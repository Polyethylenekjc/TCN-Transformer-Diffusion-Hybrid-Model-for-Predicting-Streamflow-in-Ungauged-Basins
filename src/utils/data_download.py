import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import cdsapi

try:
    import xarray as xr
except Exception:
    xr = None
import numpy as np
import pandas as pd


# ========== 认证：分别为 GLOFAS 与 ERA5 创建客户端 ==========
def get_cds_client(service: str) -> cdsapi.Client:
    """Create cdsapi client for a specific service (glofas|era5)."""
    service = service.lower()
    if service == "glofas":
        return cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key="360a5389-ec27-4bc7-ad71-d521c1995e4a")
    elif service == "era5":
        return cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="360a5389-ec27-4bc7-ad71-d521c1995e4a")
    else:
        raise ValueError("service must be 'glofas' or 'era5'")


# ========== 区域/路径配置 ==========
OUTPUT_ROOT = "/mnt/d/store/TTF"
ERA5_DIR = os.path.join(OUTPUT_ROOT, "ERA5")
GLOFAS_DIR = os.path.join(OUTPUT_ROOT, "GLOFAS")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
STATIONS_DIR = os.path.join(OUTPUT_ROOT, "stations")  # 可能为空
os.makedirs(ERA5_DIR, exist_ok=True)
os.makedirs(GLOFAS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STATIONS_DIR, exist_ok=True)

# Pakistan + India region (N, W, S, E) per CDS convention
# Exact 0.1° grid coverage for 256×256:
# latMin=7.0, latMax=32.6, lonMin=77.5, lonMax=103.1
CDS_AREA_SEASIA = [32.6, 77.5, 7.0, 103.1]


# ========== 1. 下载 GLOFAS ==========
def download_glofas(year: int):
    client = get_cds_client("glofas")
    dataset = "cems-glofas-historical"
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [str(year)],
        "hmonth": [f"{m:02d}" for m in range(1, 13)],
        "hday": [f"{d:02d}" for d in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": CDS_AREA_SEASIA,  # [N, W, S, E]
    }

    target_zip = os.path.join(GLOFAS_DIR, f"{year}.zip")
    target_dir = os.path.join(GLOFAS_DIR, str(year))
    if os.path.exists(os.path.join(target_dir, f"{year}.nc")):
        print(f"[GLOFAS] {year} exists, skip download.")
        return

    print(f"[GLOFAS] Downloading {year} ...")
    client.retrieve(dataset, request, target_zip)
    print(f"[GLOFAS] Extracting {target_zip} -> {target_dir}")
    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall(target_dir)

    # 重命名为 {year}.nc
    renamed = False
    for root, _, files in os.walk(target_dir):
        for fn in files:
            if fn.endswith('.nc'):
                old = os.path.join(root, fn)
                new = os.path.join(target_dir, f"{year}.nc")
                shutil.move(old, new)
                renamed = True
                print(f"[GLOFAS] {old} -> {new}")
                break
    if os.path.exists(target_zip):
        os.remove(target_zip)
    if not renamed:
        raise RuntimeError(f"[GLOFAS] No .nc found after extracting {target_zip}")


# ========== 2. 下载 ERA5 ==========
ERA5_VARIABLES = [
    "2m_dewpoint_temperature",
    "skin_temperature",
    "surface_latent_heat_flux",
    "surface_net_thermal_radiation",
    "surface_solar_radiation_downwards",
    "potential_evaporation",
    "runoff",
    "sub_surface_runoff",
    "total_evaporation",
    "total_precipitation",
]


def download_era5(year: int, month: int, hour: str = "12:00"):
    client = get_cds_client("era5")
    dataset = "reanalysis-era5-land"
    request = {
        "variable": ERA5_VARIABLES,
        "year": f"{year}",
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [hour],  # 每日一次，可调整
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": CDS_AREA_SEASIA,  # [N, W, S, E]
    }

    target = os.path.join(ERA5_DIR, f"{year}-{month:02d}.nc")
    if os.path.exists(target):
        print(f"[ERA5] Exists: {target}")
        return
    print(f"[ERA5] Downloading {year}-{month:02d} ...")
    client.retrieve(dataset, request).download(target=target)


# ========== 3. 合并为多通道 NPY ==========
def align_and_save_npy(years: List[int],
                       out_dir: str = IMAGES_DIR,
                       region: List[float] = [77.5, 103.1, 7.0, 32.6],
                       resolution: float = 0.1,
                       grid_h: int = 256,
                       grid_w: int = 256):
    """
    读取 ERA5 与 GLOFAS NetCDF，裁剪到区域并按日期合并为 (C,H,W) NPY。
    通道顺序：
      [ GLOFAS_discharge,
        2m_dewpoint_temperature, skin_temperature, total_precipitation,
        total_evaporation, runoff, sub_surface_runoff,
        surface_latent_heat_flux, surface_net_thermal_radiation,
        potential_evaporation ]  -> 共 10 通道
    """
    if xr is None:
        raise RuntimeError("xarray is required. Please install xarray and netCDF4.")

    lon_min, lon_max, lat_min, lat_max = region

    # 构建目标网格：使用半格偏移以得到严格的 256×256 中心点
    # 中心点覆盖 [lon_min+0.05, ..., lon_max-0.05], [lat_max-0.05, ..., lat_min+0.05]
    lons = lon_min + (resolution * 0.5) + resolution * np.arange(grid_w)
    lats = lat_max - (resolution * 0.5) - resolution * np.arange(grid_h)
    H, W = grid_h, grid_w

    # 打开 ERA5 多文件
    era5_files = [os.path.join(ERA5_DIR, f"{y}-{m:02d}.nc") for y in years for m in range(1, 13)]
    era5_files = [p for p in era5_files if os.path.exists(p)]
    if not era5_files:
        raise RuntimeError("No ERA5 files found to merge.")
    ds_era5 = xr.open_mfdataset(era5_files, combine='by_coords')
    ds_era5 = ds_era5.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    # 打开 GLOFAS 多年
    glofas_files = [os.path.join(GLOFAS_DIR, str(y), f"{y}.nc") for y in years]
    glofas_files = [p for p in glofas_files if os.path.exists(p)]
    if not glofas_files:
        raise RuntimeError("No GLOFAS files found to merge.")
    ds_glofas = xr.open_mfdataset(glofas_files, combine='by_coords')
    # GLOFAS 坐标可能叫 lat/latitude, lon/longitude
    lat_name = 'lat' if 'lat' in ds_glofas.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_glofas.coords else 'longitude'
    ds_glofas = ds_glofas.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})

    # 按日期交集输出
    era5_times = set(pd.to_datetime(ds_era5['time'].values).strftime('%Y%m%d'))
    glofas_times = set(pd.to_datetime(ds_glofas['time'].values).strftime('%Y%m%d'))
    common_days = sorted(era5_times.intersection(glofas_times))
    if not common_days:
        raise RuntimeError("No overlapping days between ERA5 and GLOFAS.")

    # 插值到统一网格（如二者经纬度完全一致为 0.1°，则可直接重索引）
    ds_era5_interp = ds_era5.interp(latitude=lats, longitude=lons)
    ds_glofas_interp = ds_glofas.interp({lat_name: lats, lon_name: lons})

    var_map = {
        'glofas': 'river_discharge_in_the_last_24_hours',
        'era5_vars': [
            'd2m' if 'd2m' in ds_era5_interp else '2m_dewpoint_temperature',
            'skt' if 'skt' in ds_era5_interp else 'skin_temperature',
            'tp' if 'tp' in ds_era5_interp else 'total_precipitation',
            'evap' if 'evap' in ds_era5_interp else 'total_evaporation',
            'ro' if 'ro' in ds_era5_interp else 'runoff',
            'ssro' if 'ssro' in ds_era5_interp else 'sub_surface_runoff',
            'slhf' if 'slhf' in ds_era5_interp else 'surface_latent_heat_flux',
            'str' if 'str' in ds_era5_interp else 'surface_net_thermal_radiation',
            'pev' if 'pev' in ds_era5_interp else 'potential_evaporation',
        ]
    }

    for day in common_days:
        # 选择该日 12:00 的 ERA5
        sel_era5 = ds_era5_interp.sel(time=slice(f"{day}T00:00:00", f"{day}T23:59:59")).isel(time=0)
        sel_glofas = ds_glofas_interp.sel(time=slice(f"{day}T00:00:00", f"{day}T23:59:59")).isel(time=0)

        channels: List[np.ndarray] = []
        # GLOFAS discharge
        if var_map['glofas'] not in sel_glofas:
            raise RuntimeError(f"Missing GLOFAS variable: {var_map['glofas']}")
        channels.append(sel_glofas[var_map['glofas']].values.astype(np.float32))

        for vn in var_map['era5_vars']:
            if vn not in sel_era5:
                raise RuntimeError(f"Missing ERA5 variable: {vn}")
            channels.append(sel_era5[vn].values.astype(np.float32))

        arr = np.stack(channels, axis=0)  # (C,H,W)
        assert arr.shape[1] == H and arr.shape[2] == W, f"Spatial shape mismatch: {arr.shape} vs {(H,W)}"
        np.save(os.path.join(out_dir, f"{day}.npy"), arr)
        print(f"[MERGE] Saved {day}.npy  shape={arr.shape}")


# ========== 4. 主程序：下载 -> 合并 ==========
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser("Download ERA5 & GLOFAS, then merge to NPY")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--hour", type=str, default="14:00", help="ERA5 hour (e.g. 00:00, 12:00, 14:00)")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    months = list(range(1, 13))

    # 先并行下载 GLOFAS（按年）
    print("🚀 Downloading GLOFAS...")
    with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(download_glofas, y) for y in years]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[ERROR] GLOFAS year failed: {e}")

    # 再并行下载 ERA5（按月）
    print("🚀 Downloading ERA5 (hour=" + args.hour + ") ...")
    with ProcessPoolExecutor(max_workers=1) as ex:
        futures = [ex.submit(download_era5, y, m, args.hour) for y in years for m in months]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[ERROR] ERA5 month failed: {e}")

    print("🔧 Merging NetCDF -> NPY (multi-channel images)...")
    try:
        align_and_save_npy(years)
        print("✅ Merge complete. NPY images at:", IMAGES_DIR)
    except Exception as e:
        print(f"[ERROR] Merge failed: {e}")