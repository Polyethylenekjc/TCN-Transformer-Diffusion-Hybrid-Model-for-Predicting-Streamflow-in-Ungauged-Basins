"""Visualization script for model predictions."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import yaml
import torch
from typing import Dict, List, Optional

def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_prediction(
    pred_path: str,
    target_path: str,
    output_path: str
):
    """
    Visualize prediction vs target.
    
    Args:
        pred_path: Path to prediction .npy file
        target_path: Path to target .npy file
        output_path: Path to save visualization
    """
    pred = np.load(pred_path)
    target = np.load(target_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Target
    im0 = axes[0].imshow(target, cmap='Blues')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0])
    
    # Raw Prediction
    im1 = axes[1].imshow(pred, cmap='Blues')
    axes[1].set_title('Raw Prediction')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_station_metrics(
    csv_path: str,
    output_dir: str
):
    """
    Visualize station metrics.
    
    Args:
        csv_path: Path to stations_eval.csv
        output_dir: Directory to save plots
    """
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    
    # Scatter plot: Predicted vs Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(df['actual_runoff'], df['predicted_runoff'], alpha=0.5)
    
    # Perfect line
    min_val = min(df['actual_runoff'].min(), df['predicted_runoff'].min())
    max_val = max(df['actual_runoff'].max(), df['predicted_runoff'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Runoff')
    plt.ylabel('Predicted Runoff')
    plt.title('Station Runoff: Predicted vs Actual')
    plt.grid(True)
    plt.savefig(output_path / 'station_scatter.png')
    plt.close()
    
    # Time series plot (if date is available and parseable)
    try:
        # Try to parse date, assuming YYYYMMDD format from filename
        # The CSV has 'date' column which might be just the filename stem
        # Let's assume it's sortable
        df_sorted = df.sort_values('date')
        
        # Group by station location (approximate by x,y)
        # Create a unique station ID based on position
        df_sorted['station_id'] = df_sorted.apply(
            lambda row: f"{row['position_x']:.1f}_{row['position_y']:.1f}", axis=1
        )
        
        unique_stations = df_sorted['station_id'].unique()
        
        for station in unique_stations:
            station_data = df_sorted[df_sorted['station_id'] == station]
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(station_data)), station_data['actual_runoff'], label='Actual', marker='o')
            plt.plot(range(len(station_data)), station_data['predicted_runoff'], label='Predicted', marker='x')
            
            plt.xlabel('Time Step')
            plt.ylabel('Runoff')
            plt.title(f'Station {station} Time Series')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_path / f'station_{station}_timeseries.png')
            plt.close()
            
    except Exception as e:
        print(f"Could not plot time series: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--config', type=str, default='./data/config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('data', {}).get('output_dir', './output'))
    
    pred_dir = output_dir / 'predictions'
    vis_dir = output_dir / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Visualizing results from {output_dir}...")
    
    # 1. Visualize Images
    # Find all prediction files
    pred_files = sorted(list(pred_dir.glob('pred_*.npy')))
    
    # Limit to first 20 for quick check, or random sample
    # For now, let's do first 10
    for pred_file in pred_files[:10]:
        date_str = pred_file.stem.replace('pred_', '')
        target_file = pred_dir / f'target_{date_str}.npy'
        
        if target_file.exists():
            output_path = vis_dir / f'vis_{date_str}.png'
            visualize_prediction(
                str(pred_file),
                str(target_file),
                str(output_path)
            )
            print(f"Saved visualization to {output_path}")
    
    # 2. Visualize Station Metrics
    csv_path = output_dir / 'stations_eval.csv'
    if csv_path.exists():
        visualize_station_metrics(str(csv_path), str(vis_dir))
        print(f"Saved station metrics plots to {vis_dir}")
    else:
        print("No station metrics CSV found.")

if __name__ == '__main__':
    main()
