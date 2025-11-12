#!/bin/bash

# TTF推理管线示例脚本

# 设置参数
CONFIG_FILE="configs/low_memory.yaml"
CHECKPOINT_FILE=""  # 如果有的话，请填写模型检查点路径
INPUT_DIR="./data/input_images"
STATIONS_CSV="./data/stations.csv"
OUTPUT_DIR="./results"

# 创建示例输入数据目录
mkdir -p $INPUT_DIR
mkdir -p ./data

# 创建示例站点CSV文件
echo "station_id,latitude,longitude,value" > $STATIONS_CSV
echo "station_1,40.1,-74.2,0.5" >> $STATIONS_CSV
echo "station_2,40.3,-74.4,0.8" >> $STATIONS_CSV

# 创建示例输入数据 (numpy数组)
python3 -c "
import numpy as np
# 创建示例输入数据 (T, C, H, W)
data = np.random.randn(3, 5, 128, 128).astype(np.float32)
np.save('$INPUT_DIR/sample_input.npy', data)
print('Created sample input data:', data.shape)
"

# 运行推理管线
python -m src.main \
    --mode pipeline \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT_FILE \
    --input-dir $INPUT_DIR \
    --stations-csv $STATIONS_CSV \
    --output-dir $OUTPUT_DIR

echo "Pipeline execution completed. Results saved to $OUTPUT_DIR"