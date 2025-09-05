#!/bin/bash

# 添加错误检查
set -e  # 如果任何命令失败则退出
set -u  # 使用未定义的变量时报错

# 定义模型和数据集的数组
MODELS=("fasterrcnn_mobilenet_320" )
DATASETS=("Full")
DATAS=("WL" "EUS")

# 检查python是否可用
if ! command -v python &> /dev/null; then
    echo "Error: python is not installed"
    exit 1
fi

# 遍历所有组合
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for DATA in "${DATAS[@]}"; do
            echo "Evaluating ${MODEL} on ${DATASET} ${DATA} dataset..."
            
            # 检查模型文件是否存在
            MODEL_PATH="output_${MODEL}_${DATA}/best_model_${MODEL}.pth"
            if [ ! -f "$MODEL_PATH" ]; then
                echo "Warning: Model file not found: $MODEL_PATH"
                continue
            fi
            
            # 检查数据目录是否存在
            IMAGE_DIR="data/Images/${DATASET}/NET-${DATA}/"
            LABEL_DIR="data/Labels/${DATASET}/NET-${DATA}/"
            if [ ! -d "$IMAGE_DIR" ] || [ ! -d "$LABEL_DIR" ]; then
                echo "Warning: Data directories not found: $IMAGE_DIR or $LABEL_DIR"
                continue
            fi
            
            # 创建输出目录
            OUTPUT_DIR="evaluation_results_${MODEL}_${DATASET}_${DATA}"
            mkdir -p "$OUTPUT_DIR"
            
            # 使用变量组装命令
            python evaluate.py --model_path "$MODEL_PATH" \
                            --model_type "$MODEL" \
                            --images_dir "$IMAGE_DIR" \
                            --labels_dir "$LABEL_DIR" \
                            --output_dir "$OUTPUT_DIR" \
                            --conf_threshold 0.5 \
                            --merge_classes
            
            if [ $? -eq 0 ]; then
                echo "Evaluation completed for ${MODEL} on ${DATASET} ${DATA} dataset"
            else
                echo "Error: Evaluation failed for ${MODEL} on ${DATASET} ${DATA} dataset"
            fi
            echo "----------------------------------------"
        done
    done
done

echo "All evaluations completed!"