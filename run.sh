#!/bin/bash
# usage:
# bash ./run.sh 0807-sa "512 960" ./val12 > ./logs/run_0807-sa.log 2>&1 &
# bash ./run.sh 0812 "512 960" ./val2508 > ./logs/run_0812.log 2>&1 &

source ~/.bashrc
export QIK_VERSION="2.19.4.240226"
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/$QIK_VERSION
source ${QNN_SDK_ROOT}/bin/envsetup.sh
export ROOT_DIR=$(pwd)

# 设置 conda 路径
# 检测 Miniconda 或 Anaconda 安装路径
if [ -d "$HOME/miniconda3" ]; then
    export CONDA_PATH="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    export CONDA_PATH="$HOME/anaconda3"
else
    echo "Neither Miniconda nor Anaconda found in the home directory."
    exit 1
fi
source $CONDA_PATH/etc/profile.d/conda.sh
export CONDA_ENV="qnn"
# 切换为 conda 环境 (用torch)
echo -e "\n----------------------------------------"
echo "Activating $CONDA_ENV environment..."
conda activate $CONDA_ENV

# 使脚本在发生错误时退出
set -e
# 设置错误处理程序
trap 'echo "Error occurred in $0 at line $LINENO"; exit 1' ERR

MODEL_VERSION=${1:-0703}
IMG_SIZE_STR=${2:-640 640}
VAL_DIR=${3:-./val2508}
BATCH_SIZE=${4:-1}
echo "The MODEL_VERSION is: $MODEL_VERSION"
echo "The Val data Path is: $VAL_DIR"
export MODEL_VERSION
export IMG_SIZE_STR
export VAL_DIR
export BATCH_SIZE

# IMG_SIZE 转为数组
IFS=' ' read -r -a IMG_SIZE <<< "$IMG_SIZE_STR"

# 检查 BATCH_SIZE 是否不等于 1
if [ "$BATCH_SIZE" -ne 1 ]; then
    # 当 BATCH_SIZE 不等于 1 时
    export OUT_VERSION="${MODEL_VERSION}_"${IMG_SIZE[0]}"_batch_${BATCH_SIZE}"
else
    # 当 BATCH_SIZE 等于 1 时
    export OUT_VERSION=${MODEL_VERSION}_${IMG_SIZE[0]}
fi

# 开始量化
echo -e "\n----------------------------------------"
echo "Starting Quantity"
echo "Running quantity.sh..."

./scripts/quantity.sh

echo "quantity.sh finished successfully, results in ./qnn_out/$OUT_VERSION"

# 开始量化模型的推理，包括前处理数据，推理以及结果后处理
echo ""
echo -e "\n----------------------------------------"
echo "Starting Infer"
echo "Running infer.sh..."

./scripts/infer.sh

echo "infer.sh finished successfully"
echo "raw results in ./infer_tmp/val_$OUT_VERSION"
echo "final results in ./output/$OUT_VERSION"

# 开始对pred结果与ground truth做验证
echo ""
echo -e "\n----------------------------------------"
echo "Starting Val"
echo "Running val.sh..."

./scripts/val.sh
echo "val.sh finished successfully"

echo ""
echo -e "\n----------------------------------------"
echo "All scripts have been executed successfully."