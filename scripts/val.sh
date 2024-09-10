#!/bin/bash
QIK_VERSION=${QIK_VERSION:-2.19.4.240226}
QNN_SDK_ROOT=${QNN_SDK_ROOT:-/opt/qcom/aistack/qnn/$QIK_VERSION}
source ${QNN_SDK_ROOT}/bin/envsetup.sh
# 设置 conda 路径
# eval "$(conda shell.bash hook)"
# export CONDA_PATH=~/miniconda3
CONDA_PATH=${CONDA_PATH:-~/miniconda3}
source $CONDA_PATH/etc/profile.d/conda.sh
CONDA_ENV=${CONDA_ENV:-yolov10}
# 切换为 conda 环境 (用torch)
echo -e "\n----------------------------------------"
echo "Activating $CONDA_ENV environment..."
conda activate $CONDA_ENV

# export MODEL_VERSION="0703_hy"
# 设置默认的输出模型版本，如果环境变量未设置则使用默认值
MODEL_VERSION="${MODEL_VERSION:-0703_hy}"
OUT_VERSION="${OUT_VERSION:-$MODEL_VERSION}"

# export VAL_DIR="./val"
# 设置默认的验证集路径，如果环境变量未设置则使用默认值
VAL_DIR="${VAL_DIR:-./val}"
export VAL_DIRNAME=$(basename "$VAL_DIR")
PRED_DIR="${PRED_DIR:-./output/$OUT_VERSION-$VAL_DIRNAME/pred_labels}"

echo -e "\n----------------------------------------"
echo "val $VAL_DIR ..."
python scripts/tools/val.py --img_dir $VAL_DIR/images --label_true_dir $VAL_DIR/labels --label_pred_dir $PRED_DIR