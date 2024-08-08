#!/bin/bash
source ~/.bashrc
QIK_VERSION=${QIK_VERSION:-2.19.4.240226}
QNN_SDK_ROOT=${QNN_SDK_ROOT:-/opt/qcom/aistack/qnn/$QIK_VERSION}
source ${QNN_SDK_ROOT}/bin/envsetup.sh
export CURRENT_DIR=$(pwd)  # /path/to/qnn/

# 设置 conda 路径
# export CONDA_PATH=~/miniconda3
CONDA_PATH=${CONDA_PATH:-~/miniconda3}
source $CONDA_PATH/etc/profile.d/conda.sh
CONDA_ENV=${CONDA_ENV:-yolov10}
# 切换为 conda 环境 (用torch)
echo -e "\n----------------------------------------"
echo "Activating $CONDA_ENV environment..."
conda activate $CONDA_ENV

# 设置默认的输出模型版本，如果环境变量未设置则使用默认值
MODEL_VERSION="${MODEL_VERSION:-0806}"
OUT_VERSION="${OUT_VERSION:-$MODEL_VERSION}"

ROOT_DIR="${ROOT_DIR:-../}"
# 设置 pt 和 onnx 格式 模型路径
export PT_MODEL="$ROOT_DIR/yolov5/runs/train/$MODEL_VERSION/weights/best.pt"
export ONNX_MODEL="$ROOT_DIR/onnx"

IMG_SIZE_STR=${IMG_SIZE_STR:-512 960}
IFS=' ' read -r -a IMG_SIZE <<< "$IMG_SIZE_STR"

# export VAL_DIR="./val"
# 设置默认的验证集路径，如果环境变量未设置则使用默认值
VAL_DIR="${VAL_DIR:-./val}"

# 生成 img 图片的 list，如果 LIST_FILE 已存在，则跳过
export RAW="false"
export ORI_DIR="$VAL_DIR/images"
export IMG_FILE="$VAL_DIR/val_img.txt"
export LIST_FILE=$IMG_FILE
echo -e "\n----------------------------------------"
echo "Running gen_input_list.sh..."
if [ ! -f "$LIST_FILE" ]; then
    $ROOT_DIR/scripts/tools/gen_input_list.sh
else
    echo "$LIST_FILE already exists, skipping generation of image list."
fi

# 检查 BATCH_SIZE 是否不等于 1
if [ "$BATCH_SIZE" -ne 1 ]; then
    # 当 BATCH_SIZE 不等于 1 时
    export RAW_DIR="$VAL_DIR/images_raw_"${IMG_SIZE[0]}"_batch_$BATCH_SIZE"  # 处理后的图片目录
    export RAW_FILE="$VAL_DIR/val_raw_"${IMG_SIZE[0]}"_batch_$BATCH_SIZE.txt"
else
    # 当 BATCH_SIZE 等于 1 时
    export RAW_DIR="$VAL_DIR/images_raw_"${IMG_SIZE[0]}""  # 处理后的图片目录
    export RAW_FILE="$VAL_DIR/val_raw_"${IMG_SIZE[0]}".txt"
fi
echo "RAW_DIR is set to: $RAW_DIR"

export RAW="true"
export LIST_FILE=$RAW_FILE
# 执行 img2raw.py 脚本，qnn前处理，如果 RAW_DIR 已存在，则跳过
echo -e "\n----------------------------------------"
echo "Running img2raw.py..."
if [ ! -d "$RAW_DIR" ]; then
    python $ROOT_DIR/scripts/tools/img2raw.py "${ORI_DIR}" "${RAW_DIR}" --img_size ${IMG_SIZE[0]} ${IMG_SIZE[1]} --batch_size $BATCH_SIZE
else
    echo "Raw images directory already exists, skipping raw image processing."
fi

# 生成 raw 图片的 list，如果 LIST_FILE 已存在，则跳过
echo -e "\n----------------------------------------"
echo "Running gen_input_list.sh..."
if [ ! -f "$LIST_FILE" ]; then
    $ROOT_DIR/scripts/tools/gen_input_list.sh
else
    echo "$LIST_FILE already exists, skipping generation of image list."
fi

# 使用 qnn模型 进行初步推理
# 数据集txt中一定不要有空格，5418个的val中有 xx (2).jpg 这种文件名，会导致无法正确解析变量，记得删去
# find "./" -type f -name "*(*)*.*" -exec echo "Deleting: {}" \; -exec rm {} \;
export ADSP_LOG_LEVEL=3
export INFER_TMP="./infer_tmp/val_$OUT_VERSION"
if [ ! -d "$INFER_TMP" ] || [ "$(ls -A $INFER_TMP | wc -l)" -eq 0 ]; then
    echo -e "\n----------------------------------------"
    echo "Running qnn-net-run for inference..."
    qnn-net-run \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so \
    --model ./qnn_out/$OUT_VERSION/model_libs/x86_64-linux-clang/lib$OUT_VERSION.so \
    --input_list $RAW_FILE \
    --output_dir $INFER_TMP #\
    # --log_level verbose
else
    echo -e "\n----------------------------------------"
    echo "Output directory $INFER_TMP already contains files, skipping inference."
fi

# 使用 python 脚本进行推理结果后处理
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/yolov5"
export RESULTS_DIR="./output/$OUT_VERSION"

echo -e "\n----------------------------------------"
echo "Running post-processing..."
python $ROOT_DIR/qnn_proc/qnn_infer/main_infer.py \
    --input_file $IMG_FILE \
    --image_size 540 960 \
    --pt $PT_MODEL\
    --results_dir $RESULTS_DIR \
    --infer_tmp $INFER_TMP \
    --onnx $ONNX_MODEL/$OUT_VERSION.onnx \
    --class_num 3 \
    --draw

echo -e "\n----------------------------------------"
echo "All scripts have been executed successfully."