#!/bin/bash

# 设置环境变量
source ~/.bashrc
QIK_VERSION=${QIK_VERSION:-2.19.4.240226}
QNN_SDK_ROOT=${QNN_SDK_ROOT:-/opt/qcom/aistack/qnn/$QIK_VERSION}
# 设置默认的输出模型版本，如果环境变量未设置则使用默认值
MODEL_VERSION="${MODEL_VERSION:-0806}"
BATCH_SIZE=${BATCH_SIZE:-1}
IMG_SIZE_STR=${IMG_SIZE_STR:-512 960}
IFS=' ' read -r -a IMG_SIZE <<< "$IMG_SIZE_STR"
OUT_VERSION="${OUT_VERSION:-$MODEL_VERSION}"

source ${QNN_SDK_ROOT}/bin/envsetup.sh
ROOT_DIR="${ROOT_DIR:-../}"
# 设置 pt 和 onnx 格式 模型路径
export PT_MODEL="$ROOT_DIR/yolov5/runs/train/$MODEL_VERSION/weights/best.pt"
export ONNX_MODEL="$ROOT_DIR/onnx"
# yolov5 export.py 脚本路径，导出 onnx用
export ONNX_EXPORT_PY="$ROOT_DIR/yolov5/export.py" 

# 设置量化参考图片目录，转换 RAW 格式输出路径 和 list 文件路径
export ORI_DIR="$ROOT_DIR/qnn_proc/source"  # 原始图片目录
# 检查 BATCH_SIZE 是否不等于 1
if [ "$BATCH_SIZE" -ne 1 ]; then
    # 当 BATCH_SIZE 不等于 1 时
    export RAW_DIR="$ROOT_DIR/qnn_proc/source_raw_"${IMG_SIZE[0]}"_batch_$BATCH_SIZE"  # 处理后的图片目录
    export LIST_FILE="$ROOT_DIR/qnn_proc/source_raw_"${IMG_SIZE[0]}"_batch_$BATCH_SIZE.txt"
else
    # 当 BATCH_SIZE 等于 1 时
    export RAW_DIR="$ROOT_DIR/qnn_proc/source_raw_"${IMG_SIZE[0]}""  # 处理后的图片目录
    export LIST_FILE="$ROOT_DIR/qnn_proc/source_raw_"${IMG_SIZE[0]}".txt"
fi
export RAW="true"
echo "RAW_DIR is set to: $RAW_DIR"

# 设置量化模型的输出路径
export OUTPUT_PATH="$ROOT_DIR/qnn_out/$OUT_VERSION"

# 根据需要启用或禁用8295推理的context文件生成
export ENABLE_8295_INFERENCE="true"  

# 确保脚本在遇到错误时停止执行
set -e

# 设置 conda 路径
# eval "$(~/miniconda3/bin/conda shell.bash hook)"
CONDA_PATH=${CONDA_PATH:-~/anaconda3}
source $CONDA_PATH/etc/profile.d/conda.sh
CONDA_ENV=${CONDA_ENV:-qnn}
# 切换为 conda 环境 (用torch)
echo -e "\n----------------------------------------"
echo "Activating $CONDA_ENV environment..."
conda activate $CONDA_ENV

# 执行 export.py，如果 ONNX 文件已存在，则跳过
echo -e "\n----------------------------------------"
echo "Exporting onnx..."
if [ ! -f "./onnx/$OUT_VERSION.onnx" ]; then
    # 运行 export.py 脚本，仅导出 ONNX 格式
    python $ONNX_EXPORT_PY \
    --weights $PT_MODEL \
    --include 'onnx' \
    --simplify \
    --imgsz ${IMG_SIZE[0]} ${IMG_SIZE[1]} \
    --device 0 \
    --batch $BATCH_SIZE

    # 移动文件到输出目录并重命名为.onnx
    mv "$(dirname "$PT_MODEL")/$(basename "$PT_MODEL" .pt).onnx" "$ONNX_MODEL/$OUT_VERSION.onnx"

    echo "Export completed: ONNX model is saved as $ONNX_MODEL/$OUT_VERSION.onnx"
else
    echo "$OUT_VERSION.onnx already exists, skipping ONNX export."
fi

# 执行 img2raw.py 脚本，处理量化参考图片，如果 RAW_DIR 已存在，则跳过
echo -e "\n----------------------------------------"
echo "Running img2raw.py..."
if [ ! -d "$RAW_DIR" ]; then
    python $ROOT_DIR/scripts/tools/img2raw.py "${ORI_DIR}" "${RAW_DIR}" --img_size ${IMG_SIZE[0]} ${IMG_SIZE[1]} --batch_size $BATCH_SIZE
else
    echo "Raw images directory already exists, skipping raw image processing."
fi

# # 执行 gen_input_list.sh，如果 LIST_FILE 已存在，则跳过
echo -e "\n----------------------------------------"
echo "Running gen_input_list.sh..."
if [ ! -f "$LIST_FILE" ]; then
    $ROOT_DIR/scripts/tools/gen_input_list.sh
else
    echo "$LIST_FILE already exists, skipping generation of image list."
fi

# # 不确定是否需要，wsl上要，但是原生ubuntu的时候没碰到这个问题
export LD_LIBRARY_PATH=$CONDA_PATH/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# 计数量化参考图片总数
TOTAL_IMAGES=$(ls -1 $ORI_DIR/*.jpg | wc -l)
# 初始化进度计数器
COUNT=0
EXECUTE_STARTED=false
FIRST_EXECUTE=true
FREEING_GRAPHSINFO=false
echo -e "\n----------------------------------------"
echo "Running qnn-onnx-converter..."
# 检查生成物是否存在，如果存在则跳过量化步骤
if [[ ! -f "${OUTPUT_PATH}/${OUT_VERSION}.cpp" ]]; then
    # 量化
    ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network $ROOT_DIR/onnx/$OUT_VERSION.onnx \
    --output_path ${OUTPUT_PATH}/$OUT_VERSION.cpp \
    --input_list ${LIST_FILE} \
    -b $BATCH_SIZE \
    2>&1 | {
        while read -r LINE; do
            if echo "$LINE" | grep -q "CpuGraph::execute"; then
                if $FIRST_EXECUTE; then
                    echo "$LINE"
                    FIRST_EXECUTE=false
                    EXECUTE_STARTED=true
                else
                    EXECUTE_STARTED=true
                    COUNT=$((COUNT + 1))
                    PERCENTAGE=$((COUNT * 100 / TOTAL_IMAGES))
                    TIMESTAMP=$(echo "$LINE" | awk '{print $1}')
                    echo -ne "$TIMESTAMP [ INFO ] Progress: $COUNT/$TOTAL_IMAGES ($PERCENTAGE%)\r"
                fi
            elif echo "$LINE" | grep -q "Freeing graphsInfo"; then
                FREEING_GRAPHSINFO=true
                echo -e "\n$LINE"
            else
                if [ "$EXECUTE_STARTED" = false ] || [ "$FREEING_GRAPHSINFO" = true ]; then
                    echo "$LINE"
                fi
            fi
        done
        echo -e "\nQuantization completed: Config is saved as ${OUTPUT_PATH}/${OUT_VERSION}.cpp"
    }
else
    echo "Quantization step skipped. ${OUTPUT_PATH}/${OUT_VERSION}.cpp already exists."
fi

# 创建输出目录
mkdir -p ${OUTPUT_PATH}/model_libs

echo -e "\n----------------------------------------"
echo "Running qnn-model-lib-generator..."
# 检查生成物是否存在，如果存在则跳过生成bin文件步骤
if [[ ! -f "${OUTPUT_PATH}/model_libs/x86_64-linux-clang/lib$OUT_VERSION.so" ]]; then
    # 生成so文件, -t 可以选择输出平台，参考官方文档
    ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c ${OUTPUT_PATH}/$OUT_VERSION.cpp \
    -b ${OUTPUT_PATH}/$OUT_VERSION.bin \
    -t x86_64-linux-clang \
    -o ${OUTPUT_PATH}/model_libs \
    2>&1 | grep -Ei "INFO|ERROR"
    echo "Model library generation completed: Model is saved as ${OUTPUT_PATH}/model_libs/x86_64-linux-clang/lib$OUT_VERSION.so"
else
    echo "Model library generation step skipped. ${OUTPUT_PATH}/model_libs/x86_64-linux-clang/lib$OUT_VERSION.so already exists."
fi

echo -e "\n----------------------------------------"
echo "Running qnn-context-binary-generator..."
# 使用环境变量ENABLE_8295_INFERENCE来控制是否执行适合8295的context生成
if [[ "${ENABLE_8295_INFERENCE}" == "true" ]]; then
    # 检查文件是否存在，如果存在则跳过推理步骤
    if [[ ! -f "${OUTPUT_PATH}/${OUT_VERSION}.serialized.bin" ]]; then
        ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator \
        --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
        --model ${OUTPUT_PATH}/model_libs/x86_64-linux-clang/lib$OUT_VERSION.so \
        --output_dir ${OUTPUT_PATH} \
        --binary_file $OUT_VERSION.serialized
    echo "Context Binary generated: Serialized Binary is saved as ${OUTPUT_PATH}/${OUT_VERSION}.serialized.bin"
    else
        echo "Context Binary generation skipped. ${OUTPUT_PATH}/${OUT_VERSION}.serialized.bin already exists."
    fi
else
    echo "Context Binary generation skipped. Flag ENABLE_8295_INFERENCE set False."
fi