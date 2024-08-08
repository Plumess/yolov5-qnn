#!/bin/bash
source ~/.bashrc
QIK_VERSION=${QIK_VERSION:-2.19.4.240226}
QNN_SDK_ROOT=${QNN_SDK_ROOT:-/opt/qcom/aistack/qnn/$QIK_VERSION}
source ${QNN_SDK_ROOT}/bin/envsetup.sh

# 从环境变量中获取文件名，提供默认值
MODEL_VERSION="${MODEL_VERSION:-0430}"
OUT_VERSION="${OUT_VERSION:-$MODEL_VERSION}"
# 从环境变量中获取输出路径，提供默认值
OUTPUT_PATH="${OUTPUT_PATH:-./qnn_out/MODEL_VERSION-batch-$BATCH_SIZE}"

# 从环境变量中获取数据路径，提供默认值
LIST_FILE="${LIST_FILE:-source_raw.txt}"

# 从环境变量中获取是否执行8295推理的标志，提供默认值
ENABLE_8295_INFERENCE="${ENABLE_8295_INFERENCE:-false}"

BATCH_SIZE="${BATCH_SIZE:-1}"

# 检查生成物是否存在，如果存在则跳过量化步骤
if [[ ! -f "${OUTPUT_PATH}/${OUT_VERSION}.cpp" ]]; then
    # 量化
    ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network ./onnx/$OUT_VERSION.onnx \
    --output_path ${OUTPUT_PATH}/$OUT_VERSION.cpp \
    --input_list ${LIST_FILE} \
    -b $BATCH_SIZE #\
    #--debug
else
    echo "Quantization step skipped. ${OUTPUT_PATH}/${OUT_VERSION}.cpp already exists."
fi    

# 创建输出目录
mkdir -p ${OUTPUT_PATH}/model_libs

# 检查生成物是否存在，如果存在则跳过生成bin文件步骤
if [[ ! -f "${OUTPUT_PATH}/model_libs/${OUT_VERSION}.bin" ]]; then
    # 生成so, bin文件, -t 可以选择输出平台，参考官方文档
    ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c ${OUTPUT_PATH}/$OUT_VERSION.cpp \
    -b ${OUTPUT_PATH}/model_libs/$OUT_VERSION.bin \
    -t x86_64-linux-clang \
    -o ${OUTPUT_PATH}/model_libs
else
    echo "Model library generation step skipped. ${OUTPUT_PATH}/model_libs/${OUT_VERSION}.bin already exists."
fi

# 使用环境变量ENABLE_8295_INFERENCE来控制是否执行8295推理步骤
if [[ "${ENABLE_8295_INFERENCE}" == "true" ]]; then
    # 检查文件是否存在，如果存在则跳过推理步骤
    if [[ ! -f "${OUTPUT_PATH}/${OUT_VERSION}.serialized.bin" ]]; then
        ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator \
        --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
        --model ${OUTPUT_PATH}/model_libs/$OUT_VERSION.so \
        --output_dir ${OUTPUT_PATH} \
        --binary_file $OUT_VERSION.serialized
    echo "8295 inference step executed."
    else
        echo "8295 inference step skipped. ${output_path}/${OUT_VERSION}.serialized.bin already exists."
    fi
else
    echo "8295 inference step skipped."
fi

