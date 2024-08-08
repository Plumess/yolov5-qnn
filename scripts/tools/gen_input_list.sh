#!/bin/bash
# infer 和 quantity 两用，RAW=true 是 quantity 生成 raw 的 list，否则是 infer

# 设置默认的源目录和输出文件路径，允许通过环境变量覆盖
#export RAW=true
#export RAW=false

if [ "$RAW" = "true" ]; then
    IMG_DIR="${RAW_DIR:-../../qnn_proc/source_raw}"
    LIST_FILE="${LIST_FILE:-../../input_raw.txt}"
else
    IMG_DIR="${ORI_DIR:-../../qnn_proc/source}"
    LIST_FILE="${LIST_FILE:-../../input_im.txt}"
fi

# 检查源目录是否存在
if [ ! -d "$IMG_DIR" ]; then
    echo "目录不存在: $IMG_DIR"
    exit 1
fi

# 获取真实路径
REAL_IMG_DIR=$(realpath "$IMG_DIR")

# 清空或创建输出文件
> "$LIST_FILE"

# 查找所有图片并写入文件，确保写入真实路径
find "$REAL_IMG_DIR" -type f \( -iname "*.jpg" -o -iname "*.raw" -o -iname "*.png" \) -print0 | xargs -0 realpath | sort > "$LIST_FILE"

# 输出结果
if [ $? -eq 0 ]; then
    echo "图片路径已保存到 $LIST_FILE"
else
    echo "生成图片路径时发生错误。"
fi
