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
echo "Activating $CONDA_ENV environment..."
conda activate $CONDA_ENV

export ROOT_DIR=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH}:$ROOT_DIR/yolov5"

export TRAIN_PY="$ROOT_DIR/yolov5/train.py"

MODEL_VERSION=${1:-0806}
MODEL_CFG=${2:-"yolov5x-im.yaml"}
IMG_SIZE_STR=${3:-512 960}
RESUME=${4:-"$ROOT_DIR/yolov5/runs/$MODEL_VERSION/weights/best.pt"}
# IMG_SIZE 转为数组
IFS=' ' read -r -a IMG_SIZE <<< "$IMG_SIZE_STR"

# usage bash train.sh 0806-de yolov5x-de.yaml

echo "Start $TRAIN_PY, check $ROOT_DIR/logs/train_$MODEL_VERSION.log"

# 多卡训练
# python -m torch.distributed.run --nproc_per_node 4 \
#                                 --master_port=29504 \
#                                 $TRAIN_PY --device 4,5,6,7 --name $MODEL_VERSION --epochs 120 \
#                                    --batch-size 64 \
#                                    --workers 32 \
#                                    --optimizer "AdamW" \
#                                    --cfg $ROOT_DIR/yolov5/models/$MODEL_CFG \
#                                    --imgsz ${IMG_SIZE[0]} ${IMG_SIZE[1]} \
#                                    --data $ROOT_DIR/yolov5/data/im.yaml \
#                                    --hyp $ROOT_DIR/yolov5/data/hyps/hyp.im-adamw.yaml \
#                                 > $ROOT_DIR/logs/train_$MODEL_VERSION.log 2>&1 &

# lsof -i :29504 | grep 'python' | awk '{print $2}' | xargs kill
# pkill -f "python -u $(pwd)/yolov5/train.py"

# 单卡训练

python $TRAIN_PY --device 0 --name $MODEL_VERSION --epochs 35 \
                 --batch-size 4 \
                 --workers 4 \
                 --optimizer "AdamW" \
                 --cfg $ROOT_DIR/yolov5/models/$MODEL_CFG \
                 --imgsz ${IMG_SIZE[0]} ${IMG_SIZE[1]} \
                 --data $ROOT_DIR/yolov5/data/im-local.yaml \
                 --hyp $ROOT_DIR/yolov5/data/hyps/hyp.im-adamw.yaml \
       > $ROOT_DIR/logs/train_$MODEL_VERSION.log 2>&1 &


# pkill -f "python $(pwd)/yolov5/train.py"

# tmux new -s plume
# bash train.sh 0808-tr-sa yolov5x-tr-sa.yaml
# ctrl+b & d 退出
# tmux attach -t plume
# tmux kill-session -t plume