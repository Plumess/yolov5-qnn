项目所需软件包和资源的下载地址：

链接：https://pan.baidu.com/s/14H1_eAN_3bez30nqjRJz8g  提取码：57jg

### 一、环境准备

1. 将以下文件放入根目录：
    - `.qik`
    - `.deb`
    - `Anaconda_*.sh`
    - `android-ndk-r26d`（解压后）
2. 运行 `env.sh` 进行环境配置，详细参考其中的注释，记得修改高通账号为自己的账号，如有必要修改其他特定的变量。
3. 若已安装 `conda` 或 YOLOv5 环境，请跳过相关步骤。若无上述环境，请取消 `env.sh` 中相关代码的注释。

### 二、数据准备

1. 解压 `source.zip`，将 200 张用于量化的参考图片放入 `./qnn_proc/source` 目录。
2. 默认使用该项目进行训练，如果有自训练的权重，将训练好的权重文件放入 `./yolov5/runs/train/<version>/weights/best.pt`。
3. 将验证集放入 `./val/` 目录。

### 三、项目结构

项目目录结构如下，标记 * 的目录/文件为项目运行的产物：

```
root
├── android-ndk-r26d  # (optional)
├── *infer_tmp  # 量化模型推理的中间产物
│   └── val_<version>
│       └── Result_*
├── logs  # 各种输出log，可以重定向到此目录
│   └── *.log
├── onnx  # pt 转 onnx 的存放路径
│   └── *<version>.onnx
├── *output  # 推理的最终结果标注图
│   └── <version>
│       ├── images  # 选用，可以用infer.sh中的--draw变量控制是否生成
│       └── pred_labels  # 保存预测信息，用于进行量化的结果评测
├── qnn_out  # 量化后的模型产物存放路径
│   └── *<version>
│       ├── model_libs
│       │   └── x86_64-linux-clang
│       │       └── lib<version>.so
│       ├── <version>.bin
│       ├── <version>.cpp
│       └── <version>_net.json
├── qnn_proc  # 推理/评测的 Python 代码
│   ├── qnn_infer  # 推理及后处理用的 Python 脚本
│   │   ├── anchors.py
│   │   ├── main_infer.py
│   │   ├── qnn_infer.py
│   │   └── reader_result.py
│   ├── source  # 量化的参考 RGB 图像目录
│   │   └── *.jpg/png
│   ├── *source_raw.txt  # 量化的参考 RGB 图像的索引 list
│   └── *source_raw  # 量化的参考 RAW 图像目录
│       └── *.raw
├── scripts  # 自动化脚本
│   ├── tools
│   │   ├── gen_input_list.sh  # 生成 raw 和 RGB 图片的索引 list
│   │   ├── img2raw.py  # 将量化参考图片由 RGB 转 RAW，模型输入数据前处理逻辑在此
│   │   ├── serialize_qnn.sh  # 生成量化模型及其 context 序列化的核心脚本
│   │   └── val.py  # 量化后结果的验证脚本
│   ├── infer.sh  # 推理的自动化脚本
│   ├── quantity.sh  # 模型量化的自动化脚本
│   └── val.sh  # 验证量化模型推理结果和指标的脚本
├── val  # 验证集存放路径
│   ├── images
│   ├── labels
│   ├── *images_raw  # 自动前处理生成的量化推理用的raw图
│   ├── *val_img.txt  # 自动生成的rgb图的索引list
│   └── *val_raw.txt  # 自动生成的量化推理用的raw图的索引list
├── yolov5 # YOLOv5 项目
│   ├── runs
│   │   └── pt模型及相关结果存放处
│   ├── check_model.py  # 用于检查配置的模型的各网络层的输出尺寸
│   ├── visual.py  # 用于将gt框标在模型输出结果上，方便可视化比对
│   └── 其他yolov5核心功能代码，cfg等
├── env.sh  # 安装 QNN 环境的自动化脚本
├── run.sh  # 全流程自动化脚本入口, bash ./run.sh <version> <val path>
└── train.sh  # yolov5 训练脚本

```

### 四、使用方法

### 0. 全流程自动

使用该项目进行训练，如果有自训练的权重，将训练好的权重文件放入 `./yolov5/runs/train/<version>/weights/best.pt`；

例如 `./yolov5/runs/train/0808/weights/best.pt`

运行如下命令，第一个参数为版本号，第二个参数为模型输入尺寸，第三个参数为验证集路径，第四个参数为需要量化输出的 batch

`bash ./run.sh 0807-sa "512 960" ./val12 > ./logs/run_0807-sa.log 2>&1 &`

### 1. 模型量化

运行 `quantity.sh`，根据需求修改以下变量：

```bash
# 设置默认的输出模型版本，如果环境变量未设置则使用默认值
MODEL_VERSION="${MODEL_VERSION:-0703}"
# 根据需要启用或禁用 8295 推理(没做维护)
ENABLE_8295_INFERENCE="false"
# 设置 conda 路径，如果非默认地址
CONDA_PATH=${CONDA_PATH:-~/miniconda3}
source $CONDA_PATH/etc/profile.d/conda.sh
```

### 2. 模型推理

运行 `infer.sh`，根据需求修改以下变量以及 QNN 运行段的输入参数：

```bash
# 设置 conda 路径
# export CONDA_PATH=~/miniconda3
CONDA_PATH=${CONDA_PATH:-~/miniconda3}
source $CONDA_PATH/etc/profile.d/conda.sh
# 切换为 qnn 环境 (用torch)
echo -e "\n----------------------------------------"
echo "Activating qnn environment..."
conda activate qnn

# export MODEL_VERSION="0703_hy"
# 设置默认的输出模型版本，如果环境变量未设置则使用默认值
MODEL_VERSION="${MODEL_VERSION:-0703_hy}"

# export VAL_DIR="./val"
# 设置默认的验证集路径，如果环境变量未设置则使用默认值
VAL_DIR="${VAL_DIR:-./val}"
```

### 3. 模型验证

运行 `val.sh`，根据需求修改 Python 命令提供的参数：

```bash
# 设置 conda 路径
CONDA_PATH=${CONDA_PATH:-~/miniconda3}
source $CONDA_PATH/etc/profile.d/conda.sh
```


### 4. pt模型验证

使用yolov5项目，但由于本项目修改了`yolo.py`检查调用栈以区分量化使用和yolov5正常使用，

使用`./yolov5/val.py`之前，先确认`models/yolo.py`的

```python
class Detect(nn.Module):
    def forward(self, x):
```
的注释内容