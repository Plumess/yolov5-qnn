import torch
import yaml
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from models.yolo import Model, Detect
from models.common import *

def print_size_hook(module, input, output):
    """打印层的名称和输出尺寸的钩子函数"""
    if isinstance(output, torch.Tensor):
        print(f"{module.__class__.__name__} - Output size: {output.shape}")
    elif isinstance(output, (tuple, list)):
        sizes = ", ".join([f"{o.shape}" for o in output if isinstance(o, torch.Tensor)])
        print(f"{module.__class__.__name__} - Output sizes: {sizes}")
    else:
        print(f"{module.__class__.__name__} - Output size: {output}")

def register_hooks(model, layers):
    """为模型的每一层注册前向钩子，并确认注册成功"""
    for i, (name, layer) in enumerate(model.named_children()):
        if i in layers:
            layer.register_forward_hook(print_size_hook)
            print(f"Hook registered on: {name} ({layer.__class__.__name__})")

def parse(cfg="./models/yolov5x-im.yaml"):
    with open(cfg, encoding="ascii", errors="ignore") as f:
        cfg_yaml = yaml.safe_load(f)  # model dict
    d = deepcopy(cfg_yaml)
    layers = []
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        layers.append(i)
    return layers

# 加载模型配置和超参数
cfg = "./models/yolov5x-hyper.yaml"  # 模型配置文件路径
hyp = "./data/hyps/hyp.im.yaml"    # 超参数文件路径
nc = 3  # 类别数
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 根据环境选择设备

# 加载超参数
if isinstance(hyp, str):
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict

# 创建模型
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # 创建模型并转移到相应的设备

# 解析模型配置
layers = parse(cfg)

# 注册钩子
register_hooks(model.model, layers)

# 创建伪造输入
dummy_input = torch.randn(1, 3, 512, 960).to(device)

# 前向传播以打印每层输出尺寸
model.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 关闭梯度计算
    output = model(dummy_input)  # 运行模型