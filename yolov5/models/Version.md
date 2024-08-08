By Plume
Based on yolov5x-de.yaml

''' yaml
    # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

    # Parameters
    nc: 3  # number of classes
    depth_multiple: 1.33  # model depth multiple
    width_multiple: 1.25  # layer channel multiple
    anchors:
    - [10,13, 16,30, 33,23]  # P3/8
    - [30,61, 62,45, 59,119]  # P4/16
    - [116,90, 156,198, 373,326]  # P5/32

    # YOLOv5 v6.0 backbone
    backbone:
    # [from, number, module, args]
    [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]],  # 9
    ]

    # YOLOv5 v6.0 head
    head:
    [[-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    [[-1, 6], 1, Concat, [1]],  # cat backbone P4
    [-1, 3, C3, [512, False]],  # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    [[-1, 4], 1, Concat, [1]],  # cat backbone P3
    [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
    [-1, 1, Conv, [256, 3,2]],  # 18 (P3/8-small)

    [17, 1, Conv, [256, 1, 2]],
    [[-1, 14], 1, Concat, [1]],  # cat head P4
    [-1, 3, C3, [512, False]],  # 21 (P4/16-medium)
    [-1, 1, Conv, [512, 1,2]],  # 22 (P3/8-small)

    [21, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]],  # cat head P5
    [-1, 3, C3, [1024, False]],  # 25 (P5/32-large)
    [-1, 1, Conv, [1024, 1,2]],  # 26 (P3/8-small)

    [[18, 22, 26], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
    ]

'''

# Version 1 -- SimAM -- 0807-sa

添加 SimAM，在 Backbone 的 每一个 C3 层后加入无参注意力机制

修改后 cfg 为 yolov5x-sa.yaml

参考：
https://proceedings.mlr.press/v139/yang21o/yang21o.pdf
https://github.com/ZjjConan/SimAM
https://jishuzhan.net/article/1695777863422185473

小规模数据验证：
1. 训练初期比较波动，但后续稳定
2. 车端版本 QNN 量化算子支持


# Version 2 -- SimAM + GhostConv -- 0807-sa-gc

在 Version 1 的基础上，将普通卷积操作，替换为轻量的 Ghost 卷积，

其中 Conv 替换为 GhostConv，C3 替换为 C3Ghost，验证算子，为 Transformer 结构加入腾出空间

小规模数据验证：
1. 参数量和GFLOPS减少至约1/4，显存占用减少至约1/2，指标掉点稍明显（20 epoch）
2. 车端版本 QNN 量化算子支持


# Version 3 -- SimAM + GhostConv + Transformer -- 0807-sa-gc-tr

在 Version 2 的基础上，将 backbone 部分 C3Ghost 替换为 C3/C3TR，C3TR中加入了Transformer

小规模数据验证：
1. Transformer 模块在车端版本 QNN 量化算子 WARNING - WARNING_GEMM: GEMM operation is not supported in the general case, attempting to interpret as FC，理论上可以由FC替代，看最后推理效果
2. backbone在P3/P4/P5都用了C3TR，小batch梯度爆炸了nan，大batch显存不够，
3. 用单个 C3TR 块测试一下，似乎很难收敛，19轮未涨点。

# Version 4 -- SimAM + Transformer -- 0807-sa-tr

在 Version 2 的基础上，只替换了一个 C3 为 C3TR，保留 SimAM

小规模数据验证：
1. 梯度消失


# Version 5 -- Transformer -- 0807-tr

在 Version 1 的基础上，只替换了一个 C3 为 C3TR

小规模数据验证：
1. C3TR 可以量化且推理，但似乎性能没有 SimAM 的 Version 1要好，可以用全量数据集试一下了。


# Version 6 -- Transformer + SimAM Head -- 0807-tr-sa

在 Version 5 的基础上，Backbone 替换了最后一个 C3 为 C3TR，在 Head 添加了 SimAM 防止梯度消失

