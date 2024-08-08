# YOLOv5x Custom Configuration

## Step 1: Adjust Hyperparameters
使用 `data/hyp` 目录下的 `hyp.im.yaml` 文件，并修改 `train.py` 中的 `--hyp` 参数。

## Step 2: Adjust Dataset Parameters
使用 `data/im.yaml` 文件，并修改 `train.py` 中的 `--data` 参数。

## Step 3: Adjust Model Structure - Replace Backbone with Swin V2
使用 `models/yolov5x-im.yaml` 文件，将 `backbone` 中的 `C3` 替换为 `swin`，并在 `train.py` 中修改 `--cfg` 参数。

### 3.1: Add Swin Block in `models/common.py`
在 `models/common.py` 中添加 Swin 相关 block。

#### 3.1.1: Import Swin Module and Modify `parse_model` Function
在 `models/yolo.py` 中引入 Swin 模块 `C3STR` 并修改 `parse_model` 函数。

```python
from models.common import (
    # ……
    C3STR
)

def parse_model(d, ch):
    # ……
    if m in {
        Conv,
        GhostConv,
        Bottleneck,
        GhostBottleneck,
        SPP,
        SPPF,
        DWConv,
        MixConv2d,
        Focus,
        CrossConv,
        BottleneckCSP,
        C3,
        C3TR,
        C3SPP,
        C3Ghost,
        nn.ConvTranspose2d,
        DWConvTranspose2d,
        C3x,
        C3STR
    }:
        c1, c2 = ch[f], args[0]
        if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, ch_mul)

        args = [c1, c2, *args[1:]]
        if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, C3STR}:
            args.insert(2, n)  # number of repeats
            n = 1
```

### 3.2: Modify Model Definition with SimAM
修改模型定义文件 `models/yolov5x-im-swin.yaml`，使用 SimAM。

#### 3.2.1: Add SimAM Class in `models/common.py`
在 `models/common.py` 中添加 SimAM 类的定义。

#### 3.2.2: Modify `parse_model` Function
在 `models/yolo.py` 中，修改 `parse_model` 函数。

```python
def parse_model(d, ch):
    # ……
    elif m is SimAM:
        c1, c2 = ch[f], args[0]
        if c2 != no:
            c2 = make_divisible(c2 * gw, 8)
        args = [c1, c2]

    elif m is nn.BatchNorm2d:
        args = [ch[f]]
```

## Step 4: Modify Input Size Definition
将固定的 640 方形修改为支持非标准矩形，例如 512\*960。

### 4.1: Modify `train.py`
#### 4.1.1: Modify `parse_opt` Function
修改参数定义函数 `parse_opt`，`imgsz` 参数要能支持非标准矩形尺寸，改为 list 类型。

```python
def parse_opt(known=False):
    # ……
    parser.add_argument('--imgsz', '--img', '--img-size', nargs=2, type=int, default=[512, 960], help='train, val image size (pixels)')
```

#### 4.1.2: Modify Multi-scale Support
修改 Multi-scale 支持非标准矩形。

```python
def train(hyp, opt, device, callbacks):
    # ……
    # Multi-scale
    if opt.multi_scale:
        if isinstance(imgsz, int):
            sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
        else:  # imgsz is a list
            min_size = min(imgsz)
            max_size = max(imgsz)
            sz = random.randrange(int(min_size * 0.5), int(max_size * 1.5) + gs) // gs * gs  # size

    hyp['obj'] *= (max(imgsz) / 640) ** 2 * 3 / nl  # scale to image size and layers

    # Log
    if RANK in {-1, 0}:
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        pbar.set_description(('%11s' * 2 + '%11.4g' * 4 + '%13s') %
                             (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], f'[{imgs.shape[2]},{imgs.shape[3]}]'))
        callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
```

### 4.2: Modify `utils/dataloaders.py`
#### 4.2.1: Modify Image and Label Loading Class
修改图片和 label 的加载类，mosaic 部分用不到可以不改。

```python
class LoadImagesAndLabels(Dataset):
    # ……
    def __init__():
        # ……
        if isinstance(self.img_size, int):
            self.mosaic_border = [-img_size // 2, -img_size // 2]
        else:
            self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2]

    def load_image(self, i):
        # ……
        if isinstance(self.img_size, int):
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        else:
            im = cv2.resize(im, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)

    # Modify cache check to avoid errors
    def check_cache_ram(self, safety_margin=0.1, prefix=""): 
        # ……
        if isinstance(self.img_size, int):
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        else:
            ratio = [self.img_size[0] / im.shape[1], self.img_size[1] / im.shape[0]]  # [width_ratio, height_ratio]
            b += im.nbytes * ratio[0] * ratio[1]  # use width and height ratios

    def load_mosaic(self, index):
        # ……
        if isinstance(self.img_size, int):
            s = self.img_size
            yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        else:
            s_h, s_w = self.img_size # (h,w)
            yc, xc = [int(random.uniform(-x, 2 * s + x)) for x, s  in zip(self.mosaic_border, self.img_size)]

        # place img in img4
        if isinstance(self.img_size, int):
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)


            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        else:
            if i == 0:  # top left
                img4 = np.full((s_h * 2, s_w * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s_w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s_h * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s_w * 2), min(s_h * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        if isinstance(self.img_size, int):
            for x in (labels4[:, 1:], *segments4):
                np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        else:
            for x in (labels4[:, 1:], *segments4):
                np.clip(x[:, 0::2], 0, 2 * s_w, out=x[:, 0::2])  # clip when using random_perspective()
                np.clip(x[:, 1::2], 0, 2 * s_h, out=x[:, 1::2])  # clip when using random_perspective()
```

### 4.3: Modify `val.py`
#### 4.3.1: Modify Log Information
修改 log 防止误解。

```python
def run():
    # ……
    if isinstance(imgsz, int):
        LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')
    else:
        LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz[0]},{imgsz[1]}) for non-PyTorch models')

    # Modify warmup input size
    if not training:
        if isinstance(imgsz, int):
            model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        else:
            model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz[0], imgsz[1]))  # warmup

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        if isinstance(imgsz, int):
            shape = (batch_size, 3, imgsz, imgsz)
        else:
            shape = (batch_size, 3, imgsz[0], imgsz[1])
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

# Modify `parse_opt` function
def parse_opt(known=False):
    # ……
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[512, 960], help='train, val image size (pixels)')
```

### 4.4: Modify `detect.py`
#### 4.4.1: Modify `parse_opt` Function
修改 `parse_opt` 函数以支持非标准矩形尺寸。

```python
def parse_opt(known=False):
    # ……
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[512, 960], help='inference size h,w')
```

## Step 5: Use AdamW Optimizer
选择 `AdamW` 优化器，使用 `--optimizer 'AdamW'` 参数。

## Step 6: Modify ONNX Export Code
修改导出 ONNX 模型的代码，去掉导出后处理的部分，方便量化处理，主要位于 `models/yolo.py`。

```python
class Detect(nn.Module):
    # ……
    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.export:  # export
                return x

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training or self.export:
            return x
        else:
            return (torch.cat(z, 1), x)
```