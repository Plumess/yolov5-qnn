import reader_result
import torch
import anchors
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='QNN INFER')
    parser.add_argument('--stride', nargs='+', type=int, default=[16, 32, 64], help='List of strides [16, 32, 64]')
    parser.add_argument('--anchor_num', type=int, default=1, help='Number of anchors')
    parser.add_argument('--class_num', type=int, default=3, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images per batch')
    # anchors
    parser.add_argument('--pt', type=str, default='./weights/best.pt', help='path to weights file')
    parser.add_argument('--device', type=str, default='0', help='device id (i.e. 0 or 0,1) or cpu')
    # reader_result
    parser.add_argument("--infer_tmp", type=str, required=True, help="Root path where the output files are stored.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")

    return parser.parse_args()

def init(args):
    global stride, anchor_num, class_num, anchors
    reader_result.init(args)

    stride = args.stride
    anchor_num = args.anchor_num
    class_num = args.class_num
    
    # anchors根据实际修改, 使用自写的 anchors.py 来加载并查询模型的anchor grid
    anchors = anchors.get_anchors(args)
    # print(anchors)
    # normalize
    # 将 stride_list 转换为 tensor，进行形状调整
    stride_tensor = torch.tensor(stride).float().view(-1, 1, 1)
    anchors = anchors / stride_tensor
    print()
    print("Normalize Anchors:")
    print(anchors)
    print()

def _make_grid(nx, ny, i=0):
    d = anchors[i].device
    t = anchors[i].dtype
    shape = 1, anchor_num, ny, nx, 2  # grid shape
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv =  torch.meshgrid(y, x)  # torch>=0.7 compatibility
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (anchors[i] * stride[i]).view((1, anchor_num, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

def pred_res(num):
    x = reader_result.reconstruct(num)
    grid = [torch.empty(0) for _ in range(3)]  # init grid
    anchor_grid = [torch.empty(0) for _ in range(3)]  # init anchor grid
    z = []  # inference output
    for i in range(3):
        bs, _,ny, nx,_ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        grid[i], anchor_grid[i] = _make_grid(nx, ny, i)
        x[i]=torch.from_numpy(x[i])
        xy, wh, conf = x[i].sigmoid().split((2, 2, class_num + 1), 4)
        xy = (xy * 2 +grid[i]) * stride[i]  # xy
        wh = (wh * 2) ** 2 * anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, anchor_num * nx * ny, 8))
    preds = torch.cat(z, 1)

    # Split the batch results into individual sample predictions
    per_pred = [preds[i].unsqueeze(0) for i in range(preds.shape[0])]
    return per_pred

if __name__ == "__main__":
    args = parse_args()
    init(args)
    pred = pred_res()
    print(pred.shape)