import argparse
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 Model Anchor Printer')
    parser.add_argument('--pt', type=str, default='./weights/best.pt', help='path to weights file')
    parser.add_argument('--device', type=str, default='0', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--data', type=str, default='./data/im.yaml', help='path to data config file')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    return parser.parse_args()

def find_anchor_grid(model):
    # 从模型中找到唯一的锚点，并以期望的格式输出
    anchor_set = set()
    for m in model.model.modules():
        if hasattr(m, 'anchor_grid'):
            for anchor in m.anchor_grid:
                for a in anchor.view(-1, 2):
                    anchor_set.add(tuple(a.tolist()))
    
    # 将集合转换为排序后的列表以保持一致性
    unique_anchors = sorted(list(anchor_set), key=lambda x: (x[0], x[1]))
    return unique_anchors

def print_anchors(model):
    unique_anchors = find_anchor_grid(model)
    # Print anchors in desired format
    formatted_anchors = 'anchors = torch.tensor([\n'
    for anchor in unique_anchors:
        formatted_anchors += f'    [[{anchor[0]:.5f}, {anchor[1]:.5f}]],\n'
    formatted_anchors += '])'
    print(formatted_anchors)

def get_anchors(args):
    device = select_device(args.device)
    model = DetectMultiBackend(args.pt, device=device)
    unique_anchors = find_anchor_grid(model)
    # 将列表转换为PyTorch张量，并调整形状以符合后续使用
    return torch.tensor(unique_anchors).view(-1, 1, 2)

if __name__ == '__main__':
    args = parse_args()
    device = select_device(args.device)
    # Load model
    model = DetectMultiBackend(args.pt, device=device)
    # Print anchors
    print_anchors(model)
