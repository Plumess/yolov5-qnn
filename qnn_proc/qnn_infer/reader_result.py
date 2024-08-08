import onnx
import argparse
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Process ONNX outputs.")
    parser.add_argument("--infer_tmp", type=str, required=True, help="Root path where the output files are stored.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    return parser.parse_args()

def init(args):
    global output_info, root_path
    output_info = load_model_outputs(args.onnx)
    root_path = args.infer_tmp

def load_model_outputs(onnx_file_path):
    model = onnx.load(onnx_file_path)
    output_info = []
    cnt = 0
    for output in model.graph.output:
        if cnt == 0:
            name = output.name
        else:
            name = "_" + output.name
        shape = tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
        output_info.append((name, shape))
        # print(f"Output Layer Name: {name}, Shape: {shape}")  # 显示每个输出层的名称和形状
        cnt += 1
    return output_info

def reconstruct(num=0):
    outputs = []
    for name, shape in output_info:
        raw_path = os.path.join(root_path, f"Result_{str(num)}/{name}.raw")
        output_data = np.fromfile(raw_path, dtype='float32').reshape(shape)
        # print(f"Loaded and reshaped {name} from {raw_path} with shape {shape}")  # 显示每次文件读取和数据重塑的信息
        outputs.append(output_data)
    return outputs

if __name__ == "__main__":
    args = parse_args()
    init(args)
    print(f"Loading ONNX model from {args.onnx}")
    output_info = load_model_outputs(args.onnx)
    print("Starting reconstruction of outputs...")
    result = reconstruct(0)
    # print("Reconstruction complete. Results:")
    # print(result)
