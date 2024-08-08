import numpy as np
import cv2
import qnn_infer
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference and process outputs.")
    parser.add_argument("--input_file", type=str, default='input_im.txt', help="File containing list of images to process.")
    parser.add_argument("--results_dir", type=str, default='./results', help="Directory to store the results.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Threshold for considering detection.")
    parser.add_argument("--nms_threshold", type=float, default=0.2, help="NMS threshold.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 640], help="Image size to use for scaling coordinates.")
    parser.add_argument("--draw", action='store_true', help="Whether to draw boxes and labels on the images.")
    # qnn_infer
    parser.add_argument('--stride', nargs='+', type=int, default=[16, 32, 64], help='List of strides [16, 32, 64]')
    parser.add_argument('--anchor_num', type=int, default=1, help='Number of anchors')
    parser.add_argument('--class_num', type=int, default=3, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images per batch')
    # anchors
    parser.add_argument('--pt', type=str, default='./weights/best.pt', help='path to weights file')
    parser.add_argument('--device', type=str, default='0', help='device id (i.e. 0 or 0,1) or cpu')
    # reader_result
    parser.add_argument("--infer_tmp", type=str, required=True, help="Root path where the output files are stored.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    qnn_infer.init(parser.parse_args())
    return parser.parse_args()

def detect(num, conf_threshold, nms_threshold):
    all_preds = qnn_infer.pred_res(num)

    batch_boxes = []
    batch_confes = []
    batch_classes = []

    for pred in all_preds:
        boxes = []
        classIds = []
        confidences = []
        for detection in pred[0]:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID] * detection[4]
            if confidence > conf_threshold:
                box = detection[0:4]
                (centerX, centerY, width, height) = box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        pred_boxes = []
        pred_confes = []
        pred_classes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                confidence = confidences[i]
                if confidence >= 0.1:
                    pred_boxes.append(boxes[i])
                    pred_confes.append(confidence)
                    pred_classes.append(classIds[i])
        
        batch_boxes.append(pred_boxes)
        batch_confes.append(pred_confes)
        batch_classes.append(pred_classes)
    
    return batch_boxes, batch_confes, batch_classes

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    坐标还原
    :param img1_shape: 旧图像的尺寸
    :param coords: 坐标
    :param img0_shape:新图像的尺寸
    :param ratio_pad: 填充率
    :return:
    """
    if ratio_pad is None:  # 从img0_shape中计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain=old/new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    """
    图片的边界处理
    :param boxes: 检测框
    :param img_shape: 图片的尺寸
    :return:
    """
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # x2

def read_list(input_file):
    input_list = []
    with open(input_file,'r') as f:
        for i in f.readlines():
            input_list.append(i.strip())
    return input_list

def create_directory(dir_path):
    """ 创建目录如果它不存在 """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    args = parse_args()
    batch = args.batch_size
    create_directory(args.results_dir)  # 确保结果目录存在
    create_directory(os.path.join(args.results_dir, 'pred_labels'))  # 确保标签目录存在
    if args.draw:
        create_directory(os.path.join(args.results_dir, 'images'))  # 确保标签目录存在
    input_list = read_list(args.input_file)
    for i in range(0, len(input_list), batch):
        batch_inputs = input_list[i:i + batch]
        num = i // batch
        # print(num)
        pred_boxes, pred_confes, pred_classes = detect(num, args.confidence_threshold, args.nms_threshold)

        for j, im in enumerate(batch_inputs):
            im_name = os.path.basename(im)
            print(im_name)
            # 获取当前图像的预测结果
            boxes, confes, classes = pred_boxes[j], pred_confes[j], pred_classes[j]

            image = cv2.imread(im)
            # print(image.shape)
            pred_labels_path = os.path.join(args.results_dir, 'pred_labels/') + im_name.replace('.jpg', '.txt')
            with open(pred_labels_path, 'w') as f:
                pass  # 打开文件并立即关闭，目的是清空文件

            h, w, c = image.shape
            if len(boxes) > 0:
                for k, _ in enumerate(boxes):
                    box = boxes[k]
                    left, top, width, height = box[0], box[1], box[2], box[3]
                    # print(f'image:({left},{top}),({width},{height})')
                    box = (left, top, left + width, top + height)
                    box = np.squeeze(
                        scale_coords(args.image_size, np.expand_dims(box, axis=0).astype("float"), image.shape[:2]).round(), axis=0).astype("int")
                    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                    # 保存 pred_labels
                    with open(pred_labels_path, 'a') as f:
                        f.write(str(x0 / w) + ' ' + str(y0 / h) + ' ' + str((x1 - x0) / w) + ' ' + str((y1 - y0) / h) + ' ' + str(float(confes[k])) + ' ' + str(int(classes[k])) + '\n')
                    if args.draw:
                        # print(f'image:({x0},{y0}),({x1},{y1})')
                        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)
                        cv2.putText(image, '{0}--{1:.2f}'.format(classes[k], confes[k]), (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
                        # 修改保存路径到指定的结果目录
                        output_path = os.path.join(os.path.join(args.results_dir, 'images'), im_name)
                        cv2.imwrite(output_path, image)

if __name__ == '__main__':
    main()