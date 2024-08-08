import os
import numpy as np
import cv2

"""
0 : police
1 : deliveryman
2 : im
"""
path = os.getcwd()
# label_path = "/mnt/sda/dataset/data/val_police/labels"
label_path = "/mnt/sda/plume/yolov5-im-plume/datasets/badtest/labels"
image_path = os.path.join(path, 'runs', 'detect', '0730-finetune-badtest-conf7')
img_folder = image_path
img_list = os.listdir(img_folder)
img_list.sort()
label_folder = label_path

# output_folder = path + '/' + str("output2")
output_folder = image_path + str('_gt')
os.makedirs(output_folder,exist_ok=True)

del image_path
del label_path

colormap = [(0, 255, 0), (132, 112, 255), (0, 191, 255),(0,255,189)]  # 色盘，可根据类别添加新颜色


# 坐标转换
def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2
    if int(label) == 0: #绿
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[0], 2)
    elif int(label) == 1: #红
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[1], 2)
    elif int(label) == 2: #黄
        cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[2], 2)
    elif int(label) == 3:
        cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[3], 2)
    return img


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img
    
def cv_imwrite(file_path,image):
    cv2.imencode('.jpg', image)[1].tofile(file_path)


if __name__ == '__main__':
    import tqdm
    interval = 1
    for i in tqdm.tqdm(range(len(img_list))):
        if i%interval!=0:
            continue
        im_name = img_list[i]
        label_name = im_name.replace('.jpg','.txt')
        image_path = img_folder + "/" + img_list[i]
        label_path = label_folder + "/" + label_name
        try:
            img = cv_imread(str(image_path))
            h, w = img.shape[:2]
        except:
            continue
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        for x in lb:
            img = xywh2xyxy(x, w, h, img)
        cv_imwrite(output_folder + '/' + '{}.jpg'.format(image_path.split('/')[-1][:-4]), img)

