import cv2
import numpy as np
import os
import math
import argparse
from PIL import Image

def process_and_save_batch(images, output_path, batch_idx):
    # Convert list of images to a numpy array
    image_batch = np.stack(images, axis=0).astype(np.float32)  # Shape: (batch_size, H, W, C)

    # Create raw file name for the batch
    raw_file_name = f"batch_{str(batch_idx).zfill(5)}.raw"
    raw_path = os.path.join(output_path, raw_file_name)
    
    # Save the batch as a raw file
    with open(raw_path, 'wb') as raw_file:
        raw_file.write(image_batch.tobytes())

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def main(root, output, image_size, batch_size):
    os.makedirs(output, exist_ok=True)
    image_list = []
    batch_idx = 0

    for i in os.listdir(root):
        im_path = os.path.join(root, i)
        img = cv2.imread(im_path)
        
        if image_size == (512, 960):
            img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
            img = img[:512, :960, :]
            img = np.ascontiguousarray(img).astype(np.float32)
            img = img[:, :, ::-1]  # Convert BGR to RGB
            img /= 255.0
        else:
            img, ratio, pad = letterbox(img, image_size, auto=False)
            img = cv2.resize(img, (640, 640))
            img = img[:, :, ::-1]  # Convert BGR to RGB
            img = np.ascontiguousarray(img).astype(np.float32)
            img = img / 255.0

        # Add the processed image to the list
        image_list.append(img)
        
        # If the list has reached the batch size, save the batch and reset the list
        if len(image_list) == batch_size:
            process_and_save_batch(image_list, output, batch_idx)
            image_list = []
            batch_idx += 1
    
    # Save any remaining images as the last batch
    if image_list:
        process_and_save_batch(image_list, output, batch_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and save as raw format.')
    parser.add_argument('input_path', help='Directory containing images to process')
    parser.add_argument('output_path', help='Directory to save processed images')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 960], help='Size to which images are resized (width height)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images per batch')
    args = parser.parse_args()
    print("raw size:", args.img_size)
    main(args.input_path, args.output_path, tuple(args.img_size), args.batch_size)
