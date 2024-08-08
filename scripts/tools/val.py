import os
import torch
import numpy as np
import argparse

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def process_batch(detections, labels, iouv): 
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
          correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # print(labels[:, 1:])
    # print(iou)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match 
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xywh2xyxy1(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]  # top left x
    y[..., 1] = x[..., 1]  # top left y
    y[..., 2] = x[..., 0] + x[..., 2]  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]  # bottom right y
    return y

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def ap_per_class(tp, conf, pred_cls, target_cls,plot=None,eps=1e-7):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]   #按置信度排序

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)  #返回列表和 每个元素在旧列表里各自出现了几次
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c   # 获取所有该类的预测值
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)  #获取该类的所有pred结果，并生成累加数组
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve  通过累加数组除以gt得到曲线  gt一定，tpc为不同conf下的tp序列
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        # 插值算法，通过conf和recall的关系俩对给定的px进行插值，表示不同conf下的的recall
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):  # 对于10个iou阈值进行使用
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    # names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    # names = dict(enumerate(names))  # to dict

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def read_info(txt_path,index,dtype='label'):
    infos = []
    # if not os.path.exists(txt_path):
    #     print(txt_path)
    #     print('1111')
    #     txt_path= txt_path.replace('(2)',' (2)')
    with open(txt_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            info =line.replace('\n','').split(' ')
            info = [float(i) for i in info] # astype float
            if dtype == "label":
                info.insert(0,index)
            infos.append(info)
    return infos

def get_infos(images_path,labels_path,preds_path):
    labels =[]
    preds = []
    shapes = []
    for index,image_name in enumerate(os.listdir(images_path)):
        if not image_name.endswith('.jpg'):
            continue
        obj_name,_ = os.path.splitext(image_name)
        ann_name  = obj_name+'.txt'
        label = read_info(os.path.join(labels_path,ann_name),index)
        try: 
            pred = read_info(os.path.join(preds_path,ann_name),index)
        except:
            pred = [[index,0.0,0.0,0.0,0.0,0.0,-1]]
        from PIL import Image
        m = Image.open(os.path.join(images_path,image_name))
        shape = m.height,m.width
        labels.extend(label)
        preds.extend(pred)
        shapes.append(shape)
    
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    shapes = torch.tensor(shapes)
    return labels,preds,shapes

def val(targets,preds,shapes,device='cpu'):
    stats = []
    max_index = int(preds[-1,0])+1
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95 iouv = torch.tensor([0.5])
    niou = iouv.numel()
    count = 0
    for i in range(max_index):
        count += 1
        shape = shapes[i]
        # gn= torch.tensor(shape)[[1, 0, 1, 0]]
        gn = shape.clone().detach()[[1, 0, 1, 0]]
        labels = targets[targets[:,0] == i,1:]
        pred = preds[preds[:,0]==i,1:]  #去index
        nl, npr = labels.shape[0], pred.shape[0]
        correct = torch.zeros(npr,niou,dtype=torch.bool,device=device)
        if npr == 0:
            if nl:
                stats.append((correct,*torch.zeros((2,0),device=device),labels[:,0]))
            continue
        predn = pred.clone()
        predn[:,:4]= predn[:,:4]*gn
        pbox = xywh2xyxy1(predn[:, :4])  # target boxes
        # print(count,pbox)
        predns = torch.cat((pbox, predn[:,4:]), 1)
        if nl:
            tbox= labels[:, 1:5]*gn
            tbox = xywh2xyxy(tbox)  # target boxes
            # print(count,tbox)
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predns, labelsn, iouv)
        # print(pbox,tbox)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # print(len(stats))
    # print(stats[0].any())
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)        
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        print(('%22s' + '%11s' * 3) % ('Class', 'Images',  'P', 'R'))
        print(('%22s' + '%11s' * 3) % ('all', count,  mp.round(3), mr.round(3)))   
        print(('%22s' + '%11s' * 3) % ('police', count,  p[0].round(3), r[0].round(3)))   
        print(('%22s' + '%11s' * 3) % ('deliveryman', count,  p[1].round(3), r[1].round(3)))   
        print(('%22s' + '%11s' * 3) % ('imcar', count,  p[2].round(3), r[2].round(3)))        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images and labels.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory of images')
    parser.add_argument('--label_true_dir', type=str, required=True, help='Directory of true labels')
    parser.add_argument('--label_pred_dir', type=str, required=True, help='Directory of predicted labels')

    args = parser.parse_args()
    
    res = get_infos(args.img_dir, args.label_true_dir, args.label_pred_dir)

    val(*res)

