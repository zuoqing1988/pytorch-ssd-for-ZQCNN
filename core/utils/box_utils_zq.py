import collections
import torch
import itertools
from typing import List
import math

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size_x', 'feature_map_size_y', 'shrinkage_x', 'shrinkage_y', 'box_sizes', 'aspect_ratios'])

def get_num_boxes_of_aspect_ratios(aspect_ratios):
    return 2 + 2*len(aspect_ratios)

def generate_ssd_priors(specs: List[SSDSpec], image_size_x, image_size_y, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 38, 8, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 19, 16, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 10, 32, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 5, 64, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 3, 100, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 1, 300, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale_x = image_size_x / spec.shrinkage_x
        scale_y = image_size_y / spec.shrinkage_y
        for j in range(spec.feature_map_size_y):
            for i in range(spec.feature_map_size_x):
                x_center = (i + 0.5) / scale_x
                y_center = (j + 0.5) / scale_y

                # small sized square box
                size = spec.box_sizes.min
                h = size / image_size_y
                w = size / image_size_x
                priors.append([
                    x_center,
                    y_center,
                    w,
                    h
                ])

                # big sized square box
                size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
                h_1 = size / image_size_y
                w_1 = size / image_size_x
                priors.append([
                    x_center,
                    y_center,
                    w_1,
                    h_1
                ])

                # change h/w ratio of the small sized box
                for ratio in spec.aspect_ratios:
                    ratio = math.sqrt(ratio)
                    priors.append([
                        x_center,
                        y_center,
                        w * ratio,
                        h / ratio
                    ])
                    priors.append([
                        x_center,
                        y_center,
                        w / ratio,
                        h * ratio
                    ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    #print(gt_boxes.shape,ious.shape)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    #print('!!')
    # size: num_targets
    #best_prior_per_target, best_prior_per_target_index = ious.max(0)
    #print('!!!')
    #for target_index, prior_index in enumerate(best_prior_per_target_index):
    #    best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    #best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels

def assign_priors3(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold, iou_threshold_mid, iou_threshold_low):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
        labels_low (num_priros): labels for priors of low iou_thresh.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    #print(gt_boxes.shape,ious.shape)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    #print('!!')
    # size: num_targets
    #best_prior_per_target, best_prior_per_target_index = ious.max(0)
    #print('!!!')
    #for target_index, prior_index in enumerate(best_prior_per_target_index):
    #    best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    #best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels_mid = gt_labels[best_target_per_prior_index]
    labels_low = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the background id
    labels_mid[best_target_per_prior < iou_threshold_mid] = 0  # the background id
    labels_low[best_target_per_prior < iou_threshold_low] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels, labels_mid, labels_low

def assign_priors4(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold_list, iou_threshold_mid_list, iou_threshold_low_list):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
        labels_low (num_priros): labels for priors of low iou_thresh.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    #print(gt_boxes.shape,ious.shape)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    #print('!!')
    # size: num_targets
    #best_prior_per_target, best_prior_per_target_index = ious.max(0)
    #print('!!!')
    #for target_index, prior_index in enumerate(best_prior_per_target_index):
    #    best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    #best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    #print(type(gt_labels))
    labels = gt_labels[best_target_per_prior_index]
    labels_mid = gt_labels[best_target_per_prior_index]
    labels_low = gt_labels[best_target_per_prior_index]
    iou_threshold = iou_threshold_list[labels-1]
    iou_threshold_mid = iou_threshold_mid_list[labels-1]
    iou_threshold_low = iou_threshold_low_list[labels-1]
    labels[best_target_per_prior < iou_threshold] = 0  # the background id
    labels_mid[best_target_per_prior < iou_threshold_mid] = 0  # the background id
    labels_low[best_target_per_prior < iou_threshold_low] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels, labels_mid, labels_low

def assign_priors5(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold_list, iou_threshold_mid_list, iou_threshold_low_list, sub_peer_range):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
        labels_low (num_priros): labels for priors of low iou_thresh.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    #print(gt_boxes.shape,ious.shape)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    is_peer = gt_labels[best_target_per_prior_index] * 0
    #print('!!')
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)
    #print(best_prior_per_target)
    #print(best_prior_per_target_index)
    #print('!!!')
    
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        if gt_labels[target_index] == 0:
            continue
        is_peer[prior_index] = 1
        iou_val = best_prior_per_target[target_index]
        #print(target_index,iou_val,gt_labels[target_index],gt_labels)
        #sub peer
        sub_peer_mask = (gt_labels[best_target_per_prior_index] == gt_labels[target_index]) & (best_target_per_prior >= iou_val-sub_peer_range)
        is_peer[sub_peer_mask] = 1
    #print(is_peer)

    labels = gt_labels[best_target_per_prior_index]
    labels_mid = gt_labels[best_target_per_prior_index]
    labels_low = gt_labels[best_target_per_prior_index]
    iou_threshold = iou_threshold_list[labels-1]
    iou_threshold_mid = iou_threshold_mid_list[labels-1]
    iou_threshold_low = iou_threshold_low_list[labels-1]
    labels[best_target_per_prior < iou_threshold] = 0  # the background id
    labels_mid[best_target_per_prior < iou_threshold_mid] = 0  # the background id
    labels_low[best_target_per_prior < iou_threshold_low] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels, labels_mid, labels_low, is_peer


def assign_priors_x(gt_boxes, gt_labels, corner_form_priors, iou_threshold, iou_threshold_mid, iou_threshold_low):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
        labels_low (num_priros): labels for priors of low iou_thresh.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    #print(gt_boxes.shape,ious.shape)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    #print('!!')
    # size: num_targets
    #best_prior_per_target, best_prior_per_target_index = ious.max(0)
    #print('!!!')
    #for target_index, prior_index in enumerate(best_prior_per_target_index):
    #    best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    # best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors

    #print(best_target_per_prior_index)
    #print(gt_labels.shape)
    #print(gt_labels[0])
    objs = gt_labels[best_target_per_prior_index]
    #print(best_target_per_prior_index.shape)
    #print(objs.shape)
    #print('--------------------!')
    #print(objs)
    objs[best_target_per_prior < iou_threshold] = 0  # the background id
    objs[best_target_per_prior >= iou_threshold] = 1  # obj 
    #print(objs)

    objs_mid = gt_labels[best_target_per_prior_index]
    objs_mid[best_target_per_prior < iou_threshold_mid] = 0  # the background id
    objs_mid[best_target_per_prior >= iou_threshold_mid] = 1  # obj

    objs_low = gt_labels[best_target_per_prior_index]
    objs_low[best_target_per_prior < iou_threshold_low] = 0  # the background id
    objs_low[best_target_per_prior >= iou_threshold_low] = 1  # obj 

    labels = gt_labels[best_target_per_prior_index] - 1
    labels[best_target_per_prior < iou_threshold] = -1  # the background id
    #for i in range(labels.shape[0]):
    #    print('%6d %8.4f %3d %3d %s'%(i, best_target_per_prior[i], objs[i],labels[i],"True" if (objs[i] > 0) == (labels[i] >= 0) else "False"))
    #labels[best_target_per_prior >= iou_threshold] = 1  # obj 
    #print(labels)
    #print('--------------------!!')
    #print(objs)
    #print('--------------------!!!')
    #labels_mid = gt_labels[best_target_per_prior_index] - 1
    #labels_low = gt_labels[best_target_per_prior_index] - 1
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, objs, objs_mid, objs_low, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    
    #enable pure negative image
    num_neg[num_neg < 2] = 2

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


hard_negative_mining7_step = 0

def hard_negative_mining7(ori_loss, pred_label, labels, labels_mid, labels_low, neg_pos_ratio_mid, neg_pos_ratio_low):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        labels_low (N, num_priors): the labels of low iou_thresh.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    global hard_negative_mining7_step


    pos_mask = labels > 0
    pos_part_mask_mid = labels_mid > 0
    pos_part_mask_low = labels_low > 0
    part_mask = (labels == 0) & (labels_mid > 0)
    neg_mask = labels_low == 0

    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg_mid = num_pos * neg_pos_ratio_mid
    num_neg_low = num_pos * neg_pos_ratio_low
    #print('num_pos=%d, num_neg_mid=%d, num_neg_low=%d'%(num_pos.long().sum(),num_neg_mid.long().sum(),num_neg_low.long().sum()))
    
    #enable pure negative image
    #num_neg_low[num_neg_low < 2] = 2
    #print('num_pos=%d, num_neg_mid=%d, num_neg_low=%d'%(num_pos.long().sum(),num_neg_mid.long().sum(),num_neg_low.long().sum()))

    loss = ori_loss.clone()
    loss[pos_part_mask_mid] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask_mid = orders < num_neg_mid

    loss = ori_loss.clone()
    loss[pos_part_mask_low] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask_low = orders < num_neg_low


    # 
    if hard_negative_mining7_step < 100:
        neg_hard_loss_thresh = math.inf
    elif hard_negative_mining7_step < 1000:
        neg_hard_loss_thresh = -math.log(0.6)
    elif hard_negative_mining7_step < 5000:
        neg_hard_loss_thresh = -math.log(0.7)
    elif hard_negative_mining7_step < 10000:
        neg_hard_loss_thresh = -math.log(0.75)
    elif hard_negative_mining7_step < 20000:
        neg_hard_loss_thresh = -math.log(0.8)
    else:
        neg_hard_loss_thresh = -math.log(0.85)
    neg_hard_mask = (ori_loss > neg_hard_loss_thresh) & neg_mask
    
    neg_hard_mask_sum = neg_hard_mask.long().sum(dim=1,keepdim=True).sum() 
    pos_mask_sum = num_pos.sum()
    if neg_hard_mask_sum > pos_mask_sum * 5:
        num_neg_hard = num_pos * 5
        loss = ori_loss.clone()
        loss[~neg_hard_mask] = -math.inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_hard_mask = orders < num_neg_hard
        if hard_negative_mining7_step % 100 == 0:
            print('neg_hard_num %d too many than pos_num %d, forced to 5*pos_num'%(neg_hard_mask_sum,pos_mask_sum))


    #print('num_pos=%d, num_neg_mid=%d, num_neg_low=%d, num_neg_hard=%d'%(num_pos.long().sum(),num_neg_mid.long().sum(),num_neg_low.long().sum(),num_neg_hard))

    # wrong pred
    #print('part_mask')
    #print(part_mask)
    wrong_pred_mask = (pred_label != labels_mid) & (pred_label != 0)
    #print('wrong_pred_mask')
    #print(wrong_pred_mask)
    wrong_part_mask = part_mask & wrong_pred_mask
    #print('pred_label==')
    #print(pred_label)
    #print('=========')
    #print(labels_mid)
    total_neg_mask = wrong_part_mask | neg_mask_mid | neg_mask_low | neg_hard_mask 


    if hard_negative_mining7_step % 100 == 0:
        total_num_neg = total_neg_mask.long().sum(dim=1, keepdim=True).sum()
    
        neg_mask_0 = total_neg_mask & (pred_label == 0)
        neg_mask_1 = total_neg_mask & (pred_label == 1)
        neg_mask_2 = total_neg_mask & (pred_label == 2)
        neg_mask_3 = total_neg_mask & (pred_label == 3)
        num_neg_0 = neg_mask_0.long().sum(dim=1, keepdim=True).sum()
        num_neg_1 = neg_mask_1.long().sum(dim=1, keepdim=True).sum()
        num_neg_2 = neg_mask_2.long().sum(dim=1, keepdim=True).sum()
        num_neg_3 = neg_mask_2.long().sum(dim=1, keepdim=True).sum()

        pos_mask_1 = pos_mask & (labels == 1)
        pos_mask_2 = pos_mask & (labels == 2)
        pos_mask_3 = pos_mask & (labels == 3)
        num_pos_1 = pos_mask_1.long().sum(dim=1, keepdim=True).sum()
        num_pos_2 = pos_mask_2.long().sum(dim=1, keepdim=True).sum()
        num_pos_3 = pos_mask_3.long().sum(dim=1, keepdim=True).sum()

        num_wrong_part = wrong_part_mask.long().sum(dim=1, keepdim=True).sum()
        num_neg_mid = neg_mask_mid.long().sum(dim=1, keepdim=True).sum()
        num_neg_low = neg_mask_low.long().sum(dim=1, keepdim=True).sum()
        num_neg_hard = neg_hard_mask.long().sum(dim=1, keepdim=True).sum()

        print('pos_num = %3d, %3d, %3d, neg_num = %3d, %3d, %3d, %3d,  | %d, %d, %d, %d, wrong_part=%d'%(num_pos_1,num_pos_2,num_pos_3, \
                                                           num_neg_0,num_neg_1,num_neg_2,num_neg_3,\
                                                           total_num_neg, num_neg_mid,num_neg_low,num_neg_hard,num_wrong_part))
    hard_negative_mining7_step += 1
    return pos_mask | total_neg_mask


hard_negative_mining9_step = 0

def hard_negative_mining9(ori_loss, pred_label, labels, labels_mid, labels_low, is_peer, neg_pos_ratio_mid, neg_pos_ratio_low):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        labels_low (N, num_priors): the labels of low iou_thresh.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    global hard_negative_mining9_step


    pos_mask = labels > 0
    pos_peer_mask = pos_mask & (is_peer > 0)
    pos_part_mask_mid = labels_mid > 0
    pos_part_mask_low = labels_low > 0
    part_mask = (labels == 0) & (labels_mid > 0)
    neg_mask = labels_low == 0

    num_pos = pos_peer_mask.long().sum(dim=1, keepdim=True)
    num_neg_mid = num_pos * neg_pos_ratio_mid
    num_neg_low = num_pos * neg_pos_ratio_low
    #print('num_pos=%d, num_neg_mid=%d, num_neg_low=%d'%(num_pos.long().sum(),num_neg_mid.long().sum(),num_neg_low.long().sum()))
    
    #enable pure negative image
    #num_neg_low[num_neg_low < 2] = 2
    #print('num_pos=%d, num_neg_mid=%d, num_neg_low=%d'%(num_pos.long().sum(),num_neg_mid.long().sum(),num_neg_low.long().sum()))

    loss = ori_loss.clone()
    loss[pos_part_mask_mid] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask_mid = orders < num_neg_mid

    loss = ori_loss.clone()
    loss[pos_part_mask_low] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask_low = orders < num_neg_low


    # 
    if hard_negative_mining9_step < 100:
        neg_hard_loss_thresh = math.inf
    elif hard_negative_mining9_step < 1000:
        neg_hard_loss_thresh = -math.log(0.6)
    elif hard_negative_mining9_step < 5000:
        neg_hard_loss_thresh = -math.log(0.7)
    elif hard_negative_mining9_step < 10000:
        neg_hard_loss_thresh = -math.log(0.75)
    elif hard_negative_mining9_step < 20000:
        neg_hard_loss_thresh = -math.log(0.8)
    else:
        neg_hard_loss_thresh = -math.log(0.85)
    neg_hard_mask = (ori_loss > neg_hard_loss_thresh) & neg_mask
    
    neg_hard_mask_sum = neg_hard_mask.long().sum(dim=1,keepdim=True).sum() 
    pos_mask_sum = num_pos.sum()
    if neg_hard_mask_sum > pos_mask_sum * 5:
        num_neg_hard = num_pos * 5
        loss = ori_loss.clone()
        loss[~neg_hard_mask] = -math.inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_hard_mask = orders < num_neg_hard
        if hard_negative_mining9_step % 100 == 0:
            print('neg_hard_num %d too many than pos_num %d, forced to 5*pos_num'%(neg_hard_mask_sum,pos_mask_sum))


    #print('num_pos=%d, num_neg_mid=%d, num_neg_low=%d, num_neg_hard=%d'%(num_pos.long().sum(),num_neg_mid.long().sum(),num_neg_low.long().sum(),num_neg_hard))

    # wrong pred
    #print('part_mask')
    #print(part_mask)
    wrong_pred_mask = (pred_label != labels_mid) & (pred_label != 0)
    #print('wrong_pred_mask')
    #print(wrong_pred_mask)
    wrong_part_mask = part_mask & wrong_pred_mask
    #print('pred_label==')
    #print(pred_label)
    #print('=========')
    #print(labels_mid)
    total_neg_mask = wrong_part_mask | neg_mask_mid | neg_mask_low | neg_hard_mask 


    if hard_negative_mining9_step % 100 == 0:
        total_num_neg = total_neg_mask.long().sum(dim=1, keepdim=True).sum()
    
        neg_mask_0 = total_neg_mask & (pred_label == 0)
        neg_mask_1 = total_neg_mask & (pred_label == 1)
        neg_mask_2 = total_neg_mask & (pred_label == 2)
        neg_mask_3 = total_neg_mask & (pred_label == 3)
        num_neg_0 = neg_mask_0.long().sum(dim=1, keepdim=True).sum()
        num_neg_1 = neg_mask_1.long().sum(dim=1, keepdim=True).sum()
        num_neg_2 = neg_mask_2.long().sum(dim=1, keepdim=True).sum()
        num_neg_3 = neg_mask_2.long().sum(dim=1, keepdim=True).sum()

        pos_mask_1 = pos_peer_mask & (labels == 1)
        pos_mask_2 = pos_peer_mask & (labels == 2)
        pos_mask_3 = pos_peer_mask & (labels == 3)
        num_pos_1 = pos_mask_1.long().sum(dim=1, keepdim=True).sum()
        num_pos_2 = pos_mask_2.long().sum(dim=1, keepdim=True).sum()
        num_pos_3 = pos_mask_3.long().sum(dim=1, keepdim=True).sum()

        num_wrong_part = wrong_part_mask.long().sum(dim=1, keepdim=True).sum()
        num_neg_mid = neg_mask_mid.long().sum(dim=1, keepdim=True).sum()
        num_neg_low = neg_mask_low.long().sum(dim=1, keepdim=True).sum()
        num_neg_hard = neg_hard_mask.long().sum(dim=1, keepdim=True).sum()

        print('pos_num = %3d, %3d, %3d, neg_num = %3d, %3d, %3d, %3d,  | %d, %d, %d, %d, wrong_part=%d'%(num_pos_1,num_pos_2,num_pos_3, \
                                                           num_neg_0,num_neg_1,num_neg_2,num_neg_3,\
                                                           total_num_neg, num_neg_mid,num_neg_low,num_neg_hard,num_wrong_part))
    hard_negative_mining9_step += 1
    return pos_peer_mask | total_neg_mask


hard_negative_mining_x_step = 0

def hard_negative_mining_x(ori_obj_loss, objs, objs_mid, objs_low, neg_pos_ratio_mid, neg_pos_ratio_low):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        labels_low (N, num_priors): the labels of low iou_thresh.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    global hard_negative_mining_x_step

    pos_mask = objs > 0
    pos_part_mask_mid = objs_mid > 0
    pos_part_mask_low = objs_low > 0
    part_mask = (objs == 0) & (objs_mid > 0)
    neg_mask = objs_low == 0

    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg_mid = num_pos * neg_pos_ratio_mid
    num_neg_low = num_pos * neg_pos_ratio_low
    
    #enable pure negative image
    #num_neg_low[num_neg_low < 2] = 2

    obj_loss = ori_obj_loss.clone()

    obj_loss[pos_part_mask_mid] = -math.inf
    _, indexes = obj_loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask_mid = orders < num_neg_mid

    obj_loss = ori_obj_loss.clone()

    obj_loss[pos_part_mask_low] = -math.inf
    _, indexes = obj_loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask_low = orders < num_neg_low

    # 
    if hard_negative_mining_x_step < 100:
        neg_hard_loss_thresh = math.inf
    elif hard_negative_mining_x_step < 1000:
        neg_hard_loss_thresh = -math.log(0.6)
    elif hard_negative_mining_x_step < 5000:
        neg_hard_loss_thresh = -math.log(0.7)
    elif hard_negative_mining_x_step < 10000:
        neg_hard_loss_thresh = -math.log(0.75)
    elif hard_negative_mining_x_step < 20000:
        neg_hard_loss_thresh = -math.log(0.8)
    else:
        neg_hard_loss_thresh = -math.log(0.85)
    neg_hard_mask = (ori_obj_loss > neg_hard_loss_thresh) & neg_mask

    neg_hard_mask_sum = neg_hard_mask.long().sum(dim=1,keepdim=True).sum() 
    pos_mask_sum = num_pos.sum()
    if neg_hard_mask_sum > pos_mask_sum * 5:
        num_neg_hard = num_pos * 5
        obj_loss = ori_obj_loss.clone()
        obj_loss[~neg_hard_mask] = -math.inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_hard_mask = orders < num_neg_hard
        if hard_negative_mining_x_step % 100 == 0:
            print('neg_hard_num %d too many than pos_num %d, forced to 5*pos_num'%(neg_hard_mask_sum,pos_mask_sum))

    total_neg_mask = neg_mask_mid | neg_mask_low | neg_hard_mask 

    if hard_negative_mining_x_step % 100 == 0:
        total_num_pos = num_pos.sum()
        total_num_neg = total_neg_mask.long().sum(dim=1, keepdim=True).sum()
        num_neg_mid = neg_mask_mid.long().sum(dim=1, keepdim=True).sum()
        num_neg_low = neg_mask_low.long().sum(dim=1, keepdim=True).sum()
        num_neg_hard = neg_hard_mask.long().sum(dim=1, keepdim=True).sum()
        print('pos_num = %3d, neg_num = %d, %d, %d, %d'%(total_num_pos, \
                                                           total_num_neg, num_neg_mid,num_neg_low,num_neg_hard))
    hard_negative_mining_x_step += 1
    return pos_mask | total_neg_mask

def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1) 


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
         boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def hard_nms_with_id(box_scores, ids, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :], ids[picked]


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)

def nms_with_id(box_scores, ids, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    return hard_nms_with_id(box_scores, ids, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])



