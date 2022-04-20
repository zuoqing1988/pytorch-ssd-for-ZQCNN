import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from collections import OrderedDict
from ..utils import box_utils_zq as box_utils
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from collections import namedtuple


def conv_only(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=True)),
        ])
    )

def conv_bn(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
        (name+'/bn', nn.BatchNorm2d(oup))
        ])
    )

def conv_bn_relu(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
        (name+'/bn', nn.BatchNorm2d(oup)),
        (name+'/relu', nn.ReLU(inplace=True))
        ])
    )

def conv_dw(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True)),
        ])
    )

def conv_dw_bn(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
        (name+'/dw/bn', nn.BatchNorm2d(inp)),
        ])
    )

def conv_dw_bn_relu(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
        (name+'/dw/bn', nn.BatchNorm2d(inp)),
        (name+'/dw/relu', nn.ReLU(inplace=True))
        ])
    )

def conv_sep(inp, oup, name):
    return nn.Sequential(OrderedDict([
        (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=True)),
        ])
    )

def conv_sep_bn(inp, oup, name):
    return nn.Sequential(OrderedDict([
        (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        (name+'/sep/bn', nn.BatchNorm2d(oup)),
        ])
    )

def conv_sep_bn_relu(inp, oup, name):
    return nn.Sequential(OrderedDict([
        (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        (name+'/sep/bn', nn.BatchNorm2d(oup)),
        (name+'/sep/relu', nn.ReLU(inplace=True))
        ])
    )

def conv_dw_bn_relu_sep_bn_relu(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
        (name+'/dw/bn', nn.BatchNorm2d(inp)),
        (name+'/dw/relu', nn.ReLU(inplace=True)),
        (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        (name+'/sep/bn', nn.BatchNorm2d(oup)),
        (name+'/sep/relu', nn.ReLU(inplace=True))
        ])
    )

def conv_dw_bn_relu_sep_bn(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
        (name+'/dw/bn', nn.BatchNorm2d(inp)),
        (name+'/dw/relu', nn.ReLU(inplace=True)),
        (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        (name+'/sep/bn', nn.BatchNorm2d(oup)),
        ])
    )

def conv_dw_bn_relu_sep(inp, oup, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
        (name+'/dw/bn', nn.BatchNorm2d(inp)),
        (name+'/dw/relu', nn.ReLU(inplace=True)),
        (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=True))
        ])
    )

def conv_dw_bn_relu_sep_bn_relu_sep(inp, oup, tmp_channel, stride, name):
    return nn.Sequential(OrderedDict([
        (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
        (name+'/dw/bn', nn.BatchNorm2d(inp)),
        (name+'/dw/relu', nn.ReLU(inplace=True)),
        (name+'/sep', nn.Conv2d(inp, tmp_channel, 1, 1, 0, bias=False)),
        (name+'/sep/bn', nn.BatchNorm2d(tmp_channel)),
        (name+'/sep/relu', nn.ReLU(inplace=True)),
        (name+'/sep1', nn.Conv2d(tmp_channel, oup, 1, 1, 0, bias=True))
        ])
    )


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.array([[0,0,1,1]],np.float32)
            gt_labels = np.array([0],np.int64)
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels, None, None


class MatchPrior2(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold, iou_threshold_low):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.iou_threshold_low = iou_threshold_low

    def __call__(self, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.array([[0,0,1,1]],np.float32)
            gt_labels = np.array([0],np.int64)
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels, labels_low = box_utils.assign_priors2(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold, self.iou_threshold_low)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels, labels_low, None


class MatchPrior3(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold, iou_threshold_mid, iou_threshold_low):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.iou_threshold_mid = iou_threshold_mid
        self.iou_threshold_low = iou_threshold_low

    def __call__(self, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.array([[0,0,1,1]],np.float32)
            gt_labels = np.array([0],np.int64)
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels, labels_mid, labels_low = box_utils.assign_priors3(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold, self.iou_threshold_mid, self.iou_threshold_low)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels, labels_mid, labels_low

class MatchPrior4(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold_list, iou_threshold_mid_list, iou_threshold_low_list):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold_list = torch.Tensor(iou_threshold_list)
        self.iou_threshold_mid_list = torch.Tensor(iou_threshold_mid_list)
        self.iou_threshold_low_list = torch.Tensor(iou_threshold_low_list)

    def __call__(self, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.array([[0,0,1,1]],np.float32)
            gt_labels = np.array([0],np.int64)
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels, labels_mid, labels_low = box_utils.assign_priors4(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold_list, self.iou_threshold_mid_list, self.iou_threshold_low_list)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels, labels_mid, labels_low

    def set_iou_threshs(self, iou_threshold_list, iou_threshold_mid_list, iou_threshold_low_list):
        self.iou_threshold_list = torch.Tensor(iou_threshold_list)
        self.iou_threshold_mid_list = torch.Tensor(iou_threshold_mid_list)
        self.iou_threshold_low_list = torch.Tensor(iou_threshold_low_list)

class MatchPrior5(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold_list, iou_threshold_mid_list, iou_threshold_low_list, sub_peer_range):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold_list = torch.Tensor(iou_threshold_list)
        self.iou_threshold_mid_list = torch.Tensor(iou_threshold_mid_list)
        self.iou_threshold_low_list = torch.Tensor(iou_threshold_low_list)
        self.sub_peer_range = sub_peer_range

    def __call__(self, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.array([[0,0,1,1]],np.float32)
            gt_labels = np.array([0],np.int64)
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels, labels_mid, labels_low, is_peer = box_utils.assign_priors5(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold_list, self.iou_threshold_mid_list, self.iou_threshold_low_list, self.sub_peer_range)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels, labels_mid, labels_low, is_peer

    def set_iou_threshs(self, iou_threshold_list, iou_threshold_mid_list, iou_threshold_low_list, sub_peer_range):
        self.iou_threshold_list = torch.Tensor(iou_threshold_list)
        self.iou_threshold_mid_list = torch.Tensor(iou_threshold_mid_list)
        self.iou_threshold_low_list = torch.Tensor(iou_threshold_low_list)
        self.sub_peer_range = sub_peer_range


