import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from .utils import box_utils_zq as box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, priors, center_variance, size_variance, is_test=False, with_softmax=False, device=None,fp16=False):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.fp16 = fp16
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.with_softmax = with_softmax
        self.priors = priors
        self.center_variance = center_variance
        self.size_variance = size_variance

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.priors = self.priors.to(self.device)
            
    def forward(self, x: torch.Tensor, print_score_and_box = False) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        with torch.cuda.amp.autocast(self.fp16):
            for end_layer_index in self.source_layer_indexes:
                if isinstance(end_layer_index, GraphPath):
                    path = end_layer_index
                    end_layer_index = end_layer_index.s0
                    added_layer = None
                elif isinstance(end_layer_index, tuple):
                    added_layer = end_layer_index[1]
                    end_layer_index = end_layer_index[0]
                    path = None
                else:
                    added_layer = None
                    path = None
                for layer in self.base_net[start_layer_index: end_layer_index]:
                    x = layer(x)
                if added_layer:
                    y = added_layer(x)
                else:
                    y = x
                if path:
                    sub = getattr(self.base_net[end_layer_index], path.name)
                    for layer in sub[:path.s1]:
                        x = layer(x)
                    y = x
                    for layer in sub[path.s1:]:
                        x = layer(x)
                    end_layer_index += 1
                start_layer_index = end_layer_index
                confidence, location = self.compute_header(header_index, y)
                header_index += 1
                confidences.append(confidence)
                locations.append(location)

            for layer in self.base_net[end_layer_index:]:
                x = layer(x)

            for layer in self.extras:
                x = layer(x)
                confidence, location = self.compute_header(header_index, x)
                header_index += 1
                confidences.append(confidence)
                locations.append(location)

            confidences = torch.cat(confidences, 1)
            locations = torch.cat(locations, 1)
        
        # fp16 --> float
        if self.fp16:
            confidences = confidences.float()
            locations = locations.float()

        if self.is_test:
            if print_score_and_box:
                scores = confidences[0]
                cpu_device = torch.device("cpu")
                scores = scores.to(cpu_device)
                print('\n\nscores')
                print(scores.shape)
                score_h, score_w = scores.shape
                for i in range(score_h):
                    line = ''
                    for j in range(score_w):
                        line = line + '%12.5f '%(scores[i][j])
                    print(line)
                    
                locs = locations[0]
                locs = locs.to(cpu_device)
                print('\n\nlocs')
                print(locs.shape)
                loc_h, loc_w = locs.shape
                for i in range(loc_h):
                    line = ''
                    for j in range(loc_w):
                        line = line + '%12.5f '%(locs[i][j])
                    print(line)
            
                print('\n\npriors')
                print(self.priors.shape)
                priors_h, priors_w = self.priors.shape
                for i in range(priors_h):
                    line = ''
                    for j in range(priors_w):
                        line = line + '%12.5f '%(self.priors[i][j])
                print(line)
            
            
            #print(self.device)
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.center_variance, self.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        elif self.with_softmax:
            #print(self.device)
            confidences = F.softmax(confidences, dim=2)
            return confidences, locations
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

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

