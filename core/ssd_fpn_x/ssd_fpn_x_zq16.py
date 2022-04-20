import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from collections import OrderedDict
from ..utils import box_utils_zq as box_utils
from torch.nn import Conv2d, Sequential, ModuleList
from collections import namedtuple
from .ssd_fpn_x_base import conv_only, conv_bn, conv_bn_relu, conv_dw, conv_dw_bn, conv_dw_bn_relu, conv_sep, conv_sep_bn
from .ssd_fpn_x_base import conv_sep_bn_relu, conv_dw_bn_relu_sep_bn_relu, conv_dw_bn_relu_sep_bn, conv_dw_bn_relu_sep, conv_dw_bn_relu_sep_bn_relu_sep


class SSD_FPN_X_ZQ16(nn.Module):
    def __init__(self, use_gray, num_classes: int, aspect_ratios,
                 priors, center_variance, size_variance, is_test=False, with_softmax=False, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD_FPN_X_ZQ16, self).__init__()

        self.fp16 = True
        self.use_gray = use_gray
        self.num_classes = num_classes
        self.is_test = is_test
        self.with_softmax = with_softmax
        self.priors = priors
        self.center_variance = center_variance
        self.size_variance = size_variance

        in_c = 1 if use_gray else 3
        self.layers = [{'in_c':in_c, 'out_c':32, 'stride':2},
                       {'in_c':32, 'out_c':32, 'stride':1},
                       {'in_c':32, 'out_c':64, 'stride':2},
                       {'in_c':64, 'out_c':64, 'stride':1},
                       {'in_c':64, 'out_c':128, 'stride':2},
                       {'in_c':128, 'out_c':128, 'stride':1},
                       {'in_c':128, 'out_c':128, 'stride':1},
                       {'in_c':128, 'out_c':128, 'stride':1},
                       {'in_c':128, 'out_c':256, 'stride':2},
                       {'in_c':256, 'out_c':256, 'stride':1},
                       {'in_c':256, 'out_c':256, 'stride':1},
                       {'in_c':256, 'out_c':256, 'stride':1},
                       {'in_c':256, 'out_c':512, 'stride':2},
                       {'in_c':512, 'out_c':512, 'stride':1}]

        self.num_layers = len(self.layers)
        self.model = nn.Sequential()
        for i in range(self.num_layers):
            cur_layer = self.layers[i]
            if i == 0:    
                self.model.add_module('conv_%d'%(i+1),conv_bn_relu(cur_layer['in_c'],cur_layer['out_c'],cur_layer['stride'],'conv_%d'%(i+1)))
            else:
                self.model.add_module('conv_%d'%(i+1),conv_dw_bn_relu_sep_bn_relu(cur_layer['in_c'],cur_layer['out_c'],cur_layer['stride'],'conv_%d'%(i+1)))


        self.source_layer_indexes = [
            8,
            12,
            14,
        ]

        self.ori_feature_channels = [
            self.layers[self.source_layer_indexes[0]-1]['out_c'],
            self.layers[self.source_layer_indexes[1]-1]['out_c'],
            self.layers[self.source_layer_indexes[2]-1]['out_c'],
            256,
            256
        ]


        num_boxes = box_utils.get_num_boxes_of_aspect_ratios(aspect_ratios)

        self.extras = ModuleList([
            Sequential(OrderedDict([
                ('conv15_1x1', conv_sep_bn_relu(self.ori_feature_channels[2], self.ori_feature_channels[3]//2, name='extra0_1x1')),
                ('conv15_3x3', conv_dw_bn_relu_sep_bn_relu(self.ori_feature_channels[3]//2, self.ori_feature_channels[3], stride=2, name='extra0_3x3'))
                ])
            ),
            Sequential(OrderedDict([
                ('conv16_1x1', conv_sep_bn_relu(self.ori_feature_channels[3], self.ori_feature_channels[4]//2, name='extra1_1x1')),
                ('conv16_3x3', conv_dw_bn_relu_sep_bn_relu(self.ori_feature_channels[4]//2, self.ori_feature_channels[4], stride=2, name='extra1_3x3'))
                ])
            )
        ])

        self.regression_headers = ModuleList([
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[0], num_boxes*4, 256, 1, 'loc_0'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[1], num_boxes*4, 256, 1, 'loc_1'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[2], num_boxes*4, 256, 1, 'loc_2'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[3], num_boxes*4, 256, 1, 'loc_3'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[4], num_boxes*4, 256, 1, 'loc_4')
        ])

        self.classification_headers = ModuleList([
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[0], num_boxes*num_classes, 256, 1, 'cls_0'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[1], num_boxes*num_classes, 256, 1, 'cls_1'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[2], num_boxes*num_classes, 256, 1, 'cls_2'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[3], num_boxes*num_classes, 256, 1, 'cls_3'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[4], num_boxes*num_classes, 256, 1, 'cls_4')
        ])

        self.object_headers = ModuleList([
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[0], num_boxes*2, 256, 1, 'obj_0'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[1], num_boxes*2, 256, 1, 'obj_1'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[2], num_boxes*2, 256, 1, 'obj_2'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[3], num_boxes*2, 256, 1, 'obj_3'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[4], num_boxes*2, 256, 1, 'obj_4')
        ])

        self.fpn_convs = ModuleList([
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[0]+self.ori_feature_channels[1], self.ori_feature_channels[0], self.ori_feature_channels[0]//2, 1, 'fpn_0'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[1]+self.ori_feature_channels[2], self.ori_feature_channels[1], self.ori_feature_channels[1]//2, 1, 'fpn_1'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[2]+self.ori_feature_channels[3], self.ori_feature_channels[2], self.ori_feature_channels[2]//2, 1, 'fpn_2'),
            conv_dw_bn_relu_sep_bn_relu_sep(self.ori_feature_channels[3]+self.ori_feature_channels[4], self.ori_feature_channels[3], self.ori_feature_channels[3]//2, 1, 'fpn_3'),
        ])


        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.priors = self.priors.to(self.device)
            
    def forward(self, x: torch.Tensor, print_score_and_box = False) -> Tuple[torch.Tensor, torch.Tensor]:
        ori_features = []
        merge_features = []
        object_confs = []
        confidences = []
        locations = []
        num_features = len(self.ori_feature_channels)
        use_fpn = True
        start_layer_index = 0
        header_index = 0
        with torch.cuda.amp.autocast(self.fp16):
            for end_layer_index in self.source_layer_indexes:
                for layer in self.model[start_layer_index: end_layer_index]:
                    x = layer(x)
                start_layer_index = end_layer_index
                ori_features.append(x)


            for layer in self.model[end_layer_index:]:
                x = layer(x)

            for layer in self.extras:
                x = layer(x)
                ori_features.append(x)

            if use_fpn:
                #upsample
                for i in range(num_features):
                    ori_feature = ori_features[num_features-1-i]
                    if i == 0:
                        merge_features.append(ori_feature)
                    else:
                        upsampled_feature = F.interpolate(merge_features[i-1], scale_factor=2, mode="bilinear")
                        #print(ori_feature.size())
                        #print(upsampled_feature.size())
                        if ori_feature.size()[2] == upsampled_feature.size()[2] and ori_feature.size()[3] == upsampled_feature.size()[3]:
                            merge_feature = torch.cat((ori_feature,upsampled_feature),1)
                            merge_feature = self.fpn_convs[num_features-1-i](merge_feature)
                        else:
                            merge_feature = ori_feature
                        merge_features.append(merge_feature)

                for i in range(num_features):
                    merge_feature = merge_features[num_features-1-i]
                    obj_conf, confidence, location = self.compute_header(i, merge_feature)
                    object_confs.append(obj_conf)
                    confidences.append(confidence)
                    locations.append(location)
            else:
                for i in range(num_features):
                    merge_feature = ori_features[i]
                    obj_conf, confidence, location = self.compute_header(i, merge_feature)
                    object_confs.append(obj_conf)
                    confidences.append(confidence)
                    locations.append(location)

            object_confs = torch.cat(object_confs, 1)
            confidences = torch.cat(confidences, 1)
            locations = torch.cat(locations, 1)

        # fp16 --> float
        if self.fp16:
            object_confs = object_confs.float()
            confidences = confidences.float()
            locations = locations.float()
        
        if self.is_test:
            if print_score_and_box:
                obj_conf = object_confs[0]
                scores = confidences[0]
                cpu_device = torch.device("cpu")
                obj_conf = obj_conf.to(cpu_device)
                print('\n\nobj_conf')
                print(obj_conf.shape)
                obj_h, obj_w = obj_conf.shape
                for i in range(obj_h):
                    line = ''
                    for j in range(obj_w):
                        line = line + '%12.5f '%(obj_conf[i][j])
                    print(line)
                    

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
            object_confs = F.softmax(object_confs, dim=2)
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.center_variance, self.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return object_confs, confidences, boxes
        elif self.with_softmax:
            #print(self.device)
            object_confs = F.softmax(object_confs, dim=2)
            confidences = F.softmax(confidences, dim=2)
            return object_confs, confidences, locations
        else:
            return object_confs, confidences, locations

    def compute_header(self, i, x):
        obj_conf = self.object_headers[i](x)
        obj_conf = obj_conf.permute(0, 2, 3, 1).contiguous()
        obj_conf = obj_conf.view(obj_conf.size(0), -1, 2)

        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return obj_conf, confidence, location

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() \
                                       if not (k.startswith("object_headers") or k.startswith("classification_headers") or k.startswith("regression_headers") or k.startswitch("fpn_convs"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.object_headers.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.fpn_convs.apply(_xavier_init_)
        self.object_headers.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

