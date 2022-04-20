import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from collections import OrderedDict
from ..utils import box_utils_zq as box_utils

def SeperableConv2d(in_channels, out_channels, kernel_size, stride, padding, name):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(OrderedDict([
        (name+'/dw', Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding)),
        (name+'/relu', ReLU()),
        (name+'/sep', Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        ])
    )

def Conv_2d(in_channels, out_channels, kernel_size, stride, padding, name):
    return Sequential(OrderedDict([
        (name+'/conv', Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        ])
    )


def headers3(backbone, num_classes, aspect_ratios):
    
    source_layer_indexes = [
        8,
        12,
        14,
    ]
    
    num_boxes = box_utils.get_num_boxes_of_aspect_ratios(aspect_ratios)

    extras = ModuleList([
        Sequential(OrderedDict([
            ('conv15_1x1', Conv2d(in_channels=backbone.layers[-1]['out_c'], out_channels=64, kernel_size=1)),
            ('conv15_1x1/relu', ReLU()),
            ('conv15_3x3', SeperableConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, name='conv15_3x3'))
            ])
        ),
        Sequential(OrderedDict([
            ('conv16_1x1', Conv2d(in_channels=128, out_channels=32, kernel_size=1)),
            ('conv16_1x1/relu', ReLU()),
            ('conv16_3x3', SeperableConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, name='conv16_3x3'))
            ])
        ),
        Sequential(OrderedDict([
            ('conv17_1x1', Conv2d(in_channels=64, out_channels=32, kernel_size=1)),
            ('conv17_1x1/relu', ReLU()),
            ('conv17_3x3', SeperableConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, name='conv17_3x3'))
            ])
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=backbone.layers[source_layer_indexes[0]-1]['out_c'], out_channels=num_boxes * 4, kernel_size=3, stride=1, padding=1, name='loc_1'),
        SeperableConv2d(in_channels=backbone.layers[source_layer_indexes[1]-1]['out_c'], out_channels=num_boxes * 4, kernel_size=3, stride=1, padding=1, name='loc_2'),
        SeperableConv2d(in_channels=backbone.layers[source_layer_indexes[2]-1]['out_c'], out_channels=num_boxes * 4, kernel_size=3, stride=1, padding=1, name='loc_3'),
        SeperableConv2d(in_channels=128, out_channels=num_boxes * 4, kernel_size=3, stride=1, padding=1, name='loc_4'),
        SeperableConv2d(in_channels=64, out_channels=num_boxes * 4, kernel_size=3, stride=1, padding=1, name='loc_5'),
        Conv_2d(in_channels=64, out_channels=num_boxes * 4, kernel_size=1, stride=1, padding=0, name='loc_6'),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=backbone.layers[source_layer_indexes[0]-1]['out_c'], out_channels=num_boxes * num_classes, kernel_size=3, stride=1, padding=1, name='cls_1'),
        SeperableConv2d(in_channels=backbone.layers[source_layer_indexes[1]-1]['out_c'], out_channels=num_boxes * num_classes, kernel_size=3, stride=1, padding=1, name='cls_2'),
        SeperableConv2d(in_channels=backbone.layers[source_layer_indexes[2]-1]['out_c'], out_channels=num_boxes * num_classes, kernel_size=3, stride=1, padding=1, name='cls_3'),
        SeperableConv2d(in_channels=128, out_channels=num_boxes * num_classes, kernel_size=3, stride=1, padding=1, name='cls_4'),
        SeperableConv2d(in_channels=64, out_channels=num_boxes * num_classes, kernel_size=3, stride=1, padding=1, name='cls_5'),
        Conv_2d(in_channels=64, out_channels=num_boxes * num_classes, kernel_size=1, stride=1, padding=0, name='cls_6'),
    ])

    return source_layer_indexes, extras, classification_headers, regression_headers

