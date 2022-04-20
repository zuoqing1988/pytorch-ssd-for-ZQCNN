import argparse
import os,sys
import pathlib
import logging
import itertools
import configparser
import torch
import numpy as np
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from core.utils.box_utils_zq import SSDSpec, SSDBoxSizes, generate_ssd_priors
import core.utils.box_utils_zq as box_utils
from core.utils import measurements
from core.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from core.ssd import MatchPrior
from core.ssd_creator import create_ssd, create_ssd_predictor
from core.datasets.voc_dataset import VOCDataset
from core.datasets.open_images import OpenImagesDataset
from core.multibox_loss import MultiboxLoss
from core.data_preprocessing import TrainAugmentation, TestTransform
from core.utils.misc import Timer
import cv2
import sys


if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <config file>  <model path> <label path> <image path>')
    sys.exit(0)
else:
    config_file = sys.argv[1]
    model_path = sys.argv[2]
    label_path = sys.argv[3]
    image_path = sys.argv[4]

    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)

    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file)

    #print(config)

    for _ in config.options("Train"):
        params[_] = eval(config.get("Train",_))



    # image_size_x, image_size_y, image_mean, image_std, use_gray
    image_size_x = int(params['image_size_x'])
    image_size_y = int(params['image_size_y'])
    image_std = float(params['image_std'])
    image_mean = params['image_mean']
    use_gray = bool(params['use_gray'])
    
    mean_values = list()
    for i in range(len(image_mean)):
        mean_values.append(float(image_mean[i]))
    image_mean = np.array(mean_values,dtype=np.float)

    # iou_thresh, center_varaiance, size_variance
    iou_thresh = float(params['iou_thresh'])
    center_variance = float(params['center_variance'])
    size_variance = float(params['size_variance'])
    
    # backbone type, header type
    backbone_type = params['backbone_type']
    header_type = params['header_type']
    
    # aspect_ratios
    aspect_ratios = params['aspect_ratios']
    if aspect_ratios is None:
        aspect_ratios = list()
    else:
        ratios = list()
        if type(aspect_ratios) == tuple:
            for j in range(len(aspect_ratios)):
                ratios.append(float(aspect_ratios[j]))
        else:
            ratios.append(float(aspect_ratios))
        aspect_ratios = ratios

    #print(aspect_ratios)
    # specs
    specs = list()
    for i in range(1,100):
        name = 'spec%d'%i
        if not name in params:
            break
        line = params[name]
        feat_map_x = int(line[0])
        feat_map_y = int(line[1])
        shrinkage_x = int(line[2])
        shrinkage_y = int(line[3])
        bbox_min = float(line[4])
        bbox_max = float(line[5])
        specs.append(SSDSpec(feat_map_x, feat_map_y, shrinkage_x, shrinkage_y, SSDBoxSizes(bbox_min, bbox_max), aspect_ratios))
        
    # priors
    priors = generate_ssd_priors(specs, image_size_x, image_size_y)

    # device
    DEVICE = torch.device("cpu")
    

    # create ssd net
    net = create_ssd(backbone_type, header_type, use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test=True, device=DEVICE)

    net.load(model_path)
    predictor = create_ssd_predictor(net, image_size_x, image_size_y, image_mean, image_std, iou_thresh, 
                                       candidate_size=200, nms_method=None, sigma=0.5, device=DEVICE)

    if use_gray:
        ori_image = cv2.imread(image_path,0)
        image = ori_image.copy()
    else:
        ori_image = cv2.imread(image_path)
        if ori_image.ndim == 2:
            image = cv2.cvtColor(ori_image,cv2.COLOR_GRAY2BGR)
        else:
            image = ori_image.copy()
    boxes, labels, probs = predictor.predict(image, 10, 0.2, True)

    show_image = ori_image
    if show_image.ndim == 2:
        show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(show_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(show_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    path = "run_ssd_example_output.jpg"
    cv2.imwrite(path, show_image)
    print(f"Found {len(probs)} objects. The output image is {path}")
