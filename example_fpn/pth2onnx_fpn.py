import configparser
import os,sys
sys.path.append(os.getcwd())
from core.ssd_fpn_creator import create_ssd,create_ssd_predictor
from core.utils.box_utils_zq import SSDSpec, SSDBoxSizes, generate_ssd_priors
from core.utils.misc import Timer
import cv2
import sys
import torch
import numpy as np
import io
import torch.onnx


def pth2onnx(config_file, in_file, out, num_valid_classes, with_softmax):
    #class_names = ['bottle','phone','cigar']

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
    
    # net type
    net_type = params['net_type']
    
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

    # create ssd net
    model = create_ssd(net_type, use_gray, num_valid_classes+1, aspect_ratios, priors, center_variance, size_variance, is_test=False, with_softmax=with_softmax, device=torch.device("cpu"))


    loaded_model = torch.load(in_file,map_location='cpu')
    #print(loaded_model)
    model.load_state_dict(loaded_model)
    in_c = 1 if use_gray else 3
    dummy_input = torch.randn(1,in_c,image_size_y,image_size_x)
    input_name=['data']
    output_name=['cls','loc']
    torch.onnx.export(model,dummy_input,out_file,verbose=True,input_names=input_name,output_names=output_name)

if __name__=='__main__':

    
    config_file = 'configs/zq1.cfg'
    in_file = 'models/backbone1-headers1-256x192/Epoch-5-Loss-5.812523530079768.pth'
    out_file = 'Epoch-5-Loss-5.812523530079768.onnx'
    with_softmax = False
    if len(sys.argv) < 5:
        print('pth2onnx.py config_file in_file out_file num_valid_classes [with_softmax]')
    else:
        config_file = sys.argv[1]
        in_file = sys.argv[2]
        out_file = sys.argv[3]
        num_valid_classes = int(sys.argv[4])
        if len(sys.argv) >=6:
            with_softmax = (int(sys.argv[5]) != 0)
        pth2onnx(config_file, in_file, out_file, num_valid_classes, with_softmax)
