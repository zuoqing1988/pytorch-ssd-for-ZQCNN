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
from core.ssd_fpn_x.ssd_fpn_x_base import MatchPrior
from core.ssd_fpn_x_creator import create_ssd, create_ssd_predictor
from core.datasets.voc_dataset_x import VOCDataset
from core.multibox_loss import MultiboxLossX
from core.data_preprocessing import TrainAugmentation, TestTransform


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--config_file',type=str, help='Specify config file')

parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument('--gpus_id', default='0', type=str, help='GPUS ID')
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
args = parser.parse_args()



def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive + 1e-9)
    if num_true_cases == 0:
        recall = 0.0
    else:
        recall = true_positive / num_true_cases
    if use_2007_metric:
        return recall[-1], precision[-1], measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return recall[-1], precision[-1], measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    logging.info(args)

    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    num_classes = len(class_names)

    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config.read(args.config_file)

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
    if torch.cuda.is_available() and args.use_cuda:
        DEVICE = torch.device('cuda:'+args.gpus_id)
        print('use cuda')
    else:
        DEVICE = torch.device("cpu")
        print('use cpu')

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")


    # create ssd net
    net = create_ssd(net_type, use_gray, num_classes-1, aspect_ratios, priors, center_variance, size_variance, is_test=True, device=DEVICE)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
   

    predictor = create_ssd_predictor(net, image_size_x, image_size_y, image_mean, image_std, iou_thresh, 
                                       candidate_size=200, nms_method=args.nms_method, sigma=0.5, device=DEVICE)

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, use_gray, is_test=True)
    
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    print(true_case_stat)
    
    results = []
    for i in range(len(dataset)):
        #print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        load_time = timer.end("Load Image")
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image, -1, 0.5)
        predict_time = timer.end("Predict")
        print("image [%d], load: %.4f s, predict: %.4f s, num_boxes = %d."%(i, load_time, predict_time, len(boxes)))
        #print(boxes, labels, probs)
        if len(boxes) == 0:
            boxes = torch.tensor([[0.0,0.0,0.0,0.0]])
            labels = torch.tensor([0.0])
            probs = torch.tensor([0.0])
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    recalls = []
    precisions = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        if class_index in true_case_stat:
            recall, precision, ap = compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                args.iou_threshold,
                args.use_2007_metric
            )
        else:
            ap = 0
            recall = 0
            precision = 0

        aps.append(ap)
        recalls.append(recall)
        precisions.append(precision)
        print(f"{class_name}: reall={recall}, precision={precision}, ap={ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
