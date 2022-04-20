import torch
from .utils import box_utils_zq as box_utils
from .data_preprocessing import PredictionTransform
from .utils.misc import Timer
import numpy as np


class Predictor:
    def __init__(self, net, size_x, size_y, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size_x, size_y, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print('init predictor')
        #print(self.device)

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None, print_score_and_box=False):
        #print('run predictor')
        #print(self.device)
        cpu_device = torch.device("cpu")
        height = image.shape[0]
        width = image.shape[1]
        if image.ndim == 2:
            image = image[:,:,np.newaxis]
        image = self.transform(image)
        images = image.unsqueeze(0)
        #print(images)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            objs, scores, boxes = self.net.forward(images, print_score_and_box)
            #print("Inference time: ", self.timer.end())
        objs = objs[0]
        scores = scores[0]
        boxes = boxes[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        #print(prob_threshold)
        # this version of nms is slower on GPU, so we move data to CPU.
        objs = objs.to(cpu_device)
        scores = scores.to(cpu_device)
        boxes = boxes.to(cpu_device)
        if print_score_and_box:
            print(objs.shape)
            obj_h, obj_w = objs.shape
            for i in range(obj_h):
                line = ''
                for j in range(obj_w):
                    line = line + '%12.5f '%(objs[i][j])
                print(line)

            print(scores.shape)
            score_h, score_w = scores.shape
            for i in range(score_h):
                line = ''
                for j in range(score_w):
                    line = line + '%12.5f '%(scores[i][j])
                print(line)

            print(boxes.shape)
            box_h, box_w = boxes.shape
            for i in range(box_h):
                print('%12.5f %12.5f %12.5f %12.5f'%(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]))
        

        probs = objs[:, 1]
        mask = probs > 0.5
        probs = probs[mask]
        if probs.size(0) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])


        indices = np.arange(0,objs.size(0))
        indices = indices[mask]
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs, indices = box_utils.nms_with_id(box_probs, indices, self.nms_method,
                                  score_threshold=prob_threshold,
                                  iou_threshold=self.iou_threshold,
                                  sigma=self.sigma,
                                  top_k=top_k,
                                  candidate_size=self.candidate_size)
        if indices.shape[0] == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        cls_scores = scores[indices,:]
        #labels = torch.argmax(cls_scores,dim=1) + 1
        max_cls_probs,labels = torch.max(cls_scores, dim=1)
        labels += 1
        #print(max_cls_probs)
        picked_mask = max_cls_probs > prob_threshold
        picked_boxes = box_probs[picked_mask]
        picked_cls_probs = max_cls_probs[picked_mask]
        picked_labels = labels[picked_mask]

        picked_boxes[:, 0] *= width
        picked_boxes[:, 1] *= height
        picked_boxes[:, 2] *= width
        picked_boxes[:, 3] *= height
        return picked_boxes[:,0:4], picked_labels, picked_cls_probs
