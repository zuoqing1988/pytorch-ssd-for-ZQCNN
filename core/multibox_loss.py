import torch.nn as nn
import torch.nn.functional as F
import torch


from .utils import box_utils_zq as box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos


class MultiboxLoss2(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss2, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_low, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining2(loss, labels, labels_low, self.neg_pos_ratio)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos


class MultiboxLoss3(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss3, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_mid, labels_low, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining3(loss, labels, labels_mid, labels_low, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos

class MultiboxLoss4(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss4, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_mid, labels_low, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining4(loss, labels, labels_mid, labels_low, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos

class MultiboxLoss5(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss5, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_mid, labels_low, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            conf = -F.log_softmax(confidence, dim=2)
            loss = conf[:, :, 0]
            #print(conf)
            #print('=======')
            #print(loss)
            mask = box_utils.hard_negative_mining5(loss, labels, labels_mid, labels_low, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos


class MultiboxLoss6(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss6, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_mid, labels_low, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            pred_label = torch.argmax(confidence, dim=2)
            conf = -F.log_softmax(confidence, dim=2)
            loss = conf[:, :, 0]
            #print(confidence)
            #print('=======')
            #print(loss)
            #print(pred_label)
            mask = box_utils.hard_negative_mining6(loss, pred_label, labels, labels_mid, labels_low, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos

class MultiboxLoss7(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss7, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_mid, labels_low, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            pred_label = torch.argmax(confidence, dim=2)
            conf = -F.log_softmax(confidence, dim=2)
            loss = conf[:, :, 0]
            #print(confidence)
            #print('=======')
            #print(loss)
            #print(pred_label)
            mask = box_utils.hard_negative_mining7(loss, pred_label, labels, labels_mid, labels_low, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            return smooth_l1_loss/num_pos, classification_loss/num_pos

class MultiboxLoss9(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss9, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, labels_mid, labels_low, is_peer, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            pred_label = torch.argmax(confidence, dim=2)
            conf = -F.log_softmax(confidence, dim=2)
            loss = conf[:, :, 0]
            #print(confidence)
            #print('=======')
            #print(loss)
            #print(pred_label)
            mask = box_utils.hard_negative_mining9(loss, pred_label, labels, labels_mid, labels_low, is_peer, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(mask)
        tmp_confidence = confidence[mask, :]
        if tmp_confidence.shape[0] == 0:
            #print(confidence.shape)
            #print(labels.shape)
            classification_loss = F.cross_entropy(confidence[0].reshape(-1, num_classes), labels[0], size_average=False)
            pos_mask = labels >= 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0
        else:  
            confidence = tmp_confidence
            #print(confidence.shape)
            #print(labels[mask].shape)
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            cls_pos_mask = mask & pos_mask
            num_cls_pos = cls_pos_mask.long().sum(dim=1, keepdim=True).sum()
            return smooth_l1_loss/num_pos, classification_loss/num_cls_pos


class MultiboxLossX(nn.Module):
    def __init__(self, priors, neg_pos_ratio_mid, neg_pos_ratio_low,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLossX, self).__init__()
        self.neg_pos_ratio_mid = neg_pos_ratio_mid
        self.neg_pos_ratio_low = neg_pos_ratio_low
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, object_conf, confidence, locations, objects, objects_mid, objects_low, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            obj_conf = -F.log_softmax(object_conf, dim=2)
            obj_loss = obj_conf[:, :, 0]
            obj_mask = box_utils.hard_negative_mining_x(obj_loss, objects, objects_mid, objects_low, self.neg_pos_ratio_mid, self.neg_pos_ratio_low)
        #print(obj_mask)
        tmp_conf = object_conf[obj_mask, :]
        if tmp_conf.shape[0] == 0:
            #print(object_conf.shape)
            #print(objects.shape)
            obj_loss = F.cross_entropy(object_conf[0].reshape(-1, 2), objects[0], size_average=False)
            pos_mask = objects >= 0
            classification_loss = F.cross_entropy(confidence[pos_mask,:].reshape(-1, num_classes), labels[pos_mask], size_average=False)
            predicted_locations = locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0)+1e-6
            return smooth_l1_loss/num_pos*0, classification_loss/num_pos*0, obj_loss/num_pos*0
        else:  
            obj_conf = tmp_conf
            #print(obj_conf.shape)
            #print(objects[mask].shape)
            obj_loss = F.cross_entropy(obj_conf.reshape(-1, 2), objects[obj_mask], size_average=False)
            pos_mask = objects > 0
            pred_confidence = confidence[pos_mask,:].reshape(-1, num_classes)
            gt_labels = labels[pos_mask] 
            #print(gt_labels)
            #print('=========')
            #print(labels)
            #print('************')
            classification_loss = F.cross_entropy(pred_confidence, gt_labels, size_average=False)
            predicted_locations = locations[pos_mask, :].reshape(-1, 4)
            gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
            num_pos = gt_locations.size(0) + 1e-6
            #print(num_pos)
            return smooth_l1_loss/num_pos, classification_loss/num_pos, obj_loss/num_pos
