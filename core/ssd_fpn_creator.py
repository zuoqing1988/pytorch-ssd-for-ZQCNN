from .ssd_fpn.ssd_fpn_zq1 import SSD_FPN_ZQ1
from .ssd_fpn.ssd_fpn_zq2 import SSD_FPN_ZQ2
from .ssd_fpn.ssd_fpn_zq3 import SSD_FPN_ZQ3
from .ssd_fpn.ssd_fpn_zq4 import SSD_FPN_ZQ4
from .ssd_fpn.ssd_fpn_zq5 import SSD_FPN_ZQ5

from .predictor import Predictor


def create_ssd(net_type, use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test=False, with_softmax=False, device=None):
    if net_type == 'ssd_fpn_zq1':
        return SSD_FPN_ZQ1(use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test, with_softmax, device=device)
    elif net_type == 'ssd_fpn_zq2':
        return SSD_FPN_ZQ2(use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test, with_softmax, device=device)
    elif net_type == 'ssd_fpn_zq3':
        return SSD_FPN_ZQ3(use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test, with_softmax, device=device)
    elif net_type == 'ssd_fpn_zq4':
        return SSD_FPN_ZQ4(use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test, with_softmax, device=device)
    elif net_type == 'ssd_fpn_zq5':
        return SSD_FPN_ZQ5(use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test, with_softmax, device=device)
    

def create_ssd_predictor(net, image_size_x, image_size_y, image_mean, image_std, iou_thresh, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, image_size_x, image_size_y, image_mean,
                          image_std,
                          nms_method=nms_method,
                          iou_threshold=iou_thresh,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
