from .ssd_fpn_x.ssd_fpn_x_zq16 import SSD_FPN_X_ZQ16

from .predictor_x import Predictor


def create_ssd(net_type, use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test=False, with_softmax=False, device=None):
    if net_type == 'ssd_fpn_zq16':
        return SSD_FPN_X_ZQ16(use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test, with_softmax, device=device)


def create_ssd_predictor(net, image_size_x, image_size_y, image_mean, image_std, iou_thresh, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, image_size_x, image_size_y, image_mean,
                          image_std,
                          nms_method=nms_method,
                          iou_threshold=iou_thresh,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
