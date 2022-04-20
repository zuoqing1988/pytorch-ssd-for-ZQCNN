from .ssd import SSD
from .predictor import Predictor
from .backbone import get_backbone as get_backbone
from .headers import get_headers as get_headers


def create_ssd(backbone_type, header_type, use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test=False, with_softmax=False, device=None, fp16=False):
    #print(backbone_type)
    base_net = get_backbone.get_backbone(backbone_type, use_gray, num_classes)
    #print(header_type)
    source_layer_indexes, extras, classification_headers, regression_headers = get_headers.get_headers(header_type, base_net, num_classes, aspect_ratios)

    return SSD(num_classes, base_net.model, source_layer_indexes,
               extras, classification_headers, regression_headers, priors, center_variance, size_variance, is_test=is_test, with_softmax=with_softmax, device=device, fp16=fp16)


def create_ssd_predictor(net, image_size_x, image_size_y, image_mean, image_std, iou_thresh, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, image_size_x, image_size_y, image_mean,
                          image_std,
                          nms_method=nms_method,
                          iou_threshold=iou_thresh,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
