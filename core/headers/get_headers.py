# -*- coding: utf-8 -*-

from . import headers1
from . import headers2
from . import headers3
from . import headers4
from . import headers10
from . import headers11
from . import headers12
from . import headers13
from . import headers14
from . import headers15
from . import headers16
from . import headers17
from . import headers18
from . import headers20
from . import headers21
from . import headers22
from . import headers23
from . import headers24
from . import headers25
from . import headers26
from . import headers27
from . import headers28


def get_headers(type, backbone, num_classes, aspect_ratios):
    if type == "headers1":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers1.headers1(backbone, num_classes, aspect_ratios)
    elif type == "headers2":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers2.headers2(backbone, num_classes, aspect_ratios)
    elif type == "headers3":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers3.headers3(backbone, num_classes, aspect_ratios)
    elif type == "headers4":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers4.headers4(backbone, num_classes, aspect_ratios)
    elif type == "headers10":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers10.headers10(backbone, num_classes, aspect_ratios)
    elif type == "headers11":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers11.headers11(backbone, num_classes, aspect_ratios)
    elif type == "headers12":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers12.headers12(backbone, num_classes, aspect_ratios)
    elif type == "headers13":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers13.headers13(backbone, num_classes, aspect_ratios)
    elif type == "headers14":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers14.headers14(backbone, num_classes, aspect_ratios)
    elif type == "headers15":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers15.headers15(backbone, num_classes, aspect_ratios)
    elif type == "headers16":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers16.headers16(backbone, num_classes, aspect_ratios)
    elif type == "headers17":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers17.headers17(backbone, num_classes, aspect_ratios)
    elif type == "headers18":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers18.headers18(backbone, num_classes, aspect_ratios)
    elif type == "headers20":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers20.headers20(backbone, num_classes, aspect_ratios)
    elif type == "headers21":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers21.headers21(backbone, num_classes, aspect_ratios)
    elif type == "headers22":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers22.headers22(backbone, num_classes, aspect_ratios)
    elif type == "headers23":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers23.headers23(backbone, num_classes, aspect_ratios)
    elif type == "headers24":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers24.headers24(backbone, num_classes, aspect_ratios)
    elif type == "headers25":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers25.headers25(backbone, num_classes, aspect_ratios)
    elif type == "headers26":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers26.headers26(backbone, num_classes, aspect_ratios)
    elif type == "headers27":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers27.headers27(backbone, num_classes, aspect_ratios)
    elif type == "headers28":
        source_layer_indexes, extras, classification_headers, regreession_headers = headers28.headers28(backbone, num_classes, aspect_ratios)
   
   
    return source_layer_indexes, extras, classification_headers, regreession_headers

