# -*- coding: utf-8 -*-

from .backbone1 import get_net as get_net1
from .backbone2 import get_net as get_net2
from .backbone3 import get_net as get_net3
from .backbone4 import get_net as get_net4
from .backbone5 import get_net as get_net5
from .backbone10 import get_net as get_net10
from .backbone11 import get_net as get_net11
from .backbone12 import get_net as get_net12
from .backbone13 import get_net as get_net13


def get_backbone(type, use_gray, num_classes):
    if type == "backbone1":
        net = get_net1(use_gray, num_classes)
    elif type == "backbone2":
        net = get_net2(use_gray, num_classes)
    elif type == "backbone3":
        net = get_net3(use_gray, num_classes)
    elif type == "backbone4":
        net = get_net4(use_gray, num_classes)
    elif type == "backbone5":
        net = get_net5(use_gray, num_classes)
    elif type == "backbone10":
        net = get_net10(use_gray, num_classes)
    elif type == "backbone11":
        net = get_net11(use_gray, num_classes)
    elif type == "backbone12":
        net = get_net12(use_gray, num_classes)
    elif type == "backbone13":
        net = get_net13(use_gray, num_classes)
   
    return net

