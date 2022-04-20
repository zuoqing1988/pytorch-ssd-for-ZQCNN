
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class backbone10(nn.Module):
    def __init__(self, use_gray, num_classes=1024):
        super(backbone10, self).__init__()
  
        def conv_bn(inp, oup, stride, name):
            return nn.Sequential(OrderedDict([
                (name+'/conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
                (name+'/bn', nn.BatchNorm2d(oup)),
                (name+'/relu', nn.ReLU(inplace=True))
                ])
            )

        def conv_dw(inp, oup, stride, name):
            return nn.Sequential(OrderedDict([
                (name+'/dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
                (name+'/dw/bn', nn.BatchNorm2d(inp)),
                (name+'/dw/relu', nn.ReLU(inplace=True)),
                (name+'/sep', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                (name+'/sep/bn', nn.BatchNorm2d(oup)),
                (name+'/sep/relu', nn.ReLU(inplace=True))
                ])
            )

        self.use_gray = use_gray
        in_c = 1 if use_gray else 3
        self.layers = [{'in_c':in_c, 'out_c':128, 'stride':2},
                       {'in_c':128, 'out_c':128, 'stride':1},
                       {'in_c':128, 'out_c':256, 'stride':2},
                       {'in_c':256, 'out_c':256, 'stride':1},
                       {'in_c':256, 'out_c':512, 'stride':2},
                       {'in_c':512, 'out_c':512, 'stride':1},
                       {'in_c':512, 'out_c':512, 'stride':1},
                       {'in_c':512, 'out_c':512, 'stride':1},
                       {'in_c':512, 'out_c':1024, 'stride':2},
                       {'in_c':1024, 'out_c':1024, 'stride':1},
                       {'in_c':1024, 'out_c':1024, 'stride':1},
                       {'in_c':1024, 'out_c':1024, 'stride':1},
                       {'in_c':1024, 'out_c':2048, 'stride':2},
                       {'in_c':2048, 'out_c':2048, 'stride':1}]

        self.num_layers = len(self.layers)
        self.model = nn.Sequential()
        for i in range(self.num_layers):
            cur_layer = self.layers[i]
            if i == 0:    
                self.model.add_module('conv_%d'%(i+1),conv_bn(cur_layer['in_c'],cur_layer['out_c'],cur_layer['stride'],'conv_%d'%(i+1)))
            else:
                self.model.add_module('conv_%d'%(i+1),conv_dw(cur_layer['in_c'],cur_layer['out_c'],cur_layer['stride'],'conv_%d'%(i+1)))


        self.fc = nn.Linear(self.layers[-1]['out_c'], num_classes)

    def forward(self, x):
        x = self.model(x)
        #x = F.avg_pool2d(x, 7)
        x = x.view(-1, self.layers[-1]['out_c'])
        x = self.fc(x)
        return x


def get_net(use_gray, num_classes):
    return backbone10(use_gray, num_classes)
