from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb


# grad_clip=0.1
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            # pdb.set_trace()
            # 裁切一下
            if hasattr(param.grad, 'data'):
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
