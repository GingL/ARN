from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LocationDecoder(nn.Module):
    def __init__(self, opt):
        super(LocationDecoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5 + 25, opt['jemb_dim']))

    def forward(self, loc_feats, total_ann_score):
        total_ann_score = total_ann_score.unsqueeze(1)
        loc_feats_fuse = torch.bmm(total_ann_score, loc_feats)
        loc_feats_fuse = loc_feats_fuse.squeeze(1)
        loc_feats_fuse = self.mlp(loc_feats_fuse)
        return loc_feats_fuse


class SubjectDecoder(nn.Module):
    def __init__(self, opt):
        super(SubjectDecoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(opt['pool5_dim'] + opt['fc7_dim'], opt['jemb_dim']))

    def forward(self, sub_feats, total_ann_score):
        total_ann_score = total_ann_score.unsqueeze(1)
        sub_feats_fuse = torch.bmm(total_ann_score, sub_feats)
        sub_feats_fuse = sub_feats_fuse.squeeze(1)
        sub_feats_fuse = self.mlp(sub_feats_fuse)
        return sub_feats_fuse


class RelationDecoder(nn.Module):
    def __init__(self, opt):
        super(RelationDecoder, self).__init__()
        self.jemb_dim = opt['jemb_dim']
        self.word_vec_size = opt['word_vec_size']
        self.fc7_dim = opt['fc7_dim']
        self.mlp = nn.Sequential(nn.Linear(self.fc7_dim + 5, self.jemb_dim))

    def forward(self, rel_feats, total_ann_score, ixs):
        sent_num, ann_num = ixs.size(0), ixs.size(1)
        total_ann_score = total_ann_score.unsqueeze(1)
        ixs = ixs.view(sent_num, ann_num, 1).unsqueeze(3).expand(sent_num, ann_num, 1,
                                                                 self.fc7_dim + 5)
        rel_feats_max = torch.gather(rel_feats, 2, ixs)
        rel_feats_max = rel_feats_max.squeeze(2)
        rel_feats_fuse = torch.bmm(total_ann_score, rel_feats_max)
        rel_feats_fuse = rel_feats_fuse.squeeze(1)
        rel_feats_fuse = self.mlp(rel_feats_fuse)
        return rel_feats_fuse
