from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        assert isinstance(bottom, Variable), 'bottom must be variable'

        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled

class LocationEncoder(nn.Module):
    def __init__(self, opt):
        super(LocationEncoder, self).__init__()
        init_norm = opt.get('visual_init_norm', 20)
        self.lfeats_normalizer = Normalize_Scale(5, init_norm)
        self.dif_lfeat_normalizer = Normalize_Scale(25, init_norm)

    def forward(self, lfeats, dif_lfeats):
        sent_num, ann_num = lfeats.size(0), lfeats.size(1)
        output = torch.cat([self.lfeats_normalizer(lfeats.contiguous().view(-1, 5)),
                            self.dif_lfeat_normalizer(dif_lfeats.contiguous().view(-1, 25))], 1)
        output = output.view(sent_num, ann_num, 5+25)

        return output
    
class SubjectEncoder(nn.Module):
    def __init__(self, opt):
        super(SubjectEncoder, self).__init__()
        self.word_vec_size = opt['word_vec_size']
        self.jemb_dim = opt['jemb_dim']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']

        self.pool5_normalizer = Normalize_Scale(opt['pool5_dim'], opt['visual_init_norm'])
        self.fc7_normalizer   = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])

    def forward(self, pool5, fc7, phrase_emb):
        sent_num, ann_num, grids = pool5.size(0), pool5.size(1), pool5.size(3)*pool5.size(4)
        batch = sent_num * ann_num

        pool5 = pool5.contiguous().view(batch, self.pool5_dim, -1)
        pool5 = pool5.transpose(1,2).contiguous().view(-1, self.pool5_dim)
        pool5 = self.pool5_normalizer(pool5)
        pool5 = pool5.view(sent_num, ann_num, 49, -1).transpose(2, 3).contiguous().mean(3)

        fc7   = fc7.contiguous().view(batch, self.fc7_dim, -1)
        fc7   = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)
        fc7   = self.fc7_normalizer(fc7)
        fc7 = fc7.view(sent_num, ann_num, 49, -1).transpose(2, 3).contiguous().mean(3)

        avg_att_feats = torch.cat([pool5, fc7], 2)

        return avg_att_feats

class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()
        self.vis_feat_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.lfeat_normalizer    = Normalize_Scale(5, opt['visual_init_norm'])

    def forward(self, cxt_feats, cxt_lfeats):
        masks = (cxt_lfeats.sum(3) != 0).float()

        sent_num, ann_num = cxt_feats.size(0), cxt_feats.size(1)
        batch, num_cxt = sent_num*ann_num, cxt_feats.size(2)
        cxt_feats = self.vis_feat_normalizer(cxt_feats.contiguous().view(batch * num_cxt, -1))
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.contiguous().view(batch * num_cxt, -1))

        rel_feats = torch.cat([cxt_feats, cxt_lfeats], 1)

        rel_feats = rel_feats.view(sent_num, ann_num, num_cxt, -1)
        return rel_feats, masks