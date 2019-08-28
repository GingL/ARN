from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Attribut Reconstruction Loss
class AttributeReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(AttributeReconstructLoss, self).__init__()
        self.att_dropout = nn.Dropout(opt['visual_drop_out'])
        self.att_fc = nn.Linear(opt['fc7_dim']+opt['pool5_dim'], opt['num_atts'])


    def forward(self, attribute_feats, total_ann_score, att_labels, select_ixs, att_weights):
        """attribute_feats.shape = (sent_num, ann_num, 512), total_ann_score.shape = (sent_num, ann_num)"""
        total_ann_score = total_ann_score.unsqueeze(1)
        att_feats_fuse = torch.bmm(total_ann_score, attribute_feats)
        att_feats_fuse = att_feats_fuse.squeeze(1)
        att_feats_fuse = self.att_dropout(att_feats_fuse)
        att_scores = self.att_fc(att_feats_fuse)
        if len(select_ixs) == 0:
            att_loss = 0
        else:
            att_loss = nn.BCEWithLogitsLoss(att_weights.cuda())(att_scores.index_select(0, select_ixs),
                                                     att_labels.index_select(0, select_ixs))
        return att_scores, att_loss

# Language Reconstruction Loss
class LangReconstructionLoss(nn.Module):
    def __init__(self, opt):
        super(LangReconstructionLoss, self).__init__()

        self.variable_lengths = opt['variable_lengths'] > 0
        self.vocab_size = opt['vocab_size']
        self.word_embedding_size = opt['word_embedding_size']
        self.word_vec_size = opt['word_vec_size']
        self.hidden_size = opt['rnn_hidden_size']
        self.bidirectional = opt['decode_bidirectional'] > 0
        self.input_dropout_p = opt['word_drop_out']
        self.dropout_p = opt['rnn_drop_out']
        self.n_layers = opt['rnn_num_layers']
        self.rnn_type = opt['rnn_type']
        self.variable_lengths = opt['variable_lengths'] > 0

        self.embedding = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.input_dropout = nn.Dropout(self.input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(self.word_embedding_size, self.word_vec_size), nn.ReLU())
        self.rnn_type = self.rnn_type
        self.rnn = getattr(nn, self.rnn_type.upper())(self.word_vec_size*2, self.hidden_size, self.n_layers,
                                                      batch_first=True, bidirectional=self.bidirectional,
                                                      dropout=self.dropout_p)
        self.num_dirs = 2 if self.bidirectional else 1

        self.fc = nn.Linear(self.num_dirs * self.hidden_size, self.vocab_size)
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, vis_att_fuse, enc_labels, dec_labels):
        seq_len = enc_labels.size(1)
        sent_num = enc_labels.size(0)
        label_mask = (dec_labels != 0).float()

        if self.variable_lengths:
            input_lengths = (enc_labels != 0).sum(1)
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
            s2r = {s: r for r, s in enumerate(sort_ixs)}
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

            assert max(input_lengths_list) == enc_labels.size(1)

            sort_ixs = enc_labels.data.new(sort_ixs).long()
            recover_ixs = enc_labels.data.new(recover_ixs).long()

            input_labels = enc_labels[sort_ixs]

        vis_att_fuse = vis_att_fuse.unsqueeze(1)
        embedded = self.embedding(input_labels)
        embedded = self.input_dropout(embedded)
        embedded = self.mlp(embedded)

        embedded = torch.cat([embedded, torch.cat([vis_att_fuse, torch.zeros(sent_num, seq_len - 1,
                                                                            self.word_vec_size).cuda()], 1)], 2)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[recover_ixs]

        output = output.view(sent_num * seq_len, -1)
        output = self.fc(output)

        dec_labels = dec_labels.view(-1)
        label_mask = label_mask.view(-1)

        rec_loss = self.cross_entropy(output, dec_labels)
        rec_loss = torch.sum(rec_loss * label_mask) / torch.sum(label_mask)

        return rec_loss

# Language Reconstruction Loss in Adaptive Reconstruction Loss
class AdapLangReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(AdapLangReconstructLoss, self).__init__()

        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']

        self.variable_lengths = opt['variable_lengths'] > 0
        self.vocab_size = opt['vocab_size']
        self.word_embedding_size = opt['word_embedding_size']
        self.word_vec_size = opt['word_vec_size']
        self.hidden_size = opt['rnn_hidden_size']
        self.bidirectional = opt['decode_bidirectional'] > 0
        self.input_dropout_p = opt['word_drop_out']
        self.dropout_p = opt['rnn_drop_out']
        self.n_layers = opt['rnn_num_layers']
        self.rnn_type = opt['rnn_type']
        self.variable_lengths = opt['variable_lengths'] > 0

        self.embedding = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.input_dropout = nn.Dropout(self.input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(self.word_embedding_size, self.word_vec_size), nn.ReLU())
        self.rnn_type = self.rnn_type
        self.rnn = getattr(nn, self.rnn_type.upper())(self.word_vec_size * 2, self.hidden_size, self.n_layers,
                                                      batch_first=True, bidirectional=self.bidirectional,
                                                      dropout=self.dropout_p)
        self.num_dirs = 2 if self.bidirectional else 1

        self.slr_mlp = nn.Sequential(nn.Linear(self.word_vec_size * 3, self.word_vec_size),
                                     nn.ReLU())

        self.fc = nn.Linear(self.num_dirs * self.hidden_size, self.vocab_size)

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, sub_phrase_emb, loc_phrase_emb, rel_phrase_emb, enc_labels, dec_labels):
        """sub_phrase_emb, loc_phrase_emb, rel_phrase_emb.shape = (sent_num, 512), labels.shape = (sent_num, sent_length)"""
        slr_embeded = torch.cat([sub_phrase_emb, loc_phrase_emb, rel_phrase_emb], 1)
        slr_embeded = self.slr_mlp(slr_embeded)

        seq_len = enc_labels.size(1)
        label_mask = (dec_labels != 0).float()
        batchsize = enc_labels.size(0)

        if self.variable_lengths:
            input_lengths = (enc_labels != 0).sum(1)
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
            s2r = {s: r for r, s in enumerate(sort_ixs)}
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

            assert max(input_lengths_list) == enc_labels.size(1)

            sort_ixs = enc_labels.data.new(sort_ixs).long()
            recover_ixs = enc_labels.data.new(recover_ixs).long()

            input_labels = enc_labels[sort_ixs]

        slr_embeded = slr_embeded.view(batchsize, 1, -1)

        embedded = self.embedding(input_labels)
        embedded = self.input_dropout(embedded)
        embedded = self.mlp(embedded)

        slr_embedded = torch.cat([embedded, torch.cat([slr_embeded, torch.zeros(batchsize, seq_len - 1,
                                                                            self.word_embedding_size).cuda()], 1)], 2)

        if self.variable_lengths:
            slr_embedded = nn.utils.rnn.pack_padded_sequence(slr_embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(slr_embedded)

        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[recover_ixs]

        output = output.view(batchsize * seq_len, -1)
        output = self.fc(output)

        dec_labels = dec_labels.view(-1)
        label_mask = label_mask.view(-1)

        lang_rec_loss = self.cross_entropy(output, dec_labels)
        lang_rec_loss = torch.sum(lang_rec_loss * label_mask) / torch.sum(label_mask)

        return lang_rec_loss

# Visual Reconstruction Loss in Adaptive Reconstruction Loss
class AdapVisualReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(AdapVisualReconstructLoss, self).__init__()

    def forward(self, sub_phrase_emb, sub_phrase_recons, loc_phrase_emb, loc_phrase_recons, rel_phrase_emb,
                rel_phrase_recons, weights):
        """
        (sub_phrase_emb, sub_phrase_recons, loc_phrase_emb, loc_phrase_recons, rel_phrase_emb, rel_phrase_recons).shape=(sent_num, 512)
        weights.shape = (sent_num, 3)
        """
        sub_loss = self.mse_loss(sub_phrase_recons, sub_phrase_emb).sum(1).unsqueeze(1)
        loc_loss = self.mse_loss(loc_phrase_recons, loc_phrase_emb).sum(1).unsqueeze(1)
        rel_loss = self.mse_loss(rel_phrase_recons, rel_phrase_emb).sum(1).unsqueeze(1)

        total_loss = (weights * torch.cat([sub_loss, loc_loss, rel_loss], 1)).sum(1).mean(0)

        return total_loss

    def mse_loss(self, recons, emb):
        return (recons-emb)**2
