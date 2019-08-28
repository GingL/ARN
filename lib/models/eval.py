from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


# IoU function
def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union


def eval_split(loader, model, split, opt):

    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0
    vis_res_loss_sum = 0
    lang_res_loss_sum = 0


    while True:
        with torch.no_grad():
            data = loader.getTestBatch(split, opt)
            att_weights = loader.get_attribute_weights()
            ann_ids = data['ann_ids']
            sent_ids = data['sent_ids']
            Feats = data['Feats']
            labels = data['labels']
            enc_labels = data['enc_labels']
            dec_labels = data['dec_labels']
            att_labels, select_ixs = data['att_labels'], data['select_ixs']

            for i, sent_id in enumerate(sent_ids):
                enc_label = enc_labels[i:i + 1]
                max_len = (enc_label != 0).sum().data[0]
                enc_label = enc_label[:, :max_len]
                dec_label = dec_labels[i:i + 1]
                dec_label = dec_label[:, :max_len]

                label = labels[i:i + 1]
                max_len = (label != 0).sum().data[0]
                label = label[:, :max_len]

                att_label = att_labels[i:i + 1]
                if i in select_ixs:
                    select_ix = torch.LongTensor([0]).cuda()
                else:
                    select_ix = torch.LongTensor().cuda()

                tic = time.time()
                scores, loss, rel_ixs, sub_attn, loc_attn, rel_attn, weights, vis_res_loss, att_res_loss, lang_res_loss = \
                    model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                          Feats['cxt_fc7'], Feats['cxt_lfeats'], label, enc_label, dec_label, att_label, select_ix, att_weights)

                scores = scores.squeeze(0).data.cpu().numpy()
                rel_ixs = rel_ixs.squeeze(0).data.cpu().numpy().tolist()

                loss = loss.data[0].item()

                if opt['loss_combined'] == 0:
                    vis_res_loss=vis_res_loss.data[0].item()
                    lang_res_loss = lang_res_loss.data[0].item()
                vis_res_loss_sum += vis_res_loss
                lang_res_loss_sum += lang_res_loss

                pred_ix = np.argmax(scores)
                gd_ix = data['gd_ixs'][i]
                loss_sum += loss
                loss_evals += 1

                pred_box = loader.Anns[ann_ids[pred_ix]]['box']
                gd_box = data['gd_boxes'][i]

                if opt['use_IoU'] > 0:
                    if computeIoU(pred_box, gd_box) >= 0.5:
                        acc += 1
                else:
                    if pred_ix == gd_ix:
                        acc += 1

                rel_ix = rel_ixs[pred_ix]

                entry = {}
                entry['sent_id'] = sent_id
                entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0]
                entry['gd_ann_id'] = data['ann_ids'][gd_ix]
                entry['pred_ann_id'] = data['ann_ids'][pred_ix]
                entry['pred_score'] = scores.tolist()[pred_ix]

                entry['sub_attn'] = sub_attn.data.cpu().numpy().tolist()
                entry['loc_attn'] = loc_attn.data.cpu().numpy().tolist()
                entry['rel_attn'] = rel_attn.data.cpu().numpy().tolist()
                entry['rel_ann_id'] = data['cxt_ann_ids'][pred_ix][rel_ix]

                entry['weights'] = weights.data.cpu().numpy().tolist()

                predictions.append(entry)
                toc = time.time()
                model_time += (toc - tic)

                if num_sents > 0  and loss_evals >= num_sents:
                    finish_flag = True
                    break
            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if verbose:
                print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
                      (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))
            model_time = 0

            if finish_flag or data['bounds']['wrapped']:
                break

    return loss_sum / loss_evals, acc / loss_evals, predictions, \
           vis_res_loss_sum / loss_evals, lang_res_loss_sum / loss_evals


