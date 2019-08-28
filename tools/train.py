from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import time
import random

# model
import _init_paths
from loaders.dataloader import DataLoader
from layers.match import AdaptiveReconstruct
import models.utils as model_utils
import models.eval as eval
from opt import parse_opt

import torch

def main(args):
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['exp_id'])
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # set random seed
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    # set up loader
    data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
    loader = DataLoader(data_h5=data_h5, data_json=data_json)

    # prepare feats
    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir = osp.join('cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)

    loader.prepare_mrcn(head_feats_dir, args)

    ann_feats = osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
    loader.loadFeats({'ann': ann_feats})

    # set up model
    opt['vocab_size'] = loader.vocab_size
    opt['fc7_dim'] = loader.fc7_dim
    opt['pool5_dim'] = loader.pool5_dim
    opt['num_atts'] = loader.num_atts
    model = AdaptiveReconstruct(opt)

    infos = {}
    if opt['start_from'] is not None:
        pass
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)


    att_weights = loader.get_attribute_weights()

    if opt['gpuid'] >= 0:
        model.cuda()


    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt['learning_rate'],
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'])

    data_time, model_time = 0, 0
    lr = opt['learning_rate']
    best_prediction, best_overall = None, None
    while True:
        model.train()
        optimizer.zero_grad()

        T = {}

        tic = time.time()
        data = loader.getBatch('train', opt)

        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        Feats = data['Feats']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']

        T['data'] = time.time() - tic

        tic = time.time()
        scores, loss,_,_,_,_,_,vis_res_loss, att_res_loss, lang_res_loss = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                                         Feats['cxt_fc7'], Feats['cxt_lfeats'], labels, enc_labels, dec_labels, att_labels, select_ixs, att_weights)

        loss.backward()
        model_utils.clip_gradient(optimizer, opt['grad_clip'])
        optimizer.step()
        T['model'] = time.time()-tic
        wrapped = data['bounds']['wrapped']

        data_time += T['data']
        model_time += T['model']

        if iter % opt['losses_log_every'] == 0:
            loss_history[iter]=(loss.data[0]).item()
            print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
                  % (iter, epoch, loss.data[0].item(), lr, data_time / opt['losses_log_every'], model_time/opt['losses_log_every']))
            data_time, model_time = 0, 0

        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor = 0.1**frac
            lr = opt['learning_rate'] * decay_factor
            model_utils.set_lr(optimizer, lr)

        if (iter) % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:
            val_loss, acc, predictions, val_vis_res_loss, val_lang_res_loss = eval.eval_split(loader, model, 'testB', opt)
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
            val_accuracies += [(iter, acc)]
            print('validation loss: %.2f' % val_loss)
            print('validation acc : %.2f%%\n' % (acc * 100.0))

            current_score = acc
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_predictions = predictions
                checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = model
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)

            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_predictions'] = predictions if best_predictions is None else best_predictions

            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            infos['word_to_ix'] = loader.word_to_ix
            infos['att_to_ix'] = loader.att_to_ix
            with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w', encoding="utf8") as io:
                json.dump(infos, io)

        iter += 1
        if wrapped:
            epoch += 1
        if iter >= opt['max_iters'] and opt['max_iters'] > 0:
            break

if __name__ == '__main__':
    args = parse_opt()
    main(args)
