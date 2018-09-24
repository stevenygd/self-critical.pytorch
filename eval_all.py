from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

opt = opts.parse_opt()

# Setting up model
opt.use_att = utils.if_use_att(opt.caption_model)
if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length

print(opt.checkpoint_path)

infos = {}
histories = {}
if opt.start_from is not None:
    # open old infos and check if models are compatible
    print(os.getcwd())
    with open(os.path.join(os.getcwd(), opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
        infos = cPickle.load(f)
        saved_model_opt = infos['opt']
        need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
            histories = cPickle.load(f)

iteration = infos.get('iter', 0)
epoch = infos.get('epoch', 0)

val_result_history = histories.get('val_result_history', {})
loss_history = histories.get('loss_history', {})
lr_history = histories.get('lr_history', {})
ss_prob_history = histories.get('ss_prob_history', {})

loader.iterators = infos.get('iterators', loader.iterators)
loader.split_ix = infos.get('split_ix', loader.split_ix)
if opt.load_best_score == 1:
    best_val_score = infos.get('best_val_score', None)

model = models.setup(opt).cuda()
# dp_model = torch.nn.DataParallel(model)
dp_model = model

update_lr_flag = True
crit = utils.LanguageModelCriterion()

splits = ['train', 'val', 'test']
predictions = []

for sp in splits:
    eval_kwargs = {'split': sp,
                   'dataset': opt.input_json,
                   'num_images': -1,
                   'lang_eval': 0}
    eval_kwargs.update(vars(opt))
    # import pdb; pdb.set_trace()
    predictions += eval_utils.eval_split(dp_model, crit, loader, eval_kwargs, no_score=True)

out = eval_utils.language_eval(None, predictions, opt.id, 'custom_eval')

from pprint import pprint
pprint(out)

with open(os.path.join(opt.checkpoint_path, 'custom_eval_'+opt.id+'.pkl'), 'wb+') as f:
    cPickle.dump(out, f)
