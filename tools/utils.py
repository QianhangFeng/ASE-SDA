#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Evaluation tools adapted from https://github.com/fartashf/vsepp/blob/master/evaluation.py
"""

import numpy as np
import torch
import random
import sys
from loguru import logger
from tools.file_io import load_pickle_file
from gensim.models.word2vec import Word2Vec
from pathlib import Path
import os


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def align_word_embedding(words_list_path, model_path, nhid):

    words_list = load_pickle_file(words_list_path)
    w2v_model = Word2Vec.load(model_path)
    ntoken = len(words_list)
    weights = np.zeros((ntoken, nhid))
    for i, word in enumerate(words_list):
        if word in w2v_model.wv.index_to_key:
            embedding = w2v_model.wv[word]
            weights[i] = embedding
    weights = torch.from_numpy(weights).float()
    return weights


def set_logger(exp_name):
    log_output_dir = Path('outputs', exp_name, 'logging')
    model_output_dir = Path('outputs', exp_name, 'models')
    # model_output_dir_lightning = Path('outputs', exp_name, 'models/lightning')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    # model_output_dir_lightning.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    return model_output_dir, log_output_dir


def decode_output(predicted_output, ref_captions, file_names):
    captions_pred, captions_gt, f_names = [], [], []

    for pred_cap, gt_caps, f_name in zip(predicted_output, ref_captions, file_names):

        f_names.append(f_name)
        captions_pred.append({'file_name': f_name, 'caption_predicted': pred_cap})
        ref_caps_dict = {'file_name': f_name}
        for i, cap in enumerate(gt_caps):
            ref_caps_dict[f"caption_{i + 1}"] = cap
        captions_gt.append(ref_caps_dict)

    return captions_pred, captions_gt
