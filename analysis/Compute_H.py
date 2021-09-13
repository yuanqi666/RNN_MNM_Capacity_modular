# basic packages #
import os
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# trial generation and network building #

import sys
sys.path.append('.')
from utils import tools
from utils.tools import model_list_parser

from task_and_network.task import generate_trials
from task_and_network.network import Model, get_perf

def compute_single_H(hp, log, model_dir, rule, model_index, trial=None, task_mode='test'):

    if trial is None:
        trial = generate_trials(rule, hp, task_mode, noise_on=False)
    
    sub_dir = model_dir+'/'+str(model_index)+'/'
    model = Model(sub_dir, hp=hp)
    with tf.Session() as sess:
        model.restore()
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y_hat = sess.run([model.h,model.y_hat], feed_dict=feed_dict)

    h *= (1000.0/hp['dt']) #1s fire rate

    task_info = dict()
    task_info['epochs'] = trial.epochs
    task_info['input_loc'] = trial.location_info
    loc_set = dict()
    try:
        for key, value in trial.location_info.items():
            loc_set[key] = sorted(set(value))
        task_info['loc_set'] = loc_set
    except:
        pass

    perf = get_perf(y_hat, trial.y_loc)
    task_info['correct_trials'] = [x==1 for x in perf]
    task_info['error_trials'] = [x==0 for x in perf]

    task_info['test_perf'] = np.mean(perf)
    task_info['train_perf'] = log['perf_'+rule][model_index//log['trials'][1]]

    return h, task_info

def compute_H(hp, log, model_dir, rule, model_list, task_mode='test'):

    model_list = model_list_parser(hp,log,rule,model_list)

    task_info_dict=dict()
    H_dict = dict()

    for m_key in model_list.keys():
        task_info_dict[m_key] = dict()
        H_dict[m_key] = dict()

        for model_index in model_list[m_key]:
            H_dict[m_key][model_index], task_info_dict[m_key][model_index] = compute_single_H(hp,log,model_dir, rule, model_index, task_mode=task_mode)

    return H_dict, task_info_dict