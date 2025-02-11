"""Main training loop"""

from __future__ import division

import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append('.')
from utils import tools
from task_and_network import *

from task_and_network.task import generate_trials
from task_and_network.network_ei import Model, get_perf
import pickle

def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    #num_ring = 1
    n_rule = task.get_num_rule(ruleset)

    #n_eachring = 8
    #n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',#modified by yichen 'cross_entropy' for one_hot input 
            # Input_location_type add by yichen
            #'in_loc_type': 'one_hot',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 0.95,
            # Mature performance
            'mature_target_perf': 0.95, #add by yichen
            # Mid performance
            'mid_target_perf': 0.65, #add by yichen
            # Early performance
            'early_target_perf': 0.35, #add by yichen
            # number of units each ring
            'n_eachring': 8,
            # number of rings
            'num_ring': 1,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': None, #1+num_ring*n_eachring,
            # number of input units
            'n_input': None, #n_input,
            # number of output units
            'n_output': None, #n_output,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            }

    return hp


def do_eval(sess, model, log, rule_train):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    rule_grow = dict()# add by yichen
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training '+rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(
                rule_test, hp, 'random', batch_size=batch_size_test_rep)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)

            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_'+rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_'+rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()
        # add by yichen=====================================================
        if np.mean(perf_tmp) <= hp['early_target_perf']:
            rule_grow[rule_test] = '=>E'
        elif np.mean(perf_tmp) <= hp['mid_target_perf']:
            rule_grow[rule_test] = 'E=>MID'
        elif np.mean(perf_tmp)<=hp['mature_target_perf']:
            rule_grow[rule_test] = 'MID=>A'
        else:
            rule_grow[rule_test] = 'MATURE'
        #===================================================================

    # TODO: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    log['growth'].append(rule_grow)# add by yichen

    # Saving the model
    model.save(log['model_dir']+'/'+str(log['trials'][-1]))
    tools.save_log(log)


    return log


def train(model_dir,
          hp=None,
          log=None,
          max_steps=1e7,
          display_step=500,
          ruleset='all_new',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          load_dir=None,
          trainables=None,
          continue_after_target_reached=False, #if set to True, the training will continue for 50 more steps after reaching the target performance
          ):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/trial_number/model.ckpt
        trial_number is also referred as model_index in downstream analysis
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_start'] = 1+hp['num_ring']*hp['n_eachring']
    hp['n_input'], hp['n_output'] = 1+hp['num_ring']*hp['n_eachring']+hp['n_rule'], hp['n_eachring']+1

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))
    tools.save_hp(hp, model_dir)

    # Build the model
    model = Model(model_dir, hp=hp)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    if log is None: #if log is not None, continue from where the training stopped last time
        log = defaultdict(list)
        log['model_dir'] = model_dir
    
    # Record time
    t_start = time.time()

    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        elif trainables == 'input':
            # train all nputs
            var_list = [v for v in model.var_list
                        if ('input' in v.name) and ('rnn' not in v.name)]
        elif trainables == 'rule':
            # train rule inputs only
            var_list = [v for v in model.var_list if 'rule_input' in v.name]
        else:
            raise ValueError('Unknown trainables')
        model.set_optimizer(var_list=var_list)

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w - w_val))

            model.set_optimizer(var_list=var_list)

        # partial weight training
        if ('p_weight_train' in hp and
            (hp['p_weight_train'] is not None) and
            hp['p_weight_train'] < 1.0):
            for w in model.weight_list:
                w_val = sess.run(w)
                w_size = sess.run(tf.size(w))
                w_mask_tmp = np.linspace(0, 1, w_size)
                hp['rng'].shuffle(w_mask_tmp)
                ind_fix = w_mask_tmp > hp['p_weight_train']
                w_mask = np.zeros(w_size, dtype=np.float32)
                w_mask[ind_fix] = 1e-1  # will be squared in l2_loss
                w_mask = tf.constant(w_mask)
                w_mask = tf.reshape(w_mask, w.shape)
                model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
            model.set_optimizer(var_list=var_list)

        if log['trials']:
            step = int(log['trials'][-1]/hp['batch_size_train']) #continue from where the training stopped last time
        else:
            step = 0
        after_target_reached_step_count=0
        gradient_list = []
        for i in range(4):
            gradient_list.append([])
        while 1:#step * hp['batch_size_train'] <= max_steps:
            try:
                # Validation
                if step % display_step == 0:
                    trial_number = step * hp['batch_size_train']# add by yichen
                    log['trials'].append(trial_number)
                    tools.mkdir_p(model_dir+'/'+str(trial_number))# add by yichen
                    time_trained = time.time()-t_start
                    log['times'].append(time_trained)
                    day = time_trained//(60*60*24)
                    hour = time_trained%(60*60*24)//(60*60)
                    minute = time_trained%(60*60)//60
                    second = time_trained%60
                    print("time trained: "+str(day)+"D-"+str(hour)+"H-"+str(minute)+"M-"+str(second)+"S")
                    log = do_eval(sess, model, log, hp['rule_trains'])
                    #check if minimum performance is above target 

                    if log['perf_min'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        if continue_after_target_reached:
                            after_target_reached_step_count+=1
                            if after_target_reached_step_count>50:
                                break
                        else:
                            break

                # Training
                rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                                  p=hp['rule_probs'])
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                        rule_train_now, hp, 'random',
                        batch_size=hp['batch_size_train'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)
                sess.run(model.train_step, feed_dict=feed_dict)
                # grad_list = sess.run(model.grad_list,feed_dict=feed_dict)
                # gradient_list[0].append(grad_list[0])
                # gradient_list[1].append(grad_list[1])
                # gradient_list[2].append(grad_list[2])
                # gradient_list[3].append(grad_list[3])
                # with open(model_dir + '/gradient_list.pickle', 'wb') as f:
                #     pickle.dump(gradient_list, f)

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")