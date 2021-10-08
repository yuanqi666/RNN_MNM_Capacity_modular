"""Main training loop"""

from __future__ import division

import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append('.')
from utils import tools
from task_and_network import *

from task_and_network.task import generate_trials
from task_and_network.network import Model, get_perf


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

def train_sequential(
        model_dir,
        rule_trains,
        hp=None,
        #max_steps=1e7,
        display_step=500,
        ruleset='mante',
        seed=0,
        continue_after_target_reached=False, #if set to True, the training will continue for 50 more steps after reaching the target performance
        ):
    '''Train the network sequentially.

    Args:
        model_dir: str, training directory
        rule_trains: a list of list of tasks to train sequentially
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps for each list of tasks
        display_step: int, display steps
        ruleset: the set of rules to train
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    '''

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

    hp['rule_trains'] = rule_trains
    # Get all rules by flattening the list of lists
    hp['rules'] = list(set([r for rs in rule_trains for r in rs]))

    # Number of training iterations for each rule
    #rule_train_iters = [len(r)*max_steps for r in rule_trains]

    tools.save_hp(hp, model_dir)
    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Using continual learning or not
    c, ksi = hp['c_intsyn'], hp['ksi_intsyn']

    # Build the model
    model = Model(model_dir, hp=hp)
    
    grad_unreg = tf.gradients(model.cost_lsq, model.var_list)

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # tensorboard summaries
    placeholders = list()
    for v_name in ['Omega0', 'omega0', 'vdelta']:
        for v in model.var_list:
            placeholder = tf.placeholder(tf.float32, shape=v.shape)
            tf.summary.histogram(v_name + '/' + v.name, placeholder)
            placeholders.append(placeholder)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(model_dir + '/tb')

    def relu(x):
        return x * (x > 0.)

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            raise NotImplementedError()

        # Looping
        step_total = 0
        for i_rule_train, rule_train in enumerate(hp['rule_trains']):
            step = 0

            # At the beginning of new tasks
            # Only if using intelligent synapses
            v_current = sess.run(model.var_list)

            if i_rule_train == 0:
                v_anc0 = v_current
                Omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
                omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
                v_delta = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
            elif c > 0:
                v_anc0_prev = v_anc0
                v_anc0 = v_current
                v_delta = [v-v_prev for v, v_prev in zip(v_anc0, v_anc0_prev)]

                # Make sure all elements in omega0 are non-negative
                # Penalty
                Omega0 = [relu(O + o / (v_d ** 2 + ksi))
                          for O, o, v_d in zip(Omega0, omega0, v_delta)]
                
                # Update cost
                model.cost_reg = tf.constant(0.)
                for v, w, v_val in zip(model.var_list, Omega0, v_current):
                    model.cost_reg += c * tf.reduce_sum(
                        tf.multiply(tf.constant(w),
                                    tf.square(v - tf.constant(v_val))))
                model.set_optimizer()

            # Store Omega0 to tf summary
            feed_dict = dict(zip(placeholders, Omega0 + omega0 + v_delta))
            summary = sess.run(merged, feed_dict=feed_dict)
            test_writer.add_summary(summary, i_rule_train)

            # Reset
            omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]

            # Keep training until reach max iterations
            after_target_reached_step_count=0
            while 1:
            #while (step * hp['batch_size_train'] <=
            #       rule_train_iters[i_rule_train]):
                # Validation
                if step % display_step == 0:
                    trial = step_total * hp['batch_size_train']
                    log['trials'].append(trial)
                    tools.mkdir_p(model_dir+'/'+str(trial))# add by yichen
                    time_trained = time.time()-t_start
                    log['times'].append(time_trained)
                    day = time_trained//(60*60*24)
                    hour = time_trained%(60*60*24)//(60*60)
                    minute = time_trained%(60*60)//60
                    second = time_trained%60
                    print("time trained: "+str(day)+"D-"+str(hour)+"H-"+str(minute)+"M-"+str(second)+"S")
                    log['rule_now'].append(rule_train)
                    log = do_eval(sess, model, log, rule_train)
                    if log['perf_avg'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        #if continue_after_target_reached:
                        if continue_after_target_reached and i_rule_train+1 == len(hp['rule_trains']):#continue train only after the last set of rules
                            after_target_reached_step_count+=1
                            #if after_target_reached_step_count>10:
                            if after_target_reached_step_count>50:
                                break
                        else:
                            break

                # Training
                rule_train_now = hp['rng'].choice(rule_train)
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                        rule_train_now, hp, 'random',
                        batch_size=hp['batch_size_train'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)

                # Continual learning with intelligent synapses
                v_prev = v_current

                # This will compute the gradient BEFORE train step
                _, v_grad = sess.run([model.train_step, grad_unreg],
                                     feed_dict=feed_dict)
                # Get the weight after train step
                v_current = sess.run(model.var_list)

                # Update synaptic importance
                omega0 = [
                    o - (v_c - v_p) * v_g for o, v_c, v_p, v_g in
                    zip(omega0, v_current, v_prev, v_grad)
                ]

                step += 1
                step_total += 1

        print("Optimization Finished!")