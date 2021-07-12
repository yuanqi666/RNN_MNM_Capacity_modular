"""Utility functions."""

import os
import errno
import six
import json
import pickle
import numpy as np


def gen_feed_dict(model, trial, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {model.x: trial.x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}
    elif hp['in_type'] == 'multi':
        n_time, batch_size = trial.x.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = \
                trial.x[:, i, :hp['rule_start']]

        feed_dict = {model.x: x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}
    else:
        raise ValueError()

    return feed_dict


def _contain_model_file(model_dir):
    """Check if the directory contains model files."""
    for f in os.listdir(model_dir):
        if 'model.ckpt' in f:
            return True
    return False


def _valid_model_dirs(root_dir):
    """Get valid model directories given a root directory."""
    return [x[0] for x in os.walk(root_dir) if _contain_model_file(x[0])]


def valid_model_dirs(root_dir):
    """Get valid model directories given a root directory(s).

    Args:
        root_dir: str or list of strings
    """
    if isinstance(root_dir, six.string_types):
        return _valid_model_dirs(root_dir)
    else:
        model_dirs = list()
        for d in root_dir:
            model_dirs.extend(_valid_model_dirs(d))
        return model_dirs


def load_log(model_dir):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, 'log.json')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log


def save_log(log): 
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, 'hparams.json')  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, 'r') as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['rng'] = np.random.RandomState(hp['seed']+1000)
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)


def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


def find_all_models(root_dir, hp_target):
    """Find all models that satisfy hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters

    Returns:
        model_dirs: list of model directories
    """
    dirs = valid_model_dirs(root_dir)

    model_dirs = list()
    for d in dirs:
        hp = load_hp(d)
        if all(hp[key] == val for key, val in hp_target.items()):
            model_dirs.append(d)

    return model_dirs


def find_model(root_dir, hp_target, perf_min=None):
    """Find one model that satisfies hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters
        perf_min: float or None. If not None, minimum performance to be chosen

    Returns:
        d: model directory
    """
    model_dirs = find_all_models(root_dir, hp_target)
    if perf_min is not None:
        model_dirs = select_by_perf(model_dirs, perf_min)

    if not model_dirs:
        # If list empty
        print('Model not found')
        return None, None

    d = model_dirs[0]
    hp = load_hp(d)

    log = load_log(d)
    # check if performance exceeds target
    if log['perf_min'][-1] < hp['target_perf']:
        print("""Warning: this network perform {:0.2f}, not reaching target
              performance {:0.2f}.""".format(
              log['perf_min'][-1], hp['target_perf']))

    return d


def select_by_perf(model_dirs, perf_min):
    """Select a list of models by a performance threshold."""
    new_model_dirs = list()
    for model_dir in model_dirs:
        log = load_log(model_dir)
        # check if performance exceeds target
        if log['perf_min'][-1] > perf_min:
            new_model_dirs.append(model_dir)
    return new_model_dirs


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def slice_H(H,time_range=None,trial_range=None,neuron_range=None):
    """
    time/trial/neuron range can be list, tuple, int or None.
    int and list refer to specific index(es)
    tuple refer to a range of indexes
    None refer to all the indexes in that axis
    """
    def expand(x,axis,shape):
        if x is None:
            return range(shape[axis])
        elif isinstance(x, tuple):
            if x[0] is None:
                start = 0
            else:
                start = x[0]
            if x[1] is None:
                end = shape[axis]
            else:
                end = x[1]
            return range(start,end)
        elif isinstance(x,int):
            return [x]
        elif isinstance(x,list):
            return x
        else:
            raise TypeError("time/trial/neuron range should be list, tuple, int or None.")
    
    H_shape = np.shape(H)
    sliced_H = np.full(H_shape, np.nan)
    time_range = expand(time_range,axis=0,shape=H_shape)
    trial_range = expand(trial_range,axis=1,shape=H_shape)
    neuron_range = expand(neuron_range,axis=2,shape=H_shape)

    sliced_H[np.ix_(time_range,trial_range,neuron_range)] = H[np.ix_(time_range,trial_range,neuron_range)]

    return sliced_H



def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H

def smooth(ori_array, smooth_window,):# if you don't need smooth, set smooth_window to 0
    smoothed_array = list()
    for i in range(len(ori_array)):    
        avg = np.mean(ori_array[i:i+smooth_window+1])
        smoothed_array.append(avg)

    return smoothed_array

def auto_model_select(hp,log,rules=None,smooth_window=9,perf_margin=0.05,max_model_num_limit=30):# if you don't need smooth, set smooth_window to 0
    model_select = dict()
    if rules is None:
        rules = hp['rules']
    for rule in rules:
        model_select[rule] = dict()
        for key in ['mature','mid','early']:
            model_select[rule][key] = list()

        smooth_growth_curve = smooth(log['perf_'+rule], smooth_window=smooth_window,)

        for i in range(len(smooth_growth_curve)):
            if hp['early_target_perf']-perf_margin <= smooth_growth_curve[i] <= hp['early_target_perf']+perf_margin:
                model_select[rule]['early']+=log['trials'][i:i+smooth_window+1]
            elif hp['mid_target_perf']-perf_margin <= smooth_growth_curve[i] <= hp['mid_target_perf']+perf_margin:
                model_select[rule]['mid']+=log['trials'][i:i+smooth_window+1]
            elif hp['mature_target_perf']-perf_margin <= smooth_growth_curve[i]:
                model_select[rule]['mature']+=log['trials'][i:i+smooth_window+1]
            else:
                continue

        for key,value in model_select[rule].items():
            model_select[rule][key] = sorted(set(value))
    
        if len(model_select[rule]['early'])>max_model_num_limit:
            model_select[rule]['early'] = model_select[rule]['early'][-max_model_num_limit:]
        if len(model_select[rule]['mid'])>max_model_num_limit:
            exee = len(model_select[rule]['mid'])-max_model_num_limit
            model_select[rule]['mid'] = model_select[rule]['mid'][int(exee//2):-int(exee//2)]
        if len(model_select[rule]['mature'])>max_model_num_limit:
            model_select[rule]['mature'] = model_select[rule]['mature'][:max_model_num_limit]

    return model_select

def get_mon_trials(stim1_locs, stim2_locs, MoNM):

    if MoNM == 'match':
        return [st[0]==st[1] for st in zip(stim1_locs,stim2_locs)]
    elif MoNM == 'non-match':
        return [st[0]!=st[1] for st in zip(stim1_locs,stim2_locs)]

def model_list_parser(hp, log, rule, model_list, margin=0):
    if isinstance(model_list, dict):
        #model_list = model_list[rule]
        return model_list
    elif isinstance(model_list, list):
        temp_list = dict()
        for m_key in ['mature','mid','early']:
            temp_list[m_key] = list()
        
        for model_index in model_list:
            growth = log['perf_'+rule][model_index//log['trials'][1]]
            #if  growth > hp['mature_target_perf']-margin:
            if  growth > hp['mid_target_perf']-margin:
                temp_list['mature'].append(model_index)
            #elif growth > hp['mid_target_perf']-margin:
            elif growth > hp['early_target_perf']-margin:
                temp_list['mid'].append(model_index)
            else:
                temp_list['early'].append(model_index)

        model_list = temp_list

    return model_list  

def max_central(max_index,tuning1,tuning2=None): #tuning1->max central, tuning2's index change with tuning1 

    temp_len = len(tuning1)
    if temp_len%2 == 0:
        mc_len = temp_len + 1
    else:
        mc_len = temp_len

    firerate_max_central = np.zeros(mc_len)
    if tuning2 is not None:
        tuning2_shift = np.zeros(mc_len)

    for i in range(temp_len):
        new_index = (i-max_index+temp_len//2)%temp_len
        firerate_max_central[new_index] = tuning1[i]
        if tuning2 is not None:
            tuning2_shift[new_index] = tuning2[i]
    if temp_len%2 == 0:
        firerate_max_central[-1] = firerate_max_central[0]
        if tuning2 is not None:
            tuning2_shift[-1] = tuning2_shift[0]

    if tuning2 is not None:
        return firerate_max_central, tuning2_shift
    else:
        return firerate_max_central

def gaussian_curve_fit(tuning):
    import math
    from scipy.optimize import curve_fit

    def gaussian(x, a,u, sig):
        return a*np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))
    
    temp_x = np.arange(len(tuning))
    gaussian_x = np.arange(-0.1,len(tuning)-0.9,0.1)
    paras , _ = curve_fit(gaussian,temp_x,tuning+(-1)*np.min(tuning),\
        p0=[np.max(tuning)+1,len(tuning)//2,1])
    gaussian_y = gaussian(gaussian_x,paras[0],paras[1],paras[2])-np.min(tuning)*(-1)
    return gaussian_x, gaussian_y, paras

def average_H_within_location(H,input_locs,loc_set):

    averaged_within_location=np.nanmean(H[:,input_locs==loc_set[0],:],axis=1,keepdims=True)

    for loc in loc_set[1:]:
        averaged_within_location = np.concatenate((averaged_within_location,np.nanmean(H[:,input_locs==loc,:],axis=1,keepdims=True)),axis=1)

    return averaged_within_location

def write_excel_xls(path, sheet_name, value):
    import xlwt
    index = len(value)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])
    workbook.save(path)

def split_test_train(x,y):

    labels = sorted(set(y))

    y_test = list()
    y_train = list()

    for label in labels:
        x_temp = x[y==label,:]
        trial_per_label = len(x_temp)
        test_index = list(range(0,trial_per_label,4))
        train_index = [i for i in range(trial_per_label) if i not in test_index]
        if label == labels[0]:
            x_train = x_temp[train_index,:]
            x_test = x_temp[test_index,:]
        else:
            x_train = np.concatenate((x_train,x_temp[train_index,:]),axis=0)
            x_test = np.concatenate((x_test,x_temp[test_index,:]),axis=0)

        for i in range(len(train_index)):
            y_train.append(label)
        for i in range(len(test_index)):
            y_test.append(label)


    return x_train, x_test, np.array(y_train), np.array(y_test)

def z_score_norm(ori_data):
    ori_data=np.array(ori_data)
    return (ori_data-np.nanmean(ori_data))/np.nanstd(ori_data)