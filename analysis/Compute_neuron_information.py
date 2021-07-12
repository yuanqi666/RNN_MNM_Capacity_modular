# basic packages #
import os
import numpy as np

# for ANOVA and paired T test analysis and plot #
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import sys
sys.path.append('.')
from utils.tools import slice_H, model_list_parser


def compute_neuron_information_single_H(hp,H,task_info,epoch, # the epoch should not ne 'fix1'
                        annova_p_thresh = 0.05,
                        paired_ttest_p_thresh = 0.05, 
                        active_thresh = 1e-3,
                        norm = True,):
    neuron_info = dict()

    if np.all(np.isnan(H)):
        raise ValueError("All elements in H are NaN")

    for neuron_type in ['all','excitatory','inhibitory','mix','significant']:
        neuron_info[neuron_type] = list()

    epoch_mean_H = np.nanmean(slice_H(H,time_range=task_info['epochs'][epoch]),axis=0)
    fix_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['fix1']),axis=0)
    epoch_mean_H_normed = epoch_mean_H/fix_epoch_H - 1

    stim1_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['stim1']),axis=0)

    for neuron in range(hp['n_rnn']):
        location_mean_firerate_abs = list()
        location_mean_firerate_normed = list()
        t_test_passed_loc_count = 0 #count how many locations that the neuron passed the paired T test

        for loc in task_info['loc_set'][epoch]:
            location_mean_firerate_abs.append(np.nanmean(epoch_mean_H[task_info['input_loc'][epoch]==loc,neuron]))
            location_mean_firerate_normed.append(np.nanmean(epoch_mean_H_normed[task_info['input_loc'][epoch]==loc,neuron]))

            #Paired T test of neuron's fix1 and stim1 epoch activity at each location
            fix_ttest = fix_epoch_H[task_info['input_loc']['stim1']==loc,neuron]
            stim1_ttest = stim1_epoch_H[task_info['input_loc']['stim1']==loc,neuron]
            paired_ttest_result = ttest_rel(fix_ttest, stim1_ttest, nan_policy='omit')[1]
            if not isinstance(paired_ttest_result,float) or np.isnan(paired_ttest_result) or paired_ttest_result <= paired_ttest_p_thresh:
                t_test_passed_loc_count += 1

        location_mean_firerate_abs = np.array(location_mean_firerate_abs)
        location_mean_firerate_normed = np.array(location_mean_firerate_normed)
        if norm:
            location_mean_firerate = location_mean_firerate_normed
        else:
            location_mean_firerate = location_mean_firerate_abs
        
        try:
            #find the location which has the maximum fire rate
            max_location = task_info['loc_set'][epoch][np.nanargmax(location_mean_firerate)]
        except:
            #exclude those neurons which has no value (nan) in all the trials
            continue

        neuron_info['all'].append((neuron,max_location,None))

        # Exclude those neurons which didn't pass paired T test in all the locations
        # and exclude those neurons whose fire rates did not pass the activity threshold in all the locations
        if np.nanmax(location_mean_firerate_abs)>active_thresh and t_test_passed_loc_count > 0:
            
            #ANOVA Analysis
            if norm:
                firerate = epoch_mean_H[:,neuron]
            else:
                firerate = epoch_mean_H_normed[:,neuron]

            df = pd.DataFrame({'Loc':task_info['input_loc'][epoch],
                               'Firerate':firerate,})
            try:
                model = ols('Firerate~C(Loc)',data=df,missing='drop').fit() #drop rows (loc-fire rate pair) that contains nan
                anova_table = anova_lm(model, typ = 2)
                anova_p = anova_table['PR(>F)'][0]
            except ValueError as e:
                error_string = str(e)
                if error_string == "must have at least one row in constraint matrix": #only one location's trials are left
                    anova_p = np.nan
                else: # other errors
                    raise ValueError(str(e))

            if np.isnan(anova_p) or anova_p <= annova_p_thresh: #selective
                if np.nanmax(location_mean_firerate_normed) < 0:
                    neuron_info['inhibitory'].append((neuron,max_location,anova_p))
                elif np.nanmin(location_mean_firerate_normed) >= 0:
                    neuron_info['excitatory'].append((neuron,max_location,anova_p))
                else:
                    neuron_info['mix'].append((neuron,max_location,anova_p))

    neuron_info['significant'] = neuron_info['excitatory'] + neuron_info['mix']

    if not neuron_info['all']:
        raise ValueError("All neurons are dropped")

    return neuron_info

def compute_neuron_information(hp, log, model_dir, rule, epoch, model_list, 
                                task_mode='test', norm=True,
                                task_info_dict=None,H_dict=None,):

    if task_info_dict is None or H_dict is None:
        from .Compute_H import compute_H
        task_info_dict, H_dict = compute_H(hp, log, model_dir, rule, model_list, task_mode=task_mode)

    model_list = model_list_parser(hp,log,rule,model_list,)

    neuron_info_dict = dict()
    for m_key in model_list.keys():
        neuron_info_dict[m_key] = dict()
        for model_index in model_list[m_key]:
            neuron_info_dict[m_key][model_index] = compute_neuron_information_single_H(hp,H_dict[m_key][model_index],task_info_dict[m_key][model_index],epoch,norm = norm,)
    
    return neuron_info_dict