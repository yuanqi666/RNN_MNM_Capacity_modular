import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,model_list_parser,mkdir_p,get_mon_trials,write_excel_xls

from .Compute_H import compute_single_H

# for 2 way ANOVA#
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def analyze_MNM_effect_2ANOVA(hp,log,model_dir,model_list,rule,epoch='stim2',
                        task_info_dict=None,H_dict=None,
                        task_mode='test',perf_type = 'train_perf',):

    model_list = model_list_parser(hp, log, rule, model_list,)
    
    # empty data
    if task_info_dict is None or H_dict is None:
        task_info_dict, H_dict = dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/MNM_effect_2way_ANOVA/'+epoch+'/'
    mkdir_p(save_dir)

    for m_key in model_list.keys():
        
        for model_index in model_list[m_key]:
            excel_content = list()
            excel_content.append(['Neuron index','Loc-df','Loc-F','Loc-PR(>F)','MNM-df','MNM-F','MNM-PR(>F)','Loc:MNM-df','Loc:MNM-F','Loc:MNM-PR(>F)'])

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]
            
            perf = task_info[perf_type]

            MNM = get_mon_trials(task_info['input_loc']['stim1'],task_info['input_loc']['stim2'],'match')
            in_locs = task_info['input_loc'][epoch]

            firerates = np.nanmean(slice_H(H,time_range=task_info['epochs'][epoch]),axis=0)

            for neuron in range(hp['n_rnn']):
                neuron_firerate = firerates[:,neuron]
                df = pd.DataFrame({'Loc':in_locs,
                           'MNM':MNM,
                           'Firerate':neuron_firerate,})
                try:
                    model = ols('Firerate ~ C(Loc) + C(MNM) + C(Loc):C(MNM)', data=df,missing='drop').fit()
                    anova_table = anova_lm(model, typ=2)

                    excel_content.append([neuron,anova_table['df'][0],anova_table['F'][0],anova_table['PR(>F)'][0],
                                        anova_table['df'][1],anova_table['F'][1],anova_table['PR(>F)'][1],
                                        anova_table['df'][2],anova_table['F'][2],anova_table['PR(>F)'][2],])
                except ValueError as e:
                    error_string = str(e)
                    if error_string == "must have at least one row in constraint matrix": #only one location's trials are left
                        continue
                    else: # other errors
                        raise ValueError(str(e))
            
            save_name = save_dir+'MNM_effect_2way_ANOVA_trialnum'+str(model_index)+'_perf%.2f'%(perf)+'.xls'
            write_excel_xls(save_name, '2 way ANOVA', excel_content)