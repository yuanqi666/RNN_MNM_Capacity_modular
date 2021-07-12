import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,max_central,model_list_parser,mkdir_p,get_mon_trials,write_excel_xls

from .Compute_H import compute_single_H
from .Analyze_tuning_curve import analyze_tuning_curve_single_H

import matplotlib.pyplot as plt

# for 2 way ANOVA#
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def plot_mix_selectivity_tuning(model_dir,rule,norm,plot_information,save_formats=['pdf','png','eps']):

    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/mix_selectivity_Analysis/'
    mkdir_p(save_dir)

    for m_key,plot_info in plot_information.items():
        for tuningtype in ['cue_higher','match_higher','average']:
            if tuningtype+'_cue' not in plot_info and tuningtype+'_match' not in plot_info:
                continue

            save_name = save_dir+'mix_selectivity_Analysis_'+tuningtype+'_'+m_key+'_'
            if norm:
                save_name += 'normalzed'
            else:
                save_name += 'raw_1s_firerate'

            fig,ax = plt.subplots(figsize=(16,10))
            x_tuning = np.arange(len(plot_info[tuningtype+'_cue']))
            ax.plot(x_tuning, plot_info[tuningtype+'_cue'], color = colors[m_key],\
                label = m_key+' cue'+' avg_perf:%.2f'%(plot_info['perf']))
            ax.plot(x_tuning, plot_info[tuningtype+'_match'], color = colors[m_key],\
                label = m_key+' match'+' avg_perf:%.2f'%(plot_info['perf']),linestyle = '--')

            title = 'Rule:'+rule+' mix_selectivity '+tuningtype
            fig.suptitle(title)
            ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
            ax.set_ylabel('activity')

            for save_format in save_formats:
                plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

            plt.close()


def analyze_mix_selectivity_tuning(hp,log,model_dir,model_list,rule,
                        task_info_dict=None,H_dict=None,
                        norm = True, task_mode='test',
                        perf_type = 'train_perf',plot=True):

    model_list = model_list_parser(hp, log, rule, model_list,)
    
    # empty data
    if task_info_dict is None or H_dict is None:
        task_info_dict, H_dict = dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()

    plot_information = dict()

    for m_key in model_list.keys():
        perf = list()
        cue_higher_cue = list()
        cue_higher_match = list()
        match_higher_cue = list()
        match_higher_match =list()
        plot_information[m_key] = dict()

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]

            perf.append(task_info[perf_type])

            Match_trials = get_mon_trials(task_info['input_loc']['stim1'],task_info['input_loc']['stim2'],'match')
            H_match = slice_H(H,trial_range=Match_trials)

            tuning_curves_cue = analyze_tuning_curve_single_H(H,task_info,epoch='stim1',norm=norm)
            tuning_curves_match = analyze_tuning_curve_single_H(H_match,task_info,epoch='stim2',norm=norm)

            for neuron in range(hp['n_rnn']):
                tuning_cue = tuning_curves_cue[:,neuron]
                tuning_match = tuning_curves_match[:,neuron]

                max_index_cue = np.nanargmax(tuning_cue)

                #move only cue tuning's max location to center
                cue_tuning, match_tuning= max_central(max_index_cue,tuning_cue,tuning_match)

                # move both max location to center 
                #max_index_match = np.nanargmax(tuning_match)
                #cue_tuning = max_central(max_index_cue,tuning_cue)
                #match_tuning = max_central(max_index_match,tuning_match)

                max_cue_firerate = np.nanmax(cue_tuning)
                max_match_firerate = np.nanmax(match_tuning)

                if np.isnan(max_cue_firerate) or np.isnan(max_match_firerate):
                    continue

                if max_cue_firerate >= max_match_firerate:
                    cue_higher_cue.append(cue_tuning)
                    cue_higher_match.append(match_tuning)
                else:
                    match_higher_cue.append(cue_tuning)
                    match_higher_match.append(match_tuning)

        plot_information[m_key]['cue_higher_cue'] = np.nanmean(np.array(cue_higher_cue),axis=0)
        plot_information[m_key]['cue_higher_match'] = np.nanmean(np.array(cue_higher_match),axis=0)
        plot_information[m_key]['match_higher_cue'] = np.nanmean(np.array(match_higher_cue),axis=0)
        plot_information[m_key]['match_higher_match'] = np.nanmean(np.array(match_higher_match),axis=0)
        plot_information[m_key]['average_cue'] = np.nanmean(np.array(cue_higher_cue+match_higher_cue),axis=0)
        plot_information[m_key]['average_match'] = np.nanmean(np.array(cue_higher_match+match_higher_match),axis=0)
        plot_information[m_key]['perf'] = np.nanmean(perf)

    if not plot:
        return plot_information

    else:
        plot_mix_selectivity_tuning(model_dir,rule,norm,plot_information,save_formats=['pdf',])


def analyze_mix_selectivity_2ANOVA(hp,log,model_dir,model_list,rule,
                        task_info_dict=None,H_dict=None,
                        task_mode='test',perf_type = 'train_perf',):

    model_list = model_list_parser(hp, log, rule, model_list,)
    
    # empty data
    if task_info_dict is None or H_dict is None:
        task_info_dict, H_dict = dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/mix_selectivity_Analysis/'
    mkdir_p(save_dir)

    for m_key in model_list.keys():
        
        for model_index in model_list[m_key]:
            excel_content = list()
            excel_content.append(['Neuron index','Loc-df','Loc-F','Loc-PR(>F)','CorM-df','CorM-F','CorM-PR(>F)','Loc:CorM-df','Loc:CorM-F','Loc:CorM-PR(>F)'])

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]
            
            perf = task_info[perf_type]

            Match_trials = get_mon_trials(task_info['input_loc']['stim1'],task_info['input_loc']['stim2'],'match')

            cue_firerates = np.nanmean(slice_H(H,time_range=task_info['epochs']['stim1']),axis=0)
            match_firerates = np.nanmean(slice_H(H,time_range=task_info['epochs']['stim2'],trial_range=Match_trials),axis=0)

            for neuron in range(hp['n_rnn']):
                cue_neuron_fr = cue_firerates[:,neuron]
                match_neuron_fr = match_firerates[:,neuron]
                firerate = np.concatenate((cue_neuron_fr,match_neuron_fr))
                in_locs = np.concatenate((task_info['input_loc']['stim1'],task_info['input_loc']['stim2']))
                cue_or_match = ['cue' for i in range(len(cue_neuron_fr))]+['match' for i in range(len(match_neuron_fr))]
                df = pd.DataFrame({'Loc':in_locs,
                           'CueOrMatch':cue_or_match,
                           'Firerate':firerate,})
                try:
                    model = ols('Firerate ~ C(Loc) + C(CueOrMatch) + C(Loc):C(CueOrMatch)', data=df,missing='drop').fit()
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
            
            save_name = save_dir+'MNM_cue_match_mix_selectivity_2wayANOVA_trialnum'+str(model_index)+'_perf%.2f'%(perf)+'.xls'
            write_excel_xls(save_name, '2 way ANOVA', excel_content)