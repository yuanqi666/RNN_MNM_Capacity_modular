import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,model_list_parser,mkdir_p,z_score_norm

from .Compute_H import compute_single_H

import matplotlib.pyplot as plt

import pandas as pd


def plot_spike_count_correlation(model_dir,rule,epoch,z_norm,plot_information,appendix_info='',save_formats=['pdf','png','eps']):

    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/spike_count_correlation_Analysis/'+epoch+'/'
    mkdir_p(save_dir)

    for plottype in ['averaged_within_model','not_averaged_within_model']:
        save_name = save_dir+'mix_spike_count_correlation_'+plottype
        if z_norm:
            save_name += '_Znormalzed'
        else:
            save_name += '_not_Znormalzed'

        save_name += appendix_info
        
        fig,axes = plt.subplots(1,3,figsize=(18,6))
        max_x_temp = np.nanmax([np.nanmax(plot_information[mk][plottype]) for mk in plot_information.keys()])
        max_x = max_x_temp*1.1
        min_x = np.nanmin([np.nanmin(plot_information[mk][plottype]) for mk in plot_information.keys()])-0.1*max_x_temp
        if np.isnan(max_x) or np.isnan(min_x):
            raise ValueError("NaN max_x/min_x encountered in determine the x range")

        bins = np.arange(min_x,max_x,0.05*max_x_temp)
        for i,m_key in enumerate(plot_information.keys()):
            axes[i].hist(plot_information[m_key][plottype], histtype="stepfilled",alpha=0.6,color=colors[m_key],bins=bins)
            axes[i].set_title(m_key+" perf %.2f"%(plot_information[m_key]['perf']))
            fig.suptitle("Neuron Fire rate correlation coefficient of "+rule+" "+epoch+" "+plottype)
            axes[i].set_xlim(min_x,max_x)

        for save_format in save_formats:
            plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

        plt.close()


def analyze_spike_count_correlation(hp,log,model_dir,model_list,rule,epoch,
                        task_info_dict=None,H_dict=None,
                        task_mode='test',z_norm = True,
                        perf_type = 'train_perf',plot=True, appendix_info = ''):

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
        plot_information[m_key] = dict()
        plot_information[m_key]['averaged_within_model'] = list()

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]

            perf.append(task_info[perf_type])

            epoch_mean_H = np.nanmean(slice_H(H,time_range=task_info['epochs'][epoch]),axis=0)

            if z_norm:
                for n in range(hp['n_rnn']):
                    epoch_mean_H[:,n] = z_score_norm(epoch_mean_H[:,n])

            df = pd.DataFrame(epoch_mean_H)

            corrs = np.array(df.corr())
            corrs = corrs[~np.eye(corrs.shape[0],dtype=bool)]

            plot_information[m_key]['averaged_within_model'].append(np.nanmean(corrs))
            if 'not_averaged_within_model' not in plot_information[m_key]:
                plot_information[m_key]['not_averaged_within_model'] = corrs
            else:
                plot_information[m_key]['not_averaged_within_model'] = np.concatenate((plot_information[m_key]['not_averaged_within_model'],corrs))

        plot_information[m_key]['averaged_within_model'] = np.array(plot_information[m_key]['averaged_within_model'])
        plot_information[m_key]['perf'] = np.nanmean(perf)

    if not plot:
        return plot_information

    else:
        plot_spike_count_correlation(model_dir,rule,epoch,z_norm,plot_information,appendix_info,save_formats=['pdf','png','eps'])