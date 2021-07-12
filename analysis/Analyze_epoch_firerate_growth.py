import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,model_list_parser,mkdir_p

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H
from .Analyze_tuning_curve import analyze_tuning_curve_single_H

import matplotlib.pyplot as plt


def plot_epoch_firerate_growth(model_dir,log,rule,epoch,neuron_type,norm,loc_type,plot_information,save_formats=['pdf','png','eps'],):
    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/epoch_firerate_growth/'+epoch+'/'
    mkdir_p(save_dir)
    save_name = save_dir+'epoch_firerate_growth_'+neuron_type+'_neurons_'+loc_type+'_'
    if norm:
        save_name += 'normalzed'
    else:
        save_name += 'raw_1s_firerate'

    fig,ax = plt.subplots(figsize=(16,10))
    for m_key,plot_info in plot_information.items():
        if 'epoch_firerate' not in plot_info:
            continue
        for data_point in plot_info['epoch_firerate']:
            ax.scatter(data_point[0],data_point[1], color = colors[m_key],marker = '+')
    ax.set_xlim(log['trials'][0],int(log['trials'][-1]*1.01))
    title = 'Rule:'+rule+' Epoch:'+epoch+' epoch firerate growth'
    fig.suptitle(title)
    ax.set_ylabel('fire rate')
    ax.set_xlabel('trials trained')

    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

    plt.close()

def analyze_epoch_firerate_growth(hp,log,model_dir,model_list,rule,epoch,
                task_info_dict=None,H_dict=None,neuron_info_dict=None,
                norm=False,loc_type='best_cue',task_mode='test',neuron_type='all',
                perf_type = 'train_perf',plot=True,):

    model_list = model_list_parser(hp, log, rule, model_list,)
    
    # empty data
    if task_info_dict is None or H_dict is None or neuron_info_dict is None:
        task_info_dict, H_dict, neuron_info_dict = dict(), dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()
            neuron_info_dict[m_key] = dict()

    plot_information = dict()

    for m_key in model_list.keys():
        perf = list()
        neuron_number = list()
        epoch_firerate_growth = list()
        plot_information[m_key] = dict()

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]

            if model_index not in neuron_info_dict[m_key]:
                neuron_info = compute_neuron_information_single_H(hp, H, task_info, epoch, norm=norm)
            else:
                neuron_info = neuron_info_dict[m_key][model_index]

            tuning_curves = analyze_tuning_curve_single_H(H,task_info,epoch,norm=norm)
            epoch_mean_fr = list()

            for neuron in neuron_info[neuron_type]:
                if loc_type=='best_cue':
                    locs = task_info['loc_set'][epoch]==neuron[1]
                elif loc_type=='all_location':
                    locs = [True]*len(task_info['loc_set'][epoch])
                else:
                    raise ValueError("location selection type can only be 'best_cue' or 'all_location'")

                epoch_mean_fr.append(np.nanmean(tuning_curves[locs,neuron[0]]))

            epoch_firerate_growth.append((model_index, np.nanmean(epoch_mean_fr)))

        plot_information[m_key]['epoch_firerate'] = epoch_firerate_growth
        plot_information[m_key]['perf'] = np.nanmean(perf)
        plot_information[m_key]['neuron_number'] = np.nanmean(neuron_number)

    if not plot:
        return plot_information
    else:
        plot_epoch_firerate_growth(model_dir,log,rule,epoch,neuron_type,norm,loc_type,plot_information,save_formats=['pdf','png','eps'])   
