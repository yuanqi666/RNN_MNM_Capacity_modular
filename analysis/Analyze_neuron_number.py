import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,model_list_parser,mkdir_p

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H

import matplotlib.pyplot as plt


def plot_neuron_number(model_dir,rule,epoch,norm,plot_information,save_formats=['pdf','png','eps'],skip_n_type=['all',]):

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/neuron_number_analysis/'+epoch+'/'
    mkdir_p(save_dir)
    save_name = save_dir+'neuron_number_analysis_'
    if norm:
        save_name += 'normalzed'
    else:
        save_name += 'raw_1s_firerate'

    fig,ax = plt.subplots(figsize=(16,10))
    for key, value in plot_information.items():
        if key not in skip_n_type and key != 'model_list':
            ax.plot(plot_information['model_list'], value, label = key)
    title = 'neuron number analysis'
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    fig.suptitle(title)
    ax.set_ylabel('neuron number')
    ax.set_xlabel('trials trained')

    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

    plt.close()

def analyze_neuron_number(hp,log,model_dir,model_list,rule,epoch,
                task_info_dict=None,H_dict=None,neuron_info_dict=None,
                norm=False,task_mode='test',
                plot=True,):

    model_list = model_list_parser(hp, log, rule, model_list,)
    
    # empty data
    if task_info_dict is None or H_dict is None or neuron_info_dict is None:
        task_info_dict, H_dict, neuron_info_dict = dict(), dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()
            neuron_info_dict[m_key] = dict()

    plot_information = dict()
    plot_information['model_list'] = list()

    neuron_numbers = dict()

    for m_key in model_list.keys():

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            if model_index in neuron_numbers:
                continue
            else:
                neuron_numbers[model_index] = dict()

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]

            if model_index not in neuron_info_dict[m_key]:
                neuron_info = compute_neuron_information_single_H(hp, H, task_info, epoch, norm=norm)
            else:
                neuron_info = neuron_info_dict[m_key][model_index]

            for neuron_type, neuron_list in neuron_info.items():
                neuron_numbers[model_index][neuron_type] = len(neuron_list)

    neuron_num_info = list()
    for model_index, n_num_info in neuron_numbers.items():
        neuron_num_info.append((model_index,n_num_info))     
    neuron_num_info = sorted(neuron_num_info, key=lambda x: x[0])

    for info in neuron_num_info:
        plot_information['model_list'].append(info[0])
        for key, value in info[1].items():
            if key not in plot_information:
                plot_information[key] = list()
            plot_information[key].append(value)

    if not plot:
        return plot_information
    else:
        plot_neuron_number(model_dir,rule,epoch,norm,plot_information,save_formats=['pdf','png','eps'])