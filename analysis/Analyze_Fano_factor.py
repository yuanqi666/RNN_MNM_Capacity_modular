import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,model_list_parser,mkdir_p

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H

import matplotlib.pyplot as plt

def plot_Fano_factor(model_dir,dt,rule,epoch,neuron_type,norm,loc_type,plot_information,save_formats=['pdf','png','eps']):
    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Fano_factor_Analysis/'+epoch+'/'
    mkdir_p(save_dir)
    save_name = save_dir+'Fano_factor_Anelysis_'+neuron_type+'_neurons_'+loc_type+'_'
    if norm:
        save_name += 'normalzed'
    else:
        save_name += 'raw_1s_firerate'

    fig,ax = plt.subplots(figsize=(16,10))
    for m_key,plot_info in plot_information.items():
        if 'Fano_factor' not in plot_info:
            continue
        x_Fano = np.arange(len(plot_info['Fano_factor']))*(dt/1000)
        ax.plot(x_Fano, plot_info['Fano_factor'], color = colors[m_key],\
            label = m_key+' avg_n_num(int):%d'%(plot_info['neuron_number'])+' avg_perf:%.2f'%(plot_info['perf']))

    title = 'Rule:'+rule+' Epoch:'+epoch+' Fano factor Anelysis'
    fig.suptitle(title)
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    ax.set_ylabel('Fano factor')
    ax.set_xlabel('time/s')

    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

    plt.close()

def analyze_Fano_factor(hp,log,model_dir,model_list,rule,epoch,
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
        Fano_factor = list()
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

            perf.append(task_info[perf_type])
            neuron_number.append(len(neuron_info[neuron_type]))

            # normalize
            fix_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['fix1']),axis=0)
            if norm:
                H = H/fix_epoch_H-1

            for neuron in neuron_info[neuron_type]:
                if loc_type=='best_cue':
                    trials = task_info['input_loc'][epoch]==neuron[1]
                elif loc_type=='all_location':
                    trials = [True]*len(task_info['input_loc'][epoch])
                else:
                    raise ValueError("location selection type can only be 'best_cue' or 'all_location'")

                H_n = H[:,trials,neuron[0]]

                Fano_factor.append(np.nanvar(H_n,axis=1)/np.nanmean(H_n,axis=1)) # σ²/μ (trialwise variance/trialwise mean)


        plot_information[m_key]['Fano_factor'] = np.nanmean(np.array(Fano_factor),axis=0)
        plot_information[m_key]['perf'] = np.nanmean(perf)
        plot_information[m_key]['neuron_number'] = np.nanmean(neuron_number)

    if not plot:
        return plot_information
    else:
        plot_Fano_factor(model_dir,hp['dt'],rule,epoch,neuron_type,norm,loc_type,plot_information,save_formats=['pdf','png','eps'])   
