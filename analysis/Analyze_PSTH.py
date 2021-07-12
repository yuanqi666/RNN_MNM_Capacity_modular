import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,average_H_within_location,model_list_parser,mkdir_p

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H

import matplotlib.pyplot as plt

def analyze_PSTH_single_H(H,task_info,epoch,norm=True):

    if norm:
        fix_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['fix1']),axis=0)
        H = H/fix_epoch_H - 1
    PSTHs = average_H_within_location(H,task_info['input_loc'][epoch],task_info['loc_set'][epoch])

    return PSTHs


def plot_PSTH(model_dir,dt,rule,epoch,neuron_type,norm,loc_type,plot_information,save_formats=['pdf','png','eps']):

    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/PSTH_Analysis/'+epoch+'/'
    mkdir_p(save_dir)
    save_name = save_dir+'PSTH_Anelysis_'+neuron_type+'_neurons_'+loc_type+'_'
    if norm:
        save_name += 'normalzed'
    else:
        save_name += 'raw_1s_firerate'

    fig,ax = plt.subplots(figsize=(16,10))
    for m_key,plot_info in plot_information.items():
        if 'PSTH' not in plot_info:
            continue
        x_PSTH = np.arange(len(plot_info['PSTH']))*(dt/1000)
        ax.plot(x_PSTH, plot_info['PSTH'], color = colors[m_key],\
            label = m_key+' avg_n_num(int):%d'%(plot_info['neuron_number'])+' avg_perf:%.2f'%(plot_info['perf']))
        if 'PSTH_oppo' in plot_info:
            ax.plot(x_PSTH, plot_info['PSTH_oppo'], color = colors[m_key],label = m_key+' opposite location',linestyle = '--')

    title = 'Rule:'+rule+' Epoch:'+epoch+' PSTH'
    fig.suptitle(title)
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    ax.set_ylabel('activity')
    ax.set_xlabel('time/s')

    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

    plt.close()



def analyze_PSTH(hp,log,model_dir,model_list,rule,epoch,
                task_info_dict=None,H_dict=None,neuron_info_dict=None,
                norm = True, loc_type='best_cue',task_mode='test',neuron_type='significant',
                perf_type = 'train_perf',plot=True,opposite_loc=False,):

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
        PSTH = list()
        if opposite_loc:
            assert (loc_type == 'best_cue'),'all_location PSTH analysis does not support opposite location PSTH analysis'
            PSTH_oppo = list()
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

            PSTH_all = analyze_PSTH_single_H(H,task_info,epoch,norm=norm)

            for neuron in neuron_info[neuron_type]:
                if loc_type=='best_cue':
                    PSTH.append(PSTH_all[:,task_info['loc_set'][epoch].index(neuron[1]),neuron[0]])
                    if opposite_loc:
                        assert (len(task_info['loc_set'][epoch])%2 == 0), "Task with odd stimulus location numbers does not support opposite location PSTH analysis"
                        op_loc = (neuron[1]+np.pi)%(2*np.pi)
                        PSTH_oppo.append(PSTH_all[:,task_info['loc_set'][epoch].index(op_loc),neuron[0]])
                elif loc_type=='all_location':
                    PSTH.append(np.nanmean(PSTH_all[:,:,neuron[0]],axis=1))
                else:
                    raise ValueError("location selection type can only be 'best_cue' or 'all_location'")

        plot_information[m_key]['PSTH'] = np.nanmean(np.array(PSTH),axis=0)
        plot_information[m_key]['perf'] = np.nanmean(perf)
        plot_information[m_key]['neuron_number'] = np.nanmean(neuron_number)
        if opposite_loc:
            plot_information[m_key]['PSTH_oppo'] = np.nanmean(np.array(PSTH_oppo),axis=0)

    if not plot:
        return plot_information
    else:
        plot_PSTH(model_dir,hp['dt'],rule,epoch,neuron_type,norm,loc_type,plot_information,save_formats=['pdf','png','eps'])