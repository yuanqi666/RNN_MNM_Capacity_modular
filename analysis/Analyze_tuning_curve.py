import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,average_H_within_location,gaussian_curve_fit,max_central,model_list_parser,mkdir_p

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H

import matplotlib.pyplot as plt


def analyze_tuning_curve_single_H(H,task_info,epoch,norm=True):
    '''
    Inputs:
        H           : Neurons' fire rate 3D tensor (time,trials,neurons)
        task_info   : A dictionary that contains information about the rule/task, 
                      for example, input locations, epoch durations etc.
        epoch       : The epoch that analyzed in this function, string (epoch name)
    Output:
        tuning_curves: A 2D tensor (locations*neurons) containing the nuerons' tuning information
                      !! Not max centered (max fire rate location not shifted to the center) !!
                      !! May contain np.nan values !!
    '''
    epoch_H = slice_H(H, time_range=task_info['epochs'][epoch])
    if norm:
        fix_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['fix1']),axis=0)
        epoch_H = epoch_H/fix_epoch_H - 1
    tuning_curves = np.nanmean(average_H_within_location(epoch_H,task_info['input_loc'][epoch],task_info['loc_set'][epoch]),axis=0)

    return tuning_curves


def plot_tuning_curve(model_dir,rule,epoch,neuron_type,norm,plot_information,save_formats=['pdf','png','eps']):

    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Tuning_Analysis/'+epoch+'/'
    mkdir_p(save_dir)
    save_name = save_dir+'Tuning_Anelysis_'+neuron_type+'_neurons_'
    if norm:
        save_name += 'normalzed'
    else:
        save_name += 'raw_1s_firerate'

    fig,ax = plt.subplots(figsize=(16,10))
    for m_key,plot_info in plot_information.items():
        if 'tuning' not in plot_info:
            continue
        x_tuning = np.arange(len(plot_info['tuning']))
        if 'gaussian_y' in plot_info:
            ax.scatter(x_tuning, plot_info['tuning'], marker = '+',color = colors[m_key], s = 70 ,\
                label = m_key+' avg_n_num(int):%d'%(plot_info['neuron_number'])+' avg_perf:%.2f'%(plot_info['perf']))
            ax.plot(plot_info['gaussian_x'], plot_info['gaussian_y'], color=colors[m_key],\
                label = m_key+' curve_width:%.2f'%(plot_info['gaussian_width']),linestyle = '--')
        else:
            ax.plot(x_tuning, plot_info['tuning'], color = colors[m_key],\
                label = m_key+' avg_n_num(int):%d'%(plot_info['neuron_number'])+' avg_perf:%.2f'%(plot_info['perf']))

    title = 'Rule:'+rule+' Epoch:'+epoch+' Tuning curve'
    fig.suptitle(title)
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    ax.set_ylabel('activity')

    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

    plt.close()


def analyze_tuning_curve(hp,log,model_dir,model_list,rule,epoch,
                        task_info_dict=None,H_dict=None,neuron_info_dict=None,
                        norm = True, gaussian_fit = True,task_mode='test',neuron_type='significant',
                        perf_type = 'train_perf',plot=True):

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
        tuning_curves_max_central = list()
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

            tuning_curves = analyze_tuning_curve_single_H(H,task_info,epoch,norm=norm)

            for neuron in neuron_info[neuron_type]:
                tuning_curves_max_central.append(max_central(task_info['loc_set'][epoch].index(neuron[1]),tuning_curves[:,neuron[0]]))

        plot_information[m_key]['tuning'] = np.nanmean(np.array(tuning_curves_max_central),axis=0)
        plot_information[m_key]['perf'] = np.nanmean(perf)
        plot_information[m_key]['neuron_number'] = np.nanmean(neuron_number)

        if gaussian_fit:
            plot_information[m_key]['gaussian_x'], plot_information[m_key]['gaussian_y'], paras = gaussian_curve_fit(plot_information[m_key]['tuning'])
            plot_information[m_key]['gaussian_width'] = paras[2]*2

    if not plot:
        return plot_information

    else:
        plot_tuning_curve(model_dir,rule,epoch,neuron_type,norm,plot_information,save_formats=['pdf','png','eps'])