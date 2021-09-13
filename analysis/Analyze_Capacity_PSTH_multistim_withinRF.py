import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,average_H_within_location,model_list_parser,mkdir_p,get_mon_trials

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H

import matplotlib.pyplot as plt

def analyze_PSTH_single_H_multistim(H,task_info,epoch='stim1',norm=True):

    if norm:
        fix_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['fix1']),axis=0)
        H = H/fix_epoch_H - 1
    PSTHs = average_H_within_location(H,task_info['input_loc'][epoch][0],sorted(set(task_info['input_loc'][epoch][0])))

    return PSTHs

def plot_Capacity_PSTH_multistim_withinRF(model_dir,dt,rule,norm,plot_information,loc_type ='best_cond',save_formats=['pdf','png','eps'],):

    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Capacity_PSTH_multistim_withinRF/'
    mkdir_p(save_dir)

    for plot_type in ['M','NM','all']:
        save_name = save_dir+plot_type+'_'+loc_type+'_'
        if norm:
            save_name += 'normalzed'
        else:
            save_name += 'raw_1s_firerate'
        
        fig,ax = plt.subplots(figsize=(16,10))
        for m_key,plot_info in plot_information.items():
            try:
                x_PSTH = np.arange(len(plot_info['PSTH_singlestim_'+plot_type]))*(dt/1000)
                ax.plot(x_PSTH, plot_info['PSTH_singlestim_'+plot_type], color = colors[m_key],\
                    label = 'Single stim '+m_key+' avg_perf:%.2f'%(plot_info['perf']))
                ax.plot(x_PSTH, plot_info['PSTH_multistim_'+plot_type], color = colors[m_key],\
                    label = 'Multiple stims '+m_key+' avg_perf:%.2f'%(plot_info['perf']),linestyle = '--')
            except:
                pass

        title = 'Rule:'+rule+' '+plot_type+' PSTH'
        fig.suptitle(title)
        ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        ax.set_ylabel('activity')
        ax.set_xlabel('time/s')

        for save_format in save_formats:
            plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

        plt.close()



def analyze_Capacity_multistim_withinRF_PSTH(hp,log,model_dir,model_list,rule,
                task_info_dict=None,H_dict=None,
                norm = True,task_mode='multistim_withinRF',loc_type='best_cond',
                perf_type = 'train_perf',plot=True,):

    model_list = model_list_parser(hp, log, rule, model_list,)
    
    # empty data
    if task_info_dict is None or H_dict is None:
        task_info_dict, H_dict= dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()

    plot_information = dict()

    for m_key in model_list.keys():
        perf = list()

        PSTH_singlestim = list()
        PSTH_singlestim_M = list()
        PSTH_singlestim_NM = list()

        PSTH_multistim = list()
        PSTH_multistim_M = list()
        PSTH_multistim_NM = list()

        plot_information[m_key] = dict()

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]

            perf.append(task_info[perf_type])

            match_trials = get_mon_trials(task_info['input_loc']['stim1'][0],task_info['input_loc']['stim2'][0],'match')
            nonmatch_trials = get_mon_trials(task_info['input_loc']['stim1'][0],task_info['input_loc']['stim2'][0],'non-match')
            singlestim_trials = np.isnan(task_info['input_loc']['stim1'][1])
            multistim_trials = ~singlestim_trials

            H_singlestim = slice_H(H,trial_range=list(singlestim_trials))
            H_singlestim_M = slice_H(H,trial_range=list(singlestim_trials*match_trials))
            H_singlestim_NM = slice_H(H,trial_range=list(singlestim_trials*nonmatch_trials))

            singlestim_PSTH_all = analyze_PSTH_single_H_multistim(H_singlestim,task_info,norm=norm)
            singlestim_PSTH_M = analyze_PSTH_single_H_multistim(H_singlestim_M,task_info,norm=norm)
            singlestim_PSTH_NM = analyze_PSTH_single_H_multistim(H_singlestim_NM,task_info,norm=norm)

            H_multistim = slice_H(H,trial_range=list(multistim_trials))
            H_multistim_M = slice_H(H,trial_range=list(multistim_trials*match_trials))
            H_multistim_NM = slice_H(H,trial_range=list(multistim_trials*nonmatch_trials))

            multistim_PSTH_all = analyze_PSTH_single_H_multistim(H_multistim,task_info,norm=norm)
            multistim_PSTH_M = analyze_PSTH_single_H_multistim(H_multistim_M,task_info,norm=norm)
            multistim_PSTH_NM = analyze_PSTH_single_H_multistim(H_multistim_NM,task_info,norm=norm)

            singlestim_stim1_epoch_mean = np.nanmean(singlestim_PSTH_all[task_info['epochs']['stim1'][0]:task_info['epochs']['stim1'][1],:,:],axis=0)

            for neuron in range(hp['n_rnn']):
                RF = np.nanargmax(singlestim_stim1_epoch_mean[:,neuron])

                if loc_type == 'best_cond':
                    PSTH_singlestim.append(singlestim_PSTH_all[:,RF,neuron])
                    PSTH_singlestim_M.append(singlestim_PSTH_M[:,RF,neuron])
                    PSTH_singlestim_NM.append(singlestim_PSTH_NM[:,RF,neuron])

                    PSTH_multistim.append(multistim_PSTH_all[:,RF,neuron])
                    PSTH_multistim_M.append(multistim_PSTH_M[:,RF,neuron])
                    PSTH_multistim_NM.append(multistim_PSTH_NM[:,RF,neuron])
                elif loc_type == 'all_cond':
                    PSTH_singlestim.append(np.nanmean(singlestim_PSTH_all[:,:,neuron],axis=1))
                    PSTH_singlestim_M.append(np.nanmean(singlestim_PSTH_M[:,:,neuron],axis=1))
                    PSTH_singlestim_NM.append(np.nanmean(singlestim_PSTH_NM[:,:,neuron],axis=1))

                    PSTH_multistim.append(np.nanmean(multistim_PSTH_all[:,:,neuron],axis=1))
                    PSTH_multistim_M.append(np.nanmean(multistim_PSTH_M[:,:,neuron],axis=1))
                    PSTH_multistim_NM.append(np.nanmean(multistim_PSTH_NM[:,:,neuron],axis=1))


        plot_information[m_key]['PSTH_singlestim_all'] = np.nanmean(np.array(PSTH_singlestim),axis=0)
        plot_information[m_key]['PSTH_singlestim_M'] = np.nanmean(np.array(PSTH_singlestim_M),axis=0)
        plot_information[m_key]['PSTH_singlestim_NM'] = np.nanmean(np.array(PSTH_singlestim_NM),axis=0)

        plot_information[m_key]['PSTH_multistim_all'] = np.nanmean(np.array(PSTH_multistim),axis=0)
        plot_information[m_key]['PSTH_multistim_M'] = np.nanmean(np.array(PSTH_multistim_M),axis=0)
        plot_information[m_key]['PSTH_multistim_NM'] = np.nanmean(np.array(PSTH_multistim_NM),axis=0)
        plot_information[m_key]['perf'] = np.nanmean(perf)

    if not plot:
        return plot_information
    else:
        plot_Capacity_PSTH_multistim_withinRF(model_dir,hp['dt'],rule,norm,plot_information,loc_type=loc_type,save_formats=['pdf',])