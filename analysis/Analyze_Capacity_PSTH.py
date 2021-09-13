import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,average_H_within_location,model_list_parser,mkdir_p,get_mon_trials

from .Compute_H import compute_single_H
from .Compute_neuron_information import compute_neuron_information_single_H
from .Analyze_PSTH import analyze_PSTH_single_H

import matplotlib.pyplot as plt

def plot_Capacity_PSTH(model_dir,dt,rule,epoch,norm,plot_information,save_formats=['pdf','png','eps']):

    colors = {'mature':'red','mid':'blue','early':'green'}

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Capacity_PSTH/'+epoch+'/'
    #save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+epoch+'/'
    mkdir_p(save_dir)

    for plot_type in ['bestcue1cond','worstcue1cond']:
        save_name = save_dir+plot_type+'_'
        if norm:
            save_name += 'normalzed'
        else:
            save_name += 'raw_1s_firerate'
        
        fig,ax = plt.subplots(figsize=(16,10))
        for m_key,plot_info in plot_information.items():
            if 'PSTH_'+plot_type+'_M' not in plot_info:
                continue
            x_PSTH = np.arange(len(plot_info['PSTH_'+plot_type+'_M']))*(dt/1000)
            ax.plot(x_PSTH, plot_info['PSTH_'+plot_type+'_M'], color = colors[m_key],\
                label = 'Match '+m_key+' avg_perf:%.2f'%(plot_info['perf']))
            ax.plot(x_PSTH, plot_info['PSTH_'+plot_type+'_NM'], color = colors[m_key],\
                label = 'Non-match '+m_key+' avg_perf:%.2f'%(plot_info['perf']),linestyle = '--')

        title = 'Rule:'+rule+' Epoch:'+epoch+' '+plot_type+' PSTH'
        fig.suptitle(title)
        ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        ax.set_ylabel('activity')
        ax.set_xlabel('time/s')

        for save_format in save_formats:
            plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

        plt.close()



def analyze_Capacity_PSTH(hp,log,model_dir,model_list,rule,epoch,
                task_info_dict=None,H_dict=None,
                norm = True,task_mode='test',#loc_type='best_cond',
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

        PSTH_bestcue1cond_M = list()
        PSTH_worstcue1cond_M = list()
        PSTH_allcue1cond_M = list()

        PSTH_bestcue1cond_NM = list()
        PSTH_worstcue1cond_NM = list()
        PSTH_allcue1cond_NM = list()

        plot_information[m_key] = dict()

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]

            perf.append(task_info[perf_type])

            match_trials = get_mon_trials(task_info['input_loc']['stim1'],task_info['input_loc']['stim2'],'match')
            H_match = slice_H(H,trial_range=match_trials)
            nonmatch_trials = get_mon_trials(task_info['input_loc']['stim1'],task_info['input_loc']['stim2'],'non-match')
            H_nonmatch = slice_H(H,trial_range=nonmatch_trials)

            
            match_PSTH_all = analyze_PSTH_single_H(H_match,task_info,epoch,norm=norm)
            nonmatch_PSTH_all = analyze_PSTH_single_H(H_nonmatch,task_info,epoch,norm=norm)

            PSTH_stim1_all = analyze_PSTH_single_H(H,task_info,'stim1',norm=norm)
            stim1_epoch_mean = np.nanmean(PSTH_stim1_all[task_info['epochs']['stim1'][0]:task_info['epochs']['stim1'][1],:,:],axis=0)

            for neuron in range(hp['n_rnn']):
                # if loc_type=='best_cue':
                #     PSTH.append(PSTH_all[:,task_info['loc_set'][epoch].index(neuron[1]),neuron[0]])
                # elif loc_type=='all_location':
                #     PSTH.append(np.nanmean(PSTH_all[:,:,neuron[0]],axis=1))
                # else:
                #     raise ValueError("location selection type can only be 'best_cue' or 'all_location'")
                PSTH_bestcue1cond_M.append(match_PSTH_all[:,np.nanargmax(stim1_epoch_mean[:,neuron]),neuron])
                PSTH_bestcue1cond_NM.append(nonmatch_PSTH_all[:,np.nanargmax(stim1_epoch_mean[:,neuron]),neuron])
                PSTH_worstcue1cond_M.append(match_PSTH_all[:,np.nanargmin(stim1_epoch_mean[:,neuron]),neuron])
                PSTH_worstcue1cond_NM.append(nonmatch_PSTH_all[:,np.nanargmin(stim1_epoch_mean[:,neuron]),neuron])

                PSTH_allcue1cond_M.append(np.nanmean(match_PSTH_all[:,:,neuron],axis=1))
                PSTH_allcue1cond_NM.append(np.nanmean(nonmatch_PSTH_all[:,:,neuron],axis=1))


        plot_information[m_key]['PSTH_bestcue1cond_M'] = np.nanmean(np.array(PSTH_bestcue1cond_M),axis=0)
        plot_information[m_key]['PSTH_bestcue1cond_NM'] = np.nanmean(np.array(PSTH_bestcue1cond_NM),axis=0)
        plot_information[m_key]['PSTH_worstcue1cond_M'] = np.nanmean(np.array(PSTH_worstcue1cond_M),axis=0)
        plot_information[m_key]['PSTH_worstcue1cond_NM'] = np.nanmean(np.array(PSTH_worstcue1cond_NM),axis=0)

        plot_information[m_key]['PSTH_allcue1cond_M'] = np.nanmean(np.array(PSTH_allcue1cond_M),axis=0)
        plot_information[m_key]['PSTH_allcue1cond_NM'] = np.nanmean(np.array(PSTH_allcue1cond_NM),axis=0)

        plot_information[m_key]['perf'] = np.nanmean(perf)

    if not plot:
        return plot_information
    else:
        plot_Capacity_PSTH(model_dir,hp['dt'],rule,epoch,norm,plot_information,save_formats=['pdf','png','eps'])