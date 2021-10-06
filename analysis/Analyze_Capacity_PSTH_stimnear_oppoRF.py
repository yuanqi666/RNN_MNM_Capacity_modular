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

def plot_Capacity_PSTH_stimnear_oppoRF(model_dir,dt,rule,norm,plot_information,loc_type ='best_cond',save_formats=['pdf','png','eps'],):

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Stimnear_oppoRF/'
    mkdir_p(save_dir)

    fig,axes = plt.subplots(1,2,figsize=(18,10),sharey=True)
    for m_key,plot_info in plot_information.items():
        if not plot_info:
            continue
        
        color_cycle = plt.rcParams['axes.prop_cycle']()

        save_name = save_dir+m_key+'_'+loc_type+'_'
        if norm:
            save_name += 'normalzed'
        else:
            save_name += 'raw_1s_firerate'

        for plot_type in ['singlestim','nearRF','oppoRF']:
            color = next(color_cycle)['color']
        
            try:
                x_PSTH = np.arange(len(plot_info['PSTH_singlestim_all']))*(dt/1000)

                axes[0].plot(x_PSTH, plot_info['PSTH_'+plot_type+'_all'], color = color,label = plot_type)

                axes[1].plot(x_PSTH, plot_info['PSTH_'+plot_type+'_M'], color = color,label = plot_type+' match')
                    
                axes[1].plot(x_PSTH, plot_info['PSTH_'+plot_type+'_NM'], color = color,label = plot_type+' non-match',linestyle = '--')
            except:
                pass

        title = 'Rule:'+rule+' '+plot_type+' PSTH'+' avg_perf:%.2f'%(plot_info['perf'])
        fig.suptitle(title)
        for ax in axes:
            ax.legend()
            ax.set_ylabel('activity')
            ax.set_xlabel('time/s')

        for save_format in save_formats:
            plt.savefig(save_name+'.'+save_format,bbox_inches='tight')

        plt.close()



def analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,model_list,rule,
                norm = True,loc_type='best_cond',
                plot=True,):#perf_type = 'train_perf',):

    model_list = model_list_parser(hp, log, rule, model_list,)

    plot_information = dict()

    for m_key in model_list.keys():
        perf = list()

        PSTH_singlestim = list()
        PSTH_singlestim_M = list()
        PSTH_singlestim_NM = list()

        PSTH_multistim_near = list()
        PSTH_multistim_M_near = list()
        PSTH_multistim_NM_near = list()

        PSTH_multistim_oppo = list()
        PSTH_multistim_M_oppo = list()
        PSTH_multistim_NM_oppo = list()

        plot_information[m_key] = dict()

        if not len(model_list[m_key]):
            continue

        for model_index in model_list[m_key]:

            H_near, task_info_near = compute_single_H(hp, log, model_dir, rule, model_index, task_mode='stimnearRF')
            H_oppo, task_info_oppo = compute_single_H(hp, log, model_dir, rule, model_index, task_mode='stimoppositeRF')

            perf.append(task_info_near['train_perf']) #same for near and opposite

            #Get the MNM and single/multi stim information
            match_trials_near = get_mon_trials(task_info_near['input_loc']['stim1'][0],task_info_near['input_loc']['stim2'][0],'match')
            nonmatch_trials_near = get_mon_trials(task_info_near['input_loc']['stim1'][0],task_info_near['input_loc']['stim2'][0],'non-match')
            singlestim_trials_near = np.isnan(task_info_near['input_loc']['stim1'][1])
            multistim_trials_near = ~singlestim_trials_near

            match_trials_oppo = get_mon_trials(task_info_oppo['input_loc']['stim1'][0],task_info_oppo['input_loc']['stim2'][0],'match')
            nonmatch_trials_oppo = get_mon_trials(task_info_oppo['input_loc']['stim1'][0],task_info_oppo['input_loc']['stim2'][0],'non-match')
            singlestim_trials_oppo = np.isnan(task_info_oppo['input_loc']['stim1'][1])
            multistim_trials_oppo = ~singlestim_trials_oppo

            #Get single stim info
            H_singlestim_near = slice_H(H_near,trial_range=list(singlestim_trials_near))
            H_singlestim_M_near = slice_H(H_near,trial_range=list(singlestim_trials_near*match_trials_near))
            H_singlestim_NM_near = slice_H(H_near,trial_range=list(singlestim_trials_near*nonmatch_trials_near))

            singlestim_PSTH_all_near = analyze_PSTH_single_H_multistim(H_singlestim_near,task_info_near,norm=norm)
            singlestim_PSTH_M_near = analyze_PSTH_single_H_multistim(H_singlestim_M_near,task_info_near,norm=norm)
            singlestim_PSTH_NM_near = analyze_PSTH_single_H_multistim(H_singlestim_NM_near,task_info_near,norm=norm)

            H_singlestim_oppo = slice_H(H_oppo,trial_range=list(singlestim_trials_oppo))
            H_singlestim_M_oppo = slice_H(H_oppo,trial_range=list(singlestim_trials_oppo*match_trials_oppo))
            H_singlestim_NM_oppo = slice_H(H_oppo,trial_range=list(singlestim_trials_oppo*nonmatch_trials_oppo))

            singlestim_PSTH_all_oppo = analyze_PSTH_single_H_multistim(H_singlestim_oppo,task_info_oppo,norm=norm)
            singlestim_PSTH_M_oppo = analyze_PSTH_single_H_multistim(H_singlestim_M_oppo,task_info_oppo,norm=norm)
            singlestim_PSTH_NM_oppo = analyze_PSTH_single_H_multistim(H_singlestim_NM_oppo,task_info_oppo,norm=norm)

            singlestim_PSTH_all = (singlestim_PSTH_all_near+singlestim_PSTH_all_oppo)/2
            singlestim_PSTH_M   = (singlestim_PSTH_M_near+singlestim_PSTH_M_oppo)/2
            singlestim_PSTH_NM  = (singlestim_PSTH_NM_near+singlestim_PSTH_NM_oppo)/2

            #Get near mode info
            H_multistim_near = slice_H(H_near,trial_range=list(multistim_trials_near))
            H_multistim_M_near = slice_H(H_near,trial_range=list(multistim_trials_near*match_trials_near))
            H_multistim_NM_near = slice_H(H_near,trial_range=list(multistim_trials_near*nonmatch_trials_near))

            multistim_PSTH_all_near = analyze_PSTH_single_H_multistim(H_multistim_near,task_info_near,norm=norm)
            multistim_PSTH_M_near = analyze_PSTH_single_H_multistim(H_multistim_M_near,task_info_near,norm=norm)
            multistim_PSTH_NM_near = analyze_PSTH_single_H_multistim(H_multistim_NM_near,task_info_near,norm=norm)

            #Get oppo mode info
            H_multistim_oppo = slice_H(H_oppo,trial_range=list(multistim_trials_oppo))
            H_multistim_M_oppo = slice_H(H_oppo,trial_range=list(multistim_trials_oppo*match_trials_oppo))
            H_multistim_NM_oppo = slice_H(H_oppo,trial_range=list(multistim_trials_oppo*nonmatch_trials_oppo))

            multistim_PSTH_all_oppo = analyze_PSTH_single_H_multistim(H_multistim_oppo,task_info_oppo,norm=norm)
            multistim_PSTH_M_oppo = analyze_PSTH_single_H_multistim(H_multistim_M_oppo,task_info_oppo,norm=norm)
            multistim_PSTH_NM_oppo = analyze_PSTH_single_H_multistim(H_multistim_NM_oppo,task_info_oppo,norm=norm)

            #timeline is same for near and oppo
            singlestim_stim1_epoch_mean = np.nanmean(singlestim_PSTH_all[task_info_near['epochs']['stim1'][0]:task_info_near['epochs']['stim1'][1],:,:],axis=0)

            for neuron in range(hp['n_rnn']):
                RF = np.nanargmax(singlestim_stim1_epoch_mean[:,neuron])

                if loc_type == 'best_cond':
                    PSTH_singlestim.append(singlestim_PSTH_all[:,RF,neuron])
                    PSTH_singlestim_M.append(singlestim_PSTH_M[:,RF,neuron])
                    PSTH_singlestim_NM.append(singlestim_PSTH_NM[:,RF,neuron])

                    PSTH_multistim_near.append(multistim_PSTH_all_near[:,RF,neuron])
                    PSTH_multistim_M_near.append(multistim_PSTH_M_near[:,RF,neuron])
                    PSTH_multistim_NM_near.append(multistim_PSTH_NM_near[:,RF,neuron])

                    PSTH_multistim_oppo.append(multistim_PSTH_all_oppo[:,RF,neuron])
                    PSTH_multistim_M_oppo.append(multistim_PSTH_M_oppo[:,RF,neuron])
                    PSTH_multistim_NM_oppo.append(multistim_PSTH_NM_oppo[:,RF,neuron])
                
                elif loc_type == 'all_cond':
                    PSTH_singlestim.append(np.nanmean(singlestim_PSTH_all[:,:,neuron],axis=1))
                    PSTH_singlestim_M.append(np.nanmean(singlestim_PSTH_M[:,:,neuron],axis=1))
                    PSTH_singlestim_NM.append(np.nanmean(singlestim_PSTH_NM[:,:,neuron],axis=1))

                    PSTH_multistim_near.append(np.nanmean(multistim_PSTH_all_near[:,:,neuron],axis=1))
                    PSTH_multistim_M_near.append(np.nanmean(multistim_PSTH_M_near[:,:,neuron],axis=1))
                    PSTH_multistim_NM_near.append(np.nanmean(multistim_PSTH_NM_near[:,:,neuron],axis=1))

                    PSTH_multistim_oppo.append(np.nanmean(multistim_PSTH_all_oppo[:,:,neuron],axis=1))
                    PSTH_multistim_M_oppo.append(np.nanmean(multistim_PSTH_M_oppo[:,:,neuron],axis=1))
                    PSTH_multistim_NM_oppo.append(np.nanmean(multistim_PSTH_NM_oppo[:,:,neuron],axis=1))


        plot_information[m_key]['PSTH_singlestim_all'] = np.nanmean(np.array(PSTH_singlestim),axis=0)
        plot_information[m_key]['PSTH_singlestim_M'] = np.nanmean(np.array(PSTH_singlestim_M),axis=0)
        plot_information[m_key]['PSTH_singlestim_NM'] = np.nanmean(np.array(PSTH_singlestim_NM),axis=0)

        plot_information[m_key]['PSTH_nearRF_all'] = np.nanmean(np.array(PSTH_multistim_near),axis=0)
        plot_information[m_key]['PSTH_nearRF_M'] = np.nanmean(np.array(PSTH_multistim_M_near),axis=0)
        plot_information[m_key]['PSTH_nearRF_NM'] = np.nanmean(np.array(PSTH_multistim_NM_near),axis=0)

        plot_information[m_key]['PSTH_oppoRF_all'] = np.nanmean(np.array(PSTH_multistim_oppo),axis=0)
        plot_information[m_key]['PSTH_oppoRF_M'] = np.nanmean(np.array(PSTH_multistim_M_oppo),axis=0)
        plot_information[m_key]['PSTH_oppoRF_NM'] = np.nanmean(np.array(PSTH_multistim_NM_oppo),axis=0)

        plot_information[m_key]['perf'] = np.nanmean(perf)

    if not plot:
        return plot_information
    else:
        plot_Capacity_PSTH_stimnear_oppoRF(model_dir,hp['dt'],rule,norm,plot_information,loc_type=loc_type,save_formats=['pdf',])