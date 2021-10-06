# basic packages #
import os
import numpy as np

import sys
sys.path.append('.')
from utils.tools import mkdir_p, smooth, model_list_parser,get_mon_trials, average_H_within_location, slice_H

# plot #
from matplotlib import pyplot as plt

# dPCA #
from dPCA import dPCA

from .Compute_H import compute_single_H


def dPCA_analysis(hp,log,model_dir,model_list,rule,epoch='stim1',
                        task_info_dict=None,H_dict=None,
                        appoint_loc_analysis=False,appoint_locs=[0,np.pi],
                        invert_tcomp_y=False,invert_scomp_y=False,invert_stcomp_y=False,
                        tcomp_ylim=(None,None),scomp_ylim=(None,None),stcomp_ylim=(None,None),
                        task_mode='test',perf_type = 'train_perf',save_formats=['pdf','png','eps']):

    model_list = model_list_parser(hp, log, rule, model_list,)

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/dPCA_Analysis/'+epoch+'/'
    mkdir_p(save_dir)
    
    # empty data
    if task_info_dict is None or H_dict is None:
        task_info_dict, H_dict = dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()

    for m_key in model_list.keys():
        if not len(model_list[m_key]):
            continue
        
        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]
            
            perf = task_info[perf_type]

            if appoint_loc_analysis:
                locs = appoint_locs
                H = H[:,[lc in appoint_locs for lc in task_info['input_loc'][epoch]],:]
            else:
                locs = task_info['loc_set'][epoch]

            trial_per_loc = int(len(task_info['input_loc'][epoch])/len(task_info['loc_set'][epoch]))
            loc_num = len(locs)
            time_len = np.size(H,0)
            neuron_num = np.size(H,2)
            reform_H = np.zeros((trial_per_loc,neuron_num,loc_num,time_len))

            for i in range(trial_per_loc):
                for n in range(neuron_num):
                    for lo in range(loc_num):
                        for t in range(time_len):
                            reform_H[i,n,lo,t] = H[t,lo*trial_per_loc+i,n]

            # trial-average data
            R = np.mean(reform_H,0)

            # center data
            R -= np.mean(R.reshape((neuron_num,-1)),1)[:,None,None]

            dpca = dPCA.dPCA(labels='st',regularizer='auto')
            dpca.protect = ['t']

            Z = dpca.fit_transform(R,reform_H)

            time = np.arange(time_len)

            fig,axes = plt.subplots(1,3,figsize=(16,7))

            for loc in range(loc_num):
                axes[0].plot(time*hp['dt']/1000,Z['t'][0,loc],label='loc:%.2fpi'%(locs[loc]/np.pi))

            axes[0].set_title('1st time component')
            axes[0].set_ylim(tcomp_ylim[0],tcomp_ylim[1])
            if invert_tcomp_y:
                axes[0].invert_yaxis()
            axes[0].legend()
    
            for loc in range(loc_num):
                axes[1].plot(time*hp['dt']/1000,Z['s'][0,loc],label='loc:%.2fpi'%(locs[loc]/np.pi))
    
            axes[1].set_title('1st stimulus component')
            axes[1].set_ylim(scomp_ylim[0],scomp_ylim[1])
            if invert_scomp_y:
                axes[1].invert_yaxis()
            axes[1].legend()

            for loc in range(loc_num):
                axes[2].plot(time*hp['dt']/1000,Z['st'][0,loc],label='loc:%.2fpi'%(locs[loc]/np.pi))
    
            axes[2].set_title('1st mixing component')
            axes[2].set_ylim(stcomp_ylim[0],stcomp_ylim[1])
            if invert_stcomp_y:
                axes[2].invert_yaxis()
            axes[2].legend()
            #plt.show()

            for i in range(3):
                axes[i].set_xlabel("time/s")
    
            save_name = save_dir+'Neuron_dPCA_'+m_key+'_trialnum_'+str(model_index)+'_perf%.2f'%(perf)
            if appoint_loc_analysis:
                save_name = save_name+'_locs_'+str(appoint_locs)
            for save_format in save_formats:
                plt.savefig(save_name+'.'+save_format,bbox_inches='tight')
            plt.close()

def Capacity_dPCA_analysis(hp,log,model_dir,model_index,rules,
                        task_mode='test',perf_type = 'train_perf',save_formats=['pdf','png']):


    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/Capacity_dPCA_Analysis/'
    mkdir_p(save_dir)

    perfs = dict()
    loc_num_list = list()
    neuron_num_list = list()
    time_len_list = list()

    PSTH_match_dict = dict()
    PSTH_nonmatch_dict = dict()

    for rule in rules:
        H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            
        perfs[rule] = task_info[perf_type]

        if task_mode == 'test' or task_mode == 'move_one_loc':
            stim1_input_loc = task_info['input_loc']['stim1']
            stim2_input_loc = task_info['input_loc']['stim2']
        else:
            stim1_input_loc = task_info['input_loc']['stim1'][0]
            stim2_input_loc = task_info['input_loc']['stim2'][0]

        stim1_locset = sorted(set(stim1_input_loc))

        time_len_list.append(np.size(H,0))
        neuron_num_list.append(np.size(H,2))
        loc_num_list.append(len(stim1_locset))

        match_trials = get_mon_trials(stim1_input_loc,stim2_input_loc,'match')
        H_match = slice_H(H,trial_range=match_trials)
        PSTH_match_dict[rule] = average_H_within_location(H_match,stim1_input_loc,stim1_locset)

        nonmatch_trials = get_mon_trials(stim1_input_loc,stim2_input_loc,'non-match')
        H_nonmatch = slice_H(H,trial_range=nonmatch_trials)
        PSTH_nonmatch_dict[rule] = average_H_within_location(H_nonmatch,stim1_input_loc,stim1_locset)

    time_max = max(time_len_list)
    loc_num_max = max(loc_num_list)
    neuron_num_max = max(neuron_num_list)
    rules_match_H = np.full((loc_num_max,time_max,len(rules),neuron_num_max), np.nan)
    rules_nonmatch_H = np.full((loc_num_max,time_max,len(rules),neuron_num_max), np.nan)

    for i_rule, rule in enumerate(rules):
        rules_match_H[:loc_num_list[i_rule],:time_len_list[i_rule],i_rule,:neuron_num_list[i_rule]]\
            = PSTH_match_dict[rule].transpose((1,0,2))
        rules_nonmatch_H[:loc_num_list[i_rule],:time_len_list[i_rule],i_rule,:neuron_num_list[i_rule]]\
            = PSTH_nonmatch_dict[rule].transpose((1,0,2))

    reform_H = np.array([rules_match_H,rules_nonmatch_H])

    trialR = reform_H.transpose((1,4,2,3,0)) #[d,nsample,t,s,n] -> [nsample,n,t,s,d]

    # trial-average data
    R = np.nanmean(trialR,0)

    # center data
    R -= np.nanmean(R.reshape((neuron_num_max,-1)),1)[:,None,None,None]

    #dpca = dPCA.dPCA(labels='tsd',regularizer='auto')
    dpca = dPCA.dPCA(labels='tsd',regularizer=None)
    #dpca.protect = ['t']

    #Z = dpca.fit_transform(R)
    Z = dpca.fit_transform(R,trialR)

    time = np.arange(time_max)

    fig,axes = plt.subplots(2,3,figsize=(20,14),)#sharey=True)

    color_cycle = plt.rcParams['axes.prop_cycle']()
    for i_rule, rule in enumerate(rules):
        color = next(color_cycle)['color']
        axes[0][0].plot(time*hp['dt']/1000,Z['t'][0,:,i_rule,0],label='rule:'+rule+' match '+'perf:%.2f'%(perfs[rule]),color=color)
        axes[0][0].plot(time*hp['dt']/1000,Z['t'][0,:,i_rule,1],label='rule:'+rule+' nonmatch '+'perf:%.2f'%(perfs[rule]),linestyle='--',color=color)

        axes[0][1].plot(time*hp['dt']/1000,Z['s'][0,:,i_rule,0],color=color)
        axes[0][1].plot(time*hp['dt']/1000,Z['s'][0,:,i_rule,1],linestyle='--',color=color)
    
        axes[0][2].plot(time*hp['dt']/1000,Z['d'][0,:,i_rule,0],color=color)
        axes[0][2].plot(time*hp['dt']/1000,Z['d'][0,:,i_rule,1],linestyle='--',color=color)
    
        axes[1][0].plot(time*hp['dt']/1000,Z['ts'][0,:,i_rule,0],color=color)
        axes[1][0].plot(time*hp['dt']/1000,Z['ts'][0,:,i_rule,1],linestyle='--',color=color)
    
        axes[1][1].plot(time*hp['dt']/1000,Z['sd'][0,:,i_rule,0],color=color)
        axes[1][1].plot(time*hp['dt']/1000,Z['sd'][0,:,i_rule,1],linestyle='--',color=color)
    
    axes[0][0].set_title('1st time component')
    axes[0][1].set_title('1st stimulus component')
    axes[0][2].set_title('1st decision component')
    axes[1][0].set_title('1st time-stimulus mixing component')
    axes[1][1].set_title('1st stimulus-decision mixing component')

    fig.text(0.5, 0.05, 'time/s', ha='center')
    fig.legend(loc='lower right',)
    
    save_name = save_dir+'Neuron_dPCA_trialnum_'+str(model_index)
    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')
    plt.close()


# def Capacity_dPCA_analysis(hp,log,model_dir,model_list,rule,#epoch='stim1',
#                         task_info_dict=None,H_dict=None,
#                         #appoint_locs=[0,np.pi],appoint_loc_analysis=False,
#                         #invert_tcomp_y=False,invert_scomp_y=False,invert_stcomp_y=False,
#                         #tcomp_ylim=(None,None),scomp_ylim=(None,None),stcomp_ylim=(None,None),
#                         task_mode='test',perf_type = 'train_perf',save_formats=['pdf','png','eps']):

#     model_list = model_list_parser(hp, log, rule, model_list,)

#     save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Capacity_dPCA_Analysis/'#+epoch+'/'
#     mkdir_p(save_dir)
    
#     # empty data
#     if task_info_dict is None or H_dict is None:
#         task_info_dict, H_dict = dict(), dict()
#         for m_key in model_list.keys():
#             task_info_dict[m_key] = dict()
#             H_dict[m_key] = dict()

#     for m_key in model_list.keys():
#         if not len(model_list[m_key]):
#             continue
        
#         for model_index in model_list[m_key]:

#             if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
#                 H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
#             else:
#                 H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]
            
#             perf = task_info[perf_type]

#             if task_mode == 'test' or task_mode == 'move_one_loc':
#                 stim1_input_loc = task_info['input_loc']['stim1']
#                 stim2_input_loc = task_info['input_loc']['stim2']
#             else:
#                 stim1_input_loc = task_info['input_loc']['stim1'][0]
#                 stim2_input_loc = task_info['input_loc']['stim2'][0]

#             stim1_locset = sorted(set(stim1_input_loc))

#             # if norm:
#             #     fix_epoch_H = np.nanmean(slice_H(H,time_range=task_info['epochs']['fix1']),axis=0)
#             #     H = H/fix_epoch_H - 1

#             time_len = np.size(H,0)
#             neuron_num = np.size(H,2)
#             loc_num = len(stim1_locset)

#             match_trials = get_mon_trials(stim1_input_loc,stim2_input_loc,'match')
#             H_match = slice_H(H,trial_range=match_trials)
#             PSTH_match = average_H_within_location(H_match,stim1_input_loc,stim1_locset)

#             # trialR_match = list()
#             # for i in stim1_locset:
#             #     trialR_match.append(H_match[:,stim1_input_loc==i,:])
#             # trialR_match = np.array(trialR_match)

#             nonmatch_trials = get_mon_trials(stim1_input_loc,stim2_input_loc,'non-match')
#             H_nonmatch = slice_H(H,trial_range=nonmatch_trials)
#             PSTH_nonmatch = average_H_within_location(H_nonmatch,stim1_input_loc,stim1_locset)

#             # trialR_nonmatch = list()
#             # for i in stim1_locset:
#             #     trialR_nonmatch.append(H_nonmatch[:,stim1_input_loc==i,:])
#             # trialR_nonmatch = np.array(trialR_nonmatch)

#             reform_H = np.array([PSTH_match,PSTH_nonmatch]) #[d,t,s,n]
#             #trialR = np.array([trialR_match,trialR_nonmatch]) #[d,s,t,n_trial,n]

#             #trialR = trialR.transpose((3,4,2,1,0)) #[d,s,t,n_trial,n] -> [n_trial,n,t,s,d]
#             R = reform_H.transpose((3,1,2,0)) #[d,t,s,n] -> [n,t,s,d]
#             #R = np.nanmean(trialR,0)

#             # center data
#             R -= np.mean(R.reshape((neuron_num,-1)),1)[:,None,None,None]

#             #dpca = dPCA.dPCA(labels='std',regularizer='auto')
#             dpca = dPCA.dPCA(labels='tsd',regularizer=None)
#             #dpca.protect = ['td']

#             Z = dpca.fit_transform(R)
#             #Z = dpca.fit_transform(R,trialR)

#             time = np.arange(time_len)

#             fig,axes = plt.subplots(2,3,figsize=(20,14),)#sharey=True)

#             for loc in range(loc_num):
#                 axes[0][0].plot(time*hp['dt']/1000,Z['t'][0,:,loc,:],)

#             axes[0][0].set_title('1st time component')
#             # axes[0][0].set_ylim(tcomp_ylim[0],tcomp_ylim[1])
#             # if invert_tcomp_y:
#             #     axes[0][0].invert_yaxis()
#             # axes[0][0].legend()
    
#             for loc in range(loc_num):
#                 axes[0][1].plot(time*hp['dt']/1000,Z['s'][0,:,loc,:],)
    
#             axes[0][1].set_title('1st stimulus component')
#             # axes[0][1].set_ylim(scomp_ylim[0],scomp_ylim[1])
#             # if invert_scomp_y:
#             #     axes[0][1].invert_yaxis()
#             # axes[0][1].legend()

#             for loc in range(loc_num):
#                 axes[0][2].plot(time*hp['dt']/1000,Z['d'][0,:,loc,:],)
    
#             axes[0][2].set_title('1st decision component')

#             for loc in range(loc_num):
#                 axes[1][0].plot(time*hp['dt']/1000,Z['ts'][0,:,loc,:],)
    
#             axes[1][0].set_title('1st time-stimulus mixing component')
#             # axes[1][0].set_ylim(stcomp_ylim[0],stcomp_ylim[1])
#             # if invert_stcomp_y:
#             #     axes[1][0].invert_yaxis()
#             # axes[2].legend()
#             #plt.show()

#             for loc in range(loc_num):
#                 axes[1][1].plot(time*hp['dt']/1000,Z['sd'][0,:,loc,:],)
    
#             axes[1][1].set_title('1st stimulus-decision mixing component')

#             fig.text(0.5, 0.05, 'time/s', ha='center')
    
#             save_name = save_dir+'Neuron_dPCA_'+m_key+'_perf%.2f'%(perf)#+'_trialnum_'+str(model_index)+'_perf%.2f'%(perf)
#             # if appoint_loc_analysis:
#             #     save_name = save_name+'_locs_'+str(appoint_locs)
#             for save_format in save_formats:
#                 plt.savefig(save_name+'.'+save_format,bbox_inches='tight')
#             plt.close()