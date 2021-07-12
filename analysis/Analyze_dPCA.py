# basic packages #
import os
import numpy as np

import sys
sys.path.append('.')
from utils.tools import mkdir_p, smooth, model_list_parser

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