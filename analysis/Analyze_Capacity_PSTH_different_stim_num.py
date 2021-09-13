import numpy as np

import sys
sys.path.append('.')
from utils.tools import model_list_parser,mkdir_p

from .Analyze_Capacity_PSTH import analyze_Capacity_PSTH

import matplotlib.pyplot as plt

def plot_Capacity_PSTH_different_stim_num(model_dir,dt,norm,plot_information,loc_type='best_cond',save_formats=['pdf','png','eps']):

    #colors = {'mature':'red','mid':'blue','early':'green'}

    if loc_type == 'best_cond':
        loctyp = 'best'
    elif loc_type == 'all_cond':
        loctyp = 'all'

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/Capacity_PSTH_diff_stim_num/'
    mkdir_p(save_dir)

    multi_stim_midmature_fig,multi_stim_midmature_ax = plt.subplots(figsize=(16,10))

    for m_key in ['mature','mid','early']:
        color_cycle = plt.rcParams['axes.prop_cycle']()

        multi_stim_fig,multi_stim_ax = plt.subplots(figsize=(16,10))
        multi_stim_sep_fig,multi_stim_sep_ax = plt.subplots(figsize=(16,10))

        ################################
        rows = len(plot_information)//3
        if len(plot_information)%3 != 0:
            rows+=1
        fig,axes = plt.subplots(rows,3,figsize=(24,14),sharey=True)

        i_rule = 0
        ################################

        for rule, plt_info in plot_information.items():
            color = next(color_cycle)['color']
            try:
                # fig,ax = plt.subplots(figsize=(16,10))
                # x_PSTH = np.arange(len(plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M']))*(dt/1000)
                # ax.plot(x_PSTH, plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M'], color = colors[m_key],\
                #     label = 'Match '+m_key+' avg_perf:%.2f'%(plt_info[m_key]['perf']))
                # ax.plot(x_PSTH, plt_info[m_key]['PSTH_'+loctyp+'cue1cond_NM'], color = colors[m_key],\
                #     label = 'Non-match '+m_key+' avg_perf:%.2f'%(plt_info[m_key]['perf']),linestyle = '--')
                # title = 'Rule:'+rule+' PSTH'
                # fig.suptitle(title)
                # ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                # ax.set_ylabel('activity')
                # ax.set_xlabel('time/s')

                # save_name = save_dir+rule+'_'+m_key+'_'+loc_type+'_'
                # if norm:
                #     save_name += 'normalzed'
                # else:
                #     save_name += 'raw_1s_firerate'

                # for save_format in save_formats:
                #     fig.savefig(save_name+'.'+save_format,bbox_inches='tight')

                # plt.close(fig)

                x_PSTH = np.arange(len(plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M']))*(dt/1000)
                axes[i_rule//3][i_rule%3].plot(x_PSTH, plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M'],color=color,\
                    label = 'Match '+m_key+' avg_perf:%.2f'%(plt_info[m_key]['perf']))
                axes[i_rule//3][i_rule%3].plot(x_PSTH, plt_info[m_key]['PSTH_'+loctyp+'cue1cond_NM'],color=color,\
                    label = 'Non-match '+m_key+' avg_perf:%.2f'%(plt_info[m_key]['perf']),linestyle = '--')
                title = 'Rule:'+rule+' PSTH'
                axes[i_rule//3][i_rule%3].set_title(title)

            except:
                pass

            ###########
            i_rule += 1
            ###########

            try:
                x_PSTH = np.arange(len(plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M']))*(dt/1000)
                PSTH = (plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M']+plt_info[m_key]['PSTH_'+loctyp+'cue1cond_NM'])/2
                multi_stim_ax.plot(x_PSTH, PSTH,label = rule+' avg_perf:%.2f'%(plt_info[m_key]['perf']),color=color)

                multi_stim_sep_ax.plot(x_PSTH, plt_info[m_key]['PSTH_'+loctyp+'cue1cond_M'],label = rule+' avg_perf:%.2f'%(plt_info[m_key]['perf']),color=color)
                multi_stim_sep_ax.plot(x_PSTH, plt_info[m_key]['PSTH_'+loctyp+'cue1cond_NM'],\
                    label = rule+' avg_perf:%.2f'%(plt_info[m_key]['perf']),color=color,linestyle = '--')

                if m_key == 'mature':
                    multi_stim_midmature_ax.plot(x_PSTH, PSTH,label = rule+' mature avg_perf:%.2f'%(plt_info[m_key]['perf']),color=color)
                elif m_key == 'mid':
                    multi_stim_midmature_ax.plot(x_PSTH, PSTH,label = rule+' mid avg_perf:%.2f'%(plt_info[m_key]['perf']),color=color,linestyle = '--')
            except:
                pass

        ###################################################################
        fig.text(0.5, 0.05, 'time/s', ha='center')
        fig.text(0.1, 0.5, 'activity', va='center', rotation='vertical')
        #for ax_ in axes.reshape(-1):
        #    plt.setp(ax_.get_yticklabels(),visible=True)
        #handles, labels = axes[-1][-1].get_legend_handles_labels()
        #fig.legend(handles, labels,bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        fig.legend(loc='lower right',)
        save_name = save_dir+'Separate_rule_PSTH_'+m_key+'_'+loc_type+'_'
        if norm:
            save_name += 'normalzed'
        else:
            save_name += 'raw_1s_firerate'

        for save_format in save_formats:
            fig.savefig(save_name+'.'+save_format,bbox_inches='tight',)

        plt.close(fig)
        ###################################################################

        title_multi_stim = 'PSTH for rules with different stimulus number '+'('+m_key+' stage)'
        multi_stim_fig.suptitle(title_multi_stim)
        multi_stim_ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        multi_stim_ax.set_ylabel('activity')
        multi_stim_ax.set_xlabel('time/s')

        save_name_multi = save_dir+'Capacity_PSTH_diff_stim_num_'+m_key+'_'+loc_type+'_'
        if norm:
            save_name_multi += 'normalzed'
        else:
            save_name_multi += 'raw_1s_firerate'

        for save_format in save_formats:
            multi_stim_fig.savefig(save_name_multi+'.'+save_format,bbox_inches='tight')

        plt.close(multi_stim_fig)

        title_multi_stim_sep = 'PSTH for rules with different stimulus number '+'('+m_key+' stage)'
        multi_stim_sep_fig.suptitle(title_multi_stim_sep)
        multi_stim_sep_ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        multi_stim_sep_ax.set_ylabel('activity')
        multi_stim_sep_ax.set_xlabel('time/s')

        save_name_multi_sep = save_dir+'Capacity_PSTH_sepMNM_diff_stim_num_'+m_key+'_'+loc_type+'_'
        if norm:
            save_name_multi_sep += 'normalzed'
        else:
            save_name_multi_sep += 'raw_1s_firerate'

        for save_format in save_formats:
            multi_stim_sep_fig.savefig(save_name_multi_sep+'.'+save_format,bbox_inches='tight')

        plt.close(multi_stim_sep_fig)

    title_multi_stim_midmature = 'PSTH for rules with different stimulus number '+'('+m_key+' stage)'
    multi_stim_midmature_fig.suptitle(title_multi_stim_midmature)
    multi_stim_midmature_ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    multi_stim_midmature_ax.set_ylabel('activity')
    multi_stim_midmature_ax.set_xlabel('time/s')

    save_name_multi_midmature = save_dir+'Capacity_PSTH_midmat_diff_stim_num_'+loc_type+'_'
    if norm:
        save_name_multi_midmature += 'normalzed'
    else:
        save_name_multi_midmature += 'raw_1s_firerate'

    for save_format in save_formats:
        multi_stim_midmature_fig.savefig(save_name_multi_midmature+'.'+save_format,bbox_inches='tight')

    plt.close(multi_stim_midmature_fig)


def analyze_Capacity_PSTH_different_stim_num(hp,log,model_dir,models_selected,rules,#epoch,
                norm = True,task_mode='test',loc_type='best_cond',
                perf_type = 'train_perf',plot=True,):

    plot_information = dict()
    
    for rule in rules:
        model_list = model_list_parser(hp, log, rule, models_selected[rule],)
        #plot_information[rule] = analyze_Capacity_PSTH(hp,log,model_dir,model_list,rule,epoch,norm=norm,task_mode=task_mode,plot=False)
        plot_information[rule] = analyze_Capacity_PSTH(hp,log,model_dir,model_list,rule,epoch='stim1',norm=norm,task_mode=task_mode,plot=False)

    if not plot:
        return plot_information
    else:
        #plot_Capacity_PSTH_different_stim_num(model_dir,hp['dt'],epoch,norm,plot_information,save_formats=['pdf',])
        plot_Capacity_PSTH_different_stim_num(model_dir,hp['dt'],norm,plot_information,loc_type=loc_type,save_formats=['pdf',])