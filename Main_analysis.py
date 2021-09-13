import os
import numpy as np

from utils import tools

from analysis.Plot_growth_curve import plot_growth_curve
from analysis.Compute_H import compute_H
from analysis.Compute_neuron_information import compute_neuron_information
from analysis.Analyze_tuning_curve import analyze_tuning_curve
from analysis.Analyze_PSTH import analyze_PSTH
from analysis.Analyze_epoch_firerate_growth import analyze_epoch_firerate_growth
from analysis.Analyze_Fano_factor import analyze_Fano_factor
from analysis.Analyze_mix_selectivity import analyze_mix_selectivity_tuning, analyze_mix_selectivity_2ANOVA
from analysis.Analyze_Decoder import Decoder_analysis
from analysis.Analyze_dPCA import dPCA_analysis
from analysis.Analyze_spike_count_correlation import analyze_spike_count_correlation
from analysis.Analyze_neuron_number import analyze_neuron_number
from analysis.Analyze_MNM_effect_2ANOVA import analyze_MNM_effect_2ANOVA

def main_analysis(hp, log, model_dir, models_selected, rule, task_info_dict, H_dict,):
    #compute neuron info
    ###################################################
    neuron_info = dict()
    #for epoch in ['stim1','delay1','stim2','delay2']:
    for epoch in ['stim1','delay1',]:
        neuron_info[epoch] = dict()
        neuron_info[epoch]['norm'] = compute_neuron_information(hp, log, model_dir, rule, epoch, models_selected[rule], \
            task_mode='test',norm=True,task_info_dict=task_info_dict,H_dict=H_dict,)
        neuron_info[epoch]['raw1sfr'] = compute_neuron_information(hp, log, model_dir, rule, epoch, models_selected[rule], \
            task_mode='test',norm=False,task_info_dict=task_info_dict,H_dict=H_dict,)
    ###################################################

    #Analyze the change of neuron number in each neuron type during the training
    ###################################################
    analyze_neuron_number(hp, log, model_dir, log['trials'], rule, 'stim1', \
        task_info_dict=task_info_dict,H_dict=H_dict,neuron_info_dict=neuron_info[epoch]['raw1sfr'],\
            norm=False, task_mode='test', plot=True) 
    #analyze_neuron_number(hp, log, model_dir, log['trials'], rule, 'stim1', \
    #    task_info_dict=task_info_dict,H_dict=H_dict,neuron_info_dict=neuron_info[epoch]['norm'],\
    #        norm=True, task_mode='test', plot=True)
    ###################################################

    #Analyze tuning curve and PSTH
    ###################################################
    #for epoch in ['stim1','delay1','stim2','delay2']:
    for epoch in ['stim1','delay1',]:
        for norm_ in ['norm', 'raw1sfr']:
            if norm_ == 'norm':
                norm = True
            elif norm_ == 'raw1sfr':
                norm = False
            for neuron_type in ['excitatory','inhibitory','significant']:
                analyze_tuning_curve(hp, log, model_dir, models_selected[rule], rule, epoch, \
                    task_info_dict=task_info_dict,H_dict=H_dict,neuron_info_dict=neuron_info[epoch][norm_],\
                        norm=norm, task_mode='test', neuron_type=neuron_type,)

                analyze_PSTH(hp, log, model_dir, models_selected[rule], rule, epoch, \
                    task_info_dict=task_info_dict,H_dict=H_dict,neuron_info_dict=neuron_info[epoch][norm_], \
                        norm=norm, loc_type='best_cue', task_mode='test', neuron_type=neuron_type,)
    ##################################################

    #Analyze epoch firerate growth
    ##################################################
    for epoch in ['stim1', 'delay1']:
        for neuron_type in ['excitatory','all','significant']:
            analyze_epoch_firerate_growth(hp, log, model_dir, log['trials'], rule, epoch, \
                task_info_dict=task_info_dict,H_dict=H_dict,neuron_info_dict=neuron_info[epoch]['raw1sfr'], \
                    norm=False, loc_type='best_cue', task_mode='test', neuron_type=neuron_type)
    ##################################################

    #Choose one model from each stage
    ##################################################
    for model_index in models_selected[rule]['early']:
        if 0 <= hp['early_target_perf'] - log['perf_'+rule][model_index//log['trials'][1]] <=0.03:
            early_model = model_index
            break
    for model_index in models_selected[rule]['mid']:
        if 0 <= hp['mid_target_perf'] - log['perf_'+rule][model_index//log['trials'][1]] <=0.03:
            mid_model = model_index
            break
    one_model_per_stage = [early_model, mid_model, models_selected[rule]['mature'][-1]]
    #################################################

    # Analyze Fano factor in stim1
    #################################################
    for epoch in ['stim1',]:
        analyze_Fano_factor(hp, log, model_dir, one_model_per_stage, rule, epoch, \
            task_info_dict=task_info_dict,H_dict=H_dict,neuron_info_dict=neuron_info[epoch]['raw1sfr'], \
                norm=False, loc_type='best_cue', task_mode='test', neuron_type='significant',)
    #################################################

    #analyze mix selectivity
    #################################################
    for norm in [True, False]:
        analyze_mix_selectivity_tuning(hp, log, model_dir, one_model_per_stage, rule, \
            task_info_dict=task_info_dict,H_dict=H_dict, \
                norm=norm, task_mode='test',)

    analyze_mix_selectivity_2ANOVA(hp, log, model_dir, one_model_per_stage, rule, \
        task_info_dict=task_info_dict,H_dict=H_dict, \
            task_mode='test',)
    #################################################

    #Decoder analysis
    #################################################
    Decoder_analysis(hp, log, model_dir, one_model_per_stage, rule,\
        task_info_dict=task_info_dict,H_dict=H_dict, \
           window_size=2, stride=1, classifier='SVC', decode_info='mnm', task_mode='test', perf_type='train_perf')

    Decoder_analysis(hp, log, model_dir, one_model_per_stage, rule, decode_epoch='stim1',\
        task_info_dict=task_info_dict,H_dict=H_dict, \
           window_size=2, stride=1, classifier='SVC', decode_info='spatial', task_mode='test', perf_type='train_perf')
    #################################################

    #dPCA analysis
    #################################################
    dPCA_analysis(hp, log, model_dir, one_model_per_stage, rule, epoch='stim1', \
        task_info_dict=task_info_dict,H_dict=H_dict, \
            task_mode='test', save_formats=['pdf','eps'])
    #################################################

    #neuron spike count correlation analysis
    #################################################
    analyze_spike_count_correlation(hp, log, model_dir, one_model_per_stage, rule, epoch='stim1', \
        task_info_dict=task_info_dict,H_dict=H_dict, \
            task_mode='test', z_norm=True,appendix_info="one_model_per_stage")

    analyze_spike_count_correlation(hp, log, model_dir, models_selected[rule], rule, epoch='stim1', \
        task_info_dict=task_info_dict,H_dict=H_dict, \
            task_mode='test', z_norm=True,)
    #################################################

    #analyze MNM effect by 2 way ANOVA
    #################################################
    analyze_MNM_effect_2ANOVA(hp, log, model_dir, one_model_per_stage, rule, epoch='stim2', \
        task_info_dict=task_info_dict,H_dict=H_dict, \
            task_mode='test', perf_type='train_perf')
    #################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/MNM_color_6tasks_8per-ring')
    #parser.add_argument('--modeldir', type=str, default='data/Capacity_6tasks_ei')
    parser.add_argument('--rule', type=str, default='MNM_color')
    #parser.add_argument('--epochs', nargs='+')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #print(args.epochs)

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    # mini test
    ######################################
    #model_list=[1072640,]
    #rule = 'MNM_color'
    #epoch = 'stim1'
    ######################################

    rule = args.rule
    
    models_selected = tools.auto_model_select(hp,log,)
    plot_growth_curve(hp, log, model_dir,models_selected=models_selected)

    H_dict, task_info_dict = compute_H(hp,log,model_dir,rule,models_selected[rule],task_mode='test')

    main_analysis(hp, log, model_dir, models_selected, rule, task_info_dict, H_dict,)