import os
import numpy as np

from utils import tools

from analysis.Plot_growth_curve import plot_growth_curve
from analysis.Compute_H import compute_H
from analysis.Analyze_Capacity_PSTH import analyze_Capacity_PSTH

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument('--modeldir', type=str, default='data/Capacity_clrwht2ch_3stim_MNM_clr2ch_6tasks_36perring_512')
    parser.add_argument('--modeldir', type=str, default='data/Capacity_clrwht_MNM_clr_6tasks_sequential_16perring_512')
    parser.add_argument('--rule', type=str, default="Capacity_color_3_stims_white_stims_2chan")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    #mini test
    ######################################
    # model_list=[100000000,]
    # rule = 'Capacity_color_3_stims_white_stims_2chan'
    # epoch = 'stim1'
    # analyze_Capacity_PSTH(hp,log,model_dir,model_list,rule,epoch,norm=False)
    ######################################

    rule = args.rule
    
    models_selected = tools.auto_model_select(hp,log,)
    plot_growth_curve(hp, log, model_dir,models_selected=models_selected)
    #plot_growth_curve(hp, log, model_dir,smooth_window=0,)

    H_dict, task_info_dict = compute_H(hp,log,model_dir,rule,models_selected[rule],task_mode='test')


    # analyze_Capacity_PSTH(hp,log,model_dir,models_selected[rule],rule,'stim1',\
    #     task_info_dict=task_info_dict,H_dict=H_dict,norm=False)
    analyze_Capacity_PSTH(hp,log,model_dir,models_selected[rule],rule,'stim1',\
        task_info_dict=task_info_dict,H_dict=H_dict,norm=True)