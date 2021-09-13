import os
import numpy as np

from utils import tools

from analysis.Plot_growth_curve import plot_growth_curve
from analysis.Analyze_Capacity_PSTH_different_stim_num import analyze_Capacity_PSTH_different_stim_num

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/Capacity_stim1-5_clrwht_6tasks_sequential_16perring_512_1s05s')
    parser.add_argument('--rules', nargs='+')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    #mini test
    ######################################
    # rules = ['Capacity_color_1_stims_white_stims_1s05s','Capacity_color_2_stims_white_stims_1s05s','Capacity_color_3_stims_white_stims_1s05s',\
    #         'Capacity_color_4_stims_white_stims_1s05s','Capacity_color_5_stims_white_stims_1s05s']
    # models_selected = dict()
    # for rule in rules:
    #     models_selected[rule] = {'early':[0,],}
    ######################################

    rules = args.rules
    
    models_selected = tools.auto_model_select(hp,log,)
    plot_growth_curve(hp, log, model_dir,models_selected=models_selected)
    #plot_growth_curve(hp, log, model_dir,smooth_window=0,)

    #analyze_Capacity_PSTH_different_stim_num(hp,log,model_dir,models_selected,rules,norm = True,task_mode='test',)
    analyze_Capacity_PSTH_different_stim_num(hp,log,model_dir,models_selected,rules,norm = True,task_mode='move_one_loc',)
    analyze_Capacity_PSTH_different_stim_num(hp,log,model_dir,models_selected,rules,norm = True,loc_type='all_cond',task_mode='move_one_loc',)
    analyze_Capacity_PSTH_different_stim_num(hp,log,model_dir,models_selected,rules,norm = False,task_mode='move_one_loc',)
    analyze_Capacity_PSTH_different_stim_num(hp,log,model_dir,models_selected,rules,norm = False,loc_type='all_cond',task_mode='move_one_loc',)


