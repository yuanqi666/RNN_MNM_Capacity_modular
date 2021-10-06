import os
import numpy as np

from utils import tools

from analysis.Plot_growth_curve import plot_growth_curve
from analysis.Analyze_dPCA import Capacity_dPCA_analysis

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--modeldir', type=str, default='data/Capacity_stim1-5_clrwht_6tasks_sequential_16perring_512_1s05s')
    # parser.add_argument('--rule', type=str, default="Capacity_color_5_stims_white_stims_1s05s")
    parser.add_argument('--modeldir', type=str, default='data/Capacity_stim1-4_clrwht_6tasks_sequential_16perring_512_1s05s_mx')
    parser.add_argument('--rule', type=str, default='Capacity_color_4_stims_wht_stims_1s05s_mx')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    rule = args.rule

    #mini test
    ######################################
    # model_list=[1280,]
    # rule = 'Capacity_color_5_stims_white_stims_1s05s'
    # epoch = 'stim1'
    # Capacity_dPCA_analysis(hp,log,model_dir,model_list,rule,task_mode='move_one_loc')
    ######################################
    
    models_selected = tools.auto_model_select(hp,log,)

    # model_list = {'mid':[models_selected[rule]['mid'][0],], 'mature':[models_selected[rule]['mature'][-1],]}

    # Capacity_dPCA_analysis(hp,log,model_dir,model_list,rule,task_mode='move_one_loc')


    rules = ['Capacity_color_1_stims_wht_stims_1s05s_mx','Capacity_color_2_stims_wht_stims_1s05s_mx','Capacity_color_3_stims_wht_stims_1s05s_mx',\
            'Capacity_color_4_stims_wht_stims_1s05s_mx']
    Capacity_dPCA_analysis(hp,log,model_dir,log['trials'][-1],rules,task_mode='move_one_loc')
    Capacity_dPCA_analysis(hp,log,model_dir,models_selected[rule]['mid'][0],rules,task_mode='move_one_loc')
