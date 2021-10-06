import os
import numpy as np

from utils import tools

from analysis.Plot_growth_curve import plot_growth_curve
from analysis.Analyze_Capacity_PSTH_stimnear_oppoRF import analyze_Capacity_PSTH_stimnear_oppoRF

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/Capacity_stim1-4_clrwht_6tasks_sequential_16perring_512_1s05s_mx')
    parser.add_argument('--rule', type=str, default="Capacity_color_4_stims_wht_stims_1s05s_mx")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    #mini test
    ######################################
    # rule = "Capacity_color_4_stims_wht_stims_1s05s_mx"
    # models_selected = dict()
    # models_selected[rule] = {'early':[0,],}
    ######################################

    rule = args.rule
    
    models_selected = tools.auto_model_select(hp,log,)
    plot_growth_curve(hp, log, model_dir,models_selected=models_selected)
    #plot_growth_curve(hp, log, model_dir,smooth_window=0,)

    # analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,models_selected[rule],rule,norm = False,loc_type='best_cond',plot=True,)
    # analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,models_selected[rule],rule,norm = False,loc_type='all_cond',plot=True,)

    analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,log['trials'][-10:],rule,norm = False,loc_type='best_cond',plot=True,)
    analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,log['trials'][-10:],rule,norm = False,loc_type='all_cond',plot=True,)
    analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,log['trials'][-10:],rule,norm = True,loc_type='best_cond',plot=True,)
    analyze_Capacity_PSTH_stimnear_oppoRF(hp,log,model_dir,log['trials'][-10:],rule,norm = True,loc_type='all_cond',plot=True,)


