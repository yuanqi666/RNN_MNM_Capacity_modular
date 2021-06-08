import os
import numpy as np

from utils import tools

from analysis.Plot_growth_curve import plot_growth_curve

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/MNM_color_6tasks_8per-ring')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    
    recompute = False
    
    models_selected = tools.auto_model_select(hp,log,)
    plot_growth_curve(hp, log, model_dir, smooth_growth=True,models_selected=models_selected)
    
