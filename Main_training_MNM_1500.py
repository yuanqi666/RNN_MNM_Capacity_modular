from training.train import train

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument('--modeldir', type=str, default='data/6tasks')
    parser.add_argument('--modeldir', type=str, default='data/MNM_color_6tasks_8per-ring_delay1500')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--neachring', type=int, default=8)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    hp = {# number of units each ring
          'n_eachring': args.neachring,
          # number of rings/modalities
          'num_ring': 3,
          'activation': 'softplus',
          'n_rnn': 256,
          'learning_rate': 0.001,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': False,
          'target_perf': 0.995,
          'mature_target_perf': 0.95,
          'mid_target_perf': 0.65,
          'early_target_perf': 0.35,}

    train(args.modeldir,
        seed=args.seed,
        hp=hp,
        # ruleset='MNM_color_6tasks',
        # rule_trains=['MNM_color','overlap','zero_gap','gap','odr','odrd','gap500',],
        ruleset='MNM_color_6tasks_1500',
        rule_trains=['MNM_color_1500','overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20,
        continue_after_target_reached=True)