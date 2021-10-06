from training.train_sequential import train_sequential

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/Capacity_mixdelay001005_clrwht_6tasks_sequential_16perring_512')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--neachring', type=int, default=16)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    hp = {# number of units each ring
          'n_eachring': args.neachring,
          # number of rings/modalities
          'num_ring': 3,
          'activation': 'softplus',
          'n_rnn': 512,
          'learning_rate': 0.001,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': False,
          'target_perf': 0.995,
          'mature_target_perf': 0.95,
          'mid_target_perf': 0.65,
          'early_target_perf': 0.35,}

    train_sequential(args.modeldir,
                    seed=args.seed,
                    hp=hp,
                    ruleset='Capacity_clr_mixuni_1-5stim_0105_6tasks',
                    rule_trains=[['overlap','zero_gap','gap','odr','odrd','gap500','Capacity_clr_mixuni_1stim_0105'],\
                        ['overlap','zero_gap','gap','odr','odrd','gap500','Capacity_clr_mixuni_1stim_0105','Capacity_clr_mixuni_2stim_0105'],\
                            ['overlap','zero_gap','gap','odr','odrd','gap500','Capacity_clr_mixuni_1stim_0105','Capacity_clr_mixuni_2stim_0105','Capacity_clr_mixuni_3stim_0105'],\
                                ['overlap','zero_gap','gap','odr','odrd','gap500','Capacity_clr_mixuni_1stim_0105','Capacity_clr_mixuni_2stim_0105','Capacity_clr_mixuni_3stim_0105','Capacity_clr_mixuni_4stim_0105'],\
                                    ['overlap','zero_gap','gap','odr','odrd','gap500','Capacity_clr_mixuni_1stim_0105','Capacity_clr_mixuni_2stim_0105','Capacity_clr_mixuni_3stim_0105',\
                                        'Capacity_clr_mixuni_4stim_0105','Capacity_clr_mixuni_5stim_0105'],],
                    display_step=500,
                    continue_after_target_reached=True,)
