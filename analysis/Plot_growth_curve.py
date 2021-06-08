import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils import tools

def plot_growth_curve(hp,log,model_dir,smooth_growth=True,smooth_window=5,models_selected=None): # if you don't need smooth, set smooth_window to 0

    print('rule trained: ', hp['rules'])
    print('minimum model index: ', log['trials'][0])
    print('maximum model index: ', log['trials'][-1])
    print('minimum step       : ', log['trials'][1])
    print('total model number : ', len(log['trials']))

    fig_pref = plt.figure(figsize=(12,9))
    for rule in hp['rules']:
        if smooth_growth:
            growth = tools.smooth(log['perf_'+rule],smooth_window)
        else:
            growth = log['perf_'+rule]

        #Plot Growth Curve
        plt.plot(log['trials'], growth, label = rule)
        
        #Mark out selected model indexes 
        if models_selected is not None:
            models_selected_rule = tools.model_list_parser(hp, log, rule, models_selected,)
            for m_c in [('early','green'),('mid','blue'),('mature','red')]:
                plt.fill_between(log['trials'], growth, where=[i in models_selected_rule[m_c[0]] for i in log['trials']],\
                    facecolor=m_c[1],alpha=0.3)

    #Mark rule start point in sequential training
    if 'rule_now' in log.keys():
        rule_set = list()
        model_index_set = list()

        for i in range(len(log['rule_now'])):
            rules_now = '_'.join(log['rule_now'][i])+'_start'
            if rules_now not in rule_set:
                rule_set.append(rules_now)
                model_index_set.append(log['trials'][i])
                plt.axvline(log['trials'][i],color="grey",linestyle = '--')
        for i in range(len(model_index_set)):
            plt.text(model_index_set[i],0,rule_set[i])

    tools.mkdir_p('figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/')

    plt.xlabel("trial trained")
    plt.ylabel("perf")
    plt.ylim(bottom=-0.05)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title('Growth of Performance')
    save_name = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/growth_of_performance'
    plt.tight_layout()
    plt.savefig(save_name+'.png', transparent=False, bbox_inches='tight')
    plt.savefig(save_name+'.pdf', transparent=False, bbox_inches='tight')
    plt.savefig(save_name+'.eps', transparent=False, bbox_inches='tight')
    plt.show()