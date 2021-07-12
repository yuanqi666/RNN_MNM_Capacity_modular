import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils import tools

def plot_growth_curve(hp,log,model_dir,rules_plot=None,smooth_window=5,models_selected=None): # if you don't need to smooth, set smooth_window to 0

    if rules_plot is None:
        rules = hp['rules']
    else:
        rules = rules_plot
    
    fig_pref = plt.figure(figsize=(12,9))
    for rule in rules:
        growth = tools.smooth(log['perf_'+rule],smooth_window)

        #Plot Growth Curve
        plt.plot(log['trials'], growth, label = rule)
        
        #Mark out selected model indexes 
        if models_selected is not None:
            models_selected_rule = tools.model_list_parser(hp, log, rule, models_selected,)[rule]
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

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/Growth_of_Performance/'
    tools.mkdir_p(save_dir)
    if rules_plot is None:
        save_name = save_dir+'growth_of_performance'
    else:
        save_name = save_dir+'growth_of_performance_'+str(rules_plot)

    plt.xlabel("trial trained")
    plt.ylabel("perf")
    plt.ylim(bottom=-0.05)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title('Growth of Performance')
    plt.tight_layout()
    plt.savefig(save_name+'.png', transparent=False, bbox_inches='tight')
    plt.savefig(save_name+'.pdf', transparent=False, bbox_inches='tight')
    plt.savefig(save_name+'.eps', transparent=False, bbox_inches='tight')
    plt.show()