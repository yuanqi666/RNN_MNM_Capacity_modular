import numpy as np

import sys
sys.path.append('.')
from utils.tools import slice_H,model_list_parser,mkdir_p,get_mon_trials,split_test_train

from .Compute_H import compute_single_H

# plot heatmap #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#SVM#
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split#, GridSearchCV


def plot_heatmap(df,save_name,save_formats=['pdf','png','eps']):

    fig, ax = plt.subplots(figsize=(20,16))
    sns.heatmap(df, annot=False, ax=ax,cmap="rainbow",vmin=0.125, vmax=1)
    ax.set_xlabel("Time-bin tested")
    ax.set_ylabel("Time-bin trained")

    for save_format in save_formats:
        plt.savefig(save_name+'.'+save_format,bbox_inches='tight')
    plt.close()


def Decoder_analysis(hp,log,model_dir,model_list,rule,decode_epoch='stim1', #which epoch's/stimulus' spatial information do we need to decode
                        task_info_dict=None,H_dict=None,
                        window_size=2,stride=1,classifier='SVC',decode_info='mnm', #mnm or spatial
                        task_mode='test',perf_type = 'train_perf'):

    model_list = model_list_parser(hp, log, rule, model_list,)

    save_dir = 'figure/'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/Decoder_Analysis/'+classifier+'/'
    mkdir_p(save_dir)
    
    # empty data
    if task_info_dict is None or H_dict is None:
        task_info_dict, H_dict = dict(), dict()
        for m_key in model_list.keys():
            task_info_dict[m_key] = dict()
            H_dict[m_key] = dict()

    for m_key in model_list.keys():

        if not len(model_list[m_key]):
            continue
        
        for model_index in model_list[m_key]:

            if model_index not in task_info_dict[m_key] or model_index not in H_dict[m_key]:
                H, task_info = compute_single_H(hp, log, model_dir, rule, model_index, task_mode=task_mode)
            else:
                H, task_info = H_dict[m_key][model_index],task_info_dict[m_key][model_index]
            
            perf = task_info[perf_type]

            #H = H*(hp['dt']/1000)

            time_len = np.size(H,0)

            time_bin_num = int((time_len-window_size)//stride+1)

            tb_set = dict()

            if decode_info == 'spatial':
                #classifier does not accept float as labels
                loc_set = list(task_info['loc_set'][decode_epoch])
                label = np.array([loc_set.index(i) for i in task_info['input_loc'][decode_epoch]])
            elif decode_info == 'mnm':
                stim1_locs = task_info['input_loc']['stim1']
                stim2_locs = task_info['input_loc']['stim2']
                label = np.array([st[0]==st[1] for st in zip(stim1_locs,stim2_locs)])
            
            row = list()
            for b_num in range(time_bin_num):
                start = b_num*stride
                end = start+window_size

                scaler = StandardScaler()
                tb_set[b_num] = scaler.fit_transform(H[start:end,:,:].mean(axis=0))
                #tb_set[b_num] = H[start:end,:,:].mean(axis=0)

                row.append(str(start*hp['dt']/1000))

            init_data = np.full([len(row),len(row)], np.nan)
            df = pd.DataFrame(data=init_data,  columns=row, index=row)

            for b_num1 in range(time_bin_num):
        
                #x_train, x_test, y_train, y_test = train_test_split(tb_set[b_num1],label,test_size=test_size)
                x_train, x_test, y_train, y_test = split_test_train(tb_set[b_num1],label,)
                if classifier == 'SVC':
                    from sklearn.svm import SVC
                    clf = SVC()
                elif classifier == 'RF':
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier()
                elif classifier == 'LinearSVC':
                    from sklearn.svm import LinearSVC
                    clf = LinearSVC()
                elif classifier == 'KNeighbors':
                    from sklearn.neighbors import KNeighborsClassifier
                    clf = KNeighborsClassifier()
                elif classifier == 'ExtraTrees':
                    from sklearn.ensemble import ExtraTreesClassifier
                    clf = ExtraTreesClassifier()
                elif classifier == 'MLP':
                    from sklearn.neural_network import MLPClassifier
                    clf = MLPClassifier()
                elif classifier == 'AdaBoost':
                    from sklearn.ensemble import AdaBoostClassifier
                    clf = AdaBoostClassifier()
                clf.fit(x_train, y_train)
                #clf.fit(tb_set[b_num1],label)

                key1 = str(b_num1*stride*hp['dt']/1000)

                for b_num2 in range(time_bin_num):
                    key2 = str(b_num2*stride*hp['dt']/1000)
            
                    if b_num1 == b_num2:
                        score = clf.score(x_test,y_test)
                    else:
                        #score = clf.score(tb_set[b_num2],label)
                        _, x1_test, _, y1_test = split_test_train(tb_set[b_num2],label,)
                        score = clf.score(x1_test,y1_test)
            
                    #score = clf.score(tb_set[b_num2],label)
            

                    df.loc[key1,key2] = score
                    #if key1>key2:
                    #    df.loc[key1,key2] = 0
            
            save_name = save_dir+"MNM_Decoder_analysis_"
            if decode_info == 'spatial':
                save_name = save_name+"decode_epoch_"+decode_epoch+"_"
            save_name = save_name+"decode_info_"+decode_info+'_'+str(model_index)+'perf%.2f'%(perf)+'_w'+str(window_size*hp['dt'])+'ms_s'+str(stride*hp['dt'])+'ms'
            plot_heatmap(df,save_name,save_formats=['pdf',])