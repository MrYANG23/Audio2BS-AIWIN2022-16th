import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

label=pd.read_csv('/data/xtx/yanghan/thirdtool/data/AIWIN/train/arkit.csv')
#print('lable:',label)
all_label=[]
for per_col in label.columns:
    all_label.append(per_col)


#csv_listdir=['/data/xtx/yanghan/thirdtool/Audio2BS/submitdir_FullLSTM_epoch10']
# csv_listdir=['/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_FullLSTM_epoch21','/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_FullLSTM_epoch10',
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_0725_LSTM__epoch48','/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_0725_LSTM__epoch40']
# csv_listdir=[
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_FullLSTM_epoch21',
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_FullLSTM_epoch10',
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_0808trainval_FullLSTM__epoch85',
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_0808trainval_FullLSTM__epoch96',
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_0725_LSTM__epoch48',
#              '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_0725_LSTM__epoch40'
#
#              ]

csv_listdir=['/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_synthesies_0812_synthesies_0725LSTM_40_TESTB',
             '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_synthesies_0812_synthesies_0725LSTM_48_TESTB',
             '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_synthesies_0812_synthesies_FullLSTM_10_TESTB',
             '/data/xtx/yanghan/thirdtool/Audio2BS/all_subimtdir/submitdir_synthesies_0812_synthesies_FullLSTM_21_TESTB'
             ]
output_dir='./submit_ensemble_submitdir_synthesies_0812_FullLSTM_10_21_LSTM_48_40_TESTB'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


ensemble_list=[]
total_csvs=[]
for perdir in tqdm(csv_listdir):
    #all_csvs=os.listdir(perdir)
    perdir_list=[]
    all_csvs=glob(perdir+'/*.csv')
    total_csvs.append(all_csvs)
    #
    #print('all_csvs',all_csvs)
    for per_csv in all_csvs:
        #print('per_csv',per_csv)
        Blendshape_path=os.path.join(perdir,per_csv)
        # print('blendshape_path',Blendshape_path)
        # exit()
        blendtensor = np.array(pd.read_csv(Blendshape_path))

        perdir_list.append(blendtensor)
    #print('------------------------len(perdir_list)',len(perdir_list))
    ensemble_list.append(perdir_list)
#print('len(ensemble_list)',len(ensemble_list))

for predA,predB,predC,predD,label_path in zip(ensemble_list[0],ensemble_list[1],ensemble_list[2],ensemble_list[3],total_csvs[0]):
    #print('label_path',label_path)
    target_blendshape=np.array(pd.read_csv(label_path))
    assert len(predA)==len(predB)==target_blendshape.shape[0]==len(predC)==len(predD)
    save_path_csv=os.path.join(output_dir,label_path.split('/')[-1])
    save_tensor=(predA+predB+predC+predD)/4
    #print('save_tensor.shape',save_tensor.shape)

    with open(save_path_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(all_label)
        writer.writerows(save_tensor)


