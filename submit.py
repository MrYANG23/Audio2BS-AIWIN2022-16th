import os
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

label=pd.read_csv('/data/xtx/yanghan/thirdtool/data/AIWIN/train/arkit.csv')
#print('lable:',label)
all_label=[]
for per_col in label.columns:
    all_label.append(per_col)

def submit_process(synthe_dir,submit_dir):
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)

    all_dirs=os.listdir(synthe_dir)
    for per_dir in tqdm(all_dirs):
        per_dir_name=[]
        per_path=os.path.join(synthe_dir,per_dir)
        #print('per_path',per_path)
        all_bs_paths=os.listdir(per_path)
        for per_index in sorted(map(lambda x: int(x.split('_')[1]), all_bs_paths)):
            per_bs_path=per_dir+'_'+str(per_index)+'_mfcc.npy'
            #print('per_bs_path',per_bs_path)
            per_bs_out=np.load(os.path.join(per_path,per_bs_path))
            per_dir_name.append(per_bs_out)

        #print('per_dir_name',np.array(per_dir_name).shape)
        # exit()

        per_dir_csv=os.path.join(submit_dir,per_dir+'.csv')
        with open(per_dir_csv,'w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(all_label)
            writer.writerows(per_dir_name)





if __name__ == '__main__':
    submit_process(synthe_dir='/data/xtx/yanghan/thirdtool/Audio2BS/all_synthesies/0812_synthesies_FullLSTM_10_TESTB',submit_dir='./all_subimtdir/submitdir_synthesies_0812_synthesies_FullLSTM_10_TESTB')