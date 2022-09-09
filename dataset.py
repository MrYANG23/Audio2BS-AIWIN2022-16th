import os


import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import torch.utils.data as data
from torch.utils.data.dataloader import Dataset
from torch.utils.data.dataloader import DataLoader
#


class Blendshape_dataset(Dataset):
    def __init__(self,feature_file,target_file,target_dir):
        self.feature_file=feature_file
        self.target_file=target_file
        self.target_dir=target_dir
        self.all_features=os.listdir(feature_file)
        self.total_features = []
        for per_feature in self.all_features:
            self.total_features.append(os.path.join(self.feature_file, per_feature))

        self.label=pd.read_csv(target_file)
        self.chose_label=[]
        for per_col in self.label:
            self.chose_label.append(per_col)

    def __len__(self):
        return len(self.total_features)

    def __getitem__(self, item):
        feature_path=self.total_features[item]
        feture=torch.from_numpy(np.load(os.path.join(self.feature_file,feature_path)))
        if feature_path.split('/')[-1].split('_')[0].isupper():
            Blendshape_path = os.path.join(self.target_dir, feature_path.split('/')[-1].split('_')[0] + '_anim.csv')
        # if os.path.exists(os.path.join(self.target_dir,feature_path.split('/')[-1].split('_')[0]+'_Anim.csv')):
        #     Blendshape_path=os.path.join(self.target_dir,feature_path.split('/')[-1].split('_')[0]+'_Anim.csv')
        else:
            Blendshape_path=os.path.join(self.target_dir, feature_path.split('/')[-1].split('_')[0] + '_Anim.csv')
        #print('--------------------Blendshape',Blendshape_path)

        blendtensor = np.array(pd.read_csv(Blendshape_path, usecols=self.chose_label))
        #rint('blendtensor.shape',blendtensor.shape)
        index=feature_path.split('/')[-1].split('_')[1]
        #print('----------------index',index)
        bs_target=torch.from_numpy(blendtensor[int(index)])
        #print('feture',feture.shape)
        #print('bs_target.shape',bs_target.shape)
        return feture.float(),bs_target.float()

class Blendshape_Test_dataset(Dataset):
    def __init__(self,feature_file):
        self.feature_file=feature_file

        self.all_features=os.listdir(feature_file)
        self.total_features = []
        for per_feature in self.all_features:
            self.total_features.append(os.path.join(self.feature_file, per_feature))


    def __len__(self):
        return len(self.total_features)

    def __getitem__(self, item):
        feature_path=self.total_features[item]
        feture=torch.from_numpy(np.load(os.path.join(self.feature_file,feature_path)))


        return feture.float(),feature_path
#

if __name__ == '__main__':
    BS_dataset=Blendshape_dataset(feature_file='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val_feature_blendshape',target_file='/data/xtx/yanghan/thirdtool/data/AIWIN/train/arkit.csv',target_dir='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val')
    #BS_dataset=Blendshape_Test_dataset(feature_file='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val_feature_blendshape')
    train_loder=DataLoader(BS_dataset,batch_size=8)
    print('type(train_loder)',type(train_loder))
    #print('---------------------len(train_loder)',len(train_loder))
    for (feature,bs_target) in train_loder:
        print(feature.shape)
        print(bs_target.shape)
        exit()
#     for index,data in enumerate(train_loder):
#         feature,bs_target=data
#         exit()
# #     #     print(data)