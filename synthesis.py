import os

import torch
from model import NvidiaNet,FullyLSTM,LSTMNvidiaNet,LSTM
import pandas as pd
import csv
from dataset import Blendshape_Test_dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm



checkpoint_file='/data/xtx/yanghan/thirdtool/Audio2BS/all_checpoint_dir/checkpoint_dir_0721_FullLSTM/Full_LSTM_chechpoint-epoch10-train_loss1.1946325685130432e-05-val_losspth.tar'

result_path='./all_synthesies/0812_synthesies_FullLSTM_10_TESTB'


#model=NvidiaNet()
model=FullyLSTM()
#model=LSTMNvidiaNet()
#model=LSTM()
checkpoint=torch.load(checkpoint_file)
#print('model epoch {} loss :{}'.format(checkpoint['epoch'],checkpoint['eval_loss']))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

test_loader=DataLoader(
    Blendshape_Test_dataset(feature_file='/data/xtx/yanghan/thirdtool/Audio2BS/data/test-B/audio_for_B/all_test_B_output'),batch_size=64,shuffle=False
)

for (feature,path) in tqdm(test_loader):
    output_bs=model(feature)
    output_bs=output_bs.detach().numpy()
    #print('output_bs.shape',output_bs.shape)

    #print(path)
    for index,per_path in enumerate(path):
        if len(per_path.split('/')[-1].split('_'))==3:
            base_dir=per_path.split('/')[-1].split('_')[0]
        if len(per_path.split('/')[-1].split('_'))==4:
            base_dir=per_path.split('/')[-1].split('_')[0]+'_'+per_path.split('/')[-1].split('_')[1]
        save_dir=os.path.join(result_path,base_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #print('base_dir',base_dir)
        per_bs=output_bs[index]
        #np.save(per_bs,)
        #print('per_bs.shape',per_bs.shape)
        np.save(os.path.join(save_dir,per_path.split('/')[-1]),per_bs)
        #exit()




