import torch
from model import NvidiaNet,FullyLSTM,LSTMNvidiaNet,LSTM
from dataset import Blendshape_dataset
from torch.utils.data.dataloader import DataLoader,Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os


num_epoch=100
batch_size=64
learning_rate=0.001
best_loss=10000
checkpoint_path='/data/xtx/yanghan/thirdtool/Audio2BS/checkpoint_dir_0809_only_val_FULLLSTM/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    #pass

#exit()
# train_loader=DataLoader(Blendshape_dataset(feature_file='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val_feature_blendshape',
#                                            target_file='/data/xtx/yanghan/thirdtool/data/AIWIN/train/arkit.csv',
#                                            target_dir='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val'),batch_size=batch_size,shuffle=True,num_workers=4)

val_loader=DataLoader(Blendshape_dataset(feature_file='/data/xtx/yanghan/thirdtool/data/AIWIN/train/feature_blendshape_dir_val',
                              target_file='/data/xtx/yanghan/thirdtool/data/AIWIN/train/arkit.csv',
                              target_dir='/data/xtx/yanghan/thirdtool/data/AIWIN/train/audio2face_data_for_val'),batch_size=64,shuffle=True,num_workers=4)

criterion=nn.MSELoss()
#model=NvidiaNet()
#model=LSTM()
model=FullyLSTM()
#model=LSTMNvidiaNet()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#
# for epoch in range(num_epoch):
#     train_loss=0
#     for index,(feature,target) in enumerate(tqdm(train_loader)):
#         #print('feature.shape',feature.shape)
#         feature=feature.to(device)
#         target=target.to(device)
#         output=model(feature)
#         loss=criterion(output,target)
#         #print('-----------------------loss:',loss)
#
#         train_loss+=loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     train_loss/=len(train_loader)
#     #print('epoch {} train_loss {}'.format(epoch,train_loss))
#
#     model.eval()
#     eval_loss=0
#     for feature_val,target_val in tqdm(val_loader):
#         feature_val=feature_val.to(device)
#         target_val=target_val.to(device)
#         output_val=model(feature_val)
#         loss_val=criterion(output_val,target_val)
#         eval_loss+=loss_val
#     eval_loss/=len(val_loader)
#     print('epoch {} | train loss {} | eval loss {}'.format(epoch,train_loss,eval_loss))
#     is_best=eval_loss<best_loss
#
#     best_loss=min(eval_loss,best_loss)
#     if is_best:
#         torch.save({'epoch':epoch+1,
#                     'state_dict':model.state_dict(),
#                     'eval_loss':eval_loss},checkpoint_path+'FullLSTM_chechpoint-epoch{}-train_loss{}-eval_loss{}'.format(epoch+1,train_loss,eval_loss)+'pth.tar')

##################################################train_with_val#################################
for epoch in range(num_epoch):
    train_loss=0
    val_loss=0
    for index,dl in enumerate([val_loader]):
        for index,(feature,target) in enumerate(tqdm(dl)):
            #print('feature.shape',feature.shape)
            feature=feature.to(device)
            target=target.to(device)
            output=model(feature)
            loss=criterion(output,target)
            #print('-----------------------loss:',loss)
            if index==0:
                train_loss+=loss
            if index==1:
                val_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_loss/=len(val_loader)
    #val_loss/=len(val_loader)
    #print('epoch {} | train loss {} | val loss {}'.format(epoch,train_loss,val_loss))
    print('epoch {} | train loss {}'.format(epoch, train_loss))

    torch.save({'epoch':epoch+1,
                'state_dict':model.state_dict()
                },checkpoint_path+'0808_only_val_FullLSTM_chechpoint-epoch{}-train_loss{}'.format(epoch+1,train_loss)+'pth.tar')







