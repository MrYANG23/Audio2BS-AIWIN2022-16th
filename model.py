import torch
import numpy as np
import os
from torch import nn



def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class NvidiaNet(nn.Module):

    def __init__(self, num_blendshapes=52):
        super(NvidiaNet, self).__init__()
        # formant analysis network
        self.num_blendshapes = num_blendshapes
        self.formant = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )

        # articulation network
        self.articulation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1,1), stride=(2,1)),
            nn.ReLU()
        )

        # output network
        self.output = nn.Sequential(
            nn.Linear(256, 150),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(150, self.num_blendshapes)
            #nn.Linear(256, self.num_blendshapes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # (-1, channel, height, width)
        # or x = x.view(-1, 1, 64, 32)
        #print('x.shape',x.shape)
        #exit()
        # convolution
        #print('----------------------type(x)',x.dtype)
        x = self.formant(x)
        #print('--------------after_formant.shape',x.shape)
        #exit()
        x = self.articulation(x)
        #print('----------------articulation.shape',x.shape)
        #exit()
        # fully connected
        x = x.view(-1, num_flat_features(x))

        x = self.output(x)
        #print('---------------------final_x.shape',x.shape)
        return x

class FullyLSTM(nn.Module):

    def __init__(self, num_features=39, num_blendshapes=52):
        super(FullyLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=256, num_layers=2,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.out = nn.Linear(256*2, num_blendshapes)

    def forward(self, input):
        # self.rnn.flatten_parameters()
        output, _ = self.rnn(input)
        #print('output.shape',output.shape)
        output = self.out(output[:, -1, :])
        return output

class LSTM(nn.Module):

    def __init__(self, num_features=39, num_blendshapes=52):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=256, num_layers=2,
                        batch_first=True, dropout=0.5, bidirectional=False)
        self.out = nn.Linear(256, num_blendshapes)

    def forward(self, input):
        # self.rnn.flatten_parameters()
        output, _ = self.rnn(input)
        #print('output.shape',output.shape)
        output = self.out(output[:, -1, :])
        return output

class LSTMNvidiaNet(nn.Module):

    def __init__(self, num_blendshapes=52, num_emotions=16):
        super(LSTMNvidiaNet, self).__init__()

        self.num_blendshapes = num_blendshapes
        self.num_emotions = num_emotions

        # emotion network with LSTM
        self.emotion = nn.LSTM(input_size=39, hidden_size=128, num_layers=1,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(128*2, 150),
            nn.ReLU(),
            nn.Linear(150, self.num_emotions)
        )


        # formant analysis network
        self.formant = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )

        # articulation network
        self.conv1 = nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.conv2 = nn.Conv2d(256+self.num_emotions, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.conv5 = nn.Conv2d(256+self.num_emotions, 256, kernel_size=(2,1), stride=(2,1))
        self.relu = nn.ReLU()

        # output network
        self.output = nn.Sequential(
            nn.Linear(256+self.num_emotions, 150),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(150, self.num_blendshapes)
        )

    def forward(self, x):
        # extract emotion state

        #print('----------------------x[:,::2].shape',x[:,::2].shape)

        e_state, _ = self.emotion(x[:, ::2]) # input features are 2* overlapping

        e_state = self.dense(e_state[:, -1, :]) # last
        e_state = e_state.view(-1, self.num_emotions, 1, 1)

        x = torch.unsqueeze(x, dim=1)
        # convolution
        x = self.formant(x)

        #print('----------------x.shape',x.shape)
        #print('e_state.shape',e_state.shape)

        # conv+concat
        x = self.relu(self.conv1(x))
        #print('----------------x.shape', x.shape)
        x = torch.cat((x, e_state.repeat(1, 1, 13, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 7, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 4, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 2, 1)), 1)
        #print('----------------conv5(x).shape',x.shape)
        x = self.relu(self.conv5(x))
        x = torch.cat((x, e_state), 1)

        # fully connected
        x = x.view(-1, num_flat_features(x))
        x = self.output(x)

        return x
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == '__main__':
    a=torch.randn(8,25,39)
    print(a.dtype)
    model=NvidiaNet()
    nvidianet_param=count_param(model)
    print('nvidianet_param',nvidianet_param)
    model_LSTM=FullyLSTM()
    BiLSTM_param=count_param(model_LSTM)
    print('BiLSTM_param',BiLSTM_param)
    model_NvidiaLSTM=LSTMNvidiaNet()

    NvidiaLstm=count_param(model_NvidiaLSTM)
    print('NvidiaLSTM', NvidiaLstm)
    model_lstm=LSTM()
    lstm_param=count_param(model_lstm)
    print('lstm_param',lstm_param)

    # print(output_lstm.shape)
    #output_NvidiaLSTM=model_NvidiaLSTM(a)
    #print('------------------------output_NvidiaLSTM.shape',output_NvidiaLSTM.shape)

