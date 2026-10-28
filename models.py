import torch
from torch import nn

class DeepConvLSTM(nn.Module):
    def __init__(self,window=24,n_conv=4,filters=64,kernel=5,hidden=128,dropout=.5,rnn='lstm',model='deepconvlstm',channels=113,**kwargs):
        super().__init__()
        length=window-n_conv*(kernel-1)
        if length<1: raise ValueError('Window too short for valid convolutions')
        conv=[]
        for i in range(n_conv):
            conv += [nn.Conv2d(1 if i==0 else filters,filters,(kernel,1)),nn.ReLU()]
        self.conv=nn.Sequential(*conv)
        features=channels*(filters if n_conv else 1)
        self.model=model
        self.drop=nn.Dropout(dropout)
        if model=='cnn':
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(length*features,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,18))
        else:
            cell=nn.LSTM if rnn=='lstm' else nn.GRU
            self.rnn1=cell(features,hidden,batch_first=True)
            self.rnn2=cell(hidden,hidden,batch_first=True)
            self.head=nn.Linear(hidden,18)

    def forward(self,x):
        x=self.conv(x.unsqueeze(1))
        # Preserve (time, filter, channel) semantics, never reshape time into batch.
        x=x.permute(0,2,1,3).contiguous().flatten(2)
        if self.model=='cnn': return self.head(x)
        x,_=self.rnn1(self.drop(x))
        x,_=self.rnn2(self.drop(x))
        return self.head(self.drop(x[:,-1]))

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.):
        super().__init__(); self.gamma=gamma
    def forward(self,logits,target):
        logpt=torch.log_softmax(logits,dim=1).gather(1,target[:,None]).squeeze(1)
        return (-((1-logpt.exp())**self.gamma)*logpt).mean()
