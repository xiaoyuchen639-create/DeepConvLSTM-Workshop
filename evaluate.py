import argparse
import json
from pathlib import Path
import torch
from data import load_split
from models import DeepConvLSTM
from train import predict

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); p.add_argument('--split',choices=['train','val','test'],default='test'); a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    c=torch.load(a.checkpoint,map_location=device,weights_only=False); cfg=c['config']
    model=DeepConvLSTM(**cfg).to(device); model.load_state_dict(c['state_dict'])
    x,y=load_split(a.split,cfg['window'],cfg['stride'],device,cfg['exclude_null'])
    print(json.dumps(predict(model,x,y)[0],indent=2))
