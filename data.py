import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

def make_windows(x,y,window=24,stride=12):
    starts=np.arange(0,len(x)-window+1,stride)
    return x[starts[:,None]+np.arange(window)[None,:]], y[starts+window-1]

def load_split(split,window=24,stride=12,device='cpu',exclude_null=False):
    folder=ROOT/'data/processed'
    manifest=json.loads((folder/'manifest.json').read_text())
    xs,ys=[],[]
    for name in manifest['splits'][split]:
        d=np.load(folder/f'{name}.npz')
        x,y=make_windows(d['x'],d['y'],window,stride)
        xs.append(x); ys.append(y)
    x,y=np.concatenate(xs),np.concatenate(ys)
    if exclude_null:
        mask=y!=0; x,y=x[mask],y[mask]
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)
