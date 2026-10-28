import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
from data import ROOT,load_split
from models import DeepConvLSTM,FocalLoss
from preprocess import CLASS_NAMES

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='reference')
    p.add_argument('--model',choices=['deepconvlstm','cnn'],default='deepconvlstm')
    p.add_argument('--rnn',choices=['lstm','gru'],default='lstm')
    for key,default in [('n-conv',4),('window',24),('stride',12),('filters',64),('kernel',5),('hidden',128),('batch-size',100),('epochs',30),('seed',42),('warmup',0)]: p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--dropout',type=float,default=.5)
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--augmentation',type=float,default=0.,help='Gaussian jitter sigma on normalized training inputs only')
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--loss',choices=['ce','focal'],default='ce')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--exclude-null',action='store_true')
    return p

@torch.no_grad()
def predict(model,x,y,batch_size=400):
    model.eval(); out=[]; losses=[]
    for i in range(0,len(x),batch_size):
        logits=model(x[i:i+batch_size]); out.append(logits.argmax(1).cpu().numpy())
        losses.append(float(torch.nn.functional.cross_entropy(logits,y[i:i+batch_size],reduction='sum')))
    pred=np.concatenate(out); actual=y.cpu().numpy()
    metrics={'accuracy':accuracy_score(actual,pred),'weighted_f1':f1_score(actual,pred,labels=list(range(18)),average='weighted',zero_division=0),'macro_f1':f1_score(actual,pred,labels=list(range(18)),average='macro',zero_division=0),'gesture_macro_f1':f1_score(actual,pred,labels=list(range(1,18)),average='macro',zero_division=0),'ce_loss':sum(losses)/len(y)}
    return metrics,actual,pred

def run(args):
    cfg=vars(args); out=ROOT/'runs'/args.name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists(): raise RuntimeError(f'Completed run exists: {out}')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    x,y=load_split('train',args.window,args.stride,args.device,args.exclude_null)
    vx,vy=load_split('val',args.window,args.stride,args.device,args.exclude_null)
    model=DeepConvLSTM(**cfg).to(args.device)
    opt=torch.optim.RMSprop(model.parameters(),lr=args.lr,alpha=.9,weight_decay=args.weight_decay)
    criterion=FocalLoss() if args.loss=='focal' else torch.nn.CrossEntropyLoss()
    metadata={'config':cfg,'parameters':sum(p.numel() for p in model.parameters()),'train_windows':len(y),'val_windows':len(vy),'torch':torch.__version__,'device':torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device,'dataset_manifest':json.loads((ROOT/'data/processed/manifest.json').read_text()),'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['train.py','models.py','data.py','preprocess.py']}}
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    print(args.name,metadata['parameters'],len(y),len(vy),flush=True)
    best=-1.; best_epoch=0; start=time.perf_counter()
    with (out/'history.csv').open('w',newline='') as f:
        fields=['epoch','lr','train_loss','val_ce_loss','val_weighted_f1','val_macro_f1','gradient_norm_mean','seconds']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.; norms=0.; batches=0
            lr=args.lr*min(1,epoch/args.warmup) if args.warmup else args.lr
            for group in opt.param_groups: group['lr']=lr
            order=torch.randperm(len(y),device=args.device)
            for ids in order.split(args.batch_size):
                xb=x[ids]
                if args.augmentation: xb=(xb+torch.randn_like(xb)*args.augmentation).clamp(0,1)
                opt.zero_grad(set_to_none=True)
                loss=criterion(model(xb),y[ids])
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step(); total+=float(loss.detach())*len(ids); norms+=float(norm); batches+=1
            val,_,_=predict(model,vx,vy)
            row=dict(epoch=epoch,lr=lr,train_loss=total/len(y),val_ce_loss=val['ce_loss'],val_weighted_f1=val['weighted_f1'],val_macro_f1=val['macro_f1'],gradient_norm_mean=norms/batches,seconds=time.perf_counter()-start)
            writer.writerow(row); f.flush()
            if val['weighted_f1']>best:
                best=val['weighted_f1']; best_epoch=epoch
                torch.save({'state_dict':model.state_dict(),'config':cfg,'epoch':epoch},out/'best.pt')
            print(args.name,epoch,'loss',round(row['train_loss'],4),'val_wF1',round(best,4),'seconds',round(row['seconds']),flush=True)
    elapsed=time.perf_counter()-start
    ckpt=torch.load(out/'best.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    del x,y,vx,vy
    tx,ty=load_split('test',args.window,args.stride,args.device,args.exclude_null)
    metrics,actual,pred=predict(model,tx,ty)
    metrics.update(best_epoch=best_epoch,best_val_weighted_f1=best,training_seconds=elapsed,test_windows=len(ty),parameters=metadata['parameters'],seed=args.seed)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    report=classification_report(actual,pred,labels=list(range(18)),target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    (out/'classification_report.json').write_text(json.dumps(report,indent=2))
    np.savetxt(out/'confusion_matrix.csv',confusion_matrix(actual,pred,labels=list(range(18))),fmt='%d',delimiter=',')
    np.savez_compressed(out/'predictions.npz',actual=actual,predicted=pred)
    print(json.dumps(metrics),flush=True)
    return metrics

if __name__=='__main__': run(parser().parse_args())
