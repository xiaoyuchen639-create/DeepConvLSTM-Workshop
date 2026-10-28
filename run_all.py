"""Sequential, restartable experiment suite. A completed run is never silently overwritten."""
import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS={
 'reference':[],
 'A_cnn':['--model','cnn'],
 'A_gru':['--rnn','gru'],
 'B_conv0':['--n-conv','0'],
 'B_conv1':['--n-conv','1'],
 'B_conv2':['--n-conv','2'],
 'C_w20':['--window','20'],
 'C_w42':['--window','42'],
 'C_w60':['--window','60'],
 'D_control':['--weight-decay','0.0001','--augmentation','0.01'],
 'D_no_dropout':['--weight-decay','0.0001','--augmentation','0.01','--dropout','0'],
 'D_no_weight_decay':['--augmentation','0.01'],
 'D_no_augmentation':['--weight-decay','0.0001'],
 'E_warmup':['--warmup','5'],
 'F_focal':['--loss','focal'],
}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--only',nargs='*'); a=p.parse_args()
    root=Path(__file__).resolve().parent
    for name,options in EXPERIMENTS.items():
        if a.only and name not in a.only: continue
        if (root/'runs'/name/'metrics.json').exists():
            print('SKIP completed',name,flush=True); continue
        subprocess.run([sys.executable,'-u','train.py','--name',name,'--epochs',str(a.epochs),*options],cwd=root,check=True)
if __name__=='__main__': main()
