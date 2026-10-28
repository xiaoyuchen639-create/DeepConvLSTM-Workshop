"""Recording-wise OPPORTUNITY preparation; never split overlapping windows randomly."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
LABEL_IDS = [0,406516,406517,404516,404517,406520,404520,406505,404505,406519,404519,406511,404511,406508,404508,408512,407521,405506]
CLASS_NAMES = ['Null','Open Door 1','Open Door 2','Close Door 1','Close Door 2','Open Fridge','Close Fridge','Open Dishwasher','Close Dishwasher','Open Drawer 1','Close Drawer 1','Open Drawer 2','Close Drawer 2','Open Drawer 3','Close Drawer 3','Clean Table','Drink from Cup','Toggle Switch']

def sensor_columns():
    # Original authors' 113-channel selection, expressed in original zero-based indices.
    excluded = set(i for a,b in [(46,50),(59,63),(72,76),(85,89),(98,102),(134,243),(244,249)] for i in range(a,b))
    return [i for i in range(250) if i not in excluded][1:114]

def fill_missing(x):
    x = x.copy()
    t = np.arange(len(x))
    for j in range(x.shape[1]):
        valid = np.isfinite(x[:,j])
        if valid.any():
            first = t[valid][0]
            x[:,j] = np.interp(t,t[valid],x[valid,j])
            x[:first,j] = 0
        else:
            x[:,j] = 0
    return x

def split_recordings():
    train = [f'S1-{a}' for a in ['Drill','ADL1','ADL2','ADL3','ADL4','ADL5']]
    train += [f'S{s}-{a}' for s in [2,3] for a in ['Drill','ADL1','ADL2']]
    return {'train':train,'val':['S2-ADL3','S3-ADL3'],'test':[f'S{s}-ADL{a}' for s in [2,3] for a in [4,5]]}

def main():
    archive = ROOT/'data/raw/opportunity.zip'
    out = ROOT/'data/processed'
    out.mkdir(parents=True,exist_ok=True)
    splits = split_recordings()
    arrays = {}
    with zipfile.ZipFile(archive) as z:
        for name in sum(splits.values(),[]):
            raw = np.loadtxt(io.BytesIO(z.read(f'OpportunityUCIDataset/dataset/{name}.dat')),dtype=np.float32)
            x = fill_missing(raw[:,sensor_columns()])
            labels = raw[:,249].astype(np.int64)
            unknown = set(np.unique(labels))-set(LABEL_IDS)
            if unknown: raise ValueError(unknown)
            y = np.array([LABEL_IDS.index(int(v)) for v in labels],dtype=np.int64)
            arrays[name]=(x,y)
            print(name,x.shape,flush=True)
    # Train-only extrema avoid using validation/test statistics.
    lo = np.min(np.stack([arrays[n][0].min(0) for n in splits['train']]),0)
    hi = np.max(np.stack([arrays[n][0].max(0) for n in splits['train']]),0)
    scale = np.maximum(hi-lo,1e-8)
    for n,(x,y) in arrays.items():
        np.savez(out/f'{n}.npz',x=np.clip((x-lo)/scale,0,1).astype(np.float32),y=y)
    np.savez(out/'normalization.npz',minimum=lo,maximum=hi,columns=sensor_columns())
    manifest={'splits':splits,'class_names':CLASS_NAMES,'label_ids':LABEL_IDS,'normalization':'train-only min-max; clip [0,1]','imputation':'within-recording linear interpolation; leading/all-missing values zero','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'recording_samples':{n:len(y) for n,(x,y) in arrays.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
