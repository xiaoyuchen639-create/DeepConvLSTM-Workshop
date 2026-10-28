"""Create auditable tables, figures, and an Overleaf-ready report from actual runs."""
import csv
import json
import re
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_all import EXPERIMENTS
from preprocess import CLASS_NAMES

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'report'
def tex(s): return str(s).replace('_',r'\_').replace('%',r'\%')
def history(name):
    with (ROOT/'runs'/name/'history.csv').open() as f:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
def main():
    REPORT.mkdir(exist_ok=True); (REPORT/'figures').mkdir(exist_ok=True)
    results={}; configs={}
    for name in EXPERIMENTS:
        p=ROOT/'runs'/name/'metrics.json'
        if p.exists():
            results[name]=json.loads(p.read_text()); configs[name]=json.loads((p.parent/'config.json').read_text())
    fields=['run','weighted_f1','macro_f1','gesture_macro_f1','accuracy','best_epoch','best_val_weighted_f1','parameters','training_seconds','test_windows']
    with (ROOT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for n,m in results.items():w.writerow({'run':n,**{k:m[k] for k in fields[1:]}})
    with (ROOT/'per_class.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['run','class','precision','recall','f1','support'])
        for n in results:
            r=json.loads((ROOT/'runs'/n/'classification_report.json').read_text())
            for c in CLASS_NAMES:
                d=r[c]; w.writerow([n,c,d['precision'],d['recall'],d['f1-score'],d['support']])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
    if results:
        names=list(results); pos=np.arange(len(names)); fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(pos-.2,[results[n]['weighted_f1'] for n in names],.4,label='Weighted F1',color='#235789')
        ax.bar(pos+.2,[results[n]['macro_f1'] for n in names],.4,label='Macro F1',color='#e58f43')
        ax.set_xticks(pos,names,rotation=45,ha='right');ax.set_ylim(0,1);ax.legend();ax.set_ylabel('Test F1')
        fig.tight_layout();fig.savefig(REPORT/'figures/comparison.pdf');plt.close(fig)
    for part,names in {'reference':['reference'],'A':['reference','A_cnn','A_gru'],'B':['reference','B_conv0','B_conv1','B_conv2'],'C':['reference','C_w20','C_w42','C_w60'],'D':['D_control','D_no_dropout','D_no_weight_decay','D_no_augmentation'],'E':['reference','E_warmup'],'F':['reference','F_focal']}.items():
        if part!='reference' and sum(n in results for n in names)<2:
            (REPORT/f'figures/curves_{part}.pdf').unlink(missing_ok=True)
            continue
        fig,axes=plt.subplots(1,2,figsize=(9,2.8)); found=False
        for n in names:
            if n not in results:continue
            h=history(n); found=True
            axes[0].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label=n)
            axes[1].plot([r['epoch'] for r in h],[r['val_weighted_f1'] for r in h],label=n)
        if found:
            axes[0].set_ylabel('Training objective');axes[1].set_ylabel('Validation weighted F1')
            for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=6)
            fig.tight_layout();fig.savefig(REPORT/f'figures/curves_{part}.pdf')
        plt.close(fig)
    ref=results.get('reference')
    if ref:
        cm=np.loadtxt(ROOT/'runs/reference/confusion_matrix.csv',delimiter=',')
        norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(18),range(18));ax.set_yticks(range(18),CLASS_NAMES,fontsize=7)
        ax.set_xlabel('Predicted class ID');ax.set_ylabel('True class');fig.colorbar(im,ax=ax,label='Row fraction')
        fig.tight_layout();fig.savefig(REPORT/'figures/confusion_reference.pdf');plt.close(fig)
    complete=len(results)==len(EXPERIMENTS)
    status='All 15 runs completed.' if complete else f'WORK IN PROGRESS: {len(results)} of 15 runs completed. Pending entries are not experimental results.'
    rows=[]
    for n in EXPERIMENTS:
        if n in results:
            m=results[n]; rows.append(f"{tex(n)} & {m['weighted_f1']:.4f} & {m['macro_f1']:.4f} & {m['gesture_macro_f1']:.4f} & {m['best_epoch']} & {m['parameters']/1e6:.2f} & {m['training_seconds']/60:.1f} \\")
        else:rows.append(tex(n)+r' & \multicolumn{6}{c}{Pending} \\')
    # Avoid backslash escaping ambiguity in generated table row endings.
    rows=[r.rstrip('\\')+r'\\' for r in rows]
    def comparison(names,control='reference'):
        if control not in results:return 'This comparison is pending.'
        parts=[]
        for n in names:
            if n not in results:continue
            m=results[n]; b=results[control]
            parts.append(f"{tex(n)} achieved weighted F1 {m['weighted_f1']:.4f} and macro F1 {m['macro_f1']:.4f}; changes relative to {tex(control)} were {(m['weighted_f1']-b['weighted_f1'])*100:+.2f} and {(m['macro_f1']-b['macro_f1'])*100:+.2f} percentage points, respectively.")
        return ' '.join(parts) or 'This comparison is pending.'
    def fig(name,caption,width='.95'):
        if not (REPORT/'figures'/f'{name}.pdf').exists():return ''
        return '\n'+r'\begin{figure}[htbp]\centering'+'\n'+r'\includegraphics[width='+width+r'\linewidth]{figures/'+name+'}\n'+r'\caption{'+caption+r'}\end{figure}'+'\n'
    reftext='The reference experiment is pending.' if not ref else f"The reference network has {ref['parameters']:,} trainable parameters. Its selected checkpoint is epoch {ref['best_epoch']}, with validation weighted F1 {ref['best_val_weighted_f1']:.4f}. Test weighted F1 is {ref['weighted_f1']:.4f}, macro F1 {ref['macro_f1']:.4f}, and gesture-only macro F1 {ref['gesture_macro_f1']:.4f}. Training took {ref['training_seconds']/60:.1f} minutes."
    # Objective stability summaries use the same CE validation loss even for focal-trained models.
    stability=[]
    for n in ['reference','E_warmup']:
        if n in results:
            h=history(n); v=np.array([r['val_ce_loss'] for r in h[:10]])
            stability.append(f"For {tex(n)}, over the first ten epochs the mean absolute consecutive change in validation CE was {np.abs(np.diff(v)).mean():.4f}, and the mean pre-clipping gradient norm was {np.mean([r['gradient_norm_mean'] for r in h[:10]]):.3f}.")
    raretext='The focal-loss comparison is pending.'
    if 'F_focal' in results and ref:
        base=json.loads((ROOT/'runs/reference/classification_report.json').read_text()); focal=json.loads((ROOT/'runs/F_focal/classification_report.json').read_text())
        rare=[c for c in CLASS_NAMES[1:] if base[c]['support']<100]
        if rare:
            raretext=f"There are {len(rare)} gesture classes with fewer than 100 test windows. Their mean F1 changed from {np.mean([base[c]['f1-score'] for c in rare]):.4f} to {np.mean([focal[c]['f1-score'] for c in rare]):.4f}. This subset is defined by test support for descriptive analysis only; it is not used to tune training."
    doc=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,amsmath,longtable}
\usepackage[colorlinks=true,urlcolor=blue,citecolor=blue,linkcolor=blue]{hyperref}
\usepackage{caption}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\title{Workshop 2: A Systematic Ablation Study of DeepConvLSTM}
\author{ELEC 872}
\date{October 2026}
\begin{document}
\maketitle
\begin{abstract}
We investigate architecture, temporal context, regularization, learning-rate warm-up, and focal loss for 18-class wearable activity recognition on OPPORTUNITY. Fifteen controlled runs cover the workshop's required configurations, the additional configurations shown in its tables, and a separate regularization control. The implementation, recording-level split, preprocessing, and checkpoint selection are documented to make comparisons auditable. Numerical conclusions refer to one fixed random seed and a common 30-epoch budget.
\end{abstract}
\textbf{Execution status:} STATUS
\section{Task and reproducibility}
The workshop asks whether the components of DeepConvLSTM improve recognition from 113 sensor channels sampled at 30 Hz. There are 17 gestures and a Null class. We implement the workshop's named scripts independently because the supplied four-page PDF contains neither source files nor a source-code link. The architecture and channel selection are checked against the authors' public implementation~\cite{ordonez,code}. This is a documented reconstruction rather than a claim to have executed an unavailable instructor codebase.

The accompanying repository includes download, preprocessing, model, training, evaluation, experiment orchestration, and summarization scripts. Each run stores its configuration, software/device information, source hashes, epoch history, best checkpoint, predictions, confusion matrix, and per-class report. Actual run outputs are the sole source of every number in this report.
\section{Experimental protocol}
\subsection{Data preparation and split}
We use the official UCI OPPORTUNITY archive~\cite{uci}. The original authors' channel exclusion indices select 113 sensor features; the gesture target is original zero-based column 249. Linear interpolation is performed separately within each recording. Leading missing values and entirely missing channels are set to zero. Per-channel minimum and maximum are fitted on training recordings only, followed by normalization and clipping to $[0,1]$.

Training uses S1-Drill and S1-ADL1--5, plus S2/S3-Drill and S2/S3-ADL1--2. Validation uses S2-ADL3 and S3-ADL3. Test uses S2/S3-ADL4--5. S4 is excluded. This is a within-participant recording-held-out evaluation, not an unseen-participant evaluation. No window crosses a recording boundary and no overlapping windows are randomly split between training and evaluation.

The reference window is 24 samples (0.8 s), with stride 12 samples (0.4 s). A window receives the gesture label of its final sample. Null windows are retained. The stride and endpoint-label rule remain fixed when window length changes. Different window lengths therefore have slightly different counts and endpoint grids; these are recorded in each run. Offline interpolation uses later observations in the same recording and should not be interpreted as a strictly causal deployment procedure.
\subsection{Model and training}
The reference model applies four valid temporal convolutions, each with 64 filters of size $5\times1$ and ReLU, followed by two 128-unit LSTM layers and an 18-way linear classifier. Convolutional filters operate across time while preserving sensor positions. The sequence is explicitly rearranged into (batch, time, filter $\times$ sensor) order. Prediction uses the last recurrent output. Dropout is 0.5 before recurrent layers and before classification.

All runs use RMSProp, target learning rate 0.001, smoothing coefficient 0.9, batch size 100, seed 42, and 30 epochs. Gradient norm clipping at 5 is applied consistently. The checkpoint with the highest validation weighted F1 is selected; test data never select epochs or hyperparameters. Each run starts from newly initialized weights. The GPU is an NVIDIA GeForce RTX 3060 laptop GPU with 6 GB memory. Exact PyTorch and device details are stored in the run configuration.

Normalization fitted on training data, independent window states, dropout placement, and the finite training budget are explicit implementation choices. Exact numerical reproduction of the original paper is not asserted. The workshop's approximate reference scores are context, not values copied into our result table.
\subsection{Metrics}
For class $c$, $F_{1,c}=2P_cR_c/(P_c+R_c)$, with zero for an undefined score. Macro F1 averages the 18 class scores equally; weighted F1 weights them by true support. Gesture-only macro F1 averages classes 1--17 while still counting confusions with Null. We report all three because frequent Null windows can conceal poor rare-gesture performance. Per-class precision, recall, F1 and support are supplied in \texttt{per\_class.csv}.
\section{Results overview}
\begin{table}[htbp]\centering\small
\caption{Actual test results. W-F1: weighted; M-F1: macro; G-F1: gesture-only macro. Parameters in millions and training time in minutes. Pending cells indicate unfinished runs.}
\begin{tabular}{lrrrrrr}\toprule
Run & W-F1 & M-F1 & G-F1 & Epoch & Params & Min.\\\midrule
ROWS
\bottomrule\end{tabular}\end{table}
OVERVIEW
\section{Part 0: Reference reproduction}
REFTEXT
REFCURVES
\section{Part A: Is recurrence useful?}
The CNN variant retains the four convolutions and replaces the recurrent stack with two 128-unit dense ReLU layers. The GRU variant replaces both LSTMs with 128-unit GRUs. All other settings match the reference. The CNN flattens the convolutional output, so its parameter count differs; this comparison concerns practical architectures rather than parameter-matched causal attribution.

ATEXT

A higher or lower score at one seed is evidence for this fixed protocol only. Differences can reflect optimization and parameter count as well as the recurrent cell's representational capacity.
ACURVES
\section{Part B: Do the convolutions matter?}
We train variants with zero, one, and two convolutional layers, retaining the two LSTMs. The zero-layer case directly presents 113 raw channels to the recurrent stack. The workshop prose requires one and two layers; zero layers is additionally included because it appears in the table.

BTEXT

Removing valid convolutions also lengthens the sequence presented to the LSTM and changes its input dimension and parameter count. Accordingly, this ablation measures the total effect of convolution depth, not an isolated feature-extraction mechanism.
BCURVES
\section{Part C: Temporal window length}
We evaluate 20, 42, and 60 samples, corresponding to approximately 0.67, 1.4, and 2.0 s. The prose requests 20 and 60, while 42 appears in the table and is included for completeness. Four kernel-5 convolutions reduce these to 4, 26, and 44 recurrent time steps, compared with 8 for the reference. Longer windows increase context and computation without changing recurrent parameter count.

CTEXT
CCURVES
\section{Part D: Regularization}
The paper-style reference has dropout 0.5, zero weight decay, and no augmentation. Turning off the latter two again would produce duplicate configurations. To make the requested ablations meaningful, we introduce a separately labelled D\_control with dropout 0.5, weight decay $10^{-4}$, and Gaussian training jitter with standard deviation 0.01 on normalized inputs. Jittered values are clipped to $[0,1]$. Each D variant removes exactly one measure from that control; D comparisons must use D\_control rather than the paper-style reference. Validation and test inputs are never augmented.

DTEXT

Training objectives and validation trajectories help diagnose overfitting, but dropout and training jitter make training loss less directly comparable across configurations. The test F1 comparisons are the primary outcome.
DCURVES
\section{Part E: Learning-rate warm-up}
The warm-up run uses learning rate $\eta_e=0.001\min(e/5,1)$ at epoch $e$. The optimizer, target learning rate, batch size, and epoch budget remain unchanged. Warm-up changes early optimization and the total learning-rate exposure within the fixed budget.

ETEXT

STABILITY

We use early validation CE changes and pre-clipping gradient norms as descriptive stability indicators. Lower fluctuations do not automatically imply better generalization, and a single trajectory cannot establish statistical robustness.
ECURVES
\section{Part F: Focal loss and class imbalance}
We replace cross-entropy with unweighted multiclass focal loss~\cite{focal},
\[\mathcal L=-\frac1N\sum_i(1-p_{i,y_i})^{2}\log p_{i,y_i}.\]
The focusing parameter is fixed at $\gamma=2$ and no class-specific alpha weighting or resampling is added. This isolates the focal modulation. Log-softmax implements the loss stably; a test verifies that $\gamma=0$ equals cross-entropy.

FTEXT

RARETEXT

Raw training objectives for focal loss and cross-entropy have different scales and must not be compared as if they were the same loss. Validation CE, class-wise F1 and macro F1 permit a common comparison. Rare-class gains can coexist with a weighted-F1 decline.
FCURVES
\section{Limitations and conclusions}
The results describe one seed, one recording split, and 30 epochs per model. They do not establish statistical significance or full convergence, and the split does not measure transfer to unseen people. Equal epoch budgets also do not imply equal compute budgets. The comparison should be repeated over multiple seeds before a general architectural recommendation is made.

CONCLUSION

The workshop labels Part 0 and Parts A--F as 10\% each; the supplied file does not specify the remaining assessment allocation, a report page limit, or a submission deadline. This report covers every experimental configuration listed and documents the extra D control.
\appendix
\section{Reference confusion matrix}
CONFUSION
\section{Reproduction commands}
\begin{verbatim}
pip install -r requirements.txt
python download_data.py
python preprocess.py
python -m pytest -q
python run_all.py --epochs 30
python summarize.py
python evaluate.py runs/reference/best.pt --split test
\end{verbatim}
For a CUDA installation, install a PyTorch build appropriate to the local driver using the official PyTorch instructions. Open \texttt{main.tex} from the report ZIP in Overleaf and compile with pdfLaTeX. Generated figures are vector PDFs. All experiment options are enumerated in \texttt{run\_all.py}. Completed runs are skipped on restart; incomplete runs restart from initialization, and source hashes identify the implementation used.
\begin{thebibliography}{9}
\bibitem{ordonez} F. J. Ord\'o\~nez and D. Roggen, ``Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition,'' \emph{Sensors}, 16(1), 115, 2016. \url{https://doi.org/10.3390/s16010115}.
\bibitem{code} STRCWearlab, \emph{DeepConvLSTM: Original authors' implementation}. \url{https://github.com/STRCWearlab/DeepConvLSTM}. Accessed October 2026. Channel selection was checked against \texttt{preprocess\_data.py}.
\bibitem{uci} D. Roggen et al., \emph{OPPORTUNITY Activity Recognition}, UCI Machine Learning Repository, 2010. \url{https://doi.org/10.24432/C5M027}. Dataset licensed CC BY 4.0.
\bibitem{focal} T.-Y. Lin et al., ``Focal Loss for Dense Object Detection,'' \emph{ICCV}, 2017. \url{https://doi.org/10.1109/ICCV.2017.324}.
\end{thebibliography}
\end{document}
'''
    conclusion='The ablation suite is still running; conclusions will be generated only after all runs complete.'
    if complete:
        bestw=max(results,key=lambda n:results[n]['weighted_f1']);bestm=max(results,key=lambda n:results[n]['macro_f1'])
        conclusion=f"Within this fixed protocol, {tex(bestw)} has the highest test weighted F1 ({results[bestw]['weighted_f1']:.4f}), while {tex(bestm)} has the highest macro F1 ({results[bestm]['macro_f1']:.4f}). These are descriptive rankings after evaluation, not a test-selected deployment recommendation. The per-part comparisons above show whether each proposed change improved the reference or the dedicated D control."
    replacements={'STATUS':status,'ROWS':'\n'.join(rows),'OVERVIEW':fig('comparison','Test weighted and macro F1 for completed runs.'),'REFTEXT':reftext,'REFCURVES':fig('curves_reference','Reference training objective and validation weighted F1.'),'ATEXT':comparison(['A_cnn','A_gru']),'BTEXT':comparison(['B_conv0','B_conv1','B_conv2']),'CTEXT':comparison(['C_w20','C_w42','C_w60']),'DTEXT':comparison(['D_no_dropout','D_no_weight_decay','D_no_augmentation'],'D_control'),'ETEXT':comparison(['E_warmup']),'FTEXT':comparison(['F_focal']),'STABILITY':' '.join(stability),'RARETEXT':raretext,'CONCLUSION':conclusion,'CONFUSION':fig('confusion_reference','Reference test confusion matrix normalized within each true class. IDs follow the class order in the dataset manifest.','.8')}
    for p in ['A','B','C','D','E','F']:replacements[p+'CURVES']=fig('curves_'+p,f'Part {p}: training objectives and validation weighted F1.')
    doc=re.sub(r'\b(?:'+ '|'.join(re.escape(k) for k in replacements)+r')\b',lambda m:replacements[m.group(0)],doc)
    if (REPORT/'onepage.tex').exists():
        (ROOT/'tmp').mkdir(exist_ok=True)
        (ROOT/'tmp/full_report.tex').write_text(doc,encoding='utf-8')
        (REPORT/'main.tex').write_text((REPORT/'onepage.tex').read_text(encoding='utf-8'),encoding='utf-8')
    else:
        (REPORT/'main.tex').write_text(doc,encoding='utf-8')
    (REPORT/'README.txt').write_text('Upload this ZIP as an Overleaf project. Main document: main.tex. Compiler: pdfLaTeX. Status: '+status+'\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'Overleaf_Report.zip','w',zipfile.ZIP_DEFLATED) as z:
        if (REPORT/'onepage.tex').exists():
            for name in ['main.tex','README.txt']:z.write(REPORT/name,name)
        else:
            for p in REPORT.rglob('*'):
                if p.is_file() and p.suffix in ['.tex','.pdf','.txt'] and p.name!='main.pdf':z.write(p,p.relative_to(REPORT))
    (ROOT/'status.json').write_text(json.dumps({'completed':list(results),'pending':[n for n in EXPERIMENTS if n not in results],'complete':complete},indent=2))
    print(status)
if __name__=='__main__': main()
