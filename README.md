# Workshop 2: DeepConvLSTM Ablation Study

An independently implemented, auditable solution to the supplied Workshop 2 specification. The PDF supplies script names but no source code or download link. This repository reconstructs the described experiment, with preprocessing/channel selection checked against the original authors' implementation. Results are measured locally, never taken from the workshop's illustrative scores.

## Execution status

All **15 experiments are complete**, each trained for **30 epochs** with seed 42. Read [results.csv](results.csv) for all test metrics and [per_class.csv](per_class.csv) for class-wise precision, recall, F1 and support.

- [Final report PDF: one page, exactly two tables](report/main.pdf)
- [LaTeX source](report/main.tex)
- [Overleaf project](https://www.overleaf.com/project/6ae0f4f540e95a9a613580d6)

The reference reaches weighted F1 **0.8792** and macro F1 **0.5576**. The 20-sample window has the highest measured weighted F1 (**0.8960**) and macro F1 (**0.6076**) in this single-seed suite. These are descriptive test results, not statistically established rankings.

## Setup and reproduction

Python 3.11 was used. Install a CUDA-enabled PyTorch wheel from the official PyTorch index if GPU training is desired; the generic requirements also support CPU training.

```bash
pip install -r requirements.txt
python download_data.py
python preprocess.py
python -m pytest -q
python run_all.py --epochs 30
python summarize.py
python evaluate.py runs/reference/best.pt --split test
```

`run_all.py` skips completed runs. An incomplete run restarts from initialization; it does not resume an optimizer checkpoint. Run the scripts from this repository. Raw data and large checkpoints are intentionally not committed.

## Protocol

- Input: 113 OPPORTUNITY channels; 18 gesture/Null classes; final-sample window label.
- Train: S1 Drill and ADL1–5; S2/S3 Drill and ADL1–2.
- Validation: S2/S3 ADL3. Test: S2/S3 ADL4–5. S4 is excluded.
- Entire recordings are assigned before windowing. Windows never cross recording boundaries.
- Missing values: recording-wise linear interpolation; leading/all-missing values zero.
- Normalization: train-only channel min/max, clipped to [0,1].
- Reference: 24 samples, stride 12; four 64-filter valid temporal convolutions with kernel 5; two 128-unit LSTMs; dropout 0.5.
- RMSProp: LR 0.001, alpha 0.9, batch 100; gradient clipping 5; seed 42; 30 epochs.
- Select the best validation weighted-F1 checkpoint; evaluate it once on test.
- Report weighted F1, macro F1, gesture-only macro F1, accuracy and per-class scores.

This is a within-participant recording-held-out evaluation, not unseen-participant generalization. The implementation is a reconstruction, not an exact execution of an unavailable instructor codebase. Dropout placement, normalization and a fixed epoch budget can differ from the paper. One seed does not establish statistical significance.

## Experiment matrix

| Part | Run | Single change from its control |
|---|---|---|
| 0 | reference | Paper-style reference: dropout .5, weight decay 0, no augmentation |
| A | A_cnn | Replace recurrent layers by two 128-unit dense layers |
| A | A_gru | Replace LSTM with GRU |
| B | B_conv0 | No convolutions (extra table configuration) |
| B | B_conv1 | One convolution |
| B | B_conv2 | Two convolutions |
| C | C_w20 | Window 20 / 0.67 seconds |
| C | C_w42 | Window 42 / 1.4 seconds (extra table configuration) |
| C | C_w60 | Window 60 / 2 seconds |
| D | D_control | Dedicated control: dropout .5, weight decay 1e-4, jitter sigma .01 |
| D | D_no_dropout | From D_control: dropout 0 |
| D | D_no_weight_decay | From D_control: weight decay 0 |
| D | D_no_augmentation | From D_control: jitter 0 |
| E | E_warmup | Linear warm-up over five epochs, same target LR |
| F | F_focal | Unweighted focal loss with gamma 2, no resampling |

D needs its own control because weight decay and augmentation are already disabled in the paper-style reference. Each requested D ablation changes one setting relative to D_control. This extra control brings the complete suite to 15 runs.

## Outputs

`runs/<name>/` contains configuration and source hashes, all 30 epochs of history, metrics, per-class report, confusion matrix, and compressed predictions. `best.pt` is saved locally; model weights and raw data are not included in this repository. `results.csv` and `per_class.csv` aggregate all completed runs. `report/main.tex` is the final one-page, two-table report; `report/main.pdf` is its compiled version. `report/figures/` contains supplementary experiment plots, separate from the one-page report. Compile `main.tex` using pdfLaTeX in Overleaf.

`python summarize.py` regenerates tables and plots. When `report/onepage.tex` exists, it preserves that final report template as `report/main.tex` and writes the extended analysis to `tmp/full_report.tex`; it does not replace the one-page submission with a longer document. Numerical edits to the one-page template must be checked against `results.csv`.

The report intentionally leaves personal authorship unspecified until student/group details are supplied.

## Sources

- Ordóñez and Roggen (2016), [DeepConvLSTM paper](https://doi.org/10.3390/s16010115).
- [Original authors' implementation](https://github.com/STRCWearlab/DeepConvLSTM), especially channel selection in `preprocess_data.py`.
- [Official OPPORTUNITY dataset](https://doi.org/10.24432/C5M027), CC BY 4.0.
- Lin et al. (2017), [Focal Loss](https://doi.org/10.1109/ICCV.2017.324).

No raw dataset is redistributed. Report all students who contributed and ensure each team member understands every experiment, as required by the workshop.
