# AUDIT-Derm: design-based audit certification for skin malignancy triage

Code and result files for the manuscript *Design-Based Audit Certification of Automated Benign
Decisions for Skin Malignancy Triage Under Compound Acquisition and Prevalence Shift*.

Everything sits in one flat folder. Run every command from inside it.

## Files

| File | Purpose |
|---|---|
| `load_harvest.py` | Decodes the ISIC thumbnail bundle (an 8-byte header length, then a JSON index, then the JPEG bytes) into `index.csv`. |
| `extract_features.py` | Runs the frozen CLIP ViT-B/32 image tower (ONNX, CPU) and writes `emb.npy`. |
| `prepare.py` | Builds cohorts, lesion-grouped splits, the paired-lesion split and the logistic head. Writes `cases.csv`. |
| `audit_derm.py` | Core method: transport, Mondrian candidates, audit design, AIPW increments, betting bounds, split certification and fixed-sequence testing. |
| `simulate.py` | Deployment simulator: scenarios, periods, delayed outcomes, all baselines and the controller. |
| `run_experiments.py` | E1 main comparison, E2 ablation, E3 sensitivity and E4 robustness. Writes `E*.csv`. |
| `aggregate.py`, `agg_sens.py` | Stream-level summaries, bootstrap CIs, Wilcoxon/Holm tests. Write `summary_raw.json` and `summary_sens.json`. |
| `analysis_transport.py` | E8: held-out pairs, prevalence estimation and shift attribution. |
| `subgroups.py` | E9: subgroup risks of the deployed thresholds. |
| `failure_dx.py` | E9b: missed-malignancy rate by diagnosis. |
| `coverage_check.py` | Monte Carlo validity check of the region-split bound. |
| `efficiency.py` | E6: timing. |
| `plots.py`, `diagram_lib.py`, `fig_arch.py` | Figures. The diagram helper checks for overlapping boxes, arrows that cross boxes and text that leaves the canvas. |

## Reproduce

```bash
pip install -r requirements.txt
export ISIC_BUNDLE=/path/to/shiftderm_isic_thumbs.bin   # ISIC thumbnails + metadata
export CLIP_ONNX=/path/to/clip_vit_b32_vision.onnx      # input pixel_values, output image_embeds
python load_harvest.py && python extract_features.py && python prepare.py
python run_experiments.py main ablation sens robust volume
python aggregate.py && python agg_sens.py && python analysis_transport.py && python subgroups.py
python failure_dx.py && python coverage_check.py && python efficiency.py
python plots.py transport tradeoff trajectory bounds budget subgroups dynamics && python fig_arch.py
```

The image bundle was assembled from the public ISIC archive API, using the HAM10000, BCN20000,
ISIC 2020 (MSKCC), HIBA, PAD-UFES-20 and MILK10k collections, as 256-pixel thumbnails. The
images themselves are not redistributed here. `cases.csv` lists every image ID together with its
split, condition, label and frozen-head logit, and `emb_fp16.npz` holds the frozen CLIP embeddings (stored as float16 to keep the archive small, so weighted-CP baseline figures can differ slightly in the last decimal), so every experiment can be rerun without the images (skip the first line of commands).

## Key settings

| Setting | Value |
|---|---|
| False-omission limit (alpha_F) | 0.05 |
| Missed-malignancy limit (alpha_M) | 0.10 |
| Error level (delta) | 0.10 |
| Audit budget | 0.40 |
| Audit floor | 0.20 |
| Window | 3 periods of 2,500 cases |
| Outcome delay | 1 period |
| Candidates | 40 |
| Minimum automation share (f_min) | 0.10 |
| Bets | J = 6, c = 0.95 |

All result files used in the manuscript tables are included (`E*.csv`, `*.json`).
