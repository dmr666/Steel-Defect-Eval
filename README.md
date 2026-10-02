# Steel-Defect-Eval

Code, per-seed metrics and raw deployment measurements for the manuscript

> **What Actually Helps Steel Surface Defect Detection at the Line Side?
> A Controlled Multi-Seed Evaluation of Capacity, Enhancement, and Imbalance Treatments**
> Lihua Dai (single author), Zhongke Shurong Technology (Zhengzhou) Co., Ltd.
> Submitted to *Signal, Image and Video Processing* (Springer).

This is a controlled **negative-result** study: instead of proposing another module, it fixes one
protocol and re-measures the habitual enhancement stack of the steel-surface-defect literature
(shallow detection heads, channel attention, IoU-type losses, class-imbalance treatments) on two
public benchmarks, two backbone scales and three seeds — 120 training runs in total — and prices
each configuration in deployment terms (parameters, GFLOPs, FP32/FP16, ONNX Runtime CPU latency,
GPU memory, model size).

## What is in here

| Directory | Contents |
|---|---|
| `code/` | Training, evaluation, aggregation, figure generation and the statistics script. `code/configs/` holds the model and dataset YAMLs used for every arm. |
| `metrics/` | **All 154 runs** (120 in the original matrix + 34 revision runs): `results.csv` (per-epoch), `args.yaml` (exact resolved configuration) and `perclass_ap.json` (per-class val/test AP at 0.5 and 0.5:0.95) per run, plus the aggregation tables (`_tables/`). Revision groups are the `*_scratch` ones (from-scratch baselines), the `*_scratch300` ones (300-epoch budget reinforcement) and `neu_s_e300` (300-epoch budget check). |
| `deploy/` | Raw per-run JSON of the deployment battery for the four benchmark groups, and the unified measurement log. |
| `tables/` | The final paper tables (English appendix + Chinese working version) in text/markdown form. |
| `MANIFEST.tsv` | Maps every run directory to its dataset/scale group. |

Run directory names are `variant_seed<N>`; variants are defined in `code/defect_train.py`.
`code/run_matrix_neu.sh`, `code/run_matrix_gc10.sh`, `code/run_nano.sh` and
`code/run_gc10_e300.sh` are the exact command sequences that produced the matrix.

## Revision additions (tag `v1.1-revision`)

Beyond the original matrix, the same layout also carries the revision controls:

| Group | Runs | What it is |
|---|---|---|
| `metrics/gc10_s`, `metrics/neu_s` | +12 | seeds 13 and 99 for the baseline, the full stack and Focal (five seeds for the key cells) |
| `metrics/gc10_s_scratch`, `gc10_nano_scratch`, `neu_s_scratch`, `neu_nano_scratch` | 12 | from-scratch baselines (structure YAML as weights, no COCO checkpoint), same protocol and budget — the reference for the pretraining-confound control |
| `metrics/neu_s_e300` | 6 | NEU-DET 300-epoch budget check |
| `metrics/*_scratch300` | 4 | from-scratch baselines trained for 300 epochs instead of 200 (seed 42), the budget reinforcement behind SI Table S13 |
| `iou_sweep/` | 2 | AP swept over IoU 0.50–0.95 for the baseline and the full stack on both benchmarks |

Code added: `code/eval_iou_sweep.py`, `code/run_review_extra_seeds.sh`, `code/run_review_scratch.sh`,
`code/run_review_neu_e300.sh`, `code/run_review_scratch_e300.sh`, `code/collect_review.sh`,
`code/reviewB_analyze.py`, `code/review_stats_v2.py`, `code/gen_si_tables_v2.py`.

Note on the IoU sweep: its absolute values come from predictions exported by the ultralytics validation
pass itself and reproduce the official held-out test mAP50-95 to within +0.012…+0.027 (systematic +0.02).
They are meant for threshold-to-threshold trend comparisons inside one consistent pipeline; the official
numbers under `metrics/` remain the reference for absolute values.

## Scope and honesty notes

* Positive findings are reported as bounded by this protocol; the paper explicitly does **not** claim
  that these modules are useless in general.
* Significance in the paper uses the pre-registered rule `|Δ| > 2 σ_pooled`. `code/review_stats_v2.py`
  additionally runs Welch two-sample t-tests with Holm adjustment across each comparison family
  (Supplementary Table S9). In the revision the baseline, the full stack and Focal were extended from
  3 to 5 seeds; under the stricter criterion the two GC10-DET effects (full stack, Focal) now survive as
  well, while on NEU-DET nothing becomes significant — stated in the paper rather than hidden.
* Deployment numbers for the CPU column were measured on a **server-class CPU (Xeon Gold 6430)**,
  i.e. a line-side rather than an embedded/ARM/NPU proxy. Values from different measurement sessions
  are not comparable (same-architecture cross-session spread ≈ 11%); only deterministic quantities
  (parameters, GFLOPs, model size) are used across sessions.

## Environment

* Python 3.10, PyTorch 2.1.2 + CUDA 11.8, Ultralytics 8.4.115, NumPy 1.26.4
* Single NVIDIA RTX 4090 (24 GB); ONNX Runtime CPU for the latency column
* Training: 640 px input, batch 16, SGD (lr0 0.01, cosine), warmup 3, 200 epochs, `degrees=10`,
  Mixup/Copy-Paste disabled, `patience=0`, `--final-test`; seeds 42/7/2024

## Datasets (not redistributed here)

* **GC10-DET** — 10 classes of cold-rolled strip defects, 2292 images (2048 x 1000). We use a fixed
  70/15/15 image split (1602/345/345) and a small-object-heavy distribution (25 % of instances are
  below 32 px at 640 px input).
* **NEU-DET** — 6 classes of hot-rolled strip defects, 1800 images (200 x 200). The public mirror's
  1620/180 train/test split is re-split into 1458 train / 162 val (balanced, 27 per class), keeping
  the 180-image test partition untouched.

Both datasets are public; download them from their original providers and lay them out as in
`code/configs/gc10_det.yaml` and `code/configs/neu_det.yaml` (paths in those YAMLs have been
replaced with placeholders). Conversion of GC10-DET to YOLO format and generation of the
sqrt-inverse-frequency resampling list are in `code/make_gc10_yolo.py` and
`code/make_resampled_list.py`.

## Reproducing

```bash
export DATA_ROOT=/path/to/your/data_root     # code/configs/*.yaml and the run scripts point here
python code/defect_train.py --variant v_base  --seed 42 --data $DATA_ROOT/neu/data/neu_det.yaml \
       --base-yaml code/configs/yolov8s-base.yaml --weights yolov8s.pt \
       --imgsz 640 --batch 16 --cls-pw 0 --epochs 200 --project $DATA_ROOT/runs --final-test
```

Per-seed evaluation on the held-out test partition writes `perclass_ap.json` next to each run;
`code/analyze_matrix.py <runs_dir>` rebuilds the matrix tables, and `code/review_stats.py`
recomputes the multiplicity-adjusted statistics and the per-seed rare-class AP50 used in Fig. 4(b).

## Not included

* Trained checkpoints (≈ 4 GB, 120 × `best.pt`) — available on request, or from the release attached
  to the published version.
* The manuscript source; the datasets themselves.

## Citation

```bibtex
@article{Dai2026SteelLineSide,
  author  = {Dai, Lihua},
  title   = {What Actually Helps Steel Surface Defect Detection at the Line Side?
             A Controlled Multi-Seed Evaluation of Capacity, Enhancement, and Imbalance Treatments},
  journal = {Signal, Image and Video Processing},
  year    = {2026},
  note    = {submitted}
}
```

## License

Code: MIT (see `LICENSE`). Metrics and deployment measurements (`metrics/`, `deploy/`):
CC BY 4.0.
