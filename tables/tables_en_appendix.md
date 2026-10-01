# Appendix A. Full result tables (submission version)

Tables 2–9 below give the complete matrices behind the figures in the main text. Protocol: best.pt re-evaluation; validation and held-out test reported separately; three seeds (42/7/2024), mean ± sample standard deviation (n−1); parameter counts measured from `best.pt`. A difference is called significant (sig.) only when |Δ| > 2·σ_pooled with σ_pooled = √((σ₁²+σ₂²)/2); otherwise n.s. (inside seed noise).

## Table 2. Main ablation matrix (GC10-DET, 10 classes, 640 px / batch 16)

| Cell | Description | val mAP50-95 | val mAP50 | test mAP50-95 | test vs baseline | Params | Epochs |
|---|---|---|---|---|---|---|---|
| v_base | YOLOv8s baseline | 0.3269±0.0046 | 0.6390±0.0121 | 0.3505±0.0138 | +0.0000 n.s. | 11,139,470 | 200 |
| v_p2 | +P2 head | 0.3019±0.0040 | 0.6370±0.0038 | 0.3187±0.0125 | -0.0318 sig. | 10,639,576 | 200 |
| v_dcsa | +DCSA | 0.3139±0.0113 | 0.6429±0.0094 | 0.3292±0.0093 | -0.0213 n.s. | 11,154,301 | 200 |
| v_wiou | +WIoU v3 | 0.3290±0.0065 | 0.6540±0.0035 | 0.3454±0.0085 | -0.0051 n.s. | 11,139,470 | 200 |
| v_p2dcsawiou | +P2+DCSA+WIoU | 0.2967±0.0065 | 0.6308±0.0011 | 0.3124±0.0062 | -0.0381 sig. | 10,654,407 | 200 |
| v_full | above + $w_c$=1.0 (full stack) | 0.2970±0.0145 | 0.6326±0.0205 | 0.3156±0.0120 | -0.0349 sig. | 10,654,407 | 200 |
| v_full_cpw0p5 | full stack, $w_c$=0.5 | 0.2940±0.0023 | 0.6339±0.0069 | 0.3117±0.0068 | -0.0388 sig. | 10,654,407 | 200 |
| v_p1_b8 | +P1 head (b8) | 0.3001±0.0028 | 0.6374±0.0174 | 0.3130±0.0098 | -0.0375 sig. | 10,026,264 | 200 |
| v_base_b8 | baseline (b8) | 0.3242±0.0023 | 0.6450±0.0091 | 0.3446±0.0050 | -0.0059 n.s. | 11,139,470 | 200 |
| v_base_s320_b8 | baseline @320 (b8) | 0.3027±0.0086 | 0.6231±0.0121 | 0.3233±0.0108 | -0.0272 sig. | 11,139,470 | 200 |
| v_base_s960_b8 | baseline @960 (b8) | 0.3226±0.0073 | 0.6486±0.0117 | 0.3377±0.0069 | -0.0128 n.s. | 11,139,470 | 200 |
| v_resample | resampling | 0.2962±0.0058 | 0.6253±0.0164 | 0.3180±0.0108 | -0.0325 sig. | 10,654,407 | 200 |
| v_focal | Focal BCE | 0.2690±0.0098 | 0.5547±0.0066 | 0.2843±0.0150 | -0.0662 sig. | 10,654,407 | 200 |
| v_base_e300 | baseline, 300 ep | 0.3192±0.0009 | 0.6427±0.0077 | 0.3501±0.0050 | -0.0004 n.s. | 11,139,470 | 300 |
| v_full_e300 | full stack, 300 ep | 0.2914±0.0037 | 0.6116±0.0058 | 0.3041±0.0021 | -0.0464 sig. | 10,654,407 | 300 |

Note: Parameters are measured from `best.pt`. The `test` column is the held-out partition (345 images). Cells suffixed `e300` use a 300-epoch budget (training-budget robustness check).

## Table 5. Second benchmark (NEU-DET, 6 classes, 640 px / batch 16)

| Cell | Description | val mAP50-95 | val mAP50 | test mAP50-95 | test vs baseline | Params | Epochs |
|---|---|---|---|---|---|---|---|
| v_base | YOLOv8s baseline | 0.4336±0.0143 | 0.7830±0.0189 | 0.3615±0.0129 | +0.0000 n.s. | 11,137,922 | 200 |
| v_p2 | +P2 head | 0.4404±0.0027 | 0.7688±0.0090 | 0.3608±0.0144 | -0.0007 n.s. | 10,638,536 | 200 |
| v_dcsa | +DCSA | 0.4371±0.0057 | 0.7834±0.0069 | 0.3687±0.0029 | +0.0072 n.s. | 11,152,753 | 200 |
| v_wiou | +WIoU v3 | 0.4341±0.0065 | 0.7805±0.0142 | 0.3548±0.0149 | -0.0067 n.s. | 11,137,922 | 200 |
| v_p2dcsawiou | +P2+DCSA+WIoU | 0.4381±0.0049 | 0.7720±0.0111 | 0.3612±0.0101 | -0.0003 n.s. | 10,653,367 | 200 |
| v_full | above + $w_c$=1.0 (full stack) | 0.4318±0.0101 | 0.7774±0.0131 | 0.3596±0.0056 | -0.0019 n.s. | 10,653,367 | 200 |
| v_full_cpw0p5 | full stack, $w_c$=0.5 | 0.4399±0.0037 | 0.7791±0.0272 | 0.3597±0.0160 | -0.0018 n.s. | 10,653,367 | 200 |
| v_p1_b8 | +P1 head (b8) | 0.4354±0.0023 | 0.7779±0.0031 | 0.3591±0.0124 | -0.0024 n.s. | 10,025,736 | 200 |
| v_base_b8 | baseline (b8) | 0.4335±0.0088 | 0.7853±0.0203 | 0.3583±0.0042 | -0.0032 n.s. | 11,137,922 | 200 |
| v_base_s320_b8 | baseline @320 (b8) | 0.4476±0.0037 | 0.7774±0.0041 | 0.3545±0.0136 | -0.0070 n.s. | 11,137,922 | 200 |
| v_base_s960_b8 | baseline @960 (b8) | 0.4157±0.0059 | 0.7779±0.0072 | 0.3458±0.0092 | -0.0157 n.s. | 11,137,922 | 200 |
| v_full_s320_b8 | full stack @320 (b8) | 0.4349±0.0041 | 0.7848±0.0116 | 0.3464±0.0091 | -0.0151 n.s. | 10,653,367 | 200 |
| v_full_s960_b8 | full stack @960 (b8) | 0.4280±0.0055 | 0.7759±0.0067 | 0.3513±0.0050 | -0.0102 n.s. | 10,653,367 | 200 |
| v_resample | resampling | 0.4465±0.0108 | 0.7857±0.0113 | 0.3462±0.0145 | -0.0153 n.s. | 10,653,367 | 200 |
| v_focal | Focal BCE | 0.4325±0.0067 | 0.7492±0.0095 | 0.3690±0.0044 | +0.0075 n.s. | 10,653,367 | 200 |

Note: Same protocol and structure as Table 2; only the dataset and the number of classes differ (NEU-DET images are 200×200, upsampled to 640).

## Table 3. Input resolution and shallow detection heads (GC10-DET, batch 8)

| Configuration | val mAP50-95 | val mAP50 | test mAP50-95 | vs 640 baseline | Params |
|---|---|---|---|---|---|
| 640→320 | 0.3027±0.0086 | 0.6231±0.0121 | 0.3233±0.0108 | -0.0215 sig. | 11,139,470 |
| 640 (batch-8 reference) | 0.3242±0.0023 | 0.6450±0.0091 | 0.3446±0.0050 | +0.0000 n.s. | 11,139,470 |
| 640→960 | 0.3226±0.0073 | 0.6486±0.0117 | 0.3377±0.0069 | -0.0016 n.s. | 11,139,470 |
| +P1 head (stride 2) | 0.3001±0.0028 | 0.6374±0.0174 | 0.3130±0.0098 | -0.0241 sig. | 10,026,264 |
| +P2 head (b16) | 0.3019±0.0040 | 0.6370±0.0038 | 0.3187±0.0125 | -0.0223 sig. | 10,639,576 |

Note: differences are computed against the batch-8 reference row; the compute cost of each change is given in Table 7.

## Table 3b. Input resolution on the second benchmark (NEU-DET, s scale, batch 8)

| Configuration | Input (px) | val mAP50-95 | val mAP50 | test mAP50-95 | test vs 640, same arm | GFLOPs | Params |
|---|---|---|---|---|---|---|---|
| baseline | 320 | 0.4476±0.0037 | 0.7774±0.0041 | 0.3545±0.0136 | -0.0038 n.s. | 7.2 | 11,137,922 |
| baseline (640 reference) | 640 | 0.4335±0.0088 | 0.7853±0.0203 | 0.3583±0.0042 | — | 28.7 | 11,137,922 |
| baseline | 960 | 0.4157±0.0059 | 0.7779±0.0072 | 0.3458±0.0092 | -0.0125 n.s. | 64.5 | 11,137,922 |
| full stack | 320 | 0.4349±0.0041 | 0.7848±0.0116 | 0.3464±0.0091 | -0.0132 n.s. | 9.2 | 10,653,367 |
| full stack (640 reference) | 640 | 0.4318±0.0101 | 0.7774±0.0131 | 0.3596±0.0056 | — | 37.0 | 10,653,367 |
| full stack | 960 | 0.4280±0.0055 | 0.7759±0.0067 | 0.3513±0.0050 | -0.0083 n.s. | 83.2 | 10,653,367 |

Note: six NEU-DET s-scale runs under the identical protocol (200 epochs, batch 8, one split, seeds 42/7/2024); the 320 and 960 rows of the full stack were trained after the main matrix with the same code, same split and same seeds. GFLOPs and parameter counts are exact properties of the exported models and are comparable across batches. The ONNX Runtime CPU column of Table 7 deliberately omits these two cells: they were measured in a separate session, and §4.6 establishes that only same-session CPU comparisons are meaningful. The same-session contrast inside that separate batch is valid and instructive: the full stack costs 932 vs 3365 ms at 320 vs 960 px (+261%), while the held-out test mAP50-95 stays inside seed noise (0.3464 vs 0.3513).

## Table 4. Class-imbalance arms (GC10-DET, shared architecture P2+DCSA+WIoU)

| Arm | val mAP50-95 | val mAP50 | rare-class AP50 | mean AP50 of remaining classes | vs untreated |
|---|---|---|---|---|---|
| untreated | 0.2967±0.0065 | 0.6308±0.0011 | 0.2096 | 0.7360±0.0053 | — |
| inverse-frequency $w_c$=0.5 | 0.2940±0.0023 | 0.6339±0.0069 | 0.2021 | 0.7419±0.0086 | -0.0027 n.s. |
| inverse-frequency $w_c$=1.0 | 0.2970±0.0145 | 0.6326±0.0205 | 0.2135 | 0.7373±0.0105 | +0.0003 n.s. |
| resampling (sqrt-inv-freq) | 0.2962±0.0058 | 0.6253±0.0164 | 0.2151 | 0.7279±0.0145 | -0.0005 n.s. |
| Focal ($\gamma$=1.5, $\alpha$=0.25) | 0.2690±0.0098 | 0.5547±0.0066 | 0.0710 | 0.6756±0.0160 | -0.0277 sig. |

Note: rare classes are the two least frequent classes of the training split (GC10-DET: zhehen, yahen); all arms share one architecture and one seed set, so only the imbalance treatment varies.

## Table 8. Cross-scale comparison (2 datasets × 2 backbone scales, five shared configurations)


**Table 8a. Validation mAP50**

| Configuration | GC10-DET s (11.1 M) | GC10-DET n (3.0 M) | Δ (n − s) | NEU-DET s (11.1 M) | NEU-DET n (3.0 M) | Δ (n − s) |
|---|---|---|---|---|---|---|
| YOLOv8s baseline | 0.6390±0.0121 | 0.6417±0.0043 | +0.0027 n.s. | 0.7830±0.0189 | 0.7977±0.0073 | +0.0147 n.s. |
| +P2 head | 0.6370±0.0038 | 0.6387±0.0014 | +0.0017 n.s. | 0.7688±0.0090 | 0.7888±0.0132 | +0.0200 n.s. |
| +DCSA | 0.6429±0.0094 | 0.6370±0.0134 | -0.0059 n.s. | 0.7834±0.0069 | 0.7899±0.0037 | +0.0065 n.s. |
| +WIoU v3 | 0.6540±0.0035 | 0.6358±0.0026 | -0.0182 sig. | 0.7805±0.0142 | 0.7758±0.0052 | -0.0047 n.s. |
| above + $w_c$=1.0 (full stack) | 0.6326±0.0205 | 0.5802±0.0238 | -0.0524 sig. | 0.7774±0.0131 | 0.7888±0.0214 | +0.0114 n.s. |

**Table 8b. Validation mAP50-95**

| Configuration | GC10-DET s (11.1 M) | GC10-DET n (3.0 M) | Δ (n − s) | NEU-DET s (11.1 M) | NEU-DET n (3.0 M) | Δ (n − s) |
|---|---|---|---|---|---|---|
| YOLOv8s baseline | 0.3269±0.0046 | 0.3156±0.0007 | -0.0113 sig. | 0.4336±0.0143 | 0.4340±0.0043 | +0.0004 n.s. |
| +P2 head | 0.3019±0.0040 | 0.2984±0.0020 | -0.0035 n.s. | 0.4404±0.0027 | 0.4352±0.0101 | -0.0052 n.s. |
| +DCSA | 0.3139±0.0113 | 0.3024±0.0154 | -0.0115 n.s. | 0.4371±0.0057 | 0.4373±0.0047 | +0.0002 n.s. |
| +WIoU v3 | 0.3290±0.0065 | 0.3197±0.0034 | -0.0093 n.s. | 0.4341±0.0065 | 0.4336±0.0054 | -0.0005 n.s. |
| above + $w_c$=1.0 (full stack) | 0.2970±0.0145 | 0.2668±0.0111 | -0.0302 sig. | 0.4318±0.0101 | 0.4441±0.0075 | +0.0123 n.s. |

**Table 8c. Test mAP50-95**

| Configuration | GC10-DET s (11.1 M) | GC10-DET n (3.0 M) | Δ (n − s) | NEU-DET s (11.1 M) | NEU-DET n (3.0 M) | Δ (n − s) |
|---|---|---|---|---|---|---|
| YOLOv8s baseline | 0.3505±0.0138 | 0.3340±0.0032 | -0.0165 n.s. | 0.3615±0.0129 | 0.3678±0.0080 | +0.0063 n.s. |
| +P2 head | 0.3187±0.0125 | 0.3128±0.0067 | -0.0059 n.s. | 0.3608±0.0144 | 0.3620±0.0172 | +0.0012 n.s. |
| +DCSA | 0.3292±0.0093 | 0.3169±0.0174 | -0.0123 n.s. | 0.3687±0.0029 | 0.3756±0.0140 | +0.0069 n.s. |
| +WIoU v3 | 0.3454±0.0085 | 0.3440±0.0058 | -0.0014 n.s. | 0.3548±0.0149 | 0.3546±0.0062 | -0.0002 n.s. |
| above + $w_c$=1.0 (full stack) | 0.3156±0.0120 | 0.2907±0.0103 | -0.0249 sig. | 0.3596±0.0056 | 0.3682±0.0108 | +0.0086 n.s. |

Note: n = YOLOv8n, s = YOLOv8s; both scales use identical data splits, protocol and seeds. Only the five shared configurations were trained at the nano scale.

## Table 9. Imbalance treatments across two datasets (s scale)

| Arm | GC10-DET val mAP50-95 | GC10-DET val mAP50 | NEU-DET val mAP50-95 | NEU-DET val mAP50 | Verdict |
|---|---|---|---|---|---|
| untreated | 0.2967±0.0065 | 0.6308±0.0011 | 0.4381±0.0049 | 0.7720±0.0111 | reference |
| inverse-frequency $w_c$=0.5 | 0.2940±0.0023 | 0.6339±0.0069 | 0.4399±0.0037 | 0.7791±0.0272 | n.s. on both |
| inverse-frequency $w_c$=1.0 | 0.2970±0.0145 | 0.6326±0.0205 | 0.4318±0.0101 | 0.7774±0.0131 | n.s. on both |
| resampling (sqrt-inv-freq) | 0.2962±0.0058 | 0.6253±0.0164 | 0.4465±0.0108 | 0.7857±0.0113 | n.s. on both |
| Focal ($\gamma$=1.5, $\alpha$=0.25) | 0.2690±0.0098 | 0.5547±0.0066 | 0.4325±0.0067 | 0.7492±0.0095 | sig. on one |

Note: identical arm definitions on both datasets; GC10-DET rare classes are zhehen/yahen (11.4:1 train imbalance) and NEU-DET rare classes are pitted_surface/scratches (2.4:1).

## Table 7. Deployment battery (unified single-session batch; batch 1; RTX 4090 + Xeon Gold 6430)

| Group | Cell | Params | GFLOPs | FP32 p50 (ms) | FP16 p50 (ms) | ONNX-CPU p50 (ms) | GPU mem (MB) | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| GC10-DET s | v_base | 11,139,470 | 28.7 | 5.54±0.60 | 5.86±0.20 | 886.8±81.3 | 193.2 | 21.5 |
| GC10-DET s | v_p2 | 10,639,576 | 37.0 | 6.38±0.20 | 7.25±0.25 | 1534.2±114.8 | 219.5 | 20.6 |
| GC10-DET s | v_dcsa | 11,154,301 | 28.7 | 6.69±0.23 | 7.53±0.04 | 968.6±150.8 | 193.2 | 21.5 |
| GC10-DET s | v_wiou | 11,139,470 | 28.7 | 5.36±0.05 | 6.12±0.02 | 795.9±99.1 | 193.2 | 21.5 |
| GC10-DET s | v_p2dcsawiou | 10,654,407 | 37.0 | 8.55±1.12 | 9.46±1.22 | 1404.9±104.1 | 219.6 | 20.7 |
| GC10-DET s | v_full | 10,654,407 | 37.0 | 7.54±0.05 | 8.30±0.06 | 1797.2±360.1 | 219.6 | 20.7 |
| GC10-DET s | v_full_cpw0p5 | 10,654,407 | 37.0 | 8.30±1.26 | 8.41±0.18 | 1466.2±49.9 | 219.6 | 20.7 |
| GC10-DET s | v_p1_b8 | 10,026,264 | 40.2 | 5.81±0.56 | 6.53±0.66 | 2628.9±60.0 | 240.4 | 19.7 |
| GC10-DET s | v_base_b8 | 11,139,470 | 28.7 | 4.86±0.14 | 5.64±0.05 | 829.8±55.5 | 193.1±0.1 | 21.5 |
| GC10-DET s | v_base_s320_b8 | 11,139,470 | 7.2 | 5.05±0.08 | 5.73±0.08 | 369.8±59.5 | 179.7 | 21.5 |
| GC10-DET s | v_base_s960_b8 | 11,139,470 | 64.5 | 5.37±0.74 | 6.08±0.83 | 1498.3±96.9 | 253.3 | 21.5 |
| GC10-DET s | v_resample | 10,654,407 | 37.0 | 8.16±0.04 | 9.10±0.04 | 1568.1±112.2 | 219.6 | 20.7 |
| GC10-DET s | v_focal | 10,654,407 | 37.0 | 8.55±0.97 | 9.37±1.02 | 1600.2±176.0 | 219.6 | 20.7 |
| GC10-DET s | v_base_e300 | 11,139,470 | 28.7 | 5.28±0.46 | 6.09±0.53 | 770.8±114.6 | 193.2 | 21.5 |
| GC10-DET s | v_full_e300 | 10,654,407 | 37.0 | 7.85±0.28 | 8.59±0.41 | 1629.7±54.7 | 219.6 | 20.7 |
| GC10-DET nano | v_base | 11,139,470 | 8.2 | 4.78±0.57 | 5.84±0.60 | 597.9±98.0 | 159.0 | 6.0 |
| GC10-DET nano | v_p2 | 10,639,576 | 12.4 | 6.97±0.01 | 8.02±0.05 | 1298.5±94.9 | 168.6 | 5.9 |
| GC10-DET nano | v_dcsa | 11,154,301 | 8.2 | 7.24±0.16 | 8.05±0.22 | 636.0±50.8 | 159.0 | 6.0 |
| GC10-DET nano | v_wiou | 11,139,470 | 8.2 | 5.51±0.60 | 6.30±0.70 | 564.3±55.0 | 159.0 | 6.0 |
| GC10-DET nano | v_full | 10,654,407 | 12.4 | 7.73±0.94 | 8.73±1.01 | 1270.1±112.5 | 168.6 | 5.9 |
| NEU-DET s | v_base | 11,137,922 | 28.7 | 5.21±0.14 | 5.80±0.13 | 780.9±137.3 | 193.3 | 21.5 |
| NEU-DET s | v_p2 | 10,638,536 | 37.0 | 6.54±0.05 | 7.44±0.07 | 1534.2±110.7 | 213.4 | 20.7 |
| NEU-DET s | v_dcsa | 11,152,753 | 28.7 | 6.71±0.24 | 7.39±0.27 | 1002.1±7.9 | 193.3 | 21.5 |
| NEU-DET s | v_wiou | 11,137,922 | 28.7 | 5.73±0.61 | 6.49±0.65 | 704.1±103.3 | 193.3 | 21.5 |
| NEU-DET s | v_p2dcsawiou | 10,653,367 | 37.0 | 8.16±0.10 | 8.89±0.19 | 1735.2±50.8 | 213.4 | 20.8 |
| NEU-DET s | v_full | 10,653,367 | 37.0 | 8.05±0.16 | 8.87±0.13 | 1664.5±116.3 | 213.4 | 20.8 |
| NEU-DET s | v_full_cpw0p5 | 10,653,367 | 37.0 | 7.95±0.19 | 8.71±0.17 | 1665.7±58.2 | 213.4 | 20.8 |
| NEU-DET s | v_p1_b8 | 10,025,736 | 40.2 | 5.75±0.13 | 6.44±0.19 | 2700.2±199.9 | 240.4 | 20.0 |
| NEU-DET s | v_base_b8 | 11,137,922 | 28.7 | 4.65±0.05 | 5.30±0.03 | 832.1±51.9 | 193.2±0.2 | 21.5 |
| NEU-DET s | v_base_s320_b8 | 11,137,922 | 7.2 | 5.52±0.66 | 6.11±0.67 | 331.3±56.8 | 179.7 | 21.5 |
| NEU-DET s | v_base_s960_b8 | 11,137,922 | 64.5 | 5.34±0.63 | 5.64±0.07 | 1365.3±112.2 | 253.3 | 21.6 |
| NEU-DET s | v_resample | 10,653,367 | 37.0 | 8.61±0.89 | 9.50±1.03 | 1535.7±57.9 | 213.4 | 20.8 |
| NEU-DET s | v_focal | 10,653,367 | 37.0 | 8.06±0.09 | 8.88±0.05 | 1935.9±253.9 | 213.4 | 20.8 |
| NEU-DET nano | v_base | 11,137,922 | 8.2 | 5.33±0.07 | 5.68±0.60 | 533.7±53.8 | 159.0 | 6.0 |
| NEU-DET nano | v_p2 | 10,638,536 | 12.4 | 5.94±0.66 | 6.74±0.74 | 1099.1±97.2 | 168.6 | 6.0 |
| NEU-DET nano | v_dcsa | 11,152,753 | 8.2 | 6.66±0.71 | 7.42±0.79 | 662.1±59.1 | 159.0 | 6.0 |
| NEU-DET nano | v_wiou | 11,137,922 | 8.2 | 4.92±0.57 | 5.56±0.61 | 536.9±54.0 | 159.0 | 6.0 |
| NEU-DET nano | v_full | 10,653,367 | 12.4 | 8.02±0.55 | 8.71±0.81 | 1229.4±59.1 | 168.6 | 6.0 |

Note: all four groups were measured back-to-back in one session so that the columns are mutually comparable; absolute values are affected by host load and must not be compared across sessions (a same-architecture control inside this batch differs by ~11% on the CPU column). FP16 is slower than FP32 for these small models at batch 1, so no half-precision speed-up is claimed.
