# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_resampled_list.py - 生成"重采样"不平衡对照臂用的训练列表（Repeat Factor Sampling, LVIS 式）

做法（写进论文的公式）：
  重复因子 r_c = min(CAP, max(1, sqrt(N_max / N_c)))，N_c 为该类训练实例数，N_max 为最大类实例数
  （即"逆频平方根"重采样：稀有类样本被重复，且随稀有度按平方根增长，避免极端放大）
  每张图的重复次数 r = max_c r_c（取该图所有类中最大的 r_c）
  训练时按列表把同一张图喂 r 次 —— 这是"重采样"处理，与 cls_pw 加权互斥（该臂 cls_pw=0）

输出：
  <data>/train_resampled.txt          绝对路径列表（AutoDL 用，路径前缀可指定）
  <data>/train_resampled_local.txt    本地版（Windows 绝对路径，供本地冒烟）
  <data>/<stem>_resampled.yaml        与基础 yaml 同结构，只把 train 换成列表文件
  <data>/resample_report.json         公式参数 + 采样前后每类实例数/图数
"""
import argparse
import collections
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="数据集目录（含 train/images、split_stats.json）")
ap.add_argument("--stem", required=True, help="基础 yaml 名（不含扩展名），如 gc10_det")
ap.add_argument("--<server>-prefix", default="${DATA_ROOT}/gc10/data/GC10-DET-yolo",
                help="AutoDL 版列表的路径前缀")
ap.add_argument("--t", type=float, default=0.0, help="保留参数（不再使用阈值法）")
ap.add_argument("--cap", type=int, default=4, help="单图最大重复次数")
args = ap.parse_args()

D = args.data
stats = json.load(open(os.path.join(D, "split_stats.json"), encoding="utf-8"))
names = stats["names"]
inst = stats["stats"]["train"]["instances"]
total = sum(inst)
freq = [c / total for c in inst]
nmax = max(inst)
r_c = [min(float(args.cap), max(1.0, math.sqrt(nmax / c))) if c > 0 else 1.0 for c in inst]

imgdir = os.path.join(D, "train", "images")
files = sorted(os.listdir(imgdir))
times = collections.defaultdict(int)
inst_after = collections.Counter()
for fn in files:
    lab = open(os.path.join(D, "train", "labels", fn[:-4] + ".txt"), encoding="utf-8").read().split()
    rows = [lab[i:i + 5] for i in range(0, len(lab), 5)]
    cls = [int(r[0]) for r in rows] or [0]
    r = int(round(max(r_c[c] for c in cls)))
    times[fn] = max(1, r)
    for c in cls:
        inst_after[c] += times[fn]

lines_local = [os.path.join(imgdir, fn) for fn in files for _ in range(times[fn])]
lines_auto = ["%s/train/images/%s" % (args.<server>_prefix.rstrip("/"), fn) for fn in files for _ in range(times[fn])]
open(os.path.join(D, "train_resampled_local.txt"), "w", encoding="utf-8").write("\n".join(lines_local) + "\n")
open(os.path.join(D, "train_resampled.txt"), "w", encoding="utf-8").write("\n".join(lines_auto) + "\n")

base = open(os.path.join(D, args.stem + ".yaml"), encoding="utf-8").read()
out_yaml = []
for ln in base.split("\n"):
    if ln.startswith("train:"):
        out_yaml.append("train: train_resampled.txt")
    else:
        out_yaml.append(ln)
open(os.path.join(D, args.stem + "_resampled.yaml"), "w", encoding="utf-8").write("\n".join(out_yaml))

rep = {"method": "inverse-sqrt-frequency resampling (r_c = min(cap, sqrt(N_max/N_c)))",
       "t": args.t, "cap": args.cap, "N_max": nmax,
       "per_class_freq": {names[i]: round(freq[i], 5) for i in range(len(names))},
       "per_class_r": {names[i]: round(r_c[i], 3) for i in range(len(names))},
       "train_images_before": len(files), "train_images_after": len(lines_local),
       "instances_before": {names[i]: inst[i] for i in range(len(names))},
       "instances_after": {names[i]: inst_after.get(i, 0) for i in range(len(names))}}
json.dump(rep, open(os.path.join(D, "resample_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"逆频平方根重采样 cap={args.cap}（N_max={nmax}）")
print(f"  图数 {len(files)} → {len(lines_local)}（×{len(lines_local)/len(files):.2f}）")
print(f"  每类 r_c: {rep['per_class_r']}")
print(f"  实例数 before: {list(inst)}")
print(f"  实例数 after : {[inst_after.get(i,0) for i in range(len(names))]}")
print("  写出: train_resampled.txt / train_resampled_local.txt /",
      f"{args.stem}_resampled.yaml / resample_report.json")

# 列表统一成 LF 行尾（Linux 上更稳；实测 CRLF 也能被 splitlines 处理，但不留隐患）
for _f in ("train_resampled.txt", "train_resampled_local.txt"):
    _p = os.path.join(D, _f)
    _b = open(_p, "rb").read().replace(bytes([13, 10]), bytes([10]))
    open(_p, "wb").write(_b)
print("  行尾已统一为 LF")
