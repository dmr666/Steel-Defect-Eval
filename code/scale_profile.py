# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两数据集的"缺陷尺度"画像：折算到 640 输入下的目标像素尺寸（决定浅层检测头 P2/P1 是否有物理依据）
输出：控制台表 + _verify/scale_profile.json
"""
import collections
import json
import math
import os
from PIL import Image

ROOT = r".\data"
NEU = os.path.join(ROOT, "NEU-DET-split")
GC10 = os.path.join(ROOT, "GC10-DET-yolo")
NEU_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def profile(root, split, names, imgsz=640):
    idir, ldir = os.path.join(root, split, "images"), os.path.join(root, split, "labels")
    sizes, per_cls, n = [], collections.defaultdict(list), 0
    for fn in os.listdir(idir):
        with Image.open(os.path.join(idir, fn)) as im:
            W, H = im.size
        for line in open(os.path.join(ldir, fn[:-4] + ".txt"), encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            w_px = float(p[3]) * W
            h_px = float(p[4]) * H
            # 等比缩放（letterbox）到 imgsz 后的等效像素
            s = imgsz / max(W, H)
            eq = math.sqrt((w_px * s) * (h_px * s))
            sizes.append(eq)
            per_cls[names[c]].append(eq)
            n += 1
    sizes.sort()
    q = lambda a, p: (sorted(a)[int(len(a) * p)] if a else 0)
    out = {"images_px": None, "instances": n,
           "eq640_median": q(sizes, 0.5), "eq640_p10": q(sizes, 0.1),
           "eq640_p90": q(sizes, 0.9), "frac_below_16px": sum(1 for v in sizes if v < 16) / max(n, 1),
           "frac_below_32px": sum(1 for v in sizes if v < 32) / max(n, 1),
           "per_class_median_eq640": {k: q(v, 0.5) for k, v in sorted(per_cls.items())},
           "per_class_n": {k: len(v) for k, v in sorted(per_cls.items())}}
    return out


res = {}
for tag, root, names in (("NEU-DET(热轧)", NEU, NEU_NAMES), ("GC10-DET(冷轧)", GC10, None)):
    with Image.open(os.path.join(root, "train", "images", os.listdir(os.path.join(root, "train", "images"))[0])) as im:
        px = im.size
    if names is None:
        names = json.load(open(os.path.join(root, "split_stats.json"), encoding="utf-8"))["names"]
    for split in ("train", "test"):
        r = profile(root, split, names)
        r["images_px"] = px
        res[f"{tag}/{split}"] = r
        print(f"\n=== {tag} / {split} （原始 {px[0]}x{px[1]}，折算到 640 输入的等效目标边长）===")
        print(f"  实例 {r['instances']} | 中位 {r['eq640_median']:.1f} px | 10分位 {r['eq640_p10']:.1f} | "
              f"90分位 {r['eq640_p90']:.1f}")
        print(f"  <16px 占比 {r['frac_below_16px']*100:.1f}% | <32px 占比 {r['frac_below_32px']*100:.1f}%")
        print("  每类中位尺寸:", {k: round(v, 1) for k, v in r["per_class_median_eq640"].items()})

outp = os.path.join(r".\_verify", "scale_profile.json")
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n写出:", outp)
