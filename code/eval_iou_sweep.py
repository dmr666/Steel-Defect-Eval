# NOTE: machine-specific paths were replaced with '.'/placeholders.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_iou_sweep.py —— review M6：per-IoU 阈值的 AP 曲线（解释 mAP50 饱和而 mAP50-95 未饱和）

对指定 cell × seed 的 best.pt，在 val/test 上导出预测（YOLO txt 格式），
再用自带的贪心匹配实现逐 IoU 阈值（0.50:0.05:0.95）算 AP，输出曲线数据。

用法（服务器上）：
  python eval_iou_sweep.py --runs ${DATA_ROOT}/neu/runs \
      --data ${DATA_ROOT}/neu/data/NEU-DET-split/neu_det.yaml \
      --cells v_base v_full --split test --out ${DATA_ROOT}/neu/iou_sweep
  # 自检：脚本会把算出的 mAP50-95 与同 run 的 perclass_ap.json（ultralytics 口径）对比并打印差值

依赖：ultralytics、numpy（服务器已有）。
"""
import argparse
import csv
import json
import os
import shutil
import statistics as st
import sys

import numpy as np

THRS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def register_custom_modules():
    """v_full / v_dcsa / v_focal 的 best.pt 里 pickled 的是 defect_train.py 定义的自定义层，
    pickle 记的是 __main__.C2f_DCSA（训练时脚本即 __main__）。本脚本作为新的 __main__
    必须先把同名类注册到 __main__，否则 torch.load 报
    "Can't get attribute 'C2f_DCSA' on <module '__main__'>"。"""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import defect_train as dt
    except Exception as e:
        print("[warn] 未能 import defect_train（%s）：仅 v_base 这类无自定义层的权重可加载" % e)
        return
    names = ("ChannelAttention", "SpatialAttention", "DCSA", "Bottleneck_DCSA",
             "C2f_DCSA", "FocalBCE")
    for n in names:
        if hasattr(dt, n):
            setattr(sys.modules["__main__"], n, getattr(dt, n))
    try:
        dt.install_dcsa_parse_patch(verbose=False)
    except Exception as e:
        print("[warn] install_dcsa_parse_patch 调用失败：%s" % e)
    print("已注册自定义层：%s" % ", ".join(n for n in names if hasattr(dt, n)))


def read_data_yaml(path):
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    root = d.get("path") or os.path.dirname(os.path.abspath(path))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(path)), root)
    d["_root"] = os.path.normpath(root)
    return d


def img_dir_of(d, split):
    v = d.get(split)
    if v is None:
        raise SystemExit("data yaml 里没有 split=%s" % split)
    p = v if os.path.isabs(v) else os.path.join(d["_root"], v)
    p = os.path.normpath(p)
    # split 目录可能是 <split>/images 布局（本项目服务器数据即此结构），
    # ultralytics 传目录不递归，直接传 <split> 会报 No images found
    sub = os.path.join(p, "images")
    return sub if os.path.isdir(sub) else p


def label_dir_of(img_dir):
    return os.path.normpath(img_dir.replace(os.sep + "images", os.sep + "labels"))


def load_gt(img_dir, names):
    """返回 {img_key: [(cls, x1,y1,x2,y2), ...]}，坐标为原图像素"""
    from PIL import Image
    ldir = label_dir_of(img_dir)
    gt = {}
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        key = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as im:
            W, H = im.size
        boxes = []
        lp = os.path.join(ldir, key + ".txt")
        if os.path.isfile(lp):
            for line in open(lp, encoding="utf-8"):
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        gt[key] = boxes
    return gt


def load_pred(txt_dir, names):
    """ultralytics predict(save_txt=True) 的输出：每张图一个 txt，行 = cls cx cy w h conf（归一化）"""
    from PIL import Image
    preds = {}
    for fn in sorted(os.listdir(txt_dir)):
        if not fn.endswith(".txt"):
            continue
        key = os.path.splitext(fn)[0]
        rows = []
        for line in open(os.path.join(txt_dir, fn), encoding="utf-8"):
            p = line.split()
            if len(p) < 6:
                continue
            rows.append((int(float(p[0])), float(p[5]), *map(float, p[1:5])))
        preds[key] = rows
    return preds


def iou_matrix(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def ap_at_iou(gt, preds, img_sizes, ncls, thr):
    """逐类 101 点插值 AP，返回 (mAP, per_class_AP)"""
    aps = {}
    for c in range(ncls):
        scores, tps, npos = [], [], 0
        for key, gboxes in gt.items():
            g = [b[1:] for b in gboxes if b[0] == c]
            npos += len(g)
            if key not in preds or key not in img_sizes:
                continue
            W, H = img_sizes[key]
            pr = [(s, ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
                  for (pc, s, cx, cy, w, h) in preds[key] if pc == c]
            pr.sort(key=lambda x: -x[0])
            if not pr:
                continue
            ious = iou_matrix([p[1] for p in pr], g)
            used = [False] * len(g)
            for i, (s, _) in enumerate(pr):
                j = int(np.argmax(ious[i])) if len(g) else -1
                if len(g) and ious[i][j] >= thr and not used[j]:
                    used[j] = True
                    scores.append(s)
                    tps.append(1)
                else:
                    scores.append(s)
                    tps.append(0)
        if npos == 0:
            continue
        order = np.argsort(-np.asarray(scores))
        tps = np.asarray(tps)[order]
        tp = np.cumsum(tps)
        fp = np.cumsum(1 - tps)
        rec = tp / (npos + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.linspace(0, 1, 101)
        aps[c] = float(np.mean(np.interp(idx, mrec, mpre)))
    if not aps:
        return 0.0, {}
    return float(np.mean(list(aps.values()))), aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cells", nargs="+", default=["v_base", "v_full"])
    ap.add_argument("--seeds", nargs="+", default=["42", "7", "2024"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    d = read_data_yaml(a.data)
    names = d["names"]
    ncls = len(names)
    img_dir = img_dir_of(d, a.split)
    gt = load_gt(img_dir, names)
    img_sizes = {}
    for k in gt:
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            f = os.path.join(img_dir, k + ext)
            if os.path.isfile(f):
                with Image.open(f) as im:
                    img_sizes[k] = im.size
                break
    print("split=%s 图像 %d 张，GT 框 %d 个，类别 %d" % (a.split, len(gt), sum(len(v) for v in gt.values()), ncls))

    register_custom_modules()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for cell in a.cells:
        for seed in a.seeds:
            for cand in ("%s_seed%s" % (cell, seed), "%s_e300_seed%s" % (cell, seed)):
                rdir = os.path.join(a.runs, cand)
                if os.path.isdir(rdir):
                    break
            else:
                print("skip %s seed%s（无目录）" % (cell, seed))
                continue
            w = os.path.join(rdir, "weights", "best.pt")
            if not os.path.isfile(w):
                print("skip %s（缺 best.pt）" % cand)
                continue
            pdir = os.path.join(a.out, "pred_%s_%s_%s" % (cell, seed, a.split))
            if os.path.isdir(pdir):
                shutil.rmtree(pdir)
            model = YOLO(w)
            # 预测来源必须与论文表格口径一致：用 ultralytics val(split) 自身的推理。
            # 同一权重下 model.predict() 与 val() 的 letterbox/NMS 结果不同。2026-10-02 实测
            # （GC10 v_base_seed42 / test）：val(save_txt) 2925 框 → 自算 AP50 0.6969（官方 0.6830）；
            # predict 4491 框 → 自算 AP50 0.5777，低 0.105，纯来源差异，与 AP 实现无关。
            vr = model.val(data=a.data, split=a.split, imgsz=a.imgsz, conf=a.conf, iou=0.7,
                           save_txt=True, save_conf=True, project=a.out,
                           name="pred_%s_%s_%s" % (cell, seed, a.split),
                           exist_ok=True, plots=False, verbose=False)
            txt_dir = os.path.join(str(vr.save_dir), "labels")
            preds = load_pred(txt_dir, names)
            ul_now = (float(vr.box.map50), float(vr.box.map))
            per_thr = {}
            for thr in THRS:
                m, _ = ap_at_iou(gt, preds, img_sizes, ncls, thr)
                per_thr[thr] = m
            m5095 = float(np.mean([per_thr[t] for t in THRS]))
            ref = None
            pc = os.path.join(rdir, "perclass_ap.json")
            if os.path.isfile(pc):
                j = json.load(open(pc, encoding="utf-8"))
                key = "test" if a.split == "test" else "val"
                ref = j.get(key, {}).get("mAP5095")
            res["%s_seed%s" % (cell, seed)] = {
                "per_iou": per_thr, "mAP50-95": m5095, "mAP50": per_thr[0.5],
                "ultralytics_ref": ref, "ul_val_mAP50": ul_now[0], "ul_val_mAP50-95": ul_now[1]}
            print("%-22s mAP50=%.4f mAP75=%.4f mAP95=%.4f | mAP50-95=%.4f | 同次val %.4f/%.4f | 归档ref %.4f Δ=%+.4f"
                  % (cand, per_thr[0.5], per_thr[0.75], per_thr[0.95], m5095,
                     ul_now[0], ul_now[1], ref if ref else -1, (m5095 - ref) if ref else 0.0))

    # 聚合（按 cell 求均值±标准差）
    agg = {}
    for cell in a.cells:
        ks = [k for k in res if k.startswith(cell + "_seed")]
        if not ks:
            continue
        agg[cell] = {("iou%.2f" % t): (st.mean([res[k]["per_iou"][t] for k in ks]),
                                      st.stdev([res[k]["per_iou"][t] for k in ks]) if len(ks) > 1 else 0.0)
                     for t in THRS}
    out = {"split": a.split, "per_run": res, "agg": agg, "thresholds": THRS}
    json.dump(out, open(os.path.join(a.out, "iou_sweep.json"), "w", encoding="utf-8"), indent=1)
    print("\n写出:", os.path.join(a.out, "iou_sweep.json"))
    for cell, d2 in agg.items():
        print("%-10s " % cell + " ".join("%.2f:%.3f" % (float(k[3:]), v[0]) for k, v in d2.items()))


if __name__ == "__main__":
    main()
