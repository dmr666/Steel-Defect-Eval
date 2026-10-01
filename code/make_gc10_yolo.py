# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_gc10_yolo.py - 把 GC10-DET（VOC XML，2048x1000 灰度）转成论文用的 YOLO 检测数据集

设计（与 NEU-DET 同纪律）：
  · 类别：按 XML 的 object/name 的数字前缀归并成 10 类（**逐框用 name**；<folder> 只是该图的"主类文件夹"，
          多目标图里附加目标的类别可能与 <folder> 不同，实测确认）
      0 chongkong(冲孔) 1 hanfeng(焊缝) 2 yueyawan(月牙湾) 3 shuiban(水斑) 4 youban(油斑)
      5 siban(丝斑)     6 yiwu(夹杂)   7 yahen(压痕)     8 zhehen(折痕)  9 yaozhe(腰折)
  · 脏数据处置（都记录在 stats 里，写进论文数据小节）：
      - `10_yaozhed`（131 实例，疑似 `10_yaozhe` 的拼写变体）→ 归并到 9 yaozhe
      - `<name>d</name>`（1 实例，非类别）→ 丢弃该框
      - 无对应 XML 的图片（18 张）→ 不进数据集，列名留档
  · 划分：固定 split seed=0，按"图像主类（第一个目标的类）"分层 70/15/15，每类 val/test 至少 2 张
  · 输出：GC10-DET-yolo/{train,val,test}/{images,labels} + gc10_det.yaml + split_stats.json + 统计报告
"""
import io
import json
import os
import re
import collections
import zipfile
from PIL import Image

ZIP = r"D:\BaiduNetdiskDownload\GC10-DET.zip"
ROOT = r".\data"
OUT = os.path.join(ROOT, "GC10-DET-yolo")
SEED = 0
RATIO = (0.70, 0.15, 0.15)
MIN_VAL = 2
MIN_TEST = 2
NAME_MAP = {                      # XML 名 -> (类号, 规范名)
    "1_chongkong": (0, "chongkong"), "2_hanfeng": (1, "hanfeng"),
    "3_yueyawan": (2, "yueyawan"), "4_shuiban": (3, "shuiban"),
    "5_youban": (4, "youban"), "6_siban": (5, "siban"),
    "7_yiwu": (6, "yiwu"), "8_yahen": (7, "yahen"),
    "9_zhehen": (8, "zhehen"), "10_yaozhe": (9, "yaozhe"),
    "10_yaozhed": (9, "yaozhe"),   # 拼写变体，归并
}
NAMES = ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
         "siban", "yiwu", "yahen", "zhehen", "yaozhe"]


def parse_xml(s):
    folder = re.search(r"<folder>(.*?)</folder>", s)
    folder = folder.group(1).strip() if folder else "?"
    w = int(re.search(r"<width>(\d+)</width>", s).group(1))
    h = int(re.search(r"<height>(\d+)</height>", s).group(1))
    fn = re.search(r"<filename>(.*?)</filename>", s).group(1)
    objs = []
    for om in re.finditer(r"<object>(.*?)</object>", s, re.S):
        o = om.group(1)
        nm = re.search(r"<name>(.*?)</name>", o).group(1).strip()
        bb = re.search(r"<bndbox>(.*?)</bndbox>", o, re.S).group(1)
        x1 = float(re.search(r"<xmin>(.*?)</xmin>", bb).group(1))
        y1 = float(re.search(r"<ymin>(.*?)</ymin>", bb).group(1))
        x2 = float(re.search(r"<xmax>(.*?)</xmax>", bb).group(1))
        y2 = float(re.search(r"<ymax>(.*?)</ymax>", bb).group(1))
        objs.append((nm, x1, y1, x2, y2))
    return fn, folder, w, h, objs


def main():
    z = zipfile.ZipFile(ZIP)
    members = [x for x in z.namelist() if x.lower().endswith(".jpg")]
    img_index = collections.defaultdict(list)
    for x in members:
        img_index[os.path.basename(x)].append(x)
    dup = {k: v for k, v in img_index.items() if len(v) > 1}
    xmls = [x for x in z.namelist() if x.endswith(".xml")]
    print(f"zip: {len(members)} 图（唯一名 {len(img_index)}，重名 {len(dup)} 组）, {len(xmls)} 标注")

    def pick_member(fn, folder):
        cands = img_index.get(fn, [])
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        for m in cands:                     # 重名：用 XML 的 <folder> 选
            if m.split('/')[0] == str(folder):
                return m
        return None

    samples, dropped_name, no_xml, bad_box, multi = {}, collections.Counter(), [], 0, 0
    inst_per_class = collections.Counter()
    dup_unresolved = []
    member_of = {}
    for x in xmls:
        fn, folder, w, h, objs = parse_xml(z.read(x).decode("utf-8", "replace"))
        m = pick_member(fn, folder)
        if m is None:
            if len(img_index.get(fn, [])) > 1:
                dup_unresolved.append(fn)
            continue
        member_of[fn] = m
        keep = []
        for nm, x1, y1, x2, y2 in objs:
            if nm not in NAME_MAP:
                dropped_name[nm] += 1
                continue
            cls, _ = NAME_MAP[nm]
            x1, y1 = max(0.0, min(x1, w)), max(0.0, min(y1, h))
            x2, y2 = max(0.0, min(x2, w)), max(0.0, min(y2, h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                bad_box += 1
                continue
            keep.append((cls, x1, y1, x2, y2))
            inst_per_class[cls] += 1
        if not keep:
            bad_box += 1
            continue
        if len(keep) > 1:
            multi += 1
        samples[fn] = (w, h, keep)
    for fn in img_index:
        if fn not in samples:
            no_xml.append(fn)

    print(f"可用图 {len(samples)} 张 | 多目标图 {multi} 张 | 丢框(非法/未知类) {bad_box} | "
          f"无标注图 {len(no_xml)} | 未知类名 {dict(dropped_name)}")
    print("每类实例数:", {NAMES[k]: v for k, v in sorted(inst_per_class.items())})

    # 分层划分：按主类（第一个目标）
    by_cls = collections.defaultdict(list)
    for fn, (w, h, keep) in samples.items():
        by_cls[keep[0][0]].append(fn)
    import random
    rng = random.Random(SEED)
    split = {"train": [], "val": [], "test": []}
    for cls, files in sorted(by_cls.items()):
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_val = max(MIN_VAL, int(round(n * RATIO[1])))
        n_test = max(MIN_TEST, int(round(n * RATIO[2])))
        if n_val + n_test > n - 1:
            n_val, n_test = max(1, n // 5), max(1, n // 5)
        split["val"] += files[:n_val]
        split["test"] += files[n_val:n_val + n_test]
        split["train"] += files[n_val + n_test:]

    # 写盘（先清掉上一轮产物，避免残留空目录干扰统计）
    import shutil
    for sp in ("train", "val", "test"):
        d = os.path.join(OUT, sp)
        if os.path.isdir(d):
            shutil.rmtree(d)
    for sp in split:
        os.makedirs(os.path.join(OUT, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT, sp, "labels"), exist_ok=True)
    stats = {"names": NAMES, "split_seed": SEED, "ratio": RATIO, "stats": {}, "source": "GC10-DET (VOC XML)"}
    for sp in ("train", "val", "test"):
        inst = collections.Counter()
        n_img = 0
        for fn in sorted(split[sp]):
            w, h, keep = samples[fn]
            with open(os.path.join(OUT, sp, "images", fn), "wb") as f:
                f.write(z.read(member_of[fn]))
            lines = []
            for cls, x1, y1, x2, y2 in keep:
                inst[cls] += 1
                lines.append(f"{cls} {((x1+x2)/2/w):.6f} {((y1+y2)/2/h):.6f} "
                             f"{((x2-x1)/w):.6f} {((y2-y1)/h):.6f}")
            with open(os.path.join(OUT, sp, "labels", fn[:-4] + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            n_img += 1
        stats["stats"][sp] = {"images": n_img,
                              "instances": [inst.get(i, 0) for i in range(len(NAMES))]}
        print(f"{sp}: {n_img} 张 | 实例 {stats['stats'][sp]['instances']}")

    yaml = f"""# GC10-DET 论文固定划分 (split seed={SEED}, 与训练种子解耦)
path: {OUT}
train: train
val: val
test: test
nc: {len(NAMES)}
names: {NAMES}
"""
    with open(os.path.join(OUT, "gc10_det.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    with open(os.path.join(OUT, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "clean_log.json"), "w", encoding="utf-8") as f:
        json.dump({"dropped_unknown_name": dict(dropped_name), "images_without_xml": sorted(no_xml),
                   "duplicate_basenames_total": len(dup), "duplicate_unresolved": sorted(dup_unresolved),
                   "multi_object_images": multi, "invalid_boxes_dropped": bad_box}, f,
                  ensure_ascii=False, indent=1)

    # nc=10 的 P1/P2 模型 yaml（直接可训练，不依赖 trainer 重建时才改 nc）
    for tag, srcf in (("p2", "yolov8s-p2.yaml"), ("p1", "yolov8s-p1.yaml")):
        dep = os.path.join(r".\autoDL_deploy", srcf)
        s = open(dep, encoding="utf-8").read()
        s = re.sub(r"^nc: \d+", f"nc: {len(NAMES)}", s, flags=re.M)
        s = re.sub(r"^names: \[.*?\]$", f"names: {NAMES}", s, flags=re.M)
        s = s.replace("NEU-DET 钢材表面缺陷", "GC10-DET 钢材表面缺陷")
        outp = os.path.join(OUT, f"yolov8s-{tag}-gc10.yaml")
        with open(outp, "w", encoding="utf-8") as f:
            f.write(s)
        print("写出模型 yaml:", os.path.basename(outp))
    print("数据集根目录:", OUT)


if __name__ == "__main__":
    main()
