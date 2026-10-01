# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_test.py - 用已有 best.pt 在 test 划分上补测（不重训，几分钟跑完全部）

背景：run_gc10.sh 只给 v_full 系列传了 --final-test，其余 cell 的 json 里没有 test 数字；
论文主证据要用 test（345 张），所以用这个脚本对**已训好的权重**逐个补测 —— 只做推理，不改权重、不改协议。

用法:
    python eval_test.py runs                                   # GC10 默认
    python eval_test.py runs --data data/GC10-DET-yolo/gc10_det.yaml
    python eval_test.py runs --batch 8                         # b8 档的 run 用 8
    python eval_test.py runs --cells v_base,v_p2,v_dcsa        # 只补几个

做三件事：
  1. 解析 run 目录名得到该 run 的 imgsz / batch（与 defect_train.run_name 同规则），保证评估口径与训练一致；
  2. 载入 best.pt（自动为 C2f_DCSA 等自定义类做 unpickle 兼容 —— 训练脚本以 __main__ 运行时保存的类路径需要桥接）；
  3. 在 test 上评估，把结果**合并进该 run 的 perclass_ap.json 的 test 字段**（analyze_matrix.py 会直接读到），
     同时写 runs_eval/<run>/ 下的常规评估产物，并打印一张汇总表。
"""
import argparse
import json
import os
import re
import sys

NAME_RE = re.compile(r"^(?P<variant>v_[a-z0-9]+?)(?:_s(?P<imgsz>\d+))?(?:_b(?P<batch>\d+))?"
                     r"(?:_cpw(?P<cpw>\d+p\d+))?(?:_e(?P<epochs>\d+))?_seed(?P<seed>\d+)$")


def bridge_custom_classes():
    """训练脚本以 `python defect_train.py` 运行时，自定义类的 __module__ 是 "__main__"，
    权重里 pickle 记录的路径也是 "__main__.<类名>" —— 在别的进程里 unpickle 就找不到。
    这里把 defect_train 里的类挂到 __main__ 上做桥接（只影响反序列化，不改任何训练行为）。"""
    import __main__
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import defect_train as dt
    except Exception as e:
        print(f"  [warn] 无法 import defect_train（若是纯 YOLOv8 权重可忽略）: {e}")
        return
    names = [n for n in dir(dt) if n.endswith("_DCSA") or n in
             ("ChannelAttention", "SpatialAttention", "DCSA", "FocalBCE")]
    for n in names:
        setattr(__main__, n, getattr(dt, n))
    print(f"  [bridge] 已把 {len(names)} 个自定义类挂到 __main__：{names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--data", default="data/GC10-DET-yolo/gc10_det.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--cells", default="", help="逗号分隔，只补这些 cell（默认全部）")
    ap.add_argument("--project", default="runs_eval")
    ap.add_argument("--overwrite", action="store_true", help="已有 test 字段也重测")
    a = ap.parse_args()

    if not os.path.isfile(a.data):
        raise SystemExit(f"找不到数据 yaml: {a.data}")

    todo = []
    for d in sorted(os.listdir(a.runs)):
        rd = os.path.join(a.runs, d)
        best = os.path.join(rd, "weights", "best.pt")
        js = os.path.join(rd, "perclass_ap.json")
        if not (os.path.isfile(best) and os.path.isfile(js)):
            continue
        m = NAME_RE.match(d)
        if not m:
            print(f"  [skip] {d}: 目录名不符合规则")
            continue
        if a.cells and m.group("variant") not in a.cells.split(","):
            continue
        try:
            cur = json.load(open(js, encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {d}: json 读不了 {e}")
            continue
        if isinstance(cur.get("test"), dict) and not a.overwrite:
            print(f"  [skip] {d}: 已有 test 结果（用 --overwrite 重测）")
            continue
        todo.append((d, best, js, int(m.group("imgsz") or 640),
                     int(m.group("batch") or 16)))

    if not todo:
        print("没有需要补测的 run。")
        return 0

    bridge_custom_classes()
    from ultralytics import YOLO

    print(f"\n待补测 {len(todo)} 个 run（test 划分）\n")
    print(f"{'run':<26} {'imgsz':>5} {'batch':>5} {'mAP50':>8} {'mAP50-95':>9}  每类AP50")
    rows = []
    for d, best, js, imgsz, batch in todo:
        print(f"---- {d} (imgsz={imgsz}, batch={batch}) ----")
        model = YOLO(best)
        res = model.val(data=a.data, split="test", imgsz=imgsz, batch=batch,
                        device=a.device, project=a.project, name=d, plots=False,
                        verbose=False)
        names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names))
        classes = {}
        for i, c in enumerate(res.ap_class_index):
            classes[str(names[int(c)])] = {"ap50": float(res.box.ap50[i]),
                                           "ap5095": float(res.box.ap[i])}
        m50, m5095 = float(res.box.map50), float(res.box.map)
        rec = json.load(open(js, encoding="utf-8"))
        rec["test"] = {"mAP50": m50, "mAP5095": m5095, "classes": classes,
                       "eval_batch": batch, "eval_imgsz": imgsz}
        json.dump(rec, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        per = " ".join(f"{k.split('_')[0][:4]}={v['ap50']:.3f}" for k, v in classes.items())
        print(f"{d:<26} {imgsz:>5} {batch:>5} {m50:>8.4f} {m5095:>9.4f}  {per}")
        rows.append((d, imgsz, batch, m50, m5095))

    print(f"\n完成 {len(rows)} 个 run；结果已写回各自 perclass_ap.json 的 test 字段。")
    print("接着跑: python analyze_matrix.py runs --out tables   （表 A 的 test 列会填上）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
