# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_deploy.py - 部署侧测量（论文表 7 / Pareto 图用）：延迟、吞吐、显存、模型体积

目的是回答"同一精度下代价是多少"，所以**只测已训好的 best.pt**，不改权重、不重训。
每条 run 测四种口径（能测的才测，缺的标 NA）：

    pt_fp32    PyTorch FP32，GPU
    pt_fp16    PyTorch FP16（half），GPU
    onnx_cpu   ONNX Runtime，CPU（同一实例的 CPU，记录线程数与型号）
    onnx_gpu   ONNX Runtime，CUDA provider（装了 onnxruntime-gpu 才有）

每个 cell 单独一个进程测温不合适（模型加载开销大），所以本脚本在一个进程里顺序测，
用 CUDA 事件 + 预热 + 多次取中位数与 p95；输入张量除 batch 外与训练同分辨率。

用法（在 ${DATA_ROOT}/gc10 下）:
    python measure_deploy.py runs                       # 全部 cell，默认 batch=1、iters=100
    python measure_deploy.py runs --batch 1 --iters 200
    python measure_deploy.py runs --cells v_base,v_p2   # 只测几个
    python measure_deploy.py runs --skip-onnx           # 只测 PyTorch 两条口径

产物：
    runs_deploy/<cell>.json    单 cell 明细（含每次迭代 ms 列表 percentiles）
    runs_deploy/summary.tsv    汇总表（cell / imgsz / batch / params / GFLOPs / 各口径 p50 p95）
    runs_deploy/env.txt        环境（GPU 名、CPU 型号与核数、torch/ort 版本）
"""
import argparse
import json
import os
import re
import statistics as st
import sys
import time

NAME_RE = re.compile(r"^(?P<variant>v_[a-z0-9]+?)(?:_s(?P<imgsz>\d+))?(?:_b(?P<batch>\d+))?"
                     r"(?:_cpw(?P<cpw>\d+p\d+))?(?:_e(?P<epochs>\d+))?_seed(?P<seed>\d+)$")


def bridge_custom_classes():
    """训练脚本以 __main__ 运行时保存的自定义类（C2f_DCSA 等）需要桥接才能 unpickle。"""
    import __main__
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import defect_train as dt
    except Exception as e:
        print(f"  [warn] import defect_train 失败（纯 YOLO 权重可忽略）: {e}")
        return
    n = 0
    for name in dir(dt):
        if name.endswith("_DCSA") or name in ("ChannelAttention", "SpatialAttention", "DCSA", "FocalBCE"):
            setattr(__main__, name, getattr(dt, name))
            n += 1
    print(f"  [bridge] {n} 个自定义类已挂到 __main__")


def pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def bench_pt(model, imgsz, batch, iters, warmup, half, device):
    import torch
    x = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    m = model.model.to(device).eval()
    if half:
        m = m.half()
        x = x.half()
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ts = []
            for _ in range(iters):
                ev0.record()
                m(x)
                ev1.record()
                torch.cuda.synchronize()
                ts.append(ev0.elapsed_time(ev1))
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            torch.cuda.reset_peak_memory_stats()
        else:
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                m(x)
                ts.append((time.perf_counter() - t0) * 1000)
            mem = None
    return ts, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", default="runs_deploy")
    ap.add_argument("--skip-onnx", action="store_true")
    a = ap.parse_args()

    bridge_custom_classes()
    import torch
    from ultralytics import YOLO

    dev = "cuda:%s" % a.device if (a.device.isdigit() and torch.cuda.is_available()) else "cpu"
    os.makedirs(a.out, exist_ok=True)
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU",
        "cpu": os.popen("lscpu | grep -E 'Model name|^CPU\\(s\\)' | head -2").read().strip() or "unknown",
        "device_used": dev,
        "batch": a.batch, "iters": a.iters, "warmup": a.warmup,
    }
    with open(os.path.join(a.out, "env.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(env, ensure_ascii=False, indent=1))
    print("环境:", json.dumps(env, ensure_ascii=False))

    todo = []
    for d in sorted(os.listdir(a.runs)):
        best = os.path.join(a.runs, d, "weights", "best.pt")
        if not os.path.isfile(best):
            continue
        m = NAME_RE.match(d)
        if not m:
            continue
        if a.cells and m.group("variant") not in a.cells.split(","):
            continue
        todo.append((d, best, int(m.group("imgsz") or 640)))
    if not todo:
        print("没有可测的 run（runs/<cell>/weights/best.pt 缺失？）")
        return 0

    print(f"\n待测 {len(todo)} 个 cell，batch={a.batch} iters={a.iters}\n")
    hdr = ["cell", "imgsz", "params", "GFLOPs",
           "pt_fp32_p50", "pt_fp32_p95", "pt_fp16_p50", "pt_fp16_p95", "gpu_mem_MB",
           "onnx_cpu_p50", "onnx_gpu_p50", "model_MB"]
    rows = [hdr]
    for d, best, imgsz in todo:
        print(f"---- {d} (imgsz={imgsz}) ----")
        y = YOLO(best)
        nparam = sum(p.numel() for p in y.model.parameters())
        try:
            from ultralytics.utils.torch_utils import get_flops
            gf = round(float(get_flops(y.model, imgsz)), 1)
        except Exception as e:
            gf = None
            print(f"  [warn] GFLOPs 取不到: {e}")
        rec = {"cell": d, "imgsz": imgsz, "params": int(nparam), "gflops": gf,
               "batch": a.batch, "iters": a.iters, "model_MB": round(os.path.getsize(best) / 1024 ** 2, 1)}
        ts, mem = bench_pt(y, imgsz, a.batch, a.iters, a.warmup, False, dev)
        rec["pt_fp32"] = {"p50": pct(ts, 0.5), "p95": pct(ts, 0.95), "mean": st.mean(ts)}
        rec["gpu_mem_MB"] = None if mem is None else round(mem, 1)
        print(f"  pt_fp32  p50={rec['pt_fp32']['p50']:.2f} ms  p95={rec['pt_fp32']['p95']:.2f} ms"
              f"  {'' if mem is None else 'mem=%.0fMB' % mem}")
        if dev.startswith("cuda"):
            tsh, memh = bench_pt(y, imgsz, a.batch, a.iters, a.warmup, True, dev)
            rec["pt_fp16"] = {"p50": pct(tsh, 0.5), "p95": pct(tsh, 0.95), "mean": st.mean(tsh)}
            print(f"  pt_fp16  p50={rec['pt_fp16']['p50']:.2f} ms  p95={rec['pt_fp16']['p95']:.2f} ms")
        if not a.skip_onnx:
            try:
                onnx_path = y.export(format="onnx", imgsz=imgsz, batch=a.batch, opset=12, dynamic=False)
                import onnxruntime as ort
                av = ort.get_available_providers()
                xnp = {n.name: n.shape for n in ort.InferenceSession(str(onnx_path)).get_inputs()}
                key = list(xnp)[0]
                shape = [1 if isinstance(v, str) or v is None else v for v in xnp[key]]
                import numpy as np
                for prov, tag in (("CPUExecutionProvider", "onnx_cpu"), ("CUDAExecutionProvider", "onnx_gpu")):
                    if prov not in av:
                        continue
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = os.cpu_count() or 8
                    sess = ort.InferenceSession(str(onnx_path), so, providers=[prov])
                    inp = np.random.rand(*[a.batch if i == 0 else v for i, v in enumerate(shape)]).astype("float32")
                    for _ in range(max(3, a.warmup // 4)):
                        sess.run(None, {key: inp})
                    tt = []
                    for _ in range(max(10, a.iters // 4)):
                        t0 = time.perf_counter()
                        sess.run(None, {key: inp})
                        tt.append((time.perf_counter() - t0) * 1000)
                    rec[tag] = {"p50": pct(tt, 0.5), "p95": pct(tt, 0.95), "mean": st.mean(tt), "providers": av}
                    print(f"  {tag}  p50={rec[tag]['p50']:.2f} ms（providers={av}）")
                # 导出的 .onnx 只是测速用，测完删掉省盘（每个 40+MB）
                try:
                    os.remove(str(onnx_path))
                    print("  [clean] 已删除临时 onnx:", os.path.basename(str(onnx_path)))
                except OSError:
                    pass
            except Exception as e:
                rec["onnx_error"] = str(e)
                print(f"  [warn] ONNX 口径失败: {e}")
        json.dump(rec, open(os.path.join(a.out, d + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        rows.append([d, imgsz, nparam, rec.get("gflops"), rec.get("pt_fp32", {}).get("p50"),
                     rec.get("pt_fp32", {}).get("p95"), rec.get("pt_fp16", {}).get("p50"),
                     rec.get("pt_fp16", {}).get("p95"), rec.get("gpu_mem_MB"),
                     rec.get("onnx_cpu", {}).get("p50"), rec.get("onnx_gpu", {}).get("p50"),
                     rec["model_MB"]])
        json.dump(rec, open(os.path.join(a.out, d + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    with open(os.path.join(a.out, "summary.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write("\t".join("" if x is None else str(round(x, 2)) if isinstance(x, float) else str(x)
                              for x in r) + "\n")
    print("\n汇总:", os.path.join(a.out, "summary.tsv"))
    print("提示：延迟要报 p50 与 p95，并写清 batch 与分辨率；不同 batch 之间不可直接比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
