# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# 统一批次部署表（2026-09-30 同一会话四组连测）→ 生成表 7 并回写 表格定稿 md
import json, os, statistics as st
from collections import defaultdict

B = "./_results/deploy_20260930"
G = {"GC10 s": "deploy_20260930_gc10_s", "GC10 nano": "deploy_20260930_gc10_nano",
     "NEU s": "deploy_20260930_neu_s", "NEU nano": "deploy_20260930_neu_nano"}
PARAMS = {  # best.pt 实测参数量（与 表_参数量订正 一致）
    ("GC10", "s"): {"v_base": 11139470, "v_wiou": 11139470, "v_dcsa": 11154301, "v_p2": 10639576,
                    "v_full": 10654407, "v_p2dcsawiou": 10654407, "v_full_cpw0p5": 10654407,
                    "v_focal": 10654407, "v_resample": 10654407, "v_p1_b8": 10026264,
                    "v_base_b8": 11139470, "v_base_s320_b8": 11139470, "v_base_s960_b8": 11139470,
                    "v_base_e300": 11139470, "v_full_e300": 10654407},
    ("NEU", "s"): {"v_base": 11137922, "v_wiou": 11137922, "v_dcsa": 11152753, "v_p2": 10638536,
                   "v_full": 10653367, "v_p2dcsawiou": 10653367, "v_full_cpw0p5": 10653367,
                   "v_focal": 10653367, "v_resample": 10653367, "v_p1_b8": 10025736,
                   "v_base_b8": 11137922, "v_base_s320_b8": 11137922, "v_base_s960_b8": 11137922},
    ("GC10", "nano"): {"v_base": 3012798, "v_wiou": 3012798, "v_dcsa": 3018413, "v_p2": 2927880, "v_full": 2933495},
    ("NEU", "nano"): {"v_base": 3012018, "v_wiou": 3012018, "v_dcsa": 3017633, "v_p2": 2927352, "v_full": 2932967},
}

def read_group(d):
    per = defaultdict(lambda: defaultdict(list))
    for fn in sorted(os.listdir(os.path.join(B, d))):
        if not fn.endswith(".json"):
            continue
        j = json.load(open(os.path.join(B, d, fn), encoding="utf-8"))
        key = (j.get("cell") or fn[:-5]).rsplit("_seed", 1)[0]
        per[key]["fp32"].append(j["pt_fp32"]["p50"])
        per[key]["fp16"].append(j["pt_fp16"]["p50"])
        per[key]["cpu"].append(j["onnx_cpu"]["p50"])
        per[key]["mem"].append(j["gpu_mem_MB"])
        per[key]["MB"].append(j["model_MB"])
        per[key]["gf"].append(j["gflops"])
    return {k: {m: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) for m, v in d2.items()} for k, d2 in per.items()}

D = {k: read_group(v) for k, v in G.items()}
def f(t, dp=2):
    m, s = t
    return "%.*f±%.*f" % (dp, m, dp, s) if s else "%.*f" % (dp, m)

def pget(ds, sc, c):
    d = PARAMS.get((ds, sc))
    if not d:
        return 0
    if c in d:
        return d[c]
    if c.endswith("_e300") and c[:-5] in d:
        return d[c[:-5]]
    return 0

L = []
w = L.append
w("## 表 7 部署指标（统一批次：同一会话四组连测，batch=1，RTX 4090 + Xeon Gold 6430）")
w("")
w("| 组 | 配置 | 参数量 | GFLOPs | FP32 p50 (ms) | FP16 p50 (ms) | ONNX-CPU p50 (ms) | 显存 (MB) | 体积 (MB) |")
w("|---|---|---|---|---|---|---|---|---|")
CELLS_S = ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_p2dcsawiou", "v_full", "v_full_cpw0p5",
           "v_p1_b8", "v_base_b8", "v_base_s320_b8", "v_base_s960_b8", "v_resample", "v_focal"]
for gname, ds, sc in [("GC10 s", "GC10", "s"), ("GC10 nano", "GC10", "nano"),
                      ("NEU s", "NEU", "s"), ("NEU nano", "NEU", "nano")]:
    cells = ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_full"] if sc == "nano" else \
            [c for c in CELLS_S if c in D[gname]]
    for c in cells:
        a = D[gname].get(c)
        if not a:
            continue
        w("| %s | %s | %s | %.1f | %s | %s | %s | %s | %.1f |" % (
            gname, c, format(pget(ds, sc, c), ","), a["gf"][0], f(a["fp32"]), f(a["fp16"]),
            f(a["cpu"], 1), f(a["mem"], 1), a["MB"][0]))
    w("| | *（其余 s 尺度 cell）* | | | | | | | |")
w("")
w("表注：同一会话内连续测量四组（batch=1，100 次计时 + 20 次预热），组间可比；绝对值受宿主负载影响，"
   "不与其他批次的测量跨批比较。参数量为 best.pt 实测值。FP16 在这些小模型上普遍慢于 FP32（batch=1 时"
   "kernel 启动开销占主导），本文不据此主张半精度加速。")
w("")

path = "./_paper/paper3_表格定稿_20260930.md"
txt = open(path, encoding="utf-8").read()
head = txt.split("## 表 7")[0]
open(path, "w", encoding="utf-8").write(head + "\n".join(L) + "\n")

print("\n".join(L))
print("=" * 100)
print("关键对照（统一批次）")
print("=" * 100)
def show(ds, sc, c, m, dp=2):
    a = D[f"{ds} {sc}"].get(c)
    return "—" if not a else "%.*f" % (dp, a[m][0])

for ds in ("GC10", "NEU"):
    for m, nm in (("fp32", "FP32 延迟"), ("fp16", "FP16 延迟"), ("cpu", "ONNX-CPU 延迟"),
                  ("mem", "显存"), ("MB", "体积"), ("gf", "GFLOPs")):
        s = D[f"{ds} s"].get("v_base"); n = D[f"{ds} nano"].get("v_base")
        if not s or not n:
            continue
        d = (n[m][0] - s[m][0]) / s[m][0] * 100
        print("  %-5s %-14s s=%8.2f  nano=%8.2f   nano−s = %+6.1f%%" % (ds, nm, s[m][0], n[m][0], d))
    print()

print("同一 nano 架构在两数据集上（统一批次，验证组间可比性）: ")
for c in ("v_base", "v_full"):
    print("  %-8s GC10 nano CPU=%s ms / NEU nano CPU=%s ms | GC10 nano FP32=%s / NEU nano FP32=%s" % (
        c, show("GC10", "nano", c, "cpu", 1), show("NEU", "nano", c, "cpu", 1),
        show("GC10", "nano", c, "fp32"), show("NEU", "nano", c, "fp32")))

print()
print("P2 与四件套在 nano 上的代价（相对 nano 基线）:")
for ds in ("GC10", "NEU"):
    b = D[f"{ds} nano"].get("v_base")
    for c in ("v_p2", "v_full"):
        a = D[f"{ds} nano"].get(c)
        if not a or not b:
            continue
        print("  %-5s nano %-8s GFLOPS %+5.1f%%   FP32 %+6.1f%%   CPU %+6.1f%%" % (
            ds, c, (a["gf"][0]/b["gf"][0]-1)*100, (a["fp32"][0]/b["fp32"][0]-1)*100, (a["cpu"][0]/b["cpu"][0]-1)*100))
print()
print("已回写:", path)
