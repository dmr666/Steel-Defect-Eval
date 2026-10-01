# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# paper3：英文版表格（投稿用）→ _paper/tables_en_20260930.md
# 数据源与中文版完全一致（best.pt 复评口径，3 种子均值±样本标准差）
import os, math

R = "./_results"
SRC = {
    ("GC10", "s"): f"{R}/gc10_20260929/tables",
    ("NEU", "s"): f"{R}/neu_20260929/tables",
    ("GC10", "nano"): f"{R}/gc10_nano_20260929/tables_nano",
    ("NEU", "nano"): f"{R}/neu_nano_20260930/small/tables_nano",
}
DEP = {
    ("GC10", "s"): f"{R}/deploy_20260930/deploy_20260930_gc10_s",
    ("GC10", "nano"): f"{R}/deploy_20260930/deploy_20260930_gc10_nano",
    ("NEU", "s"): f"{R}/deploy_20260930/deploy_20260930_neu_s",
    ("NEU", "nano"): f"{R}/deploy_20260930/deploy_20260930_neu_nano",
}
PARAMS = {
    ("GC10", "s"): {"v_base": 11139470, "v_wiou": 11139470, "v_dcsa": 11154301, "v_p2": 10639576,
                    "v_full": 10654407, "v_p2dcsawiou": 10654407, "v_full_cpw0p5": 10654407,
                    "v_focal": 10654407, "v_resample": 10654407, "v_p1_b8": 10026264,
                    "v_base_b8": 11139470, "v_base_s320_b8": 11139470, "v_base_s960_b8": 11139470,
                    "v_base_e300": 11139470, "v_full_e300": 10654407},
    ("NEU", "s"): {"v_base": 11137922, "v_wiou": 11137922, "v_dcsa": 11152753, "v_p2": 10638536,
                   "v_full": 10653367, "v_p2dcsawiou": 10653367, "v_full_cpw0p5": 10653367,
                   "v_focal": 10653367, "v_resample": 10653367, "v_p1_b8": 10025736,
                   "v_base_b8": 11137922, "v_base_s320_b8": 11137922, "v_base_s960_b8": 11137922,
                   "v_full_s320_b8": 10653367, "v_full_s960_b8": 10653367},
    ("GC10", "nano"): {"v_base": 3012798, "v_wiou": 3012798, "v_dcsa": 3018413, "v_p2": 2927880, "v_full": 2933495},
    ("NEU", "nano"): {"v_base": 3012018, "v_wiou": 3012018, "v_dcsa": 3017633, "v_p2": 2927352, "v_full": 2932967},
}
DESC = {"v_base": "YOLOv8s baseline", "v_p2": "+P2 head", "v_dcsa": "+DCSA", "v_wiou": "+WIoU v3",
        "v_p2dcsawiou": "+P2+DCSA+WIoU", "v_full": "above + $w_c$=1.0 (full stack)",
        "v_full_cpw0p5": "full stack, $w_c$=0.5", "v_p1_b8": "+P1 head (b8)", "v_base_b8": "baseline (b8)",
        "v_base_s320_b8": "baseline @320 (b8)", "v_base_s960_b8": "baseline @960 (b8)",
        "v_full_s320_b8": "full stack @320 (b8)", "v_full_s960_b8": "full stack @960 (b8)",
        "v_resample": "resampling", "v_focal": "Focal BCE", "v_base_e300": "baseline, 300 ep",
        "v_full_e300": "full stack, 300 ep"}
ORDER = ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_p2dcsawiou", "v_full", "v_full_cpw0p5", "v_p1_b8",
         "v_base_b8", "v_base_s320_b8", "v_base_s960_b8", "v_full_s320_b8", "v_full_s960_b8",
         "v_resample", "v_focal", "v_base_e300", "v_full_e300"]

def num(s):
    s = (s or "").strip()
    if not s or s == "—":
        return None, None
    if "±" in s:
        a, b = s.split("±")
        return float(a), float(b)
    return float(s), None

def read_main(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").rstrip("\r").split("\t")
        if not line.strip():
            continue
        if hdr is None:
            hdr = f; continue
        if len(f) < 9:
            continue
        d = dict(zip(hdr, f))
        if int(d["n"]) == 0:
            continue
        out[d["cell"]] = {"val50": num(d["val mAP50"]), "val95": num(d["val mAP50-95"]),
                          "test95": num(d.get("test mAP50-95", "")), "n": int(d["n"])}
    return out

def read_imbal(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").rstrip("\r").split("\t")
        if not line.strip():
            continue
        if hdr is None:
            hdr = f; continue
        if len(f) < 7:
            continue
        out[f[0].strip()] = {"val95": num(f[3]), "val50": num(f[4]), "rare": f[5], "rest": f[6]}
    return out

def read_dep(d):
    import json, statistics as st
    from collections import defaultdict
    per = defaultdict(lambda: defaultdict(list))
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        k = (j.get("cell") or fn[:-5]).rsplit("_seed", 1)[0]
        per[k]["fp32"].append(j["pt_fp32"]["p50"]); per[k]["fp16"].append(j["pt_fp16"]["p50"])
        per[k]["cpu"].append(j["onnx_cpu"]["p50"]); per[k]["mem"].append(j["gpu_mem_MB"])
        per[k]["MB"].append(j["model_MB"]); per[k]["gf"].append(j["gflops"])
    return {k: {m: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) for m, v in dd.items()} for k, dd in per.items()}

M = {k: read_main(os.path.join(v, "table_main.tsv")) for k, v in SRC.items()}
# NEU ext（v_full@320 / v_full@960，2026-09-30 单独一批训练；口径与主表一致：perclass_ap.json）
EXT_TABLES = f"{R}/neu_ext_20260930/small/tables/table_main_ext.tsv"
EXT_DEP = f"{R}/neu_ext_20260930/ext_deploy/runs_deploy_ext"
if os.path.isfile(EXT_TABLES):
    M[("NEU", "s")].update(read_main(EXT_TABLES))
IM = {"GC10": read_imbal(f"{SRC[('GC10','s')]}/table_imbalance.tsv"),
      "NEU": read_imbal(f"{SRC[('NEU','s')]}/table_imbalance.tsv")}
D = {k: read_dep(v) for k, v in DEP.items()}
DEXT = read_dep(EXT_DEP) if os.path.isdir(EXT_DEP) else {}

def gflops_of(ds, sc, c):
    """GFLOPs 为确定性量（get_flops 与测量会话无关）；ext cell 取 ext 批 json"""
    a = (DEXT if c in DEXT else D[(ds, sc)]).get(c)
    return a["gf"][0] if a else None
CELLS5 = ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_full"]
ARM_EN = {"无处理": "untreated", "指数加权 0.5": "inverse-frequency $w_c$=0.5",
          "指数加权 1.0": "inverse-frequency $w_c$=1.0", "重采样": "resampling (sqrt-inv-freq)",
          "Focal": "Focal ($\\gamma$=1.5, $\\alpha$=0.25)"}

def s(t, dp=4):
    if not t:
        return "—"
    m, sd = t
    return ("%.*f±%.*f" % (dp, m, dp, sd)) if sd else ("%.*f" % (dp, m))

def sd(t, dp=2):
    if not t:
        return "—"
    m, sdd = t
    return ("%.*f±%.*f" % (dp, m, dp, sdd)) if sdd else ("%.*f" % (dp, m))

def delta(base, cur, dp=4):
    if not base or not cur:
        return "—"
    d = cur[0] - base[0]
    p = math.sqrt(((base[1] or 0) ** 2 + (cur[1] or 0) ** 2) / 2)
    if p == 0:
        return "%+.*f" % (dp, d)
    return "%+.*f %s" % (dp, d, "sig." if abs(d) > 2 * p else "n.s.")

def pget(ds, c):
    d = PARAMS[(ds, "s")]
    if c in d:
        return d[c]
    if c.endswith("_e300") and c[:-5] in d:
        return d[c[:-5]]
    return 0

L = []
w = L.append
w("# Appendix A. Full result tables (submission version)")
w("")
w("Tables 2–9 below give the complete matrices behind the figures in the main text. "
  "Protocol: best.pt re-evaluation; validation and held-out test reported separately; three seeds "
  "(42/7/2024), mean ± sample standard deviation (n−1); parameter counts measured from `best.pt`. "
  "A difference is called significant (sig.) only when |Δ| > 2·σ_pooled with σ_pooled = √((σ₁²+σ₂²)/2); "
  "otherwise n.s. (inside seed noise).")

for ds, tno, tname, note in [
    ("GC10", 2, "Main ablation matrix (GC10-DET, 10 classes, 640 px / batch 16)",
     "Parameters are measured from `best.pt`. The `test` column is the held-out partition (345 images). "
     "Cells suffixed `e300` use a 300-epoch budget (training-budget robustness check)."),
    ("NEU", 5, "Second benchmark (NEU-DET, 6 classes, 640 px / batch 16)",
     "Same protocol and structure as Table 2; only the dataset and the number of classes differ "
     "(NEU-DET images are 200×200, upsampled to 640).")]:
    t = M[(ds, "s")]
    base = t["v_base"]["test95"]
    w("")
    w("## Table %d. %s" % (tno, tname))
    w("")
    w("| Cell | Description | val mAP50-95 | val mAP50 | test mAP50-95 | test vs baseline | Params | Epochs |")
    w("|---|---|---|---|---|---|---|---|")
    for c in ORDER:
        if c not in t:
            continue
        r = t[c]
        w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c, DESC[c], s(r["val95"]), s(r["val50"]), s(r["test95"]), delta(base, r["test95"]),
            format(pget(ds, c), ","), "300" if c.endswith("e300") else "200"))
    w("")
    w("Note: %s" % note)

w("")
w("## Table 3. Input resolution and shallow detection heads (GC10-DET, batch 8)")
w("")
w("| Configuration | val mAP50-95 | val mAP50 | test mAP50-95 | vs 640 baseline | Params |")
w("|---|---|---|---|---|---|")
t = M[("GC10", "s")]
ref = t["v_base_b8"]["val95"]
for c, nm in [("v_base_s320_b8", "640→320"), ("v_base_b8", "640 (batch-8 reference)"),
              ("v_base_s960_b8", "640→960"), ("v_p1_b8", "+P1 head (stride 2)"), ("v_p2", "+P2 head (b16)")]:
    if c not in t:
        continue
    r = t[c]
    w("| %s | %s | %s | %s | %s | %s |" % (nm, s(r["val95"]), s(r["val50"]), s(r["test95"]),
                                          delta(ref, r["val95"]), format(pget("GC10", c), ",")))
w("")
w("Note: differences are computed against the batch-8 reference row; the compute cost of each change is "
  "given in Table 7.")

w("")
w("## Table 3b. Input resolution on the second benchmark (NEU-DET, s scale, batch 8)")
w("")
w("| Configuration | Input (px) | val mAP50-95 | val mAP50 | test mAP50-95 | test vs 640, same arm | GFLOPs | Params |")
w("|---|---|---|---|---|---|---|---|")
t = M[("NEU", "s")]
PX = {"v_base_s320_b8": "320", "v_base_b8": "640", "v_base_s960_b8": "960",
      "v_full_s320_b8": "320", "v_full": "640", "v_full_s960_b8": "960"}
ref_b = t["v_base_b8"]["test95"]
ref_f = t["v_full"]["test95"]
for c, nm, ref in [("v_base_s320_b8", "baseline", ref_b), ("v_base_b8", "baseline (640 reference)", None),
                   ("v_base_s960_b8", "baseline", ref_b),
                   ("v_full_s320_b8", "full stack", ref_f), ("v_full", "full stack (640 reference)", None),
                   ("v_full_s960_b8", "full stack", ref_f)]:
    if c not in t:
        continue
    r = t[c]
    gf = gflops_of("NEU", "s", c)
    w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
        nm, PX[c], s(r["val95"]), s(r["val50"]), s(r["test95"]),
        "—" if ref is None else delta(ref, r["test95"]),
        ("%.1f" % gf) if gf else "—", format(pget("NEU", c), ",")))
w("")
w("Note: six NEU-DET s-scale runs under the identical protocol (200 epochs, batch 8, one split, seeds "
  "42/7/2024); the 320 and 960 rows of the full stack were trained after the main matrix with the same code, "
  "same split and same seeds. GFLOPs and parameter counts are exact properties of the exported models and are "
  "comparable across batches. The ONNX Runtime CPU column of Table 7 deliberately omits these two cells: they "
  "were measured in a separate session, and §4.6 establishes that only same-session CPU comparisons are "
  "meaningful. The same-session contrast inside that separate batch is valid and instructive: the full stack "
  "costs 932 vs 3365 ms at 320 vs 960 px (+261%), while the held-out test mAP50-95 stays inside seed noise "
  "(0.3464 vs 0.3513).")

w("")
w("## Table 4. Class-imbalance arms (GC10-DET, shared architecture P2+DCSA+WIoU)")
w("")
w("| Arm | val mAP50-95 | val mAP50 | rare-class AP50 | mean AP50 of remaining classes | vs untreated |")
w("|---|---|---|---|---|---|")
b = IM["GC10"].get("无处理") or IM["GC10"].get(list(IM["GC10"])[0])
for arm, r in IM["GC10"].items():
    w("| %s | %s | %s | %s | %s | %s |" % (ARM_EN.get(arm, arm), s(r["val95"]), s(r["val50"]),
                                          r["rare"], r["rest"],
                                          "—" if r is b else delta(b["val95"], r["val95"])))
w("")
w("Note: rare classes are the two least frequent classes of the training split (GC10-DET: zhehen, yahen); "
  "all arms share one architecture and one seed set, so only the imbalance treatment varies.")

w("")
w("## Table 8. Cross-scale comparison (2 datasets × 2 backbone scales, five shared configurations)")
w("")
for key, sub in (("val50", "8a. Validation mAP50"), ("val95", "8b. Validation mAP50-95"),
                 ("test95", "8c. Test mAP50-95")):
    w("")
    w("**Table %s**" % sub)
    w("")
    w("| Configuration | GC10-DET s (11.1 M) | GC10-DET n (3.0 M) | Δ (n − s) | NEU-DET s (11.1 M) | NEU-DET n (3.0 M) | Δ (n − s) |")
    w("|---|---|---|---|---|---|---|")
    for c in CELLS5:
        a1 = M[("GC10", "s")].get(c, {}).get(key)
        b1 = M[("GC10", "nano")].get(c, {}).get(key)
        a2 = M[("NEU", "s")].get(c, {}).get(key)
        b2 = M[("NEU", "nano")].get(c, {}).get(key)
        w("| %s | %s | %s | %s | %s | %s | %s |" % (DESC[c], s(a1), s(b1), delta(a1, b1),
                                                    s(a2), s(b2), delta(a2, b2)))
w("")
w("Note: n = YOLOv8n, s = YOLOv8s; both scales use identical data splits, protocol and seeds. "
  "Only the five shared configurations were trained at the nano scale.")

w("")
w("## Table 9. Imbalance treatments across two datasets (s scale)")
w("")
w("| Arm | GC10-DET val mAP50-95 | GC10-DET val mAP50 | NEU-DET val mAP50-95 | NEU-DET val mAP50 | Verdict |")
w("|---|---|---|---|---|---|")
first = list(IM["GC10"])[0]
for arm in IM["GC10"]:
    g, n = IM["GC10"].get(arm), IM["NEU"].get(arm)
    if not g or not n:
        continue
    vg = "reference" if arm == first else delta(IM["GC10"][first]["val95"], g["val95"]).split(" ", 1)[-1]
    vn = "reference" if arm == first else delta(IM["NEU"][first]["val95"], n["val95"]).split(" ", 1)[-1]
    verdict = "reference" if arm == first else ("n.s. on both" if vg == "n.s." and vn == "n.s." else
                                               ("sig. on both" if vg == "sig." and vn == "sig." else "sig. on one"))
    w("| %s | %s | %s | %s | %s | %s |" % (ARM_EN.get(arm, arm), s(g["val95"]), s(g["val50"]),
                                          s(n["val95"]), s(n["val50"]), verdict))
w("")
w("Note: identical arm definitions on both datasets; GC10-DET rare classes are zhehen/yahen (11.4:1 train "
  "imbalance) and NEU-DET rare classes are pitted_surface/scratches (2.4:1).")

w("")
w("## Table 7. Deployment battery (unified single-session batch; batch 1; RTX 4090 + Xeon Gold 6430)")
w("")
w("| Group | Cell | Params | GFLOPs | FP32 p50 (ms) | FP16 p50 (ms) | ONNX-CPU p50 (ms) | GPU mem (MB) | Size (MB) |")
w("|---|---|---|---|---|---|---|---|---|")
for gname, ds, sc in [("GC10-DET s", "GC10", "s"), ("GC10-DET nano", "GC10", "nano"),
                      ("NEU-DET s", "NEU", "s"), ("NEU-DET nano", "NEU", "nano")]:
    cells = CELLS5 if sc == "nano" else [c for c in ORDER if c in D[(ds, sc)]]
    for c in cells:
        a = D[(ds, sc)].get(c)
        if not a:
            continue
        w("| %s | %s | %s | %.1f | %s | %s | %s | %s | %.1f |" % (
            gname, c, format(pget(ds, c), ","), a["gf"][0], sd(a["fp32"]), sd(a["fp16"]),
            sd(a["cpu"], 1), sd(a["mem"], 1), a["MB"][0]))
w("")
w("Note: all four groups were measured back-to-back in one session so that the columns are mutually "
  "comparable; absolute values are affected by host load and must not be compared across sessions "
  "(a same-architecture control inside this batch differs by ~11% on the CPU column). FP16 is slower than "
  "FP32 for these small models at batch 1, so no half-precision speed-up is claimed.")

path = "./_paper/tables_en_20260930.md"
open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("已写入:", path, "(%d 行)" % len(L))
print("覆盖: GC10 s %d cells | NEU s %d | GC10 nano %d | NEU nano %d" % (
    len(M[("GC10", "s")]), len(M[("NEU", "s")]), len(M[("GC10", "nano")]), len(M[("NEU", "nano")])))
