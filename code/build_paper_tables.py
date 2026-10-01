# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# paper3：按论文体例生成 9 张表（markdown，题注/表注齐全），数据全部取自本地归档
# 口径：best.pt 复评（val 与 test 分列）；3 种子均值±样本标准差；参数量取 best.pt 实测
import os, json, math, statistics as st

R = "./_results"
GC10_S = f"{R}/gc10_20260929/tables"
NEU_S = f"{R}/neu_20260929/tables"
GC10_N = f"{R}/gc10_nano_20260929/tables_nano"
NEU_N = f"{R}/neu_nano_20260930/small/tables_nano"

PARAMS = {
    ("GC10", "s"): {"v_base": 11139470, "v_wiou": 11139470, "v_dcsa": 11154301, "v_p2": 10639576,
                    "v_full": 10654407, "v_p2dcsawiou": 10654407, "v_full_cpw0p5": 10654407,
                    "v_focal": 10654407, "v_resample": 10654407, "v_p1_b8": 10026264,
                    "v_base_b8": 11139470, "v_base_s320_b8": 11139470, "v_base_s960_b8": 11139470},
    ("NEU", "s"): {"v_base": 11137922, "v_wiou": 11137922, "v_dcsa": 11152753, "v_p2": 10638536,
                   "v_full": 10653367, "v_p2dcsawiou": 10653367, "v_full_cpw0p5": 10653367,
                   "v_focal": 10653367, "v_resample": 10653367, "v_p1_b8": 10025736,
                   "v_base_b8": 11137922, "v_base_s320_b8": 11137922, "v_base_s960_b8": 11137922,
                   "v_full_s320_b8": 10653367, "v_full_s960_b8": 10653367},
    ("GC10", "nano"): {"v_base": 3012798, "v_wiou": 3012798, "v_dcsa": 3018413, "v_p2": 2927880, "v_full": 2933495},
    ("NEU", "nano"): {"v_base": 3012018, "v_wiou": 3012018, "v_dcsa": 3017633, "v_p2": 2927352, "v_full": 2932967},
}

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
        out[d["cell"]] = {"variant": d["variant"], "imgsz": d["imgsz"], "batch": d["batch"], "cls_pw": d["cls_pw"],
                          "n": int(d["n"]), "val50": num(d["val mAP50"]), "val95": num(d["val mAP50-95"]),
                          "test50": num(d.get("test mAP50", "")), "test95": num(d.get("test mAP50-95", ""))}
    return out

def f2(t, dp=4):
    if not t:
        return "—"
    m, s = t
    return "%.*f±%.*f" % (dp, m, dp, s) if s else "%.*f" % (dp, m)

def f1(t, dp=4):
    return "—" if not t else "%.*f" % (dp, t[0])

def sig_word(base, cur):
    """cur 相对 base 的差值 + 显著性（|Δ| > 2×合并标准差）"""
    if not base or not cur:
        return None, "—"
    d = cur[0] - base[0]
    s1, s2 = base[1] or 0.0, cur[1] or 0.0
    pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2) if (s1 or s2) else 0.0
    if pooled == 0:
        return d, "无离散度"
    return d, ("显著" if abs(d) > 2 * pooled else "噪声内")

M = {("GC10", "s"): read_main(f"{GC10_S}/table_main.tsv"), ("NEU", "s"): read_main(f"{NEU_S}/table_main.tsv"),
     ("GC10", "nano"): read_main(f"{GC10_N}/table_main.tsv"), ("NEU", "nano"): read_main(f"{NEU_N}/table_main.tsv")}
# NEU ext（v_full@320 / @960，2026-09-30 第二批训练；口径同主表：perclass_ap.json）
EXT_TABLES = f"{R}/neu_ext_20260930/small/tables/table_main_ext.tsv"
EXT_DEP = f"{R}/neu_ext_20260930/ext_deploy/runs_deploy_ext"
if os.path.isfile(EXT_TABLES):
    M[("NEU", "s")].update(read_main(EXT_TABLES))

def gf_of(cell, default=None):
    """GFLOPs（确定性量，与测量会话无关）：先查 ext 部署目录，再回落 NEU s 统一批"""
    import json, glob
    for d in (EXT_DEP, f"{R}/deploy_20260930/deploy_20260930_neu_s"):
        for fn in sorted(glob.glob(os.path.join(d, "%s_seed*.json" % cell))):
            return json.load(open(fn, encoding="utf-8"))["gflops"]
    return default

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
        out[f[0].strip()] = {"cell": f[1], "n": f[2], "val95": num(f[3]), "val50": num(f[4]),
                            "rare": f[5], "rest": f[6]}
    return out

IM = {"GC10": read_imbal(f"{GC10_S}/table_imbalance.tsv"), "NEU": read_imbal(f"{NEU_S}/table_imbalance.tsv")}

L = []
w = L.append
w("# paper3 表格定稿（2026-09-30）")
w("")
w("口径：best.pt 复评；val 与 test 分列；3 种子（42/7/2024）均值±**样本标准差**；参数量取 best.pt 实测值。")
w("显著性与否一律按 |Δ| > 2×合并标准差（sqrt((sd1²+sd2²)/2)）判定，其余写\"噪声内\"。")
w("")

ORDER_S = ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_p2dcsawiou", "v_full", "v_full_cpw0p5",
           "v_p1_b8", "v_base_b8", "v_base_s320_b8", "v_base_s960_b8",
           "v_full_s320_b8", "v_full_s960_b8", "v_resample", "v_focal",
           "v_base_e300", "v_full_e300"]
DESC = {"v_base": "YOLOv8s 基线", "v_p2": "+P2 检测头", "v_dcsa": "+DCSA 注意力", "v_wiou": "+WIoU v3",
        "v_p2dcsawiou": "+P2+DCSA+WIoU", "v_full": "上者 + cls_pw=1.0（四件套）",
        "v_full_cpw0p5": "四件套 + cls_pw=0.5", "v_p1_b8": "+P1 检测头(b8)", "v_base_b8": "基线(b8)",
        "v_base_s320_b8": "基线@320(b8)", "v_base_s960_b8": "基线@960(b8)", "v_resample": "重采样",
        "v_full_s320_b8": "四件套@320(b8)", "v_full_s960_b8": "四件套@960(b8)",
        "v_focal": "Focal BCE", "v_base_e300": "基线 300 轮", "v_full_e300": "四件套 300 轮"}

def pget(ds, c):
    """参数量查询：e300 后缀与同架构 200 轮 cell 相同"""
    d = PARAMS[(ds, "s")]
    if c in d:
        return d[c]
    if c.endswith("_e300") and c[:-5] in d:
        return d[c[:-5]]
    return 0

def table_main(ds, num_no, title, extra_note):
    t = M[(ds, "s")]
    base = t["v_base"]["test95"]
    w("## 表 %d %s" % (num_no, title))
    w("")
    w("| 配置 | 说明 | val mAP50-95 | val mAP50 | test mAP50-95 | test 相对基线 | 参数量 | 轮数 |")
    w("|---|---|---|---|---|---|---|---|")
    for c in ORDER_S:
        if c not in t:
            continue
        r = t[c]
        d, word = sig_word(base, r["test95"])
        dd = "—" if d is None else "%+.4f %s" % (d, word)
        w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c, DESC.get(c, r["variant"]), f2(r["val95"]), f2(r["val50"]), f2(r["test95"]), dd,
            format(pget(ds, c), ","), "300" if c.endswith("e300") else "200"))
    w("")
    w("表注：%s" % extra_note)
    w("")

table_main("GC10", 2, "主矩阵消融（GC10-DET，640/b16，10 类）",
           "参数量为 best.pt 实测（含检测头偏置），与训练日志 fused 口径不同；"
           "test 列为独立测试划分（345 张）。同一后缀 e300 表示训练预算改为 300 轮（见正文 4.x 节预算稳健性）。")
table_main("NEU", 5, "第二基准主矩阵（NEU-DET，640/b16，6 类）",
           "与表 2 同协议、同结构，仅数据集与类别数不同；NEU-DET 原始 200×200，上采样到 640。")

w("## 表 3 输入分辨率与浅层检测头（GC10-DET，统一 batch=8）")
w("")
w("| 配置 | val mAP50-95 | val mAP50 | test mAP50-95 | 相对 640 基线 | 参数量 |")
w("|---|---|---|---|---|---|")
t = M[("GC10", "s")]
ref = t["v_base_b8"]["val95"]
for c, nm in [("v_base_s320_b8", "640→320"), ("v_base_b8", "640（同批基准）"), ("v_base_s960_b8", "640→960"),
              ("v_p1_b8", "+P1 头（stride=2）"), ("v_p2", "+P2 头（b16，见表 2）")]:
    if c not in t:
        continue
    r = t[c]
    d, word = sig_word(ref, r["val95"])
    w("| %s | %s | %s | %s | %s | %s |" % (nm, f2(r["val95"]), f2(r["val50"]), f2(r["test95"]),
                                          "—" if d is None else "%+.4f %s" % (d, word), format(pget("GC10", c), ",")))
w("")
w("表注：基准行为 v_base_b8（batch 8 同批口径），其余行与之比较；分辨率与头结构改变时算力代价见\"部署表\"（表 7）。")
w("")

w("## 表 3-NEU 输入分辨率扫描（NEU-DET，s 尺度，batch=8）")
w("")
w("| 配置 | 输入(px) | val mAP50-95 | val mAP50 | test mAP50-95 | test 相对 640 同架构 | GFLOPs | 参数量 |")
w("|---|---|---|---|---|---|---|---|")
t = M[("NEU", "s")]
PX = {"v_base_s320_b8": "320", "v_base_b8": "640", "v_base_s960_b8": "960",
      "v_full_s320_b8": "320", "v_full": "640", "v_full_s960_b8": "960"}
ref_b = t["v_base_b8"]["test95"]
ref_f = t["v_full"]["test95"]
for c, nm, ref in [("v_base_s320_b8", "基线", ref_b), ("v_base_b8", "基线（640 基准行）", None),
                   ("v_base_s960_b8", "基线", ref_b),
                   ("v_full_s320_b8", "四件套", ref_f), ("v_full", "四件套（640 基准行）", None),
                   ("v_full_s960_b8", "四件套", ref_f)]:
    if c not in t:
        continue
    r = t[c]
    d, word = (None, "—") if ref is None else sig_word(ref, r["test95"])
    gf = gf_of(c)
    w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
        nm, PX[c], f2(r["val95"]), f2(r["val50"]), f2(r["test95"]),
        "—" if d is None else "%+.4f %s" % (d, word),
        ("%.1f" % gf) if gf else "—", format(pget("NEU", c), ",")))
w("")
w("表注：6 个 NEU-DET s 尺度 run 协议完全一致（200 轮、batch 8、同划分、种子 42/7/2024）；其中 320/960 的"
   "四件套为 2026-09-30 补充的第二批训练（同代码、同划分、同种子，与主矩阵连续队列不同批）。GFLOPs 与参数量是"
   "模型自身属性（确定性），可跨批比较；ONNX-CPU 延迟**不列**——ext 批是另一次测量会话，按 4.6 节口径只有同会话"
   "比较有效。ext 会话内部的同会话对照仍有效：四件套 320 vs 960 = 931.8 vs 3365.3 ms（+261%），"
   "test mAP50-95 却在种子噪声内（0.3464 vs 0.3513）。")
w("")
w("")

w("## 表 4 类别不平衡处置五臂（GC10-DET，同架构 P2+DCSA+WIoU，只换不平衡处置）")
w("")
w("| 臂 | val mAP50-95 | val mAP50 | 稀有类 AP50 | 其余类均值 AP50 | 相对无处理 |")
w("|---|---|---|---|---|---|")
base_arm = "无处理"
for arm, r in IM["GC10"].items():
    d, word = sig_word(IM["GC10"][base_arm]["val95"], r["val95"])
    w("| %s | %s | %s | %s | %s | %s |" % (arm, f2(r["val95"]), f2(r["val50"]), r["rare"], r["rest"],
                                          "—" if arm == base_arm else "%+.4f %s" % (d, word)))
w("")
w("表注：稀有类按训练集实例数最少的类判定（GC10：zhehen、yahen）；五臂共享同一架构与种子集合，"
   "差异仅在不平衡处置。\"显著\"为 |Δ|>2×合并标准差。")
w("")

w("## 表 8 跨尺度对照：两数据集 × 骨干尺度（5 个共享配置，640/b16）")
w("")
for key, name in [("val50", "val mAP50"), ("val95", "val mAP50-95"), ("test95", "test mAP50-95")]:
    w("**表 8a-%s**" % name if key == "val50" else "**表 8b-%s**" % name if key == "val95" else "**表 8c-%s**" % name)
    w("")
    w("| 配置 | GC10 s(11.1M) | GC10 n(3.0M) | Δ(nano−s) | NEU s(11.1M) | NEU n(3.0M) | Δ(nano−s) |")
    w("|---|---|---|---|---|---|---|")
    for c in ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_full"]:
        row = []
        for ds in ("GC10", "NEU"):
            a = M[(ds, "s")][c][key] if c in M[(ds, "s")] else None
            b = M[(ds, "nano")][c][key] if c in M[(ds, "nano")] else None
            d, word = sig_word(a, b)
            row += [f2(a), f2(b), "—" if d is None else "%+.4f %s" % (d, word)]
        w("| %s | %s | %s | %s | %s | %s | %s |" % (DESC[c], row[0], row[1], row[2], row[3], row[4], row[5]))
    w("")
w("表注：n = nano（YOLOv8n）尺度，s = YOLOv8s；两尺度同协议、同种子、同数据划分；参数量见括注（best.pt 实测）。"
   "Δ 为 nano 减 s，判据同全表。nano 组仅跑 5 个共享配置（其余 8 个配置只在 s 尺度评测）。")
w("")

w("## 表 9 类别不平衡处置的跨数据集复现（两数据集，s 尺度）")
w("")
w("| 臂 | GC10 val mAP50-95 | GC10 val mAP50 | NEU val mAP50-95 | NEU val mAP50 | 结论 |")
w("|---|---|---|---|---|---|")
for arm in IM["GC10"]:
    g, n = IM["GC10"].get(arm), IM["NEU"].get(arm)
    dg, wg = sig_word(IM["GC10"].get(base_arm, {}).get("val95"), g["val95"]) if g else (None, "—")
    dn, wn = sig_word(IM["NEU"].get(base_arm, {}).get("val95"), n["val95"]) if n else (None, "—")
    verdict = "两处皆噪声内" if (wg == "噪声内" and wn == "噪声内") else \
              ("两处皆显著" if (wg == "显著" and wn == "显著") else "一处显著/一处噪声内")
    if arm == base_arm:
        verdict = "对照"
    w("| %s | %s | %s | %s | %s | %s |" % (arm, f2(g["val95"]) if g else "—", f2(g["val50"]) if g else "—",
                                          f2(n["val95"]) if n else "—", f2(n["val50"]) if n else "—", verdict))
w("")
w("表注：GC10 稀有类 = zhehen/yahen（训练集最悬殊 11.4:1），NEU 稀有类 = pitted_surface/scratches（2.4:1）；"
   "两数据集的臂定义完全一致。Focal 在两数据集的 val mAP50 上一致下降（GC10 −0.076、NEU −0.023），"
   "是本表中唯一跨数据集一致的显著负向处置。")
w("")

w("## 表 7 部署指标（2 尺度 × 2 数据集，batch=1，单卡 RTX 4090；onnx_cpu = ONNX Runtime CPU）")
w("")
w("**待替换**：统一批次重测（服务器 15:11 起跑，四组同会话连续测量）完成后，本表用新数据整体替换；")
w("下列数值为分批测量的旧口径，仅作占位与趋势参考 —— 其中 ONNX-CPU 一列受宿主 CPU 负载影响，")
w("同一模型跨批次可差 27%（GC10 nano 497.9 ms vs NEU nano 630.0 ms），故不可跨批次比较。")
w("")

path = "./_paper/paper3_表格定稿_20260930.md"
open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
print("...")
print("已写入:", path, "(%d 行)" % len(L))
