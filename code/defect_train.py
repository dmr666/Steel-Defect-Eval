# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
defect_train.py - Paper 3 (NEU-DET steel surface defect detection) training.

Protocol (see PLAN_研究方案.md):
  imgsz=640, epochs=200(default), batch=16, SGD lr0=0.01, cos_lr, warmup=3,
  momentum=0.937, weight_decay=5e-4, close_mosaic=epochs-50, degrees=10,
  mixup=0, copy_paste=0, patience=0, COCO pretrained, seeds 42/7/2024.
  best-epoch on val(162); optional final report on test(180).

Variants (6 configs, all @640):
  v_base       : stock YOLOv8s
  v_p2         : + P2 head (stride=4)
  v_dcsa       : + C2f_DCSA on backbone layers 4/6/8
  v_wiou       : + WIoU v3
  v_p2dcsawiou : + P2 + C2f_DCSA + WIoU
  v_full       : v_p2dcsawiou + class-weighted cls loss (ultralytics built-in cls_pw=1.0, inverse frequency)

Note on cls_pw (ultralytics 8.4.115, verified against cfg/default.yaml + engine/trainer.py + models/yolo/detect/train.py):
  cls_pw is a FLOAT exponent in [0,1] (0.0=disable, 1.0=full inverse frequency);
  the trainer computes inverse-frequency weights, normalizes them to mean 1.0 and
  attaches them to model.class_weights, which nn/loss.py multiplies into the BCE term.
  Do NOT pass a per-class list here (assert 0 <= cls_pw <= 1.0 would fail).

Usage:
  python defect_train.py --variant v_full --seed 42 \
      --data ${DATA_ROOT}/neu/data/NEU-DET-split/neu_det.yaml \
      --project ${DATA_ROOT}/neu/runs [--epochs 4] [--final-test]
"""
import os
import argparse
import json

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Bottleneck

# ---------------------------------------------------------------------
# DCSA (CBAM-style), identical to paper-2 verified code
# ---------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.gap(x)) + self.mlp(self.gmp(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DCSA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Bottleneck_DCSA(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.dcsa = DCSA(c2)

    def forward(self, x):
        y = self.cv1(x)
        y = self.cv2(y)
        y = self.dcsa(y)
        return x + y if self.add else y


class C2f_DCSA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.ModuleList([
            Bottleneck_DCSA(c_, c_, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        ])


DCSA_LAYER_INDICES = (4, 6, 8)


def _layer_seq(obj):
    """把 YOLO包装 / DetectionModel / nn.Sequential 统一取到层的 nn.Sequential。
    （parse_model 返回的本来就是 Sequential；trainer 重建路径与 .pt 加载路径给的是包装对象）"""
    seen = 0
    while not isinstance(obj, nn.Sequential) and hasattr(obj, "model") and seen < 3:
        obj = obj.model
        seen += 1
    return obj


def replace_c2f_with_dcsa(model, layer_indices=DCSA_LAYER_INDICES):
    layers = _layer_seq(model)
    replaced = 0
    for i in layer_indices:
        layer = layers[i]
        if isinstance(layer, C2f) and not isinstance(layer, C2f_DCSA):
            c1 = layer.cv1.conv.in_channels
            c2 = layer.cv2.conv.out_channels
            n = len(layer.m)
            shortcut = layer.m[0].add
            new = C2f_DCSA(c1, c2, n=n, shortcut=shortcut)
            for attr in ("i", "f", "type", "np"):   # 图属性必须抄过去，否则前向报 no attribute 'f'
                if hasattr(layer, attr):
                    setattr(new, attr, getattr(layer, attr))
            layers[i] = new
            replaced += 1
            print(f"  [dcsa] layer {i}: C2f -> C2f_DCSA (c1={c1}, c2={c2}, n={n})")
    return replaced


# =====================================================================
# DCSA 注入的“真·入口”：包装 ultralytics.nn.tasks.parse_model
# ---------------------------------------------------------------------
# 为什么必须这样做（2026-09-26 本地实测，与论文二 2026-09-14 踩的是同一个坑）：
#   ultralytics 的 Model.train() 里有
#       self.trainer.model = self.trainer.get_model(weights=..., cfg=self.model.yaml)
#   即**训练前会用 yaml 重建模型**，把训练前在内存里替换掉的层静默丢弃。
#   实测证据：只做 in-place 替换时，训练日志里 DCSA 参数量增量出现，但
#   runs/<name>/weights/best.pt 里 C2f_DCSA 模块数 = 0、参数量回到纯模型值，
#   也就是白跑一场、结果其实是裸模型（论文二当时 15 个 run 全部如此）。
#   包装 parse_model 后，任何建模路径（trainer 重建、它生成的 EMA）都会注入 DCSA。
# =====================================================================
def install_dcsa_parse_patch(verbose=True):
    import ultralytics.nn.tasks as tasks
    if getattr(tasks, "_hermes_dcsa_patched", False):
        return
    _orig = tasks.parse_model

    def _patched_parse(cfg, ch, *args, **kwargs):
        model, save = _orig(cfg, ch, *args, **kwargs)
        n = replace_c2f_with_dcsa(model)
        if not n:
            raise RuntimeError(
                "parse_model 包装后未替换到任何 C2f —— 层结构变了，停下排查，不要开跑")
        return model, save

    tasks.parse_model = _patched_parse
    tasks._hermes_dcsa_patched = True
    if verbose:
        print("  [dcsa] parse_model 已包装：此后任何建模（含 trainer 重建与 EMA）都自动注入 C2f_DCSA（层 4/6/8）")


def verify_dcsa(model, expect=DCSA_LAYER_INDICES):
    """断言模型里第 4/6/8 层确实是 C2f_DCSA，否则立刻抛错（不要带着坏配置开跑）。"""
    seq = list(_layer_seq(model))
    types = [type(l).__name__ for l in seq]
    got = [i for i in expect if i < len(types) and types[i] == "C2f_DCSA"]
    print("  [dcsa] 校验：层 %s 实际类型 = %s" % (list(expect), [types[i] for i in expect]))
    if got != list(expect):
        raise RuntimeError("C2f_DCSA 未进入模型（注入失败），不要开跑")
    return True


def verify_p1(model, expect=(2, 8, 16, 32)):
    """P1 头结构自检：Detect 的 strides 必须是 [2, 8, 16, 32]（P1/P3/P4/P5，不含 P2）。"""
    det = list(_layer_seq(model))[-1]
    try:
        strides = [int(s) for s in det.stride.tolist()]
    except Exception as e:
        raise RuntimeError(f"取 Detect strides 失败（结构不对）: {e}")
    print(f"  [p1] Detect strides = {strides}（期望 {list(expect)}）")
    if strides != list(expect):
        raise RuntimeError("P1 头结构不对（strides 与预期不符），不要开跑")
    return strides


def checkpoint_has_dcsa(path):
    """字节扫 best.pt：必须出现 DCSA 类名之一（pickle 记忆化，出现 >=1 次即可）。"""
    blob = open(path, "rb").read()
    keys = ("C2f_DCSA", "Bottleneck_DCSA", "ChannelAttention", "SpatialAttention", "DCSA")
    hits = {k: blob.count(k.encode()) for k in keys}
    return any(hits.values()), hits


# ---------------------------------------------------------------------
# WIoU v3 (identical to paper-2 verified code)
# ---------------------------------------------------------------------

def apply_wiou(momentum=0.1, alpha=1.9, delta=3):
    import ultralytics.utils.loss as loss_mod
    if getattr(loss_mod, "_hermes_wiou_patched", False):
        return
    _orig = loss_mod.bbox_iou
    state = {"l_iou_mean": 1.0}

    def _wrapper(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, *a, **k):
        if not CIoU:
            return _orig(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps, *a, **k)
        iou = _orig(box1, box2, xywh=xywh, GIoU=False, DIoU=False, CIoU=False, eps=eps)
        b1, b2 = box1, box2
        if xywh:
            (x1, y1, w1, h1) = b1.chunk(4, -1)
            (x2, y2, w2, h2) = b2.chunk(4, -1)
            b1_x1, b1_y1 = x1 - w1 / 2, y1 - h1 / 2
            b1_x2, b1_y2 = x1 + w1 / 2, y1 + h1 / 2
            b2_x1, b2_y1 = x2 - w2 / 2, y2 - h2 / 2
            b2_x2, b2_y2 = x2 + w2 / 2, y2 + h2 / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = b1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = b2.chunk(4, -1)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        rho2 = (b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 / 4 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2 / 4
        R = torch.exp(rho2 / (cw ** 2 + ch ** 2 + 1e-7)).detach().clamp(max=10.0)
        L = (1.0 - iou).detach()
        beta = (L / (state["l_iou_mean"] + 1e-7)).clamp(max=10.0)
        r = (beta / (delta * torch.pow(alpha, beta - delta))).clamp(max=10.0).detach()
        state["l_iou_mean"] = (1 - momentum) * state["l_iou_mean"] + momentum * L.mean().item()
        return 1.0 - r * R * (1.0 - iou)

    loss_mod.bbox_iou = _wrapper
    loss_mod._hermes_wiou_patched = True
    print("  [wiou] patched")


# ---------------------------------------------------------------------
# Focal BCE（不平衡对照臂用）：只替换 v8DetectionLoss.__init__ 里的 self.bce
#   ultralytics 8.4.115 的分类损失是 self.bce = nn.BCEWithLogitsLoss(reduction="none")，
#   之后 bce_loss *= self.class_weights、再 .sum()/target_scores_sum —— 所以只要换掉 self.bce
#   这个模块本身（返回逐元素损失），其余流程不变。
# ---------------------------------------------------------------------

class FocalBCE(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        import torch.nn.functional as F
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        if self.alpha is not None and self.alpha >= 0:
            a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = a_t * loss
        return loss


def apply_focal(gamma=1.5, alpha=0.25):
    import ultralytics.utils.loss as loss_mod
    if getattr(loss_mod, "_hermes_focal_patched", False):
        return
    cls = loss_mod.v8DetectionLoss
    orig_init = cls.__init__

    def patched_init(self, model, *a, **k):
        orig_init(self, model, *a, **k)
        self.bce = FocalBCE(gamma=gamma, alpha=alpha)

    cls.__init__ = patched_init
    loss_mod._hermes_focal_patched = True
    print(f"  [focal] 分类损失已换成 Focal BCE (gamma={gamma}, alpha={alpha})")


# ---------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------

VARIANTS = ["v_base", "v_p1", "v_p2", "v_dcsa", "v_wiou", "v_p2dcsawiou", "v_full",
            "v_resample", "v_focal"]
CW_VARIANTS = ("v_full", "v_p2dcsawiou")          # 支持类加权的变体
DEFAULT_CLS_PW = {"v_full": 1.0}                   # 其余变体默认 0.0（不加权）
# 不平衡对照臂（2026-09-27 加）：模型结构与 v_p2dcsawiou 完全相同，只换"不平衡处理方式"——
#   v_resample：训练用重采样列表（逆频平方根重复，见 code/make_resampled_list.py），cls_pw=0
#   v_focal   ：分类损失换成 Focal BCE（gamma/alpha 可调），cls_pw=0
# 这样 4 个臂（不加权 / cls_pw=1.0 / 重采样 / Focal）之间只差一个因素。
DCSA_VARIANTS = ("v_dcsa", "v_p2dcsawiou", "v_full", "v_resample", "v_focal")
WIOU_VARIANTS = ("v_wiou", "v_p2dcsawiou", "v_full", "v_resample", "v_focal")
P2_VARIANTS = ("v_p2", "v_p2dcsawiou", "v_full", "v_resample", "v_focal")
RESAMPLE_VARIANTS = ("v_resample",)
FOCAL_VARIANTS = ("v_focal",)
NEU_NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
             "rolled-in_scale", "scratches"]


def guess_base_yaml(data_yaml=None, p2_yaml=None):
    """找一个"标准检测头 + 本数据集 nc"的 base yaml。

    为什么需要：`YOLO("xxx.pt")` 是**反序列化**，不经过 parse_model —— 靠 parse_model 包装注入的
    自定义模块（本项目的 C2f_DCSA）不会出现在这种模型里。实测在 AutoDL 上：v_dcsa 变体因为
    直接 load .pt，verify_dcsa 直接抛 "C2f_DCSA 未进入模型（注入失败）" 并中止队列。
    所以"需要注入的变体"必须**先按 yaml 建模**（触发 parse_model），再 load 预训练权重。
    """
    if not data_yaml:
        return "yolov8s.yaml"
    d = os.path.dirname(os.path.abspath(data_yaml))
    stem = os.path.basename(data_yaml).replace(".yml", "").replace(".yaml", "").replace("_det", "")
    cands = []
    if stem:
        cands.append(os.path.join(d, "yolov8s-%s.yaml" % stem))
    try:
        for f in sorted(os.listdir(d)):
            if f.startswith("yolov8s-") and f.endswith(".yaml") and "-p1" not in f and "-p2" not in f:
                cands.append(os.path.join(d, f))
    except OSError:
        pass
    for c in cands:
        if os.path.isfile(c):
            return c
    return "yolov8s.yaml"


def build_model(variant, p2_yaml, p1_yaml=None, weights="yolov8s.pt", base_yaml=None,
                data_yaml=None):
    v = variant.lower()
    needs_dcsa = v in DCSA_VARIANTS
    random_init = str(weights).endswith((".yaml", ".yml"))  # 冒烟用：不加载预训练
    if needs_dcsa:
        install_dcsa_parse_patch()      # 必须在任何 YOLO(...) 之前安装
    if v == "v_p1":
        model = YOLO(p1_yaml)
        if not random_init:
            model = model.load(weights)
    elif v in ("v_base", "v_dcsa", "v_wiou"):
        if needs_dcsa and not random_init:
            y = base_yaml or guess_base_yaml(data_yaml, p2_yaml)
            print(f"  [dcsa] 需要注入 -> 先按 yaml 建模（触发 parse_model），再加载预训练权重；base yaml = {y}")
            model = YOLO(y).load(weights)
        else:
            model = YOLO(weights)
    else:  # v_p2 / v_p2dcsawiou / v_full
        model = YOLO(p2_yaml)
        if not random_init:
            model = model.load(weights)
    if v == "v_p1":
        verify_p1(model)
    if needs_dcsa:
        verify_dcsa(model)
    if v in WIOU_VARIANTS:
        apply_wiou()
    if v in FOCAL_VARIANTS:
        apply_focal()
    params = sum(p.numel() for p in model.model.parameters())
    print(f"  [params] {params:,} ({params/1e6:.2f}M)")
    return model, params


def resolve_cls_pw(variant, cli_value):
    """CLI 未给 --cls-pw 时按变体取默认：v_full=1.0（全逆频），其余 0.0（关闭）。"""
    if cli_value is not None:
        return float(cli_value)
    return float(DEFAULT_CLS_PW.get(variant, 0.0))


def run_name(variant, seed, imgsz, batch, cls_pw):
    """目录/收数名：非默认配置编进名字，run_matrix.sh 的跳过判定与这里必须一致。"""
    tag = variant
    if imgsz != 640:
        tag += f"_s{imgsz}"
    if batch != 16:
        tag += f"_b{batch}"
    if variant in CW_VARIANTS and cls_pw != float(DEFAULT_CLS_PW.get(variant, 0.0)):
        tag += f"_cpw{str(cls_pw).replace(chr(46), chr(112))}"
    return f"{tag}_seed{seed}"


def class_weight_report(data_yaml, variant, cls_pw):
    """报告数据集类别频数与理论逆频权重（供论文写公式/数值），实际加权交给 ultralytics 内置 cls_pw。

    仅 v_full 启用；返回要传给 trainer 的 cls_pw 浮点值（0.0 = 关闭）。
    """
    if variant not in CW_VARIANTS or float(cls_pw) <= 0.0:
        return 0.0
    stats_path = os.path.join(os.path.dirname(os.path.abspath(data_yaml)),
                              "split_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        counts = stats["stats"]["train"]["instances"]
        mx = max(counts)
        raw = [mx / c for c in counts]
        mean = sum(raw) / len(raw)
        norm = [x / mean for x in raw]
        print("  [clspw] train instance counts:", counts, f"(total {sum(counts)})")
        print("  [clspw] raw inverse-freq w_i = max(count)/count_i:",
              [round(x, 3) for x in raw])
        print("  [clspw] normalized to mean=1 (as the trainer attaches to model.class_weights):",
              [round(x, 3) for x in norm])
    else:
        print("  [clspw] split_stats.json not found -> only the exponent is used")
    print(f"  [clspw] ultralytics cls_pw exponent = {cls_pw} "
          f"(0.0 = off, 1.0 = full inverse frequency)")
    return float(cls_pw)


def per_class_ap(model, data, imgsz, batch, device, split="val"):
    try:
        m = model.val(data=data, imgsz=imgsz, batch=batch, device=device,
                      split=split, plots=False, verbose=False)
    except Exception as e:
        print(f"  [val] fallback (err={e})")
        m = model.val(data=data, imgsz=imgsz, batch=batch, device=device,
                      split=split, plots=False)
    out = {"mAP50": float(m.box.map50), "mAP5095": float(m.box.map)}
    try:
        all_ap = m.box.all_ap
        idx = m.box.ap_class_index
        names = [m.names[i] for i in idx]
        per = {}
        for k, ci in enumerate(idx):
            per[names[k]] = {"ap50": float(all_ap[k, 0]),
                             "ap5095": float(all_ap[k].mean())}
        out["classes"] = per
    except Exception as e:
        print(f"  [val] per-class unavailable: {e}")
    print(f"  [{split}] mAP50={out['mAP50']:.4f} mAP50-95={out['mAP5095']:.4f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="${DATA_ROOT}/neu/data/NEU-DET-split/neu_det.yaml")
    p.add_argument("--p2-yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "yolov8s-p2.yaml"))
    p.add_argument("--base-yaml", default="",
                   help="标准检测头的 base yaml（需要 parse_model 注入的变体用它建模）；默认自动在数据目录里找 yolov8s-<数据集>.yaml")
    p.add_argument("--p1-yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "yolov8s-p1.yaml"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="${DATA_ROOT}/neu/runs")
    p.add_argument("--cls-pw", type=float, default=None,
                   help="ultralytics cls_pw exponent in [0,1]；不传时按变体默认（v_full=1.0，其余=0.0）")
    p.add_argument("--weights", default="yolov8s.pt",
                   help="初始化权重：.pt（默认，AutoDL 会用/下载）/ .yaml（随机初始化，仅供本地冒烟）")
    p.add_argument("--no-pretrained", action="store_true",
                   help="随机初始化冒烟时用（不给 ultralytics 传 pretrained，仅供本地自测）")
    p.add_argument("--final-test", action="store_true",
                   help="also evaluate best.pt on test split")
    args = p.parse_args()

    cls_pw = resolve_cls_pw(args.variant, args.cls_pw)
    name = run_name(args.variant, args.seed, args.imgsz, args.batch, cls_pw)
    print(f"\n=== defect variant={args.variant} seed={args.seed} "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} cls_pw={cls_pw} name={name} ===")
    data_yaml = args.data
    if args.variant in RESAMPLE_VARIANTS:
        data_yaml = (args.data if args.data.endswith("_resampled.yaml")
                     else args.data.replace(".yaml", "_resampled.yaml"))
        if not os.path.exists(data_yaml):
            raise SystemExit(f"重采样臂需要 {data_yaml}（先生成 train_resampled.txt 与 *_resampled.yaml）")
        print(f"  [resample] 本臂训练数据改用重采样列表: {data_yaml}")
    model, params = build_model(args.variant, args.p2_yaml, args.p1_yaml, args.weights,
                                base_yaml=args.base_yaml, data_yaml=data_yaml)
    cls_pw = class_weight_report(data_yaml, args.variant, cls_pw)

    cm = max(0, args.epochs - 50)  # close_mosaic; 0 for short smoke runs
    train_kw = dict(
        data=data_yaml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, project=args.project, name=name,
        lr0=0.01, cos_lr=True, warmup_epochs=3, optimizer="SGD", exist_ok=True,
        momentum=0.937, weight_decay=5e-4, close_mosaic=cm,
        pretrained=(not args.no_pretrained), save=True, plots=True, cache=True,
        mixup=0.0, copy_paste=0.0, degrees=10.0, patience=0, seed=args.seed,
        cls_pw=cls_pw,
    )
    model.train(**train_kw)

    print("\n=== val best.pt ===")
    best = os.path.join(args.project, name, "weights", "best.pt")
    if os.path.exists(best):
        model = YOLO(best)
    else:
        print("  [val] WARNING best.pt missing")
    val = per_class_ap(model, data_yaml, args.imgsz, args.batch, args.device, "val")
    val["variant"] = args.variant
    val["seed"] = args.seed
    val["imgsz"] = args.imgsz
    val["params"] = params
    val["cls_pw"] = cls_pw
    val["data_yaml"] = os.path.basename(data_yaml)
    val["imbalance_arm"] = ("resample" if args.variant in RESAMPLE_VARIANTS else
                            "focal" if args.variant in FOCAL_VARIANTS else
                            "cls_pw" if cls_pw > 0 else "none")
    # 存档自检：需要 DCSA 的变体，best.pt 里必须能找到 DCSA 类名，否则这轮结果无效
    if os.path.exists(best):
        has, hits = checkpoint_has_dcsa(best)
        val["ckpt_has_dcsa"] = bool(has)
        need = args.variant in DCSA_VARIANTS
        if need and not has:
            print("\n" + "!" * 78)
            print("!! [dcsa] best.pt 里查不到 DCSA —— 本轮结果无效（模型其实是纯基线），请勿采信 !!")
            print("!" * 78 + "\n")
        elif need:
            print(f"  [dcsa] 存档自检通过：best.pt 含 DCSA 痕迹 {hits}")
    if args.final_test and os.path.exists(best):
        tst = per_class_ap(YOLO(best), data_yaml, args.imgsz, args.batch,
                           args.device, "test")
        val["test"] = tst

    out_dir = os.path.join(args.project, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "perclass_ap.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(out_dir, "perclass_ap.json"))


if __name__ == "__main__":
    main()
