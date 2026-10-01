# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_matrix.py - 兼容入口（2026-09-27 起改为直接调用 analyze_matrix.py）

历史问题（当日实测）：
    旧版的分组函数签名里有 batch 参数，但**从未真正按 batch 过滤**；主矩阵段落只靠
    `not dir.endswith("_b8")` 排除 b8 档，而目录名实际是 `v_base_s640_b8_seed7`（结尾是 `_seed7`），
    过滤条件永远为真 → 分辨率扫描的 b8 run 被并进主变体均值（v_base n=6）；
    `v_full` 段也没按 cls_pw 分档 → cpw=0.5 的三个 run 被并进 cpw=1.0（v_full n=6）。
    后果：均值±标准差口径被污染（逐种子原始行仍是真的，json 没问题）。
    另外它的"缺失 cell"检查只覆盖 6 个主变体，看不到分辨率/P1/重采样/Focal 的缺口。

修法：不再维护第二套分组逻辑 —— 统一走 analyze_matrix.py 的"按规范 cell 名精确匹配"
（cell 名由 defect_train.run_name 的同一规则生成，两侧已逐名 diff 验证 39/39）。

用法（与旧版一致）:
    python collect_matrix.py            # 默认 runs/，表写到 tables/
    python collect_matrix.py runs --out tables
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "analyze_matrix.py")

if not os.path.isfile(TARGET):
    raise SystemExit("找不到 analyze_matrix.py（应与 collect_matrix.py 同目录）")

sys.argv = [TARGET] + sys.argv[1:]
runpy.run_path(TARGET, run_name="__main__")
