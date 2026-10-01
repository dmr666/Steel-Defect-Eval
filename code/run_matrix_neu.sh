#!/bin/bash
# NOTE: paths below were machine-specific and have been replaced with ${DATA_ROOT}.
# Set DATA_ROOT (and DATA/PROJ inside) to match your own layout before running.
#!/bin/bash
# run_matrix.sh - Paper3 NEU-DET 主矩阵 + 分辨率扫描 + 类加权扫描 + P1 头（AutoDL, 2026-09-26 版）
#
# 用法:
#   bash run_matrix.sh              # 全跑（已完成的 cell 自动跳过）
#   bash run_matrix.sh 0            # 指定 GPU 0
#   EPOCHS=300 bash run_matrix.sh   # 改轮数（默认 200；改协议才用）
#   DRY=1 bash run_matrix.sh        # 只打印将要跑什么，不训练
#
# cell 三项参数：variant | imgsz | batch | cls_pw
#   · 主矩阵 6 变体 @640 batch16（原有纪律：batch 与种子对齐）
#   · 分辨率扫描：v_base @320/640/960，**统一 batch 8**（跨分辨率必须同 batch，否则比的是 batch）
#     其中 v_base@640_b8 同时充当 P1 头的同批对照
#   · 类加权指数三档（同一架构）：v_p2dcsawiou@0（主矩阵，未加权）→ v_full@cpw0.5 → v_full@cpw1.0（主矩阵）
#     注意：v_p2dcsawiou@cpw1.0 与 v_full 是**同一个配置**（只是目录名不同），故不重复设 cell
#   · P1 头：v_p1 @640 batch 8（P1 anchor 网格 320x320/图，约 P2 方案的 3.3 倍，故降 batch 防空 assigner）
# 目录名规则必须与 defect_train.py 的 run_name() 完全一致，否则跳过判定失效。
set -u
cd "$(dirname "$0")"
DATA=${DATA_ROOT}/neu/data/NEU-DET-split/neu_det.yaml
PROJ=${DATA_ROOT}/neu/runs
GPU=${1:-0}
EP=${EPOCHS:-200}
DRY=${DRY:-0}
SEEDS=(42 7 2024)

CELLS=(
  "v_base|640|16|0"
  "v_p2|640|16|0"
  "v_dcsa|640|16|0"
  "v_wiou|640|16|0"
  "v_p2dcsawiou|640|16|0"
  "v_full|640|16|1.0"
  "v_base|320|8|0"
  "v_base|640|8|0"
  "v_base|960|8|0"
  "v_full|640|16|0.5"
  "v_p1|640|8|0"
  "v_resample|640|16|0"
  "v_focal|640|16|0"
)

cell_name() {   # variant imgsz batch cls_pw seed
  local tag="$1"
  local C
  C=$(printf '%.1f' "$4")            # 归一化：0 -> 0.0，1 -> 1.0，避免与 python 的 0.0/1.0 比较时错位
  [ "$2" != "640" ] && tag="${tag}_s$2"
  [ "$3" != "16" ] && tag="${tag}_b$3"
  case "$1" in
    v_full)       if [ "$C" != "1.0" ]; then tag="${tag}_cpw$(echo "$C" | tr '.' 'p')"; fi ;;
    v_p2dcsawiou) if [ "$C" != "0.0" ]; then tag="${tag}_cpw$(echo "$C" | tr '.' 'p')"; fi ;;
  esac
  echo "${tag}_seed$5"
}

echo "GPU=${GPU} epochs=${EP} cells=${#CELLS[@]} seeds=${SEEDS[*]} dry=${DRY}"
T0=$(date +%s)
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r VAR IMG BS CPW <<< "$CELL"
  for SEED in "${SEEDS[@]}"; do
    NAME=$(cell_name "$VAR" "$IMG" "$BS" "$CPW" "$SEED")
    if [ -f "${PROJ}/${NAME}/perclass_ap.json" ]; then
      echo "skip ${NAME} (done)"
      continue
    fi
    FT=""
    case "$VAR" in v_full) FT="--final-test" ;; esac
    echo "==== ${NAME}  variant=${VAR} imgsz=${IMG} batch=${BS} cls_pw=${CPW} epochs=${EP} ===="
    if [ "$DRY" = "1" ]; then continue; fi
    python defect_train.py --variant "$VAR" --seed "$SEED" --data "$DATA" \
      --imgsz "$IMG" --batch "$BS" --cls-pw "$CPW" \
      --epochs "$EP" --device "$GPU" --workers 8 --project "$PROJ" $FT
    RC=$?
    echo "---- ${NAME} exit=${RC} ----"
    if [ "$RC" != "0" ]; then
      echo "!! ${NAME} 非 0 退出，已停止整个队列（避免静默跑错配置）。修好后重跑本脚本会跳过已完成 cell。"
      exit "$RC"
    fi
  done
done
T1=$(date +%s)
echo "ALL DONE  用时 $(( (T1 - T0) / 60 )) 分钟"
echo "收数: python collect_matrix.py   （或 README 第 5 节里的内联脚本）"
