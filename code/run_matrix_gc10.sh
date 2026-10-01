#!/bin/bash
# NOTE: paths below were machine-specific and have been replaced with ${DATA_ROOT}.
# Set DATA_ROOT (and DATA/PROJ inside) to match your own layout before running.
#!/bin/bash
# run_gc10.sh - Paper3 GC10-DET 矩阵（主线数据集：10 类冷轧缺陷，25% 目标 <32px@640，不平衡 12:1）
# 与 NEU 版结构完全一致：主矩阵 + 分辨率扫描 + 类加权指数 + P1 头
#
# 用法:
#   bash run_gc10.sh                 # 全跑（已完成 cell 自动跳过）
#   bash run_gc10.sh 0               # 指定 GPU
#   EPOCHS=300 bash run_gc10.sh      # 改轮数（默认 200）
#   DRY=1 bash run_gc10.sh           # 只打印将跑什么
#
# 目录名规则必须与 defect_train.py 的 run_name() 一致（两侧已逐名 diff 验证过）。
set -u
cd "$(dirname "$0")"
ROOT=${DATA_ROOT}/gc10
DATA=${ROOT}/data/GC10-DET-yolo/gc10_det.yaml
P2Y=${ROOT}/data/GC10-DET-yolo/yolov8s-p2-gc10.yaml
P1Y=${ROOT}/data/GC10-DET-yolo/yolov8s-p1-gc10.yaml
PROJ=${ROOT}/runs
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
  C=$(printf '%.1f' "$4")
  [ "$2" != "640" ] && tag="${tag}_s$2"
  [ "$3" != "16" ] && tag="${tag}_b$3"
  case "$1" in
    v_full)       if [ "$C" != "1.0" ]; then tag="${tag}_cpw$(echo "$C" | tr '.' 'p')"; fi ;;
    v_p2dcsawiou) if [ "$C" != "0.0" ]; then tag="${tag}_cpw$(echo "$C" | tr '.' 'p')"; fi ;;
  esac
  echo "${tag}_seed$5"
}

echo "GC10 矩阵：GPU=${GPU} epochs=${EP} cells=${#CELLS[@]} seeds=${SEEDS[*]} dry=${DRY}"
echo "数据: ${DATA}"
T0=$(date +%s)
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r VAR IMG BS CPW <<< "$CELL"
  for SEED in "${SEEDS[@]}"; do
    NAME=$(cell_name "$VAR" "$IMG" "$BS" "$CPW" "$SEED")
    if [ -f "${PROJ}/${NAME}/perclass_ap.json" ]; then echo "skip ${NAME} (done)"; continue; fi
    FT=""
    case "$VAR" in v_full) FT="--final-test" ;; esac
    echo "==== ${NAME}  variant=${VAR} imgsz=${IMG} batch=${BS} cls_pw=${CPW} epochs=${EP} ===="
    if [ "$DRY" = "1" ]; then continue; fi
    python defect_train.py --variant "$VAR" --seed "$SEED" --data "$DATA" \
      --p2-yaml "$P2Y" --p1-yaml "$P1Y" \
      --imgsz "$IMG" --batch "$BS" --cls-pw "$CPW" \
      --epochs "$EP" --device "$GPU" --workers 8 --project "$PROJ" $FT
    RC=$?
    echo "---- ${NAME} exit=${RC} ----"
    if [ "$RC" != "0" ]; then echo "!! ${NAME} 非 0 退出，停止队列"; exit "$RC"; fi
  done
done
T1=$(date +%s)
echo "GC10 ALL DONE  用时 $(( (T1 - T0) / 60 )) 分钟"
echo "收数: python collect_matrix.py ${PROJ}"
