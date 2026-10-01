#!/bin/bash
# NOTE: paths below were machine-specific and have been replaced with ${DATA_ROOT}.
# Set DATA_ROOT (and DATA/PROJ inside) to match your own layout before running.
#!/bin/bash
# run_matrix_ext.sh - 第二阶段（可选）：部署 Pareto 用——全改进模型在 320/960 下的精度与代价
# 用途：论文里的"精度-延迟 Pareto"曲线需要全改进模型（v_full）在不同输入分辨率下的同批数据；
#       分辨率扫描（run_matrix.sh 里那三个）跑的是裸基线 v_base，两者互补。
# 用法: bash run_matrix_ext.sh [GPU]      （已完成的 cell 自动跳过）
set -u
cd "$(dirname "$0")"
DATA=${DATA_ROOT}/neu/data/NEU-DET-split/neu_det.yaml
PROJ=${DATA_ROOT}/neu/runs
GPU=${1:-0}
EP=${EPOCHS:-200}
SEEDS=(42 7 2024)
CELLS=( "v_full|320|8|1.0" "v_full|960|8|1.0" )

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

echo "GPU=${GPU} epochs=${EP} cells=${#CELLS[@]}（第二阶段：部署 Pareto）"
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r VAR IMG BS CPW <<< "$CELL"
  for SEED in "${SEEDS[@]}"; do
    NAME=$(cell_name "$VAR" "$IMG" "$BS" "$CPW" "$SEED")
    if [ -f "${PROJ}/${NAME}/perclass_ap.json" ]; then echo "skip ${NAME} (done)"; continue; fi
    echo "==== ${NAME} imgsz=${IMG} batch=${BS} ===="
    python defect_train.py --variant "$VAR" --seed "$SEED" --data "$DATA" \
      --imgsz "$IMG" --batch "$BS" --cls-pw "$CPW" \
      --epochs "$EP" --device "$GPU" --workers 8 --project "$PROJ" --final-test
    RC=$?
    echo "---- ${NAME} exit=${RC} ----"
    if [ "$RC" != "0" ]; then echo "!! ${NAME} 非 0 退出，停止队列"; exit "$RC"; fi
  done
done
echo "EXT DONE"
