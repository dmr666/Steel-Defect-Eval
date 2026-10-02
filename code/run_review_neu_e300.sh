# NOTE: machine-specific paths were replaced with ${DATA_ROOT}/. placeholders.
#!/bin/bash
# run_review_neu_e300.sh —— review Minor 9：NEU-DET 的 300 轮加固（对齐 GC10 已有的 e300 对照）
#
# cells: v_base / v_full，各 3 种子，300 轮 → 6 个 run（NEU 每 run 约 30 分钟）
# 写入独立目录 $ROOT/runs_e300n（不与被 200 轮目录重名）；跳过判定兼容两种命名。
#
# 用法：bash run_review_neu_e300.sh 0
set -u
PY=${PY:-${CONDA_ROOT}/bin/python}   # 非交互 SSH 里 python 不在 PATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPU=${1:-0}
EP=300
DRY=${DRY:-0}
ROOT=${DATA_ROOT}/neu
DATA=$ROOT/data/NEU-DET-split/neu_det.yaml
PROJ=$ROOT/runs_e300n
SEEDS=(42 7 2024)

[ -f "$DATA" ] || { echo "缺 $DATA"; exit 1; }
for CELL in "v_base|0.0" "v_full|1.0"; do
  IFS='|' read -r VAR CPW <<< "$CELL"
  for SEED in "${SEEDS[@]}"; do
    if [ -f "${PROJ}/${VAR}_e${EP}_seed${SEED}/perclass_ap.json" ] || [ -f "${PROJ}/${VAR}_seed${SEED}/perclass_ap.json" ]; then
      echo "skip ${VAR} seed${SEED} (done)"; continue
    fi
    [ "$DRY" = "1" ] && { echo "would run ${VAR} seed${SEED} epochs=${EP} cls_pw=${CPW}"; continue; }
    echo "==== ${VAR} seed${SEED} epochs=${EP} cls_pw=${CPW} ===="
    "$PY" "$ROOT/defect_train.py" --variant "$VAR" --seed "$SEED" --data "$DATA" \
      --p2-yaml "$ROOT/yolov8s-p2.yaml" --base-yaml "$ROOT/yolov8s-neu.yaml" \
      --weights "$ROOT/yolov8s.pt" --imgsz 640 --batch 16 --cls-pw "$CPW" \
      --epochs "$EP" --device "$GPU" --workers 8 --project "$PROJ" --final-test
    RC=$?
    echo "---- ${VAR} seed${SEED} exit=${RC} ----"
    [ "$RC" != "0" ] && { echo "!! 非 0 退出，停止队列"; exit "$RC"; }
  done
done
echo
echo "NEU E300 ALL DONE"
