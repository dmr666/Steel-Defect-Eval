# NOTE: machine-specific paths were replaced with ${DATA_ROOT}/. placeholders.
#!/bin/bash
# run_review_scratch_e300.sh —— 300 轮 from-scratch 加固队列
#
# 目的：封死"你只训了 200 轮"的质疑 —— 在 from-scratch 条件下把预算加到 300 轮，
#       看 s vs nano 的容量结论是否改变。
#
# 与 run_review_scratch.sh 的差别：
#   1) epochs=300；
#   2) 只跑 seed 42（4 个格子，与同种子 200 轮对照；跑满 3 种子会变成 12 run / 4~5 小时）；
#   3) 写入独立 project 目录 runs_scratch_s300 / runs_scratch_n300 ——
#      defect_train.run_name() 不带轮数后缀，若写回 runs_scratch_s 会覆盖 200 轮的 run。
#
# 用法：bash run_review_scratch_e300.sh 0        （DRY=1 只打印）
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PY=${PY:-${CONDA_ROOT}/bin/python}
GPU=${1:-0}
EP=${EPOCHS:-300}
DRY=${DRY:-0}
SEEDS=(42)

run_one() {   # root data base_yaml proj
  local ROOT="$1" DATA="$2" BASE="$3" PROJ="$4"
  for SEED in "${SEEDS[@]}"; do
    if [ -f "${PROJ}/v_base_seed${SEED}/perclass_ap.json" ]; then
      echo "skip ${PROJ}/v_base_seed${SEED} (done)"; continue
    fi
    echo "==== scratch300 $(basename "$PROJ") seed${SEED}  struct=${BASE} epochs=${EP} ===="
    [ "$DRY" = "1" ] && continue
    "$PY" "$ROOT/defect_train.py" --variant v_base --seed "$SEED" --data "$DATA" \
      --base-yaml "$BASE" --weights "$BASE" --no-pretrained \
      --imgsz 640 --batch 16 --cls-pw 0 --epochs "$EP" --device "$GPU" --workers 8 \
      --project "$PROJ" --final-test
    RC=$?
    echo "---- scratch300 $(basename "$PROJ") seed${SEED} exit=${RC} ----"
    [ "$RC" != "0" ] && { echo "!! 非 0 退出，停止队列"; exit "$RC"; }
  done
}

NEU=${DATA_ROOT}/neu
GC=${DATA_ROOT}/gc10

run_one "$GC"  "$GC/data/GC10-DET-yolo/gc10_det.yaml"  "$GC/yolov8s-gc10.yaml"  "$GC/runs_scratch_s300"
run_one "$GC"  "$GC/data/GC10-DET-yolo/gc10_det.yaml"  "$GC/yolov8n-gc10.yaml"  "$GC/runs_scratch_n300"
run_one "$NEU" "$NEU/data/NEU-DET-split/neu_det.yaml"  "$NEU/yolov8s-neu.yaml"   "$NEU/runs_scratch_s300"
run_one "$NEU" "$NEU/data/NEU-DET-split/neu_det.yaml"  "$NEU/yolov8n-neu.yaml"   "$NEU/runs_scratch_n300"

echo
echo "SCRATCH300 ALL DONE"
