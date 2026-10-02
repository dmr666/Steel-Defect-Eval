# NOTE: machine-specific paths were replaced with ${DATA_ROOT}/. placeholders.
#!/bin/bash
# run_review_extra_seeds.sh —— review M1：关键结论加种子（baseline / full stack / Focal 各 +2 种子）
#
# 只对"结论所依赖的格子"加种子（不是全矩阵），共 12 个 run：
#   GC10-DET s 与 NEU-DET s 各：v_base / v_full / v_focal × 2 新种子（13、99）
# 结果写回各自的 runs/；跳过判定同时接受带/不带 _e300 的目录名（防重跑）。
#
# 用法：bash run_review_extra_seeds.sh 0
set -u
PY=${PY:-${CONDA_ROOT}/bin/python}   # 非交互 SSH 里 python 不在 PATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

GPU=${1:-0}
EP=${EPOCHS:-200}
DRY=${DRY:-0}
SEEDS=(13 99)

synth_train() {   # root data p2 base proj
  local ROOT="$1" DATA="$2" P2="$3" BASE="$4" PROJ="$5"
  for CELL in "v_base|0.0" "v_full|1.0" "v_focal|0.0"; do
    IFS='|' read -r VAR CPW <<< "$CELL"
    for SEED in "${SEEDS[@]}"; do
      NAME="${VAR}_seed${SEED}"
      if [ -f "${PROJ}/${NAME}/perclass_ap.json" ]; then echo "skip ${NAME} (done)"; continue; fi
      [ "$DRY" = "1" ] && { echo "would run ${NAME} variant=${VAR} cls_pw=${CPW}"; continue; }
      echo "==== ${NAME} variant=${VAR} cls_pw=${CPW} epochs=${EP} ===="
      "$PY" "$ROOT/defect_train.py" --variant "$VAR" --seed "$SEED" --data "$DATA" \
        --p2-yaml "$P2" --base-yaml "$BASE" --weights "$ROOT/yolov8s.pt" \
        --imgsz 640 --batch 16 --cls-pw "$CPW" \
        --epochs "$EP" --device "$GPU" --workers 8 --project "$PROJ" --final-test
      RC=$?
      echo "---- ${NAME} exit=${RC} ----"
      [ "$RC" != "0" ] && { echo "!! 非 0 退出，停止队列"; exit "$RC"; }
    done
  done
}

GC_ROOT=${DATA_ROOT}/gc10
NEU_ROOT=${DATA_ROOT}/neu
synth_train "$GC_ROOT" "$GC_ROOT/data/GC10-DET-yolo/gc10_det.yaml" \
            "$GC_ROOT/data/GC10-DET-yolo/yolov8s-p2-gc10.yaml" "$GC_ROOT/yolov8s-gc10.yaml" "$GC_ROOT/runs"
synth_train "$NEU_ROOT" "$NEU_ROOT/data/NEU-DET-split/neu_det.yaml" \
            "$NEU_ROOT/yolov8s-p2.yaml" "$NEU_ROOT/yolov8s-neu.yaml" "$NEU_ROOT/runs"

echo
echo "EXTRA SEEDS ALL DONE"
