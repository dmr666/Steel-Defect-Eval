# NOTE: machine-specific paths were replaced with ${DATA_ROOT}/. placeholders.
#!/bin/bash
# run_review_scratch.sh —— review M6：COCO 预训练混淆的对照（from-scratch 的 s vs nano 基线）
#
# 目的：论文"容量在 nano 尺度饱和"是在 COCO 预训练条件下得到的。本队列补 from-scratch 对照，
#       判定饱和是"任务饱和"还是"预训练抹平了骨干差异"。
# 实现：--weights 传结构 yaml（build_model 里 random_init=True → 随机初始化）+ --no-pretrained。
# 命名：defect_train.run_name 只编 variant/imgsz/batch/cpw/epochs，不含尺度标签 —— 因此
#       s 与 nano 必须用**不同的 project 目录**（runs_scratch_s / runs_scratch_n）避免撞名。
#
# 用法：bash run_review_scratch.sh 0        （DRY=1 只打印）
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PY=${PY:-${CONDA_ROOT}/bin/python}   # 非交互 SSH 里 python 不在 PATH
GPU=${1:-0}
EP=${EPOCHS:-200}
DRY=${DRY:-0}
SEEDS=(42 7 2024)

run_one() {   # root data base_yaml proj
  local ROOT="$1" DATA="$2" BASE="$3" PROJ="$4"
  for SEED in "${SEEDS[@]}"; do
    if [ -f "${PROJ}/v_base_seed${SEED}/perclass_ap.json" ]; then echo "skip ${PROJ}/v_base_seed${SEED} (done)"; continue; fi
    echo "==== scratch $(basename "$PROJ") seed${SEED}  struct=${BASE} epochs=${EP} ===="
    [ "$DRY" = "1" ] && continue
    "$PY" "$ROOT/defect_train.py" --variant v_base --seed "$SEED" --data "$DATA" \
      --base-yaml "$BASE" --weights "$BASE" --no-pretrained \
      --imgsz 640 --batch 16 --cls-pw 0 --epochs "$EP" --device "$GPU" --workers 8 \
      --project "$PROJ" --final-test
    RC=$?
    echo "---- scratch $(basename "$PROJ") seed${SEED} exit=${RC} ----"
    [ "$RC" != "0" ] && { echo "!! 非 0 退出，停止队列"; exit "$RC"; }
  done
}

NEU=${DATA_ROOT}/neu
GC=${DATA_ROOT}/gc10

# GC10-DET：s 与 nano 各自的结构 yaml + 各自 project 目录
run_one "$GC" "$GC/data/GC10-DET-yolo/gc10_det.yaml" "$GC/yolov8s-gc10.yaml" "$GC/runs_scratch_s"
run_one "$GC" "$GC/data/GC10-DET-yolo/gc10_det.yaml" "$GC/yolov8n-gc10.yaml" "$GC/runs_scratch_n"
# NEU-DET
run_one "$NEU" "$NEU/data/NEU-DET-split/neu_det.yaml" "$NEU/yolov8s-neu.yaml" "$NEU/runs_scratch_s"
run_one "$NEU" "$NEU/data/NEU-DET-split/neu_det.yaml" "$NEU/yolov8n-neu.yaml" "$NEU/runs_scratch_n"

echo
echo "SCRATCH ALL DONE"
