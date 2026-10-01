#!/bin/bash
# NOTE: paths below were machine-specific and have been replaced with ${DATA_ROOT}.
# Set DATA_ROOT (and DATA/PROJ inside) to match your own layout before running.
#!/bin/bash
# run_nano.sh —— paper3 YOLOv8n 尺度对照（回答"容量是不是瓶颈 / 轻量化该怎么做"）
#
# 目的：s 尺度上"增强组合无增益"，需要一条 nano 尺度的对照来判定：
#   · 若 nano ≈ s（噪声内）→ 该任务在 nano 即饱和，轻量化不该靠加模块、而该换更小骨干（正向结论）
#   · 若 nano 明显更差   → 容量确实是瓶颈，与"增强无益"形成张力（同样是真实发现）
#   两格核心：v_base（裸 nano） vs v_full（nano+P2+DCSA+WIoU+cls_pw=1.0）
#   另加 v_p2 / v_dcsa / v_wiou 做拆解（nano 单 run 只要几分钟，拆解很便宜）
#
# 用法：
#   bash run_nano.sh                 # 默认 GC10，GPU 0
#   DS=neu bash run_nano.sh 0        # NEU（换数据/ yaml 路径）
#   DRY=1 bash run_nano.sh           # 只打印将跑什么（不训练）
#   EPOCHS=200 bash run_nano.sh      # 轮数（默认 200，与主矩阵同协议）
#
# 前置：
#   1) 先跑 python preflight_nano.py --data ... --base-yaml ... --p2-yaml ...（不通过别开跑）
#   2) GPU 必须空闲（不要与 e300 队列/部署测量同时跑）
#   3) 结果写进 $ROOT/runs_nano（与 s 尺度的 runs/ 完全隔离，命名相同也不冲突）
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DS=${DS:-gc10}
case "$DS" in
  gc10)
    ROOT=${ROOT:-${DATA_ROOT}/gc10}
    DATA=$ROOT/data/GC10-DET-yolo/gc10_det.yaml
    BASE=$ROOT/yolov8n-gc10.yaml
    P2=$ROOT/yolov8n-p2-gc10.yaml
    ;;
  neu)
    ROOT=${ROOT:-${DATA_ROOT}/neu}
    DATA=$ROOT/data/NEU-DET-split/neu_det.yaml
    BASE=$ROOT/yolov8n-neu.yaml
    P2=$ROOT/yolov8n-p2-neu.yaml
    ;;
  *) echo "DS 只能是 gc10 或 neu（当前：$DS）"; exit 1 ;;
esac

W=${WEIGHTS:-$ROOT/yolov8n.pt}
PROJ=${PROJ:-$ROOT/runs_nano}
EP=${EPOCHS:-200}
IMG=${IMGSZ:-640}
BS=${BATCH:-16}
GPU=${1:-0}
DRY=${DRY:-0}

echo "nano 对照：DS=$DS GPU=$GPU epochs=$EP imgsz=$IMG batch=$BS dry=$DRY"
echo "  ROOT = $ROOT"
echo "  DATA = $DATA"
echo "  BASE = $BASE"
echo "  P2   = $P2"
echo "  W    = $W"
echo "  PROJ = $PROJ"

for f in "$DATA" "$BASE" "$P2"; do
  [ -f "$f" ] || { echo "缺文件: $f"; exit 1; }
done
if [ ! -f "$W" ]; then
  echo "缺 nano 预训练权重: $W"
  echo "  取法一（推荐，随本包附带）：把包里的 yolov8n.pt 一起解压到 $ROOT"
  echo "  取法二（服务器能通 GitHub 时）：wget -c https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt -O $W"
  echo "  取法三（ghproxy 镜像）：wget -c https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt -O $W"
  echo "  取法四：本机下载后经 JupyterLab 上传（务必核对大小约 6 MB，别是半截文件）"
  exit 1
fi

# cell：variant | cls_pw
CELLS=(
  "v_base|0"
  "v_p2|0"
  "v_dcsa|0"
  "v_wiou|0"
  "v_full|1.0"
)
SEEDS=(42 7 2024)

for CELL in "${CELLS[@]}"; do
  IFS='|' read -r VAR CPW <<< "$CELL"
  for SEED in "${SEEDS[@]}"; do
    NAME="${VAR}_seed${SEED}"
    if [ -f "${PROJ}/${NAME}/perclass_ap.json" ]; then
      echo "skip ${NAME} (done)"
      continue
    fi
    echo "==== ${NAME}  variant=${VAR} cls_pw=${CPW} epochs=${EP} imgsz=${IMG} batch=${BS} ===="
    if [ "$DRY" = "1" ]; then continue; fi

    python defect_train.py --variant "$VAR" --seed "$SEED" --data "$DATA" \
      --p2-yaml "$P2" --base-yaml "$BASE" --weights "$W" \
      --imgsz "$IMG" --batch "$BS" --cls-pw "$CPW" \
      --epochs "$EP" --device "$GPU" --workers 8 --project "$PROJ" \
      --final-test
    RC=$?
    echo "---- ${NAME} exit=${RC} ----"
    if [ "$RC" != "0" ]; then echo "!! ${NAME} 非 0 退出，停止队列"; exit "$RC"; fi
  done
done

echo
echo "nano 对照 ALL DONE（DS=$DS）"
echo "收数：cd $(dirname "$ROOT") 不需要；直接在 $ROOT 下跑："
echo "  RUNS=runs_nano bash 10_eval_test_collect.sh"
echo "注意：analyze_matrix 的完整性报告会把它期望的 13 个 cell 中缺失的那些报成\"缺\"，"
echo "      那是主矩阵的期望，nano 目录只有 5 个 cell，忽略即可；表 A/B/C 里的 nano 行是有效的。"
