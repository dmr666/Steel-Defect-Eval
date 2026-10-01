#!/bin/bash
# NOTE: paths below were machine-specific and have been replaced with ${DATA_ROOT}.
# Set DATA_ROOT (and DATA/PROJ inside) to match your own layout before running.
#!/bin/bash
# run_gc10_e300.sh - 300 轮同算力加固对照（paper3 / GC10-DET）
#
# 目的：回应"200 轮是否训练不足、模块来不及发挥作用"这一质疑。
#   两格：v_base@300 与 v_full@300（各 3 种子 = 6 run ≈ 3 GPU 时）
#   对照读法：若 300 轮下基线仍然 ≥ 四件套 → "轮数不是短板" 的结论被坐实；
#             若四件套在 300 轮下反超 → 增益与训练预算相关（同样是可发表结论）。
#
# 与主矩阵的关系：目录名带 _e300（defect_train.run_name 的轮数标签），
#   与 200 轮的 runs/v_base_seed42 等**不撞名**，可共存于同一个 runs/；
#   analyze_matrix.py 会把它们作为"可选加固 cell"单列，不计入主表缺失。
#
# 用法：
#   bash run_gc10_e300.sh            # 跑（已完成的自动跳过）
#   bash run_gc10_e300.sh 0          # 指定 GPU
#   DRY=1 bash run_gc10_e300.sh      # 只打印将跑什么
#
# 注意：跑之前 GPU 必须空闲（不要与 NEU 矩阵或部署测量同时进行）。
set -u
cd "$(dirname "$0")"
ROOT=${GC10_ROOT:-${DATA_ROOT}/gc10}   # 本地做命名 dry-run 时可用 GC10_ROOT 覆盖
DATA=${ROOT}/data/GC10-DET-yolo/gc10_det.yaml
P2Y=${ROOT}/data/GC10-DET-yolo/yolov8s-p2-gc10.yaml
P1Y=${ROOT}/data/GC10-DET-yolo/yolov8s-p1-gc10.yaml
PROJ=${ROOT}/runs
GPU=${1:-0}
EP=${EPOCHS:-300}
DRY=${DRY:-0}

if [ "${DRY}" != "1" ] && [ ! -f "${DATA}" ]; then echo "找不到数据 yaml: ${DATA}"; exit 1; fi

# cell: variant | imgsz | batch | cls_pw
CELLS=(
  "v_base|640|16|0"
  "v_full|640|16|1.0"
)
SEEDS="42 7 2024"

echo "GC10 300 轮加固：GPU=${GPU} epochs=${EP} cells=${#CELLS[@]} seeds=${SEEDS} dry=${DRY}"
echo "数据: ${DATA}"

for CELL in "${CELLS[@]}"; do
  IFS='|' read -r VAR IMG BS CPW <<< "${CELL}"
  for SEED in ${SEEDS}; do
    TAG="${VAR}"
    if [ "${IMG}" != "640" ]; then TAG="${TAG}_s${IMG}"; fi
    if [ "${BS}" != "16" ]; then TAG="${TAG}_b${BS}"; fi
    # 类加权只对 v_full / v_p2dcsawiou 生效；且只在"与默认不同"时进名字
    if [ "${VAR}" = "v_full" ] || [ "${VAR}" = "v_p2dcsawiou" ]; then
      DEF=$( [ "${VAR}" = "v_full" ] && echo "1.0" || echo "0.0" )
      if [ "$(printf '%.1f' "${CPW}")" != "${DEF}" ]; then
        TAG="${TAG}_cpw$(echo "$(printf '%.1f' "${CPW}")" | tr '.' 'p')"
      fi
    fi
    # 轮数标签：非 200 才加（必须与 defect_train.run_name 的规则完全一致）
    if [ "${EP}" != "200" ]; then TAG="${TAG}_e${EP}"; fi
    NAME="${TAG}_seed${SEED}"

    if [ -f "${PROJ}/${NAME}/perclass_ap.json" ]; then
      echo "skip ${NAME} (done)"; continue
    fi
    echo "==== ${NAME}  variant=${VAR} imgsz=${IMG} batch=${BS} cls_pw=${CPW} epochs=${EP} ===="
    if [ "${DRY}" = "1" ]; then continue; fi

    python defect_train.py --variant "${VAR}" --seed "${SEED}" --data "${DATA}" \
      --p2-yaml "${P2Y}" --p1-yaml "${P1Y}" \
      --imgsz "${IMG}" --batch "${BS}" --cls-pw "${CPW}" \
      --epochs "${EP}" --device "${GPU}" --workers 8 --project "${PROJ}" \
      --final-test
    RC=$?
    echo "---- ${NAME} exit=${RC} ----"
    if [ "${RC}" != "0" ]; then echo "!! ${NAME} 非 0 退出，停止队列"; exit "${RC}"; fi
  done
done

echo
echo "GC10 300 轮加固 ALL DONE"
echo "收数: python analyze_matrix.py ${PROJ} --out tables   （可选 cell 会自动单列）"
