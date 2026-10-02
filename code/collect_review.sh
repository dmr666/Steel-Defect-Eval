# NOTE: machine-specific paths were replaced with ${DATA_ROOT}/. placeholders.
#!/bin/bash
# collect_review.sh —— B 档收数：把新增 run 的小文件 + IoU 曲线打包（不含权重）
# 用法（在 ${DATA_ROOT} 下）: bash collect_review.sh
set -u
OUT=${DATA_ROOT}/reviewB_small.tar.gz
NEU=${DATA_ROOT}/neu
GC=${DATA_ROOT}/gc10

cd ${DATA_ROOT} || exit 1
LIST=$(mktemp)
{
  find "$GC/runs_scratch_s" "$GC/runs_scratch_n" "$NEU/runs_scratch_s" "$NEU/runs_scratch_n" -maxdepth 3 -type f \( -name 'results.csv' -o -name 'args.yaml' -o -name 'perclass_ap.json' \) 2>/dev/null
  find "$NEU/runs_e300n" -maxdepth 3 -type f \( -name 'results.csv' -o -name 'args.yaml' -o -name 'perclass_ap.json' \) 2>/dev/null
  find "$GC/runs" "$NEU/runs" -maxdepth 2 -type d \( -name '*_seed13' -o -name '*_seed99' \) 2>/dev/null | while read -r d; do
      find "$d" -maxdepth 2 -type f \( -name 'results.csv' -o -name 'args.yaml' -o -name 'perclass_ap.json' \)
  done
  find "$NEU/iou_sweep" "$GC/iou_sweep" -maxdepth 2 -name 'iou_sweep.json' 2>/dev/null
  find "$NEU" "$GC" -maxdepth 1 -name 'review_*.log' 2>/dev/null
} > "$LIST"

tar czf "$OUT" -T "$LIST" 2>/dev/null
echo "条目数: $(wc -l < "$LIST")"
ls -la "$OUT"
echo
echo "拉回本地（在你本机执行）："
echo "  scp -P <ssh-port> root@<your-server-host>:$OUT  ./_results/"
