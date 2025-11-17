#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <image_or_folder> <output_root> [--device DEVICE]" >&2
  exit 1
fi

INPUT=$1
OUTPUT_ROOT=$2
DEVICE=${3:---device cpu}

process_image() {
  local image_path=$1
  local out_dir=$2
  mkdir -p "$out_dir"
  image_ts_train --image "$image_path" --output "$out_dir" --device ${DEVICE#--device }
  image_ts_eval --reference "$image_path" --bitstream "$out_dir/bitstream.bin" \
    --metrics-out "$out_dir/metrics_eval.json" --rd-csv "report/assets/data/rd_curve.csv" \
    --rd-plot "report/assets/plots/rd_curve.png"
}

if [[ -d "$INPUT" ]]; then
  while IFS= read -r img; do
    rel=$(basename "${img%.*}")
    process_image "$img" "$OUTPUT_ROOT/$rel"
  done < <(find "$INPUT" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \ ))
else
  process_image "$INPUT" "$OUTPUT_ROOT"
fi
