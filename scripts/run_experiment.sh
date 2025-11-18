#!/usr/bin/env bash
set -euo pipefail

# Master experiment orchestrator for Image-TS
#
# Examples:
#  Single image:
#    ./scripts/run_experiment.sh single \
#      --image data/example.png \
#      --exp_name example_default
#
#  Benchmark folder:
#    ./scripts/run_experiment.sh benchmark \
#      --image_dir data/benchmark \
#      --exp_name clic_low_bpp
#
#  Decode a bitstream:
#    ./scripts/run_experiment.sh decode \
#      --bitstream out/exp1/sample1.bin \
#      --out_image out/exp1/sample1.png

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://github.com/astral-sh/uv" >&2
  exit 1
fi

MODE=${1:-}
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <single|benchmark|decode> [args]" >&2
  exit 1
fi
shift

single_usage() {
  cat >&2 <<USAGE
Usage: $0 single --image PATH --exp_name NAME [--device cpu] [--iterations 1000] [--target 1500] [--max 3000] [--topk 8]
USAGE
}

benchmark_usage() {
  cat >&2 <<USAGE
Usage: $0 benchmark --image_dir DIR --exp_name NAME [--device cpu] [--iterations 1000] [--target 1500] [--max 3000] [--topk 8]
USAGE
}

decode_usage() {
  cat >&2 <<USAGE
Usage: $0 decode --bitstream PATH --out_image PATH [--topk 8]
USAGE
}

run_single() {
  local image="" exp_name="" device="cpu" iterations=1000 target=1500 max_tris=3000 topk=8
  while [[ $# -gt 0 ]]; do
    case $1 in
      --image) image=$2; shift 2 ;;
      --exp_name) exp_name=$2; shift 2 ;;
      --device) device=$2; shift 2 ;;
      --iterations) iterations=$2; shift 2 ;;
      --target) target=$2; shift 2 ;;
      --max) max_tris=$2; shift 2 ;;
      --topk) topk=$2; shift 2 ;;
      *) echo "Unknown arg: $1" >&2; single_usage; exit 1 ;;
    esac
  done
  [[ -z "$image" || -z "$exp_name" ]] && { single_usage; exit 1; }

  local out_dir="out/${exp_name}"
  mkdir -p "$out_dir"
  uv run image_ts_train \
    --image "$image" \
    --output "$out_dir" \
    --device "$device" \
    --iterations "$iterations" \
    --target-triangles "$target" \
    --max-triangles "$max_tris" \
    --topk "$topk"

  local data_dir="report/assets/data/${exp_name}"
  local plots_dir="report/assets/plots/${exp_name}"
  mkdir -p "$data_dir" "$plots_dir"
  uv run image_ts_eval \
    --reference "$image" \
    --bitstream "$out_dir/bitstream.bin" \
    --metrics-out "$data_dir/metrics.json" \
    --rd-csv "$data_dir/rd_curve.csv" \
    --rd-plot "$plots_dir/rd_curve.png"
}

run_benchmark() {
  local image_dir="" exp_name="" device="cpu" iterations=1000 target=1500 max_tris=3000 topk=8
  while [[ $# -gt 0 ]]; do
    case $1 in
      --image_dir) image_dir=$2; shift 2 ;;
      --exp_name) exp_name=$2; shift 2 ;;
      --device) device=$2; shift 2 ;;
      --iterations) iterations=$2; shift 2 ;;
      --target) target=$2; shift 2 ;;
      --max) max_tris=$2; shift 2 ;;
      --topk) topk=$2; shift 2 ;;
      *) echo "Unknown arg: $1" >&2; benchmark_usage; exit 1 ;;
    esac
  done
  [[ -z "$image_dir" || -z "$exp_name" ]] && { benchmark_usage; exit 1; }
  find "$image_dir" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | while read -r img; do
    base=$(basename "${img%.*}")
    run_single --image "$img" --exp_name "${exp_name}/${base}" --device "$device" --iterations "$iterations" --target "$target" --max "$max_tris" --topk "$topk"
  done
}

run_decode() {
  local bitstream="" out_image="" topk=8
  while [[ $# -gt 0 ]]; do
    case $1 in
      --bitstream) bitstream=$2; shift 2 ;;
      --out_image) out_image=$2; shift 2 ;;
      --topk) topk=$2; shift 2 ;;
      *) echo "Unknown arg: $1" >&2; decode_usage; exit 1 ;;
    esac
  done
  [[ -z "$bitstream" || -z "$out_image" ]] && { decode_usage; exit 1; }
  uv run image_ts_decode --bitstream "$bitstream" --output "$out_image" --topk "$topk"
}

case "$MODE" in
  single) run_single "$@" ;;
  benchmark) run_benchmark "$@" ;;
  decode) run_decode "$@" ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
