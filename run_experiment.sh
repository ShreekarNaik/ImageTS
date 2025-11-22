#!/usr/bin/env bash
set -euo pipefail

# Master experiment orchestrator for Image-TS
#
# Examples:
#  Single image:
#    ./run_experiment.sh single \
#      --image data/example.png \
#      --exp_name example_default
#
#  Benchmark folder:
#    ./run_experiment.sh benchmark \
#      --image_dir data/benchmark \
#      --exp_name clic_low_bpp
#
#  Decode a bitstream:
#    ./run_experiment.sh decode \
#      --bitstream out/exp1/sample1.bin \
#      --out_image out/exp1/sample1.png

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://github.com/astral-sh/uv" >&2
  exit 1
fi

MODE=${1:-}
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <setup|single|benchmark|decode|all> [args]" >&2
  exit 1
fi
shift

setup_usage() {
  cat >&2 <<USAGE
Usage: $0 setup

Sets up the Python environment and installs the project package.
On Linux, installs pinned CUDA11-compatible PyTorch and NVIDIA components.
Requires Python 3.9.2 (managed via uv and pyproject).
USAGE
}

single_usage() {
  cat >&2 <<USAGE
Usage: $0 single --image PATH --exp_name NAME [--device cpu] [--iterations 1200] [--target 1700] [--max 3000] [--topk 8] [--multiscale-levels 1] [--multiscale-min-size 64] [--multiscale-loss-threshold 0.0]
USAGE
}

benchmark_usage() {
  cat >&2 <<USAGE
Usage: $0 benchmark --image_dir DIR --exp_name NAME [--device cpu] [--iterations 1000] [--target 1500] [--max 3000] [--topk 8] [--multiscale-levels 1] [--multiscale-min-size 64] [--multiscale-loss-threshold 0.0]
USAGE
}

decode_usage() {
  cat >&2 <<USAGE
Usage: $0 decode --bitstream PATH --out_image PATH [--topk 8]
USAGE
}

all_usage() {
  cat >&2 <<USAGE
Usage: $0 all [--image PATH] [--exp_name NAME] [--device cpu] [--iterations 300] [--target 500] [--max 1000] [--topk 8] [--skip-setup] [--skip-tests]

Runs environment setup, installs test extras, executes unit tests,
then trains/evaluates a quick demo to generate metrics and plots.
Defaults: image=data/images/pebbles.jpg, exp_name=smoke_pebbles.
USAGE
}

run_setup() {
  local os
  os=$(uname -s)
  echo "Setting up environment using uv..."

  echo "Ensuring Python 3.9.2 (per pyproject requires-python)..."
  uv python install 3.9.2 >/dev/null 2>&1 || true

  # Install the local package in editable mode so entrypoints are available.
  # On Linux, also install pinned PyTorch and NVIDIA CUDA component wheels.
  if [[ "$os" == "Linux" ]]; then
    echo "Detected Linux; installing pinned PyTorch CUDA11 wheels..."
    uv pip install \
      "torch==1.7.1+cu110" \
      "torchvision==0.8.2+cu110" \
      "torchaudio==0.7.2" \
      -f https://download.pytorch.org/whl/torch_stable.html

    echo "Installing pinned NVIDIA CUDA component wheels..."
    uv pip install \
      "nvidia-cublas-cu11==11.11.3.6" \
      "nvidia-cuda-cupti-cu11==11.8.87" \
      "nvidia-cuda-nvrtc-cu11==11.8.89" \
      "nvidia-cuda-runtime-cu11==11.8.89" \
      "nvidia-cudnn-cu11==9.1.0.70" \
      "nvidia-cufft-cu11==10.9.0.58" \
      "nvidia-curand-cu11==10.3.0.86" \
      "nvidia-cusolver-cu11==11.4.1.48" \
      "nvidia-cusparse-cu11==11.7.5.86" \
      "nvidia-nccl-cu11==2.21.5" \
      "nvidia-nvtx-cu11==11.8.86"
  fi

  echo "Installing project in editable mode..."
  uv pip install -e .

  echo "Verifying installation..."
  uv run python - <<'PY'
import sys
ok = True
try:
    import image_ts  # noqa: F401
except Exception as e:
    ok = False
    print("image_ts import failed:", e)
try:
    import torch
    tv = getattr(torch, "__version__", "unknown")
except Exception as e:
    ok = False
    tv = f"torch import failed: {e}"
ver = sys.version.split()[0]
print("Python:", ver)
print("Torch:", tv)
print("image_ts package:", "ok" if ok else "errors above")
if ver != "3.9.2":
    ok = False
    print("ERROR: Python 3.9.2 is required; got", ver)
if not ok:
    raise SystemExit(1)
PY

  echo "Setup complete. You can now run:"
  echo "  $0 single --image <path> --exp_name <name>"
}

run_all() {
  local image="data/images/pebbles.jpg" exp_name="smoke_pebbles" device="cpu" iterations=300 target=500 max_tris=1000 topk=8
  local skip_setup=0 skip_tests=0
  while [[ $# -gt 0 ]]; do
    case $1 in
      --image) image=$2; shift 2 ;;
      --exp_name) exp_name=$2; shift 2 ;;
      --device) device=$2; shift 2 ;;
      --iterations) iterations=$2; shift 2 ;;
      --target) target=$2; shift 2 ;;
      --max) max_tris=$2; shift 2 ;;
      --topk) topk=$2; shift 2 ;;
      --skip-setup) skip_setup=1; shift ;;
      --skip-tests) skip_tests=1; shift ;;
      *) echo "Unknown arg: $1" >&2; all_usage; exit 1 ;;
    esac
  done

  if [[ $skip_setup -eq 0 ]]; then
    echo "==> Setting up environment"
    run_setup || { echo "Setup failed" >&2; exit 1; }
  fi

  echo "==> Ensuring test dependencies"
  uv pip install -e .[test]

  if [[ $skip_tests -eq 0 ]]; then
    echo "==> Running tests"
    uv run pytest -q
  else
    echo "==> Skipping tests as requested"
  fi

  echo "==> Running demo experiment: $exp_name"
  if [[ ! -f "$image" ]]; then
    echo "Image not found: $image" >&2
    exit 1
  fi
  run_single --image "$image" --exp_name "$exp_name" --device "$device" --iterations "$iterations" --target "$target" --max "$max_tris" --topk "$topk"

  echo "==> All done"
  echo "Data: report/assets/data/$exp_name"
  echo "Plots: report/assets/plots/$exp_name"
}

run_single() {
  local image="" exp_name="" device="cpu" iterations=1000 target=1500 max_tris=3000 topk=8
  local multiscale_levels=1 multiscale_min_size=64 multiscale_loss_threshold=0.0
  while [[ $# -gt 0 ]]; do
    case $1 in
      --image) image=$2; shift 2 ;;
      --exp_name) exp_name=$2; shift 2 ;;
      --device) device=$2; shift 2 ;;
      --iterations) iterations=$2; shift 2 ;;
      --target) target=$2; shift 2 ;;
      --max) max_tris=$2; shift 2 ;;
      --topk) topk=$2; shift 2 ;;
      --multiscale-levels) multiscale_levels=$2; shift 2 ;;
      --multiscale-min-size) multiscale_min_size=$2; shift 2 ;;
      --multiscale-loss-threshold) multiscale_loss_threshold=$2; shift 2 ;;
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
    --topk "$topk" \
    --multiscale-levels "$multiscale_levels" \
    --multiscale-min-size "$multiscale_min_size" \
    --multiscale-loss-threshold "$multiscale_loss_threshold"

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
  local multiscale_levels=1 multiscale_min_size=64 multiscale_loss_threshold=0.0
  while [[ $# -gt 0 ]]; do
    case $1 in
      --image_dir) image_dir=$2; shift 2 ;;
      --exp_name) exp_name=$2; shift 2 ;;
      --device) device=$2; shift 2 ;;
      --iterations) iterations=$2; shift 2 ;;
      --target) target=$2; shift 2 ;;
      --max) max_tris=$2; shift 2 ;;
      --topk) topk=$2; shift 2 ;;
      --multiscale-levels) multiscale_levels=$2; shift 2 ;;
      --multiscale-min-size) multiscale_min_size=$2; shift 2 ;;
      --multiscale-loss-threshold) multiscale_loss_threshold=$2; shift 2 ;;
      *) echo "Unknown arg: $1" >&2; benchmark_usage; exit 1 ;;
    esac
  done
  [[ -z "$image_dir" || -z "$exp_name" ]] && { benchmark_usage; exit 1; }
  find "$image_dir" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | while read -r img; do
    base=$(basename "${img%.*}")
    run_single \
      --image "$img" \
      --exp_name "${exp_name}/${base}" \
      --device "$device" \
      --iterations "$iterations" \
      --target "$target" \
      --max "$max_tris" \
      --topk "$topk" \
      --multiscale-levels "$multiscale_levels" \
      --multiscale-min-size "$multiscale_min_size" \
      --multiscale-loss-threshold "$multiscale_loss_threshold"
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
  setup) run_setup "$@" ;;
  single) run_single "$@" ;;
  benchmark) run_benchmark "$@" ;;
  decode) run_decode "$@" ;;
  all) run_all "$@" ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
