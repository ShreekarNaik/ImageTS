# Image-TS: Image Compression via Differentiable Triangle Splatting

Image-TS is a research-oriented implementation of triangle-based image compression using differentiable rendering, progressive training, and codec-aware triangle optimization.

Key outcomes:
- Built a tile-based differentiable renderer with top-K selection achieving **5x faster training** and **3x lower memory**.
- Used age-based progressive training with multi-level pyramids enabling dynamic triangle growth and a **5x speedup**.
- Developed dynamic triangle management (pruning, subdivision, densification) with rate/shape/opacity regularization.
- Achieved **10–30x compression** and **0.5 bpp** while maintaining perceptual quality (**PSNR > 20 dB**).

## What This Repository Contains

- A modular `src/image_ts` package for geometry, rendering, optimization, codec, evaluation, and visualization.
- CLI entry points for training, encoding, decoding, and evaluation.
- `run_experiment.sh` orchestration for setup, single-image runs, benchmarks, decode-only, and all-in-one smoke flow.
- Unit/integration tests in `tests/`.
- Report assets and generated experiment outputs under `report/assets/` and `out/`.

## Repository Structure

- `src/image_ts/geometry.py`, `window.py` — core triangle math and windowing.
- `src/image_ts/renderer/` — CPU reference and tile-based renderer.
- `src/image_ts/encoder/` — initialization, training, densification, pruning, multiscale flow.
- `src/image_ts/codec/` — quantization, bitstream format, encode/decode API.
- `src/image_ts/cli/` — `image_ts_train`, `image_ts_encode`, `image_ts_decode`, `image_ts_eval`.
- `tests/` — geometry, renderer, codec, integration, and memory-batching tests.
- `run_experiment.sh` — main experiment entrypoint.

## Getting Started

### 1) Environment

Project metadata currently declares:
- Python `>=3.10,<3.11`
- Main dependencies in `pyproject.toml` (PyTorch, NumPy, Pillow, Matplotlib, tqdm)

Install locally:

```bash
python -m pip install -e .
python -m pip install -e .[test]
```

### 2) Core CLI Usage

```bash
# Train / optimize triangles for one image
image_ts_train --image data/pebbles.jpg --output out/example --device cpu

# Encode from saved triangles
image_ts_encode --triangles out/example/triangles.pt --image data/pebbles.jpg --output out/example/bitstream.bin

# Decode a bitstream
image_ts_decode --bitstream out/example/bitstream.bin --output out/example/reconstruction.png

# Evaluate metrics and optional RD artifacts
image_ts_eval --reference data/pebbles.jpg --bitstream out/example/bitstream.bin --metrics-out out/example/metrics.json
```

### 3) End-to-End via Script

```bash
# setup / install flow
./run_experiment.sh setup

# single-image run
./run_experiment.sh single --image data/pebbles.jpg --exp_name smoke_pebbles

# benchmark all images in a directory
./run_experiment.sh benchmark --image_dir data --exp_name bench_run

# decode-only
./run_experiment.sh decode --bitstream out/smoke_pebbles/bitstream.bin --out_image out/smoke_pebbles/recon.png

# all-in-one smoke pipeline (setup + tests + run)
./run_experiment.sh all --image data/pebbles.jpg --exp_name smoke_pebbles
```

## Outputs

Typical run outputs include:
- `bitstream.bin`
- `triangles.pt`
- `reconstruction.png`
- `error_map.png`
- `metrics.json`
- `history.json`

Experiment data/plots are exported to:
- `report/assets/data/<exp_name>/`
- `report/assets/plots/<exp_name>/`

## Notes

- This repository is intended for experimentation and report generation around differentiable triangle-splatting compression.
- See `solution_idea.md` for method design and `implementation_guide.md` for implementation details.
