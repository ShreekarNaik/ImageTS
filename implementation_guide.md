# Implementation Guide for Image-TS (Triangle-Based Image Compression)

This guide translates the design in `solution_idea.md` into a concrete, modular implementation plan that a strong coder can follow. It focuses on:

- A clean Python + PyTorch + CUDA implementation (adaptable to other stacks).
- A clear package/module layout under `src/`.
- A master shell script to run experiments, test new images, and export plots + raw data to `report/assets/`.
- Testing and instrumentation to keep the system debuggable and extensible.

If you prefer a different language/framework (e.g., C++/CUDA only), the structure still applies conceptually: just mirror the module boundaries.

---

## 1. High-Level Implementation Phases

Recommended implementation order:

1. **Core geometry & window function (CPU)**  
   - 2D triangle utilities (barycentrics, edge equations, SDF, incenter).  
   - Triangle window function `I_t(p)` with σ.
2. **Reference CPU renderer**  
   - Simple, non-tiled triangle splatting for correctness checks.
3. **GPU tile-based renderer**  
   - Tile partitioning, per-tile triangle lists, top-K per pixel.  
   - PyTorch-friendly forward (and backward if training in PyTorch).
4. **Encoder (per-image optimization)**  
   - Data loading, gradient/importance maps, initialization, loss, optimizer.  
   - Densification (splitting/cloning) and pruning in 2D.
5. **Codec (quantization + bitstream + decoder wrapper)**  
   - Quantization, packing, entropy coding.  
   - Decoder that reconstructs triangles and calls the GPU renderer.
6. **CLI tools + master shell script**  
   - Command-line entrypoints for training, encoding, decoding, evaluating.  
   - `run_experiment.sh` (or similar) to orchestrate workflows and export plots/data.
7. **Tests, metrics, and reporting**  
   - Unit tests for geometry/rendering.  
   - Integration tests for encode→decode.  
   - Scripts to generate rate–distortion plots and save raw metrics in `report/assets/`.

---

## 2. Repository Layout

Target layout (adapt if you already have preferences):

- `src/`
  - `image_ts/` (Python package)
    - `__init__.py`
    - `config.py` – experiment configuration management.
    - `geometry.py` – triangle math utilities.
    - `window.py` – triangle window function (σ-based) and helpers.
    - `renderer/`
      - `cpu_renderer.py` – simple reference triangle splatting.
      - `tile_renderer.py` – GPU-accelerated, tile-based renderer.
    - `encoder/`
      - `init.py` – initialization (gradient + importance-based sampling and triangulation).
      - `densify.py` – splitting/cloning logic.
      - `prune.py` – triangle pruning logic.
      - `train.py` – main training loop (per-image optimization).
    - `codec/`
      - `quantization.py` – quantize/dequantize positions, colors, σ, opacity.
      - `bitstream.py` – pack/unpack bitstreams, LOD structure, headers.
      - `api.py` – high-level `encode_image()` / `decode_image()` functions.
    - `data/`
      - `datasets.py` – image loading utilities (single image + dataset mode).
      - `importance.py` – loading / normalizing external importance maps.
    - `losses.py` – reconstruction (L1, MS-SSIM), regularizers, importance weighting.
    - `metrics.py` – PSNR, SSIM/MS-SSIM, LPIPS hooks (if used).
    - `viz/`
      - `plots.py` – RD curves, histograms, convergence plots.
      - `images.py` – utilities to save reconstructions, error maps.
    - `cli/`
      - `train_cli.py` – entrypoint for training.
      - `encode_cli.py` – entrypoint for encoding an image to a bitstream.
      - `decode_cli.py` – entrypoint for decoding a bitstream to an image.
      - `eval_cli.py` – entrypoint for running metrics/benchmarks.
    - `utils/`
      - `logging.py` – structured logging, progress bars.
      - `seed.py` – reproducibility utilities.
      - `profiling.py` – timing/profiling helpers.

- `scripts/`
  - `run_experiment.sh` – master shell script orchestrating common workflows.
  - (optional) `dev_check.sh` – quick sanity checks, linting, basic tests.

- `report/`
  - `report.typ`
  - `assets/`
    - `plots/` – generated figures (PNG/PDF).
    - `data/` – corresponding raw data (CSV/JSON: RD curves, metrics, etc.).

- `tests/`
  - `test_geometry.py`, `test_window.py`, `test_renderer.py`, `test_codec.py`, etc.

---

## 3. Core Geometry & Window Function

### 3.1 `image_ts/geometry.py`

Implement basic, well-tested geometric utilities:

- `barycentric_coords(p, v1, v2, v3)`  
  - Inputs: `p`, `v1`, `v2`, `v3` in `[0,1]^2`.  
  - Output: `(w1, w2, w3)` with `w1 + w2 + w3 = 1`.  
  - Handle degenerate triangles robustly (e.g., clamp denominators, early exit if area ~ 0).

- `edge_equations(v1, v2, v3)`  
  - Returns `(n1, d1), (n2, d2), (n3, d3)` for edge-based SDF:  
    `L_i(p) = n_i · p + d_i`.

- `signed_distance_field(p, edges)`  
  - `ϕ(p) = max_i L_i(p)` using the edge equations.

- `triangle_area(v1, v2, v3)` – for regularization and sanity checks.

- `triangle_incenter(v1, v2, v3)`  
  - Compute incenter `s` within the triangle; fall back gracefully for degenerate cases.

Implementation guideline:

- Prefer vectorized operations (PyTorch tensors) where possible.  
- Provide both scalar and batched versions if useful:
  - e.g., support `p` as `[N, 2]` and `v` as `[T, 3, 2]`.

### 3.2 `image_ts/window.py`

Implement Triangle Splatting–style window function adapted to 2D:

- Core function: `triangle_window(p, v1, v2, v3, sigma)`  
  - Use geometry utilities to compute `ϕ(p)` and `ϕ(s)`, where `s` is the incenter.  
  - Implement:
    - Inside triangle: `I(p) = ReLU(1 + sigma * ϕ(p) / ϕ(s))`.  
    - Outside: `I(p) = 0`.
  - Clamp σ and `ϕ(s)` to avoid numerical explosions.

- Batch-friendly API:  
  - Inputs: batched pixels, batched triangles, and per-triangle σ.  
  - Goal: easily used inside GPU renderer kernels.

Testing:

- Unit tests verifying:
  - `I(p) ≈ 0` on and outside edges.  
  - `I(s) ≈ 1` at the incenter.  
  - Monotonic falloff from center to edges for varying σ.

---

## 4. Renderer Implementation

### 4.1 `renderer/cpu_renderer.py` (Reference Renderer)

Implement a straightforward, CPU-only renderer for debugging:

- Inputs:
  - Image size `(H, W)`.
  - List of triangles: vertex positions, vertex colors, σ, opacity.  
  - Optional: tile size, but CPU version can ignore tiling initially.
- For each pixel:
  - For each triangle:
    - Check if pixel center lies inside triangle (fast barycentric test).  
    - If inside, compute `I_t(p)`, barycentric color, weight, and accumulate.
- Normalize at the end per pixel.

Use this to:

- Validate geometry/window logic.  
- Compare with GPU results on small images.

### 4.2 `renderer/tile_renderer.py` (GPU Renderer)

Implement a GPU-friendly, tile-based renderer:

- **Data layout:**
  - Tensors for:
    - `verts`: `[T, 3, 2]` in normalized coords.  
    - `colors`: `[T, 3, C]`.  
    - `sigma`: `[T]`.  
    - Optional `opacity`: `[T]`.
  - Tile grid: tile size `T_h × T_w` (e.g., `16 × 16`).

- **Tile index building:**
  - Python-side:
    - Compute per-triangle bounding boxes in pixel space.  
    - Map to tile indices; assign triangle to all overlapped tiles.  
  - Store per-tile triangle lists in a compressed format:
    - Arrays: `tile_tri_indices`, `tile_offsets` (CSR-like structure).

- **Forward rendering kernel:**
  - For each tile:
    - Load the triangle indices for that tile.  
    - For each pixel in the tile:
      - Loop over triangles in tile’s list:
        - Compute coverage; if covered:
          - Evaluate window `I_t(p)` and barycentric color.  
          - Maintain top-K contributions by weight (K small, e.g., 8–16).  
      - Normalize aggregated contributions to obtain final pixel color.

- **Autograd:**
  - Option 1: Write as a PyTorch custom autograd Function with custom backward.  
  - Option 2: Start with PyTorch-only operations (possibly slower) to get gradients for a first working version, then optimize.

Performance guidelines:

- Ensure operations are batched and memory-coalesced as much as possible.  
- Keep K small but sufficient to avoid quality loss.

---

## 5. Encoder: Per-Image Optimization

### 5.1 `data/datasets.py` and `data/importance.py`

- `datasets.py`:
  - `SingleImageDataset` – wraps a single image for per-image training.  
  - `ImageFolderDataset` – optional, for batched experiments.

- `importance.py`:
  - `load_importance_map(path, image_shape)` – loads and resizes an external map if present.  
  - `default_importance_map(image)` – returns ones if no importance map is supplied.  
  - Normalize to `[0,1]` per `solution_idea.md`.

### 5.2 `encoder/init.py`

Responsibilities:

- Compute gradient magnitude map G for an image.  
- Combine gradient and importance map into a sampling distribution:
  - `P(x) ∝ (1 - λ_grad - λ_imp) * U + λ_grad * G̃(x) + λ_imp * M̃(x)`.
- Sample initial points `{p_k}` according to P.  
- Run 2D Delaunay triangulation to obtain initial triangles.
- Initialize:
  - Vertex positions from `{p_k}`.  
  - Vertex colors from ground-truth image at those positions.  
  - σ from local gradients (e.g., smaller near strong edges).

Implementation notes:

- Use an existing Delaunay implementation (e.g., via SciPy or a simple library), or a custom triangulation if external deps are restricted.  
- Keep a clean separation between sampling logic and triangulation so either can be swapped.

### 5.3 `encoder/densify.py`

Implements densification (splitting/cloning):

- Tracks per-triangle statistics:
  - Approximate reconstruction error over pixels where triangle contributes.  
  - Average importance over those pixels.

- Densification step:
  - Compute densification probability `P_densify(t)` based on error and importance.  
  - Sample a subset of triangles to split.  
  - Apply midpoint subdivision:
    - Replace parent with four smaller triangles; copy colors and σ, with optional small perturbation.  
    - For very small but important triangles, clone instead of splitting further.

Provide a function like:

- `densify(triangles, stats, budget, config)` – returns updated triangle set.

### 5.4 `encoder/prune.py`

Implements pruning:

- Maintain per-triangle maximum contribution `max_w_t` across training steps.  
- Periodically remove triangles where `max_w_t < τ_prune` and optionally area is small and/or removal minimally affects validation loss.

Provide:

- `prune(triangles, stats, config)` – returns pruned triangle set.

### 5.5 `encoder/train.py`

Main training loop for per-image optimization:

- Inputs:
  - Image, optional importance map.  
  - Config (triangle budgets, learning rates, number of steps, densification/pruning schedule).

- High-level loop:
  1. Initialize triangles (via `encoder/init.py`).  
  2. For `step` in `0..max_steps`:
     - Render current reconstruction with GPU renderer.  
     - Compute loss:
       - Importance-weighted reconstruction loss (L1 + MS-SSIM) from `losses.py`.  
       - Regularization terms for area, σ, etc.  
     - Backprop + optimizer step on triangle parameters.  
     - Accumulate triangle statistics (error, importance, contribution).  
     - Every `K_densify` steps: call `densify()`.  
     - Every `K_prune` steps: call `prune()`.  
     - Log metrics and occasional visualizations.
  3. After convergence, export final triangle set to codec API.

Keep:

- Config-driven behavior (no magic constants embedded in code).  
- Logging of metrics to JSON or TensorBoard for later plotting.

---

## 6. Codec and Bitstream

### 6.1 `codec/quantization.py`

Implement:

- `quantize_positions(verts, B_pos)` / `dequantize_positions(...)`.  
- `quantize_colors(colors, B_col)` / `dequantize_colors(...)`.  
- `quantize_sigma(sigma, config)` / `dequantize_sigma(...)`.  
- Optional: `quantize_opacity` if opacity is retained.

Guidelines:

- Work in normalized coordinates for positions, convert to integer grid for storage.  
- Prefer log-domain for σ.  
- Keep interfaces pure (functions should not depend on global state).

### 6.2 `codec/bitstream.py`

Bitstream responsibilities:

- Header:
  - Image dimensions, tile size, channels, quantization bits, number of triangles, LOD boundaries.  
- Payload:
  - Sorted triangle parameters (in importance / LOD order).  
  - Optional tile index data if not reconstructed at decode time.

Implement:

- `encode_bitstream(triangles, config) -> bytes`  
  - Apply quantization.  
  - Pack into header + payload (binary format with clear, documented layout).  
  - Optionally apply entropy coding.

- `decode_bitstream(blob) -> (triangles, header)`  
  - Parse header.  
  - Decode triangle parameters, reconstruct quantized values.

Keep the format versioned in case you evolve it later.

### 6.3 `codec/api.py`

High-level user-facing API:

- `encode_image(image, importance_map=None, config) -> bitstream`  
  - Internally calls encoder training and bitstream encoding.

- `decode_to_image(bitstream, lod_level=None, config=None) -> image`  
  - Decode header and triangles.  
  - Optionally select subset based on `lod_level`.  
  - Call GPU renderer to produce the image.

---

## 7. CLI Tools and Master Shell Script

### 7.1 CLI entrypoints in `image_ts/cli/`

Define simple, argument-driven scripts (exposed via `console_scripts` if using setuptools/poetry):

- `image_ts_train`:
  - Args: `--image PATH`, `--importance PATH?`, `--config PATH`, `--out_dir PATH`.  
  - Runs per-image optimization and saves:
    - Final triangle set (e.g., as JSON or binary).  
    - Training logs (metrics over iterations).

- `image_ts_encode`:
  - Args: `--image PATH`, `--importance PATH?`, `--config PATH`, `--bitstream PATH`.  
  - Runs full encode pipeline and outputs bitstream.

- `image_ts_decode`:
  - Args: `--bitstream PATH`, `--lod LEVEL?`, `--out_image PATH`.  
  - Decodes and saves reconstructed image.

- `image_ts_eval`:
  - Args: `--image PATH`, `--bitstream PATH`, `--metrics_out PATH`.  
  - Decodes, computes metrics against ground truth, saves metrics (JSON/CSV).

### 7.2 `scripts/run_experiment.sh` (Master Shell Script)

Design `run_experiment.sh` as the main entry for experiments:

- Usage patterns (examples to document in comments at top of script):

  - **Train + encode + evaluate on a single image:**
    - `./scripts/run_experiment.sh single \`
      `--image data/example.png \`
      `--config configs/default.yaml \`
      `--exp_name example_default`

  - **Run a benchmark on a folder of images:**
    - `./scripts/run_experiment.sh benchmark \`
      `--image_dir data/benchmark/ \`
      `--config configs/low_bpp.yaml \`
      `--exp_name clic_low_bpp`

  - **Test decoding on a new bitstream:**
    - `./scripts/run_experiment.sh decode \`
      `--bitstream out/exp1/sample1.bin \`
      `--lod 2 \`
      `--out_image out/exp1/sample1_lod2.png`

- Script responsibilities:

  - Parse a `mode` (e.g., `single`, `benchmark`, `decode`, `plot`).  
  - Set up experiment directories under `out/EXPERIMENT_NAME`.  
  - Call the Python CLIs (`image_ts_train`, `image_ts_encode`, `image_ts_eval`) with appropriate arguments.  
  - Collect all metrics into CSV/JSON in:
    - `report/assets/data/EXPERIMENT_NAME/`.  
  - Optionally invoke plotting helper (section 8) to generate figures in:
    - `report/assets/plots/EXPERIMENT_NAME/`.

Important: keep the script **thin**; most logic stays in Python. The shell script is an orchestration layer only.

---

## 8. Metrics, Plots, and Reporting

### 8.1 `metrics.py`

Implement standard metrics:

- PSNR (per image and dataset-level averages).  
- SSIM/MS-SSIM.  
- Optional LPIPS (if you include a pre-trained model).

APIs:

- `compute_metrics(gt_image, recon_image) -> dict`  
  - Returns metrics in a consistent dictionary with keys used across the codebase.

### 8.2 `viz/plots.py` and `viz/images.py`

- `viz/images.py`:
  - Functions to save:
    - Reconstructions (`*_recon.png`).  
    - Error maps (`*_error.png`).  
    - Side-by-side comparisons.

- `viz/plots.py`:
  - Functions to generate:
    - Rate–distortion (RD) curves (PSNR vs bpp, SSIM vs bpp).  
    - Training curves (loss vs iteration).  
  - Save:
    - Figures in `report/assets/plots/EXPERIMENT_NAME/`.  
    - Raw data as CSV in `report/assets/data/EXPERIMENT_NAME/`.

Guidelines:

- Always save a machine-readable CSV/JSON along with each figure.  
- Name files systematically, e.g., `rd_curve_clic2020.csv`, `rd_curve_clic2020.png`.

---

## 9. Testing and Quality Checks

### 9.1 Unit Tests (`tests/`)

Suggested tests:

- `test_geometry.py`:
  - Barycentric sums, area computations, incenter inside triangle, etc.

- `test_window.py`:
  - Window value behavior at incenter, on edges, outside triangle.  
  - Monotonic variation with σ.

- `test_renderer.py`:
  - CPU vs GPU renderer consistency on small synthetic scenes.  
  - Rendering of simple patterns (e.g., a single colored triangle).

- `test_codec.py`:
  - Encode→decode triangle parameters is nearly lossless within quantization tolerance.  
  - Encode→decode→render preserves PSNR above a minimal threshold on a simple test image.

### 9.2 Integration / Regression Tests

Add a small set of tiny images (e.g., `tests/data/`) and:

- Run a shortened training pipeline (few steps) and ensure it runs without error.  
- Encode→decode each and log metrics; track them as regression baselines.

Optional: a separate script (or `run_experiment.sh regression`) that runs these checks in one command.

---

## 10. Coding Style Guidelines

To keep the codebase “human-style” and maintainable:

- **Modular design** – each module has a single, clear responsibility. Avoid god-classes or giant scripts.
- **Type hints & docstrings** – annotate function signatures and document inputs/outputs, especially in geometry/rendering/codec layers.
- **Configuration-driven** – avoid hard-coded constants; expose them via config files (`configs/*.yaml`) and `config.py`.
- **Logging over prints** – use a simple logger wrapper so you can switch verbosity and redirect logs easily.
- **Determinism where possible** – set random seeds for sampling; log seeds in experiment metadata.
- **Clear separation of concerns**:
  - Geometry/window functions should not know about bitstreams or file I/O.  
  - Codec should not implement training logic.  
  - CLIs and shell scripts should not contain core algorithm logic.

Following this guide alongside `solution_idea.md` should give you a clear, incremental path to a working, extensible Image-TS implementation with a solid experiment harness and reproducible reporting pipeline.

