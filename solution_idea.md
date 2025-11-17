# Image-TS: Content-Adaptive Image Compression with Triangle Splatting

This document describes a triangle-only, content-adaptive image representation and compression scheme that conceptually fuses:

- Image-GS: per-image optimization, content-aware initialization, progressive error-guided refinement, tile-based top-K rendering, and level-of-detail (LOD) behavior.
- Triangle Splatting: triangle primitives with a smooth window function, a learnable sharpness parameter σ, and adaptive pruning/splitting.

The target is 2D image compression (no multi-view or 3D reconstruction), with:

- High rate–distortion performance for large images.
- Very fast GPU decoding using only triangle rasterization + a simple triangle-window evaluation (no neural networks at decode time).
- Progressive / LOD-style refinement.
- Optional user-provided importance map to steer triangle allocation and reconstruction quality.
- Representation strictly in terms of triangles (no Gaussians as primitives in the final codec).

We refer to this approach as **Image-TS** (Image Triangle Splatting) in what follows.

---

## 1. High-Level Overview

**Representation.**  
An image is represented as a set of 2D triangles in normalized image coordinates. Each triangle carries:

- Three 2D vertex positions in the image plane.
- Three vertex colors (or a compact per-triangle color basis).
- A per-triangle smoothness (sharpness) parameter σ controlling a triangle window function.
- Optionally a per-triangle opacity/weight scalar o.

Rendering is done by **triangle splatting**:

- For any pixel x, we consider triangles whose support covers x.
- Each triangle contributes a color via barycentric interpolation and a scalar weight via a compactly-supported window function that depends on its signed distance field and σ.
- Final pixel color is a normalized weighted combination of the contributing triangles, optionally restricted to the top-K contributors per pixel for efficiency and locality (Image-GS-style top-K).

**Compression.**  
To encode an image:

1. Initialize a coarse triangle set guided by image gradients and an optional importance map.
2. Optimize triangle parameters via differentiable triangle splatting to minimize a weighted reconstruction loss.
3. Progressively **densify** (split or clone) triangles in high-error / high-importance regions and **prune** triangles with negligible contribution.
4. After convergence, **quantize** and entropy-code triangle parameters in a carefully structured bitstream that:
   - Allows progressive decoding (LOD).
   - Supports fast GPU decoding with simple data structures.
   - Optionally supports importance-aware bit allocation.

**Decoding.**  
The decoder:

- Parses the bitstream to reconstruct a list of triangles and their attributes.
- Optionally truncates the triangle list at some LOD level to meet bitrate or performance constraints.
- Performs GPU rasterization using a custom but simple shader that:
  - Evaluates the triangle window function and barycentric colors for relevant triangles per pixel.
  - Combines them via normalized weighted blending, optionally top-K inside per-tile lists.

---

## 2. Core Data Structures and Parameterization

### 2.1 Coordinate System

- Normalize the image domain to `[0, 1] × [0, 1]`:
  - A pixel center at integer coordinates `(i, j)` maps to `(x, y) = ((i + 0.5) / W, (j + 0.5) / H)`.
- All vertex positions are stored and optimized in this normalized domain, then quantized.

### 2.2 Triangle Primitive

Each triangle `t` is defined by:

- Vertex positions:  
  - `v1 = (x1, y1)`, `v2 = (x2, y2)`, `v3 = (x3, y3)` in `[0,1]^2`.
- Vertex colors:  
  - For RGB images: `c1, c2, c3 ∈ R^3`.  
  - Optionally store colors in a decorrelated color space (e.g., YCoCg or YUV) for better entropy coding.
- Smoothness (sharpness) parameter:  
  - `σ > 0` controlling the window function smoothness (sharp vs smooth).
- Optional opacity:  
  - `o ∈ (0, 1]`, primarily useful during optimization and in pruning; can be dropped or coarsely quantized in the final codec if not needed.

In practice, we can store parameters in a more compression-friendly form:

- Vertex positions as quantized integers on a grid (e.g., 16-bit per coordinate).
- Colors as 8–10-bit per channel after color-space transform.
- σ in log-space or via a small discrete codebook.
- o as a low-bit scalar or omitted in the final representation if replaced by geometric and σ-based pruning.

### 2.3 Triangle Window Function (2D Triangle Splat)

We adapt the **Triangle Splatting** window function to 2D image triangles:

1. For each triangle, precompute edge equations in image space:
   - For edges `i ∈ {1,2,3}`, define signed distance-like linear functions:  
     `L_i(p) = n_i · p + d_i`, where `n_i` is the outward unit normal.
2. Define the signed distance field (SDF) approximation:  
   - `ϕ(p) = max_i L_i(p)`  
     - Negative inside the triangle, zero at edges, positive outside.
3. Compute the incenter `s` of the triangle (point of minimal ϕ inside).
4. Define window function `I_t(p)` as in Triangle Splatting (adapted for 2D-only):
   - Inside the triangle:  
     `I_t(p) = ReLU(1 + σ * ϕ(p) / ϕ(s))`  
   - On boundary and outside:  
     `I_t(p) = 0`.

Properties:

- Compact support: `I_t(p)` is zero outside the triangle; support tightly matches the triangle footprint (good for rasterization and locality).
- Smoothness: controlled by σ; small σ approximates a hard triangle, larger σ yields smoother falloff from center to boundary.
- Differentiability: piecewise differentiable w.r.t. vertex positions and σ, enabling backpropagation-based optimization.

### 2.4 Per-Pixel Color from a Single Triangle

For a pixel center at `p` inside triangle `t`:

- Compute barycentric coordinates `(w1, w2, w3)` of `p` with respect to `v1, v2, v3`.
- Interpolate color:
  - `c_t(p) = w1 * c1 + w2 * c2 + w3 * c3`.
- Compute window weight:
  - `w_t(p) = o * I_t(p)` (opacity-scaled window value).

This yields a triangle-specific contribution `(w_t(p), c_t(p))`.

---

## 3. Rendering and Decoding Pipeline

### 3.1 Tile-Based Acceleration and Top-K Selection

For fast GPU rendering and random access, we adopt an Image-GS-style **tile-based** design:

1. **Tiling.**
   - Partition the image into non-overlapping tiles, e.g., `T = 16 × 16` pixels.
   - Assign each triangle to all tiles it overlaps by checking bounding box–tile intersections (plus a small margin for smoothing near edges).
2. **Per-tile triangle lists.**
   - For each tile `τ`, store a list `S_τ` of triangle indices that intersect τ.
   - These lists are stored in the bitstream or derived after decoding vertex positions.
3. **Top-K selection per pixel (optional but recommended).**
   - For a pixel `p` in tile `τ`, evaluate contributions only from `S_τ`.
   - Compute `w_t(p)` for each triangle `t ∈ S_τ` that covers `p`.
   - Optionally keep only the **top K** triangles by weight (e.g., `K=8–16`) to:
     - Bound per-pixel work.
     - Preserve locality and hardware-friendly access.

This mirrors Image-GS’s top-K tile-based accumulation but uses triangles instead of Gaussians.

### 3.2 Pixel Color Aggregation

Let `S_τ^K(p)` denote the top-K contributing triangles for pixel `p` in tile `τ`. Then the decoded pixel color is:

- `C(p) = (∑_{t ∈ S_τ^K(p)} w_t(p) * c_t(p)) / (∑_{t ∈ S_τ^K(p)} w_t(p) + ε)`

where `ε` is a small constant to avoid division by zero; if the denominator is near zero, we can fall back to a default background color.

This aggregation:

- Keeps brightness stable regardless of the number of overlapping triangles.
- Lets triangles naturally cooperate to approximate local color distributions.
- Can be implemented as a simple compute shader / fragment shader on GPU.

### 3.3 Progressive / LOD Decoding

To support LOD and progressive transmission:

1. **Importance ordering of triangles.**
   - After training, estimate the importance of each triangle, e.g., by:
     - Measuring change in reconstruction error when the triangle is removed or masked.
     - Or using internal training statistics (e.g., accumulated contribution weights).
   - Sort triangles from most to least important.
2. **LOD levels.**
   - Partition the sorted triangle list into LOD stages, e.g.:
     - LOD0: first `N0` triangles (very coarse reconstruction).
     - LOD1: next `N1` triangles (adds more structure).
     - …
   - Alternatively, define LOD boundaries by cumulative rate or error thresholds.
3. **Bitstream structure.**
   - Encode LOD boundaries in the header.
   - Store triangles in order; truncating the bitstream at any LOD boundary yields a valid, lower-quality reconstruction.

At decode time, the client chooses how many LOD stages to read, trading bitrate vs quality and decode time.

---

## 4. Encoder: Per-Image Optimization Pipeline

The encoder performs per-image optimization (like Image-GS) to construct and refine the triangle set. It can be implemented in PyTorch/JAX with a differentiable triangle splatting module.

### 4.1 Inputs

- Target image `I_gt` of size `H × W × C`.
- Optional user-supplied **importance map** `M` of size `H × W` (values in `[0,1]`), controlling prioritization.
  - `M` can be generated by any external network or manually.
  - If absent, treat `M` as all ones (or default to a purely gradient-based scheme).

### 4.2 Preprocessing

1. Normalize image coordinates to `[0,1]^2`.
2. Compute a **gradient magnitude map** `G(x)` of the image:
   - E.g., via Sobel filters on a luminance channel.
3. Normalize both `G` and `M`:
   - `G̃(x) = G(x) / (mean(G) + δ)` or min-max normalization.
   - `M̃(x) = M(x) / (mean(M) + δ)` if provided; otherwise `M̃(x) = 1`.

### 4.3 Initialization: Content- and Importance-Aware Triangulation

We adapt Image-GS’s content-adaptive initialization to a triangle mesh:

1. **Sampling points.**
   - Sample an initial set of `N_init` 2D points `{p_k}` in `[0,1]^2` according to a mixture distribution:
     - `P(x) ∝ (1 - λ_grad - λ_imp) * U + λ_grad * G̃(x) + λ_imp * M̃(x)`  
       with:
       - `U`: uniform density over the image domain.
       - `λ_grad`: weight for gradient-based emphasis (high-frequency regions).
       - `λ_imp`: weight for importance map emphasis.
     - Example: `λ_grad = 0.3`, `λ_imp = 0.3`.
2. **Triangulation.**
   - Perform 2D Delaunay triangulation over `{p_k}` to obtain an initial mesh.
   - Treat resulting triangles as a **triangle soup**: ignore connectivity for optimization, but connectivity can still be reused for initialization and regularization.
3. **Initial colors.**
   - For each vertex, sample initial color as the ground-truth pixel color at that position (bilinear sampling).
4. **Initial σ and o.**
   - Set `σ` to a moderate value (e.g., corresponding to slightly soft edges) but optionally modulate by local gradient:
     - `σ_init ∝ 1 / (α + local_gradient)`, so regions with stronger edges start sharper.
   - Set opacity `o` initially to 1.0 for all triangles.

This initialization gives more vertices/triangles where either gradients or importance are high, emulating Image-GS’s content- and semantics-aware primitive distribution.

### 4.4 Differentiable Rendering and Loss

Implement a differentiable triangle splatting renderer:

1. Given current triangles, render `I_hat` via tile-based Top-K triangle splatting.
2. Define a **weighted reconstruction loss**, combining:
   - Pixel-wise L1 or L2:
     - `L_rec = ∑_p w(p) * ||I_hat(p) - I_gt(p)||_1`
   - Structural similarity term (e.g., MS-SSIM-based):
     - `L_ssim = 1 - MS-SSIM(I_hat, I_gt)` (possibly also importance-weighted).
3. Combine into final image loss:
   - `L_img = (1 - λ_ssim) * L_rec + λ_ssim * L_ssim`.
4. **Importance-weighted loss.**
   - Choose pixel weights:
     - `w(p) = (1 - α_imp) + α_imp * M̃(p)`.
   - This increases loss contributions in important regions without forcing the user to rely on implicit learning of importance.

### 4.5 Regularization and Triangle-Specific Losses

Borrowing ideas from Triangle Splatting:

- **Smoothness/size regularization:**
  - Encourage triangles to have reasonable sizes and avoid degenerate shapes.
  - E.g., `L_area = β_area * ∑_t (1 / (area_t + ε))` to avoid overly tiny triangles unless needed; or a term that mildly encourages larger triangles where possible.
- **Sharpness regularization:**
  - Encourage σ to be neither too small everywhere nor too large:
  - `L_σ = β_σ * ∑_t (σ_t - σ_prior)^2` or a piecewise penalty that:
    - Prefers small σ near edges / high gradient / high importance.
    - Allows larger σ in smooth, low-importance regions.
- **Color smoothness regularization (optional):**
  - Encourage color variation within a triangle to be consistent with the image:
  - For each triangle, penalize large color differences between vertices unless justified by local gradients.

Total training loss:

- `L_total = L_img + L_area + L_σ + L_other`  
  - `L_other` can include simple priors (e.g., limiting the dynamic range of colors, vertex positions within [0,1]^2).

### 4.6 Progressive Densification (Triangle Splitting / Cloning)

Analogous to both Image-GS’s progressive Gaussian addition and Triangle Splatting’s densification:

1. **Error / importance map for triangles.**
   - Track per-triangle statistics during training:
     - Average reconstruction error over pixels where the triangle significantly contributes.
     - Average importance `M̃` over those pixels.
     - Maybe average gradient magnitude.
2. **Sampling triangles to densify.**
   - Define a probability for selecting triangle `t` for densification:
     - `P_densify(t) ∝ (error_t)^γ * (1 + κ_imp * importance_t)`
   - Sample a batch of triangles according to this distribution once every `K` training steps.
3. **Splitting operation (midpoint subdivision).**
   - For each selected triangle, split it into four smaller triangles by joining edge midpoints (as in Triangle Splatting):
     - Compute midpoints of edges `m12 = (v1+v2)/2`, `m23`, `m31`.
     - Create four new triangles: `(v1, m12, m31)`, `(m12, v2, m23)`, `(m31, m23, v3)`, `(m12, m23, m31)`.
   - Initialize new vertices’ colors by interpolating existing vertex colors.
   - Inherit σ and o from the parent triangle, optionally with small noise.
4. **Cloning small triangles (optional).**
   - If a triangle’s area is below a threshold but its contribution/error is high, cloning (instead of further splitting) can allow local refinement by adjusting positions and colors.
5. **Budget and schedule.**
   - Start with `N_init` triangles.
   - Densify in stages every `K` optimization steps until a target budget `N_max` is reached or desired quality is achieved.

This achieves content- and importance-adaptive refinement similar to Image-GS but using triangle primitives and triangle-splatting geometry.

### 4.7 Pruning Low-Contribution Triangles

To keep representation compact and reduce redundancy:

1. During training, track for each triangle:
   - Maximum contribution weight `max_p w_t(p)` across the image (or over mini-batches).
2. Periodically prune triangles whose:
   - `max_p w_t(p) < τ_prune` (very low influence), and
   - Area is tiny and their removal minimally worsens the loss (optional re-check).
3. Optionally, consider per-triangle visibility:
   - If a triangle never significantly contributes to any pixel over several iterations, remove it.

This is a 2D simplification of Triangle Splatting’s pruning heuristic, tailored to images rather than multi-view scenes.

---

## 5. Compression and Bitstream Design

After optimization, we have a set of triangles with real-valued parameters. We transform this into a compressed bitstream.

### 5.1 Quantization

Apply quantization schemes mindful of rate–distortion trade-offs:

- **Vertex positions:**
  - Map `[0,1]` to integer grid of size `2^B_pos` per axis (e.g., `B_pos = 16`).
  - Quantize each coordinate: `x̂ = round(x * (2^B_pos - 1))`.
- **Colors:**
  - Convert RGB to a decorrelated color space (YCoCg / YUV).
  - Quantize each channel with `B_col` bits (e.g., 8–10 bits).
  - Optionally, use non-uniform quantization or per-channel step sizes based on statistics.
- **Smoothness σ:**
  - Work in log domain: `s = log σ`.
  - Quantize `s` to a small number of bits (e.g., 6–8 bits), or use a learned/shared codebook index.
- **Opacity o (if kept):**
  - Quantize to a few bits or eliminate if pruning already ensures triangles are either useful or removed.

Quantization schemes can be tuned for the desired bitrate range; Image-GS’s analysis of bitrate vs number of primitives is a guide for setting triangle counts and precision.

### 5.2 Entropy Coding

Encode quantized parameters with simple entropy coding (arithmetic or range coding):

- Exploit spatial coherence:
  - Delta-code vertex positions relative to a coarse grid or a reference point.
  - Encode triangles in a spatial order (e.g., tile-scan or Morton order).
- Exploit parameter correlations:
  - Separate streams for positions, colors, σ, and o.
  - Use context models based on neighboring triangles or LOD level.

While Image-GS originally avoids entropy coding to preserve data locality, here we can use light-weight entropy coding while still structuring the data for tile-based access. A trade-off can be chosen depending on implementation complexity.

### 5.3 Bitstream Structure and LOD

One possible high-level bitstream layout:

1. **Header:**
   - Image width `W`, height `H`.
   - Tile size `T`.
   - Number of channels `C`.
   - Quantization bit-depths (`B_pos`, `B_col`, etc.).
   - Number of triangles and LOD boundaries: `[N0, N1, …, NL]`.
2. **Triangle parameter stream:**
   - Triangles ordered by importance (LOD order).
   - For each triangle:
     - Quantized vertex positions.
     - Quantized vertex colors.
     - Quantized σ (and o if used).
3. **(Optional) Tile index:**
   - Either reconstruct per-tile triangle lists on the fly from vertex positions, or store a compact tile-to-triangle index to accelerate rendering.

The decoder can:

- Read header, allocate buffers.
- Read only the subset of triangles corresponding to desired LOD(s).
- Build per-tile triangle lists and start splatting on GPU.

---

## 6. Semantics / Importance Map Integration

The algorithm assumes the **importance map M is provided externally** (by a network or user) and focuses on how to use it:

- **Initialization:**
  - Importance enters the sampling distribution for initial points (Section 4.3) via weight `λ_imp`.
  - High-importance regions get denser initial triangulation.
- **Loss weighting:**
  - Pixel weights `w(p)` in `L_rec` and `L_ssim` incorporate M̃:
    - Larger errors in high-importance regions contribute more strongly to the total loss.
- **Densification:**
  - Per-triangle importance statistics (average M̃ over triangle support) influence densification probability.
  - Triangles covering high M̃ areas are more likely to be selected for splitting.
- **LOD design:**
  - When assigning triangles to LOD levels, triangles that improve high-importance regions can be pushed earlier in the hierarchy, so early LODs already respect user priorities.

This design keeps semantic / importance guidance:

- Optional but simple to plug in.
- Under explicit control of the user (who supplies M).
- Orthogonal to the core representation and codec.

---

## 7. Comparison to Image-GS and Triangle Splatting

Conceptually, Image-TS:

- Adopts from **Image-GS**:
  - Content-adaptive, gradient-guided initialization.
  - Error-guided progressive refinement / LOD behavior.
  - Tile-based rendering with top-K selection for locality and fast random access.
  - Optional semantics-aware / importance-weighted compression.
- Adopts from **Triangle Splatting**:
  - Triangle primitives as the single representation.
  - A compactly-supported window function with a smoothness parameter σ, used for differentiable rasterization.
  - Adaptive splitting/pruning strategies to allocate representation power spatially.

But it differs from both by:

- Restricting the domain to **2D images** (no cameras or 3D geometry).
- Using **no Gaussian primitives** at all, only triangles with a splatting-like window.
- Focusing on **image compression** and progressive LOD coding rather than view synthesis.

---

## 8. Implementation Notes and Possible Extensions

This section outlines practical considerations that a competent developer should account for when implementing the design.

- **Renderer implementation:**
  - Implement the triangle-splatting renderer as:
    - A CUDA / compute shader kernel operating per tile, or
    - A fragment shader with per-pixel evaluation of local triangle lists.
  - Ensure stable gradients by:
    - Clamping σ within reasonable bounds.
    - Using soft top-K with straight-through gradients if strict top-K is non-differentiable.
- **Training schedule:**
  - Use Adam or similar optimizer.
  - Train for a fixed number of steps or until validation loss plateaus, interleaving densification and pruning stages.
- **Quality control / rate control:**
  - Run multiple training passes with different triangle budgets to map out rate–distortion curves.
  - Alternatively, run a single progressive training run with growing triangle budget and snapshot the model at desired bitrates, similar to Image-GS.
- **Random access / streaming:**
  - To support fast random access to subregions (tiles), structure the bitstream such that tile-level triangle subsets can be located quickly, or store a coarse index.
- **Optional neural components (future extension, not required now):**
  - A learned predictor for initial vertex positions / colors to speed up convergence.
  - A neural module to predict σ initialization or to propose densification candidates.

This design should be implementable by a single experienced developer familiar with GPU programming and differentiable rendering, and it leaves room for later experimentation with rate–distortion trade-offs, alternative losses, and entropy coding strategies.
