# Technical Specification v2: Point-Cloud-Encoded Structure-Preserving Generator for Fast Meshfree Shape Function Construction

**Status:** Implementation-Ready (Final Revised)  
**Version:** v2 — incorporates all mandatory corrections from three review rounds

---

## 0. Executive Summary

This document specifies the complete implementation plan for a neural shape function generator that maps local point cloud patches to meshfree shape function values. The method decouples a **geometric encoder backbone** (responsible for learning local spatial relationships) from a **structure-preserving output head** (responsible for enforcing all numerical constraints by construction). Four competing backbone architectures are systematically compared. The output head enforces partition of unity, locality, and first-order consistency through explicit architectural layers rather than soft-penalty losses.

**Key method assumption (stated upfront):** The default scheme prioritizes ensuring zeroth-order and first-order consistency together with fast approximate usability. Strict non-negativity of the final corrected shape functions is not imposed as a hard constraint, because the MLS reproducing correction may introduce small negative values. This trade-off is appropriate for the target application of fast shape function generation in large-deformation meshfree analysis, where moderate precision at high speed is preferred over exact non-negativity at higher cost.

---

## 1. Problem Definition

### 1.1 Formal Operator Learning Statement

The task is to learn a parametric mapping:

$$\mathcal{G}_\theta : (\mathbf{x}_q, \{\mathbf{X}_i\}_{i=1}^{k}, \beta, \rho_q) \longmapsto \{\phi_i(\mathbf{x}_q)\}_{i=1}^{k}$$

| Symbol | Meaning | Type / Shape |
|--------|---------|-------------|
| $\mathbf{x}_q$ | Query (evaluation) point | $\mathbb{R}^d$ (d = 2 or 3) |
| $\{\mathbf{X}_i\}$ | k-nearest neighbor nodes within support domain | $\mathbb{R}^{k \times d}$ |
| $\beta$ | Thermodynamic / dilation parameter (max-ent width) | $\mathbb{R}^+$ |
| $\rho_q$ | Optional local geometric context (see Section 3.1) | $\mathbb{R}^{n_\rho}$ |
| $\{\phi_i(\mathbf{x}_q)\}$ | Shape function values at the query point for each neighbor | $\mathbb{R}^k$ |

### 1.2 What This Problem Is NOT

This formulation is distinct from:

- **Global field prediction:** FNO and Transolver learn global PDE solution fields; our output is local weights on a small patch.
- **Fixed-dimensional regression:** KAN treats this as a fixed-input function; our input is variable-geometry and unordered.
- **Regular-grid operator learning:** FNO requires structured grids; our input is inherently irregular point clouds.

### 1.3 Typical Patch Dimensions

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| k (number of neighbors) | 8–32 | 2D: 8–20; 3D: 15–32 |
| d (spatial dimension) | 2 or 3 | Start with 2D |
| Patch radius | 1.5h–3.0h | h = average nodal spacing |

---

## 2. Method Assumptions and Scope

This section explicitly states the design choices and their rationale, so that the scope of the method is clear before implementation details are presented.

**Assumption 1: Consistency over strict non-negativity.** The output head guarantees zeroth-order consistency (partition of unity) and first-order consistency (linear reproducibility) by construction. The MLS reproducing correction in Step B4 may introduce small negative values in some shape function entries. This is accepted as a design trade-off. The rationale is twofold: (a) the target application (fast shape function generation for large deformation) prioritizes speed and consistency over pointwise non-negativity; (b) enforcing strict non-negativity after MLS correction would require an iterative clamp-and-re-correct procedure that destroys the closed-form nature of the output head and adds computational cost. If strict non-negativity is required for a specific downstream application, an optional post-processing step can be applied (documented in Section 9), but this is not the default.

**Assumption 2: Approximate accuracy is acceptable.** The generator is designed to produce shape functions that are "good enough" for use as initial approximations in large-deformation meshfree analysis — either directly in a fast simulation or as warm-start inputs for iterative refinement. The target relative error against reference max-ent shape functions is on the order of 1–5%, not machine precision.

**Assumption 3: 2D first, 3D as extension.** The primary development and validation is in 2D. Extension to 3D is straightforward (the basis vector $\mathbf{p}$ gains one additional entry; patch sizes increase) but is treated as a follow-up, not a core deliverable.

---

## 3. Module A — Local Geometric Encoder (Backbone)

### 3.1 Input Preprocessing (Shared Across All Backbones)

All backbones receive the same preprocessed input. For each patch:

**Step 1: Query-centered normalization.** Translate all coordinates so the query point is at the origin:

$$\mathbf{r}_i = \mathbf{X}_i - \mathbf{x}_q, \quad i = 1, \ldots, k$$

**Step 2: Scale normalization.** Divide by the support radius to obtain dimensionless coordinates:

$$\hat{\mathbf{r}}_i = \mathbf{r}_i / r_{\max}$$

where $r_{\max}$ is the support domain radius (set as the distance to the farthest neighbor in the patch, or a fixed multiple of the average nodal spacing $h$).

**Step 3: Per-node feature vector construction.** For each neighbor $i$, assemble:

$$\mathbf{f}_i = \bigl[\hat{\mathbf{r}}_i,\; \|\hat{\mathbf{r}}_i\|,\; \beta,\; \rho_q\bigr] \in \mathbb{R}^{d_f}$$

The local geometric context $\rho_q$ consists of two categories of features:

*Local density features:*
- Mean nearest-neighbor distance within the patch (normalized by $r_{\max}$)
- Number of neighbors $k$ (as a scalar, or encoded as $k / k_{\max}$)

*Local anisotropy features:*
- Eigenvalue ratio of the neighborhood covariance matrix $\text{Cov}(\{\hat{\mathbf{r}}_i\})$, i.e., $\lambda_{\min} / \lambda_{\max}$. This captures whether the patch is isotropic ($\approx 1$) or elongated ($\ll 1$).

For 2D with both density and anisotropy context: $d_f = 2 + 1 + 1 + 3 = 7$ (two coordinates, distance, $\beta$, mean NN distance, $k/k_{\max}$, eigenvalue ratio). These additional context features are optional and will be evaluated via feature ablation in the experiments. The minimum feature set is $\mathbf{f}_i = [\hat{\mathbf{r}}_i, \|\hat{\mathbf{r}}_i\|, \beta] \in \mathbb{R}^4$ for 2D.

**Tensor shape going into the backbone:** $\mathbf{F} \in \mathbb{R}^{k \times d_f}$.

### 3.2 Backbone Candidate 1: Kernel-Integral Operator (Recommended Primary)

**Rationale.** This architecture directly mirrors the kernel integral operator framework from Li et al. (2020b), adapted for local patches. It speaks the native language of meshfree methods — kernels, support domains, distance-based weighting — and provides built-in geometric inductive bias.

**Architecture.** The backbone consists of $L$ message-passing layers ($L = 2$–$3$ is sufficient for small patches). In each layer:

**Lifting (first layer only):**

$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{lift}}(\mathbf{f}_i) \in \mathbb{R}^{C}$$

with $C = 32$–$64$ (hidden channels).

**Message passing (layer $\ell$):**

$$\mathbf{m}_i^{(\ell)} = \sum_{j=1}^{k} \alpha_{ij}^{(\ell)} \, \mathbf{v}_j^{(\ell)}$$

where:

$$\alpha_{ij}^{(\ell)} = k_{\text{base}}\!\left(\|\hat{\mathbf{r}}_i - \hat{\mathbf{r}}_j\|\right) \cdot g_\theta^{(\ell)}(\mathbf{h}_i^{(\ell)}, \mathbf{h}_j^{(\ell)}, \hat{\mathbf{r}}_i, \hat{\mathbf{r}}_j) \in \mathbb{R}$$

$$\mathbf{v}_j^{(\ell)} = \text{Linear}_V^{(\ell)}(\mathbf{h}_j^{(\ell)}) \in \mathbb{R}^{C}$$

Here $k_{\text{base}}$ is a fixed compactly-supported radial kernel (cubic B-spline or Wendland C2) and $g_\theta^{(\ell)}$ is a small learnable network:

$$g_\theta^{(\ell)}(\mathbf{h}_i, \mathbf{h}_j, \hat{\mathbf{r}}_i, \hat{\mathbf{r}}_j) = \text{MLP}^{(\ell)}\!\bigl([\mathbf{h}_i - \mathbf{h}_j,\; \hat{\mathbf{r}}_i,\; \hat{\mathbf{r}}_j,\; \|\hat{\mathbf{r}}_i - \hat{\mathbf{r}}_j\|]\bigr) \in \mathbb{R}$$

This pair-feature construction includes both the feature difference $\mathbf{h}_i - \mathbf{h}_j$ and the individual query-relative positions $\hat{\mathbf{r}}_i, \hat{\mathbf{r}}_j$. This is important because the task is query-centered: simply using $\mathbf{h}_i - \mathbf{h}_j$ alone may compress the role difference of nodes relative to the query point. The inter-node distance $\|\hat{\mathbf{r}}_i - \hat{\mathbf{r}}_j\|$ is the distance between node $i$ and node $j$ (not query-to-node), which is the natural pair distance for message aggregation.

**Update:**

$$\mathbf{h}_i^{(\ell+1)} = \sigma\!\bigl(\mathbf{W}^{(\ell)} \mathbf{h}_i^{(\ell)} + \mathbf{m}_i^{(\ell)}\bigr)$$

where $\sigma$ is GELU activation and $\mathbf{W}^{(\ell)} \in \mathbb{R}^{C \times C}$ is a local linear transform (analogous to the bypass branch in FNO's Fourier layer).

**Readout:** After $L$ layers, project to a scalar logit per node:

$$l_i = \text{Linear}_{\text{out}}(\mathbf{h}_i^{(L)}) \in \mathbb{R}$$

**Tensor shapes per layer:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{h}_i^{(\ell)}$ | $\mathbb{R}^{k \times C}$ | Node hidden states |
| $\alpha_{ij}^{(\ell)}$ | $\mathbb{R}^{k \times k}$ | Kernel-modulated attention weights |
| $\mathbf{v}_j^{(\ell)}$ | $\mathbb{R}^{k \times C}$ | Value embeddings |
| $l_i$ | $\mathbb{R}^{k}$ | Output logits |

**Parameter count estimate:** With $C = 64$, $L = 2$, and small MLPs (2 layers, 64 hidden), total parameters are approximately 30k–50k.

### 3.3 Backbone Candidate 2: DeepSets / Set Transformer-lite (Strong Baseline)

**Architecture.** Per-node MLP feature extraction followed by permutation-equivariant pooling:

$$\mathbf{h}_i = \text{MLP}_{\text{node}}(\mathbf{f}_i)$$

$$\mathbf{c} = \frac{1}{k}\sum_{j=1}^{k} \mathbf{h}_j \quad \text{(global context via mean pooling)}$$

$$l_i = \text{MLP}_{\text{out}}([\mathbf{h}_i,\; \mathbf{c}])$$

**Variant (Set Transformer-lite):** Replace mean pooling with a single induced-point attention layer (1–4 induced points) for richer inter-node interaction.

**Purpose in experiments:** Establishes whether simple permutation-equivariant processing is sufficient, or whether explicit geometric kernel structure (Candidate 1) provides meaningful improvement.

### 3.4 Backbone Candidate 3: Transolver-lite (Competing Implementation)

**Architecture.** A single-layer Physics-Attention from Wu et al., with $M = 4$–$8$ slices:

$$\mathbf{w}_i = \text{Softmax}\!\bigl(\text{Linear}(\mathbf{h}_i)\bigr) \in \mathbb{R}^M$$

$$z_j = \frac{\sum_i w_{ij} \mathbf{h}_i}{\sum_i w_{ij}} \in \mathbb{R}^C \quad \text{(aggregate to tokens)}$$

$$\mathbf{z}' = \text{Self-Attention}(\mathbf{z}) \quad \text{(attend among $M$ tokens)}$$

$$\mathbf{h}_i' = \sum_j w_{ij} z_j' \quad \text{(broadcast back)}$$

$$l_i = \text{Linear}_{\text{out}}(\mathbf{h}_i')$$

**Purpose in experiments:** Tests whether Transolver's slice-based token compression provides meaningful benefit at the small-patch scale (8–32 nodes), where the quadratic-to-linear complexity reduction may not be the primary concern.

### 3.5 Backbone Candidate 4: MLP Baseline (Minimum Benchmark)

**Architecture.** Sort neighbors by distance from query point (to impose a canonical ordering), concatenate all features into a single vector, pass through a standard MLP:

$$\mathbf{l} = \text{MLP}\!\bigl([\mathbf{f}_{\pi(1)},\; \mathbf{f}_{\pi(2)},\; \ldots,\; \mathbf{f}_{\pi(k)}]\bigr) \in \mathbb{R}^k$$

where $\pi$ is the distance-based sorting permutation.

**Important limitation:** This baseline requires a fixed input dimension, which means it must be trained and evaluated at a single fixed $k$. This is a known fairness limitation and will be acknowledged explicitly in the experimental discussion (see Section 6). The MLP baseline serves only as a minimum benchmark to establish that geometric inductive bias matters, not as a universal competitor.

---

## 4. Variable-$k$ Handling

Since the number of neighbors $k$ varies across query points and patch configurations, a consistent strategy for handling variable-length inputs is required. This section specifies how variable-$k$ enters each backbone.

### 4.1 Training Scheme

**Primary approach: fixed $k$ within each batch, variable $k$ across batches.** During training, patches are grouped into batches by their neighbor count $k$. Within each batch, all patches have the same $k$, eliminating the need for padding. Across batches, $k$ varies (e.g., $k \in \{8, 10, 12, 14, 16, 18, 20\}$ for 2D).

**Alternative approach (if batch-by-$k$ is impractical): padding + masking.** Pad all patches to $k_{\max}$ and use a binary mask $\mathbf{mask} \in \{0, 1\}^{k_{\max}}$ to indicate valid nodes.

### 4.2 How the Mask Enters Each Backbone

**Kernel-integral operator (Candidate 1):** The pairwise kernel matrix $\alpha_{ij}$ must be masked: set $\alpha_{ij} = 0$ whenever node $i$ or node $j$ is a padding node. The aggregation $\mathbf{m}_i = \sum_j \alpha_{ij} \mathbf{v}_j$ then naturally excludes padding nodes. The readout layer only outputs logits for valid nodes.

**DeepSets / Set Transformer-lite (Candidate 2):** Mean pooling divides by the actual number of valid nodes, not by $k_{\max}$. In the Set Transformer variant, attention scores for padding nodes are set to $-\infty$ before softmax.

**Transolver-lite (Candidate 3):** The slice weight computation masks padding nodes by setting their hidden features to zero before the projection layer, or by masking the softmax. Token aggregation (Eq. 2) divides by the sum of valid weights only.

**MLP baseline (Candidate 4):** This backbone is only evaluated on a fixed-$k$ subset. This is a known limitation and does not affect the primary conclusions of the backbone comparison.

### 4.3 How the Mask Enters the Output Head

In the output head (Module B):

- **B1 (softplus):** Applied only to valid logits. Padding positions output zero.
- **B2 (window modulation):** Padding positions receive $w_i = 0$.
- **B3 (normalization):** Sum in the denominator runs over valid nodes only.
- **B4 (MLS correction):** The moment matrix $\mathbf{M}$ and the basis evaluations $\mathbf{p}(\mathbf{X}_i)$ are computed over valid nodes only.

### 4.4 How the Mask Enters the Loss

The data loss $\mathcal{L}_{\text{data}}$ sums only over valid (unmasked) entries. Padding positions contribute zero loss.

---

## 5. Module B — Structure-Preserving Output Head

This module is identical for all backbones. It transforms raw logits $\{l_i\}_{i=1}^k$ into shape function values through four steps, expressed as a unified pipeline.

### 5.1 Complete Pipeline (Unified Notation)

**Step B1 — Non-negative raw weights:**

$$a_i = \text{softplus}(l_i) = \ln(1 + e^{l_i})$$

Softplus is chosen over ReLU because it is everywhere differentiable and avoids dead neurons.

**Step B2 — Window modulation (locality enforcement):**

$$\tilde{a}_i = a_i \cdot w_i, \qquad w_i = w\!\left(\frac{\|\mathbf{X}_i - \mathbf{x}_q\|}{r_{\max}}\right)$$

where $w(s) = (1 - s^2)^2_+$ (quartic spline) or $w(s) = \max(0, 1-s)^4(4s+1)$ (Wendland C2). Locality is jointly achieved by neighborhood truncation (only k-nearest neighbors are in the patch) and window modulation within the patch (smooth attenuation toward the support boundary).

**Step B3 — Normalization (partition of unity for base weights):**

$$\phi_i^{\text{base}} = \frac{\tilde{a}_i}{\sum_{j=1}^{k} \tilde{a}_j}$$

At this point: $\phi_i^{\text{base}} \geq 0$ and $\sum_i \phi_i^{\text{base}} = 1$.

**Step B4 — Linear reproducing correction using $\phi^{\text{base}}$ as the weight function:**

$$\mathbf{M}(\mathbf{x}_q) = \sum_{j=1}^{k} \phi_j^{\text{base}} \, \mathbf{p}(\mathbf{X}_j)\,\mathbf{p}(\mathbf{X}_j)^T \in \mathbb{R}^{n_p \times n_p}$$

$$\mathbf{c}(\mathbf{x}_q) = \mathbf{M}(\mathbf{x}_q)^{-1}\,\mathbf{p}(\mathbf{x}_q) \in \mathbb{R}^{n_p}$$

$$\phi_i(\mathbf{x}_q) = \phi_i^{\text{base}} \cdot \mathbf{p}(\mathbf{X}_i)^T \mathbf{c}(\mathbf{x}_q)$$

where $\mathbf{p}(\mathbf{x}) = [1, x_1, x_2]^T \in \mathbb{R}^3$ for 2D linear reproduction ($n_p = 3$).

This is a standard reproducing correction (Liu et al. 1995; Chen et al. 1996; Lancaster and Salkauskas 1981) that uses the base shape functions $\phi_i^{\text{base}}$ as the weight function to construct a moment matrix, then applies a corrective factor to enforce linear consistency. It is NOT a generic "MLS correction" but specifically a "linear reproducing correction with $\phi^{\text{base}}$ as the weight function."

### 5.2 Properties Guaranteed After B4

- **Zeroth-order consistency:** $\sum_i \phi_i(\mathbf{x}_q) = 1$. This follows from $\mathbf{p}(\mathbf{x}_q)^T \mathbf{M}^{-1} \sum_j \phi_j^{\text{base}} \mathbf{p}(\mathbf{X}_j) \mathbf{p}(\mathbf{X}_j)^T = \mathbf{p}(\mathbf{x}_q)^T$, by definition of $\mathbf{M}$.
- **First-order consistency:** $\sum_i \phi_i(\mathbf{x}_q) \mathbf{X}_i = \mathbf{x}_q$. This follows from the same reproducing property applied to the linear basis functions.

Both properties are guaranteed by the algebraic structure of the correction, not by training. Under double precision and well-conditioned patches, the numerical residuals should be close to machine precision. However, the actual residuals depend on the condition number of $\mathbf{M}$, the precision format (float32 vs. float64), and implementation details. **Empirical residual statistics will be reported in the experiments without presupposing a fixed threshold.**

### 5.3 Non-Negativity After Correction

The MLS correction may produce small negative values in some $\phi_i$. Per the method assumptions stated in Section 2, this is accepted as a design trade-off. The default scheme does not impose a post-correction non-negativity clamp.

To monitor the extent of negative values during training and evaluation, a lightweight negative-weight monitoring term is included in the loss (see Section 6.1):

$$\mathcal{L}_{\text{neg}} = \frac{1}{N_s} \sum_s \sum_i \max(0, -\phi_i^{(s)})$$

This term is weighted very lightly ($\lambda_{\text{neg}} = 10^{-5}$). Its purpose is diagnostic: it allows tracking whether certain patch configurations systematically produce large negative corrections, which would indicate problems with the base weights from the backbone.

### 5.4 Numerical Stability of $\mathbf{M}^{-1}$

The moment matrix $\mathbf{M}$ can become ill-conditioned for degenerate patches (e.g., nearly collinear nodes in 2D). Mitigation:

- Add diagonal regularization: $\mathbf{M}_{\text{reg}} = \mathbf{M} + \epsilon \mathbf{I}$ with $\epsilon = 10^{-10}$.
- During data generation, compute and log $\text{cond}(\mathbf{M})$ for each training sample. Flag patches with $\text{cond}(\mathbf{M}) > 10^6$ for inspection.
- In the experimental results, report the distribution of $\text{cond}(\mathbf{M})$ across the test set.

### 5.5 Differentiability

The entire B1–B4 pipeline is differentiable with respect to the backbone parameters $\theta$. The moment matrix inversion $\mathbf{M}^{-1}$ is differentiable via standard linear algebra backpropagation. No re-normalization is applied after B4, as this would disturb first-order consistency.

### 5.6 Do NOT Re-Normalize After B4

This is critical and worth stating explicitly: do not apply additional normalization after the reproducing correction. The correction formula simultaneously guarantees both zeroth-order and first-order consistency. An additional normalization step would preserve zeroth-order but break first-order consistency.

---

## 6. Training Strategy

### 6.1 Phase 1: Reference-Guided Warm Start (Primary)

**Data generation.** Generate training data offline by solving the standard max-ent optimization for a large variety of random patch configurations.

**Patch sampling protocol:**

| Parameter | Sampling Distribution |
|-----------|-----------------------|
| Number of neighbors $k$ | Uniform integer from [8, 25] (2D) |
| Node positions | Uniform random in circular/square domain, with controlled irregularity |
| Regularity | Mix of: uniform (20%), mildly perturbed (30%), highly random (20%), clustered (15%), boundary-truncated (15%) |
| $\beta$ (dilation) | Log-uniform from [0.5, 8.0] |
| Query position | Uniform within convex hull of neighbors |
| Boundary proximity | 15% of samples include boundary truncation (partial support domain) |

**Critical note on data distribution (primary risk factor).** The success of this work depends heavily on the training patch distribution covering the relevant geometric configurations. The true risk is not computational cost of data generation (each max-ent solve takes milliseconds), but insufficient coverage of pathological patch geometries: extreme boundary truncation, highly anisotropic node distributions, clustered nodes near the query point, and mixed sparse-dense transitions. The sampling protocol above is a starting point; it should be iteratively refined based on failure analysis during development.

**Dataset sizes:**

| Split | Number of Patches | Purpose |
|-------|------------------|---------|
| Training | 50,000–100,000 | Backbone + head training |
| Validation | 5,000 | Hyperparameter selection |
| Test (in-distribution) | 10,000 | Standard evaluation |
| Test (OOD geometry) | 5,000 | Generalization: unseen patch types |
| Test (OOD $\beta$) | 3,000 | Generalization: $\beta$ outside training range |

Each sample is a tuple: $(\mathbf{x}_q, \{\mathbf{X}_i\}, \beta, \{\phi_i^{\text{ref}}\})$, where $\phi_i^{\text{ref}}$ are obtained from the standard max-ent optimization solver.

**Loss function:**

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_c \mathcal{L}_{\text{cons}} + \lambda_{\text{neg}} \mathcal{L}_{\text{neg}}$$

$$\mathcal{L}_{\text{data}} = \frac{1}{N_s} \sum_{s=1}^{N_s} \sum_{i=1}^{k^{(s)}} \bigl(\phi_i^{(s)} - \phi_i^{\text{ref},(s)}\bigr)^2$$

$$\mathcal{L}_{\text{cons}} = \frac{1}{N_s} \sum_{s=1}^{N_s} \left[\left(\sum_i \phi_i^{(s)} - 1\right)^{\!2} + \left\|\sum_i \phi_i^{(s)} \mathbf{X}_i^{(s)} - \mathbf{x}_q^{(s)}\right\|^2\right]$$

$$\mathcal{L}_{\text{neg}} = \frac{1}{N_s} \sum_{s=1}^{N_s} \sum_i \max(0, -\phi_i^{(s)})$$

| Term | Weight | Purpose |
|------|--------|---------|
| $\mathcal{L}_{\text{data}}$ | 1.0 | Primary: match reference shape functions |
| $\mathcal{L}_{\text{cons}}$ | $\lambda_c = 10^{-4}$ | Monitoring + mild regularization (should be near zero if output head works) |
| $\mathcal{L}_{\text{neg}}$ | $\lambda_{\text{neg}} = 10^{-5}$ | Monitoring: track extent of negative corrections |

The consistency and negativity terms are NOT the primary enforcement mechanism — the output head handles constraints structurally. These terms serve as diagnostic signals: if $\mathcal{L}_{\text{cons}}$ is not near zero during training, something is wrong with the output head implementation.

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | $10^{-3}$ with cosine annealing to $10^{-5}$ |
| Batch size | 256–512 patches (grouped by $k$) |
| Epochs | 200–500 (with early stopping on validation loss) |
| Precision | float64 for output head; float32 acceptable for backbone |

### 6.2 Phase 2: Physics-Regularized Refinement (Optional)

After Phase 1 converges, optionally fine-tune with additional physics-motivated terms:

- Smoothness of shape functions across neighboring query points (spatial gradient regularization).
- Max-ent variational principle: maximize $-\sum_i \phi_i \ln \phi_i$ subject to constraints.

This phase is optional and should not be activated until Phase 1 produces a stable, well-performing model.

---

## 7. Experimental Plan

### Experiment 1: Backbone Comparison (Core Result)

**Setup:** Fix the output head (Module B). Train each of the four backbone candidates on the same dataset. Evaluate on the same held-out test set.

**Metrics:**

| Metric | Definition | Notes |
|--------|-----------|-------|
| L2 relative error | $\|\boldsymbol\phi - \boldsymbol\phi^{\text{ref}}\| / \|\boldsymbol\phi^{\text{ref}}\|$, averaged | Primary accuracy metric |
| Max pointwise error | $\max_i |\phi_i - \phi_i^{\text{ref}}|$, averaged | Worst-case per-node accuracy |
| PoU residual | $|\sum_i \phi_i - 1|$ | Should be near zero (output head guarantee) |
| Linear reprod. residual | $\|\sum_i \phi_i \mathbf{X}_i - \mathbf{x}_q\|$ | Should be near zero (output head guarantee) |
| Negative fraction | Fraction of samples with any $\phi_i < 0$ | Monitors correction-induced negativity |
| Max negative magnitude | $\max_i |\min(0, \phi_i)|$, averaged | Severity of negative values |
| Inference time per patch | Wall-clock (GPU and CPU) | Speed metric |
| Training convergence | Epochs to 95% of final accuracy | Efficiency metric |
| Parameter count | Total trainable parameters | Complexity metric |

**Key narrative to establish:** Without a good backbone, the output head can guarantee consistency but not proximity to the reference max-ent shape functions. The backbone determines the quality, smoothness, and generalization of the base weights, while the output head guarantees their numerical legitimacy. Both are necessary; neither is sufficient alone.

### Experiment 2: Output Head Ablation

**Setup:** Fix the best backbone from Experiment 1. Compare four output head variants:

| Variant | B1 (softplus) | B2 (window) | B3 (normalize) | B4 (MLS correction) |
|---------|:---:|:---:|:---:|:---:|
| Full (proposed) | ✓ | ✓ | ✓ | ✓ |
| No correction | ✓ | ✓ | ✓ | — |
| Soft constraint only | — | — | — | — (loss penalties only) |
| No constraints | — | — | — | — (bare regression) |

**Expected outcome:** Full head vastly outperforms soft-constraint and no-constraint variants. The "no correction" variant shows good PoU but poor linear reproducibility, demonstrating that B4 is essential.

### Experiment 3: Cross-Patch Generalization

**Setup:** Train on standard patches. Test on unseen configurations in two categories:

**Geometric OOD:**
- Extreme aspect ratio patches (elongated support domains, eigenvalue ratio $< 0.2$)
- Heavily truncated boundary patches ($< 50\%$ of neighbors present)
- Highly non-uniform density (10:1 density ratio within patch)
- Different $k$ values than training (train on $k \in [10, 20]$, test on $k = 8$ and $k = 25$)

**$\beta$ OOD:**
- Train: $\beta \in [1, 4]$
- Test: $\beta \in [0.5, 1) \cup (4, 8]$

This separation directly answers the question: is the network learning a genuine local geometric mapping, or merely interpolating within a narrow range of $\beta$?

### Experiment 4: Large-Deformation Solver Embedding

**Setup:** Integrate the trained generator into an updated-Lagrangian meshfree solver. At each load increment, query the neural generator for shape functions at the deformed configuration.

**Critical fairness control:** Except for the shape function generation module, all other solver components remain identical across all methods: integration scheme, solution procedure, material model, convergence criteria, and load stepping.

**Benchmark problems:**

| Problem | Type | Why It Matters |
|---------|------|---------------|
| Cook's membrane | Hyperelastic, moderate deformation | Standard benchmark (PFEM uses this) |
| Tensile bar necking | Large strain localization | Tests quality under severe node distortion |
| Punch indentation | Contact + large deformation | Tests boundary truncation handling |

**Comparison methods:**

| Method | Shape Function Source | Expected Result |
|--------|----------------------|-----------------|
| Standard meshfree | Max-ent optimization at every step | Accurate but slow |
| Neural generator | Trained model (single forward pass) | Fast, slightly less accurate |
| KAN (current pipeline) | Your existing KAN + soft constraints | Slow training, mediocre accuracy |

**Metrics:** Total simulation wall-clock time, final displacement error vs. reference, and number of equilibrium iterations per load step.

### Experiment 5: Head-to-Head vs. Current KAN Pipeline

**Setup:** Implement your current KAN + soft-constraint approach with the same training data and compare directly against the best configuration from Experiments 1–2.

**Metrics:** Training time to convergence, inference speed, accuracy, and consistency residuals. This provides the "before vs. after" narrative for the paper.

### Feature Ablation (Supplementary)

Test the impact of enriched input features ($\rho_q$) by comparing: (a) minimum features $[\hat{\mathbf{r}}_i, \|\hat{\mathbf{r}}_i\|, \beta]$, (b) + density features, (c) + anisotropy features, (d) full feature set. This determines whether the additional geometric context features are worth the engineering effort.

---

## 8. Paper Organization

The paper decomposes into three layers of contribution:

**Layer 1 — Problem Definition (Section 2).** For the first time, meshfree shape function generation is formally defined as operator learning over local point cloud patches. This reframes a classical numerical construction problem as a modern machine learning problem with the correct inductive structure.

**Layer 2 — Methodological Contribution (Section 3).** The structure-preserving output head (softplus → window → normalize → reproducing correction). This is the hardest technical contribution: it demonstrates that numerical constraints can and should be enforced architecturally, not via loss penalties.

**Layer 3 — Empirical Comparison (Sections 4–5).** Systematic comparison of four local geometric encoders, plus application validation in large-deformation meshfree analysis.

**Suggested title:** *A Point-Cloud-Encoded Structure-Preserving Generator for Fast Meshfree Shape Function Construction*

---

## 9. Connection to Referenced Literature

| Paper | What It Contributes to This Work |
|-------|----------------------------------|
| **FNO** (Li et al. 2021) | Kernel integral operator formulation (Def. 2) directly inspires the primary backbone. Key adaptation: local patches in physical space, not global domains in Fourier space. |
| **Transolver** (Wu et al.) | Physics-Attention provides one competing backbone. Theorem 3.4 (attention ≡ learnable integral) provides theoretical backing for attention-based encoders on point clouds. |
| **Galerkin/Fourier Transformer** (Cao 2021) | Softmax-free attention interpretation as Petrov-Galerkin projection provides theoretical language. Layer normalization insights may improve training stability. |
| **PFEM** (Wang et al. 2026) | Pretraining–warm-start paradigm inspires the application framework. Explicit function differentiation using shape functions is a natural downstream consumer of the generated shape functions. |
| **Chen et al. 2023** (encoder-decoder meshfree) | Demonstrates attention-enhanced architectures for meshfree surrogate models. The reproducing kernel shape function formula (Eq. 1) is the direct mathematical ancestor of Step B4. |
| **Reproducing kernel theory** (Liu et al. 1995; Chen et al. 1996) | Provides the mathematical foundation for the MLS correction in Step B4. |

---

## 10. Risk Assessment and Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Moment matrix $\mathbf{M}$ ill-conditioned for degenerate patches | Medium | Diagonal regularization $\mathbf{M} + \epsilon\mathbf{I}$. Flag and log ill-conditioned samples. Report $\text{cond}(\mathbf{M})$ distribution. |
| MLS correction produces negative values | Medium | Accepted as design trade-off (Section 2). Monitor via $\mathcal{L}_{\text{neg}}$. Document as method limitation. |
| Insufficient training data coverage of pathological geometries | **Medium-High** | This is the primary data risk. Iteratively refine sampling protocol based on failure analysis. Augment with targeted pathological patches. |
| Neural generator not faster than direct optimization | Low | Single forward pass through 30k-parameter network is inherently cheaper than iterative optimization. Verify with wall-clock benchmarks. |
| MLP baseline unfairly disadvantaged by variable-$k$ | Medium | Acknowledged explicitly. MLP trained on fixed-$k$ subset only. Stated as minimum benchmark, not universal competitor. |
| Reviewers ask "if B4 already guarantees consistency, why does A matter?" | Medium | Address in Experiment 1 + paper narrative: A determines base weight quality (smoothness, accuracy, generalization); B4 only guarantees consistency, not proximity to reference. |

---

## 11. Implementation Checklist

**Phase 0: Data Generation (Week 1)**

- [ ] Implement patch sampling protocol (Section 6.1), including pathological patch types
- [ ] Implement max-ent optimization solver for reference shape functions
- [ ] Compute and log $\text{cond}(\mathbf{M})$ for each sample during data generation
- [ ] Generate and validate training/validation/test/OOD datasets
- [ ] Implement data loading pipeline with batch-by-$k$ grouping (Section 4.1)

**Phase 1: Module B Implementation (Week 2)**

- [ ] Implement softplus → window → normalize → reproducing correction pipeline
- [ ] Implement masking for variable-$k$ in the output head (Section 4.3)
- [ ] **Validation test:** For 1000 random logit vectors, verify that PoU and linear reproducibility residuals are near machine precision (float64)
- [ ] **Validation test:** Log the fraction and magnitude of negative $\phi_i$ values
- [ ] Verify end-to-end differentiability of B1–B4 (compute gradients w.r.t. dummy logits)
- [ ] Implement moment matrix regularization ($\mathbf{M} + \epsilon\mathbf{I}$)

**Phase 2: Module A Implementation (Weeks 3–4)**

- [ ] Implement kernel-integral operator backbone with enriched pair features (Section 3.2)
- [ ] Implement DeepSets/Set Transformer-lite baseline (Section 3.3)
- [ ] Implement Transolver-lite backbone (Section 3.4)
- [ ] Implement MLP baseline on fixed-$k$ subset (Section 3.5)
- [ ] Implement variable-$k$ masking for each backbone (Section 4.2)
- [ ] Verify all backbones produce correct output shapes

**Phase 3: Training (Weeks 4–5)**

- [ ] Train all four backbones with identical settings
- [ ] Monitor $\mathcal{L}_{\text{data}}$, $\mathcal{L}_{\text{cons}}$, and $\mathcal{L}_{\text{neg}}$ curves
- [ ] Run Experiment 1 (backbone comparison)
- [ ] Run Experiment 2 (output head ablation)
- [ ] Run feature ablation (supplementary)

**Phase 4: Evaluation (Weeks 5–6)**

- [ ] Run Experiment 3 (geometric OOD + $\beta$ OOD)
- [ ] Run Experiment 5 (comparison with KAN)
- [ ] Compile results tables and figures
- [ ] Analyze failure cases: which patch types produce worst errors or largest negative values?

**Phase 5: Application Validation (Weeks 6–8)**

- [ ] Integrate best model into meshfree solver
- [ ] Run Experiment 4 (large-deformation benchmarks)
- [ ] Verify fairness control: all solver settings identical except shape function source
- [ ] Compile speed-accuracy tradeoff analysis

**Phase 6: Writing (Weeks 8–10)**

- [ ] Draft paper following the three-layer structure (Section 8)
- [ ] Prepare figures: architecture diagram, output head pipeline, backbone comparison, solver results
- [ ] Explicitly address the "why does A matter if B4 guarantees consistency" question
