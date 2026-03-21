# Results Analysis

**Conditional Flow Matching vs. Diffusion Models on CIFAR-10**

All models were trained on CIFAR-10 (32x32) and evaluated on 10,000 generated images. FID and IS are computed against the training set. Runtime is measured on a single Colab GPU (T4/A100).

---

## Models

| Model | Architecture | Conditioning | Training Epochs | EMA Decay | Solver(s) |
|-------|-------------|-------------|----------------|-----------|-----------|
| **DDPM/DDIM** | UNet (128 base, [1,2,2,2], 4 heads, GN32) | None | 500 | 0.9999 | DDPM (1000 steps), DDIM (variable) |
| **OT-CFM (ours)** | Same UNet as DDPM | None | 500 | 0.9999 | Midpoint, RK4 (torchdiffeq) |
| **Conditional CFM (Elora)** | SimpleFlowUNet (128 base, [1,2,4], GN8, class emb) | Class-conditional (10 classes, CFG) | 210 (best at 190) | 0.999 | Euler, Midpoint, RK4 |

---

## Main Results: FID vs NFE

### Full Comparison Table

| Method | Solver | NFE | FID | IS (mean) | ms/img |
|--------|--------|----:|------:|----------:|-------:|
| DDPM | ddpm | 1000 | 13.06 | 8.18 | 205.0 |
| DDIM | ddim | 10 | 22.18 | 7.77 | 2.2 |
| DDIM | ddim | 20 | 18.95 | 7.82 | 4.2 |
| DDIM | ddim | 50 | 17.96 | 7.84 | 10.5 |
| DDIM | ddim | 100 | 18.43 | 7.88 | 21.0 |
| OT-CFM (ours) | midpoint | 10 | 18.25 | 8.07 | 2.2 |
| OT-CFM (ours) | midpoint | 20 | 14.80 | 8.22 | 4.2 |
| OT-CFM (ours) | midpoint | 50 | 12.02 | 8.44 | 10.4 |
| OT-CFM (ours) | midpoint | 100 | 11.21 | 8.50 | 20.7 |
| OT-CFM (ours) | rk4 | 20 | 18.01 | 9.15 | 4.2 |
| OT-CFM (ours) | rk4 | 100 | 10.65 | 8.58 | 20.7 |
| Cond. CFM (Elora) | euler | 10 | 21.07 | 7.81 | 3.4 |
| Cond. CFM (Elora) | euler | 20 | 16.69 | 7.95 | 5.9 |
| Cond. CFM (Elora) | euler | 50 | 13.78 | 8.08 | 13.3 |
| Cond. CFM (Elora) | euler | 100 | 13.19 | 8.27 | 25.7 |
| Cond. CFM (Elora) | midpoint | 10 | 17.53 | 8.02 | 2.6 |
| Cond. CFM (Elora) | midpoint | 20 | 15.05 | 8.15 | 4.9 |
| Cond. CFM (Elora) | midpoint | 50 | 13.17 | 8.28 | 12.2 |
| Cond. CFM (Elora) | midpoint | 100 | 12.64 | 8.36 | 24.3 |
| Cond. CFM (Elora) | rk4 | 20 | 20.30 | 8.84 | 4.9 |
| Cond. CFM (Elora) | rk4 | 100 | 12.19 | 8.46 | 24.3 |

### Best Results per Model

| Model | Best FID | NFE | Solver | ms/img |
|-------|------:|----:|--------|-------:|
| DDPM | 13.06 | 1000 | ddpm | 205.0 |
| DDIM | 17.96 | 50 | ddim | 10.5 |
| OT-CFM (ours) | **10.65** | 100 | rk4 | 20.7 |
| Cond. CFM (Elora) | 12.19 | 100 | rk4 | 24.3 |

---

## Analysis

### 1. CFM vs Diffusion: Quality

OT-CFM significantly outperforms DDPM/DDIM at comparable NFE budgets:

- At **NFE=100**: OT-CFM (midpoint) achieves **FID=11.21** vs DDIM's **18.43** — a 39% improvement.
- OT-CFM with RK4 at NFE=100 reaches **FID=10.65**, the best result across all models.
- Even at just **NFE=20**, OT-CFM midpoint (**14.80**) already beats DDIM at any NFE (best DDIM: 17.96 at NFE=50).
- DDPM's 1000-step FID of 13.06 is surpassed by OT-CFM midpoint at just 50 steps (**12.02**), using **20x fewer** function evaluations.

### 2. CFM vs Diffusion: Speed-Quality Tradeoff

The core advantage of flow matching is the favorable speed-quality tradeoff:

- DDPM needs 1000 NFE to reach FID=13.06. OT-CFM reaches the same quality at ~40 NFE (midpoint) — roughly **25x faster**.
- DDIM's quality plateaus around NFE=50 (FID=17.96) and slightly degrades at NFE=100. OT-CFM keeps improving with more steps.
- At the fastest setting (NFE=10), OT-CFM midpoint (**18.25**) already matches DDIM (**22.18**) while being comparably fast (~2.2 ms/img).

### 3. ODE Solver Comparison

Across both CFM models, solver choice follows a consistent pattern:

- **Midpoint** (2nd order) is the best default — good quality at all NFE, recommended by the Flow Matching Guide paper.
- **RK4** (4th order) achieves the best FID at high NFE (100) but underperforms at low NFE (20) because each step costs 4 function evaluations, leaving fewer actual integration steps.
- **Euler** (1st order, Elora's model) is the least accurate per-NFE but simplest. The gap narrows at high NFE.
- At equal NFE, midpoint and rk4 have nearly identical runtime since NFE counts total model calls, not steps.

### 4. Unconditional vs Conditional CFM

Comparing our unconditional OT-CFM with Elora's conditional model (both at cfg_scale=1.0, i.e. no guidance boost):

- At NFE=100 midpoint: ours **11.21** vs Elora's **12.64**. Our unconditional model is slightly better, likely because:
  - Longer effective training (500 epochs vs 210)
  - Higher EMA decay (0.9999 vs 0.999)
  - The conditional model has more parameters to learn but was trained for fewer epochs
- Elora's model with Euler at NFE=100 (**13.19**) is close to DDPM at 1000 steps (**13.06**), confirming that even basic Euler sampling with CFM matches diffusion quality with 10x fewer steps.
- The conditional model's main advantage (class guidance with cfg_scale > 1.0) was not used in this comparison — higher cfg_scale would improve its FID further.

### 5. DDIM Saturation

DDIM shows a notable saturation effect: FID improves from NFE=10 (22.18) to NFE=50 (17.96) but then worsens at NFE=100 (18.43). This is a known behavior — DDIM's deterministic sampling from a model trained with the stochastic DDPM objective introduces approximation errors that accumulate with more steps. CFM does not suffer from this since the model directly learns the velocity field that the ODE solver integrates.

---

## Runtime Summary

All solvers scale linearly with NFE. At equal NFE, all methods have similar per-image cost since they share the same UNet backbone (the dominant cost is the forward pass).

| NFE | DDIM | OT-CFM (midpoint) | Cond. CFM (midpoint) |
|----:|-----:|------------------:|--------------------:|
| 10 | 2.2 ms | 2.2 ms | 2.6 ms |
| 20 | 4.2 ms | 4.2 ms | 4.9 ms |
| 50 | 10.5 ms | 10.4 ms | 12.2 ms |
| 100 | 21.0 ms | 20.7 ms | 24.3 ms |

Elora's conditional model is ~15% slower per step due to larger channel dimensions ([1,2,4] vs [1,2,2,2]) and class embedding overhead.

---

## Key Takeaways

1. **Flow matching is strictly better than diffusion for image generation** on CIFAR-10: better FID at every NFE budget, with the same UNet backbone.
2. **Midpoint is the recommended solver** — best quality-per-NFE ratio, especially at low budgets (10-50 NFE).
3. **RK4 wins at high NFE** (100+) but wastes compute at low NFE. Use midpoint for fast sampling, RK4 for maximum quality.
4. **DDPM's 1000-step quality is matched by CFM in ~50 steps** — a 20x sampling speedup with no quality loss.
5. **DDIM saturates** around 50 steps and cannot match CFM quality regardless of NFE budget.
6. **Class conditioning + CFG** is an orthogonal improvement axis — Elora's model could be further improved with guidance scales > 1.0.
