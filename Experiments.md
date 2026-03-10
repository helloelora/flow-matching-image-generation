# Flow Matching (Euler) - Experiment Notes

## Goal
Quickly identify why generated images looked poor and decide which Euler setup to keep for comparison against DDPM/DDIM.

## Runs tested

### Run 1 - Baseline
- Standard Euler flow-matching training.
- Result: images not convincing, not enough evidence to isolate the bottleneck.

### Run 2 - Longer training (no EMA)
- Same setup as Run 1, more epochs (100).
- Purpose: test whether undertraining was the main issue.

### Run 3 - EMA enabled
- Same as Run 2, but with EMA (`decay ~0.999`) and EMA weights for sampling.
- Purpose: test weight stabilization effect at generation time.

## Locked A/B evaluation (final, comparable)
- Same eval protocol for both runs:
  - same script family
  - same NFEs: `10, 20, 50, 100`
  - same number of generated images: `5000`
  - explicit checkpoints

### Final table

| NFE | FID Run2 (no EMA) | FID Run3 (EMA) | Delta (EMA - Run2) | sec/img Run2 | sec/img Run3 | Delta sec/img |
|---:|---:|---:|---:|---:|---:|---:|
| 10  | 155.552872 | 158.977325 | +3.424454 | 0.003656 | 0.003632 | -0.000025 |
| 20  | 157.015427 | 159.907486 | +2.892059 | 0.006364 | 0.006363 | -0.000001 |
| 50  | 156.509140 | 154.382004 | -2.127136 | 0.014594 | 0.014592 | -0.000002 |
| 100 | 157.733276 | 152.857712 | -4.875565 | 0.028315 | 0.028320 | ~0.000000 |

## Takeaway
- EMA is worse at very low NFE (10/20), but better at higher NFE (50/100).
- Runtime is effectively the same.
- For quality-focused comparisons (typical report setting at NFE 50/100), keep **Run 3 (EMA)** as Euler baseline.

## Decision for project
- Use **Run 3 (EMA)** for Euler vs DDPM vs DDIM comparison.
- Mention low-NFE behavior separately (Run 2 slightly better at 10/20).
