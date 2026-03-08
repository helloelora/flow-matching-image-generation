# Project architecture and tracking

This file documents what is already implemented and what must be reported for fair comparison across models.

## Implemented in notebook

Implemented in `flow-matching-euler.ipynb` (Euler-based CFM version):

1. Run folder creation: `runs/cfm_<timestamp>/`
2. Config + env snapshot: `config.json`
3. Training logs:
   - `train_log.jsonl` (per step)
   - `epoch_log.jsonl` (per epoch)
4. Checkpoints:
   - `checkpoints/last.pt`
   - `checkpoints/best.pt`
   - `checkpoints/epoch_XXX.pt`
5. Progress images during training:
   - `samples/progress/progress_epoch_XXX.png` (fixed seed, fixed NFE)
6. Final generated images for comparison:
   - `samples/eval/nfe_10/`
   - `samples/eval/nfe_20/`
   - `samples/eval/nfe_50/`
   - `samples/eval/nfe_100/`
7. Metrics artifacts:
   - `metrics/metrics_summary.json`
   - `metrics/metrics_summary.jsonl`
   - `metrics/fid_results.json`
   - `metrics/runtime_results.json`
8. Run notes: `notes.txt`
9. Resume support:
   - set `cfg.resume_checkpoint` to `runs/<run_id>/checkpoints/last.pt`
   - training continues from next epoch and appends to existing JSONL logs

Note: FID/IS are computed only if `torchmetrics` is available. If not, generated images + runtime are still saved for external evaluation.

## Shared comparison protocol

Apply the same protocol for all methods (`CFM`, `DDPM`, `DDIM`):

1. Same CIFAR-10 preprocessing and image resolution.
2. Same FID real-data stats and same number of generated images.
3. Same NFE points: `10, 20, 50, 100`.
4. Report solver/sampler details explicitly.

## Metrics to report

Primary:

1. `FID` vs `NFE`

Secondary:

1. `IS`
2. Runtime (`sec/image` or `images/sec`) vs `NFE`

Optional (inversion/editing):

1. `PSNR`, `LPIPS`, `MSE` for `x -> z -> x_hat`
2. Qualitative edit grids

## Minimal result schema (per model, per NFE)

- `model`
- `checkpoint_name`
- `nfe`
- `solver_sampler`
- `num_generated`
- `fid`
- `is_mean`
- `is_std`
- `sec_per_image`
- `total_sampling_sec`
- `seed`
- `notes`
