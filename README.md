# Ours Method Package

This package contains the runtime code for the proposed method only.

## Included

- model definition
- DAVA/CAVA alignment module
- MIL head
- MLPR reweighting
- EMA teacher-student training
- training and evaluation entry scripts

## Excluded

- dataset files
- data split generation scripts
- benchmark runners
- plotting and reviewer-only utilities

## Main Files

- `scripts/strong_trainer.py`
- `scripts/eval_enhanced.py`
- `configs/default_eswa_retry5_16f_scaled.yaml`

## Training

```bash
python scripts/strong_trainer.py --config configs/default_eswa_retry5_16f_scaled.yaml --output outputs/run_ours
```

## Evaluation

```bash
python scripts/eval_enhanced.py --config configs/default_eswa_retry5_16f_scaled.yaml --checkpoint outputs/run_ours/checkpoints/best_f1.pth --output outputs/eval_ours
```

## Notes

- Prepare the CSV files referenced by the config before running.
- This package is intended for the proposed method path only.
