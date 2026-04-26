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
- `configs/config.yaml`

## Training

```bash
python scripts/strong_trainer.py --config configs/config.yaml --output outputs/run_ours
```

## Evaluation

```bash
python scripts/eval_enhanced.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_f1.pth --output outputs/eval_ours
```

## Notes

- Prepare the CSV files referenced by the config before running.
- This package is intended for the proposed method path only.
- This project is currently undergoing peer review. Once the paper is accepted, we will further refine the codebase.
- The gradient update in the original network relies on GPU computing power during training. If you encounter an “Out Of Memory Error: CUDA out of memory” or similar error during runtime, please stop the process and adjust the settings based on your computer's computing power.
