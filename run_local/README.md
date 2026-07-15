# `run_local` — Local experiment runners

Local, scriptable version of the Kaggle notebook `it5429e-workspace`. It runs
the deep-network / over-smoothing experiments on **OGBN-ARXIV** and produces the
report artifacts (Table 3, Table 4, Figures 5–7) directly from the command line,
without a notebook.

The models (`../models/`) and the dataset loader (`../utils.py`) from the parent
repository are reused as-is; only the training loop (`learning.py`, previously a
Kaggle-only utility script) and the experiment drivers are added here.

## Task coverage

This package implements the assigned technical tasks:

| Task | Where | Output |
|---|---|---|
| Deep-network experiments at 2 / 10 / 20 / 50 layers | `experiments/oversmoothing.py` | `results/figures/`, `results/tables/depth_stability_*.csv` |
| Training accuracy per epoch | `experiments/oversmoothing.py` | `results/figures/fig_acc_epoch_*.jpg` |
| Stability vs. depth | `experiments/oversmoothing.py` | `results/figures/fig5_*_acc_vs_depth.jpg` |
| Ablation study (GCN & GraphSAGE) | `experiments/ablation.py` | `results/tables/table4_ablation.*` |
| Aggregate results over multiple runs | `experiments/main_result.py`, `experiments/ablation.py` | mean ± std tables |
| Table 3 (main result) | `experiments/main_result.py` | `results/tables/table3_main_result.*` |
| Table 4 (ablation) | `experiments/ablation.py` | `results/tables/table4_ablation.*` |
| Figures 5–7 (over-smoothing) | `experiments/oversmoothing.py` | `results/figures/*.jpg` |

## Layout

```
run_local/
├── common.py        # sys.path setup, device, seeding, cached dataset loader
├── config.py        # model registry + per-experiment hyper-parameters
├── learning.py      # train() / evaluate() (full-batch transductive)
├── reporting.py     # table writers (CSV/Markdown) and figure plotters
├── run_all.py       # CLI entry point
├── experiments/
│   ├── main_result.py    # Table 3
│   ├── oversmoothing.py  # Figures 5–7 + depth-stability tables
│   └── ablation.py       # Table 4
└── results/         # generated at runtime (checkpoints / tables / figures)
```

## Requirements

Same as the parent repo (see `../requirements.txt`): `torch`, `torch_geometric`,
`ogb`, `numpy`, `scikit-learn`, `matplotlib`. On first run, `ogb` downloads and
processes OGBN-ARXIV into `../dataset/`.

## Usage

From the repository root:

```bash
# everything (Table 3 + Figures 5–7 + Table 4)
python run_local/run_all.py --experiment all

# individual experiments
python run_local/run_all.py --experiment main
python run_local/run_all.py --experiment oversmoothing
python run_local/run_all.py --experiment ablation

# custom output directory
python run_local/run_all.py --experiment all --results-dir /path/to/out
```

Each experiment module is also runnable on its own:

```bash
python run_local/experiments/oversmoothing.py
```

## Notes

- Hyper-parameters match the notebook: main result / ablation use
  `hidden=128, epochs=300, 3 runs`; the over-smoothing sweep uses
  `hidden=64, epochs=100, depths=[2,10,20,50]`.
- Runs are seeded (`common.set_seed`) so the aggregated mean ± std is
  reproducible across machines.
- The plotting backend is forced to headless `Agg`, so figures are written to
  disk even on servers without a display.
