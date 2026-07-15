"""Figures 5-7 - deep-network / over-smoothing analysis on OGBN-ARXIV."""

import os
import sys

_RUN_LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUN_LOCAL_DIR not in sys.path:
    sys.path.insert(0, _RUN_LOCAL_DIR)

import time

import torch

import common
import config
from learning import train, evaluate
from reporting import (
    plot_acc_vs_depth,
    plot_acc_vs_epoch,
    save_results_table,
)

CHECKPOINT_SUBDIR = "oversmoothing"


def _run_family(family_name, variants, cfg, data, split_idx, in_channels,
                loss_fn, ckpt_dir, fig_dir, table_dir):
    from utils import num_params

    depths = cfg["depths"]
    test_acc_by_model = {}
    train_hist_by_model = {}

    total_time = 0.0
    for name, ctor in variants.items():
        test_acc_by_model[name] = []
        train_hist_by_model[name] = []
        for depth in depths:
            common.set_seed(0)
            start = time.perf_counter()
            model = ctor(in_channels, config.OUT_CHANNELS,
                         cfg["hidden_channels"], depth, cfg["dropout"])
            print(f"{name:<10} [depth={depth}] params={num_params(model):,}")

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            best_path = os.path.join(ckpt_dir, f"{name.lower()}_{depth}_best.pt")
            last_path = os.path.join(ckpt_dir, f"{name.lower()}_{depth}_last.pt")
            ckpt = train(model, data, split_idx, optimizer, loss_fn, cfg["epochs"],
                         best_path, last_path, save_train_acc=True)

            model.load_state_dict(ckpt["model_state_dict"])
            _, _, test_acc = evaluate(model, data, split_idx)
            test_acc_by_model[name].append(test_acc * 100)
            train_hist_by_model[name].append(ckpt["train_acc_history"])

            elapsed = time.perf_counter() - start
            total_time += elapsed
            print(f"  test acc {test_acc * 100:.2f}%  ({elapsed / 60:.2f} mins)")

    plot_acc_vs_depth(
        depths, test_acc_by_model,
        out_path=os.path.join(fig_dir, f"fig5_{family_name}_acc_vs_depth.jpg"),
    )

    for name in variants:
        safe = name.lower().replace("-", "_")
        plot_acc_vs_epoch(
            name, depths, train_hist_by_model[name],
            out_path=os.path.join(fig_dir, f"fig_acc_epoch_{safe}.jpg"),
        )

    columns = ["Model"] + [f"{d} layers" for d in depths]
    rows = [
        {"Model": name,
         **{f"{d} layers": f"{acc:.2f}" for d, acc in zip(depths, accs)}}
        for name, accs in test_acc_by_model.items()
    ]
    save_results_table(
        rows, columns,
        csv_path=os.path.join(table_dir, f"depth_stability_{family_name}.csv"),
        md_path=os.path.join(table_dir, f"depth_stability_{family_name}.md"),
        title=f"Test accuracy (%) vs depth - {family_name.upper()} variants",
    )
    return total_time


def run(results_dir=None):
    results_dir = results_dir or common.RESULTS_DIR
    cfg = config.OVERSMOOTHING

    ckpt_dir = common.ensure_dir(os.path.join(results_dir, "checkpoints", CHECKPOINT_SUBDIR))
    fig_dir = common.ensure_dir(os.path.join(results_dir, "figures"))
    table_dir = common.ensure_dir(os.path.join(results_dir, "tables"))

    data, split_idx = common.get_dataset(gnn=True)
    in_channels = data.x.shape[1]
    loss_fn = torch.nn.CrossEntropyLoss()

    total_time = 0.0
    total_time += _run_family("gcn", config.GCN_VARIANTS, cfg, data, split_idx,
                              in_channels, loss_fn, ckpt_dir, fig_dir, table_dir)
    total_time += _run_family("sage", config.SAGE_VARIANTS, cfg, data, split_idx,
                              in_channels, loss_fn, ckpt_dir, fig_dir, table_dir)

    print(f"ALL DONE (over-smoothing). Total {total_time / 60:.2f} mins")


if __name__ == "__main__":
    run()
