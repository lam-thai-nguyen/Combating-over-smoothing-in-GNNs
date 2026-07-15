import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 14
COLORS = ["b", "r", "g", "m", "y", "k"]


def save_results_table(rows, columns, csv_path, md_path=None, title=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {csv_path}")

    if md_path is not None:
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        lines = []
        if title:
            lines.append(f"## {title}\n")
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[saved] {md_path}")


def summarize_runs(train_accs, valid_accs, test_accs):
    def fmt(values):
        arr = np.asarray(values, dtype=float) * 100.0
        return arr.mean(), arr.std()

    tr_m, tr_s = fmt(train_accs)
    va_m, va_s = fmt(valid_accs)
    te_m, te_s = fmt(test_accs)
    return {
        "train": f"{tr_m:.2f} ± {tr_s:.2f}",
        "valid": f"{va_m:.2f} ± {va_s:.2f}",
        "test": f"{te_m:.2f} ± {te_s:.2f}",
        "test_mean": te_m,
    }


def plot_acc_vs_depth(depths, series, out_path, ylabel="Test accuracy (%)"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    for (name, accs), c in zip(series.items(), COLORS):
        plt.plot(depths, accs, marker="o", linestyle="-", linewidth=3, c=c, label=name)
    plt.xlabel("Depth", fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.xticks(depths, fontsize=FONT_SIZE - 2)
    plt.legend(fontsize=FONT_SIZE - 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_acc_vs_epoch(name, depths, histories, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    for depth, hist, c in zip(depths, histories, COLORS):
        accs = [a * 100 for a in hist]
        epochs = range(1, len(accs) + 1)
        plt.plot(epochs, accs, linestyle="-", linewidth=3, c=c, label=f"{name}-{depth}")
    plt.xlabel("Epoch", fontsize=FONT_SIZE)
    plt.ylabel("Training accuracy (%)", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE - 2)
    plt.legend(fontsize=FONT_SIZE - 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")
