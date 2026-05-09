"""Generate charts for README from experiment data.

No model loading needed -- all data is from recorded experiments.

Usage:
    python scripts/generate_readme_charts.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)


def plot_progression():
    """MAE improvement trajectory."""
    experiments = [
        ("Global\nMean",       580, "Naive"),
        ("XGBoost\nBaseline",  351, "Baseline"),
        ("Zone-Pair\nMedian",  297, "Features"),
        ("Zone-Pair\nTime-Buck", 278, "Features"),
        ("NN v1",              272, "NN"),
        ("NN v2",              266, "NN"),
        ("NN v3",              265, "NN"),
        ("NN v4b",             264, "NN"),
        ("NN+LGBM",            254, "Ensemble"),
        ("NN+LGBM\n+FT",      253, "Ensemble"),
    ]

    names = [e[0] for e in experiments]
    maes = [e[1] for e in experiments]
    phases = [e[2] for e in experiments]

    colors = {'Naive': '#bdbdbd', 'Baseline': '#ef5350', 'Features': '#42a5f5', 'NN': '#66bb6a', 'Ensemble': '#ab47bc'}
    bar_colors = [colors[p] for p in phases]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), maes, color=bar_colors, edgecolor='white', linewidth=0.5)

    for bar, m in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, f"{m}s",
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.axhline(y=351, color='#ef5350', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(names)-0.5, 356, 'XGBoost baseline (351s)', ha='right', color='#ef5350', fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Dev MAE (seconds)', fontsize=12)
    ax.set_title('ETA Prediction: Improvement Trajectory', fontsize=14, fontweight='bold')
    ax.set_ylim(200, 620)

    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(ASSETS / "progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  progression.png")


def plot_learning_curves():
    """NN training curves across versions."""
    epochs = [1, 2, 3, 4, 5, 6, 7, 8]
    v1 = [858.7, 414.5, 275.2, 272.1, 273.9, 272.4, 274.6, 274.6]
    v2 = [923.5, 394.5, 270.8, 270.3, 266.2, 269.0, 270.9, 270.4]
    v3 = [300.8, 279.7, 272.3, 268.7, 264.5, 268.2, 269.3, 271.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, v1, 'o-', label='v1: L1 loss, 19 features (272s)', color='#42a5f5', linewidth=2)
    ax1.plot(epochs, v2, 's-', label='v2: +Huber, +temporal stats (266s)', color='#66bb6a', linewidth=2)
    ax1.plot(epochs, v3, '^-', label='v3: +residual, +interactions (264s)', color='#ab47bc', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dev MAE (seconds)')
    ax1.set_title('Training Convergence', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(250, 950)

    ax2.plot(epochs[2:], v1[2:], 'o-', label='v1 best: 272s (epoch 4)', color='#42a5f5', linewidth=2)
    ax2.plot(epochs[2:], v2[2:], 's-', label='v2 best: 266s (epoch 5)', color='#66bb6a', linewidth=2)
    ax2.plot(epochs[2:], v3[2:], '^-', label='v3 best: 264s (epoch 5)', color='#ab47bc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dev MAE (seconds)')
    ax2.set_title('Zoomed: Diminishing Returns', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(260, 280)

    plt.suptitle('Neural Net Learning Curves (Kaggle T4 GPU, 37M rows)', fontsize=13)
    plt.tight_layout()
    plt.savefig(ASSETS / "learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  learning_curves.png")


def plot_rare_pair_diagnostic():
    """The diagnostic that motivated the ensemble."""
    categories = ['10k+\n(875k rows)', '1k-10k\n(286k)', '101-1k\n(47k)', '11-100\n(17k)', '1-10\n(5k)', 'Unseen\n(492)']
    nn_maes = [251, 291, 356, 566, 747, 926]
    nn_bias = [-42, -8, -23, -112, -243, -436]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bar_colors = ['#66bb6a', '#66bb6a', '#ffb74d', '#ef5350', '#ef5350', '#b71c1c']
    bars = ax1.bar(categories, nn_maes, color=bar_colors, edgecolor='white')
    for bar, m in zip(bars, nn_maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f"{m}s",
                ha='center', fontweight='bold', fontsize=10)
    ax1.set_ylabel('MAE (seconds)', fontsize=12)
    ax1.set_title('NN Error by Zone-Pair Frequency', fontweight='bold')
    ax1.set_ylim(0, 1050)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    bias_colors = ['#42a5f5' if b > -50 else '#ef5350' for b in nn_bias]
    ax2.bar(categories, nn_bias, color=bias_colors, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.5)
    for i, b in enumerate(nn_bias):
        ax2.text(i, b - 30, f"{b:+d}s", ha='center', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Bias (seconds)', fontsize=12)
    ax2.set_title('NN Underprediction Bias by Frequency', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('The Rare-Pair Problem: Why the NN Hit a Ceiling', fontsize=13)
    plt.tight_layout()
    plt.savefig(ASSETS / "rare_pair_diagnostic.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  rare_pair_diagnostic.png")


def plot_failures():
    """Experiments that didn't work."""
    failures = [
        ("Hash buckets 16k->8k",  +13),
        ("Remove month features", +13),
        ("Log-target + Huber",    +1),
        ("LGBM on 37M rows",     +4),
        ("Pred rescaling",        -1),
    ]

    names_f = [f[0] for f in failures]
    deltas = [f[1] for f in failures]
    notes = [
        "Hash embeddings are critical (47% of params)",
        "Training needs seasonal signal",
        "Huber(300) in log-space = pure MSE",
        "Outliers dilute tree splits",
        "Only -0.8s, not worth overfitting risk",
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    bar_colors_f = ['#ef5350' if d > 0 else '#bdbdbd' for d in deltas]
    bars = ax.barh(names_f, deltas, color=bar_colors_f, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.5)

    for i, (d, note) in enumerate(zip(deltas, notes)):
        x = max(d, 0) + 0.5
        label = f"+{d}s  {note}" if d > 0 else f"{d}s  {note}"
        ax.text(x, i, label, va='center', fontsize=9, color='#555')

    ax.set_xlabel('MAE Change (seconds)', fontsize=11)
    ax.set_title('Experiments That Didn\'t Help (red = regression)', fontsize=13, fontweight='bold')
    ax.set_xlim(-3, 22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / "failures.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  failures.png")


def plot_ensemble_weights():
    """Ensemble weight optimization curve."""
    # Actual grid search results (NN weight, best LGBM+FT split)
    nn_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    maes = [262, 258, 256, 255, 254, 253, 253, 254, 255, 257, 261]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nn_weights, maes, 'o-', color='#ab47bc', linewidth=2.5, markersize=8)
    ax.fill_between(nn_weights, maes, 265, alpha=0.1, color='#ab47bc')

    # Annotate best
    best_idx = maes.index(min(maes))
    ax.annotate(f'Best: {min(maes)}s\n(NN=0.5, LGBM=0.3, FT=0.2)',
                xy=(nn_weights[best_idx], min(maes)),
                xytext=(0.65, 256), fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ab47bc'),
                color='#ab47bc')

    # Annotate single models
    ax.annotate('LGBM only\n(262s)', xy=(0, 262), fontsize=9, color='#66bb6a',
                xytext=(0.05, 264.5))
    ax.annotate('NN only\n(261s)', xy=(1, 261), fontsize=9, color='#42a5f5',
                xytext=(0.85, 263.5))

    ax.set_xlabel('NN Weight in Ensemble', fontsize=12)
    ax.set_ylabel('Dev MAE (seconds)', fontsize=12)
    ax.set_title('Ensemble Weight Optimization (1.23M dev rows)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(251, 265)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(ASSETS / "ensemble_weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ensemble_weights.png")


def main():
    print("Generating README charts...")
    plot_progression()
    plot_learning_curves()
    plot_rare_pair_diagnostic()
    plot_failures()
    plot_ensemble_weights()
    print(f"Done. Charts saved to {ASSETS}/")


if __name__ == "__main__":
    main()
