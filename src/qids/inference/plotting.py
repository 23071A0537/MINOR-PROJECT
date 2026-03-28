from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["NORMALL", "DoSD", "PROBE", "EXPLOIT", "MALWARE"]


def plot_confusion_matrix(cm: List[List[int]], out_path: Path) -> None:
    matrix = np.array(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Hybrid Ensemble Confusion Matrix")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_f1_scores(report: Dict[str, Any], out_path: Path) -> None:
    f1_scores = [report[name]["f1-score"] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CLASS_NAMES, f1_scores, color="#2a6fdb")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-Class F1 Scores")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
