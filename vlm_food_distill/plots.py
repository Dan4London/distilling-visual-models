"""Figures: student confusion matrix and teacher-vs-truth example grid.

Matplotlib only (no seaborn/sklearn) to keep the dependency footprint small.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .config import humanize
from .data import make_loader, read_pseudo_labels
from .student import load_student, pick_device


@torch.no_grad()
def _student_predictions(
    student_path: str,
    test_items: list[tuple[str, str]],
    classes: list[str],
    img_size: int,
    batch_size: int,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(true_ids, pred_ids)`` over the test items (order preserved)."""
    device = pick_device(device)
    model, ckpt_classes = load_student(student_path, device)
    class_to_idx = {c: i for i, c in enumerate(ckpt_classes)}
    loader = make_loader(test_items, class_to_idx, img_size, batch_size,
                         train=False)
    trues, preds = [], []
    for images, targets in loader:
        out = model(images.to(device)).argmax(dim=1).cpu().numpy()
        preds.append(out)
        trues.append(targets.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def plot_confusion_matrix(
    student_path: str,
    test_items: list[tuple[str, str]],
    classes: list[str],
    img_size: int,
    batch_size: int,
    out_path: str | Path,
    device: str = "auto",
) -> None:
    """Row-normalised confusion matrix of the distilled student on the test set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trues, preds = _student_predictions(student_path, test_items, classes,
                                        img_size, batch_size, device)
    n = len(classes)
    cm = np.zeros((n, n), dtype=float)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    acc = float((trues == preds).mean())
    labels = [humanize(c) for c in classes]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.5), max(6, n * 0.5)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Distilled student confusion matrix (top-1 {100 * acc:.1f}%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_teacher_examples(
    labels_csv: str,
    out_path: str | Path,
    n_correct: int = 4,
    n_wrong: int = 4,
    seed: int = 0,
) -> None:
    """Grid of example images with the teacher's pseudo-label vs the true label.

    Deliberately shows a mix of correct (green) and incorrect (red) teacher calls
    so the pseudo-label noise is visible rather than cherry-picked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [r for r in read_pseudo_labels(labels_csv) if r["pseudo_class"]]
    correct = [r for r in rows if r["pseudo_class"] == r["true_class"]]
    wrong = [r for r in rows if r["pseudo_class"] != r["true_class"]]
    rng = random.Random(seed)
    rng.shuffle(correct); rng.shuffle(wrong)
    picks = correct[:n_correct] + wrong[:n_wrong]
    if not picks:
        return

    cols = min(4, len(picks))
    nrows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, picks):
        ax.imshow(Image.open(r["image_path"]).convert("RGB"))
        ax.axis("off")
        ok = r["pseudo_class"] == r["true_class"]
        colour = "#1a7f37" if ok else "#cf222e"
        ax.set_title(f"true: {humanize(r['true_class'])}\n"
                     f"teacher: {humanize(r['pseudo_class'])}",
                     fontsize=9, color=colour)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(colour); spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis("on")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle("Teacher pseudo-label vs ground truth "
                 "(green = match, red = mismatch)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
