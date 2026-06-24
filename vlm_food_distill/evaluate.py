"""Evaluation: student / oracle top-1 on the test split, teacher quality, report."""

from __future__ import annotations

import torch
from tqdm import tqdm

from .data import make_loader, read_pseudo_labels
from .student import load_student, pick_device


@torch.no_grad()
def student_top1(
    student_path: str,
    test_items: list[tuple[str, str]],
    img_size: int,
    batch_size: int,
    device: str = "auto",
) -> float:
    """Top-1 accuracy of a saved student on the (true-labelled) test items."""
    device = pick_device(device)
    model, classes = load_student(student_path, device)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    loader = make_loader(test_items, class_to_idx, img_size, batch_size,
                         train=False)
    correct = total = 0
    for images, targets in tqdm(loader, desc="eval student"):
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / max(1, total)


def pseudo_label_accuracy(labels_csv: str) -> dict[str, float]:
    """How often the teacher's pseudo-label matched the true label (train set).

    This is the *teacher quality* on the training images — the ceiling on what
    a student trained purely on these labels could learn.
    """
    rows = read_pseudo_labels(labels_csv)
    matched = sum(1 for r in rows if r["pseudo_class"] and
                  r["pseudo_class"] == r["true_class"])
    labelled = sum(1 for r in rows if r["pseudo_class"])
    total = len(rows)
    return {
        "pseudo_vs_true_acc": matched / max(1, total),
        "coverage": labelled / max(1, total),
        "n_total": float(total),
    }


@torch.no_grad()
def teacher_test_top1(
    teacher,  # QwenTeacher
    test_items: list[tuple[str, str]],
    classes: list[str],
) -> float:
    """Zero-shot teacher accuracy on the test split (the expensive headline)."""
    correct = total = 0
    for path, true_class in tqdm(test_items, desc="eval teacher"):
        pred, _ = teacher.classify(path, classes)
        correct += int(pred == true_class)
        total += 1
    return correct / max(1, total)


def render_report(results: dict) -> str:
    """Pretty Markdown table from a merged results dict."""
    def pct(x):
        return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "—"

    lines = ["| Model | Params | Top-1 (test) |", "|---|---|---|"]
    teacher = results.get("teacher", {})
    lines.append(f"| Teacher ({teacher.get('model_id', 'Qwen2-VL')}, zero-shot) "
                 f"| billions | {pct(teacher.get('test_top1'))} |")
    student = results.get("student", {})
    lines.append(f"| **Student (distilled)** | {student.get('params', '—')} "
                 f"| {pct(student.get('test_top1'))} |")
    oracle = results.get("oracle", {})
    if oracle:
        lines.append(f"| Oracle student (true labels) | {oracle.get('params', '—')} "
                     f"| {pct(oracle.get('test_top1'))} |")
    pl = results.get("pseudo_labels", {})
    if pl:
        lines.append(f"| _Pseudo-label accuracy (teacher vs true, train)_ | — "
                     f"| {pct(pl.get('pseudo_vs_true_acc'))} |")
    return "\n".join(lines)
