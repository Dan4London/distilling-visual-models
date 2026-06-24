"""Compact timm student: train on a set of ``(path, class_name)`` items.

Used both for the distilled student (items labelled by the teacher) and the
oracle student (items with their true labels) — identical recipe, only the label
source differs, so the comparison is fair.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import make_loader


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_student(arch: str, num_classes: int) -> nn.Module:
    return timm.create_model(arch, pretrained=True, num_classes=num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_student(
    items: list[tuple[str, str]],
    classes: list[str],
    student_cfg: dict,
    out_path: str | Path,
    device: str = "auto",
) -> dict:
    """Train a student and save ``{state_dict, classes, arch}`` to ``out_path``.

    Returns a small dict with the architecture, parameter count, and final
    training loss for the run report.
    """
    device = pick_device(device)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    torch.manual_seed(int(student_cfg.get("seed", 42)))
    loader = make_loader(
        items, class_to_idx,
        img_size=int(student_cfg.get("img_size", 224)),
        batch_size=int(student_cfg.get("batch_size", 64)),
        train=True,
        num_workers=int(student_cfg.get("num_workers", 2)),
    )

    model = build_student(student_cfg["arch"], len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(student_cfg.get("lr", 8e-4)),
        weight_decay=float(student_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(student_cfg.get("epochs", 12))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(student_cfg.get("label_smoothing", 0.1)))

    model.train()
    last_loss = float("nan")
    for epoch in range(epochs):
        running, seen = 0.0, 0
        pbar = tqdm(loader, desc=f"train {epoch + 1}/{epochs}")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
            pbar.set_postfix(loss=running / max(1, seen))
        scheduler.step()
        last_loss = running / max(1, seen)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes,
         "arch": student_cfg["arch"]},
        out_path,
    )
    return {
        "arch": student_cfg["arch"],
        "params": count_params(model),
        "final_train_loss": last_loss,
        "n_train_items": len(items),
    }


def load_student(path: str | Path, device: str = "auto") -> tuple[nn.Module, list[str]]:
    device = pick_device(device)
    ckpt = torch.load(path, map_location=device)
    model = build_student(ckpt["arch"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]
