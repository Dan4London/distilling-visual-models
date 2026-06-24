"""Food-101 subset selection, item lists, and torch datasets.

We deliberately treat the Food-101 *train* split as if it were unlabelled — the
true labels are kept only to (a) build the class vocabulary, (b) measure
pseudo-label quality, and (c) train the oracle student for comparison.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Food101

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_classes(data_root: str | Path, requested: list[str]) -> list[str]:
    """Return the ordered class vocabulary.

    If ``requested`` is non-empty it is used as-is (after validating against the
    real Food-101 classes); otherwise all 101 classes are used.
    """
    ds = Food101(root=str(data_root), split="test", download=True)
    all_classes = list(ds.classes)
    if not requested:
        return all_classes
    unknown = [c for c in requested if c not in all_classes]
    if unknown:
        raise ValueError(f"Unknown Food-101 classes in config: {unknown}")
    # Preserve the config's order so class ids are stable/readable.
    return list(requested)


def list_items(
    data_root: str | Path,
    split: str,
    classes: list[str],
    max_per_class: int | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(image_path, true_class_name), ...]`` for ``split``.

    Restricted to ``classes`` and optionally capped at ``max_per_class`` images
    per class (deterministic: first-N by the dataset's own ordering).
    """
    ds = Food101(root=str(data_root), split=split, download=True)
    keep = set(classes)
    counts: Counter[str] = Counter()
    items: list[tuple[str, str]] = []
    for path, label_idx in zip(ds._image_files, ds._labels):  # noqa: SLF001
        name = ds.classes[label_idx]
        if name not in keep:
            continue
        if max_per_class is not None and counts[name] >= max_per_class:
            continue
        counts[name] += 1
        items.append((str(path), name))
    return items


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    resize = int(img_size * 1.15)
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class FoodImageDataset(Dataset):
    """Image dataset over ``(path, class_name)`` items.

    The label each sample yields comes from ``class_to_idx[class_name]``, so the
    same item list can be paired with teacher pseudo-labels or true labels.
    """

    def __init__(
        self,
        items: list[tuple[str, str]],
        class_to_idx: dict[str, int],
        img_size: int,
        train: bool,
    ) -> None:
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, name = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.class_to_idx[name]


def make_loader(
    items: list[tuple[str, str]],
    class_to_idx: dict[str, int],
    img_size: int,
    batch_size: int,
    train: bool,
    num_workers: int = 2,
) -> DataLoader:
    ds = FoodImageDataset(items, class_to_idx, img_size, train)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def write_items_csv(path: str | Path, items: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "true_class"])
        w.writerows(items)


def read_pseudo_labels(path: str | Path) -> list[dict[str, str]]:
    """Read the teacher pseudo-label CSV produced by :mod:`teacher`."""
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
