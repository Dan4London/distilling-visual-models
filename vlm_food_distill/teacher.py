"""Qwen2-VL teacher: turn images into class pseudo-labels.

This is "output distillation" — the VLM makes a hard class decision per image and
the student later learns from those. A VLM doesn't expose calibrated class
logits, so we use its parsed text answer rather than soft targets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from tqdm import tqdm

from .config import canonicalize, humanize


def _match_class(reply: str, classes: list[str]) -> str | None:
    """Map a free-text teacher reply to one of ``classes`` (or ``None``)."""
    canon = canonicalize(reply)
    if not canon:
        return None
    class_set = set(classes)
    if canon in class_set:
        return canon
    # Substring either way (e.g. reply "a slice of pizza" -> "pizza").
    for c in classes:
        if c in canon or canon in c:
            return c
    # Fall back to best token overlap.
    reply_tokens = set(canon.split("_"))
    best, best_score = None, 0
    for c in classes:
        score = len(reply_tokens & set(c.split("_")))
        if score > best_score:
            best, best_score = c, score
    return best if best_score > 0 else None


def _build_prompt(classes: list[str]) -> str:
    options = ", ".join(humanize(c) for c in classes)
    return (
        "You are a food image classifier. Look at the image and choose the "
        "single best matching food category from this list:\n"
        f"{options}\n"
        "Reply with ONLY the category name exactly as written above, and "
        "nothing else."
    )


class QwenTeacher:
    """Lazy Qwen2-VL wrapper that classifies an image into a fixed class list."""

    def __init__(self, model_id: str, max_new_tokens: int = 24,
                 device: str = "auto") -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype,
            device_map=self.device if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def classify(self, image_path: str, classes: list[str]) -> tuple[str | None, str]:
        """Return ``(matched_class_or_None, raw_reply)`` for one image."""
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _build_prompt(classes)},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True,
                                 return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        reply = self._processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0].strip()
        return _match_class(reply, classes), reply


def label_items(
    teacher: QwenTeacher,
    items: list[tuple[str, str]],
    classes: list[str],
    out_csv: str | Path,
    log_every: int = 25,
) -> dict[str, int]:
    """Pseudo-label every item and write ``image_path,true_class,pseudo_class,raw``.

    Returns simple counts (labelled / dropped) for the run summary.
    """
    labelled = dropped = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "true_class", "pseudo_class", "raw_reply"])
        for i, (path, true_class) in enumerate(tqdm(items, desc="teacher")):
            pseudo, raw = teacher.classify(path, classes)
            if pseudo is None:
                dropped += 1
            else:
                labelled += 1
            writer.writerow([path, true_class, pseudo or "", raw])
            if log_every and (i + 1) % log_every == 0:
                fh.flush()
    return {"labelled": labelled, "dropped": dropped, "total": len(items)}
