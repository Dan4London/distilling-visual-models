"""CLI: subset | label | train | eval | report.

Typical order (see README):
    subset -> label -> train (x2: distilled + oracle) -> eval -> report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data as data_mod
from .config import load_config
from .evaluate import (
    pseudo_label_accuracy,
    render_report,
    student_top1,
    teacher_test_top1,
)
from .student import count_params, load_student, train_student


def _resolved(cfg: dict, data_root: str) -> list[str]:
    return data_mod.resolve_classes(data_root, cfg["classes"])


def _merge_results(out_path: str, patch: dict) -> dict:
    path = Path(out_path)
    results = {}
    if path.exists():
        results = json.loads(path.read_text())
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(results.get(key), dict):
            results[key].update(value)
        else:
            results[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return results


def cmd_subset(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    classes = _resolved(cfg, args.data_root)
    train_items = data_mod.list_items(args.data_root, "train", classes,
                                      cfg["max_train_per_class"])
    test_items = data_mod.list_items(args.data_root, "test", classes,
                                     cfg["max_test_per_class"])
    print(f"Classes: {len(classes)}")
    print(f"Train images: {len(train_items)}")
    print(f"Test images:  {len(test_items)}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from .teacher import QwenTeacher, label_items  # heavy import, defer

    cfg = load_config(args.config)
    classes = _resolved(cfg, args.data_root)
    train_items = data_mod.list_items(args.data_root, "train", classes,
                                      cfg["max_train_per_class"])
    tcfg = cfg["teacher"]
    teacher = QwenTeacher(tcfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
                          max_new_tokens=int(tcfg.get("max_new_tokens", 24)),
                          device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = label_items(teacher, train_items, classes, args.out,
                          log_every=int(tcfg.get("batch_log_every", 25)))
    print(f"Pseudo-labelled {summary['labelled']}/{summary['total']} "
          f"({summary['dropped']} unparseable) -> {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    classes = _resolved(cfg, args.data_root)
    if args.source == "teacher":
        rows = data_mod.read_pseudo_labels(args.labels)
        items = [(r["image_path"], r["pseudo_class"]) for r in rows
                 if r["pseudo_class"] in set(classes)]
        print(f"Training distilled student on {len(items)} pseudo-labelled images.")
    else:  # true
        items = data_mod.list_items(args.data_root, "train", classes,
                                    cfg["max_train_per_class"])
        print(f"Training oracle student on {len(items)} true-labelled images.")
    info = train_student(items, classes, cfg["student"], args.out, args.device)
    print(f"Saved {args.source} student -> {args.out}  "
          f"({info['params'] / 1e6:.1f}M params, "
          f"final loss {info['final_train_loss']:.3f})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    classes = _resolved(cfg, args.data_root)
    test_items = data_mod.list_items(args.data_root, "test", classes,
                                     cfg["max_test_per_class"])
    scfg = cfg["student"]
    img_size, bs = int(scfg.get("img_size", 224)), int(scfg.get("batch_size", 64))

    patch: dict = {"config": {"classes": len(classes), "n_test": len(test_items)}}

    acc = student_top1(args.student, test_items, img_size, bs, args.device)
    model, _ = load_student(args.student, args.device)
    patch["student"] = {"test_top1": acc, "params": count_params(model)}
    print(f"Student top-1: {100 * acc:.1f}%")

    if args.oracle:
        oacc = student_top1(args.oracle, test_items, img_size, bs, args.device)
        omodel, _ = load_student(args.oracle, args.device)
        patch["oracle"] = {"test_top1": oacc, "params": count_params(omodel)}
        print(f"Oracle top-1:  {100 * oacc:.1f}%")

    if args.labels:
        patch["pseudo_labels"] = pseudo_label_accuracy(args.labels)
        print(f"Pseudo-label accuracy (train): "
              f"{100 * patch['pseudo_labels']['pseudo_vs_true_acc']:.1f}%")

    if args.eval_teacher:
        from .teacher import QwenTeacher  # heavy import, defer

        tcfg = cfg["teacher"]
        teacher = QwenTeacher(tcfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
                              max_new_tokens=int(tcfg.get("max_new_tokens", 24)),
                              device=args.device)
        tacc = teacher_test_top1(teacher, test_items, classes)
        patch["teacher"] = {"model_id": tcfg.get("model_id", "Qwen2-VL"),
                            "test_top1": tacc}
        print(f"Teacher top-1: {100 * tacc:.1f}%")

    _merge_results(args.out, patch)
    print(f"Results -> {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text())
    print(render_report(results))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="vlm_food_distill", description=__doc__)
    p.add_argument("--device", default="auto",
                   help="auto | cuda | mps | cpu (default: auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict()
    s = sub.add_parser("subset", help="Download Food-101 and report subset sizes.")
    s.add_argument("--config", required=True)
    s.add_argument("--data-root", default="./data")
    s.set_defaults(func=cmd_subset)

    s = sub.add_parser("label", help="Teacher pseudo-labels the train images.")
    s.add_argument("--config", required=True)
    s.add_argument("--data-root", default="./data")
    s.add_argument("--out", default="runs/pseudo_labels.csv")
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("train", help="Train a student (teacher or true labels).")
    s.add_argument("--config", required=True)
    s.add_argument("--data-root", default="./data")
    s.add_argument("--source", choices=("teacher", "true"), default="teacher")
    s.add_argument("--labels", default="runs/pseudo_labels.csv",
                   help="Pseudo-label CSV (used when --source teacher).")
    s.add_argument("--out", default="runs/student.pt")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="Evaluate student/oracle/teacher on test.")
    s.add_argument("--config", required=True)
    s.add_argument("--data-root", default="./data")
    s.add_argument("--student", required=True)
    s.add_argument("--oracle", default=None)
    s.add_argument("--labels", default=None,
                   help="Pseudo-label CSV for the teacher-quality metric.")
    s.add_argument("--eval-teacher", action="store_true",
                   help="Also run the VLM teacher zero-shot on the test split.")
    s.add_argument("--out", default="runs/results.json")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("report", help="Print the Markdown results table.")
    s.add_argument("--results", default="runs/results.json")
    s.set_defaults(func=cmd_report)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
