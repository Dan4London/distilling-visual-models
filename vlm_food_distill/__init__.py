"""VLM-as-annotator knowledge distillation on Food-101.

An open vision-language model (Qwen2-VL) pseudo-labels images; a compact timm
student is trained on those labels only; accuracy is compared to the teacher
and to an oracle student trained on the true labels.
"""

__version__ = "0.1.0"
