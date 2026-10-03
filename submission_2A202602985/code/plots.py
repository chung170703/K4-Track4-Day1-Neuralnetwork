"""plots.py — ảnh biểu đồ là sản phẩm nộp (README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

OPT_LABEL = {"sgd": "SGD", "sgd_momentum": "SGD+mom", "adam": "Adam", "adamw": "AdamW"}


def _cfg_text(cfg: dict) -> str:
    clip = "none" if cfg.get("clip_norm") is None else cfg["clip_norm"]
    return (f"{cfg['loss'].upper()} | {OPT_LABEL[cfg['optimizer']]} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} | "
            f"batch={cfg['batch']} ep={cfg['epochs']} | hidden={'-'.join(map(str, cfg['hidden']))} "
            f"drop={cfg['dropout']:g} clip={clip} {cfg['precision']} init={cfg['init']} seed={cfg['seed']}")


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục; train đo ở eval mode)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip): trung bình và lớn nhất
    Đường thẳng đứng đánh dấu best_epoch (val loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = np.array(h["epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    best = s.get("best_epoch")

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax.plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel(f"loss ({cfg['loss'].upper()})"); ax.set_title("Loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax.axhline(0.4876, color="gray", ls=":", lw=1, label="đoán lớp đa số (0.4876)")
    ax.set_xlabel("epoch"); ax.set_ylabel("điểm"); ax.set_title("Val accuracy / macro-F1")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="grad_norm trung bình")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="grad_norm lớn nhất")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls=":", lw=1, label=f"ngưỡng clip c={cfg['clip_norm']:g}")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖₂ (trước khi clip)"); ax.set_title("Chuẩn gradient")

    for ax in axes:
        if best:
            ax.axvline(best, color="green", ls="--", lw=1, alpha=0.6, label=f"best epoch = {best}")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    if s.get("diverged"):
        fig.text(0.5, 0.02, "DIVERGED (loss NaN/inf): huấn luyện dừng sớm", ha="center", color="red")
    fig.suptitle(f"{cfg['exp_id']}\n{_cfg_text(cfg)}", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (history[metric], ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều
    thí nghiệm trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        h = r["history"]
        ax.plot(h["epoch"], h[metric], "o-", ms=3, lw=1.4, label=r["cfg"]["exp_id"])
    ax.set_xlabel("epoch"); ax.set_ylabel(metric)
    if metric == "grad_norm":
        ax.set_yscale("log")
    ax.set_title(title or metric)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
