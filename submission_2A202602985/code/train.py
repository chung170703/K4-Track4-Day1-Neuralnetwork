"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train loss đo ở eval mode trên 50 000 mẫu train đầu tiên (cố định cho mọi lần chạy)

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val (Part 2), không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
    scheduler=None,
    notes="",
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có). torch.manual_seed cũng đặt seed cho mps."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py).
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits (eval mode, không xáo)."""
    model.eval()
    out = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).to(torch.int64)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y; F.mse_loss lấy trung bình trên MỌI phần tử (B x 7),
               không có hệ số 1/2 (giống nn.MSELoss). reduction="sum" cộng mọi phần tử.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        onehot = F.one_hot(y, N_CLASSES).to(logits.dtype)
        return F.mse_loss(logits, onehot, reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt), no_grad, luôn FP32.

    Loss cộng dồn theo reduction="sum" rồi chia cho số phần tử (N với CE, N*7 với MSE),
    nên không bị lệch khi lô cuối nhỏ hơn.
    """
    model.eval()
    n = len(X)
    total = 0.0                                   # cộng dồn bằng float Python (mps không có float64)
    cm = torch.zeros(N_CLASSES * N_CLASSES, device=X.device, dtype=torch.int64)
    for i in range(0, n, batch_size):
        logits = model(X[i:i + batch_size])
        yb = y[i:i + batch_size]
        total += compute_loss(logits, yb, loss_name, reduction="sum").item()
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(dim=1), minlength=N_CLASSES * N_CLASSES)
    cm = cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
    denom = n * (N_CLASSES if loss_name == "mse" else 1)
    return dict(loss=total / denom, acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm))


def _sync(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def _mem_mb(device_type: str):
    """cuda: bộ nhớ cực đại thật; mps: bộ nhớ đang cấp phát tại thời điểm đo (không có API 'peak'); cpu: None."""
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device_type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG; khoá thiếu lấy giá trị mặc định)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, ... trên device)

    Trả về dict:
        {"cfg", "history", "summary", "best_state"}
    history (mỗi epoch): epoch, train_loss (eval mode, 50 000 mẫu train cố định), val_loss, val_acc,
        val_macro_f1, grad_norm (trung bình, đo TRƯỚC clip), grad_norm_max, clip_frac (tỉ lệ bước bị cắt),
        epoch_time_s (chỉ phần huấn luyện, không tính đánh giá).
    summary: step0_loss, best_val_loss, best_epoch, final_train_loss, final_val_loss, val_acc, val_macro_f1
        (tại best_epoch), time_per_epoch_s, peak_mem_MB, diverged, act_std_step0 (std kích hoạt sau mỗi nn.Linear
        ở bước 0).
    Không dùng X_eval ở đây: epoch tốt nhất chỉ chọn bằng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    assert cfg["lr"] is not None, "cần đặt cfg['lr']"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev = device.type

    set_seed(cfg["seed"])
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]], "số tham số không khớp EXPECTED_PARAMS"
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], cfg["epochs"] * steps_per_epoch)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"])
    scaler = torch.amp.GradScaler(dev) if cfg["precision"] == "fp16" else None
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # bước 0: std kích hoạt và loss trên val, TRƯỚC bước cập nhật đầu tiên
    act_std0 = activation_stats(model, X_val[:1024])
    step0 = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    n_sub = min(TRAIN_EVAL_SUBSET, len(X_tr))
    X_sub, y_sub = X_tr[:n_sub], y_tr[:n_sub]
    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    best = dict(val_loss=float("inf"), epoch=0, acc=float("nan"), f1=float("nan"), state=None)
    diverged, peak_mem = False, 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(dev)
        t0 = time.time()
        gns, clipped = [], 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], gen):
            if amp_dtype is not None:
                with torch.autocast(dev, dtype=amp_dtype):          # chỉ bọc forward + loss
                    logits = model(xb)
                    loss = compute_loss(logits.float(), yb, cfg["loss"])
            else:
                loss = compute_loss(model(xb), yb, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                           # unscale TRƯỚC khi đo/cắt gradient
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt
            finite = math.isfinite(gn)
            if finite:
                gns.append(gn)
                if cfg["clip_norm"] is not None and gn > cfg["clip_norm"]:
                    clipped += 1
            if scaler is not None:
                scaler.step(optimizer)       # tự bỏ qua bước khi gradient inf/NaN (FP16 tràn số); giảm hệ số s
                scaler.update()
                if not torch.isfinite(loss):
                    diverged = True
            else:
                if not finite or not torch.isfinite(loss):
                    diverged = True
                else:
                    optimizer.step()
            if diverged:
                break
            if scheduler is not None:
                scheduler.step()
        _sync(dev)
        epoch_time = time.time() - t0
        if diverged:
            break

        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        if not math.isfinite(va["loss"]):
            diverged = True
            break
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"]); hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"]); hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.mean(gns)) if gns else float("nan"))
        hist["grad_norm_max"].append(float(np.max(gns)) if gns else float("nan"))
        hist["clip_frac"].append(clipped / max(len(gns), 1))
        hist["epoch_time_s"].append(epoch_time)
        mem = _mem_mb(dev)
        if mem is not None:
            peak_mem = max(peak_mem, mem)
        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=epoch, acc=va["acc"], f1=va["macro_f1"],
                        state={k: v.detach().clone().cpu() for k, v in model.state_dict().items()})

    if best["state"] is None:   # phân kỳ ngay epoch đầu: giữ trọng số hiện tại để các hàm sau không lỗi
        best["state"] = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    n_ep = len(hist["epoch"])
    summary = dict(
        step0_loss=step0,
        best_val_loss=best["val_loss"] if n_ep else float("nan"), best_epoch=best["epoch"],
        final_train_loss=hist["train_loss"][-1] if n_ep else float("nan"),
        final_val_loss=hist["val_loss"][-1] if n_ep else float("nan"),
        val_acc=best["acc"], val_macro_f1=best["f1"],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])) if n_ep else float("nan"),
        peak_mem_MB=peak_mem if dev != "cpu" else None,
        diverged=bool(diverged), act_std_step0=act_std0,
    )
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best["state"])


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`, mỗi row_id đúng một lần."""
    import pandas as pd
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) == len(set(row_id.tolist())), "row_id phải khớp preds và không lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)}).to_csv(path, index=False)


def score_predictions(pred_path: str, repo_root: str, out_json: str | None = None) -> dict:
    """Chạy ĐÚNG scripts/evaluate.py (từ repo_root) trên file dự đoán và trả về dict kết quả
    (accuracy, macro_f1, per_class, confusion_matrix). Nếu out_json là None, ghi tạm rồi xoá."""
    tmp = None
    if out_json is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        out_json = tmp.name
    subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path),
                    "--out", os.path.abspath(out_json)], cwd=repo_root, check=True,
                   stdout=subprocess.DEVNULL)
    with open(out_json) as f:
        res = json.load(f)
    if tmp is not None:
        os.remove(out_json)
    return res


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> torch.Tensor:
    """Dùng cho baseline và cấu hình cuối cùng: nạp best_state (epoch val loss thấp nhất), dự đoán TOÀN BỘ eval
    ở fp32/eval mode và ghi `pred_path`. Trả về nhãn dự đoán. Chấm điểm bằng score_predictions()."""
    device = data["X_eval"].device
    model = MLP(tuple(cfg["hidden"]), cfg["dropout"], cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    return preds
