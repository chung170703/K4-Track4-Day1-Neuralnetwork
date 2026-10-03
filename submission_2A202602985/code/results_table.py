"""results_table.py — lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx.

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, không ghi đè)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OPT_NAME = {"sgd": "SGD", "sgd_momentum": "SGD+momentum", "adam": "Adam", "adamw": "AdamW"}
FORMULA_COLS = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"}


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi cfg, history, summary (KHÔNG ghi best_state) ra <results_dir>/<exp_id>.json. Trả về đường dẫn."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / f"{result['cfg']['exp_id']}.json"
    payload = {k: result[k] for k in ("cfg", "history", "summary")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir (trừ eval_*), trả về danh sách dict sắp theo exp_id."""
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        if "cfg" in r and "history" in r:
            out.append(r)
    return sorted(out, key=lambda r: r["cfg"]["exp_id"])


def _num(x):
    """None/NaN -> None để ô Excel để trống thay vì ghi 'nan'."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return x


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    c, s = result["cfg"], result["summary"]
    row = dict(
        exp_id=c["exp_id"], group=c["group"], description=c["description"],
        loss=c["loss"].upper(), optimizer=OPT_NAME[c["optimizer"]], lr=c["lr"],
        weight_decay=c["weight_decay"], batch=c["batch"], epochs=c["epochs"],
        hidden="-".join(str(h) for h in c["hidden"]), dropout=c["dropout"],
        clip_norm="none" if c.get("clip_norm") is None else c["clip_norm"],
        precision=c["precision"], init=c["init"], seed=c["seed"],
        step0_loss=_num(s["step0_loss"]), best_val_loss=_num(s["best_val_loss"]), best_epoch=_num(s["best_epoch"]),
        final_train_loss=_num(s["final_train_loss"]), final_val_loss=_num(s["final_val_loss"]),
        val_acc=_num(s["val_acc"]), val_macro_f1=_num(s["val_macro_f1"]),
        time_per_epoch_s=_num(s["time_per_epoch_s"]), peak_mem_MB=_num(s["peak_mem_MB"]),
        diverged="Y" if s["diverged"] else "N",
        eval_acc=None, eval_macro_f1=None,
        figure_file=f"figures/{c['exp_id']}.png",
        notes=notes or c.get("notes", ""),
    )
    if eval_scores is not None:
        row["eval_acc"] = eval_scores["accuracy"]
        row["eval_macro_f1"] = eval_scores["macro_f1"]
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str,
               seed_ids: list[str] | None = None, summary_notes: dict | None = None) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu (từ dòng 2), rồi lưu thành out_path.

    - Giữ công thức: không dùng data_only; bỏ qua các cột công thức.
    - seed_ids: exp_id các lần chạy baseline, ghi vào cột A của sheet "Seeds" (từ dòng 2, tối đa 5).
    - summary_notes: {group: nhận xét} ghi vào cột H của sheet "Summary".
    """
    import openpyxl
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    col = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    assert len(rows) <= 60, "mẫu chỉ có sẵn 60 dòng công thức"
    for r_idx, row in enumerate(rows, start=2):
        for key, val in row.items():
            if key in FORMULA_COLS or key not in col:
                continue
            ws.cell(r_idx, col[key]).value = val
    for r_idx in range(len(rows) + 2, 62):      # xoá dòng baseline mẫu còn sót phía dưới
        for key in ("exp_id", "group", "description", "loss", "optimizer", "weight_decay", "batch", "epochs",
                    "hidden", "dropout", "clip_norm", "precision", "init", "seed"):
            ws.cell(r_idx, col[key]).value = None
    if seed_ids is not None:
        sw = wb["Seeds"]
        assert len(seed_ids) <= 5, "sheet Seeds có công thức cho 5 dòng"
        for i in range(5):
            sw.cell(2 + i, 1).value = seed_ids[i] if i < len(seed_ids) else None
    if summary_notes:
        sm = wb["Summary"]
        for r in range(2, 12):
            g = sm.cell(r, 1).value
            if g in summary_notes:
                sm.cell(r, 8).value = summary_notes[g]
    wb.save(out_path)
