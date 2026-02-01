import os
import sys
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import argparse
import csv
import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import exp_cfg as cfg
from models import build_model
from utils_audio import (
    load_audio,
    split_fixed,
    mel_spectrogram,
    normalize_mel,
    slice_clip,
    uniform_starts,
)


def load_label_map(path="models/label_map.json"):
    m = json.load(open(path, "r", encoding="utf-8"))
    genres = [m[str(i)] for i in range(len(m))]
    return genres


def load_split_items(split_json, split_name):
    items = json.load(open(split_json, "r", encoding="utf-8"))
    return [it for it in items if it["split"] == split_name]


@torch.no_grad()
def predict_one_file_meanprob(model, device, path, sr=22050, clip_seconds=3.0, hop_seconds=1.5):
    """
    多切片：softmax 后按概率平均融合
    返回 proba: np.ndarray [C]
    """
    y = load_audio(path, sr=sr)
    clips = split_fixed(y, sr, clip_seconds=clip_seconds, hop_seconds=hop_seconds)

    probs = []
    for seg in clips:
        m = normalize_mel(mel_spectrogram(seg, sr=sr))
        x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,128,T]
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        probs.append(p)

    if len(probs) == 0:
        return None
    return np.mean(probs, axis=0)


@torch.no_grad()
def predict_one_file_fixed(
    model,
    device,
    path,
    sr=22050,
    clip_seconds=5.0,
    num_clips=6,
    trim_ratio=0.0,
    trim_seconds=0.0,
):
    y = load_audio(path, sr=sr)
    clip_len = int(sr * clip_seconds)
    starts = uniform_starts(
        len(y), sr, clip_seconds, num_clips,
        trim_ratio=trim_ratio, trim_seconds=trim_seconds
    )

    probs = []
    for s in starts:
        seg = slice_clip(y, s, clip_len)
        m = normalize_mel(mel_spectrogram(seg, sr=sr))
        x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,128,T]
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        probs.append(p)

    if len(probs) == 0:
        return None
    return np.mean(probs, axis=0)


def save_confusion_matrix(cm, labels, out_path):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, aspect="auto", origin="upper")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()

    # 写数字（非零）
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v != 0:
                plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def upsert_summary(summary_path, exp_name, test_acc, macro_f1):
    """
    在 runs/summary.csv 里找到 exp_name 那行，写入 test_acc / macro_f1。
    若不存在该 exp_name 行，则追加一行（仅包含这三列）。
    """
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["exp_name", "test_acc", "macro_f1"])
            w.writeheader()
            w.writerow({"exp_name": exp_name, "test_acc": f"{test_acc:.6f}", "macro_f1": f"{macro_f1:.6f}"})
        return

    # 读取全部
    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = list(r.fieldnames) if r.fieldnames else []

    # 确保字段存在
    for fn in ["test_acc", "macro_f1"]:
        if fn not in fieldnames:
            fieldnames.append(fn)

    found = False
    for row in rows:
        if row.get("exp_name") == exp_name:
            row["test_acc"] = f"{test_acc:.6f}"
            row["macro_f1"] = f"{macro_f1:.6f}"
            found = True
            break

    if not found:
        # 追加一行，其他列留空
        new_row = {k: "" for k in fieldnames}
        new_row["exp_name"] = exp_name
        new_row["test_acc"] = f"{test_acc:.6f}"
        new_row["macro_f1"] = f"{macro_f1:.6f}"
        rows.append(new_row)

    # 写回
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--exp", type=str, required=True, help="例如 E0_baseline / E1_specaug")
    # ap.add_argument("--split", type=str, default="test", help="test | val")
    # ap.add_argument("--k", type=int, default=0, help="num clips for uniform sampling; 0 uses cfg default")
    # args = ap.parse_args()

    cfg.EXP_NAME = cfg.EXP_NAME
    out_dir = os.path.join(cfg.EXPERIMENTS_DIR, cfg.EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 读取训练时保存的 exp_cfg.json（确保评估参数与训练一致）
    exp_cfg_path = os.path.join(out_dir, "exp_cfg.json")
    if os.path.exists(exp_cfg_path):
        exp_cfg = json.load(open(exp_cfg_path, "r", encoding="utf-8"))
        for k, v in exp_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    genres = load_label_map(cfg.LABEL_MAP_PATH)
    cfg.NUM_CLASSES = len(genres)

    # 模型：按实验配置构建
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.MODEL_TYPE, cfg.NUM_CLASSES).to(device).eval()

    # 权重路径：runs/<exp>/best.pt
    ckpt_path = os.path.join(out_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. 请先训练该实验。")

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    # 数据
    items = load_split_items(cfg.SPLIT_JSON, cfg.EVAL_SPLIT)
    if len(items) == 0:
        raise RuntimeError(f"No items found for split={cfg.EVAL_SPLIT} in {cfg.SPLIT_JSON}")

    y_true, y_pred = [], []
    skipped = 0

    for it in items:
        fp = it["path"]
        label = int(it["label"])
        try:
            num_clips = args.k if args.k > 0 else cfg.EVAL_NUM_CLIPS
            if num_clips > 0:
                proba = predict_one_file_fixed(
                    model, device, fp,
                    sr=cfg.SR,
                    clip_seconds=cfg.EVAL_CLIP_SECONDS,
                    num_clips=num_clips,
                    trim_ratio=cfg.EVAL_TRIM_RATIO,
                    trim_seconds=cfg.EVAL_TRIM_SECONDS,
                )
            else:
                proba = predict_one_file_meanprob(
                    model, device, fp,
                    sr=cfg.SR,
                    clip_seconds=cfg.EVAL_CLIP_SECONDS,
                    hop_seconds=cfg.HOP_SECONDS
                )
            if proba is None:
                skipped += 1
                continue
            pred = int(np.argmax(proba))
            y_true.append(label)
            y_pred.append(pred)
        except Exception:
            skipped += 1

    report = classification_report(y_true, y_pred, target_names=genres, digits=4)
    acc = (np.array(y_true) == np.array(y_pred)).mean() if len(y_true) else 0.0

    # macro f1：从 report 文本里取不够稳，这里用 sklearn 的输出更靠谱
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0

    print(report)
    print(f"{args.split}_acc={acc:.4f}  macro_f1={macro_f1:.4f}  skipped={skipped}  total={len(items)}")

    # 保存报告
    report_path = os.path.join(out_dir, f"{args.split}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\n\n{args.split}_acc={acc:.6f}\nmacro_f1={macro_f1:.6f}\nskipped={skipped}\n")
    print("saved:", report_path)

    # 混淆矩阵（仅对 test/val 都可）
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(genres))))
    cm_path = os.path.join(out_dir, f"{args.split}_confusion_matrix.png")
    save_confusion_matrix(cm, genres, cm_path)
    print("saved:", cm_path)

    # 更新 summary.csv（只对 test 写入更常见；这里 split=test 才写）
    summary_path = os.path.join(cfg.OUT_ROOT, "summary.csv")
    if args.split == "test":
        upsert_summary(summary_path, cfg.EXP_NAME, acc, macro_f1)
        print("updated:", summary_path)


if __name__ == "__main__":
    main()
