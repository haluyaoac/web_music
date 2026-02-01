# src/infer.py
import os
# 关键：避免你之前遇到的 torch 导入/compile 等问题（对 Streamlit 也很重要）
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import json
import numpy as np
import torch

from . import exp_cfg as cfg
from models import build_model
from src.utils_audio import (
    load_audio,
    split_fixed,
    mel_spectrogram,
    normalize_mel,
    slice_clip,
    uniform_starts,
)


def load_label_map(map_path=cfg.LABEL_MAP_PATH):
    with open(map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # label_map keys may be strings
    genres = [label_map[str(i)] for i in range(len(label_map))]
    return genres


def load_model(model_path="models/cnn_melspec.pth", n_classes=5, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.MODEL_TYPE, n_classes=n_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict_proba_file(
    path: str,
    model_path="models/cnn_melspec.pth",
    map_path="models/label_map.json",
    sr=cfg.SR,
    clip_seconds=cfg.INFER_CLIP_SECONDS,
    hop_seconds=cfg.HOP_SECONDS,
    num_clips=cfg.INFER_NUM_CLIPS,
    trim_ratio=cfg.INFER_TRIM_RATIO,
    trim_seconds=cfg.INFER_TRIM_SECONDS,
):
    """
    返回：genres(list), mean_proba(np.ndarray shape [C]), clip_count(int)
    做法：整段音频切片 -> 每片算 log-mel -> softmax -> 平均融合
    """
    genres = load_label_map(map_path)
    model, device = load_model(model_path, n_classes=len(genres))

    y = load_audio(path, sr=sr)
    if num_clips and num_clips > 0:
        clip_len = int(sr * clip_seconds)
        starts = uniform_starts(
            len(y), sr, clip_seconds, num_clips,
            trim_ratio=trim_ratio, trim_seconds=trim_seconds
        )
        clips = [slice_clip(y, s, clip_len) for s in starts]
    else:
        clips = split_fixed(y, sr, clip_seconds=clip_seconds, hop_seconds=hop_seconds)

    probs = []
    for seg in clips:
        m = mel_spectrogram(seg, sr=sr)
        m = normalize_mel(m)
        x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,128,T]
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        probs.append(p)

    mean_p = np.mean(probs, axis=0) if len(probs) else np.ones(len(genres)) / len(genres)
    return genres, mean_p, len(clips)


def topk_from_proba(genres, proba, k=5):
    idx = np.argsort(proba)[::-1][:k]
    return [(genres[i], float(proba[i])) for i in idx]
