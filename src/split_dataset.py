# split_dataset.py
import json
import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import exp_cfg as cfg

# 引入 utils_audio 中的新函数
from src.utils_audio import (
    get_duration_ffprobe,
    load_audio_ffmpeg,
    mel_spectrogram,
    normalize_mel,
    spec_augment,
    is_silent,
    pad_or_trim_1d,
    get_clip_starts  # <--- 使用统一的切片逻辑
)

class SplitDataset(Dataset):
    def __init__(self,
                 split_json_path=cfg.SPLIT_JSON,
                 split="train",
                 sr=cfg.SR,
                 clip_seconds=cfg.CLIP_SECONDS,
                 hop_seconds=cfg.HOP_SECONDS, # 这个参数其实在 get_clip_starts 里不再强依赖，但为了兼容保留
                 clips_per_item=1,
                 covering_clips=True,
                 trim_ratio=0.0,
                 trim_seconds=0.0,
                 unique_clips=True,
                 random_clip=False,
                 max_retry=20,
                 use_specaug=False,
                 freq_mask_param=cfg.FREQ_MASK_PARAM,
                 time_mask_param=cfg.TIME_MASK_PARAM,
                 num_masks=cfg.NUM_MASKS,
                 silence_threshold_db=None,
    ):
        # ... (参数赋值保持不变) ...
        self.sr = int(sr)
        self.split = split
        self.clip_seconds = float(clip_seconds)
        self.clips_per_item = max(1, int(clips_per_item))
        self.covering_clips = bool(covering_clips)
        self.trim_ratio = float(trim_ratio)
        self.trim_seconds = float(trim_seconds)
        self.unique_clips = bool(unique_clips)
        self.random_clip = bool(random_clip)
        self.max_retry = int(max_retry)
        
        self.use_specaug = bool(use_specaug)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_masks = int(num_masks)
        self.silence_threshold_db = float(silence_threshold_db) if silence_threshold_db is not None else None
        
        # ... (加载 JSON 逻辑保持不变) ...
        with open(split_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.items = [it for it in items if it["split"] == split]
        if len(self.items) == 0:
            raise RuntimeError(f"No items found for split='{split}' in {split_json_path}")
        self._dur_cache = {}

    # ... (__len__, _get_track_seconds 保持不变) ...
    def __len__(self):
        return len(self.items)

    def _get_track_seconds(self, fp: str) -> float:
        if fp in self._dur_cache: return self._dur_cache[fp]
        sec = get_duration_ffprobe(fp)
        sec = float(sec) if sec and sec > 0 else 0.0
        self._dur_cache[fp] = sec
        return sec

    def _decode_segment(self, fp, start_sample):
        # ... (保持不变，FFmpeg Seek 逻辑) ...
        offset_sec = float(start_sample) / float(self.sr)
        seg = load_audio_ffmpeg(fp, sr=self.sr, mono=True, offset=offset_sec, duration=self.clip_seconds)
        return pad_or_trim_1d(seg, int(self.sr * self.clip_seconds))

    def _segments_to_tensor(self, segments):
        # ... (保持不变) ...
        xs = []
        for seg in segments:
            # 注意：mel_spectrogram 需要传参数
            m = normalize_mel(mel_spectrogram(
                seg, sr=self.sr, n_mels=cfg.N_MELS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
            ))
            if self.use_specaug and self.split == "train":
                m = spec_augment(m, self.freq_mask_param, self.time_mask_param, self.num_masks)
            xs.append(torch.tensor(m).unsqueeze(0))
        if len(xs) == 1: return xs[0]
        return torch.stack(xs, dim=0)

    def __getitem__(self, idx):
        for _ in range(self.max_retry * 2):
            it = self.items[idx]
            fp, label = it["path"], it["label"]
            try:
                track_seconds = self._get_track_seconds(fp)
                y_len = int(track_seconds * self.sr)

                # 【修改点】调用 utils_audio 中的 get_clip_starts
                starts = get_clip_starts(
                    y_len=y_len,
                    sr=self.sr,
                    clip_seconds=self.clip_seconds,
                    num_clips=self.clips_per_item,
                    mode="uniform" if not self.random_clip else "random",
                    allow_overlap=self.covering_clips,
                    trim_ratio=self.trim_ratio,
                    trim_seconds=self.trim_seconds,
                    seed=cfg.RANDOM_SEED
                )

                segments = [self._decode_segment(fp, s) for s in starts]

                # 静音过滤
                if self.silence_threshold_db is not None:
                    valid = [s for s in segments if not is_silent(s, self.silence_threshold_db)]
                    if len(valid) < len(segments): raise ValueError("Silent clips")
                    segments = valid
                
                x = self._segments_to_tensor(segments)
                y_t = torch.tensor(int(label)).long()
                return x, y_t

            except Exception as e:
                print(f"[Warning] Skipping {fp}: {e}")
                idx = random.randrange(len(self.items))
        
        raise RuntimeError("Too many bad files.")