import json
import random
import torch
from torch.utils.data import Dataset

import exp_cfg as cfg
from utils_audio import (
    # FFmpeg/ffprobe
    get_duration_ffprobe,
    load_audio_ffmpeg,

    # 特征与增强
    mel_spectrogram,
    normalize_mel,
    spec_augment,
    is_silent,

    # 起点采样
    candidate_starts,
    uniform_starts,
)

# 如果你 utils_audio.py 没写 pad_or_trim_1d，这里提供一个兜底
def _pad_or_trim_1d(y, target_len: int):
    if y.shape[0] < target_len:
        import numpy as np
        return np.pad(y, (0, target_len - y.shape[0]))
    return y[:target_len]


class SplitDataset(Dataset):
    """
    Dataset that reads a fixed split JSON and returns audio clips as model inputs.
    FFmpeg segment decoding version: decode only clip(s) with -ss/-t, avoid full-track decode.
    """

    def __init__(
        self,
        split_json_path=cfg.SPLIT_JSON,
        split="train",    # "train" | "val" | "test"
        sr=cfg.SR,
        clip_seconds=cfg.CLIP_SECONDS,
        hop_seconds=cfg.HOP_SECONDS,
        max_retry=20,
        random_clip=True,
        use_specaug=cfg.USE_SPECAUG,
        freq_mask_param=cfg.FREQ_MASK_PARAM,
        time_mask_param=cfg.TIME_MASK_PARAM,
        num_masks=cfg.NUM_MASKS,
        clips_per_item=1,
        unique_clips=True,
        trim_ratio=0.0,
        trim_seconds=0.0,
        deterministic=False,
        deterministic_num_clips=1,
        deterministic_trim_ratio=0.0,
        deterministic_trim_seconds=0.0,
    ):
        self.sr = int(sr)
        self.clip_seconds = float(clip_seconds)
        self.hop_seconds = float(hop_seconds)
        self.max_retry = int(max_retry)
        self.random_clip = bool(random_clip)

        self.use_specaug = bool(use_specaug)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_masks = int(num_masks)

        self.clips_per_item = max(1, int(clips_per_item))
        self.unique_clips = bool(unique_clips)
        self.trim_ratio = float(trim_ratio)
        self.trim_seconds = float(trim_seconds)

        self.deterministic = bool(deterministic)
        self.deterministic_num_clips = max(1, int(deterministic_num_clips))
        self.deterministic_trim_ratio = float(deterministic_trim_ratio)
        self.deterministic_trim_seconds = float(deterministic_trim_seconds)

        with open(split_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.items = [it for it in items if it["split"] == split]
        if len(self.items) == 0:
            raise RuntimeError(f"No items found for split='{split}' in {split_json_path}")

        # 可选：缓存每个文件的时长（首次读取后写入，后续更快）
        self._dur_cache = {}

    def __len__(self):
        return len(self.items)

    def _get_track_seconds(self, fp: str) -> float:
        if fp in self._dur_cache:
            return self._dur_cache[fp]
        sec = get_duration_ffprobe(fp)  # 可能抛异常，外层会 catch
        # 防御：ffprobe 可能返回 0 或负数
        sec = float(sec) if sec and sec > 0 else 0.0
        self._dur_cache[fp] = sec
        return sec

    def _sample_train_starts(self, y_len: int):
        starts = candidate_starts(
            y_len, self.sr, self.clip_seconds, self.hop_seconds,
            trim_ratio=self.trim_ratio, trim_seconds=self.trim_seconds
        )
        if not starts:
            starts = [0]

        if not self.random_clip:
            indices = [0] * self.clips_per_item
        elif self.unique_clips and self.clips_per_item <= len(starts):
            indices = random.sample(range(len(starts)), self.clips_per_item)
        else:
            indices = [random.randrange(len(starts)) for _ in range(self.clips_per_item)]

        return [starts[i] for i in indices]

    def _sample_deterministic_starts(self, y_len: int):
        starts = uniform_starts(
            y_len, self.sr, self.clip_seconds, self.deterministic_num_clips,
            trim_ratio=self.deterministic_trim_ratio, trim_seconds=self.deterministic_trim_seconds
        )
        return starts if starts else [0]

    def _decode_segment(self, fp: str, start_sample: int):
        """
        start_sample: sample index
        return: 1D float32 waveform of fixed length clip_len
        """
        clip_len = int(self.sr * self.clip_seconds)
        offset_sec = float(start_sample) / float(self.sr)

        seg, _ = load_audio_ffmpeg(
            fp,
            sr=self.sr,
            mono=True,
            offset=offset_sec,
            duration=self.clip_seconds,
        )

        # 固定长度（防止结尾不足）
        try:
            from utils_audio import pad_or_trim_1d  # 如果你已经实现了这个
            seg = pad_or_trim_1d(seg, clip_len)
        except Exception:
            seg = _pad_or_trim_1d(seg, clip_len)

        return seg

    def _segments_to_tensor(self, segments):
        xs = []
        for seg in segments:
            m = normalize_mel(mel_spectrogram(seg, sr=self.sr))
            if self.use_specaug and not self.deterministic:
                m = spec_augment(
                    m,
                    freq_mask_param=self.freq_mask_param,
                    time_mask_param=self.time_mask_param,
                    num_masks=self.num_masks,
                )
            xs.append(torch.tensor(m).unsqueeze(0))  # [1, n_mels, time]
        if len(xs) == 1:
            return xs[0]          # [1, n_mels, time]
        return torch.stack(xs, dim=0)  # [K, 1, n_mels, time]

    def __getitem__(self, idx):
        # 增加重试次数，防止文件损坏/静音导致卡死
        for _ in range(self.max_retry * 2):
            it = self.items[idx]
            fp, label = it["path"], it["label"]

            try:
                # 1) 取总时长 -> y_len（samples）
                track_seconds = self._get_track_seconds(fp)
                y_len = int(track_seconds * self.sr)

                # 2) 采样起点（samples）
                if self.deterministic:
                    starts = self._sample_deterministic_starts(y_len)
                else:
                    starts = self._sample_train_starts(y_len)

                # 3) 逐段 FFmpeg 解码
                segments = [self._decode_segment(fp, s) for s in starts]

                # 4) 静音过滤（训练时）
                if not self.deterministic:
                    valid_segments = [s for s in segments if not is_silent(s, threshold_db=-50)]
                    if len(valid_segments) < len(segments):
                        raise ValueError("Silent clips detected")
                    segments = valid_segments

                x = self._segments_to_tensor(segments)
                y_t = torch.tensor(int(label)).long()
                return x, y_t

            except Exception:
                # 解码失败/静音/损坏：随机换一个样本
                idx = random.randrange(len(self.items))

        raise RuntimeError("Too many unreadable/silent audio files.")
