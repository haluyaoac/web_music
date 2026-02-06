# utils_audio.py
import os
import sys
import random
import numpy as np
import librosa
import ffmpeg
# 移除 cfg 依赖，保持工具库纯净，参数由外部传入
# import exp_cfg as cfg 

# ---------------------------
# 1. FFmpeg 基础 I/O
# ---------------------------
def get_duration_ffprobe(path: str) -> float:
    try:
        info = ffmpeg.probe(path)
        return float(info["format"]["duration"])
    except Exception as e:
        # 容错：如果 probe 失败，返回 0，外层处理
        return 0.0

def load_audio_ffmpeg(
    path: str,
    sr: int = 44100,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None
) -> np.ndarray:
    """使用 FFmpeg 解码指定片段"""
    try:
        input_kwargs = {}
        if offset and offset > 0:
            input_kwargs["ss"] = offset
        if duration and duration > 0:
            input_kwargs["t"] = duration
        stream = ffmpeg.input(path, **input_kwargs)
        stream = stream.output(
            'pipe:',
            format='f32le',
            ac=1 if mono else 2,
            ar=sr,
            vn=None,
            loglevel="error" # 仅输出错误，避免静默失败
        )
        out, err = stream.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        # 【关键修改】从 ffmpeg 异常中提取 stderr 并解码，这才是真正的错误原因
        error_log = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "No stderr captured"
        raise RuntimeError(f"ffmpeg decode failed: {path}\n--- FFmpeg Log ---\n{error_log}")
    except Exception as e:
        # 处理其他非 ffmpeg 错误
        raise RuntimeError(f"ffmpeg decode failed: {path}\n{e}")

    y = np.frombuffer(out, dtype=np.float32)
    if not mono:
        y = y.reshape(-1, 2).T
    return y

def load_audio_whole(path: str, sr: int = 44100, mono: bool = True) -> np.ndarray:
    """加载整首音频到内存 (用于推理/评估)"""
    return load_audio_ffmpeg(path, sr=sr, mono=mono, offset=0.0, duration=None)

# ---------------------------
# 2. 核心切片逻辑 (统一算法)
# ---------------------------
def _fill_remainder(base_items, target_count, method="cycle"):
    """辅助函数：填充不足的部分"""
    result = base_items[:]
    remainder = target_count - len(base_items)
    if method == "random":
        result.extend([random.choice(base_items) for _ in range(remainder)])
    elif method == "cycle":
        for i in range(remainder):
            result.append(base_items[i % len(base_items)])
    return result

def get_clip_starts(
    y_len: int,
    sr: int,
    clip_seconds: float,
    num_clips: int,
    mode: str = "uniform",   # "random" | "uniform"
    allow_overlap: bool = True,
    trim_ratio: float = 0.0,
    trim_seconds: float = 0.0,
    seed: int = None
):
    """
    【核心函数】计算切片起点。
    Train/Val/Test/Infer 必须全部调用此函数以保证逻辑一致性。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clip_len = int(sr * clip_seconds)
    
    # 1. Trim
    trim = 0
    if trim_ratio > 0:
        trim = int(y_len * trim_ratio)
    elif trim_seconds > 0:
        trim = int(sr * trim_seconds)

    start_min = trim
    end_max = y_len - clip_len - trim
    valid_region_len = end_max - start_min
    
    # 2. 异常处理：音频太短
    if valid_region_len < 0:
        return [0] * num_clips

    effective_total_len = (y_len - 2 * trim)
    max_possible = effective_total_len // clip_len
    max_possible = max(1, max_possible)
    is_insufficient = num_clips > max_possible
    
    final_starts = []

    if mode == "uniform":
        if not is_insufficient:
            if num_clips == 1:
                final_starts = [start_min + valid_region_len // 2]
            else:
                final_starts = np.linspace(start_min, end_max, num=num_clips).astype(int).tolist()
        else:
            # 资源不足：先生成 max_possible 个均匀切片，再循环补充
            base_starts = np.linspace(start_min, end_max, num=max_possible).astype(int).tolist()
            final_starts = _fill_remainder(base_starts, num_clips, method="cycle")

    elif mode == "random":
        if allow_overlap:
            final_starts = [random.randint(start_min, end_max) for _ in range(num_clips)]
            final_starts.sort()
        else:
            # 不允许重叠 (简略版：资源不足时退化为随机填充)
            if not is_insufficient:
                # 随机间隙算法 (简化实现，保证不重叠)
                slack = effective_total_len - (num_clips * clip_len)
                gaps = np.random.rand(num_clips + 1)
                gaps = (gaps / gaps.sum() * slack).astype(int)
                # 修正误差
                gaps[-1] = slack - gaps[:-1].sum()
                
                curr = start_min + gaps[0]
                for i in range(num_clips):
                    final_starts.append(curr)
                    curr += clip_len + gaps[i+1]
            else:
                # 资源不足：先填满，再随机补
                slack = effective_total_len - (max_possible * clip_len)
                gaps = np.random.rand(max_possible + 1)
                gaps = (gaps / gaps.sum() * slack).astype(int)
                
                base_starts = []
                curr = start_min + gaps[0]
                for i in range(max_possible):
                    base_starts.append(curr)
                    curr += clip_len + gaps[i+1]
                
                final_starts = _fill_remainder(base_starts, num_clips, method="random")

    return sorted(final_starts)

# ---------------------------
# 3. 数组处理
# ---------------------------
def pad_or_trim_1d(y: np.ndarray, target_len: int) -> np.ndarray:
    if y.shape[0] < target_len:
        return np.pad(y, (0, target_len - y.shape[0]))
    return y[:target_len]

def slice_clip(y: np.ndarray, start_sample: int, clip_len: int) -> np.ndarray:
    """内存中切片"""
    start = max(0, int(start_sample))
    end = start + int(clip_len)
    # 简单处理单声道
    if y.ndim == 1:
        seg = y[start:end]
        return pad_or_trim_1d(seg, clip_len)
    return y

def is_silent(audio: np.ndarray, threshold_db: float = -60.0) -> bool:
    if audio is None or len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio ** 2)))
    db = 20 * np.log10(rms + 1e-9)
    return db < threshold_db

# ---------------------------
# 4. Mel 特征
# ---------------------------
def mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)

def normalize_mel(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / (m.std() + 1e-6)

def spec_augment(m: np.ndarray, freq_mask_param, time_mask_param, num_masks) -> np.ndarray:
    m = m.copy()
    n_mels, n_steps = m.shape
    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, n_mels - f))
            m[f0:f0 + f, :] = 0
        t = np.random.randint(0, time_mask_param + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, n_steps - t))
            m[:, t0:t0 + t] = 0
    return m
