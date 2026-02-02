# utils_audio.py
import subprocess
import random
import numpy as np
import librosa
import exp_cfg as cfg


# ---------------------------
# FFprobe: 获取音频总时长（秒）
# ---------------------------
def get_duration_ffprobe(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffprobe failed: {path}\n{err}")
    s = p.stdout.decode("utf-8", errors="ignore").strip()
    return float(s) if s else 0.0


# ---------------------------
# FFmpeg: 按 offset/duration 解码到 float32 PCM
# ---------------------------
def load_audio_ffmpeg(
    path: str,
    sr: int = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None
) -> tuple[np.ndarray, int]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]

    # 快速 seek：-ss 放在 -i 前
    if offset and offset > 0:
        cmd += ["-ss", str(offset)]

    cmd += ["-i", path]

    if duration and duration > 0:
        cmd += ["-t", str(duration)]

    cmd += ["-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "f32le", "pipe:1"]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed: {path}\n{err}")

    y = np.frombuffer(p.stdout, dtype=np.float32)
    if not mono:
        y = y.reshape(-1, 2).T  # (2, n)

    return y, sr


def pad_or_trim_1d(y: np.ndarray, target_len: int) -> np.ndarray:
    if y.shape[0] < target_len:
        return np.pad(y, (0, target_len - y.shape[0]))
    return y[:target_len]


# Slice a fixed-length clip from an in-memory waveform.
def slice_clip(y: np.ndarray, start_sample: int, clip_len: int) -> np.ndarray:
    start = max(0, int(start_sample))
    end = start + int(clip_len)
    if y.ndim == 1:
        seg = y[start:end]
        return pad_or_trim_1d(seg, clip_len)
    # Stereo: (2, n) or (n, 2) not expected, handle (2, n)
    if y.shape[0] == 2:
        seg = y[:, start:end]
        if seg.shape[1] < clip_len:
            pad = clip_len - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, pad)))
        else:
            seg = seg[:, :clip_len]
        return seg
    # Fallback: treat last axis as time
    seg = y[..., start:end]
    if seg.shape[-1] < clip_len:
        pad = clip_len - seg.shape[-1]
        pad_width = [(0, 0)] * seg.ndim
        pad_width[-1] = (0, pad)
        seg = np.pad(seg, pad_width)
    else:
        seg = seg[..., :clip_len]
    return seg


# （可选）仍保留一个“随机取一段”的便捷函数
def load_audio(
    path: str,
    sr: int = 22050,
    mono: bool = True,
    clip_seconds: float = 3.0,
    random_offset: bool = True,
) -> tuple[np.ndarray, int]:
    offset = 0.0
    if random_offset:
        try:
            track_seconds = get_duration_ffprobe(path)
        except Exception:
            track_seconds = 0.0

        if track_seconds > clip_seconds:
            offset = random.uniform(0.0, track_seconds - clip_seconds)

    y, _sr = load_audio_ffmpeg(path, sr=sr, mono=mono, offset=offset, duration=clip_seconds)

    target_len = int(sr * clip_seconds)
    if mono:
        y = pad_or_trim_1d(y, target_len)
    else:
        # stereo: (2, n)
        if y.shape[1] < target_len:
            pad = target_len - y.shape[1]
            y = np.pad(y, ((0, 0), (0, pad)))
        else:
            y = y[:, :target_len]

    return y, _sr


# ---------------------------
# 静音检测
# ---------------------------
def is_silent(audio: np.ndarray, threshold_db: float = -60.0) -> bool:
    if audio is None or len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio ** 2)))
    db = 20 * np.log10(rms + 1e-9)
    return db < threshold_db


# ---------------------------
# 切片起点生成
# ---------------------------
def candidate_starts(
    y_len: int,
    sr: int,
    clip_seconds: float,
    hop_seconds: float,
    trim_ratio: float = 0.0,
    trim_seconds: float = 0.0
):
    clip_len = int(sr * clip_seconds)
    hop_len = max(1, int(sr * hop_seconds))
    if y_len <= 0:
        return [0]

    trim = 0
    if trim_ratio > 0:
        trim = max(trim, int(y_len * trim_ratio))
    if trim_seconds > 0:
        trim = max(trim, int(sr * trim_seconds))

    start_min = min(trim, max(0, y_len - 1))
    end_max = y_len - clip_len - trim
    if end_max < start_min:
        return [0]

    starts = list(range(start_min, end_max + 1, hop_len))
    return starts if starts else [0]


def uniform_starts(
    y_len: int,
    sr: int,
    clip_seconds: float,
    num_clips: int,
    trim_ratio: float = 0.0,
    trim_seconds: float = 0.0
):
    clip_len = int(sr * clip_seconds)
    if y_len <= 0:
        return [0] * max(1, num_clips)

    trim = 0
    if trim_ratio > 0:
        trim = max(trim, int(y_len * trim_ratio))
    if trim_seconds > 0:
        trim = max(trim, int(sr * trim_seconds))

    start_min = min(trim, max(0, y_len - 1))
    end_max = y_len - clip_len - trim
    if end_max < start_min:
        return [0] * max(1, num_clips)

    if num_clips <= 1:
        return [start_min]

    positions = np.linspace(start_min, end_max, num=num_clips)
    return [int(round(p)) for p in positions]


# Split a waveform into fixed-length clips with a hop.
def split_fixed(
    y: np.ndarray,
    sr: int,
    clip_seconds: float = 3.0,
    hop_seconds: float = 1.5,
):
    clip_len = int(sr * clip_seconds)
    y_len = int(y.shape[-1])
    starts = candidate_starts(y_len, sr, clip_seconds, hop_seconds)
    return [slice_clip(y, s, clip_len) for s in starts]


# ---------------------------
# Mel 计算与增强
# ---------------------------
def mel_spectrogram(
    y: np.ndarray,
    sr: int = cfg.SR,
    n_mels: int = cfg.N_MELS,
    n_fft: int = cfg.N_FFT,
    hop_length: int = cfg.HOP_LENGTH,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_mel(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / (m.std() + 1e-6)


def spec_augment(
    m: np.ndarray,
    freq_mask_param: int = cfg.FREQ_MASK_PARAM,
    time_mask_param: int = cfg.TIME_MASK_PARAM,
    num_masks: int = cfg.NUM_MASKS
) -> np.ndarray:
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
