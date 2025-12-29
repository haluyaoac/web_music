# 读取音频、算 mel、切片、归一化
import numpy as np
import librosa

def load_audio(path, sr=22050, mono=True):
    y, _sr = librosa.load(path, sr=sr, mono=mono)
    return y

def split_fixed(y, sr, clip_seconds=3.0, hop_seconds=1.5):
    clip_len = int(sr * clip_seconds)
    hop_len  = int(sr * hop_seconds)
    clips = []
    for start in range(0, max(1, len(y) - clip_len + 1), hop_len):
        seg = y[start:start + clip_len]
        if len(seg) < clip_len:
            seg = np.pad(seg, (0, clip_len - len(seg)))
        clips.append(seg)
    return clips

def mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)

def normalize_mel(m):
    # 标准化到 0 均值/1 方差（也可以用全局 mean/std）
    return (m - m.mean()) / (m.std() + 1e-6)
