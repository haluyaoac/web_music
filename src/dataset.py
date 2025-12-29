import os, glob, json
import torch
from torch.utils.data import Dataset
from src.utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel

class GenreDataset(Dataset):
    def __init__(self, root_dir, genres, sr=22050, clip_seconds=3.0, hop_seconds=1.5):
        self.items = []
        self.genres = genres
        self.sr = sr
        self.clip_seconds = clip_seconds
        self.hop_seconds = hop_seconds

        for gi, g in enumerate(genres):
            for fp in glob.glob(os.path.join(root_dir, g, "*")):
                self.items.append((fp, gi))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, label = self.items[idx]
        y = load_audio(fp, sr=self.sr)
        # 随机取一段（训练时更像数据增强）
        clips = split_fixed(y, self.sr, self.clip_seconds, self.hop_seconds)
        seg = clips[torch.randint(low=0, high=len(clips), size=(1,)).item()]

        m = mel_spectrogram(seg, sr=self.sr)
        m = normalize_mel(m)

        # [1, n_mels, time]
        x = torch.tensor(m).unsqueeze(0)
        y = torch.tensor(label).long()
        return x, y
