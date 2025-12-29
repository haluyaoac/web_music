import json, torch
import numpy as np
from src.model import SimpleCNN
from src.utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel

def predict_file(path, model_path="models/cnn_melspec.pth", map_path="models/label_map.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = json.load(open(map_path, "r", encoding="utf-8"))
    genres = [label_map[str(i)] for i in range(len(label_map))]

    model = SimpleCNN(n_classes=len(genres))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    y = load_audio(path, sr=22050)
    clips = split_fixed(y, 22050, clip_seconds=3.0, hop_seconds=1.5)

    probs = []
    with torch.no_grad():
        for seg in clips:
            m = normalize_mel(mel_spectrogram(seg, sr=22050))
            x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,128,T]
            p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            probs.append(p)

    mean_p = np.mean(probs, axis=0)
    top = mean_p.argsort()[::-1][:3]
    return [(genres[i], float(mean_p[i])) for i in top]
