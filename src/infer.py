# infer.py
import os
import json
import torch
import numpy as np

import exp_cfg as cfg
from models import build_model
from src.utils_audio import (
    load_audio_whole,
    get_clip_starts,
    slice_clip,
    mel_spectrogram,
    normalize_mel
)

class AudioPredictor:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载 Label Map
        with open(cfg.LABEL_MAP_JSON, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.genres = [self.label_map[str(i)] for i in range(len(self.label_map))]
        self.num_classes = len(self.genres)

        # 2. 构建模型
        # 如果未指定路径，尝试自动寻找 (exp_cfg 中的 OUT_ROOT/EXP_NAME/best.pt)
        if model_path is None:
            model_path = os.path.join(cfg.OUT_ROOT, cfg.EXP_NAME, "best.pt")
        
        print(f"Loading model from: {model_path}")
        self.model = build_model(cfg.MODEL_TYPE, self.num_classes)
        
        # 加载权重
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
        else:
            print(f"[Warning] Model path {model_path} not found. Using random weights (Debug).")
            
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, audio_path, topk=3):
        """
        对单个音频文件进行预测
        :return: (topk_list, mean_probs, raw_logits)
        """
        # 1. 加载全长音频
        y = load_audio_whole(audio_path, sr=cfg.SR)
        if len(y) == 0:
            return None, None

        # 2. 统一切片计算 (使用 INFER 配置)
        starts = get_clip_starts(
            y_len=len(y),
            sr=cfg.SR,
            clip_seconds=cfg.INFER_CLIP_SECONDS,
            num_clips=cfg.INFER_NUM_CLIPS, 
            mode="uniform", # 推理通常用均匀
            trim_ratio=cfg.INFER_TRIM_RATIO,
            trim_seconds=cfg.INFER_TRIM_SECONDS
            seed = cfg.RANDOM_SEED
        )

        # 3. 提取特征
        clip_len = int(cfg.SR * cfg.INFER_CLIP_SECONDS)
        batch_mels = []
        for s in starts:
            seg = slice_clip(y, s, clip_len)
            m = mel_spectrogram(seg, cfg.SR, cfg.N_MELS, cfg.N_FFT, cfg.HOP_LENGTH)
            m = normalize_mel(m)
            batch_mels.append(m)
        
        # 4. 推理
        if not batch_mels: return None, None

        # Stack -> [K, 1, F, T]
        x = torch.tensor(np.array(batch_mels)).unsqueeze(1).to(self.device)
        
        logits = self.model(x) # [K, C]
        probs = torch.softmax(logits, dim=1).cpu().numpy() # [K, C]
        
        # 5. 融合 (Mean)
        mean_prob = np.mean(probs, axis=0) # [C]
        
        # Top K
        idx = np.argsort(mean_prob)[::-1][:topk]
        topk_res = [(self.genres[i], float(mean_prob[i])) for i in idx]
        
        return topk_res, mean_prob

# 简单测试入口
if __name__ == "__main__":
    predictor = AudioPredictor() # 自动加载 best.pt
    # 测试一个文件 (替换为你的实际路径)
    test_file = "data/raw_fma8/test_song.mp3" 
    if os.path.exists(test_file):
        res, _ = predictor.predict(test_file)
        print("Result:", res)