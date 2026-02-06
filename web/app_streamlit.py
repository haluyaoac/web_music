# web/app_streamlit.py
import os
import sys
from pathlib import Path

# 关键：Streamlit 启动时也要禁用 torch 的 autoload/compile（否则可能又卡）
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

# Ensure project root is on sys.path when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import tempfile
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 【修改点 1】引入配置和新版工具函数
import src.exp_cfg as cfg
from src.infer import AudioPredictor
from src.utils_audio import (
    load_audio_whole, 
    get_clip_starts, 
    slice_clip, 
    mel_spectrogram, 
    normalize_mel
)

# 可选：让页面干净一点
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed*")

st.set_page_config(page_title="Music Genre Classifier", layout="wide")
st.title("🎵 音乐风格识别 Demo")
st.caption("上传 MP3/WAV → 提取 Log-Mel 频谱 → CNN/ResNet 预测风格")

# 侧边栏配置
with st.sidebar:
    st.header("推理参数")
    
    # 默认模型路径尝试从 config 读取
    default_model = os.path.join(cfg.OUT_ROOT, cfg.EXP_NAME, "best.pt")
    if not os.path.exists(default_model):
        default_model = "models/cnn_melspec.pth" # Fallback
        
    model_path = st.text_input("模型权重路径", default_model)
    
    # 允许修改 Label Map 路径 (实际上 infer.py 默认读 cfg.LABEL_MAP_JSON)
    # 这里我们通过修改 cfg 来生效
    map_path = st.text_input("类别映射路径", cfg.LABEL_MAP_JSON)
    
    st.divider()
    
    topk = st.slider("Top-K 展示", 1, 10, 5)
    
    # 【修改点 2】参数适配：不再使用 hop_seconds，改为 num_clips
    clip_seconds = st.slider("切片长度（秒）", 1.0, 10.0, float(cfg.INFER_CLIP_SECONDS), 0.5)
    num_clips = st.slider("推理采样切片数", 1, 20, 5, 1, help="将音频切成多少段进行投票")

    st.header("频谱显示")
    preview_seconds = st.slider("频谱预览音频长度（秒）", 1.0, 20.0, 6.0, 1.0)


# 上传文件
up = st.file_uploader("上传音频文件（MP3/WAV/FLAC 等）", type=["wav", "mp3", "flac", "ogg", "m4a", "aac"])

if up is None:
    st.info("请上传音频文件。")
    st.stop()

st.audio(up)

# 保存临时文件
suffix = "." + up.name.split(".")[-1].lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
    f.write(up.getbuffer())
    tmp_path = f.name

try:
    # 更新全局配置 (Monkey Patching) 以适配当前用户的侧边栏选择
    cfg.LABEL_MAP_JSON = map_path
    cfg.INFER_CLIP_SECONDS = clip_seconds
    cfg.INFER_NUM_CLIPS = num_clips
    # SR, N_MELS 等保持 cfg 默认

    col1, col2 = st.columns([1, 1])

    # ---------------------------
    # 左侧：频谱预览
    # ---------------------------
    with col1:
        st.subheader("📈 Log-Mel 频谱图（预览）")
        
        # 1. 加载音频
        y = load_audio_whole(tmp_path, sr=cfg.SR)
        
        # 2. 截取预览长度
        max_len = int(cfg.SR * preview_seconds)
        y_preview = y[:max_len] if len(y) > max_len else y
        
        # 3. 切片 (仅为了画图，取中间一段)
        # 使用统一的 get_clip_starts
        starts = get_clip_starts(
            y_len=len(y_preview), 
            sr=cfg.SR, 
            clip_seconds=clip_seconds, 
            num_clips=1, 
            mode="uniform"
        )
        
        if len(starts) > 0:
            seg = slice_clip(y_preview, starts[0], int(cfg.SR * clip_seconds))
            
            # 4. Mel 计算
            m = mel_spectrogram(seg, sr=cfg.SR, n_mels=cfg.N_MELS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
            m = normalize_mel(m)

            fig = plt.figure(figsize=(10, 4))
            plt.imshow(m, aspect="auto", origin="lower", cmap="viridis")
            plt.xlabel("Time Frames")
            plt.ylabel("Mel Bins")
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Segment @ {starts[0]/cfg.SR:.2f}s")
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("音频太短，无法生成频谱。")

    # ---------------------------
    # 右侧：模型推理
    # ---------------------------
    with col2:
        st.subheader("🤖 预测结果")
        t0 = time.time()
        
        # 【修改点 3】调用新的 AudioPredictor
        # 注意：Predictor 内部会读取 cfg.INFER_CLIP_SECONDS 等参数
        predictor = AudioPredictor(model_path=model_path)
        
        # 执行预测
        top_res, mean_probs = predictor.predict(tmp_path, topk=topk)
        
        dt = time.time() - t0

        if top_res is not None:
            # 整理结果
            df = pd.DataFrame(top_res, columns=["Genre", "Probability"])
            
            st.write(f"采样切片数：**{num_clips}** |  推理耗时：**{dt:.2f}s**")
            st.dataframe(df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

            st.subheader("概率分布")
            fig2 = plt.figure(figsize=(10, 4))
            # 绘制 Top-K
            plt.bar(df["Genre"], df["Probability"], color="skyblue")
            plt.ylim(0, 1.0)
            plt.ylabel("Confidence")
            plt.title(f"Top-{topk} Predictions")
            st.pyplot(fig2, clear_figure=True)
            
            # 展开全量
            with st.expander("查看所有类别概率"):
                all_genres = predictor.genres
                all_df = pd.DataFrame({"Genre": all_genres, "Probability": mean_probs})
                all_df = all_df.sort_values("Probability", ascending=False)
                st.dataframe(all_df.style.format({"Probability": "{:.4f}"}), use_container_width=True)
        else:
            st.error("推理返回空结果（可能音频过短或静音）。")

except Exception as e:
    st.error(f"发生错误：{e}")
    # 打印堆栈以便调试
    import traceback
    st.text(traceback.format_exc())

finally:
    # 清理临时文件
    try:
        os.remove(tmp_path)
    except Exception:
        pass