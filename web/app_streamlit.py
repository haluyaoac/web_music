import os, tempfile
import streamlit as st
from src.infer import predict_file

st.title("🎵 音乐风格识别 Demo")
st.write("上传 MP3/WAV，系统会提取 Mel 频谱并预测风格。")

up = st.file_uploader("上传音频文件", type=["mp3","wav"])
if up is not None:
    # 播放
    st.audio(up)

    # 保存到临时文件（很多音频库更喜欢文件路径）:contentReference[oaicite:6]{index=6}
    suffix = "." + up.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(up.getbuffer())
        tmp_path = f.name

    with st.spinner("分析中..."):
        top3 = predict_file(tmp_path)

    os.remove(tmp_path)

    st.subheader("预测结果（Top-3）")
    for g, p in top3:
        st.write(f"**{g}**: {p:.3f}")
