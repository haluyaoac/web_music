music_genre_classifier/
  README.md
  requirements.txt
  .gitignore

  data/
    raw/                 # 原始音频（GTZAN/FMA 解压到这里）
    processed/           # 预处理后的特征（.npy/.pt）
    splits/              # train/val/test 切分文件（csv/json）

  models/
    cnn_melspec.pth      # 训练好的模型权重
    label_map.json       # 类别映射（index->genre）

  src/
    config.py
    utils_audio.py       # 读取音频、算 mel、切片、归一化
    dataset.py           # PyTorch Dataset/DataLoader
    model.py             # CNN 网络
    train.py             # 训练入口
    eval.py              # 评估/混淆矩阵
    infer.py             # 单文件推理（mp3/wav）

  web/
    app_streamlit.py     # Web 演示（上传音频→预测）
