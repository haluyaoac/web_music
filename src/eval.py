# eval.py
import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import exp_cfg as cfg
from infer import AudioPredictor # <--- 直接复用推理引擎

def load_split_items(split_json, split_name):
    with open(split_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [it for it in items if it["split"] == split_name]

def save_confusion_matrix(cm, labels, out_path):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, aspect="auto", origin="upper", cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v > 0:
                plt.text(j, i, str(v), ha="center", va="center",
                         color="white" if v > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    # 1. 实验目录准备
    out_dir = os.path.join(cfg.OUT_ROOT, cfg.EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    # 尝试加载训练时的配置覆盖当前配置 (Optional)
    exp_cfg_json = os.path.join(out_dir, "exp_cfg.json")
    if os.path.exists(exp_cfg_json):
        print(f"Loading training config from {exp_cfg_json}")
        with open(exp_cfg_json, "r") as f:
            saved_cfg = json.load(f)
            # 仅覆盖关键参数
            for k in ["N_MELS", "N_FFT", "HOP_LENGTH", "MODEL_TYPE"]:
                if k in saved_cfg: setattr(cfg, k, saved_cfg[k])

    # 2. 初始化预测器
    # 指定 model_path 为 best.pt
    ckpt_path = os.path.join(out_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    predictor = AudioPredictor(model_path=ckpt_path)
    genres = predictor.genres

    # 3. 加载评估数据
    items = load_split_items(cfg.SPLIT_JSON, cfg.EVAL_SPLIT)
    print(f"Evaluating on {cfg.EVAL_SPLIT} set: {len(items)} items")

    y_true, y_pred = [], []
    skipped = 0
    cfg.INFER_CLIP_SECONDS = cfg.VAL_CLIP_SECONDS
    cfg.INFER_NUM_CLIPS = cfg.EVAL_NUM_CLIPS
    # 4. 循环预测
    for i, it in enumerate(items):
        fp = it["path"]
        label = int(it["label"])
        
        try:
            # 调用 infer 的 predict 接口
            # 这里 predict 内部会自动根据 cfg.INFER_NUM_CLIPS 进行切片
            topk, probs = predictor.predict(fp)
            
            if probs is None:
                skipped += 1
                continue
                
            pred_idx = int(np.argmax(probs))
            
            y_true.append(label)
            y_pred.append(pred_idx)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(items)}...")

        except Exception as e:
            print(f"Error on {fp}: {e}")
            skipped += 1

    # 5. 计算指标
    if len(y_true) == 0:
        print("No samples evaluated.")
        return

    acc = (np.array(y_true) == np.array(y_pred)).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    print("\n" + "="*40)
    print(f"Result for {cfg.EXP_NAME} ({cfg.EVAL_SPLIT})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Skipped:  {skipped}")
    print("="*40)
    
    report = classification_report(y_true, y_pred, target_names=genres, digits=4)
    print(report)

    # 6. 保存结果
    with open(os.path.join(out_dir, f"{cfg.EVAL_SPLIT}_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAcc: {acc:.4f}\nF1: {macro_f1:.4f}\n")

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, genres, os.path.join(out_dir, f"{cfg.EVAL_SPLIT}_cm.png"))

    # Summary.csv
    summary_path = os.path.join(cfg.OUT_ROOT, "summary.csv")
    if os.path.exists(summary_path):
        # 简单追加或更新逻辑(略)，建议手动维护或使用 pandas
        pass

if __name__ == "__main__":
    main()