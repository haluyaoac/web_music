import os
import sys
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
# ... 其他 import
import json
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

import exp_cfg as cfg

from split_dataset import SplitDataset
from models import build_model


def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''
    返回混合后的输入 mixed_x，以及混合前的两组标签 y_a, y_b 和混合比例 lam
    '''
    if alpha > 0:
        # 从 Beta 分布中采样 lambda
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # 生成乱序索引
    index = torch.randperm(batch_size).to(device)

    # 混合输入
    # 注意：如果 x 是 [B, K, C, H, W]，这里是对 dim=0 (歌曲维度) 进行混合，逻辑完全通顺
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # 记录对应的标签
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    混合 Loss 计算
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if x.dim() == 5:
            b, k = x.shape[0], x.shape[1]
            x_flat = x.reshape(b * k, *x.shape[2:])      # 更安全：reshape
            logits = model(x_flat).reshape(b, k, -1)     # [B,K,C]
            logits_song = torch.logsumexp(logits, dim=1) - np.log(logits.shape[1])     # [B,C]  ✅歌曲级
            pred = logits_song.argmax(dim=1)
        else:
            pred = model(x).argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_jsonable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def save_curves(log_csv_path, out_png_path):
    epochs, train_loss, train_acc, train_song_acc, val_acc = [], [], [], [], []
    with open(log_csv_path, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头
        for line in f:
            # 按实际列数解包：epoch, train_loss, train_acc, train_song_acc, val_acc, seconds
            parts = line.strip().split(",")
            # 确保列数正确（防止异常行）
            if len(parts) >= 6:
                e = parts[0]
                tl = parts[1]
                ta = parts[2]
                tsa = parts[3]
                va = parts[4]  # val_acc 是第5列（索引4）
                # sec = parts[5]  # 不需要可以注释
                
                epochs.append(int(e))
                train_loss.append(float(tl))
                train_acc.append(float(ta))
                train_song_acc.append(float(tsa))
                val_acc.append(float(va))

    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_loss, color='blue', label='Train Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, color='red', label='Val Acc', linestyle='--')
    ax2.plot(epochs, train_acc, color='green', label='Train Clip Acc', linestyle='--')
    ax2.plot(epochs, train_song_acc, color='orange', label='Train Song Acc', linestyle='--')
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close(fig)

def main():

    seed_everything(cfg.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # 读 label_map（确保 n_classes 对齐）
    with open(cfg.LABEL_MAP_JSON, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    genres = [label_map[str(i)] for i in range(len(label_map))]
    cfg.NUM_CLASSES = len(genres)

    out_dir = os.path.join(cfg.OUT_ROOT, cfg.EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 保存本次 cfg
    with open(os.path.join(out_dir, "exp_cfg.json"), "w", encoding="utf-8") as f:
        cfg_dict = {
            k: v
            for k, v in cfg.__dict__.items()
            if not k.startswith("_") and is_jsonable(v)
        }
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    # 数据
    train_ds = SplitDataset(
        cfg.SPLIT_JSON, split="train",
        sr=cfg.SR,
        clip_seconds=cfg.CLIP_SECONDS,
        hop_seconds=cfg.HOP_SECONDS,
        random_clip=cfg.RANDOM_CLIP,
        use_specaug=cfg.USE_SPECAUG,
        freq_mask_param=cfg.FREQ_MASK_PARAM,
        time_mask_param=cfg.TIME_MASK_PARAM,
        num_masks=cfg.NUM_MASKS,
        clips_per_item=cfg.TRAIN_CLIPS_PER_ITEM,      
        unique_clips=cfg.TRAIN_UNIQUE_CLIPS,
        trim_ratio=cfg.TRAIN_TRIM_RATIO,
        trim_seconds=cfg.TRAIN_TRIM_SECONDS,
    )
    val_ds = SplitDataset(
        cfg.SPLIT_JSON, split="val",
        sr=cfg.SR,
        clip_seconds=cfg.EVAL_CLIP_SECONDS,
        hop_seconds=cfg.HOP_SECONDS,
        random_clip=False,
        use_specaug=False,
        deterministic=True,
        deterministic_num_clips=cfg.EVAL_NUM_CLIPS,
        deterministic_trim_ratio=cfg.EVAL_TRIM_RATIO,
        deterministic_trim_seconds=cfg.EVAL_TRIM_SECONDS,
    )

    # 1. 动态计算 worker 数量
    # os.cpu_count() 获取核心数，为了稳定性通常上限设为 8 或 16
    # 如果是在 Windows 上运行报错，请尝试改回 0 或 2
    num_workers = min(os.cpu_count(), 8) 
    print(f"Using {num_workers} dataloader workers")

    # 2. 优化 DataLoader 配置
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,   # [关键] 开启多进程
        pin_memory=True,           # [加速] 开启锁页内存，加快 CPU->GPU 传输
        persistent_workers=(num_workers > 0) # [加速] 避免每个 epoch 结束后销毁进程重建
    )

    val_loader = DataLoader(
        val_ds,   
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,   # 验证集也可以开启
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )


    # 模型
    model = build_model(cfg.MODEL_TYPE, cfg.NUM_CLASSES).to(device)
    n_params = count_params(model)

    # 训练策略
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    scheduler = None
    if cfg.USE_COSINE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)

    best_acc = -1.0
    bad_epochs = 0
    best_path = os.path.join(out_dir, "best.pt")

    log_csv = os.path.join(out_dir, "train_log.csv")
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,train_song_acc,val_acc,seconds\n")

    warmup_epochs = getattr(cfg, "WARMUP_EPOCHS", 3)

    

    for epoch in range(1, cfg.EPOCHS + 1):
        train_correct = 0
        train_total = 0
        train_song_correct = 0
        train_song_total = 0

        t0 = time.time()
        # warmup: 线性从 0 -> lr
        if scheduler is not None and epoch <= warmup_epochs:
            warmup_lr = cfg.LR * epoch / max(1, warmup_epochs)
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr

        model.train()

        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            # --- [Step 1] Mixup 准备数据 ---
            # 仅在训练且启用时执行
            use_mixup = getattr(cfg, "USE_MIXUP", False) and (cfg.MIXUP_ALPHA > 0)
            
            if use_mixup:
                # 生成混合数据和两组标签
                mixed_x, y_a, y_b, lam = mixup_data(x, y, cfg.MIXUP_ALPHA, device)
                input_data = mixed_x
            else:
                input_data = x

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                # --- 修改开始：Clip-level Strong Supervision ---
                
                # 1. 前向传播：将 [B, K, ...] 拍平成 [B*K, ...]
                if input_data.dim() == 5:
                    b, k = input_data.shape[0], input_data.shape[1]
                    # 融合 Batch 和 Clip 维度
                    x_flat = input_data.reshape(b * k, *input_data.shape[2:]) 
                    final_pred = model(x_flat)  # 输出 [B*K, n_classes]
                    
                    # 2. 标签对齐：将 Song 标签扩展到每个 Clip
                    # y_a, y_b 原本是 [B], 现在需要变成 [B*K]
                    if use_mixup:
                        y_a_expanded = y_a.repeat_interleave(k)
                        y_b_expanded = y_b.repeat_interleave(k)
                    else:
                        y_expanded = y.repeat_interleave(k)
                else:
                    # 如果本来就是 4D 输入 (Batch=1 或其他情况)
                    final_pred = model(input_data)
                    y_a_expanded = y_a
                    y_b_expanded = y_b
                    y_expanded = y

                # 3. 计算 Loss (对每个片段计算)
                if use_mixup:
                    loss = mixup_criterion(crit, final_pred, y_a_expanded, y_b_expanded, lam)
                else:
                    loss = crit(final_pred, y_expanded)
                
                # --- 修改结束 ---

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

            # --- [Step 3] 准确率统计 (修复 Bug) ---
            with torch.no_grad():
                # 修改点：选择对应的已扩展（Expanded）标签
                if use_mixup:
                    target_for_acc = y_a_expanded  # 形状为 [B*K]
                else:
                    target_for_acc = y_expanded    # 形状为 [B*K]
                
                # 获取预测类别
                pred_label = final_pred.argmax(dim=1) # 形状为 [B*K]
                
                # 现在 pred_label [B*K] 和 target_for_acc [B*K] 维度匹配了
                train_correct += (pred_label == target_for_acc).sum().item()
                train_total += target_for_acc.numel()

                # ====== song-level acc（新增，和 val 的融合口径对齐）======
                # 注意：song-level acc 用原始 y（song 标签），不要用 mixup 后的 y_a/y_b
                # 因为 val_acc 也是用真实标签 y 来算的；mixup 下算 song_acc 只是做“可比指标”
                if input_data.dim() == 5:
                    # final_pred: [B*K, C] -> [B, K, C]
                    logits_bkc = final_pred.reshape(b, k, -1)

                    # 融合方式建议与 eval_acc 一致：logsumexp（更稳定）
                    fused_logits = torch.logsumexp(logits_bkc, dim=1)  # [B, C]

                    song_pred = fused_logits.argmax(dim=1)  # [B]
                    train_song_correct += (song_pred == y).sum().item()
                    train_song_total += y.numel()

        train_acc = train_correct / max(1, train_total)
        train_song_acc = train_song_correct / max(1, train_song_total)

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_acc = eval_acc(model, val_loader, device)
        sec = time.time() - t0

        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()


        print(
            f"[{cfg.EXP_NAME}] epoch={epoch:02d} loss={train_loss:.4f} "
            f"train_clip_acc={train_acc:.4f} train_song_acc={train_song_acc:.4f} "
            f"val_acc={val_acc:.4f} time={sec:.1f}s"
        )


        with open(log_csv, "a", encoding="utf-8") as f:
           f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{train_song_acc:.6f},{val_acc:.6f},{sec:.3f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if cfg.EARLY_STOP_PATIENCE > 0 and bad_epochs >= cfg.EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch}, best_val_acc={best_acc:.4f}")
                break

    # 曲线
    curve_png = os.path.join(out_dir, "train_curves.png")
    save_curves(log_csv, curve_png)

    # 写入 summary.csv（追加一行）
    summary_path = os.path.join(cfg.OUT_ROOT, "summary.csv")
    header = "exp_name,model_type,use_specaug,use_cosine,label_smoothing,weight_decay,params,best_val_acc\n"
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(header)

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"{cfg.EXP_NAME},{cfg.MODEL_TYPE},{int(cfg.USE_SPECAUG)},{int(cfg.USE_COSINE)},"
                f"{cfg.LABEL_SMOOTHING},{cfg.WEIGHT_DECAY},{n_params},{best_acc:.6f}\n")

    print("saved:", best_path)
    print("saved:", log_csv)
    print("saved:", curve_png)
    print("updated:", summary_path)


if __name__ == "__main__":
    main()
