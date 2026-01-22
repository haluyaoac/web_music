from dataclasses import dataclass

@dataclass
class ExpCFG:
    # ========== 基础 ==========
    exp_name: str = "E0_baseline"
    seed: int = 42

    # ========== 数据 ==========
    split_json: str = "data/splits/split_v1_strat.json"
    sr: int = 22050
    clip_seconds: float = 3.0
    hop_seconds: float = 1.5

    # ========== 模型 ==========
    model_type: str = "small_cnn"  # small_cnn | small_cnn_v2 | resnet18
    n_classes: int = 5

    # ========== 增强 ==========
    use_specaug: bool = False
    freq_mask_param: int = 20
    time_mask_param: int = 30
    num_masks: int = 2

    # ========== 训练 ==========
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    use_cosine: bool = False
    early_stop_patience: int = 0  # 0 表示不启用

    # ========== 保存 ==========
    out_root: str = "runs"
