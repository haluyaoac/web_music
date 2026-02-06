# exp_cfg.py
import os

# ==============================================================================
# 1. 项目路径与基础配置 (Project & Paths)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'runs')

# 数据集选择
USE_DATA: str = "fma8"  # fma8 | fma_medium | fma_large
DATA_SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
SPLIT_JSON: str = os.path.join(DATA_SPLITS_DIR, f"split_{USE_DATA}.json")
LABEL_MAP_JSON = os.path.join(BASE_DIR, "models", "label_map", USE_DATA + ".json")
FMA_AUDIO_DIR = os.path.join(DATA_DIR, "raw_fma8")
SPLIT_RATIO = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

# 随机种子 (复现性)
RANDOM_SEED: int = 42

# ==============================================================================
# 2. 音频特征参数 (Audio & Mel Spec)
# ==============================================================================
SR: int = 44100            # 采样率
CLIP_SECONDS: float = 5.0  # 标准切片时长
HOP_SECONDS: float = 2.5   # 切片步长 (仅用于推理或离线切片)

# STFT & Mel 参数
N_MELS: int = 256
N_FFT: int = 2048
HOP_LENGTH: int = 512

# ==============================================================================
# 3. 采样与切片策略 (Sampling Strategy)
# ==============================================================================
# --- 训练集策略 (Training) ---
TRAIN_CLIPS_PER_ITEM: int = 3       # 强监督力度：每个音频文件采样几个片段
TRAIN_RANDOM_CLIP: bool = True      # 是否随机选择起点 (必须 True 以覆盖全曲)
TRAIN_UNIQUE_CLIPS: bool = True     # True=无放回采样(互不重叠)，False=允许重复
TRAIN_COVERING_CLIPS: bool = True   # 仅在随机模式下生效，是否允许片段间重叠

TRAIN_TRIM_RATIO: float = 0.0       # 首尾修剪比例
TRAIN_TRIM_SECONDS: float = 0.0     # 首尾修剪时长

# --- 验证集策略 (Validation) ---
VAL_CLIP_SECONDS: float = CLIP_SECONDS
VAL_NUM_CLIPS: int = 15             # 验证时均匀采样片段数
VAL_TRIM_RATIO: float = 0.0
VAL_TRIM_SECONDS: float = 0.0

# --- 评估与推理策略 (Eval & Infer) ---
EVAL_SPLIT: str = "test"
EVAL_NUM_CLIPS: int = 0             # 0 = 使用所有可能的均匀切片
INFER_CLIP_SECONDS: float = CLIP_SECONDS
INFER_NUM_CLIPS: int = 0            # 0 = 使用所有可能的均匀切片
INFER_TRIM_RATIO: float = 0.0
INFER_TRIM_SECONDS: float = 0.0

# ==============================================================================
# 4. 数据清洗与过滤 (Data Cleaning)
# ==============================================================================
# [新增] 静音检测阈值 (dB)。低于此分贝的片段被视为静音并被丢弃/重试。
# 建议：环境音设 -40，录音棚音乐设 -60。
SILENCE_THRESHOLD_DB: float = None

# ==============================================================================
# 5. 模型结构 (Model Architecture)
# ==============================================================================
MODEL_TYPE: str = "resnet18"  # small_cnn | small_cnn_v2 | resnet18
MODEL_VERSION: str = 'E0'      # 版本号，用于区分实验名
EXP_NAME: str = MODEL_VERSION  # 实验文件夹名称

# ==============================================================================
# 6. 训练与增强策略 (Training Strategy & Augmentation)
# ==============================================================================
# --- 数据增强 (Augmentation) ---
USE_SPECAUG: bool = True        # [策略开关] 频谱遮挡 (SpecAugment)
FREQ_MASK_PARAM: int = 20       # 频率遮挡最大宽度
TIME_MASK_PARAM: int = 30       # 时间遮挡最大宽度
NUM_MASKS: int = 2              # 遮挡块数量

USE_MIXUP: bool = False         # [策略开关] 混合增强 (Mixup)
MIXUP_ALPHA: float = 0.4        # Beta分布参数 (0.2~1.0)，越大混合越强

# --- 优化与正则化 (Optimization & Regularization) ---
EPOCHS: int = 30
BATCH_SIZE: int = 32
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-4      # AdamW 的权重衰减

USE_COSINE: bool = True         # [策略开关] 余弦退火学习率调度
WARMUP_EPOCHS: int = 5          # [新增] 预热轮次，防止初期梯度爆炸

LABEL_SMOOTHING: float = 0.1    # [策略开关] 标签平滑 (推荐 0.1)

# --- 硬件加速与稳定性 (Hardware & Stability) ---
USE_AMP: bool = True            # [新增] 自动混合精度 (节省显存，加速训练)
GRAD_CLIP_NORM: float = 1.0     # [新增] 梯度裁剪阈值 (防止梯度爆炸)

EARLY_STOP_PATIENCE: int = 5    # 早停轮次

# ==============================================================================
# 7. 输出配置 (Output)
# ==============================================================================
OUT_ROOT: str = "runs"

# 辅助函数
def get_config_dict():
    return {k: v for k, v in globals().items() 
            if k.isupper() and not k.startswith('_')}

# 自动推导 NUM_CLASSES (防止循环导入，简单的逻辑)
# 实际运行中 train.py 会重新加载 label_map 覆盖此值
if USE_DATA == 'fma8':
    NUM_CLASSES = 8
else:
    # 默认占位，train.py 会修正
    NUM_CLASSES = 8