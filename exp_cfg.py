import os

# ==========================================
# 项目路径配置 (Paths)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FMA_AUDIO_DIR = os.path.join(DATA_DIR, 'raw_fma8') 
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'runs')
DATA_SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
# Data
USE_DATA: str = "fma8"  # fma8 | fma_medium | fma_large
SPLIT_JSON: str = os.path.join(DATA_SPLITS_DIR, f"split_{USE_DATA}.json")
LABEL_MAP_JSON = os.path.join(BASE_DIR, "models", "label_map", USE_DATA + ".json")
# ==========================================
# 数据策略配置 (Data Strategy)
# ==========================================
# 选择数据集大小: 'small', 'medium', 'large'
# small: 8,000 tracks, 8 genres (平衡)
# medium: 25,000 tracks, 16 genres (不平衡)
DATASET_SUBSET = 'small'

# 类别映射 (根据 fma 定义，后续可根据 subset 自动加载)
# 这里先手动定义 small 的8个类别，用于 E0 阶段测试
GENRES_SMALL = {
    'Hip-Hop': 0, 'Pop': 1, 'Folk': 2, 'Experimental': 3, 
    'Rock': 4, 'International': 5, 'Electronic': 6, 'Instrumental': 7
}
NUM_CLASSES = len(GENRES_SMALL)
# 数据集划分比例
SPLIT_RATIO = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1
}
RANDOM_SEED = 42  # 保证结果可复现

# ==========================================
# 音频处理参数 (Audio Processing)
# ==========================================
SR: int = 22050
DURATION = 30        # FMA_small 是 30s，Medium/Large 是整首
CLIP_SECONDS: float = 5.0
HOP_SECONDS: float = 2.5
N_MELS: int = 128
N_FFT: int = 2048
HOP_LENGTH: int = 512

# ==========================================
# 4. 模型版本控制 (Model Versioning)
# ==========================================
# 切换这里的字符串来改变训练使用的模型结构
# 可选: 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'VGGish'
MODEL_VERSION = 'E0'
EXP_NAME = MODEL_VERSION



# 采样策略
# Train: L=5s, K=6 (spread across positions)
TRAIN_CLIPS_PER_ITEM: int = 4                   # 每个样本采样多少片段
RANDOM_CLIP: bool = True                        # 是否随机采样片段
TRAIN_TRIM_RATIO: float = 0.0                   # 开头修剪比例/秒数
TRAIN_TRIM_SECONDS: float = 0.0                 # 结尾修剪比例/秒数
TRAIN_UNIQUE_CLIPS: bool = True                 # 仅在 RANDOM_CLIP=True 时生效,放回切片是否唯一

# Val: fixed 6 uniform clips, average
EVAL_CLIP_SECONDS: float = CLIP_SECONDS         # 片段时长
EVAL_NUM_CLIPS: int = 4                         # 采样多少个均匀分布的片段
EVAL_TRIM_RATIO: float = 0.0                    # 开头修剪比例/秒数
EVAL_TRIM_SECONDS: float = 0.0                  # 结尾修剪比例/秒数

#Eval:
EVAL_SPLIT = "test"                    # "val" | "test"
EVAL_NUM_CLIPS: int = 0                         # 0 = use all uniform clips

# Infer: multi-clip mean fusion
INFER_CLIP_SECONDS: float = CLIP_SECONDS
INFER_NUM_CLIPS: int = 0                    # 0 = use all uniform clips
INFER_TRIM_RATIO: float = 0.0
INFER_TRIM_SECONDS: float = 0.0

# Model
MODEL_TYPE: str = "small_cnn"  # small_cnn | small_cnn_v2 | resnet18

# Augmentation
USE_SPECAUG: bool = True         # 是否使用 SpecAugment
FREQ_MASK_PARAM: int = 20
TIME_MASK_PARAM: int = 30
NUM_MASKS: int = 2

# Training
EPOCHS: int = 60
BATCH_SIZE: int = 16
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-2
LABEL_SMOOTHING: float = 0.0
USE_COSINE: bool = True
EARLY_STOP_PATIENCE: int = 8

# Output
OUT_ROOT: str = "runs"

# 在 cfg 中添加
USE_MIXUP = True
MIXUP_ALPHA = 0.4  # Beta分布参数，越大混合越强烈


# ==========================================
# 辅助函数：将配置导出为字典
# ==========================================
def get_config_dict():
    """
    将当前配置导出为字典，用于保存到实验记录中 (JSON)
    以便后续复现这次实验。
    """
    return {k: v for k, v in globals().items() 
            if k.isupper() and not k.startswith('_')}

if __name__ == "__main__":
    # 测试打印配置
    print(f"当前配置模式: {MODEL_VERSION}")
    #print(f"输入数据形状: {INPUT_SHAPE}")
    print(f"数据存放路径: {FMA_AUDIO_DIR}")
    print(f"数据集划分路径: {SPLIT_JSON}")
    print(f"音乐风格类型:{LABEL_MAP_JSON}")
