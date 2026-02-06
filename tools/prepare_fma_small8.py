import os
import shutil
import pandas as pd
from tqdm import tqdm
import sys

# ================= 配置区域 =================
# 1. tracks.csv 的路径 (通常在 fma_metadata.zip 解压后)
CSV_PATH = 'C:/Code/python/vscode/music_genre_classifier/data/fma_metadata/tracks.csv'

# 2. 原始 fma_small 数据的根目录 (里面应该是 000, 001, ... 等文件夹)
SOURCE_DIR = 'C:/Users/14367/Desktop/fma_small'
# 3. 目标输出目录
DEST_DIR = os.path.join('data', 'raw_fma8')

# ===========================================

def load_tracks(csv_path):
    """
    加载 tracks.csv，处理 FMA 特有的多级表头
    """
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到文件 {csv_path}")
        print("请下载 fma_metadata.zip 并解压，确保路径正确。")
        sys.exit(1)

    print("正在读取 tracks.csv，这可能需要几秒钟...")
    # header=[0, 1] 指示前两行是表头
    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    
    # 筛选出 subset 为 'small' 的数据
    small_tracks = tracks[tracks[('set', 'subset')] == 'small']
    
    # 只保留我们要的列：流派 (genre_top)
    # 注意：FMA small 的 genre_top 应该没有空值，但为了保险还是 dropna 一下
    return small_tracks[[('track', 'genre_top')]].dropna()

def organize_files():
    # 1. 加载元数据
    df = load_tracks(CSV_PATH)
    print(f"✅ 成功加载元数据，共有 {len(df)} 条 'small' 数据集记录。")

    # 2. 准备计数器
    success_count = 0
    missing_count = 0

    # 3. 遍历每一行进行处理
    print(f"🚀 开始整理文件到: {DEST_DIR}")
    
    # 使用 tqdm 显示进度条
    for track_id, row in tqdm(df.iterrows(), total=len(df)):
        genre = row[('track', 'genre_top')]
        
        # FMA 的文件名是 6 位数字，例如 ID 2 -> 000002.mp3
        track_id_str = f"{int(track_id):06d}"
        
        # FMA 的原始目录结构是前3位数字作为子文件夹，例如 000002.mp3 在 000/ 文件夹下
        src_folder = track_id_str[:3]
        src_filename = track_id_str + ".mp3"
        
        # 拼接源文件路径
        src_path = os.path.join(SOURCE_DIR, src_folder, src_filename)
        
        # 拼接目标文件路径: data/raw_fma8/Hip-Hop/000002.mp3
        # 处理一下 genre 名字，防止有非法字符（虽然 FMA small 的类别名都很干净）
        safe_genre = genre.replace('/', '_') 
        dest_folder = os.path.join(DEST_DIR, safe_genre)
        dest_path = os.path.join(dest_folder, src_filename)

        # 检查源文件是否存在
        if os.path.exists(src_path):
            # 确保目标文件夹存在
            os.makedirs(dest_folder, exist_ok=True)
            
            # 复制文件 (使用 copy2 保留元数据，如果想移动用 move)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
            
            success_count += 1
        else:
            # print(f"⚠️ 文件丢失: {src_path}") # 如果丢失太多，可以取消注释查看详情
            missing_count += 1

    print("=" * 30)
    print("🎉 整理完成！")
    print(f"✅ 成功复制: {success_count} 个文件")
    print(f"❌ 源文件缺失: {missing_count} 个文件")
    print(f"📂 数据已保存在: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    organize_files()