import os
import sys
import glob
import shutil
import logging
import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保能导入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import exp_cfg as cfg

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("cleaning_strict.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)

class StrictValidator:
    FFMPEG_PATH = shutil.which("ffmpeg")

    @staticmethod
    def try_decode(file_path):
        """
        模拟训练时的读取过程：强制将音频解码为 PCM 数据流。
        如果这里报错，训练时 100% 会报错。
        """
        if not StrictValidator.FFMPEG_PATH:
            return False, "No FFmpeg"

        # 构造命令：解码为 44100Hz 单声道 32位浮点数，输出到管道
        cmd = [
            StrictValidator.FFMPEG_PATH,
            "-v", "error",          # 只打印错误
            "-i", file_path,        # 输入
            "-f", "f32le",          # 强制输出格式：32位浮点 raw data
            "-ac", "1",             # 单声道
            "-ar", "44100",         # 采样率
            "-"                     # 输出到 stdout
        ]

        try:
            # 这里的逻辑是：必须拿到数据！
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,  # 捕获数据流
                stderr=subprocess.PIPE,  # 捕获错误日志
                timeout=10               # 10秒读不出来就算坏
            )

            # 判据 1: FFmpeg 报错退出
            if proc.returncode != 0:
                err = proc.stderr.decode('utf-8', 'ignore').strip()
                return False, f"FFmpeg Error: {err[:100]}"

            # 判据 2: 解码出的数据为空 (Zombie File)
            # 32位浮点数，4字节。如果数据量少于 4KB (约0.02秒)，认为无效
            if len(proc.stdout) < 4096:
                return False, f"Empty Stream (Size: {len(proc.stdout)} bytes)"

            return True, "OK"

        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, f"Exception: {e}"

def process_file(fp):
    is_valid, msg = StrictValidator.try_decode(fp)
    return fp, is_valid, msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="进程数，设为1最稳定")
    parser.add_argument("--dry-run", action="store_true", help="不移动文件，只扫描")
    args = parser.parse_args()

    src_dir = cfg.FMA_AUDIO_DIR
    bad_dir = os.path.join(cfg.DATA_DIR, 'bad_files_strict')
    
    if not args.dry_run:
        os.makedirs(bad_dir, exist_ok=True)

    # 1. 扫描文件
    print(f"正在扫描目录: {src_dir} ...")
    files = []
    for ext in ['*.mp3', '*.wav']:
        files.extend(glob.glob(os.path.join(src_dir, '**', ext), recursive=True))
    print(f"找到 {len(files)} 个音频文件。开始严格清洗...")

    bad_count = 0
    
    # 2. 并行处理
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in files}
        
        with tqdm(total=len(files), desc="Strict Cleaning") as pbar:
            for future in as_completed(futures):
                fp, is_valid, msg = future.result()
                
                if not is_valid:
                    bad_count += 1
                    fname = os.path.basename(fp)
                    logging.warning(f"损坏: {fname} -> {msg}")
                    
                    if not args.dry_run:
                        try:
                            dst = os.path.join(bad_dir, fname)
                            if os.path.exists(dst): # 防止重名
                                import uuid
                                dst = os.path.join(bad_dir, f"{uuid.uuid4().hex[:4]}_{fname}")
                            shutil.move(fp, dst)
                        except Exception as e:
                            logging.error(f"移动失败: {e}")
                
                pbar.update(1)

    print("="*40)
    print(f"清洗完成。共发现损坏文件: {bad_count} 个")
    print(f"损坏文件已移至: {bad_dir}")
    print("="*40)

if __name__ == "__main__":
    main()