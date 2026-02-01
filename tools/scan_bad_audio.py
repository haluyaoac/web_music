import os
import shutil
import glob
import logging
import sys
import subprocess
import tempfile
import contextlib
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import exp_cfg as cfg

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cleaning.log", encoding='utf-8'), # 加上 encoding 防止乱码
            logging.StreamHandler()
        ]
    )

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

FFMPEG_PATH = shutil.which("ffmpeg")

# --- 新增：上下文管理器，用于捕获底层 C 库的 stderr 输出 ---
@contextlib.contextmanager
def capture_c_stderr():
    """
    捕获 C 语言层面的 stderr 输出 (如 libmpg123, libsndfile 等)。
    原理是使用 os.dup2 重定向文件描述符 2 (stderr)。
    """
    # 1. 创建一个临时文件用于接收错误日志
    # 使用 w+b 模式避免编码问题
    temp = tempfile.TemporaryFile(mode='w+b')
    
    # 2. 保存原本的 stderr 文件描述符
    try:
        # 在某些环境(如IDLE或某些IDE) stderr可能没有fileno，这种情况下无法捕获，只能跳过
        orig_stderr = sys.stderr.fileno()
        saved_stderr_fd = os.dup(orig_stderr)
    except Exception:
        # 如果无法获取句柄，就直接 yield 一个空对象，不进行捕获
        yield temp
        temp.close()
        return

    try:
        # 3. 将 stderr (fd 2) 重定向到临时文件
        sys.stderr.flush()
        os.dup2(temp.fileno(), orig_stderr)
        
        yield temp
        
    finally:
        # 4. 恢复 stderr
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, orig_stderr)
        os.close(saved_stderr_fd)
        temp.close()

def ffmpeg_strict_check(file_path):
    """
    使用 ffmpeg 严格解码全文件。
    修改：使用 -v warning 而不是 error，以便捕获 'dequantization failed' 等警告级错误。
    """
    if not FFMPEG_PATH:
        return None, "ffmpeg not found"

    # 改为 warning 级别，更敏感
    cmd = [FFMPEG_PATH, "-v", "warning", "-i", file_path, "-f", "null", "-"]
    
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', # 显式指定编码
            errors='ignore'   # 忽略解码错误
        )
        stderr = (proc.stderr or "").strip()
        
        # 只要有 warning 输出，或者返回码不对，就视为坏文件
        if proc.returncode != 0 or stderr:
            msg_lines = stderr.splitlines()
            # 过滤掉一些无关紧要的 info，只看 warning/error
            # 但既然开了 -v warning，通常出来的都是问题
            msg = "\n".join(msg_lines[:3]) if msg_lines else f"ffmpeg exit {proc.returncode}"
            return False, msg
        return True, "OK"
    except Exception as e:
        return False, f"ffmpeg run error: {e}"

def is_valid_audio(file_path):
    """
    验证音频完整性，包含 ffmpeg 检查和 librosa 加载检查。
    """
    # 1. 先做 ffmpeg 检查 (由 -v error 改为了 -v warning)
    strict_ok, strict_msg = ffmpeg_strict_check(file_path)
    if strict_ok is False:
        return False, f"ffmpeg: {strict_msg}"

    # 2. Librosa 加载检查 (捕获底层报错)
    try:
        with capture_c_stderr() as stderr_file:
            # 尝试加载
            y, sr = librosa.load(file_path, sr=None, mono=True, duration=cfg.DURATION)
            
            # 检查是否有 C 库报错
            stderr_file.seek(0)
            c_errors = stderr_file.read().decode('utf-8', 'ignore').strip()
        
        # 3. 判定逻辑
        # 如果捕获到了错误日志，且包含特定关键字
        error_keywords = ["error", "failed", "corrupt", "too large", "invalid"]
        if c_errors and any(k in c_errors.lower() for k in error_keywords):
            return False, f"Librosa C-Error: {c_errors[:100]}..."

        # 4. 检查数据有效性
        if y is None or len(y) < 100:
            return False, "Empty or too short"
        
        # 检查是否全是静音 (可选，防止解码出纯静音)
        if np.max(np.abs(y)) < 1e-4:
            return False, "Silent audio"

        return True, "OK"
    except Exception as e:
        return False, str(e)

def clean_dataset():
    source_dir = cfg.FMA_AUDIO_DIR
    bad_dir = os.path.join(cfg.DATA_DIR, 'bad')
    
    ensure_dir(bad_dir)
    logging.info(f"开始清洗数据 (增强模式)...")
    logging.info(f"源目录: {source_dir}")
    
    if not FFMPEG_PATH:
        logging.warning("未检测到 ffmpeg!")

    audio_files = []
    for ext in ['*.mp3', '*.wav']:
        audio_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))

    logging.info(f"共发现 {len(audio_files)} 个音频文件。")

    bad_count = 0
    for file_path in tqdm(audio_files, desc="Checking files"):
        valid, message = is_valid_audio(file_path)
        
        if not valid:
            bad_count += 1
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(bad_dir, file_name)
            
            # 处理重名
            if os.path.exists(dest_path):
                base, extension = os.path.splitext(file_name)
                dest_path = os.path.join(bad_dir, f"{base}_dup_{bad_count}{extension}")

            try:
                shutil.move(file_path, dest_path)
                # 使用 debug 级别减少刷屏，或者保留 warning
                # 将 \n 替换为空格以保持日志整洁
                clean_msg = message.replace('\n', ' ')
                logging.warning(f"移除: {file_name} -> {clean_msg}")
            except OSError as e:
                logging.error(f"移动失败 {file_name}: {e}")

    logging.info(f"清洗完成。")
    logging.info(f"损坏/移除: {bad_count}")
    logging.info(f"剩余有效: {len(audio_files) - bad_count}")

if __name__ == "__main__":
    setup_logging()
    clean_dataset()