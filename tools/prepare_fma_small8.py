import os, shutil, argparse
from pathlib import Path

import pandas as pd

def read_tracks_csv(tracks_csv: Path) -> pd.DataFrame:
    # FMA 的 tracks.csv 是多级表头
    df = pd.read_csv(tracks_csv, header=[0,1], index_col=0)
    return df

def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", type=str, required=True, help="fma_small 解压后的根目录")
    ap.add_argument("--tracks_csv", type=str, required=True, help="fma_metadata 里的 tracks.csv 路径")
    ap.add_argument("--out_root", type=str, default="data/raw_fma8", help="输出到你的项目 raw 目录")
    ap.add_argument("--top_k", type=int, default=8, help="取样本最多的前K个 genre_top")
    ap.add_argument("--mode", type=str, default="copy", choices=["copy","link"], help="copy 或 link(硬链接)")
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    tracks_csv = Path(args.tracks_csv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tracks = read_tracks_csv(tracks_csv)

    # 兼容不同版本列名
    col_subset = get_col(tracks, [( "set", "subset" ), ( "set", "split" )])
    col_genre = get_col(tracks, [( "track", "genre_top" ), ( "track", "genre" )])

    # 只取 small 子集
    small = tracks[tracks[col_subset] == "small"].copy()
    small = small[small[col_genre].notna()]

    # 统计 genre_top，选前 top_k
    counts = small[col_genre].value_counts()
    top_genres = list(counts.head(args.top_k).index)
    print("Selected genres:", top_genres)
    print("Counts:", counts.head(args.top_k).to_dict())

    # 建立 genre -> track_id 集合
    small = small[small[col_genre].isin(top_genres)]
    genre_of = small[col_genre].to_dict()  # key=index(track_id), value=genre

    # 遍历音频文件
    n_ok = 0
    n_skip = 0
    for mp3 in audio_root.rglob("*.mp3"):
        try:
            track_id = int(mp3.stem)  # 文件名就是 track_id
        except:
            n_skip += 1
            continue

        genre = genre_of.get(track_id, None)
        if genre is None:
            n_skip += 1
            continue

        dst_dir = out_root / genre
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / mp3.name

        if dst.exists():
            n_skip += 1
            continue

        if args.mode == "link":
            os.link(mp3, dst)   # 硬链接，不占双份空间（Windows 可能不支持/权限问题）
        else:
            shutil.copy2(mp3, dst)

        n_ok += 1

    print(f"Done. copied/linked: {n_ok}, skipped: {n_skip}")
    print("Output:", out_root)

if __name__ == "__main__":
    main()
