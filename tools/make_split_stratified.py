import os, glob, json, random, argparse, sys
from collections import defaultdict, Counter
Base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if Base not in sys.path:
    sys.path.insert(0, Base)
import exp_cfg as cfg

def load_label_map(path=cfg.LABEL_MAP_JSON):
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    genres = [label_map[str(i)] for i in range(len(label_map))]
    return genres

def stratified_split(files_by_label, seed=42, train=0.8, val=0.1, test=0.1):
    rnd = random.Random(seed)
    out = []
    split_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for label, files in files_by_label.items():
        files = files[:]  # copy
        rnd.shuffle(files)
        n = len(files)

        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # 保证总和不超过 n
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        for fp in train_files:
            out.append({"path": fp, "label": label, "split": "train"})
            split_counts["train"][label] += 1
        for fp in val_files:
            out.append({"path": fp, "label": label, "split": "val"})
            split_counts["val"][label] += 1
        for fp in test_files:
            out.append({"path": fp, "label": label, "split": "test"})
            split_counts["test"][label] += 1

    rnd.shuffle(out)
    return out, split_counts

def main():
        # ap = argparse.ArgumentParser()
        # ap.add_argument("--root", type=str, default="data/raw_fma8")
        # ap.add_argument("--out", type=str, default="data/splits/split_fma8_v1.json")
        # ap.add_argument("--seed", type=int, default=42)
        # ap.add_argument("--train", type=float, default=0.8)
        # ap.add_argument("--val", type=float, default=0.1)
        # ap.add_argument("--test", type=float, default=0.1)
        # args = ap.parse_args()

    if abs((cfg.SPLIT_RATIO["train"] + cfg.SPLIT_RATIO["val"] + cfg.SPLIT_RATIO["test"]) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    genres = load_label_map(cfg.LABEL_MAP_JSON)

    files_by_label = defaultdict(list)
    for label, g in enumerate(genres):
        files = glob.glob(os.path.join(cfg.FMA_AUDIO_DIR, g, "*"))
        files = [f for f in files if os.path.isfile(f)]
        files_by_label[label].extend(files)

    # 打印每类总数
    print("Per-class totals:")
    for label, g in enumerate(genres):
        print(f"  [{label}] {g}: {len(files_by_label[label])}")

    items, split_counts = stratified_split(
        files_by_label,
        seed=cfg.RANDOM_SEED,
        train=cfg.SPLIT_RATIO["train"],
        val=cfg.SPLIT_RATIO["val"],
        test=cfg.SPLIT_RATIO["test"],
    )

    # 补齐 genre 字段（方便你论文/调试）
    inv = {i: genres[i] for i in range(len(genres))}
    for it in items:
        it["genre"] = inv[it["label"]]

    os.makedirs(os.path.dirname(cfg.SPLIT_JSON), exist_ok=True)
    with open(cfg.SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print("\nSaved:", cfg.SPLIT_JSON, "total:", len(items))
    for sp in ["train", "val", "test"]:
        n_sp = sum(split_counts[sp].values())
        print(f"{sp}: {n_sp}  per-class: {dict(split_counts[sp])}")

if __name__ == "__main__":
    main()
