"""Download FineWeb GPT-2 tokenized shards (from kjj0/fineweb10B-gpt2) via curl and build data/counts.npy."""
import os
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = [
    "fineweb_val_000000.bin",
    "fineweb_train_000001.bin",
]

ENDPOINTS = [
    os.environ.get("HF_ENDPOINT", "").rstrip("/"),
    "https://huggingface.co",
    "https://hf-mirror.com",
]
ENDPOINTS = [e for e in ENDPOINTS if e]
CURL = "curl.exe" if os.name == "nt" else "curl"


def download_file(fname: str) -> str:
    dst = os.path.join(DATA_DIR, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000_000:
        print(f"[skip] {dst} already exists ({os.path.getsize(dst) / 1e6:.1f} MB)")
        return dst

    for base in ENDPOINTS:
        url = f"{base}/datasets/kjj0/fineweb10B-gpt2/resolve/main/{fname}"
        print(f"[download] Trying {url} ...")
        res = subprocess.run([CURL, "-fL", "--retry", "3", url, "-o", dst])
        if res.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
            print(f"[ok] Saved {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
            return dst
        if os.path.exists(dst):
            os.remove(dst)
    raise RuntimeError(f"Could not download {fname} from any endpoint.")


def build_counts(train_bin: str):
    counts_path = os.path.join(DATA_DIR, "counts.npy")
    if os.path.exists(counts_path):
        print(f"[skip] {counts_path} already exists")
        return
    print("[build] Computing token counts from fineweb_train_000001.bin ...")
    tr = np.memmap(train_bin, dtype=np.uint16, mode="r", offset=1024)
    counts = np.bincount(np.asarray(tr, dtype=np.int64), minlength=65536).astype(np.int64)
    np.save(counts_path, counts)
    print(f"[ok] Saved {counts_path} (total tokens: {counts.sum():,})")


if __name__ == "__main__":
    for f in FILES:
        download_file(f)
    build_counts(os.path.join(DATA_DIR, "fineweb_train_000001.bin"))
    print("All data ready in ./data/")
