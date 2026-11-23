"""Concurrent FineWeb GPT-2 shard downloader (from kjj0/fineweb10B-gpt2).

Each training shard contains 100M GPT-2 uint16 tokens (~200 MB).
Examples:
  # Default local setup (val + 1 train shard = 100M train tokens):
  python download_data.py

  # Cloud 125M x 2.5B tokens (val + 25 train shards = 5.0 GB) with 8 concurrent workers:
  python download_data.py --shards 25 --workers 8 --data-dir ./data

  # Domestic mirror with 12 workers:
  HF_ENDPOINT=https://hf-mirror.com python download_data.py --shards 25 --workers 12
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data")
CURL = "curl.exe" if os.name == "nt" else "curl"
MIN_SHARD_BYTES = 100_000_000  # Each valid 100M-token shard is ~200 MB
LOCK = threading.Lock()


def get_endpoints(preferred: str = ""):
    eps = [
        preferred.strip().rstrip("/"),
        os.environ.get("HF_ENDPOINT", "").strip().rstrip("/"),
        "https://huggingface.co",
        "https://hf-mirror.com",
    ]
    seen = []
    for e in eps:
        if e and e not in seen:
            seen.append(e)
    return seen


def is_valid_shard(path: str) -> bool:
    if not os.path.exists(path):
        return False
    sz = os.path.getsize(path)
    if sz < MIN_SHARD_BYTES or (sz - 1024) % 2 != 0:
        return False
    return True


def download_one(fname: str, data_dir: str, endpoints: list, retries: int = 3) -> tuple:
    dst = os.path.join(data_dir, fname)
    part = dst + ".part"
    if is_valid_shard(dst):
        return fname, "skip", os.path.getsize(dst), 0.0

    if os.path.exists(part):
        try:
            os.remove(part)
        except OSError:
            pass

    t0 = time.time()
    last_err = ""
    for base in endpoints:
        url = f"{base}/datasets/kjj0/fineweb10B-gpt2/resolve/main/{fname}"
        with LOCK:
            print(f"[start] {fname} <- {base}", flush=True)
        cmd = [
            CURL, "-fL", "-sS",
            "--retry", str(retries),
            "--retry-delay", "2",
            "--connect-timeout", "15",
            url, "-o", part,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and is_valid_shard(part):
            os.replace(part, dst)
            dt = time.time() - t0
            sz = os.path.getsize(dst)
            mb_s = (sz / 1e6) / max(1e-3, dt)
            with LOCK:
                print(f"[done]  {fname} ({sz / 1e6:.1f} MB in {dt:.1f}s, {mb_s:.1f} MB/s)", flush=True)
            return fname, "ok", sz, dt
        last_err = (res.stderr or "").strip()
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
    raise RuntimeError(f"Failed to download {fname}: {last_err}")


def build_counts(data_dir: str, count_shards: int = 1, force: bool = False):
    counts_path = os.path.join(data_dir, "counts.npy")
    if os.path.exists(counts_path) and not force:
        print(f"[skip] {counts_path} already exists (pass --rebuild-counts to overwrite)")
        return
    counts = np.zeros(65536, dtype=np.int64)
    used = 0
    for idx in range(1, count_shards + 1):
        p = os.path.join(data_dir, f"fineweb_train_{idx:06d}.bin")
        if is_valid_shard(p):
            tr = np.memmap(p, dtype=np.uint16, mode="r", offset=1024)
            counts += np.bincount(np.asarray(tr, dtype=np.int64), minlength=65536)
            used += 1
    if used == 0:
        raise FileNotFoundError(f"No valid training shards found in {data_dir} to build counts.npy")
    np.save(counts_path, counts)
    print(f"[ok] Saved {counts_path} over {used} shard(s) (total tokens: {counts.sum():,})")


def verify_all(data_dir: str, files: list):
    ok_n, tot_tok = 0, 0
    for fname in files:
        p = os.path.join(data_dir, fname)
        if not is_valid_shard(p):
            print(f"[MISSING/BAD] {fname}")
            continue
        mm = np.memmap(p, dtype=np.uint16, mode="r", offset=1024)
        max_id = int(mm[:min(100000, len(mm))].max())
        assert max_id < 50257, f"Unexpected token ID {max_id} in {fname}"
        ok_n += 1
        if fname.startswith("fineweb_train_"):
            tot_tok += len(mm)
    print(f"[verify] {ok_n}/{len(files)} shards valid | Train tokens ready: {tot_tok / 1e9:.3f}B ({tot_tok:,})")


def main():
    p = argparse.ArgumentParser(description="Concurrent FineWeb10B-GPT2 shard downloader")
    p.add_argument("--shards", type=int, default=1, help="Number of 100M-token train shards (1..103; 25 = 2.5B tokens)")
    p.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Output directory for .bin shards")
    p.add_argument("--endpoint", default="", help="Preferred HF base URL (e.g. https://hf-mirror.com)")
    p.add_argument("--retries", type=int, default=3, help="Retries per endpoint per shard")
    p.add_argument("--counts-shards", type=int, default=1, help="How many train shards to aggregate into counts.npy (0 to skip)")
    p.add_argument("--rebuild-counts", action="store_true", help="Force rebuild counts.npy")
    p.add_argument("--verify", action="store_true", help="Verify shard sizes and token IDs after completion")
    args = p.parse_args()

    if not (1 <= args.shards <= 103):
        raise ValueError("--shards must be between 1 and 103")

    os.makedirs(args.data_dir, exist_ok=True)
    endpoints = get_endpoints(args.endpoint)
    files = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, args.shards + 1)]

    print(f"Downloading {len(files)} file(s) -> {args.data_dir} | workers={args.workers} | endpoints={endpoints}")
    t_start = time.time()
    downloaded_bytes = 0
    skipped = 0
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, fn, args.data_dir, endpoints, args.retries): fn for fn in files}
        for fut in concurrent.futures.as_completed(futs):
            fn = futs[fut]
            try:
                _, status, sz, _ = fut.result()
                if status == "skip":
                    skipped += 1
                    print(f"[skip]  {fn} ({sz / 1e6:.1f} MB)", flush=True)
                else:
                    downloaded_bytes += sz
            except Exception as e:
                failed.append((fn, str(e)))
                print(f"[FAIL]  {fn}: {e}", file=sys.stderr, flush=True)

    wall = time.time() - t_start
    if downloaded_bytes > 0:
        print(f"Downloaded {downloaded_bytes / 1e9:.2f} GB in {wall:.1f}s (aggregate {downloaded_bytes / 1e6 / max(1e-3, wall):.1f} MB/s); skipped {skipped}.")
    else:
        print(f"All {skipped} shard(s) already present and valid.")

    if failed:
        print(f"ERROR: {len(failed)} shard(s) failed!", file=sys.stderr)
        sys.exit(1)

    if args.counts_shards > 0:
        build_counts(args.data_dir, count_shards=min(args.shards, args.counts_shards), force=args.rebuild_counts)

    if args.verify:
        verify_all(args.data_dir, files)


if __name__ == "__main__":
    main()
