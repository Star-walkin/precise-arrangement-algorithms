"""Resumable verified download of the one official frozen TAPNext++ object."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

URL = "https://storage.googleapis.com/download/storage/v1/b/dm-tapnet/o/tapnextpp%2Ftapnextpp_ckpt.pt?alt=media&generation=1774639633910846"
SIZE = 2532282370
MD5_BASE64 = "pjiOC5EfLjZE+vqKmhGN+w=="


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget-seconds", type=int, default=1200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "weights"
    parts = root / "official_range_parts"
    parts.mkdir(parents=True, exist_ok=True)
    part_size = 8 * 1024 * 1024
    starts = list(range(0, SIZE, part_size))
    deadline = time.monotonic() + args.budget_seconds
    def fetch(start):
        end = min(SIZE, start + part_size) - 1
        size = end - start + 1
        path = parts / f"{start:012d}.part"
        if path.exists() and path.stat().st_size == size:
            return start, "reused"
        for attempt in range(3):
            if time.monotonic() > deadline:
                raise TimeoutError("bounded download budget exhausted")
            offset = path.stat().st_size if path.exists() else 0
            if offset > size:
                raise ValueError("invalid existing part size")
            req = Request(URL, headers={"Range": f"bytes={start+offset}-{end}"})
            try:
                with urlopen(req, timeout=30) as response:
                    expected = f"bytes {start+offset}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"range response mismatch: {response.status}, {response.headers}")
                    with path.open("ab") as stream:
                        while chunk := response.read(min(128*1024, size - offset)):
                            stream.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() > deadline:
                                raise TimeoutError("bounded download budget exhausted")
                if path.stat().st_size == size:
                    return start, "downloaded"
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        raise RuntimeError("incomplete range")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(fetch, start): start for start in starts}
        for number, future in enumerate(as_completed(future_map), 1):
            try:
                future.result()
            except Exception as error:
                failures.append({"start": future_map[future], "error": str(error)})
            if number % 10 == 0:
                print(json.dumps({"parts_completed": number, "total": len(starts), "failures": len(failures)}), flush=True)
    if failures:
        print(json.dumps({"status": "incomplete", "failures": failures}, ensure_ascii=False), flush=True)
        raise SystemExit(2)
    final = root / "tapnextpp_verified.pt"
    if final.exists():
        raise FileExistsError(final)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with final.open("xb") as out:
        for start in starts:
            data = (parts / f"{start:012d}.part").read_bytes()
            out.write(data)
            md5.update(data)
            sha.update(data)
    actual_md5 = base64.b64encode(md5.digest()).decode()
    if actual_md5 != MD5_BASE64:
        raise ValueError(f"Google object MD5 mismatch: {actual_md5}")
    print(json.dumps({"status": "verified", "path": str(final), "bytes": SIZE,
                      "sha256": sha.hexdigest(), "md5_base64": actual_md5, "url": URL}), flush=True)


if __name__ == "__main__":
    main()
