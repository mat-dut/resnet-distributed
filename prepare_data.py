from __future__ import annotations

import argparse
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/EliSchwartz/imagenet-sample-images/archive/refs/heads/master.zip"


def download(dest: Path) -> None:
    """Download the 1000 ImageNet sample images into dest."""
    print(f"downloading {URL} ...")
    data = urllib.request.urlopen(URL).read()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.filename.endswith(".JPEG"):
                (dest / Path(info.filename).name).write_bytes(z.read(info))


def make_copies(src: Path, dest: Path, copies: int) -> None:
    """Create dest/copy0 .. copyN-1, each with all images from src."""
    for i in range(copies):
        out = dest / f"copy{i}"
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.JPEG")):
            target = out / f.name
            if target.exists():
                continue
            try:
                os.link(f, target)
            except OSError:
                shutil.copy(f, target)


def main() -> None:
    """Prepare data/bench10k (same layout on every machine)."""
    p = argparse.ArgumentParser(description="Download sample images and build the benchmark folder.")
    p.add_argument("--copies", type=int, default=10)
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()

    src = args.data / "imagenet-sample-images"
    if len(list(src.glob("*.JPEG"))) < 1000:
        download(src)
    make_copies(src, args.data / "bench10k", args.copies)
    print(f"ready: {args.data / 'bench10k'} ({args.copies * 1000} images)")


if __name__ == "__main__":
    main()
