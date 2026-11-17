from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
from typing import Any

import requests
import torch
from torch import nn

from common import Result, iter_batches, load_model, to_results


def process(model: nn.Module, data_dir: Path, files: list[str], batch_size: int, topk: int) -> list[Result]:
    """Run the model on the given files and return results with names relative to data_dir."""
    results: list[Result] = []
    for batch, names in iter_batches([data_dir / f for f in files], batch_size):
        rel = [Path(n).relative_to(data_dir).as_posix() for n in names]
        results.extend(to_results(model(batch), rel, topk))
    return results


def register(server: str, name: str) -> int:
    """Register with the server, retrying until it answers."""
    while True:
        try:
            r = requests.post(f"{server}/register", json={"name": name}, timeout=10)
            r.raise_for_status()
            return int(r.json()["agent_id"])
        except requests.RequestException as e:
            print(f"[agent] server not reachable ({e}), retrying...")
            time.sleep(3)


def main() -> None:
    """Register, then keep asking the server for files and sending back results."""
    p = argparse.ArgumentParser(description="Distributed ResNet-50: worker agent.")
    p.add_argument("server", help="Server address, e.g. http://1.2.3.4:8000")
    p.add_argument("data_dir", type=Path, help="Local folder with the images")
    p.add_argument("--name", default=socket.gethostname())
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--poll", type=float, default=2.0, help="Seconds between polls when there is no work")
    args = p.parse_args()
    server = args.server.rstrip("/")

    model = load_model()
    agent_id = register(server, args.name)
    print(f"[agent] registered as #{agent_id} ({args.name}), {torch.get_num_threads()} threads, waiting for start")

    run, seconds = 0, 0.0
    results: list[Result] = []
    with torch.no_grad():
        while True:
            payload: dict[str, Any] = {"agent_id": agent_id, "run": run, "results": results, "seconds": seconds}
            try:
                r = requests.post(f"{server}/work", json=payload, timeout=30)
                if r.status_code == 404:
                    agent_id = register(server, args.name)
                    results, seconds = [], 0.0
                    continue
                r.raise_for_status()
                job = r.json()
            except requests.RequestException as e:
                print(f"[agent] connection problem ({e}), retrying...")
                time.sleep(args.poll)
                continue

            results, seconds = [], 0.0
            if not job["files"]:
                time.sleep(args.poll)
                continue

            run = job["run"]
            t0 = time.perf_counter()
            results = process(model, args.data_dir, job["files"], args.batch_size, args.topk)
            seconds = time.perf_counter() - t0
            print(f"[agent] run {run}: {len(results)} images in {seconds:.2f}s ({len(results) / seconds:.1f} img/s)")


if __name__ == "__main__":
    main()
