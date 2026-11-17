from __future__ import annotations

import time
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from common import Result, base_parser, build_stages, find_images, iter_batches, report, save_results, to_results

STOP = None

Item = tuple[list[str], np.ndarray] | None


def loader_worker(paths: list[Path], batch_size: int, q_out: Queue[Item], ready: Queue[str], go: Event) -> None:
    """Load images and send them to the first stage."""
    ready.put("loader")
    go.wait()
    for batch, names in iter_batches(paths, batch_size):
        q_out.put((names, batch.numpy()))
    q_out.put(STOP)


def stage_worker(module: nn.Module, q_in: Queue[Item], q_out: Queue[Item], ready: Queue[str]) -> None:
    """Run one part of the model on everything from q_in and pass results to q_out."""
    torch.set_num_threads(1)
    ready.put("stage")
    with torch.no_grad():
        while True:
            item = q_in.get()
            if item is STOP:
                q_out.put(STOP)
                break
            names, x = item
            q_out.put((names, module(torch.from_numpy(x)).numpy()))


def main() -> None:
    """Start the loader and 4 stage processes, collect and save results."""
    p = base_parser("Parallel (pipelined) ResNet-50 inference.")
    args = p.parse_args()

    paths = find_images(args.image_dir, args.limit)
    stages, desc = build_stages()
    for i, d in enumerate(desc, 1):
        print(f"  stage {i}: {d}")

    ctx = mp.get_context("spawn")
    queues: list[Queue[Item]] = [ctx.Queue(maxsize=4) for _ in range(len(stages) + 1)]
    ready: Queue[str] = ctx.Queue()
    go: Event = ctx.Event()
    procs: list[SpawnProcess] = [ctx.Process(target=loader_worker, args=(paths, args.batch_size, queues[0], ready, go))]
    procs += [ctx.Process(target=stage_worker, args=(stage, queues[i], queues[i + 1], ready))
              for i, stage in enumerate(stages)]
    for pr in procs:
        pr.start()
    for _ in procs:
        ready.get()
    print(f"[parallel] {len(paths)} images, micro-batch {args.batch_size}, "
          f"1 loader + {len(stages)} stage processes")

    results: list[Result] = []
    t0 = time.perf_counter()
    go.set()
    while True:
        item = queues[-1].get()
        if item is STOP:
            break
        names, logits = item
        results.extend(to_results(torch.from_numpy(logits), names, args.topk))
        print(f"  {len(results)}/{len(paths)}", end="\r", flush=True)
    elapsed = time.perf_counter() - t0
    print()

    for pr in procs:
        pr.join()

    save_results(results, args.output, args.topk)
    report("parallel", len(results), elapsed, args.output)


if __name__ == "__main__":
    main()
