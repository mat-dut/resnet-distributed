import time

import torch

from common import Result, base_parser, find_images, iter_batches, load_model, report, save_results, to_results


def main() -> None:
    """Run the whole model on one image at a time in a single thread."""
    p = base_parser("Sequential ResNet-50 inference.")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.set_defaults(batch_size=1)
    args = p.parse_args()

    torch.set_num_threads(1)
    paths = find_images(args.image_dir, args.limit)
    model = load_model().to(args.device)
    print(f"[sequential] {len(paths)} images, batch {args.batch_size}, 1 thread, {args.device}")

    results: list[Result] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch, names in iter_batches(paths, args.batch_size):
            results.extend(to_results(model(batch.to(args.device)).cpu(), names, args.topk))
            print(f"  {len(results)}/{len(paths)}", end="\r", flush=True)
    elapsed = time.perf_counter() - t0
    print()

    save_results(results, args.output, args.topk)
    report("sequential", len(results), elapsed, args.output)


if __name__ == "__main__":
    main()
