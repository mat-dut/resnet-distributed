import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models

IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
WEIGHTS = models.ResNet50_Weights.DEFAULT
CATEGORIES: list[str] = WEIGHTS.meta["categories"]

Result = dict[str, Any]


def base_parser(description: str) -> argparse.ArgumentParser:
    """Command-line arguments shared by all apps."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("image_dir", type=Path, help="Folder with images (searched recursively)")
    p.add_argument("-o", "--output", type=Path, default=Path("results.csv"), help="Output file (.csv or .json)")
    p.add_argument("--batch-size", type=int, default=8, help="Images per batch / micro-batch")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    return p


def find_images(image_dir: Path, limit: int | None = None) -> list[Path]:
    """List image files in a folder, sorted."""
    paths = sorted(f for f in Path(image_dir).rglob("*") if f.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {image_dir}")
    return paths[:limit] if limit else paths


def iter_batches(paths: list[Path], batch_size: int) -> Iterator[tuple[torch.Tensor, list[str]]]:
    """Load and preprocess images, yield (batch tensor, paths)."""
    transform = WEIGHTS.transforms()
    imgs: list[torch.Tensor] = []
    names: list[str] = []
    for path in paths:
        try:
            imgs.append(transform(Image.open(path).convert("RGB")))
            names.append(str(path))
        except OSError as e:
            print(f"warning: skipping {path}: {e}")
            continue
        if len(imgs) == batch_size:
            yield torch.stack(imgs), names
            imgs, names = [], []
    if imgs:
        yield torch.stack(imgs), names


def load_model() -> models.ResNet:
    """Pretrained ResNet-50 in eval mode."""
    return models.resnet50(weights=WEIGHTS).eval()


def model_blocks(model: models.ResNet) -> list[tuple[str, nn.Module]]:
    """ResNet-50 as a list of 18 named blocks."""
    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    head = nn.Sequential(model.avgpool, nn.Flatten(1), model.fc)
    blocks: list[tuple[str, nn.Module]] = [("stem", stem)]
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        for i, blk in enumerate(getattr(model, lname)):
            blocks.append((f"{lname}.{i}", blk))
    blocks.append(("head", head))
    return blocks


STAGES: list[list[str]] = [
    ["stem", "layer1.0"],
    ["layer1.1", "layer1.2", "layer2.0", "layer2.1"],
    ["layer2.2", "layer2.3", "layer3.0", "layer3.1", "layer3.2", "layer3.3"],
    ["layer3.4", "layer3.5", "layer4.0", "layer4.1", "layer4.2", "head"],
]


def build_stages() -> tuple[list[nn.Sequential], list[str]]:
    """Split ResNet-50 into the stages from STAGES."""
    blocks = dict(model_blocks(load_model()))
    stages = [nn.Sequential(*(blocks[name] for name in names)).eval() for names in STAGES]
    desc = [f"{names[0]} .. {names[-1]}  ({len(names)} blocks)" for names in STAGES]
    return stages, desc


def to_results(logits: torch.Tensor, names: list[str], k: int) -> list[Result]:
    """Top-k labels and probabilities for each image."""
    probs = logits.softmax(dim=1)
    top_p, top_i = probs.topk(k, dim=1)
    return [
        {"image": name,
         "predictions": [{"label": CATEGORIES[i], "class_id": i, "prob": round(p, 5)}
                         for p, i in zip(ps, idxs)]}
        for name, ps, idxs in zip(names, top_p.tolist(), top_i.tolist())
    ]


def save_results(results: list[Result], output: Path, k: int) -> None:
    """Write results to a .csv or .json file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(results, indent=2))
        return
    with output.open("w", newline="") as f:
        w = csv.writer(f)
        header: list[str] = ["image"]
        for i in range(1, k + 1):
            header += [f"top{i}_label", f"top{i}_id", f"top{i}_prob"]
        w.writerow(header)
        for r in results:
            row: list[Any] = [r["image"]]
            for pred in r["predictions"]:
                row += [pred["label"], pred["class_id"], pred["prob"]]
            w.writerow(row)


def report(name: str, n_images: int, seconds: float, output: Path) -> None:
    """Print time and throughput."""
    print(f"[{name}] {n_images} images in {seconds:.2f}s  ->  {n_images / seconds:.1f} img/s")
    print(f"[{name}] results saved to {output}")
