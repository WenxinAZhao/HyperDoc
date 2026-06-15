#!/usr/bin/env python3
"""Run ColPali page retrieval and write HyperDoc retrieval fields."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_samples(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def clean_doc_id(doc_id: str) -> str:
    return re.sub(r"\.pdf$", "", Path(doc_id).name)


def collect_page_images(page_image_dir: Path, doc_id: str) -> List[Path]:
    prefix = clean_doc_id(doc_id)
    paths = sorted(
        page_image_dir.glob(f"{prefix}_*.png"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No rendered page images found for {doc_id} under {page_image_dir}")
    return paths


def load_image(path: Path) -> Image.Image:
    return Image.open(path)


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if device.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{device}")
    return torch.device("cpu")


def encode_document(
    model,
    processor,
    device: torch.device,
    page_paths: Sequence[Path],
    batch_size: int,
) -> torch.Tensor:
    images = [load_image(path) for path in page_paths]
    loader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: processor.process_images(batch),
    )
    embeds = []
    for batch in loader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            embeds.extend(model(**batch))
    return torch.stack(embeds, dim=0)


def encode_query(
    model,
    processor,
    device: torch.device,
    query: str,
) -> torch.Tensor:
    inputs = processor.process_queries([query]).to(device)
    with torch.no_grad():
        return model(**inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperDoc ColPali page retrieval.")
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument("--page-image-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model-path", default=os.environ.get("HYPERDOC_COLPALI_MODEL"))
    parser.add_argument("--cache-file", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--doc-key", default="doc_id")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    if not args.model_path:
        parser.error("Provide --model-path or set HYPERDOC_COLPALI_MODEL.")
    model_path = args.model_path
    device = resolve_device(args.device)
    samples = load_samples(args.samples_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_file:
        args.cache_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ColPali model from {model_path}")
    from colpali_engine.models import ColPali, ColPaliProcessor

    model = ColPali.from_pretrained(model_path, dtype=torch.bfloat16, device_map=None).eval().to(device)
    processor = ColPaliProcessor.from_pretrained(model_path)

    if args.cache_file and args.cache_file.exists() and not args.force_cache:
        with args.cache_file.open("rb") as f:
            document_cache: Dict[str, Tuple[List[str], torch.Tensor]] = pickle.load(f)
    else:
        document_cache = {}

    doc_ids = sorted({sample[args.doc_key] for sample in samples})
    for doc_id in tqdm(doc_ids, desc="Encoding documents"):
        if doc_id in document_cache:
            continue
        page_paths = collect_page_images(args.page_image_dir, doc_id)
        document_cache[doc_id] = (
            [str(path) for path in page_paths],
            encode_document(model, processor, device, page_paths, args.batch_size).cpu(),
        )

    if args.cache_file:
        with args.cache_file.open("wb") as f:
            pickle.dump(document_cache, f)

    image_key = f"image-top-{args.top_k}-{args.question_key}"
    score_key = f"{image_key}_score"
    output_samples = []
    for sample in tqdm(samples, desc="Retrieving pages"):
        row = dict(sample)
        doc_id = row[args.doc_key]
        page_paths, document_embed = document_cache[doc_id]
        query_embed = encode_query(model, processor, device, row[args.question_key])
        document_embed = document_embed.to(device=device, dtype=query_embed.dtype)
        scores = processor.score_multi_vector(query_embed, document_embed)
        top = torch.topk(scores, min(args.top_k, scores.shape[-1]), dim=-1)
        row[image_key] = top.indices.tolist()[0]
        row[score_key] = top.values.tolist()[0]
        output_samples.append(row)

    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
