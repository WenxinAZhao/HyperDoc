#!/usr/bin/env python3
"""Build HyperDoc hypergraphs directly from OCR JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hyperdoc.extract_hypergraph_from_ocr import HypergraphExtractorFromOCR


def build_one(
    ocr_path: Path,
    output_dir: Path,
    page_image_dir: Path,
    semantic: bool,
) -> Path:
    extractor = HypergraphExtractorFromOCR(
        output_dir=str(output_dir),
        page_image_dir=str(page_image_dir),
    )
    return extractor.write_document_hypergraph(str(ocr_path), semantic=semantic)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HyperDoc hypergraphs from OCR JSON.")
    parser.add_argument("--ocr-path", type=Path)
    parser.add_argument("--ocr-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-image-dir", type=Path, required=True)
    parser.add_argument("--doc-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()

    if not args.ocr_path and not args.ocr_dir:
        parser.error("Provide --ocr-path or --ocr-dir.")

    if args.ocr_path:
        files = [args.ocr_path]
    else:
        files = sorted(args.ocr_dir.glob("*_ocr.json"))
        if args.doc_id:
            files = [p for p in files if p.name == f"{args.doc_id}_ocr.json"]
        if args.limit is not None:
            files = files[: args.limit]

    if not files:
        raise FileNotFoundError("No OCR JSON files matched the requested input.")

    for path in files:
        out_path = build_one(
            path,
            args.output_dir,
            args.page_image_dir,
            semantic=not args.no_semantic,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
