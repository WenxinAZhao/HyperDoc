#!/usr/bin/env python3
"""Render PDF pages to PNG files using poppler's pdftoppm."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int, max_pages: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = pdf_path.stem
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / doc_id
        cmd = ["pdftoppm", "-png", "-r", str(dpi)]
        if max_pages:
            cmd.extend(["-f", "1", "-l", str(max_pages)])
        cmd.extend([str(pdf_path), str(prefix)])
        subprocess.check_call(cmd)
        rendered = sorted(Path(tmp).glob(f"{doc_id}-*.png"))
        for idx, src in enumerate(rendered):
            dst = output_dir / f"{doc_id}_{idx}.png"
            if not dst.exists():
                src.replace(dst)
        return len(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HyperDoc PDF pages.")
    parser.add_argument("--pdf-path", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    if not args.pdf_path and not args.pdf_dir:
        parser.error("Provide --pdf-path or --pdf-dir.")

    pdfs = [args.pdf_path] if args.pdf_path else sorted(args.pdf_dir.glob("*.pdf"))
    total = 0
    for pdf_path in pdfs:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        n_pages = render_pdf(pdf_path, args.output_dir, args.dpi, args.max_pages)
        total += n_pages
        print(f"{pdf_path.name}: {n_pages} page image(s)")
    print(f"Rendered {total} page image(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
