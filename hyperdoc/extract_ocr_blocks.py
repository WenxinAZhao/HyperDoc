#!/usr/bin/env python3
from __future__ import annotations
"""
OCR Block Extraction
====================
Extract OCR blocks from PDF pages and save to JSON.
This is the first stage of the HyperGraph pipeline.

Usage:
    python scripts/extract_HyperDoc_ocr.py \
        --pdf-path "data/MMLongBench/documents/PH_2016.06.08_Economy-Final.pdf" \
        --img-dir "tmp/MMLongBench" \
        --output-dir ocr_results/MMLongBench
        
    # Batch process all PDFs
    python scripts/extract_HyperDoc_ocr.py \
        --pdf-dir "data/MMLongBench/documents" \
        --img-dir "tmp/MMLongBench" \
        --output-dir ocr_results
"""

import argparse
import json
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

CAPTION_PATTERNS = [
    (r'^\s*Figure\s+\d+[\.\:]', 'figure_caption'),
    (r'^\s*FIGURE\s+\d+[\.\:]', 'figure_caption'),
    (r'^\s*Fig\.\s*\d+[\.\:]', 'figure_caption'),
    (r'^\s*Table\s+\d+[\.\:]', 'table_caption'),
    (r'^\s*TABLE\s+\d+[\.\:]', 'table_caption'),
    (r'^\s*Chart\s+\d+[\.\:]', 'chart_caption'),
    (r'^\s*Map\s+\d+[\.\:]', 'map_caption'),
    (r'^\s*Diagram\s+\d+[\.\:]', 'diagram_caption'),
    (r'^\s*Graph\s+\d+[\.\:]', 'graph_caption'),
]

def refine_block_type(text: str, current_type: str) -> str:
    """Refine block type based on text content (e.g., detect captions)."""
    if not text or current_type not in ('text', 'unknown', 'caption'):
        return current_type
    
    # Check first 50 chars for caption patterns
    text_start = text[:50]
    for pattern, caption_type in CAPTION_PATTERNS:
        if re.match(pattern, text_start, re.IGNORECASE):
            return caption_type
            
    return current_type


def get_page_image_paths(
    pdf_path: str,
    max_pages: Optional[int] = None,
    start_page: int = 0,
    end_page: Optional[int] = None,
    img_dir: Optional[str] = None,
    page_ids: Optional[List[int]] = None,
) -> List[str]:
    """Get pre-rendered page image paths."""
    doc_name = Path(pdf_path).stem
    script_dir = Path(__file__).parent
    if img_dir is not None:
        image_dir = Path(img_dir)
    else:
        image_dir = script_dir.parent / "tmp" / "MMLongBench"
    
    # Scan all pre-rendered images
    image_files = sorted(image_dir.glob(f"{doc_name}_*.png"))
    
    if not image_files:
        raise FileNotFoundError(
            f"No pre-rendered images found for '{doc_name}'\n"
            f"Expected location: {image_dir}/{doc_name}_*.png"
        )
    
    # Extract page indices and sort
    page_images = []
    for img_file in image_files:
        stem = img_file.stem
        page_index = int(stem.split('_')[-1])
        page_images.append((page_index, str(img_file)))
    
    page_images.sort(key=lambda x: x[0])
    
    # Apply exact page-id filter first when provided.
    if page_ids is not None:
        allowed_pages = {int(p) for p in page_ids}
        page_images = [(page_idx, img_path) for page_idx, img_path in page_images if page_idx in allowed_pages]

    # Apply page range filters
    if start_page > 0 or end_page is not None:
        filtered = []
        for page_idx, img_path in page_images:
            if page_idx < start_page:
                continue
            if end_page is not None and page_idx >= end_page:
                break
            filtered.append((page_idx, img_path))
        page_images = filtered
    
    # Apply max_pages limit
    if max_pages:
        page_images = page_images[:max_pages]
    
    return [img_path for _, img_path in page_images]


def extract_ocr_blocks(
    image_paths: List[str],
    detector: BlockDetector
) -> List[Dict[str, Any]]:
    """
    Extract blocks from all pages using BlockDetector.
    
    Returns:
        List of blocks with complete OCR information
    """
    print("\n" + "="*80)
    print("OCR Block Extraction")
    print("="*80)
    
    all_blocks = []
    block_id_counter = 0
    
    for page_idx, image_path in enumerate(image_paths):
        print(f"\n📄 Processing page {page_idx+1}/{len(image_paths)}")
        
        t0 = time.time()
        raw_results, img = detector.detect_blocks(image_path)
        t1 = time.time()
        
        print(f"   ✓ Detected {len(raw_results)} blocks in {t1-t0:.2f}s")
        
        # Normalize to unified format
        for raw_block in raw_results:
            text = raw_block.get("text", "")
            raw_type = raw_block.get("type", "unknown")
            refined_type = refine_block_type(text, raw_type)
            
            block = {
                "block_id": block_id_counter,
                "page": page_idx,
                "bbox": raw_block.get("bbox", [0, 0, 0, 0]),
                "type": refined_type,
                "text": text,
                "rec_conf": raw_block.get("rec_conf", 0.0),
            }
            all_blocks.append(block)
            block_id_counter += 1
    
    print(f"\n✓ Total blocks extracted: {len(all_blocks)}")
    
    # Statistics
    type_counts = defaultdict(int)
    for block in all_blocks:
        type_counts[block["type"]] += 1
    
    print("\nBlock type distribution:")
    for block_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {block_type:15s}: {count:4d}")
    
    return all_blocks


def extract_semantic_text_markers(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract semantic text markers from blocks:
    - Section titles (type=title)
    - Figure captions (type=figure_caption)
    - All text blocks (type=text)
    
    These can be used for text-based retrieval and semantic understanding.
    """
    markers = {
        "section_titles": [],
        "figure_captions": [],
        "text_blocks": []
    }
    
    for block in blocks:
        block_type = block.get("type", "")
        text = block.get("text", "").strip()
        
        if not text:
            continue
        
        marker_entry = {
            "block_id": block["block_id"],
            "page": block["page"],
            "text": text,
            "bbox": block["bbox"]
        }
        
        if block_type == "title":
            markers["section_titles"].append(marker_entry)
        elif block_type == "figure_caption":
            markers["figure_captions"].append(marker_entry)
        elif block_type == "text":
            markers["text_blocks"].append(marker_entry)
    
    return markers


def save_ocr_results(
    pdf_path: str,
    blocks: List[Dict[str, Any]],
    output_dir: Path,
    max_pages: Optional[int] = None
) -> Path:
    """
    Save OCR results to JSON file.
    
    Format:
    {
        "document_id": "...",
        "pdf_path": "...",
        "total_pages": int,
        "blocks": [...],
        "semantic_markers": {
            "section_titles": [...],
            "figure_captions": [...],
            "text_blocks": [...]
        },
        "statistics": {...},
        "created_at": "..."
    }
    """
    doc_name = Path(pdf_path).stem
    output_path = output_dir / f"{doc_name}_ocr.json"
    
    # Extract semantic markers
    semantic_markers = extract_semantic_text_markers(blocks)
    
    # Calculate statistics
    type_counts = defaultdict(int)
    for block in blocks:
        type_counts[block["type"]] += 1
    
    total_pages = max(block["page"] for block in blocks) + 1 if blocks else 0
    
    stats = {
        "total_blocks": len(blocks),
        "total_pages": total_pages,
        "block_types": dict(type_counts),
        "num_section_titles": len(semantic_markers["section_titles"]),
        "num_figure_captions": len(semantic_markers["figure_captions"]),
        "num_text_blocks": len(semantic_markers["text_blocks"]),
    }
    
    ocr_data = {
        "document_id": doc_name,
        "pdf_path": str(pdf_path),
        "total_pages": total_pages,
        "max_pages_processed": max_pages,
        "blocks": blocks,
        "semantic_markers": semantic_markers,
        "statistics": stats,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ OCR results saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\n📊 Statistics:")
    print(f"   Total blocks: {stats['total_blocks']}")
    print(f"   Total pages: {stats['total_pages']}")
    print(f"   Section titles: {stats['num_section_titles']}")
    print(f"   Figure captions: {stats['num_figure_captions']}")
    print(f"   Text blocks: {stats['num_text_blocks']}")
    
    return output_path


def process_single_pdf(
    pdf_path: str,
    output_dir: Path,
    max_pages: Optional[int] = None,
    start_page: int = 0,
    end_page: Optional[int] = None,
    detector: Optional[BlockDetector] = None,
    skip_existing: bool = True,
    img_dir: Optional[str] = None,
    page_ids: Optional[List[int]] = None,
) -> Path:
    """Process a single PDF file."""
    doc_name = Path(pdf_path).stem
    output_path = output_dir / f"{doc_name}_ocr.json"
    
    # Check if output already exists
    if skip_existing and output_path.exists():
        print("="*80)
        print(f"⏭️  Skipping (already exists): {pdf_path}")
        print(f"   Output file: {output_path}")
        print("="*80)
        return output_path
    
    print("="*80)
    print(f"Processing: {pdf_path}")
    if start_page > 0 or end_page is not None:
        print(f"Page range: {start_page} - {end_page if end_page else 'end'}")
    if page_ids is not None:
        print(f"Exact page_ids filter: {sorted(page_ids)}")
    print("="*80)
    
    # Get page images
    try:
        image_paths = get_page_image_paths(
            pdf_path,
            max_pages,
            start_page,
            end_page,
            img_dir=img_dir,
            page_ids=page_ids,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

    
    print(f"✓ Found {len(image_paths)} page images")
    
    # Create detector if not provided
    if detector is None:
        from .blocker import BlockDetector

        detector = BlockDetector(ocr_verbose=False)
    
    # Extract OCR blocks
    blocks = extract_ocr_blocks(image_paths, detector)
    
    # Save results
    output_path = save_ocr_results(pdf_path, blocks, output_dir, max_pages)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract OCR blocks from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single PDF
    python scripts/extract_HyperDoc_ocr.py \\
        --pdf-path "data/MMLongBench/documents/PH_2016.06.08_Economy-Final.pdf" \\
        --output-dir ocr_results
    
    # Process all PDFs in a directory
    python scripts/extract_HyperDoc_ocr.py \\
        --pdf-dir "data/MMLongBench/documents" \\
        --output-dir ocr_results \\
        --max-pages 10
        """
    )
    
    parser.add_argument(
        "--pdf-path",
        help="Path to single PDF document"
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing multiple PDF documents"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ocr_results",
        help="Output directory for OCR results (default: ocr_results)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to process per PDF (default: all pages)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Start page index (0-based, default: 0)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="End page index (0-based, exclusive, default: all pages)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip PDFs that already have OCR results (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if OCR results exist (overrides --skip-existing)"
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default=None,
        help="Directory containing pre-rendered page images (default: tmp/MMLongBench)"
    )
    parser.add_argument(
        "--page-ids",
        type=str,
        default=None,
        help="Exact 0-based page ids to process, comma-separated (e.g. 0,1,4,9). "
             "Optional; when omitted, process all pages."
    )
    
    args = parser.parse_args()
    
    if not args.pdf_path and not args.pdf_dir:
        parser.error("Either --pdf-path or --pdf-dir must be specified")
    
    output_dir = Path(args.output_dir)
    
    # Determine skip_existing behavior
    skip_existing = args.skip_existing and not args.force
    page_ids = None
    if args.page_ids:
        page_ids = [int(p.strip()) for p in args.page_ids.split(",") if p.strip()]
    
    # Process single PDF
    if args.pdf_path:
        process_single_pdf(
            args.pdf_path, 
            output_dir, 
            args.max_pages,
            args.start_page,
            args.end_page,
            skip_existing=skip_existing,
            img_dir=args.img_dir,
            page_ids=page_ids,
        )
    
    # Process directory of PDFs
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            return
        
        print(f"\n📁 Found {len(pdf_files)} total PDF files")
        
        # Pre-scan to find files that need processing
        if skip_existing:
            print("🔍 Scanning for missing OCR files...")
            files_to_process = []
            skipped_files = []
            
            for pdf_path in pdf_files:
                doc_name = pdf_path.stem
                output_path = output_dir / f"{doc_name}_ocr.json"
                if output_path.exists():
                    skipped_files.append(pdf_path)
                else:
                    files_to_process.append(pdf_path)
            
            print(f"   ✓ Already have OCR: {len(skipped_files)} files")
            print(f"   📝 Need to process: {len(files_to_process)} files")
            
            if not files_to_process:
                print("\n✅ All files already have OCR results!")
                return
        else:
            files_to_process = pdf_files
            print(f"   📝 Will process all {len(files_to_process)} files (--force mode)")
        
        # Create single detector for all PDFs
        print("\n🤖 Loading BlockDetector...")
        from .blocker import BlockDetector

        detector = BlockDetector(ocr_verbose=False)
        print("   ✓ BlockDetector ready")
        
        print(f"\n{'='*80}")
        print(f"Starting OCR extraction for {len(files_to_process)} PDFs")
        print(f"{'='*80}\n")
        
        success_count = 0
        failed_count = 0
        
        # Process with tqdm progress bar
        for pdf_path in tqdm(files_to_process, desc="Processing PDFs", unit="file"):
            try:
                result = process_single_pdf(
                    str(pdf_path), 
                    output_dir, 
                    args.max_pages,
                    args.start_page,
                    args.end_page,
                    detector,
                    skip_existing=False,  # Already filtered, so don't skip in function
                    img_dir=args.img_dir,
                    page_ids=page_ids,
                )
                if result:
                    success_count += 1
            except Exception as e:
                tqdm.write(f"❌ Error processing {pdf_path.name}: {e}")
                failed_count += 1
                continue
        
        print("\n" + "="*80)
        print(f"✅ Batch processing complete!")
        print(f"   Successfully processed: {success_count}/{len(files_to_process)} PDFs")
        if failed_count > 0:
            print(f"   Failed: {failed_count} PDFs")
        if skip_existing and skipped_files:
            print(f"   Skipped (already exist): {len(skipped_files)} PDFs")
        print(f"   Output directory: {output_dir}")
        print("="*80)


if __name__ == "__main__":
    main()
