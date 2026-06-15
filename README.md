# HyperDoc

HyperDoc is a hypergraph-based pipeline for long multimodal document QA.

## Environments

Use separate environments for retrieval/preprocessing and reasoning. The
retriever stack may require PaddleOCR, OpenCLIP, and ColPali, while the
reasoner stack loads the VLM.

```bash
pip install -r requirements-retriever.txt
pip install -r requirements-reasoner.txt
pip install -r requirements-eval.txt
```

Set model paths explicitly:

```bash
export HYPERDOC_REASONER_MODEL=/path/to/Qwen3-VL-8B-Instruct
export HYPERDOC_OPENCLIP_MODEL=/path/to/open_clip_pytorch_model.bin
export HYPERDOC_COLPALI_MODEL=/path/to/colpali-v1.3
```

The released configs default to Qwen3-VL. To reproduce other backbone settings
reported in the paper, set `model.type` and `HYPERDOC_REASONER_MODEL` to the
corresponding Qwen2.5-VL or Qwen3-VL checkpoint.

For exact reproduction, use the package versions in the authors' environment;
minor version differences in `transformers`, `qwen-vl-utils`, `paddleocr`, and
`colpali-engine` may affect preprocessing or VLM inference.

PDF rendering uses `pdftoppm` from poppler.

## Dataset

Download the prepared dataset package from Google Drive:

https://drive.google.com/file/d/12IhxegKc6BNyIDN-um0r2LWBDI7AGHGz/view?usp=sharing

After downloading and extracting the package, place the files under the
following layout:

```text
data/
  MMLongBench/
    samples.json
    documents/
      *.pdf
  LongDocURL/
    samples.json
    documents/
      *.pdf
  docbench/
    samples.json
    documents/
      *.pdf
```

The `data/MMLongBench/` directory corresponds to MMLongBench-Doc in the
official release.

This repository includes a tiny raw-format subset under `data/` for smoke
testing. Replace it with the full official datasets for full experiments. Page
images are written under `tmp/<DatasetName>/`, OCR JSON under
`ocr_results/<DatasetName>/`, and hypergraphs under `hypergraph/<DatasetName>/`.

Prepare ColPali page retrieval fields before online answering:

```bash
python scripts/render_HyperDoc_pages.py \
  --pdf-path data/MMLongBench/documents/PH_2016.06.08_Economy-Final.pdf \
  --output-dir tmp/MMLongBench
```

```bash
python scripts/retrieve_HyperDoc_colpali.py \
  --samples-file data/MMLongBench/samples.json \
  --page-image-dir tmp/MMLongBench \
  --output-file data/MMLongBench/sample-with-retrieval-results.json \
  --model-path "$HYPERDOC_COLPALI_MODEL" \
  --top-k 10 \
  --batch-size 2 \
  --device cuda:0 \
  --cache-file tmp/ColPaliRetrieval/MMLongBench_embed.pkl
```

The command writes `data/MMLongBench/sample-with-retrieval-results.json`, which
is the dataset file consumed by the released HyperDoc configs.

## Offline Pipeline

Step 1: render PDF pages.

```bash
python scripts/render_HyperDoc_pages.py \
  --pdf-path data/MMLongBench/documents/PH_2016.06.08_Economy-Final.pdf \
  --output-dir tmp/MMLongBench
```

Step 2: run OCR block extraction.

```bash
python scripts/extract_HyperDoc_ocr.py \
  --pdf-path data/MMLongBench/documents/PH_2016.06.08_Economy-Final.pdf \
  --img-dir tmp/MMLongBench \
  --output-dir ocr_results/MMLongBench
```

Step 3: build hypergraphs from OCR JSON.

```bash
python scripts/build_HyperDoc_hypergraph.py \
  --ocr-path ocr_results/MMLongBench/PH_2016.06.08_Economy-Final_ocr.json \
  --output-dir hypergraph/MMLongBench \
  --page-image-dir tmp/MMLongBench
```

## Dynamic K

Compute the VLM-budget-derived cap after hypergraph construction and before
online answering:

```bash
python scripts/compute_HyperDoc_dynamic_k.py \
  --config configs/HyperDoc_mmlb.yaml \
  --output-config configs/HyperDoc_mmlb.dynamic.yaml \
  --meta-output results/HyperDoc_mmlb.dynamic_k.json
```

The generated config is the input for online answering.

## Online QA

Run HyperDoc:

```bash
python scripts/run_HyperDoc.py \
  --config configs/HyperDoc_mmlb.dynamic.yaml
```

For a short smoke run:

```bash
python scripts/run_HyperDoc.py \
  --config configs/HyperDoc_mmlb.dynamic.yaml \
  --max-samples 2
```

## Evaluation

Run the MMLongBench evaluator on raw HyperDoc outputs:

```bash
python scripts/evaluate_HyperDoc.py \
  --dataset mmlb \
  --result-file results/HyperDoc_mmlb/mmlb_qwen3vl_vl_hyperedge.jsonl \
  --output-root results/eval_reports \
  --run-name HyperDoc_mmlb \
  --metric-mode mmlb_official \
  --samples-file data/MMLongBench/samples.json
```

For LongDocURL reporting, first run answer extraction with the official
benchmark-side prompt in `scripts/evaluation_prompts.py` and write the extracted
short answer to `predicted_answer_eval`. Then run the LongDocURL rule-based
scorer:

```bash
python scripts/evaluate_HyperDoc.py \
  --dataset ldu \
  --result-file results/HyperDoc_ldu/ldu_qwen3vl_vl_hyperedge.jsonl \
  --output-root results/eval_reports \
  --run-name HyperDoc_ldu \
  --metric-mode ldu_official \
  --samples-file data/LongDocURL/samples.json \
  --pred-field predicted_answer_eval
```

For DocBench reporting, first run the GPT judge with the official benchmark-side
judge prompt in `scripts/evaluation_prompts.py` and write `judge_score` as `0`
or `1` for each row. Then aggregate the judge output:

```bash
python scripts/evaluate_HyperDoc.py \
  --dataset docbench \
  --result-file results/HyperDoc_docbench/docbench_qwen3vl_vl_hyperedge.judged.jsonl \
  --output-root results/eval_reports \
  --run-name HyperDoc_docbench \
  --metric-mode docbench_judge
```

The benchmark-side LongDocURL answer extraction prompt and DocBench GPT-judge
prompt are provided in `scripts/evaluation_prompts.py`; the comments in that
file mark them as copied from the official benchmark evaluation code.
