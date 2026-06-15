#!/usr/bin/env python3
"""Evaluate HyperDoc JSONL prediction files with benchmark-aligned metrics."""

from __future__ import annotations

import argparse
import json
import re
from math import isclose
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_prediction(record: Dict[str, Any], pred_field: str) -> str:
    if pred_field != "auto":
        return str(record.get(pred_field, ""))
    for key in ("predicted_answer", "prediction", "answer"):
        if record.get(key) is not None:
            return str(record[key])
    return ""


def sample_key(record: Dict[str, Any]) -> tuple[str, str]:
    return str(record.get("doc_id", "")).strip(), str(record.get("question", "")).strip()


def load_sample_sidecar(path: Path | None) -> Dict[tuple[str, str], Dict[str, Any]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {sample_key(row): row for row in rows}


def load_mmlb_sidecar(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        question = mmlb_normalize_text(row.get("question", ""))
        if doc_id and question:
            out[f"{doc_id}\t{question}"] = row
    return out


# Copied from the MMLongBench evaluation helper used for the paper reports.
def mmlb_normalize_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"[.,!?;:\"'()\[\]{}%$#@&*+=/<>\\|`~^]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Copied from the MMLongBench evaluation helper used for the paper reports.
def mmlb_extract_answer_from_thinking_format(prediction: str) -> str:
    if not prediction:
        return ""
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", prediction, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    think_end_match = re.search(r"</think>\s*(.*?)$", prediction, re.DOTALL | re.IGNORECASE)
    if think_end_match:
        return re.sub(r"<[^>]+>", "", think_end_match.group(1)).strip()
    return prediction.strip()


# Copied from the MMLongBench evaluation helper used for the paper reports.
def mmlb_smart_match(ground_truth: str, prediction: str, normalize: bool = True) -> bool:
    if normalize:
        gt_norm = mmlb_normalize_text(ground_truth)
        pred_norm = mmlb_normalize_text(prediction)
    else:
        gt_norm = ground_truth
        pred_norm = prediction
    gt_words = gt_norm.split()
    if len(gt_words) <= 2 and len(gt_norm) < 15:
        return bool(re.search(r"\b" + re.escape(gt_norm) + r"\b", pred_norm))
    return gt_norm in pred_norm


def mmlb_is_oom_error(prediction: str) -> bool:
    if not prediction:
        return False
    indicators = ["CUDA out of memory", "OutOfMemoryError"]
    prediction_lower = prediction.lower()
    return any(indicator.lower() in prediction_lower for indicator in indicators)


def evaluate_mmlb_official(
    records: List[Dict[str, Any]],
    sidecar: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    matches = 0
    oom_errors = 0
    evaluated = 0

    for record in records:
        raw_prediction = str(record.get("prediction", record.get("predicted_answer", "")))
        prediction = mmlb_extract_answer_from_thinking_format(raw_prediction)
        if mmlb_is_oom_error(raw_prediction):
            oom_errors += 1
            continue

        side = sidecar.get(f"{str(record.get('doc_id', '')).strip()}\t{mmlb_normalize_text(record.get('question', ''))}", {})
        answer = record.get("ground_truth") if record.get("ground_truth") is not None else side.get("answer", "")
        if not str(answer).strip() or not prediction.strip():
            continue
        matches += int(mmlb_smart_match(str(answer), prediction, normalize=True))
        evaluated += 1

    valid_samples = len(records) - oom_errors
    return {
        "num_records": len(records),
        "evaluated": evaluated,
        "oom_errors": oom_errors,
        "valid_samples": valid_samples,
        "mmlb_accuracy": matches / valid_samples if valid_samples else 0.0,
        "mmlb_matches": matches,
    }


# Copied from the LongDocURL official rule-based evaluation implementation.
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(distances[i1], distances[i1 + 1], distances_[-1]))
        distances = distances_
    return distances[-1]


# Copied from the LongDocURL official rule-based evaluation implementation.
def anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


# Copied from the LongDocURL official rule-based evaluation implementation.
def is_float_equal(reference: Any, prediction: Any, include_percentage: bool = False, is_close: bool = False) -> bool:
    def get_precision(gt_ans: Any) -> int:
        precision = 3
        if "." in str(gt_ans):
            precision = len(str(gt_ans).split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
    for item in gt_result:
        try:
            if is_close and isclose(item, prediction, rel_tol=0.01):
                return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


# Copied from the LongDocURL official rule-based evaluation implementation.
def get_clean_string(value: Any) -> str:
    text = str(value).lower().strip()
    text = text.replace(",", "")
    for suffix in [
        "kg",
        "mm",
        "meters",
        "acres",
        "minutes",
        "miles",
        "mile",
        "million",
        "thousand",
        "billion",
        "m",
    ]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"^['\"]|['\"]$", "", text).strip()
    return text.lstrip("$").lstrip("£").rstrip("%").strip()


# Copied from the LongDocURL official rule-based evaluation implementation.
def is_exact_match(text: str) -> bool:
    if "https://" in text:
        return True
    if text.endswith(".py") or text.endswith("ipynb"):
        return True
    if text.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", text):
        return True
    if "a.m." in text or "p.m." in text:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        return True
    return False


def isfloat(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


# Copied from the LongDocURL official rule-based evaluation implementation.
def ldu_eval_score(ground_truth: Any, prediction: Any, answer_type: str) -> float:
    if answer_type == "Integer":
        try:
            ground_truth = get_clean_string(str(ground_truth))
            if re.findall(r"\d+,\s*\d+", ground_truth):
                ground_truth = "".join(c for c in ground_truth if c != ",")
            ground_truth = int(ground_truth)
        except Exception:
            pass
        try:
            prediction = get_clean_string(str(prediction))
            if re.findall(r"\d+,\s*\d+", prediction):
                prediction = "".join(c for c in prediction if c != ",")
            prediction = int(prediction)
        except Exception:
            prediction = ""
        return float(ground_truth == prediction)

    if answer_type == "Float":
        gt_s = get_clean_string(str(ground_truth))
        pred_s = get_clean_string(str(prediction))
        if re.findall(r"\d+,\s*\d+", gt_s):
            gt_s = "".join(c for c in gt_s if c != ",")
        if re.findall(r"\d+,\s*\d+", pred_s):
            pred_s = "".join(c for c in pred_s if c != ",")
        try:
            gt_f: Any = float(gt_s)
        except Exception:
            gt_f = gt_s
        try:
            pred_f: Any = float(pred_s)
        except Exception:
            pred_f = pred_s
        try:
            return float(is_float_equal(gt_f, pred_f, include_percentage=True, is_close=True))
        except Exception:
            return 0.0

    if answer_type in ("String", "None"):
        gt_c = get_clean_string(ground_truth)
        pred_c = get_clean_string(prediction)
        if is_exact_match(gt_c):
            return float(gt_c == pred_c)
        return anls_compute(gt_c, pred_c)

    if isinstance(ground_truth, str) and ground_truth.startswith("["):
        try:
            ground_truth = eval(ground_truth)
        except Exception:
            pass
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    if isinstance(prediction, str) and prediction.startswith("["):
        try:
            prediction = eval(prediction)
        except Exception:
            pass
    if not isinstance(prediction, list):
        prediction = [prediction]
    if not prediction:
        return 0.0
    if isinstance(ground_truth[0], dict):
        ground_truth = ["-".join(str(v) for v in item.values()) for item in ground_truth]
    if isinstance(prediction[0], dict):
        prediction = ["-".join(str(v) for v in item.values()) for item in prediction]

    gt_c = [get_clean_string(item) for item in ground_truth]
    pred_c = [get_clean_string(item) for item in prediction]
    if isfloat(gt_c[0]) or is_exact_match(gt_c[0]):
        return float("-".join(gt_c) == "-".join(pred_c))
    greedy = [max(anls_compute(str(gv), str(pv)) for pv in pred_c) for gv in gt_c]
    return sum(greedy) / len(gt_c) * min(1, len(gt_c) / len(pred_c)) ** 0.5


def evaluate_ldu_official(
    records: List[Dict[str, Any]],
    pred_field: str,
    sidecar: Dict[tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    score_total = 0.0
    evaluated = 0
    skipped = 0
    missing_answer_format = 0

    for record in records:
        if record.get("status") not in (None, "success"):
            skipped += 1
            continue
        sample = sidecar.get(sample_key(record), {})
        answer_type = str(sample.get("answer_format", record.get("answer_format", "")) or "")
        if not answer_type:
            missing_answer_format += 1
            skipped += 1
            continue
        prediction = record.get(pred_field) if pred_field != "auto" else extract_prediction(record, pred_field)
        ground_truth: Any = sample.get("answer", None)
        if ground_truth is None:
            ground_truth = record.get("ground_truth", record.get("answer", ""))
        score_total += ldu_eval_score(ground_truth, prediction, answer_type)
        evaluated += 1

    return {
        "num_records": len(records),
        "evaluated": evaluated,
        "skipped": skipped,
        "ldu_official_accuracy": score_total / evaluated if evaluated else 0.0,
        "missing_answer_format": missing_answer_format,
    }


def evaluate_docbench_judge(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    evaluated = 0
    skipped = 0
    for record in records:
        if "judge_score" not in record:
            skipped += 1
            continue
        try:
            correct += int(record.get("judge_score", 0))
            evaluated += 1
        except Exception:
            skipped += 1
    return {
        "num_records": len(records),
        "evaluated": evaluated,
        "skipped": skipped,
        "docbench_judge_accuracy": correct / evaluated if evaluated else 0.0,
    }


def choose_metric_mode(dataset: str, records: List[Dict[str, Any]], samples_file: Path | None) -> str:
    if dataset == "mmlb" and samples_file is not None:
        return "mmlb_official"
    if dataset == "docbench" and any("judge_score" in record for record in records):
        return "docbench_judge"
    if dataset == "ldu" and samples_file is not None:
        return "ldu_official"
    raise SystemExit(
        "Cannot infer evaluation mode. Provide --samples-file for mmlb/ldu, "
        "or provide a DocBench result file containing judge_score."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HyperDoc predictions.")
    parser.add_argument("--dataset", choices=["mmlb", "ldu", "docbench"], required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="HyperDoc")
    parser.add_argument("--pred-field", default="auto")
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=None,
        help="Dataset sample metadata JSON. Required for MMLongBench and LongDocURL official scoring.",
    )
    parser.add_argument(
        "--metric-mode",
        choices=["auto", "mmlb_official", "ldu_official", "docbench_judge"],
        default="auto",
        help=(
            "Evaluation mode. Use mmlb_official for MMLongBench reports, "
            "ldu_official after GPT answer extraction, and docbench_judge after GPT judging."
        ),
    )
    args = parser.parse_args()

    records = load_jsonl(args.result_file)
    metric_mode = args.metric_mode
    if metric_mode == "auto":
        metric_mode = choose_metric_mode(args.dataset, records, args.samples_file)

    if metric_mode == "ldu_official":
        sidecar = load_sample_sidecar(args.samples_file)
        if not sidecar:
            raise SystemExit("--samples-file is required for --metric-mode ldu_official")
        summary = evaluate_ldu_official(records, args.pred_field, sidecar)
    elif metric_mode == "mmlb_official":
        sidecar = load_mmlb_sidecar(args.samples_file)
        if not sidecar:
            raise SystemExit("--samples-file is required for --metric-mode mmlb_official")
        summary = evaluate_mmlb_official(records, sidecar)
    elif metric_mode == "docbench_judge":
        summary = evaluate_docbench_judge(records)
    else:
        raise SystemExit(f"Unsupported metric mode: {metric_mode}")

    summary.update({
        "dataset": args.dataset,
        "run_name": args.run_name,
        "result_file": str(args.result_file),
        "pred_field": args.pred_field,
        "metric_mode": metric_mode,
    })

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"{args.run_name}_{args.dataset}_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
