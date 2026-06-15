"""Dataset loading helpers for the HyperDoc release pipeline."""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


DATASET_NAME_MAPPING: Dict[str, str] = {
    "mmlb": "MMLongBench",
    "ldu": "LongDocURL",
    "docbench": "docbench",
}


@dataclass
class DatasetConfig:
    name: str
    config_name: str = ""
    question_key: str = "question"
    gt_key: str = "answer"

    @property
    def data_dir(self) -> str:
        return f"./data/{self.name}"

    @property
    def extract_path(self) -> str:
        return f"./tmp/{self.name}"

    @property
    def sample_path(self) -> str:
        return f"{self.data_dir}/samples.json"


class BaseDataset:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sample_path_override: Optional[str] = None

    def load_data(self, use_retrieval: bool = True) -> List[Dict[str, Any]]:
        path = self.sample_path_override or self.config.sample_path
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        processed_samples = []
        for sample in samples:
            processed = dict(sample)
            processed["golden_pages"] = normalize_page_list(
                sample.get("evidence_pages", sample.get("golden_pages", []))
            )
            q_uid = processed.get("q_uid") or processed.get("question_id") or ""
            processed["q_uid"] = str(q_uid)
            processed_samples.append(processed)
        return processed_samples

    def get_sample_for_vl_experiment(self, idx: int) -> Dict[str, Any]:
        samples = self.load_data(use_retrieval=True)
        if idx >= len(samples):
            raise IndexError(f"Sample index {idx} is out of range for {len(samples)} samples.")

        sample = samples[idx]
        doc_id = sample.get("doc_id", "")
        clean_doc_id = re.sub(r"\.pdf$", "", doc_id).split("/")[-1]
        images = []
        for page_idx in sample.get("golden_pages", []):
            image_path = os.path.join(self.config.extract_path, f"{clean_doc_id}_{int(page_idx) - 1}.png")
            if os.path.exists(image_path):
                images.append(image_path)

        return {
            "q_uid": sample.get("q_uid", ""),
            "doc_id": doc_id,
            "question": sample.get(self.config.question_key, ""),
            "answer": sample.get(self.config.gt_key, sample.get("answer", "")),
            "images": images,
            "golden_pages": sample.get("golden_pages", []),
            "image_count": len(images),
            "image-top-10-question": sample.get("image-top-10-question", []),
            "image-top-10-question_score": sample.get("image-top-10-question_score", []),
        }


def normalize_page_list(value: Any) -> List[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, list):
        return []
    pages = []
    for item in value:
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.append(page)
    return pages


def resolve_dataset_name(input_name: str) -> tuple[str, str]:
    if input_name in DATASET_NAME_MAPPING:
        return input_name, DATASET_NAME_MAPPING[input_name]
    for config_name, actual_name in DATASET_NAME_MAPPING.items():
        if input_name == actual_name:
            return config_name, actual_name
    return input_name, input_name


def get_supported_datasets() -> Dict[str, str]:
    return DATASET_NAME_MAPPING.copy()


def create_dataset_config(dataset_name: str) -> DatasetConfig:
    config_name, actual_name = resolve_dataset_name(dataset_name)
    return DatasetConfig(name=actual_name, config_name=config_name)
