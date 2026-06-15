#!/usr/bin/env python3
""""""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """"""
    model_id: str
    model_type: str = "qwen25"  # qwen25, qwen2vl, qwen25vl, qwen3vl, qwen35
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    device: str = "auto"
    is_local: bool = False
    local_path: Optional[str] = None
    enable_thinking: Optional[bool] = None
    
    def __post_init__(self):
        """"""
        #
        if os.path.exists(self.model_id) or self.model_id.startswith('./') or self.model_id.startswith('/'):
            self.is_local = True
            self.local_path = os.path.abspath(self.model_id)
            
    def to_dict(self) -> Dict[str, Any]:
        """"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """"""
        return cls(**config_dict)


#
PREDEFINED_MODELS = {
    "qwen25-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "qwen25"
    },
    "qwen25-14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct", 
        "model_type": "qwen25"
    },
    "qwen25-32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "model_type": "qwen25"
    },
    "qwen2vl-7b": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_type": "qwen2vl"
    },
    "qwen2vl-72b": {
        "model_id": "Qwen/Qwen2-VL-72B-Instruct",
        "model_type": "qwen2vl"
    },
    "qwen25vl-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_type": "qwen25vl"
    },
    "qwen25vl-72b": {
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "model_type": "qwen25vl"
    },
    "qwen3vl-8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_type": "qwen3vl"
    },
    "qwen3-14b": {
        "model_id": "Qwen/Qwen3-14B",
        "model_type": "qwen3"
    },
    "qwen3coder": {
        "model_id": "Qwen/Qwen3-Coder-Next",
        "model_type": "qwen3coder"
    },
    "qwen35-9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_type": "qwen35"
    },
}


def create_config(
    model_id: str,
    model_type: Optional[str] = None,
    **kwargs
) -> ModelConfig:
    """"""
    #
    if model_id in PREDEFINED_MODELS:
        predefined = PREDEFINED_MODELS[model_id]
        model_id = predefined["model_id"]
        if model_type is None:
            model_type = predefined["model_type"]
    
    #
    if model_type is None:
        if "qwen3.5" in model_id.lower() or "qwen35" in model_id.lower():
            model_type = "qwen35"
        elif "3-vl" in model_id.lower() or "3vl" in model_id.lower():
            model_type = "qwen3vl"
        elif "2.5-vl" in model_id.lower() or "25vl" in model_id.lower():
            model_type = "qwen25vl"
        elif "vl" in model_id.lower():
            model_type = "qwen2vl"
        elif "coder" in model_id.lower():
            model_type = "qwen3coder"
        elif "qwen3" in model_id.lower() or "qwen/qwen3-" in model_id.lower():
            model_type = "qwen3"
        else:
            model_type = "qwen25"
    
    return ModelConfig(
        model_id=model_id,
        model_type=model_type,
        **kwargs
    )


def load_config(config_path: Union[str, Path]) -> ModelConfig:
    """"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return ModelConfig.from_dict(config_dict)


def save_config(config: ModelConfig, config_path: Union[str, Path]) -> None:
    """"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def list_local_models(base_path: Union[str, Path]) -> Dict[str, str]:
    """"""
    base_path = Path(base_path)
    models = {}
    
    if not base_path.exists():
        return models
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            #
            has_model = any([
                (model_dir / "config.json").exists(),
                (model_dir / "pytorch_model.bin").exists(),
                (model_dir / "model.safetensors").exists(),
                any(model_dir.glob("*.safetensors")),
                any(model_dir.glob("*.bin"))
            ])
            
            if has_model:
                models[model_dir.name] = str(model_dir.absolute())
    
    return models

if __name__ == "__main__":
    #
    print("Testing model config")
    
    #
    config1 = create_config("qwen25-7b")
    print(f"Remote model config: {config1}")
    
    #
    config2 = create_config("/path/to/Qwen2.5-VL-7B-Instruct")
    print(f"Local model config: {config2}")
    
    #
    for name in PREDEFINED_MODELS:
        config = create_config(name)
        print(f"{name}: {config.model_id} ({config.model_type})") 
