#!/usr/bin/env python3
""""""

from typing import Optional, Union
from .config import ModelConfig, create_config
from .qwen import Qwen25
from .qwen3 import Qwen3, Qwen3Coder
from .qwenvl import Qwen2VL, Qwen2_5VL, Qwen3VL, Qwen35


def create_reasoner(
    model_id: str,
    model_type: Optional[str] = None,
    **kwargs
) -> Union[Qwen25, Qwen3, Qwen3Coder, Qwen2VL, Qwen2_5VL, Qwen3VL, Qwen35]:
    """"""
    #
    config = create_config(model_id, model_type, **kwargs)
    
    #
    if config.model_type == "qwen25":
        return Qwen25(config)
    elif config.model_type == "qwen3":
        return Qwen3(config)
    elif config.model_type == "qwen3coder":
        return Qwen3Coder(config)
    elif config.model_type == "qwen2vl":
        return Qwen2VL(config)
    elif config.model_type == "qwen25vl":
        return Qwen2_5VL(config)
    elif config.model_type == "qwen3vl":
        return Qwen3VL(config)
    elif config.model_type == "qwen35":
        return Qwen35(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


def create_reasoner_from_config(config: ModelConfig) -> Union[Qwen25, Qwen3, Qwen3Coder, Qwen2VL, Qwen2_5VL, Qwen3VL, Qwen35]:
    """"""
    if config.model_type == "qwen25":
        return Qwen25(config)
    elif config.model_type == "qwen3":
        return Qwen3(config)
    elif config.model_type == "qwen3coder":
        return Qwen3Coder(config)
    elif config.model_type == "qwen2vl":
        return Qwen2VL(config)
    elif config.model_type == "qwen25vl":
        return Qwen2_5VL(config)
    elif config.model_type == "qwen3vl":
        return Qwen3VL(config)
    elif config.model_type == "qwen35":
        return Qwen35(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


#
def create_model(model_id: str, model_type: Optional[str] = None, **kwargs):
    """"""
    return create_reasoner(model_id, model_type, **kwargs) 