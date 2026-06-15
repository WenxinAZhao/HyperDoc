#!/usr/bin/env python3
""""""

import torch
from typing import List, Optional, Dict, Any, Union

from .config import ModelConfig


class BaseModel:
    """"""
    
    def __init__(self, config: ModelConfig):
        """"""
        self.config = config
        self.last_predict_meta: Optional[Dict[str, Any]] = None
    
    def predict(
        self, 
        question: str, 
        texts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None, 
        history: Optional[List[Dict[str, Any]]] = None
    ):
        """"""
        pass
    
    def clean_up(self):
        """"""
        torch.cuda.empty_cache()

    def reset_predict_meta(self):
        """Reset metadata from the previous predict call."""
        self.last_predict_meta = None

    def set_predict_meta(self, **kwargs):
        """Store model-layer usage metadata for the latest predict call."""
        self.last_predict_meta = kwargs
    
    def process_message(
        self, 
        question: str, 
        texts: Optional[List[str]], 
        images: Optional[List[str]], 
        history: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """"""
        if history is not None:
            assert self.is_valid_history(history)
            messages = history.copy()
        else:
            messages = []
        
        #
        if texts is not None and len(texts) > 0:
            messages.append(self.create_text_message(texts, question))
        #
        elif images is not None and len(images) > 0:
            messages.append(self.create_image_message(images, question))
        #
        else:
            messages.append(self.create_ask_message(question))
        
        return messages
    
    def create_ask_message(self, question: str) -> Dict[str, Any]:
        """"""
        pass
    
    def create_ans_message(self, answer: str) -> Dict[str, Any]:
        """"""
        pass
    
    def create_text_message(self, texts: List[str], question: str) -> Dict[str, Any]:
        """"""
        pass
    
    def create_image_message(self, images: List[str], question: str) -> Dict[str, Any]:
        """"""
        pass
    
    def is_valid_history(self, history: List[Dict[str, Any]]) -> bool:
        """"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """"""
        return {
            'model_id': self.config.model_id,
            'model_type': self.config.model_type,
            'is_local': self.config.is_local,
            'local_path': self.config.local_path,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p
        } 
