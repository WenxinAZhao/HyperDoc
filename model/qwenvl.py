#!/usr/bin/env python3
""""""

import os
import time
import tempfile
import importlib.util
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForImageTextToText,
)
import torch

from .config import ModelConfig
from .base_model import BaseModel


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _maybe_enable_auto_device(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if _has_accelerate():
        kwargs["device_map"] = "auto"
    return kwargs


def _finalize_loaded_model(model):
    if not _has_accelerate() and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def _extract_visual_usage_from_inputs(processor, inputs, images):
    """Extract visual token usage from multimodal processor outputs."""
    image_grid_thw = inputs.get("image_grid_thw")
    merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", None)
    image_count = len(images) if images else 0
    visual_tokens = None
    image_grid_thw_list = None

    if image_grid_thw is not None:
        try:
            image_grid_thw_cpu = image_grid_thw.detach().cpu()
            image_grid_thw_list = image_grid_thw_cpu.tolist()
            if merge_size is not None:
                merge_length = int(merge_size) ** 2
                visual_tokens = int(sum(int(grid.prod().item()) // merge_length for grid in image_grid_thw_cpu))
            image_count = int(len(image_grid_thw_list))
        except Exception:
            pass

    return {
        "image_count": image_count,
        "visual_tokens": visual_tokens,
        "image_grid_thw": image_grid_thw_list,
    }


def _get_qwenvl_max_pixels() -> Optional[int]:
    raw = os.getenv("QWENVL_MAX_PIXELS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _prepare_qwenvl_images(images: Optional[List[Any]]) -> Optional[List[Any]]:
    """Optionally resize image inputs to a max-pixel budget before processor tokenization."""
    if not images:
        return images
    max_pixels = _get_qwenvl_max_pixels()
    if not max_pixels:
        return images

    prepared: List[Any] = []
    for image in images:
        if not isinstance(image, str) or not os.path.exists(image):
            prepared.append(image)
            continue
        try:
            with Image.open(image) as pil_img:
                pil_img = pil_img.convert("RGB")
                width, height = pil_img.size
                pixels = width * height
                if pixels <= max_pixels:
                    prepared.append(image)
                    continue
                scale = (max_pixels / pixels) ** 0.5
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".png",
                    prefix="qwenvl_resized_",
                )
                resized.save(tmp.name)
                tmp.close()
                prepared.append(tmp.name)
        except Exception:
            prepared.append(image)
    return prepared


class Qwen2VL(BaseModel):
    """"""
    
    def __init__(self, config: ModelConfig):
        """"""
        super().__init__(config)
        
        print(f"Loading Qwen2-VL model: {self.config.model_id}")
        if self.config.is_local:
            print(f"Local model path: {self.config.local_path}")
        
        #
        load_kwargs = _maybe_enable_auto_device({
            "torch_dtype": torch.float16,
        })
        self.model = _finalize_loaded_model(Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        ))
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )
        
        print("Model loaded")
        
        #
        self.create_ask_message = lambda question: {
            "role": "user", 
            "content": [{"type": "text", "text": question}]
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant", 
            "content": [{"type": "text", "text": ans}]
        }
    
    def create_text_message(self, texts: List[str], question: str) -> Dict[str, Any]:
        """"""
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        return {"role": "user", "content": content}
    
    def create_image_message(self, images: List[str], question: str) -> Dict[str, Any]:
        """"""
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        return {"role": "user", "content": content}
    
    @torch.no_grad()
    def predict(
        self, 
        question: str, 
        texts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """"""
        self.clean_up()
        self.reset_predict_meta()
        images = _prepare_qwenvl_images(images)
        
        #
        messages = self.process_message(question, texts, images, history)
        
        try:
            #
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            #
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            
            #
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            
            #
            _gen_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
            if hasattr(self.config, 'temperature'):
                _temp = self.config.temperature
                if _temp is not None and _temp > 0:
                    _gen_kwargs['do_sample'] = True
                    _gen_kwargs['temperature'] = _temp
                    if hasattr(self.config, 'top_p') and self.config.top_p is not None:
                        _gen_kwargs['top_p'] = self.config.top_p
                else:
                    _gen_kwargs['do_sample'] = False
            generated_ids = self.model.generate(**_gen_kwargs)
            
            #
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            prompt_tokens = int(inputs.input_ids.shape[-1])
            completion_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
            visual_usage = _extract_visual_usage_from_inputs(self.processor, inputs, images)
            
            #
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            self.set_predict_meta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                image_count=visual_usage["image_count"],
                visual_tokens=visual_usage["visual_tokens"],
                image_grid_thw=visual_usage["image_grid_thw"],
                model_type=self.config.model_type,
            )
            
            #
            messages.append(self.create_ans_message(output_text))
            
            self.clean_up()
            
            return output_text, messages
            
        except Exception as e:
            print(f"Inference error: {e}")
            self.clean_up()
            return f"Inference failed: {str(e)}", messages

    def is_valid_history(self, messages: List[Dict[str, Any]]) -> bool:
        """"""
        if not isinstance(messages, list):
            return False

        for item in messages:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False

            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in ["text", "image"]:
                    return False

        return True


class Qwen2_5VL(Qwen2VL):

    def __init__(self, config):
        #
        BaseModel.__init__(self, config)
        load_kwargs = _maybe_enable_auto_device({
            "torch_dtype": torch.float16,
        })
        self.model = _finalize_loaded_model(Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id, 
            **load_kwargs,
        ))
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        self.create_ask_message = lambda question: {
            "role": "user", 
            "content": [{"type": "text", "text": question}]
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant", 
            "content": [{"type": "text", "text": ans}]
        }


class Qwen3VL(Qwen2VL):
    """"""
    
    def __init__(self, config: ModelConfig):
        """"""
        #
        BaseModel.__init__(self, config)
        
        print(f"Loading Qwen3-VL model: {self.config.model_id}")
        if self.config.is_local:
            print(f"Local model path: {self.config.local_path}")
        
        #
        load_kwargs = _maybe_enable_auto_device({
            "dtype": "auto",
        })
        self.model = _finalize_loaded_model(Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        ))
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id
        )
        
        print("Model loaded")
        
        #
        self.create_ask_message = lambda question: {
            "role": "user", 
            "content": [{"type": "text", "text": question}]
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant", 
            "content": [{"type": "text", "text": ans}]
        }
    
    @torch.no_grad()
    def predict(
        self, 
        question: str, 
        texts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """"""
        self.clean_up()
        self.reset_predict_meta()
        images = _prepare_qwenvl_images(images)
        
        #
        messages = self.process_message(question, texts, images, history)
        
        try:
            #
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            #
            _gen_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
            if hasattr(self.config, 'temperature'):
                _temp = self.config.temperature
                if _temp is not None and _temp > 0:
                    _gen_kwargs['do_sample'] = True
                    _gen_kwargs['temperature'] = _temp
                    if hasattr(self.config, 'top_p') and self.config.top_p is not None:
                        _gen_kwargs['top_p'] = self.config.top_p
                else:
                    _gen_kwargs['do_sample'] = False
            generated_ids = self.model.generate(**_gen_kwargs)
            
            #
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            prompt_tokens = int(inputs.input_ids.shape[-1])
            completion_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
            visual_usage = _extract_visual_usage_from_inputs(self.processor, inputs, images)
            
            #
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            self.set_predict_meta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                image_count=visual_usage["image_count"],
                visual_tokens=visual_usage["visual_tokens"],
                image_grid_thw=visual_usage["image_grid_thw"],
                model_type=self.config.model_type,
            )
            
            #
            messages.append(self.create_ans_message(output_text))
            
            self.clean_up()
            
            return output_text, messages
            
        except Exception as e:
            print(f"Inference error: {e}")
            self.clean_up()
            return f"Inference failed: {str(e)}", messages


class Qwen35(Qwen3VL):
    """"""

    def __init__(self, config: ModelConfig):
        BaseModel.__init__(self, config)

        print(f"Loading Qwen3.5 model: {self.config.model_id}")
        if self.config.is_local:
            print(f"Local model path: {self.config.local_path}")

        load_kwargs = _maybe_enable_auto_device({
            "torch_dtype": "auto",
            "trust_remote_code": True,
        })
        self.model = _finalize_loaded_model(AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        ))
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )

        print("Model loaded")

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [{"type": "text", "text": ans}]
        }

    @torch.no_grad()
    def predict(
        self,
        question: str,
        texts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.clean_up()
        self.reset_predict_meta()

        messages = self.process_message(question, texts, images, history)

        try:
            chat_template_kwargs = {}
            if hasattr(self.config, "enable_thinking") and self.config.enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = bool(self.config.enable_thinking)

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **chat_template_kwargs
            )
            inputs = inputs.to(self.model.device)

            _gen_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
            if hasattr(self.config, 'temperature'):
                _temp = self.config.temperature
                if _temp is not None and _temp > 0:
                    _gen_kwargs['do_sample'] = True
                    _gen_kwargs['temperature'] = _temp
                    if hasattr(self.config, 'top_p') and self.config.top_p is not None:
                        _gen_kwargs['top_p'] = self.config.top_p
                else:
                    _gen_kwargs['do_sample'] = False
            generated_ids = self.model.generate(**_gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            prompt_tokens = int(inputs.input_ids.shape[-1])
            completion_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
            visual_usage = _extract_visual_usage_from_inputs(self.processor, inputs, images)

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            self.set_predict_meta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                image_count=visual_usage["image_count"],
                visual_tokens=visual_usage["visual_tokens"],
                image_grid_thw=visual_usage["image_grid_thw"],
                model_type=self.config.model_type,
            )

            messages.append(self.create_ans_message(output_text))
            self.clean_up()
            return output_text, messages

        except Exception as e:
            print(f"Inference error: {e}")
            self.clean_up()
            return f"Inference failed: {str(e)}", messages
