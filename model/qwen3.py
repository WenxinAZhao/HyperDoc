#!/usr/bin/env python3
""""""

import torch
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig
from .base_model import BaseModel


class Qwen3(BaseModel):
    """"""
    
    def __init__(self, config: ModelConfig):
        """"""
        super().__init__(config)
        
        print(f"Loading Qwen3 model: {self.config.model_id}")
        if self.config.is_local:
            print(f"Local model path: {self.config.local_path}")
        
        #
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )
        
        #
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype="auto",
            device_map=self.config.device,
            trust_remote_code=True
        )
        
        #
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded")
        
        #
        self.create_ask_message = lambda question: {"role": "user", "content": question}
        self.create_ans_message = lambda ans: {"role": "assistant", "content": ans}
    
    def create_text_message(self, texts: List[str], question: str) -> Dict[str, str]:
        """"""
        if texts:
            combined_text = "\n\n".join([f"Text {i+1}:\n{text}" for i, text in enumerate(texts)])
            full_content = f"{combined_text}\n\nQuestion: {question}"
        else:
            full_content = f"Question: {question}"
            
        return {"role": "user", "content": full_content}
    
    def create_image_message(self, images: List[str], question: str) -> Dict[str, str]:
        """"""
        content = f"[Note: This is a text-only model and cannot process images directly]\nQuestion: {question}"
        return {"role": "user", "content": content}
    
    @torch.no_grad()
    def predict(
        self, 
        question: str, 
        texts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """"""
        self.clean_up()
        self.reset_predict_meta()
        
        #
        messages = self.process_message(question, texts, images, history)
        
        try:
            #
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            #
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            #
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][input_length:]
            prompt_tokens = int(input_length)
            completion_tokens = int(generated_tokens.shape[-1])
            
            #
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            self.set_predict_meta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                image_count=len(images) if images else 0,
                model_type=self.config.model_type,
            )
            
            #
            messages.append(self.create_ans_message(output_text))
            
            self.clean_up()
            
            return output_text, messages
            
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            self.clean_up()
            return f"Inference failed: {str(e)}", messages
    
    def is_valid_history(self, messages: List[Dict[str, str]]) -> bool:
        """"""
        if not isinstance(messages, list):
            return False
        
        for item in messages:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                return False
        
        return True
    
    def batch_predict(
        self, 
        questions: List[str], 
        texts_list: Optional[List[List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """"""
        results = []
        
        for i, question in enumerate(questions):
            texts = texts_list[i] if texts_list and i < len(texts_list) else None
            
            try:
                answer, messages = self.predict(question, texts=texts)
                result = {
                    'question': question,
                    'answer': answer,
                    'text_count': len(texts) if texts else 0,
                    'status': 'success'
                }
            except Exception as e:
                result = {
                    'question': question,
                    'answer': None,
                    'text_count': len(texts) if texts else 0,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
            
        return results


class Qwen3Coder(BaseModel):
    """"""
    
    def __init__(self, config: ModelConfig):
        """"""
        super().__init__(config)
        
        print(f"Loading Qwen3-Coder model: {self.config.model_id}")
        if self.config.is_local:
            print(f"Local model path: {self.config.local_path}")
        
        #
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )
        
        #
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype="auto",
            device_map=self.config.device,
            trust_remote_code=True
        )
        
        #
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded")
        
        #
        self.create_ask_message = lambda question: {"role": "user", "content": question}
        self.create_ans_message = lambda ans: {"role": "assistant", "content": ans}
    
    def create_text_message(self, texts: List[str], question: str) -> Dict[str, str]:
        """"""
        if texts:
            combined_text = "\n\n".join([f"Text {i+1}:\n{text}" for i, text in enumerate(texts)])
            full_content = f"{combined_text}\n\nQuestion: {question}"
        else:
            full_content = f"Question: {question}"
            
        return {"role": "user", "content": full_content}
    
    def create_image_message(self, images: List[str], question: str) -> Dict[str, str]:
        """"""
        content = f"[Note: This is a text-only model and cannot process images directly]\nQuestion: {question}"
        return {"role": "user", "content": content}
    
    @torch.no_grad()
    def predict(
        self, 
        question: str, 
        texts: Optional[List[str]] = None, 
        images: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """"""
        self.clean_up()
        self.reset_predict_meta()
        
        #
        messages = self.process_message(question, texts, images, history)
        
        try:
            #
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            #
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            #
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            #
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            prompt_tokens = int(model_inputs.input_ids.shape[-1])
            completion_tokens = int(generated_ids[0].shape[-1]) if generated_ids else 0
            
            #
            output_text = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            self.set_predict_meta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                image_count=len(images) if images else 0,
                model_type=self.config.model_type,
            )
            
            #
            messages.append(self.create_ans_message(output_text))
            
            self.clean_up()
            
            return output_text, messages
            
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            self.clean_up()
            return f"Inference failed: {str(e)}", messages
    
    def is_valid_history(self, messages: List[Dict[str, str]]) -> bool:
        """"""
        if not isinstance(messages, list):
            return False
        
        for item in messages:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                return False
        
        return True
    
    def batch_predict(
        self, 
        questions: List[str], 
        texts_list: Optional[List[List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """"""
        results = []
        
        for i, question in enumerate(questions):
            texts = texts_list[i] if texts_list and i < len(texts_list) else None
            
            try:
                answer, messages = self.predict(question, texts=texts)
                result = {
                    'question': question,
                    'answer': answer,
                    'text_count': len(texts) if texts else 0,
                    'status': 'success'
                }
            except Exception as e:
                result = {
                    'question': question,
                    'answer': None,
                    'text_count': len(texts) if texts else 0,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
            
        return results
