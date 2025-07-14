from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import LLMBackend
import google.generativeai as genai
import torch
import os
import logging

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

#TODO: Fix this
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LocalHFBackend(LLMBackend):
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(model_name=model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(DEVICE)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
