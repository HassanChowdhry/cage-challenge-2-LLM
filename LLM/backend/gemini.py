from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import LLMBackend
import google.generativeai as genai
import torch
import os
import logging

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

class GeminiBackend(LLMBackend):
    def __init__(self, hyperparams: Dict):
        model_name = hyperparams['model_name']
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        logger.info(f"Configured gemini model {model_name}")

        self.model = create_react_agent(
            model=model_name,
        )
        # self.model = genai.GenerativeModel(
        #     model_name=self.model_name,
        #     generation_config=genai.types.GenerationConfig(
        #         temperature=self.temperature,
        #         max_output_tokens=self.max_tokens,
        #     )
        # )
        logger.info("Initialised Model")
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.model.generate_content(prompt)
        logger.info("Get Response from Gemini")
            
        if response.text: return response.text.strip()
        else:
            logger.warning("Empty response from Gemini")
            return ""