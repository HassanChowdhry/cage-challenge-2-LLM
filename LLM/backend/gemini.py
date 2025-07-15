from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from LLM.backend.model import LLMBackend
import google.generativeai as genai
import torch
import os
import logging

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

class GeminiBackend(LLMBackend):
    def __init__(self, hyperparams: Dict):
        self.model_name = hyperparams['model_name']
        self.temperature = hyperparams.get('temperature', 0.7)
        self.max_tokens = hyperparams.get('max_new_tokens', 100)
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        logger.info(f"Configured gemini model {self.model_name}")

        # self.model = create_react_agent(
            # model=model_name,
        # )
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        logger.info("Initialised Model")
    
    def generate(self, prompt: str) -> str:
        logger.info("Getting response from Gemini")
        response = self.model.generate_content(prompt)
        logger.info("Received response from Gemini")
            
        if response.text: return response.text.strip()
        else:
            logger.warning("Empty response from Gemini")
            return ""