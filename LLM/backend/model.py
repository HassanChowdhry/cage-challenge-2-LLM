from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)