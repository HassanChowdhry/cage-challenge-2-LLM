from __future__ import annotations

from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

from .base import BaseLLMAgent
from .langraph_utils import build_generation_graph


def _load_default_model():
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required for LLM agents")
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


class FinetunedAgent(BaseLLMAgent):
    """Agent that assumes the underlying model has been fine-tuned."""

    def __init__(self, *args: Any, model=None, tokenizer=None, device=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if model is None or tokenizer is None:
            model, tokenizer, device = _load_default_model()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        try:
            self.graph = build_generation_graph(self.model, self.tokenizer, self.device)
        except Exception:  # pragma: no cover - optional dependency
            self.graph = None

    def get_action(self, observation, action_space=None, hidden=None):
        prompt = f"State: {observation}\nAction:"
        if self.graph:
            result = self.graph.invoke({"prompt": prompt})
            text = result.get("action", "0")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=1)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            return int(text.strip().split()[-1])
        except Exception:
            return 0

    def train(self, results):
        self._log_transition(results.observation, results.action, results.reward)
