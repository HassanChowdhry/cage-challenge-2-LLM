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
from .memory_db import TrajectoryDatabase


def _load_default_model():
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required for LLM agents")
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


class RAGFinetunedAgent(BaseLLMAgent):
    """Agent that combines retrieval with a fine-tuned model."""

    def __init__(self, *args: Any, db: TrajectoryDatabase | None = None, model=None, tokenizer=None, device=None, **kwargs: Any) -> None:
        super().__init__(db=db)
        if model is None or tokenizer is None:
            model, tokenizer, device = _load_default_model()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _build_context(self, observation: Any) -> str:
        context = []
        if self.db:
            for s, a, r in list(self.db.all_transitions())[-5:]:
                context.append(f"State: {s} Action: {a} Reward: {r}")
        context.append(f"Current state: {observation}")
        return "\n".join(context)

    def get_action(self, observation, action_space=None, hidden=None):
        prompt = self._build_context(observation) + "\nAction:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=1)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            return int(text.strip().split()[-1])
        except Exception:
            return 0

    def train(self, results):
        self._log_transition(results.observation, results.action, results.reward)
