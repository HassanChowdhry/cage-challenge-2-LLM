from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from CybORG.Agents import BaseAgent

# Store in a database?? Or In Memory with ReAct 
class BaseLLMAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
    def _log_transition(self, state, action, reward: float) -> None:
        pass
    def end_episode(self) -> None:
        pass