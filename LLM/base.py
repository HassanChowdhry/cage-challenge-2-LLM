from __future__ import annotations

from CybORG.Agents import BaseAgent
from .memory_db import TrajectoryDatabase


class BaseLLMAgent(BaseAgent):
    """Base class for LLM-driven agents with trajectory logging."""

    def __init__(self, db: TrajectoryDatabase | None = None) -> None:
        super().__init__()
        self.db = db or TrajectoryDatabase()

    def _log_transition(self, state, action, reward: float) -> None:
        if self.db:
            self.db.insert(state, action, reward)

    def end_episode(self) -> None:
        pass
