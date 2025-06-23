import torch
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.PPO import PPO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent(BaseAgent):
    """Wraps the :class:`PPO` algorithm in a CybORG agent interface."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        eps_clip: float = 0.2,
        K_epochs: int = 4,
    ) -> None:
        super().__init__()
        self.algo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            K_epochs=K_epochs,
        )

    def end_episode(self) -> None:
        self.algo.memory.clear()
        self.algo.hidden = self.algo.policy.init_hidden()

    def train(self, results: Results):
        self.algo.memory.rewards.append(results.reward)
        self.algo.memory.is_terminals.append(results.done)
        if results.done:
            self.algo.update()

    def get_action(self, observation, action_space=None, hidden=None):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action, self.algo.hidden = self.algo.select_action(state)
        return int(action.item())

    def set_initial_values(self, action_space, observation):
        pass
