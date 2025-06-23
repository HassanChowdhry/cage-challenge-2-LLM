import torch
import torch.nn as nn
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.Recurrent_ActorCritic import Memory, RecurrentActorCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent(BaseAgent):
    """Minimal recurrent PPO agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        eps_clip: float = 0.2,
        entropy_coeff: float = 0.0,
        K_epochs: int = 4,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.K_epochs = K_epochs

        self.memory = Memory()
        self.policy = RecurrentActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.hidden = self.policy.init_hidden()

    def end_episode(self) -> None:
        self.memory.clear()
        self.hidden = self.policy.init_hidden()

    def _compute_returns(self):
        discounted = 0
        returns = []
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            discounted = reward + self.gamma * discounted * (1 - float(done))
            returns.insert(0, discounted)
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _update(self) -> None:
        returns = self._compute_returns()
        states = torch.cat(self.memory.states, 0)
        hiddens = torch.cat(self.memory.hiddens).detach()
        actions = torch.stack(self.memory.actions).squeeze()
        old_logprobs = torch.stack(self.memory.logprobs)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, hiddens, actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse(state_values, returns)
            entropy_loss = -dist_entropy.mean()
            loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        self.end_episode()

    def train(self, results: Results):
        self.memory.rewards.append(results.reward)
        self.memory.is_terminals.append(results.done)
        if results.done:
            self._update()

    def get_action(self, observation, action_space=None, hidden=None):
        hidden = self.hidden if hidden is None else hidden
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action, self.hidden = self.policy.act(state, hidden, self.memory)
        return int(action.item())

    def set_initial_values(self, action_space, observation):
        pass
