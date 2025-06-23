import torch
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.PPO import PPO
from Agents.ICM_template import ICM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOICMAgent(BaseAgent):
    """PPO agent augmented with an Intrinsic Curiosity Module."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        eps_clip: float = 0.2,
        K_epochs: int = 4,
        curiosity_beta: float = 0.01,
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
        self.icm = ICM(state_dim, action_dim).to(device)
        self.icm_beta = curiosity_beta
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=lr)
        self.last_state: torch.Tensor | None = None

    def end_episode(self) -> None:
        self.algo.memory.clear()
        self.algo.hidden = self.algo.policy.init_hidden()
        self.last_state = None

    def train(self, results: Results):
        next_state = torch.tensor(results.observation, dtype=torch.float32).unsqueeze(0).to(device)
        if self.last_state is None:
            # first step of episode, no curiosity reward yet
            intrinsic = 0.0
            icm_loss = None
        else:
            action_idx = self.algo.memory.actions[-1]
            action_onehot = torch.zeros(1, self.algo.policy.actor[0].out_features, device=device)
            action_onehot[0, action_idx.item()] = 1.0
            intrinsic_r, inv_logits, _ = self.icm(self.last_state, next_state, action_onehot)
            intrinsic = intrinsic_r.item()
            inv_loss = torch.nn.functional.cross_entropy(inv_logits, action_idx.unsqueeze(0))
            fwd_loss = intrinsic_r.mean()
            icm_loss = inv_loss + fwd_loss
            self.icm_opt.zero_grad()
            icm_loss.backward()
            self.icm_opt.step()
        reward = results.reward + self.icm_beta * intrinsic
        self.algo.memory.rewards.append(reward)
        self.algo.memory.is_terminals.append(results.done)
        if results.done:
            self.algo.update()
        self.last_state = next_state

    def get_action(self, observation, action_space=None, hidden=None):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action, self.algo.hidden = self.algo.select_action(state)
        self.last_state = state
        return int(action.item())

    def set_initial_values(self, action_space, observation):
        pass
