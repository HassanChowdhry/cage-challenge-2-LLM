import torch, torch.nn as nn
from torch import Tensor
import numpy as np
from torch.distributions import Categorical
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.Recurrent_ActorCritic import Memory
from PPO.Recurrent_ActorCritic import RecurrentActorCritic
from ICM import ICM
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent(BaseAgent):
  def __init__(
    self, input_dims=52, action_space=[*range(158)] + [1000+i for i in range(9)], gamma=0.99, lr=0.002, beta=0.9, eps_clip=0.2,
    eps_ratio=1e-5, entropy_coeff=0.005, K_epochs=4, ckpt=None, deterministic=False, training=True,
  ):
    super().__init__()
    
    # RL dims
    self.scanables = 10 # defender + hosts
    self.state_dims = input_dims + self.scanables
    self.action_space = action_space
    self.action_dims = len(self.action_space)

    # hyperparamas/coeff
    self.gamma = gamma
    self.lr = lr
    self.beta = beta
    self.eps_clip = eps_clip
    self.eps_ratio = eps_ratio
    self.entropy_coeff = entropy_coeff
    self.K_epochs = K_epochs
    
    # rest
    self.ckpt = ckpt
    self.deterministic = deterministic
    self.training = training
    
    # reset
    self.end_episode()
    
    # decoy tables (host→preferred decoy list)
    self.greedy_decoys = {
      1000:[55,107,120,29], 
      1001:[43], 
      1002:[44],
      1003:[37,115,76,102], 
      1004:[51,116,38,90],
      1005:[130,91], 
      1006:[131], 
      1007:[54,106,28,119],
      1008:[61,35,113,126]
    }
    self.curr_decoys = {hidden_idx: [] for hidden_idx in self.greedy_decoys.keys()}
    self.scan = np.zeros(self.scanables)
    
    # make a mapping of restores to decoys
    self.restore_decoy_mapping = dict()
    # decoys for defender host
    base_list = [28, 41, 54, 67, 80, 93, 106, 119]
    # add for all hosts
    for i in range(13):
        self.restore_decoy_mapping[132 + i] = [x + i for x in base_list]
    
    #? restore policy
    #? if self.restore:
    #?   pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
    #?   self.policy.load_state_dict(pretained_model)
    
    # policy
    self.memory = Memory()
    self.policy = RecurrentActorCritic(state_dim=self.state_dims, action_dim=self.action_dims).to(device=device)
    self.icm = ICM(state_dim=self.state_dims, action_dim=self.action_dims)
    self.hidden = self.policy.init_hidden() # GRU hidden (internal)
    
    # optimizer / loss
    self.optimizer = torch.optim.Adam(params=list(self.policy.parameters()) + list(self.icm.parameters()), lr=self.lr, betas=(0.9, 0.99))
    self.mse = torch.nn.MSELoss()
  
  def _add_scan(self, observation):
    # obs = 0 = never scanned
    # 1 = scanned sometime earlier this episode
    # 2 = scanned *this* timestep (most recent)

    idx = [0, 4, 8, 12, 28, 32, 36, 40, 44, 48] # 10 start positions = 10 hosts
    for i, j in enumerate(idx):
      if observation[j] == 1 and observation[j+1] == 0:
        self.scan[:] = np.where(self.scan == 2, 1, self.scan)  # demote old “latest”
        self.scan[i] = 2 
        break
  
  def _pad_observation(self, observation):
    return np.concatenate([observation, self.scan])
  
  def _map_network_action(self, action_index):
    pass
  
  def _curiosity_bonus(self, observation, action_idx):
    """Compute scalar intrinsic reward for the latest transition."""
    state_prev = self.memory.states[-1]
    state_next = torch.FloatTensor(self._pad_observation(observation)).unsqueeze(0).to(device)

    onehot = torch.zeros(1, self.action_dims, device=device)
    onehot[0, action_idx] = 1.0
    
    # intrinsic reward = forward-model prediction error
    return float(self.icm(state_prev, state_next, onehot).item())
  
  def _update(self):
    discounted_rewards = 0
    returns = deque()
    
    for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
      discounted_rewards = reward + self.gamma * discounted_rewards * ( 1.0 - float(done) ) # reset if done is 1
      returns.appendleft(discounted_rewards)
    
    returns_tensor = torch.tensor(returns, device=device)
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + self.eps_ratio) # normalize
    
    old_states = torch.cat(self.memory.states, 0)
    old_hidden = torch.cat(self.memory.hiddens).detach()
    old_actions = torch.stack(self.memory.actions).squeeze()
    old_action_logprobs = torch.stack(self.memory.logprobs)
    
    for _ in range(self.K_epochs):
      action_logprobs, state_values, dist_entropy = self.policy.evaluate(states=old_states, hiddens=old_hidden, actions=old_actions)
      ratio = torch.exp(action_logprobs - old_action_logprobs)
      advantage = returns_tensor - state_values.detach()
      
      surr1 = ratio * advantage
      surr2 = torch.clamp(ratio, 1 + self.eps_clip, 1 - self.eps_clip) * advantage
      
      actor_loss = -torch.min(surr1, surr2).mean()
      critic_loss = self.mse(returns, state_values) * 0.5
      entropy_loss = -(self.beta * dist_entropy.mean())
      loss: Tensor = actor_loss + critic_loss + entropy_loss
      
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
      self.optimizer.step()
    self.memory.clear()
  
  def end_episode(self):
    self.scan = np.fill(0)
    self.curr_decoys = { hidden_idx: [] for hidden_idx in self.greedy_decoys.keys() }
    self.hidden = self.policy.init_hidden()

  def train(self, results: Results):
    reward = results.reward
    done = results.done
    
    intrinsic_curiosity_bonus = self._curiosity_bonus(results.observation, self.memory.actions[-1])
    self.memory.rewards.append(reward + self.beta * intrinsic_curiosity_bonus)
    self.memory.is_terminals.append(done)
    
    if done:
      self._update()
      self.end_episode()
  
  def get_action(self, observation, action_space=None, hidden=None):
    """Return CybORG action integer."""
    hidden = self.hidden if hidden is None else hidden
    self._add_scan(observation=observation)
    observation_tensor = torch.FloatTensor(self._pad_observation(observation)).unsqueeze(0).to(device)
    
    action_idx, h2 = self.policy.act(state=observation_tensor, hidden=hidden, memory=self.memory)
    self.h2 = h2
    
    env_action = self._map_network_action(action_index=action_idx)
    return env_action

  def set_initial_values(self, action_space, observation): pass