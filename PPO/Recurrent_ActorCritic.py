# modified from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch, torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions = [], []
        self.rewards, self.is_terminals = [], []
        self.logprobs = []
        self.hiddens = []

HIDDEN = 64 # size of GRU hidden state
class RecurrentActorCritic(nn.Module):
  def __init__(self, state_dim: int, action_dim: int):
    super(RecurrentActorCritic, self).__init__()
    
    self.encoder = nn.Sequential(
      nn.Linear(state_dim, HIDDEN),
      nn.ReLU(),
    )
    
    self.gru = nn.GRU(
      input_size=HIDDEN,
      output_size=HIDDEN,
      num_layers=1,
      batch_first=True # B,T,F
    )
    
    self.actor = nn.Sequential(
      nn.Linear(HIDDEN, action_dim),
      nn.Softmax(dim=-1),
    )
    
    self.critic = nn.Sequential(
      nn.Linear(HIDDEN, 1)
    )
  
  def init_hidden(self, batch_size: int = 1, device = None):
    device = device or next(self.parameters()).device
    return torch.zeros(1, batch_size, HIDDEN, device=device)

  def forward_core(self, x: torch.Tensor, h: torch.Tensor):
    """
    x  : (B, state_dim)           -- single timestep
    h  : (1, B, HIDDEN)           -- previous hidden
    returns: z (B, HIDDEN), h' (1,B,HIDDEN)
    """
    z = self.encoder(x)                 # (B, H)
    z, h = self.gru(z.unsqueeze(1), h)  # add seq len dim
    z = z.squeeze(1)                    # back to (B, H)
    return z, h
  
  def act(self, state, hidden, memory: Memory=None, deterministic=False, full=False):
    """
    state : (B,state_dim) tensor
    hidden: (1,B,HIDDEN)  tensor
    returns: action_idx  (and optionally next hidden)
    """
    z, h_next = self.forward_core(state, hidden)
    
    action_probs = self.actor(z)
    dist = Categorical(action_probs)
    
    action = torch.argmax(action_probs, dim=1) if deterministic else dist.sample()
    
    action_logp = dist.log_prob(action)
    
    if memory is not None:
      memory.states.append(state.detach())
      memory.hiddens.append(hidden.detach())
      memory.actions.append(action.detach())
      memory.logprobs.append(action_logp.detach())
    
    if full: return action_probs, h_next
      
    return action, h_next
  
  def evaluate(self, states, hiddens, actions):
    """
    states  : (B,state_dim)
    hiddens : (1,B,HIDDEN)
    """
    z, _ = self.forward_core(states, hiddens)
    
    action_probs = self.actor(z)
    dist = Categorical(action_probs)
    
    action_logprobs = dist.log_prob(actions)
    dist_entropy = dist.entropy()
    state_values = self.critic(z).squeeze(-1)
    
    return action_logprobs, state_values, dist_entropy
