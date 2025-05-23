import torch, torch.nn as nn

class ICM(nn.Module):
  def __init__(self, state_dim, action_dim):
    super().__init__()

    self.phi = nn.Sequential(
      nn.Linear(state_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
    )
    
    self.fwd = nn.Sequential(
      nn.Linear(action_dim + 128, 256),
      nn.ReLU(),
      nn.Linear(256, 128)
    )
    
    self.MSELoss = nn.MSELoss(reduction=None)
  
  def forward(
    self, 
    state_prev, # φ(s_t)
    state_next, # φ(s_{t+1})  target
    action_onehot # one-hot action at t
  ):
    phi_prev = self.phi(state_prev).detach()
    phi_next = self.phi(state_next)
    
    pred = self.fwd(torch.cat([phi_prev, action_onehot], 1))
    return self.MSELoss(pred, phi_next).mean(1)
  # if we find feature collapse, add inverse head to help phi learn controllable aspects