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
  
  def forward(self, state, action_onehot):
    phi = self.phi(state)
    pred = self.fwd(torch.concat([phi.detach(), action_onehot], 1))
    return self.MSELoss(pred, phi).mean(1)
  # if we find feature collapse, add inverse head to help phi learn controllable aspects