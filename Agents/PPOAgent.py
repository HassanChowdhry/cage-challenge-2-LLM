import torch, torch.nn as nn
from torch.distributions import Categorical
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.Recurrent_ActorCritic import Memory
from PPO.Recurrent_ActorCritic import RecurrentActorCritic
from ICM import ICM

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
    self.old_policy = RecurrentActorCritic(state_dim=self.state_dims, action_dim=self.action_dims).to(device=device)
    self.old_policy.load_state_dict(self.policy.state_dict())
    
    # optimizer / loss
    self.optimizer = torch.optim.Adam(params=list(self.policy.parameters()) + list(self.icm.parameters()), lr=self.lr, betas=(0.9, 0.99))
    self.mse = torch.nn.MSELoss()
  
  def _add_scan(self, observation):
    pass
  
  def _pad_observation(self, observation):
    pass

  def train(self, results: Results):
    return super().train(results)
  
  def get_action(self, observation, action_space):
    return super().get_action(observation, action_space)
  
  def end_episode(self):
    return super().end_episode()

  def set_initial_values(self, action_space, observation):
    return super().set_initial_values(action_space, observation)