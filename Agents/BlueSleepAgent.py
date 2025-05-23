from CybORG.Agents import SleepAgent

class BlueSleepAgent(SleepAgent):
  def __init__(self):
    super().__init__()
    
  def train(self, results):
    return super().train(results)
  
  def end_episode(self):
    return super().end_episode()

  def set_initial_values(self, action_space, observation):
    return super().set_initial_values(action_space, observation)
  
  def get_action(self, observation, action_space):
    return super().get_action(observation, action_space)