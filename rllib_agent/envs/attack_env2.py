"""
Red Meander Attack Environment for RLlib
"""

import gym
import numpy as np
from gym import spaces
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
import inspect

class RedMeanderAttackEnv(gym.Env):
    """
    Environment wrapper for Red Meander attack scenario
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Set up CybORG environment
        self.path = str(inspect.getfile(CybORG))
        self.path = self.path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        
        # Initialize CybORG with Red Meander agent
        self.cyborg = CybORG(self.path, 'sim', agents={'Red': RedMeanderAgent})
        self.env = ChallengeWrapper(env=self.cyborg, agent_name="Blue")
        
        # Set up spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Configuration
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 100)
        self.current_step = 0
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        
    def reset(self):
        """Reset the environment for a new episode"""
        # Reinitialize CybORG for fresh episode
        self.cyborg = CybORG(self.path, 'sim', agents={'Red': RedMeanderAgent})
        self.env = ChallengeWrapper(env=self.cyborg, agent_name="Blue")
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0
        
        # Get initial observation
        observation = self.env.reset()
        return observation
    
    def step(self, action):
        """Take a step in the environment"""
        # Take action
        observation, reward, done, info = self.env.step(action)
        
        # Update episode tracking
        self.current_step += 1
        self.episode_reward += reward
        self.episode_length += 1
        
        # Check if episode should end due to max steps
        if self.current_step >= self.max_steps:
            done = True
        
        # Add episode info
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        info['attack_type'] = 'red_meander'
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment (not implemented for CybORG)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass 