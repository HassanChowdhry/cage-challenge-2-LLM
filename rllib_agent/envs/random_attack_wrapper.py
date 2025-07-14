import gym
import numpy as np
from gym import spaces
from typing import Dict, Any, Optional
import random


class RandomAttackWrapper(gym.Env):
    """
    Wrapper environment that randomly selects between different attack types
    each episode, allowing a single agent to learn to handle multiple scenarios.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        
        self.attack_envs = {
            'b_line': BLineAttackEnv(config),
            'red_meander': RedMeanderAttackEnv(config)
        }
        
        # Set up spaces (assuming both envs have the same spaces)
        self.action_space = self.attack_envs['b_line'].action_space
        self.observation_space = self.attack_envs['b_line'].observation_space
        
        # Current environment tracking
        self.current_env = None
        self.current_attack_type = None
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        
        # Random seed for reproducibility
        self.seed = self.config.get('random_seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def reset(self):
        """Reset the environment and randomly select an attack type"""
        # Randomly select an attack type
        attack_types = list(self.attack_envs.keys())
        self.current_attack_type = random.choice(attack_types)
        self.current_env = self.attack_envs[self.current_attack_type]
        
        # Reset episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        
        # Reset the selected environment
        observation = self.current_env.reset()
        
        return observation
    
    def step(self, action):
        """Take a step in the current environment"""
        if self.current_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Take action in current environment
        observation, reward, done, info = self.current_env.step(action)
        
        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1
        
        # Add wrapper-specific info
        info['attack_type'] = self.current_attack_type
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the current environment"""
        if self.current_env is not None:
            return self.current_env.render(mode)
    
    def close(self):
        """Close all environments"""
        for env in self.attack_envs.values():
            env.close()
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        for env in self.attack_envs.values():
            if hasattr(env, 'seed'):
                env.seed(seed)
        return [seed]
    
    def get_current_attack_type(self):
        """Get the current attack type"""
        return self.current_attack_type
    
    def get_attack_type_distribution(self):
        """Get the distribution of attack types used so far"""
        # This would need to be implemented with episode tracking
        # For now, return None
        return None 