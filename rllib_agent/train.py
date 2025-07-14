import os
import sys
import argparse
import ray
import torch
import random 
import numpy as np

from ray import tune
from ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from .envs.random_attack_wrapper import RandomAttackWrapper
from .configs import get_config, update_config

def train_agent(config, results_dir, checkpoint_path=None):
    print(f"Config: {pretty_print(config)}")
    
    ray.init(ignore_reinit_error=True)
    
    stop = {
        "training_iteration": 10000000,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,
        "episode_reward_mean": -0.1,
    }
    
    log_dir = '../../logs/training/'
    algo = ppo.PPOTrainer
    results = tune.run(
        algo,
        config=train_config,
        name="" # algo_name + adv_name + time
        local_dir=results_dir,
        stop=stop,
        checkpoint_at_end=True,
        checkpoint_freq=config.get('checkpoint_freq', 50),
        keep_checkpoints_num=3,
        checkpoint_score_attr="episode_reward_mean",
        # restore=checkpoint_path,
    )
    
    print(f"Results saved to: {results_dir}")
    best_trial = results.get_best_trial("episode_reward_mean", "max")
    
    if best_trial:
        print(f"Best trial: {best_trial}")
        print(f"Best episode reward mean: {best_trial.last_result['episode_reward_mean']}")
    
    ray.shutdown()
    return results
    

def main():    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    config = get_config("ppo")
    
    results_dir = config.get('results_dir', 'results/rllib_ppo')
    checkpoint_dir = config.get('checkpoint_dir', 'Models/rllib_ppo')
    
    os.makedirs('results/rllib_ppo', exist_ok=True)
    os.makedirs('Models/rllib_ppo', exist_ok=True)
    
    results = train_agent(config, results_dir, checkpoint_dir)
    
    return results

if __name__ == "__main__":
    main() 