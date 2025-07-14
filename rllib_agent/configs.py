import os
from typing import Dict, Any
from envs import RandomAttackWrapper

PPO_CONFIG = {
    "env": RandomAttackEnv,
    "framework": "torch",
    "num_workers": 4,
    "num_gpus": 0,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    },
    
    # Training parameters
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.5,
    "entropy_coeff": 0.01,
    "vf_clip_param": 5.0,
    "vf_loss_coeff": 1.0,
    
    # Environment configuration
    "env_config": {
        "max_steps": 100,
        "random_seed": 0,
    },
    
    # Logging and checkpointing
    "log_level": "INFO",
    "evaluation_interval": 100,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    
    # Checkpointing
    "checkpoint_freq": 50,
    "checkpoint_at_end": True,
}

PPO_ICM_CONFIG = {
    "env": RandomAttackEnv,
    "framework": "torch",
    "num_workers": 4,
    "num_gpus": 0,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    },
    
    # Training parameters
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.5,
    "entropy_coeff": 0.01,
    "vf_clip_param": 5.0,
    "vf_loss_coeff": 1.0,
    
    # Environment configuration
    "env_config": {
        "max_steps": 100,
        "random_seed": 0,
    },
    
    # Logging and checkpointing
    "log_level": "INFO",
    "evaluation_interval": 100,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    
    # Checkpointing
    "checkpoint_freq": 50,
    "checkpoint_at_end": True,

    "exploration_config": {
    "type": Curiosity,  # <- Use the Curiosity module for exploring.
    "framework": "torch",
    "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
    "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
    "feature_dim": 53,  # Dimensionality of the generated feature vectors.
    # Setup of the feature net (used to encode observations into feature (latent) vectors).
    "feature_net_config": {
        "fcnet_hiddens": [],
        "fcnet_activation": "relu",
        'framework': 'torch',
        #'device': 'cuda:0'
    },
    "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
    "inverse_net_activation": "relu",  # Activation of the "inverse" model.
    "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
    "forward_net_activation": "relu",  # Activation of the "forward" model.
    "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
    # Specify, which exploration sub-type to use (usually, the algo's "default"
    # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
    "sub_exploration": {
        "type": "StochasticSampling",
    }
},
}

ENV_CONFIGS = {
    "b_line_attack": {
        "red_agent": "B_lineAgent",
        "max_steps": 100,
    },
    "red_meander_attack": {
        "red_agent": "RedMeanderAgent", 
        "max_steps": 100,
    }
}

TRAINING_CONFIG = {
    "max_episodes": 500000,
    "max_timesteps": 100,
    "update_timestep": 20000,
    "save_interval": 200,
    "print_interval": 50,
    "checkpoint_dir": "Models/rllib_ppo",
    "results_dir": "results/rllib_ppo",
}

EVALUATION_CONFIG = {
    "num_episodes": [30, 50, 100],
    "render": False,
    "save_videos": False,
    "video_dir": "videos",
}

def get_config(config_name: str = "base") -> Dict[str, Any]:
    if config_name == "ppo": return PPO_CONFIG.copy()
    elif config_name == "ppo_icm": return PPO_CONFIG.copy()
    elif config_name == "training": return TRAINING_CONFIG.copy()
    elif config_name == "evaluation": return EVALUATION_CONFIG.copy()