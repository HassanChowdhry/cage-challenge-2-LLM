import os
import sys
import argparse
import ray
from ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import numpy as np
import json
from collections import defaultdict
import time
import logging

from .envs.random_attack_wrapper import RandomAttackWrapper
from .configs import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def env_creator(env_config):
    """
    Environment creator function for RLlib
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        RandomAttackWrapper environment instance
    """
    return RandomAttackWrapper(env_config)

def register_environments():
    """Register the custom environment with RLlib"""
    register_env("RandomAttackEnv", env_creator)
    logger.info("✓ Registered RandomAttackEnv with RLlib")

def load_agent(checkpoint_path, config):
    """
    Load a trained PPO agent from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Agent configuration
        
    Returns:
        Loaded PPOTrainer instance
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Register environments
    register_environments()
    
    # Create agent
    agent = PPOTrainer(config=config, env="RandomAttackEnv")
    
    # Restore from checkpoint
    agent.restore(checkpoint_path)
    logger.info(f"✓ Loaded agent from checkpoint: {checkpoint_path}")
    
    return agent

def evaluate_episode(agent, env, max_steps=100, render=False):
    """
    Evaluate a single episode
    
    Args:
        agent: Trained PPO agent
        env: Environment instance
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        
    Returns:
        Dictionary with episode results
    """
    obs = env.reset()
    total_reward = 0
    episode_length = 0
    done = False
    
    while not done and episode_length < max_steps:
        # Get action from agent
        action = agent.compute_action(obs)
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        if render:
            env.render()
    
    return {
        'episode_reward': total_reward,
        'episode_length': episode_length,
        'attack_type': info.get('attack_type', 'unknown'),
        'done': done
    }

def evaluate_agent(agent, config, num_episodes=100, render=False):
    """
    Evaluate the agent over multiple episodes
    
    Args:
        agent: Trained PPO agent
        config: Evaluation configuration
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"🔍 Evaluating agent over {num_episodes} episodes...")
    
    # Create environment
    env = RandomAttackWrapper(config.get('env_config', {}))
    
    # Evaluation results
    results = {
        'episodes': [],
        'attack_type_results': defaultdict(list),
        'summary': {}
    }
    
    for episode in range(num_episodes):
        logger.debug(f"Episode {episode + 1}/{num_episodes}")
        
        # Evaluate episode
        episode_result = evaluate_episode(
            agent, env, 
            max_steps=config.get('max_steps', 100),
            render=render
        )
        
        # Store results
        results['episodes'].append(episode_result)
        results['attack_type_results'][episode_result['attack_type']].append(
            episode_result['episode_reward']
        )
    
    # Calculate summary statistics
    all_rewards = [ep['episode_reward'] for ep in results['episodes']]
    results['summary'] = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'min_reward': np.min(all_rewards),
        'max_reward': np.max(all_rewards),
        'total_episodes': len(results['episodes']),
        'attack_type_stats': {}
    }
    
    # Calculate stats per attack type
    for attack_type, rewards in results['attack_type_results'].items():
        results['summary']['attack_type_stats'][attack_type] = {
            'count': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    env.close()
    logger.info("✅ Evaluation completed!")
    
    return results

def save_results(results, output_path):
    """
    Save evaluation results to file
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return obj
    
    # Convert results
    results_json = json.loads(json.dumps(results, default=convert_numpy))
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"💾 Results saved to: {output_path}")

def print_summary(results):
    """
    Print evaluation summary
    
    Args:
        results: Evaluation results dictionary
    """
    summary = results['summary']
    
    logger.info("\n" + "="*50)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Episodes: {summary['total_episodes']}")
    logger.info(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    logger.info(f"Reward Range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
    
    logger.info("\n📈 Attack Type Breakdown:")
    for attack_type, stats in summary['attack_type_stats'].items():
        logger.info(f"  {attack_type.upper()}:")
        logger.info(f"    Episodes: {stats['count']}")
        logger.info(f"    Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        logger.info(f"    Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    
    logger.info("="*50)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file to evaluate"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes during evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name to use"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return
    
    # Get configuration
    config = get_config(args.config)
    
    # Load agent
    agent = load_agent(args.checkpoint_path, config)
    
    # Evaluate agent
    results = evaluate_agent(
        agent, 
        config, 
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)
    
    logger.info("🎉 Evaluation completed!")

if __name__ == "__main__":
    main() 