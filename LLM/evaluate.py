import os
import sys
import argparse
import logging
import time
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import statistics

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.SimpleAgents import BlueMonitorAgent
from CybORG.Shared.Results import Results
import inspect

from LLM.blue_agent import LLMAgent, LLMPolicy
from LLM.backend import create_backend
from LLM.configs.prompts import PROMPT_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EpisodeResult:
    episode_id: int
    total_reward: float
    steps: int
    actions_taken: List[str]
    final_state: str
    success: bool
    duration: float
    red_agent_type: str

@dataclass
class EvaluationResults:
    config: Dict[str, Any]
    episodes: List[EpisodeResult]
    summary: Dict[str, Any]

class LLMAgentEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
        
        PATH = str(inspect.getfile(CybORG))
        self.PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        
    def create_agent(self, env) -> LLMAgent:
        # env: ChallengeWrapper
        obs_space = env.observation_space
        agent = LLMAgent(
            name="Blue",
            policy=LLMPolicy,
            obs_space=obs_space,
            llm_config=config
        )
        logger.info(f"Created LLM agent with LLMPolicy and obs_space: {obs_space}")
        return agent
    
    def create_red_agent(self) -> Any:
        red_agent_type = self.config.get('red_agent', 'random')
        
        if red_agent_type == 'sleep':
            return SleepAgent()
        elif red_agent_type == 'bline':
            return B_lineAgent()
        elif red_agent_type == 'meander':
            return RedMeanderAgent()
        elif red_agent_type == 'random':
            return random.choice([B_lineAgent, RedMeanderAgent, SleepAgent])
        else:
            return SleepAgent()
    
    def create_environment(self, red_agent) -> tuple:
        cyborg = CybORG(self.PATH, 'sim', agents={'Red': red_agent})
        env = ChallengeWrapper(env=cyborg, agent_name="Blue")
        logger.info(f"Created CybORG environment with red agent: {type(red_agent).__name__}")
        return cyborg, env
    
    def run_episode(self, agent: LLMAgent, episode_id: int, env=None) -> EpisodeResult:
        logger.info(f"Starting episode {episode_id}")
        start_time = time.time()
        actions_taken = []
        total_reward = 0.0
        steps = 0
        red_agent = self.create_red_agent()
        cyborg, env = self.create_environment(red_agent) if env is None else (None, env)
        state = env.reset()
        for step in range(100):
            action = agent.get_action(state)
            actions_taken.append(f"Step {step}: Action {action}")
            next_state, reward, done, _ = env.step(action)
            result = Results(observation=state, action=action, reward=reward)
            state = next_state
            total_reward += reward
            steps += 1
            if done: break
        duration = time.time() - start_time
        success = total_reward > 0 and steps > 0
        result = EpisodeResult(
            episode_id=episode_id,
            total_reward=total_reward,
            steps=steps,
            actions_taken=actions_taken,
            final_state=str(state),
            success=success,
            duration=duration,
            red_agent_type=type(red_agent).__name__
        )
        logger.info(f"Episode {episode_id} completed: reward={total_reward:.2f}, steps={steps}, success={success}")
        agent.end_episode()
        return result
    
    def evaluate(self, episodes=None, max_steps=None) -> EvaluationResults:
        logger.info("Starting LLM Blue Agent evaluation")
        # logger.info(f"Configuration: {self.config}")
        # Create a temp env to get obs_space
        red_agent = self.create_red_agent()
        _, env = self.create_environment(red_agent)
        agent = self.create_agent(env)
        episode_results = []
        n_episodes = episodes if episodes is not None else self.config.get('episodes', 10)
        max_steps = max_steps if max_steps is not None else self.config.get('max_steps', 100)
        
        for episode_id in range(n_episodes):
            result = self.run_episode(agent, episode_id, env=env)
            episode_results.append(result)
            
        summary = self._calculate_summary(episode_results)
        results = EvaluationResults(
            config=self.config,
            episodes=episode_results,
            summary=summary
        )
        self._save_results(results)
        logger.info("Evaluation completed")
        return results
    
    def _calculate_summary(self, episode_results: List[EpisodeResult]) -> Dict[str, Any]:
        if not episode_results:
            return {"error": "No episodes completed"}
        
        rewards = [ep.total_reward for ep in episode_results]
        steps = [ep.steps for ep in episode_results]
        durations = [ep.duration for ep in episode_results]
        successes = [ep.success for ep in episode_results]
        
        red_agent_results = {}
        for ep in episode_results:
            agent_type = ep.red_agent_type
            if agent_type not in red_agent_results:
                red_agent_results[agent_type] = []
            red_agent_results[agent_type].append(ep.total_reward)
        
        red_agent_stats = {}
        for agent_type, agent_rewards in red_agent_results.items():
            red_agent_stats[agent_type] = {
                "count": len(agent_rewards),
                "avg_reward": statistics.mean(agent_rewards),
                "success_rate": sum(1 for r in agent_rewards if r > 0) / len(agent_rewards)
            }
        
        summary = {
            "total_episodes": len(episode_results),
            "success_rate": sum(successes) / len(successes),
            "avg_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_steps": statistics.mean(steps),
            "avg_duration": statistics.mean(durations),
            "total_duration": sum(durations),
            "red_agent_breakdown": red_agent_stats
        }
        
        return summary
    
    def _save_results(self, results: EvaluationResults):
        try:
            output_file = self.config.get('output_file', 'llm_evaluation_results.json')
            data = {
                "config": {
                    "backend_type": results.config.get('backend_type'),
                    "prompt_name": results.config.get('prompt_name'),
                    "episodes": results.config.get('episodes'),
                    "max_steps": results.config.get('max_steps'),
                    "red_agent": results.config.get('red_agent')
                },
                "summary": results.summary,
                "episodes": [
                    {
                        "episode_id": ep.episode_id,
                        "total_reward": ep.total_reward,
                        "steps": ep.steps,
                        "success": ep.success,
                        "duration": ep.duration,
                        "red_agent_type": ep.red_agent_type
                    }
                    for ep in results.episodes
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def print_results(results: EvaluationResults):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("LLM BLUE AGENT EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Backend: {results.config.get('backend_type', 'gemini')}")
    print(f"  Prompt: {results.config.get('prompt_name', 'zero_shot')}")
    print(f"  Episodes: {results.config.get('episodes', 10)}")
    print(f"  Red Agent: {results.config.get('red_agent', 'random')}")
    
    print(f"\nSummary:")
    print(f"  Total Episodes: {results.summary['total_episodes']}")
    print(f"  Success Rate: {results.summary['success_rate']:.2%}")
    print(f"  Average Reward: {results.summary['avg_reward']:.2f}")
    print(f"  Reward Std Dev: {results.summary['std_reward']:.2f}")
    print(f"  Min/Max Reward: {results.summary['min_reward']:.2f} / {results.summary['max_reward']:.2f}")
    print(f"  Average Steps: {results.summary['avg_steps']:.1f}")
    print(f"  Average Duration: {results.summary['avg_duration']:.2f}s")
    print(f"  Total Duration: {results.summary['total_duration']:.2f}s")
    
    if 'red_agent_breakdown' in results.summary:
        print(f"\nRed Agent Breakdown:")
        for agent_type, stats in results.summary['red_agent_breakdown'].items():
            print(f"  {agent_type}: {stats['count']} episodes, "
                  f"avg reward: {stats['avg_reward']:.2f}, "
                  f"success rate: {stats['success_rate']:.2%}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    config = {
        'llm': "local",
        'hyperparams': {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "max_new_tokens": 64, "temperature": 0.9},
        'max_steps': 100,
        'red_agent': "random",
    }
    # config = {
    #     'llm': "gemini",
    #     'hyperparams': {"model_name": "gemini-2.0-flash-lite", "max_new_tokens": 32, "temperature": 0.9},
    #     'max_steps': 100,
    #     'red_agent': "random",
    # }
    random.seed(0)
    np.random.seed(0)
    # for n_episodes in [30, 50, 100]:
    for n_episodes in [5]:
        print(f"\n{'='*20} Running evaluation for {n_episodes} episodes {'='*20}")
        config['episodes'] = n_episodes
        evaluator = LLMAgentEvaluator(config)
        results = evaluator.evaluate(episodes=n_episodes)
        print_results(results)