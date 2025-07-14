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

from .blue_agent import LLMBlueAgent
from .backends import create_backend
from .prompts import get_prompt_template, PROMPT_TEMPLATES

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
        
    def create_agent(self) -> LLMBlueAgent:
        agent = LLMBlueAgent(
            backend_type=self.config.get('backend_type', 'gemini'),
            backend_config=self.config.get('backend_config', {}),
            prompt_template=self.config.get('prompt_template'),
            prompt_name=self.config.get('prompt_name', 'zero_shot'),
            max_history_length=self.config.get('max_history_length', 10)
        )
        logger.info(f"Created LLM agent with backend: {self.config.get('backend_type', 'gemini')}")
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
    
    def run_episode(self, agent: LLMBlueAgent, episode_id: int) -> EpisodeResult:
        logger.info(f"Starting episode {episode_id}")
        
        start_time = time.time()
        actions_taken = []
        total_reward = 0.0
        steps = 0
    
        red_agent = self.create_red_agent()
        cyborg, env = self.create_environment(red_agent)
        
        state = env.reset()
        
        for step in range(self.config.get('max_steps', 100)):
            action = agent.get_action(state)
            actions_taken.append(f"Step {step}: Action {action}")
            
            next_state, reward, done, _ = env.step(action)
            
            result = Results(observation=state, action=action, reward=reward)
            agent.train(result)
            
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
    
    def evaluate(self) -> EvaluationResults:
        logger.info("Starting LLM Blue Agent evaluation")
        logger.info(f"Configuration: {self.config}")
    
        agent = self.create_agent()
        
        episode_results = []
        for episode_id in range(self.config.get('episodes', 10)):
            result = self.run_episode(agent, episode_id)
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
    parser = argparse.ArgumentParser(description="Evaluate LLM Blue Agent")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "local"], help="LLM backend to use")
    parser.add_argument("--prompt", default="zero_shot", choices=list(PROMPT_TEMPLATES.keys()), help="Prompt template to use")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--red-agent", default="random", choices=["random", "sleep", "bline", "meander"], help="Red agent type")
    parser.add_argument("--output", default="llm_evaluation_results.json", help="Output file for results")
    parser.add_argument("--api-key", help="API key for the LLM backend")
    
    args = parser.parse_args()
    
    backend_config = {}
    if args.api_key:
        if args.backend == "gemini":
            backend_config["api_key"] = args.api_key
    
    config = {
        'backend_type': args.backend,
        'backend_config': backend_config,
        'prompt_name': args.prompt,
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'red_agent': args.red_agent,
        'output_file': args.output,
        'max_history_length': 10
    }
    
    random.seed(0)
    np.random.seed(0)
    
    evaluator = LLMAgentEvaluator(config)
    results = evaluator.evaluate()
    
    print_results(results)