import inspect, random, numpy as np
import logging, time, copy, pprint
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from prettytable import PrettyTable

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper, BlueTableWrapper
from CybORG.Agents.SimpleAgents import BlueMonitorAgent
from CybORG.Shared.Results import Results

from blue_agent import LLMAgent, LLMPolicy
from utils import EpisodeResult, EvaluationResults, save_results, print_results, calculate_summary, calculate_time_step_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAgentEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
        
        PATH = str(inspect.getfile(CybORG))
        self.PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        
    def create_agent(self, env) -> LLMAgent:
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
            return SleepAgent
        elif red_agent_type == 'bline':
            return B_lineAgent
        elif red_agent_type == 'meander':
            return RedMeanderAgent
        elif red_agent_type == 'random':
            return random.choice([B_lineAgent])
            # return random.choice([RedMeanderAgent])
            # return random.choice([B_lineAgent, RedMeanderAgent, SleepAgent])
        else:
            return SleepAgent()
    
    def create_environment(self, red_agent) -> tuple:
        cyborg = CybORG(self.PATH, 'sim', agents={'Red': red_agent})
        env = ChallengeWrapper(env=cyborg, agent_name="Blue")
        return cyborg, env
    
    def run_episode(self, agent: LLMAgent, episode_id: int, env=None, max_steps:int =100) -> EpisodeResult:
        logger.info(f"Starting episode {episode_id}")
        start_time = time.time()
        actions_taken = []
        total_reward = 0.0
        steps = 0
        red_agent = self.create_red_agent()
        cyborg, env = self.create_environment(red_agent)
        state = env.reset()
        # pprint([action for action in actions if actions[action]])

        for step in tqdm(range(max_steps), desc="Evaluating"):
            action = agent.get_action(state)
            actions_taken.append(f"Step {step}: Action {action}")
            next_state, reward, done, _ = env.step(action)
            result = Results(observation=state, action=action, reward=reward)
            logger.info(f"Step {step}: Action {action} resulted in state: {next_state} and reward: {reward}")
            state = next_state
            total_reward += reward
            steps += 1
            if done: break
            logger.info((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red')), _))
            time.sleep(60)
        duration = time.time() - start_time
        
        result = EpisodeResult(
            episode_id=episode_id,
            total_reward=total_reward,
            steps=steps,
            actions_taken=actions_taken,
            final_state=str(state),
            duration=duration,
            red_agent_type=type(red_agent).__name__,
            max_steps=max_steps
        )
        logger.info(f"Episode {episode_id} completed: reward={total_reward:.2f}, steps={steps}")
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
        max_steps = max_steps if max_steps is not None else self.config.get('max_steps', 30)
        
        for episode_id in range(n_episodes):
            result = self.run_episode(agent, episode_id, max_steps=max_steps)
            episode_results.append(result)
            
        summary = calculate_summary(episode_results)
        results = EvaluationResults(
            config=self.config,
            episodes=episode_results,
            summary=summary
        )
        save_results(self.config, results)
        logger.info("Evaluation completed")
        return results    

if __name__ == "__main__":
    # config = {
    #     'llm': "local",
    #     'hyperparams': {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "max_new_tokens": 1024, "temperature": 0.9},
    #     'max_steps': 100,
    #     'red_agent': "random",
    # }
    config = {
        'llm': "gemini",
        'hyperparams': {"model_name": "gemini-2.0-flash-lite", "temperature": 0.7}, # 0.5 -> 1
        'red_agent': "bline",
        # 'hyperparams': {"model_name": "gemini-2.0-flash-lite", "temperature": 0.9},
        # 'max_steps': 100,
        # 'red_agent': "random",
    }
    random.seed(0)
    np.random.seed(0)
    
    all_episode_results = []
    
    res = []
    for max_steps in [30]:
    # for max_steps in [30, 50, 100]:
        config['max_steps'] = max_steps
        for n_episodes in [1]:
            print(f"\n{'='*20} Running evaluation for {n_episodes} episodes with max_steps={max_steps} {'='*20}")
            config['episodes'] = n_episodes
            evaluator = LLMAgentEvaluator(config)
            results = evaluator.evaluate(episodes=n_episodes, max_steps=max_steps)
            all_episode_results.extend(results.episodes)
        
    # Create comprehensive summary with time step breakdown
    overall_summary = calculate_summary(all_episode_results)
    time_step_results = calculate_time_step_summary(all_episode_results)
    
    comprehensive_results = EvaluationResults(
        config=config,
        episodes=all_episode_results,
        summary=overall_summary,
        time_step_results=time_step_results
    )
    
    save_results(config, comprehensive_results)
    print_results(comprehensive_results)