"""

"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_evaluation():
    """Example of basic evaluation with Gemini backend."""
    logger.info("Running basic evaluation example...")
    
    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set, skipping Gemini evaluation")
        return
    
    try:
        from .evaluate import LLMAgentEvaluator
        
        config = {
            'backend_type': 'gemini',
            'backend_config': {
                'model_name': 'gemini-2.0-flash-exp',
                'temperature': 0.0,
                'max_tokens': 50
            },
            'prompt_name': 'zero_shot',
            'episodes': 3,
            'max_steps': 50,
            'red_agent': 'random',
            'output_file': 'example_gemini_results.json'
        }
        
        evaluator = LLMAgentEvaluator(config)
        results = evaluator.evaluate()
        
        logger.info("Basic evaluation completed!")
        logger.info(f"Success rate: {results.summary['success_rate']:.2%}")
        logger.info(f"Average reward: {results.summary['avg_reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Basic evaluation failed: {e}")

def example_local_evaluation():
    """Example of evaluation with local backend."""
    logger.info("Running local evaluation example...")
    
    try:
        from .evaluate import LLMAgentEvaluator
        
        config = {
            'backend_type': 'local',
            'backend_config': {
                'model_name': 'microsoft/DialoGPT-medium',
                'temperature': 0.0,
                'max_tokens': 30
            },
            'prompt_name': 'simple',
            'episodes': 2,
            'max_steps': 30,
            'red_agent': 'sleep',
            'output_file': 'example_local_results.json'
        }
        
        evaluator = LLMAgentEvaluator(config)
        results = evaluator.evaluate()
        
        logger.info("Local evaluation completed!")
        logger.info(f"Success rate: {results.summary['success_rate']:.2%}")
        logger.info(f"Average reward: {results.summary['avg_reward']:.2f}")
        
    except Exception as e:
        logger.warning(f"Local evaluation failed (expected if transformers not installed): {e}")

def example_different_prompts():
    """Example of evaluating different prompt templates."""
    logger.info("Running prompt comparison example...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set, skipping prompt comparison")
        return
    
    try:
        from .evaluate import LLMAgentEvaluator
        
        prompts = ['zero_shot', 'adaptive', 'tactical', 'simple']
        results_summary = {}
        
        for prompt_name in prompts:
            logger.info(f"Testing prompt: {prompt_name}")
            
            config = {
                'backend_type': 'gemini',
                'backend_config': {
                    'model_name': 'gemini-2.0-flash-exp',
                    'temperature': 0.0,
                    'max_tokens': 50
                },
                'prompt_name': prompt_name,
                'episodes': 2,
                'max_steps': 30,
                'red_agent': 'random',
                'output_file': f'prompt_comparison_{prompt_name}.json'
            }
            
            evaluator = LLMAgentEvaluator(config)
            results = evaluator.evaluate()
            
            results_summary[prompt_name] = {
                'success_rate': results.summary['success_rate'],
                'avg_reward': results.summary['avg_reward']
            }
        
        # Print comparison
        logger.info("\nPrompt Comparison Results:")
        for prompt_name, metrics in results_summary.items():
            logger.info(f"  {prompt_name}: success={metrics['success_rate']:.2%}, "
                       f"reward={metrics['avg_reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Prompt comparison failed: {e}")

def example_different_red_agents():
    """Example of evaluating against different red agents."""
    logger.info("Running red agent comparison example...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set, skipping red agent comparison")
        return
    
    try:
        from .evaluate import LLMAgentEvaluator
        
        red_agents = ['sleep', 'bline', 'meander', 'random']
        results_summary = {}
        
        for red_agent in red_agents:
            logger.info(f"Testing against red agent: {red_agent}")
            
            config = {
                'backend_type': 'gemini',
                'backend_config': {
                    'model_name': 'gemini-2.0-flash-exp',
                    'temperature': 0.0,
                    'max_tokens': 50
                },
                'prompt_name': 'zero_shot',
                'episodes': 2,
                'max_steps': 30,
                'red_agent': red_agent,
                'output_file': f'red_agent_comparison_{red_agent}.json'
            }
            
            evaluator = LLMAgentEvaluator(config)
            results = evaluator.evaluate()
            
            results_summary[red_agent] = {
                'success_rate': results.summary['success_rate'],
                'avg_reward': results.summary['avg_reward']
            }
        
        # Print comparison
        logger.info("\nRed Agent Comparison Results:")
        for red_agent, metrics in results_summary.items():
            logger.info(f"  {red_agent}: success={metrics['success_rate']:.2%}, "
                       f"reward={metrics['avg_reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Red agent comparison failed: {e}")

def main():
    """Run all examples."""
    logger.info("Starting LLM evaluation examples...")
    
    # Run examples
    example_basic_evaluation()
    example_local_evaluation()
    example_different_prompts()
    example_different_red_agents()
    
    logger.info("\nAll examples completed!")
    logger.info("\nTo run evaluation from command line:")
    logger.info("  python -m LLM.evaluate --backend gemini --episodes 10")
    logger.info("  python -m LLM.evaluate --backend local --episodes 5")

if __name__ == "__main__":
    main() 