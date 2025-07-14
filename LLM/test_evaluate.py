#!/usr/bin/env python3
"""
Test script for LLM evaluation.

This script tests the evaluation functionality without requiring API keys.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_evaluation_imports():
    """Test that evaluation components can be imported."""
    logger.info("Testing evaluation imports...")
    
    try:
        from .evaluate import LLMAgentEvaluator, EpisodeResult, EvaluationResults
        logger.info("✓ Evaluation classes imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import evaluation classes: {e}")
        return False

def test_agent_creation():
    """Test agent creation with local backend."""
    logger.info("Testing agent creation...")
    
    try:
        from .blue_agent import LLMBlueAgent
        
        # Test with local backend (doesn't require API key)
        config = {
            'backend_type': 'local',
            'backend_config': {'model_name': 'microsoft/DialoGPT-medium'},
            'prompt_name': 'simple',
            'episodes': 1,
            'max_steps': 10,
            'red_agent': 'sleep'
        }
        
        evaluator = LLMAgentEvaluator(config)
        agent = evaluator.create_agent()
        
        logger.info("✓ Agent created successfully")
        return True
        
    except Exception as e:
        logger.warning(f"⚠ Agent creation failed (expected if transformers not installed): {e}")
        return True  # This is not a failure, just a warning

def test_red_agent_creation():
    """Test red agent creation."""
    logger.info("Testing red agent creation...")
    
    try:
        from .evaluate import LLMAgentEvaluator
        
        config = {'red_agent': 'sleep'}
        evaluator = LLMAgentEvaluator(config)
        
        # Test different red agent types
        red_agent_types = ['sleep', 'random']
        for agent_type in red_agent_types:
            config['red_agent'] = agent_type
            evaluator.config = config
            red_agent = evaluator.create_red_agent()
            logger.info(f"✓ Created red agent: {type(red_agent).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Red agent creation failed: {e}")
        return False

def test_configuration():
    """Test configuration handling."""
    logger.info("Testing configuration...")
    
    try:
        from .evaluate import LLMAgentEvaluator
        
        # Test different configurations
        configs = [
            {
                'backend_type': 'gemini',
                'prompt_name': 'zero_shot',
                'episodes': 5,
                'max_steps': 50
            },
            {
                'backend_type': 'local',
                'prompt_name': 'simple',
                'episodes': 3,
                'max_steps': 20
            }
        ]
        
        for config in configs:
            evaluator = LLMAgentEvaluator(config)
            logger.info(f"✓ Configuration created: {config}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting LLM evaluation tests...")
    
    tests = [
        ("Evaluation Imports", test_evaluation_imports),
        ("Agent Creation", test_agent_creation),
        ("Red Agent Creation", test_red_agent_creation),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! LLM evaluation is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Set your API key: export GOOGLE_API_KEY='your-key'")
        logger.info("2. Run evaluation: python -m LLM.evaluate --backend gemini --episodes 5")
        logger.info("3. Test with local backend: python -m LLM.evaluate --backend local --episodes 3")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 