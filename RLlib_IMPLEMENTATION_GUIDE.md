# RLlib PPO Implementation Guide

This guide explains how to use the new RLlib PPO implementation for training agents that can handle multiple attack types in the Cage Challenge 2 environment.

## 🎯 Overview

The RLlib implementation provides:
- **Single Agent, Multiple Attacks**: One PPO agent trained to handle both B-line and Red Meander attacks
- **Random Attack Selection**: Each episode randomly selects an attack type
- **RLlib Integration**: Uses Ray RLlib for scalable, distributed training
- **Complete Pipeline**: Training, evaluation, and inference scripts

## 📁 Structure

```
rllib_agent/
├── __init__.py                 # Package initialization
├── configs.py                  # All configuration settings
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── example.py                  # Usage examples
├── test_env.py                 # Environment testing
├── requirements.txt            # Dependencies
├── README.md                   # Detailed documentation
└── envs/                       # Environment implementations
    ├── __init__.py
    ├── random_attack_wrapper.py    # Main wrapper (randomly selects attacks)
    ├── attack_env1.py             # B-line attack environment
    └── attack_env2.py             # Red Meander attack environment
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install RLlib and dependencies
pip install -r rllib_agent/requirements.txt

# Or install RLlib separately
pip install "ray[rllib]>=2.8.0"
```

### 2. Test the Environment

```bash
# Test that everything works
python rllib_agent/test_env.py
```

### 3. Run Training

```bash
# Basic training
python rllib_agent/train.py

# Training with custom parameters
python rllib_agent/train.py \
    --seed 42 \
    --max-episodes 100000

# Resume from checkpoint
python rllib_agent/train.py \
    --checkpoint path/to/checkpoint \
    --max-episodes 500000
```

### 4. Evaluate Trained Agent

```bash
# Evaluate a trained agent
python rllib_agent/evaluate.py path/to/checkpoint

# Evaluate with custom parameters
python rllib_agent/evaluate.py path/to/checkpoint \
    --num-episodes 200 \
    --output results.json
```

## 🔧 Key Features

### Random Attack Wrapper

The `RandomAttackWrapper` is the core innovation:

```python
class RandomAttackWrapper(gym.Env):
    def __init__(self, config=None):
        # Create both attack environments
        self.attack_envs = {
            'b_line': BLineAttackEnv(config),
            'red_meander': RedMeanderAttackEnv(config)
        }
    
    def reset(self):
        # Randomly select attack type each episode
        self.current_attack_type = random.choice(list(self.attack_envs.keys()))
        self.current_env = self.attack_envs[self.current_attack_type]
        return self.current_env.reset()
```

### Configuration Management

All settings are centralized in `configs.py`:

```python
PPO_CONFIG = {
    "env": "RandomAttackEnv",
    "framework": "torch",
    "num_workers": 4,
    "train_batch_size": 4000,
    "lr": 3e-4,
    "gamma": 0.99,
    # ... more parameters
}
```

### Training Pipeline

The training script provides:
- **Automatic checkpointing**: Saves models every 50 iterations
- **TensorBoard logging**: Automatic metric logging
- **Resume capability**: Can resume from any checkpoint
- **Progress tracking**: Real-time training progress

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir results/rllib_ppo
```

### Checkpoints

Checkpoints are saved to `Models/rllib_ppo/` with format:
- `checkpoint_<iteration>/`
- Contains model weights, optimizer state, and metadata

### Training Metrics

RLlib automatically tracks:
- Episode reward mean/std
- Policy loss
- Value function loss
- Entropy
- Learning rate
- Training time

## 🧪 Evaluation

The evaluation script provides comprehensive metrics:

```bash
python rllib_agent/evaluate.py checkpoint_path --num-episodes 100
```

**Output includes:**
- Overall performance statistics
- Per-attack-type breakdown
- Episode-by-episode results
- JSON export for further analysis

## 🔄 Comparison with Original Implementation

| Feature | Original PPO | RLlib PPO |
|---------|-------------|-----------|
| **Architecture** | Custom PPO implementation | RLlib's optimized PPO |
| **Attack Handling** | Separate agents per attack | Single agent, multiple attacks |
| **Scalability** | Single-threaded | Multi-worker, distributed |
| **Monitoring** | Basic logging | TensorBoard + RLlib metrics |
| **Checkpointing** | Manual PyTorch saves | Automatic Ray checkpoints |
| **Configuration** | Hardcoded parameters | Centralized config system |

## 🎯 Advantages of RLlib Implementation

1. **Unified Policy**: Single agent learns to handle both attack types
2. **Better Generalization**: Random attack selection improves robustness
3. **Scalability**: Multi-worker training for faster convergence
4. **Production Ready**: RLlib's battle-tested implementation
5. **Easy Monitoring**: Built-in TensorBoard integration
6. **Flexible Configuration**: Easy to experiment with hyperparameters

## 🔧 Customization

### Adding New Attack Types

1. Create new environment in `envs/`:
```python
class NewAttackEnv(gym.Env):
    def __init__(self, config=None):
        self.cyborg = CybORG(path, 'sim', agents={'Red': NewAttackAgent})
        # ... implementation
```

2. Add to wrapper:
```python
self.attack_envs = {
    'b_line': BLineAttackEnv(config),
    'red_meander': RedMeanderAttackEnv(config),
    'new_attack': NewAttackEnv(config)  # Add here
}
```

### Modifying Training Parameters

```python
from rllib_agent.configs import get_config, update_config

config = get_config("default")
config = update_config(config, {
    "lr": 1e-4,
    "train_batch_size": 8000,
    "num_workers": 8
})
```

## 🚨 Troubleshooting

### Common Issues

1. **Ray initialization errors**:
   ```bash
   ray.shutdown()
   ray.init()
   ```

2. **Memory issues**:
   - Reduce `num_workers` or `train_batch_size`
   - Use fewer parallel environments

3. **Environment import errors**:
   - Ensure CybORG is properly installed
   - Check Python path includes parent directory

### Performance Tips

- **Multi-GPU**: Set `num_gpus` in config
- **Parallel Workers**: Increase `num_workers` for faster training
- **Batch Size**: Larger `train_batch_size` for more stable training

## 📈 Expected Results

With proper training, you should see:
- **Convergence**: Episode rewards increasing over time
- **Attack Handling**: Similar performance on both attack types
- **Generalization**: Agent performs well on unseen scenarios

## 🎉 Next Steps

1. **Test the environment**: Run `python rllib_agent/test_env.py`
2. **Start training**: Run `python rllib_agent/train.py`
3. **Monitor progress**: Use TensorBoard
4. **Evaluate results**: Use `python rllib_agent/evaluate.py`
5. **Experiment**: Try different configurations and hyperparameters

The RLlib implementation provides a robust, scalable foundation for training agents that can handle multiple attack types in the Cage Challenge 2 environment. 