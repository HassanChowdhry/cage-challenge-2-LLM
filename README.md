# PPO Baseline

This repository contains a minimal implementation of a recurrent PPO agent. Two
agent variants are provided:

* **`Agents/PPOAgent.py`** – a basic wrapper around the lightweight PPO
  algorithm located in `PPO/PPO.py` (adapted from the `PPO_discrete.py`
  example).  It uses the network defined in `PPO/Recurrent_ActorCritic.py`.

* **`Agents/PPOICMAgent.py`** – extends the base agent with an Intrinsic
  Curiosity Module (ICM) to encourage exploration.  This demonstrates how the
  template curiosity module can be integrated into the training loop.

## Intrinsic Curiosity Module

An implementation of the Intrinsic Curiosity Module (ICM) is provided in
`Agents/ICM_template.py`.  The module contains both the forward and inverse
prediction heads described in the original paper.  To integrate curiosity
driven exploration:

1. Instantiate the `ICM` module in your agent and add its parameters to the
   optimizer.
2. Call the module with the previous state, next state and one-hot encoded
   action to obtain an intrinsic reward and auxiliary predictions.
3. Combine the curiosity reward with the environment reward and include the
   forward and inverse losses when updating the policy.

These steps allow you to extend the base agent with curiosity while keeping its
core training loop simple.

## LLM Agents

The `LLM` package contains experimental agents that make decisions with large
language models. Four strategies are provided:

1. **OneShotAgent** – prompts an LLM for an action given the current state.
2. **RAGAgent** – augments the prompt with recently observed transitions.
3. **FinetunedAgent** – assumes the underlying model has been fine-tuned on
   relevant data.
4. **RAGFinetunedAgent** – combines retrieval augmented prompts with a
   fine-tuned model.

All agents log the observed state, chosen action and reward to a small SQLite
database via `TrajectoryDatabase`.
