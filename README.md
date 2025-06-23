# PPO Baseline

This repository contains a minimal implementation of a recurrent PPO agent. The
agent is defined in `Agents/PPOAgent.py` and uses the network in
`PPO/Recurrent_ActorCritic.py`.

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
