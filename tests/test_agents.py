import pytest

torch = pytest.importorskip("torch")

from Agents import PPOAgent, PPOICMAgent


class DummyResults:
    def __init__(self, obs, reward, done):
        self.observation = obs
        self.reward = reward
        self.done = done


def test_icm_agent_step():
    agent = PPOICMAgent(state_dim=4, action_dim=2)
    obs = [0.0, 0.0, 0.0, 0.0]
    agent.get_action(obs)
    results = DummyResults(obs, 1.0, True)
    agent.train(results)


def test_base_agent_update():
    agent = PPOAgent(state_dim=4, action_dim=2)
    obs = [0.0, 0.0, 0.0, 0.0]
    agent.get_action(obs)
    results = DummyResults(obs, 1.0, True)
    agent.train(results)
