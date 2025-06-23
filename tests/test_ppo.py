import pytest

torch = pytest.importorskip("torch")
from PPO.Recurrent_ActorCritic import RecurrentActorCritic


def test_act():
    model = RecurrentActorCritic(state_dim=4, action_dim=2)
    h = model.init_hidden()
    state = torch.zeros(1, 4)
    action, h2 = model.act(state, h)
    assert action.item() in {0, 1}
    assert h2.shape == h.shape
