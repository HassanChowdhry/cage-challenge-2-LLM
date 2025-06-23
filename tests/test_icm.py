import pytest

torch = pytest.importorskip("torch")
from Agents.ICM_template import ICM

def test_icm_shapes():
    icm = ICM(state_dim=4, action_dim=2)
    s1 = torch.zeros(1, 4)
    s2 = torch.ones(1, 4)
    action = torch.tensor([[1., 0.]])
    r, action_logits, next_feat = icm(s1, s2, action)
    assert r.shape == (1,)
    assert action_logits.shape == (1, 2)
    assert next_feat.shape[0] == 1
    assert next_feat.shape[1] == icm.feature_dim
