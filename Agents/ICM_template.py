"""Intrinsic Curiosity Module (ICM).

This module implements the curiosity mechanism from `Curiosity-driven
Exploration by Self-supervised Prediction` (Pathak et al.).  It contains
two components:

1. **Forward Model** – predicts the feature representation of the next
   observation given the current observation and action.
2. **Inverse Model** – predicts the action taken given the current and
   next feature representations.

The prediction error of the forward model is used as an intrinsic reward
to encourage exploration of unseen states.
"""

import torch
import torch.nn as nn


class ICM(nn.Module):
    """Simple implementation of the Intrinsic Curiosity Module."""

    def __init__(self, state_dim: int, action_dim: int, feature_dim: int = 128):
        super().__init__()

        self.feature_dim = feature_dim

        # Encoder that maps raw observations to a latent feature space
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
        )

        # Forward model predicts next state features from current features
        # and the executed action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Inverse model predicts the taken action from the pair of
        # consecutive state features
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
        )

        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        state_prev: torch.Tensor,
        state_next: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return intrinsic reward and predictions.

        Parameters
        ----------
        state_prev : Tensor
            Observation at time *t* with shape ``(B, state_dim)``.
        state_next : Tensor
            Observation at time *t+1* with shape ``(B, state_dim)``.
        action_onehot : Tensor
            One-hot encoded action taken at time *t* with shape
            ``(B, action_dim)``.

        Returns
        -------
        reward : Tensor
            Scalar intrinsic reward for each item in the batch ``(B,)``.
        pred_action_logits : Tensor
            Logits predicting the taken action ``(B, action_dim)``.
        pred_next_feat : Tensor
            Predicted feature representation of ``state_next``.
        """

        # Encode states into feature space
        feat_prev = self.encoder(state_prev)
        feat_next = self.encoder(state_next)

        # Predict next feature vector given previous feature and action
        forward_input = torch.cat([feat_prev, action_onehot], dim=-1)
        pred_next_feat = self.forward_model(forward_input)

        # Predict the taken action (inverse dynamics) from state features
        inverse_input = torch.cat([feat_prev, feat_next], dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)

        # Curiosity reward is the squared error of the forward model
        reward = 0.5 * self.mse(pred_next_feat, feat_next).mean(dim=-1)

        return reward, pred_action_logits, pred_next_feat
