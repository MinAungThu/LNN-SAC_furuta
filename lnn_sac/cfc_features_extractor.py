import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from cfc import CfCCell


class CfCFeaturesExtractor(BaseFeaturesExtractor):
    """Unrolls a CfC cell over a stacked observation history.

    Expects flattened input of shape (history_len * obs_dim,) as produced by
    HistoryWrapper; reshapes it back into a sequence and feeds it through the
    liquid cell step by step, returning the final hidden state as the feature
    vector for SAC's actor/critic heads.
    """

    def __init__(self, observation_space, obs_dim: int, history_len: int, hidden_size: int = 64):
        super().__init__(observation_space, features_dim=hidden_size)
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.hidden_size = hidden_size
        self.cell = CfCCell(obs_dim, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        seq = observations.view(batch, self.history_len, self.obs_dim)
        h = torch.zeros(batch, self.hidden_size, device=observations.device, dtype=observations.dtype)
        for t in range(self.history_len):
            h = self.cell(seq[:, t, :], h)
        return h
