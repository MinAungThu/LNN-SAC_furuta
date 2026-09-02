from collections import deque

import gymnasium as gym
import numpy as np


class HistoryWrapper(gym.Wrapper):
    """Stacks the last `history_len` raw observations into one flat vector.

    A CfC (or any recurrent) feature extractor unrolls over this window each
    forward pass, giving it genuine short-horizon temporal context without
    needing a stateful policy across the SAC replay buffer.
    """

    def __init__(self, env: gym.Env, history_len: int = 16):
        super().__init__(env)
        self.history_len = history_len
        self.obs_dim = env.observation_space.shape[0]
        self.history: deque[np.ndarray] = deque(maxlen=history_len)

        low = np.tile(env.observation_space.low, history_len)
        high = np.tile(env.observation_space.high, history_len)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self.history), axis=0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        for _ in range(self.history_len - 1):
            self.history.append(np.zeros(self.obs_dim, dtype=np.float32))
        self.history.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs)
        return self._stacked(), reward, terminated, truncated, info
