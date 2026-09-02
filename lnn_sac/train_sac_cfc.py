import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(PARENT_DIR)  # FurutaEnv loads "furuta.xml" via a relative path

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from env import FurutaEnv
from history_wrapper import HistoryWrapper
from cfc_features_extractor import CfCFeaturesExtractor

HISTORY_LEN = 16
OBS_DIM = 5
HIDDEN_SIZE = 64
TOTAL_TIMESTEPS = 150_000  # SAC is off-policy/sample-efficient; smaller budget than the PPO baseline's 500k


def make_env():
    env = FurutaEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    env = HistoryWrapper(env, history_len=HISTORY_LEN)
    return env


if __name__ == "__main__":
    vec_env = make_vec_env(make_env, n_envs=4)

    policy_kwargs = dict(
        features_extractor_class=CfCFeaturesExtractor,
        features_extractor_kwargs=dict(
            obs_dim=OBS_DIM, history_len=HISTORY_LEN, hidden_size=HIDDEN_SIZE
        ),
        net_arch=[64, 64],
    )

    print("Initializing SAC + CfC model...")
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=os.path.join(THIS_DIR, "sac_cfc_tb"),
    )

    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    save_path = os.path.join(THIS_DIR, "furuta_sac_cfc_model")
    model.save(save_path)
    print(f"Training complete! Model saved as '{save_path}.zip'.")
