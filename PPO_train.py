import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import FurutaEnv

def make_env():
    env = FurutaEnv()
    return TimeLimit(env, max_episode_steps=1000)

if __name__ == "__main__":
    # Create 4 parallel environments for faster RL training
    num_cpu = 4
    vec_env = make_vec_env(make_env, n_envs=num_cpu)

    print("Initializing PPO Model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./furuta_tb/"
    )

    print("Training policy... (This will take a few minutes)")
    # 500,000 steps is sufficient for basic swing up
    model.learn(total_timesteps=500_000)

    # Save the trained policy
    model.save("furuta_ppo_model")
    print("Training complete! Model saved as 'furuta_ppo_model.zip'.")