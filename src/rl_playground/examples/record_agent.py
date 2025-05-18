"""
Train an agent using Stable Baselines3 and record its performance in a video.

https://gymnasium.farama.org/main/introduction/record_agent/
https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
"""

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import A2C


def main():
    # Environment for training
    train_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Train the model
    model = A2C("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=10_000)
    train_env.close()

    # Environment for evaluation and video recording
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env, video_folder="results/cartpole-agent", name_prefix="eval", episode_trigger=lambda x: True
    )
    eval_env = RecordEpisodeStatistics(eval_env, buffer_length=1)
    obs, info = eval_env.reset()
    episode_over = False
    while not episode_over:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        episode_over = terminated or truncated

    eval_env.close()

    print(f"Episode time taken: {eval_env.time_queue}")
    print(f"Episode total rewards: {eval_env.return_queue}")
    print(f"Episode lengths: {eval_env.length_queue}")


if __name__ == "__main__":
    main()
