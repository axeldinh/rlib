from rlib.learning import PPO

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env_kwargs = {
    'id': 'BipedalWalker-v3',
}

actor_kwargs = {'hidden_sizes': [256, 256], 'activation': 'tanh'} 
critic_kwargs = {'hidden_sizes': [256, 256], 'activation': 'tanh'} 

ppo_kwargs = {
    "env_kwargs": env_kwargs,
    "actor_kwargs": actor_kwargs,
    "critic_kwargs": critic_kwargs,
    "num_envs": 10,
    "num_steps_per_iter": 4096,
    "num_updates_per_iter": 10,
    "total_timesteps": 4096 * 4_000,
    #"max_episode_length": 2_000,
    "test_every": 4096 * 10 * 10,
    "num_test_agents": 10,
    "batch_size": 64,
    "discount": 0.99,
    "use_gae": True,
    "lambda_gae": 0.95,
    "policy_loss_clip": 0.2,
    "clip_value_loss": True,
    "value_loss_clip": 0.2,
    "value_loss_coef": 0.5,
    "entropy_loss_coef": 0.01,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "lr_annealing": True,
    "norm_advantages": True,
    "seed": 0,
    "save_folder": "ppo_bipedalwalker",
}


if __name__ == "__main__":
    model = PPO(**ppo_kwargs)
    model.train()
    model.save_plots()
    model.save_videos()