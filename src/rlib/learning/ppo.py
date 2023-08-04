import torch

from rlib.learning import BaseAlgorithm
from rlib.agents import get_agent

class PPO(BaseAlgorithm):

    def __init__(self, env_kwargs, actor_kwargs, critic_kwargs,
                 max_episode_length=-1, max_total_reward=-1, save_folder="ppo",
                 normalize_observations=False):
        """
        :param env_kwargs: Keyword arguments to call `gym.make(**env_kwargs, render_mode=render_mode)`
        :type env_kwargs: dict
        :param max_episode_length: Maximum length of an episode. Defaults to -1 (no limit).
        :type max_episode_length: int
        :param max_total_reward: Maximum total reward of an episode. Defaults to -1 (no limit).
        :type max_total_reward: float
        :param normalize_observations: Whether to normalize observations in `[-1, 1]`. Defaults to False.
        :type normalize_observations: bool

        """

        super.__init__(env_kwargs, max_episode_length, max_total_reward, 
                       save_folder, normalize_observations)
        
        self.actor_kwargs = actor_kwargs
        self.critic_kwargs = critic_kwargs

        # Build the actor and critic

        if self.obs_shape.__len__() == 1:

            num_obs = self.obs_shape[0]
            num_actions = self.action_shape[0]

            actor_kwargs["input_size"] = num_obs
            actor_kwargs["output_size"] = num_actions
            actor_kwargs['requires_grad'] = True

            critic_kwargs["input_size"] = num_obs
            critic_kwargs["output_size"] = 1
            critic_kwargs['requires_grad'] = True        
            self.actor = get_agent("mlp", **actor_kwargs)
            self.critic = get_agent("mlp", **critic_kwargs)

        else:
            raise NotImplementedError("PPO only supports 1D observations currently.")
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_kwargs["lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_kwargs["lr"])

        