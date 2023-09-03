import os
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from gymnasium.wrappers import ClipAction, TransformObservation, NormalizeReward, TransformReward
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from rlib.learning.base_algorithm import BaseAlgorithm
from rlib.learning.rollout_buffer import RolloutBuffer
from rlib.agents import get_agent

def init_layer(layer, std=np.sqrt(2.), bias_constant=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_constant)
    return layer

class PPOAgent(nn.Module):
    """
    Agent used for PPO algorithm.

    It is composed of one actor (the policy network) and one critic (the value network).
    This agent automatically handles discrete and continuous action spaces.

    """

    def __init__(self, state_space, action_space, actor_kwargs={}, critic_kwargs={}):

        super().__init__()

        self.state_space = state_space
        self.action_space = action_space

        if isinstance(action_space, Discrete):
            action_shape = action_space.n
            self.continuous = False
        elif isinstance(action_space, MultiDiscrete):
            action_shape = action_space.nvec
            self.continuous = False
        elif isinstance(action_space, Box):
            action_shape = action_space.shape[1:]
            self.continuous = True
            self.action_high = torch.tensor(action_space.high, dtype=torch.float32)
            self.action_low = torch.tensor(action_space.low, dtype=torch.float32)

        act_kwargs = actor_kwargs.copy()
        act_kwargs['init_weights'] = 'ppo_actor'
        if 'activation' not in act_kwargs:
            act_kwargs['activation'] = 'tanh'
        if 'hidden_sizes' not in act_kwargs:
            act_kwargs['hidden_sizes'] = [64, 64]

        crit_kwargs = critic_kwargs.copy()
        crit_kwargs['init_weights'] = 'ppo_critic'
        if 'activation' not in crit_kwargs:
            crit_kwargs['activation'] = 'tanh'
        if 'hidden_sizes' not in crit_kwargs:
            crit_kwargs['hidden_sizes'] = [64, 64]

        self.actor = get_agent(self.state_space, self.action_space, act_kwargs)
        self.critic = get_agent(self.state_space, self.action_space, crit_kwargs, ppo_critic=True)
        
        if self.continuous:
            self.actor_std = nn.Parameter(torch.ones(np.prod(action_shape).prod()))

    def forward(self, state, action=None):

        value = self.get_value(state)
        dist = self.get_distribution(state)
        if action is None:
            action = dist.sample()
        if self.continuous:
            log_prob = dist.log_prob(action).sum(1)
            entropy = dist.entropy().sum(1)
        else:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, state):
        return self.critic(state)
    
    def get_distribution(self, state):

        logits = self.actor(state)

        if self.continuous:
            distribution = Normal(logits, self.actor_std.expand_as(logits))
        else:
            distribution = Categorical(logits=logits)

        return distribution
    
    def get_log_prob(self, state, action):

        dist = self.get_distribution(state)

        if self.continuous:
            print(dist.log_prob(action))
            return dist.log_prob(action).sum(1)
        else:
            return dist.log_prob(action)
    
    def get_action(self, state):

        is_numpy = isinstance(state, np.ndarray)

        if is_numpy:
            state = torch.tensor(state, dtype=torch.float32)

        # Return the mode of the distribution, i.e. the action with the highest probability
        action = self.get_distribution(state).mode

        if is_numpy:
            action = action.detach().numpy()

        return action

class PPO(BaseAlgorithm):
    """
    Implementation of the Proximal Policy Optimization algorithm (`Paper <https://arxiv.org/abs/1707.06347>`_).
    
    For this algorithm, the policy network is optimized using the clipped surrogate objective function:

    .. math::

        L^{CLIP} = \\hat{\\mathbb{E}}_t \\left[ \\min \\left( r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1 - \\epsilon, 1 + \\epsilon) \\hat{A}_t \\right) \\right]

    where :math:`r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}` is the probability ratio between the new and old policies,
    and :math:`\\hat{A}_t` is the advantage function :math:`A(s, a) = Q(s, a) - V(s)` estimated using Generalized Advantage Estimation (GAE) or not.

    Note that :math:`V(s)` is estimated using a critic network, and that :math:`\\epsilon` allows to control the magnitude of the policy update.

    Example:
    
    .. code-block:: python

        from rlib.learning import PPO

        env_kwargs = {'id': 'CartPole-v1'}
        actor_kwargs = {'hidden_sizes': [64, 64]}
        critic_kwargs = {'hidden_sizes': [64, 64]}
        ppo = PPO(env_kwargs, actor_kwargs, critic_kwargs, batch_size=10,
                  num_steps_per_iter=2_000, total_timesteps=200_000,   # 10 iterations, 2_000 * num_envs steps per iteration
                  num_envs=10, seed=42)

        ppo.train()

    :ivar buffer: Buffer used to store the transitions.
    :vartype buffer: rlib.learning.rollout_buffer.RolloutBuffer
    :ivar current_agent: Current agent used to interact with the environment.
    :vartype current_agent: rlib.learning.ppo.PPOAgent

    """

    def __init__(
            self,
            env_kwargs,
            actor_kwargs={},
            critic_kwargs={},
            num_envs=10,
            save_folder='ppo',
            normalize_observation=False,
            seed=42,
            num_steps_per_iter=2048,
            num_updates_per_iter=10,
            total_timesteps=1_000_000,
            max_episode_length=-1,
            max_total_reward=-1,
            test_every=10_000,
            num_test_agents=10,
            batch_size=64,
            discount=0.99,
            use_gae=True,
            lambda_gae=0.95,
            policy_loss_clip=0.2,
            clip_value_loss=True,
            value_loss_clip=0.2,
            value_loss_coef=0.5,
            entropy_loss_coef=0.01,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            lr_annealing=True,
            norm_advantages=True,
    ):
        """
        Initialize the PPO algorithm.

        :param env_kwargs: Keyword arguments to pass to the gym environment.
        :type env_kwargs: dict
        :param actor_kwargs: Keyword arguments to pass to the actor network, as :meth:`get_agent(kwargs=actor_kwargs)<rlib.agents.get_agent>`.
        :type actor_kwargs: dict
        :param critic_kwargs: Keyword arguments to pass to the critic network, as :meth:`get_agent(kwargs=critic_kwargs, ppo_critic=True)<rlib.agents.get_agent>`.
        :type critic_kwargs: dict
        :param num_envs: Number of parallel environments. Default is 10.
        :type num_envs: int
        :param save_folder: Folder where to save the model. Default is 'ppo'.
        :type save_folder: str
        :param normalize_observation: Whether to normalize the observations or not. Default is False.
        :type normalize_observation: bool
        :param seed: Seed for the random number generator. Default is 42.
        :type seed: int
        :param num_steps_per_iter: Number of steps per iteration per environment. Default is 2048.
        :type num_steps_per_iter: int
        :param num_updates_per_iter: Number of network updates per iteration. Default is 10.
        :type num_updates_per_iter: int
        :param total_timesteps: Total number of steps to train the agent, should be divisible by `num_steps_per_iter x num_envs`. Default is 1_000_000. 
        :type total_timesteps: int
        :param max_episode_length: Maximum length of an episode. If -1, there is no maximum length. Default is -1.
        :type max_episode_length: int
        :param max_total_reward: Maximum total reward of an episode. If -1, there is no maximum total reward. Default is -1.
        :type max_total_reward: int
        :param test_every: Number of steps between each test. Default is 10_000.
        :type test_every: int
        :param num_test_agents: Number of agents to test. Default is 10.
        :type num_test_agents: int
        :param batch_size: Batch size. Default is 64.
        :type batch_size: int
        :param discount: Discount factor. Default is 0.99.
        :type discount: float
        :param use_gae: Whether to use Generalized Advantage Estimation (GAE), if not the advantage estimate is :math:`A(s, a) = \\hat{Q}(s, a) - V(s)`. Default is True.
        :type use_gae: bool
        :param lambda_gae: Lambda parameter for GAE. Default is 0.95.
        :type lambda_gae: float
        :param policy_loss_clip: Epsilon parameter for the clipped surrogate objective function. Default is 0.2.
        :type policy_loss_clip: float
        :param clip_value_loss: Whether to clip the value loss or not. Default is True.
        :type clip_value_loss: bool
        :param value_loss_clip: Epsilon parameter for the clipped value loss. Default is 0.2.
        :type value_loss_clip: float
        :param value_loss_coef: Coefficient for the value loss. Default is 0.5.
        :type value_loss_coef: float
        :param entropy_loss_coef: Coefficient for the entropy loss. Default is 0.01.
        :type entropy_loss_coef: float
        :param max_grad_norm: Maximum norm of the gradients. Default is 0.5.
        :type max_grad_norm: float
        :param learning_rate: Learning rate. Default is 3e-4.
        :type learning_rate: float
        :param lr_annealing: Whether to linearly anneal the learning rate or not. Default is True.
        :type lr_annealing: bool
        :param norm_advantages: Whether to normalize the advantages or not. Default is True.

        """
        
        self.kwargs = locals()

        del self.kwargs['self']
        del self.kwargs['__class__']
        
        super().__init__(env_kwargs, num_envs, max_episode_length, max_total_reward,
                         save_folder, normalize_observation, seed)

        # If the action space is continuous, apply some wrappers
        if isinstance(self.action_space, Box):
            self.envs_wrappers = [
                ClipAction, lambda env: TransformObservation(env, lambda obs: np.clip(obs, -10, 10)),
                NormalizeReward, lambda env: TransformReward(env, lambda rew: np.clip(rew, -10, 10))
            ]
        
        self.actor_kwargs = actor_kwargs
        self.critic_kwargs = critic_kwargs
        self.num_steps_per_iter = num_steps_per_iter
        self.num_updates_per_iter = num_updates_per_iter
        self.total_timesteps = total_timesteps
        self.test_every = test_every
        self.num_test_agents = num_test_agents
        self.batch_size = batch_size
        self.discount = discount
        self.use_gae = use_gae
        self.lambda_gae = lambda_gae
        self.policy_loss_clip = policy_loss_clip
        self.clip_value_loss = clip_value_loss
        self.value_loss_clip = value_loss_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.lr_annealing = lr_annealing
        self.norm_advantages = norm_advantages
        
        self.buffer = RolloutBuffer(
            num_steps_per_iter, self.num_envs, self.obs_space, self.action_space, 
            batch_size, discount,
            use_gae, lambda_gae
        )

        self.current_agent = PPOAgent(self.obs_space, self.action_space, 
                                      actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)

        self.optimizer = torch.optim.Adam(self.current_agent.parameters(), lr=self.learning_rate, eps=1e-5)

        self.env = self.make_env()

        self.global_step = 0

        self.next_test = 0

        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_divs = []

        if total_timesteps & num_updates_per_iter != 0:
            print("Warning: total_iterations should be divisible by num_updates_per_iter")
            self.total_timesteps = ((total_timesteps // num_steps_per_iter) + 1) * num_steps_per_iter
            print("total_iterations is set to: ", self.total_timesteps)

    def update_agent(self, writer):
        """
        Updates the agent using data stored in the buffer.

        :param writer: Tensorboard writer.
        :type writer: torch.utils.tensorboard.SummaryWriter
        """

        for _ in range(self.num_updates_per_iter):

            mean_loss = 0 
            mean_policy_loss = 0
            mean_value_loss = 0
            mean_entropy_loss = 0
            mean_kl_div = 0

            for batch in self.buffer.batches():

                actions, states, log_probs, advantages, returns, values = batch

                _, new_log_prob, new_entropy, new_value = self.current_agent(states, actions)

                log_ratio = new_log_prob - log_probs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl_div = ((ratio - 1) - log_ratio).mean()

                # Normalized Advantage Function
                if self.norm_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                policy_loss_1 = ratio * advantages
                policy_loss_2 = torch.clamp(ratio, 1 - self.policy_loss_clip, 1 + self.policy_loss_clip) * advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                new_value = new_value.squeeze(-1)

                if self.clip_value_loss:  # Clipped value loss
                    value_loss_unclipped = (returns - new_value) ** 2
                    value_loss_clipped = (values + torch.clamp(new_value - values, -self.value_loss_clip, self.value_loss_clip) - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                else:  # Unclipped value loss
                    value_loss = (returns - new_value) ** 2
                    value_loss = 0.5 * value_loss.mean()

                entropy_loss = new_entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_loss_coef * entropy_loss
    
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.current_agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_loss += loss.item()
                mean_policy_loss += policy_loss.item()
                mean_value_loss += value_loss.item()
                mean_entropy_loss += entropy_loss.item()
                mean_kl_div += approx_kl_div.item()

            mean_loss /= (self.num_envs * self.num_steps_per_iter) / self.batch_size
            mean_policy_loss /= (self.num_envs * self.num_steps_per_iter) / self.batch_size
            mean_value_loss /= (self.num_envs * self.num_steps_per_iter) / self.batch_size
            mean_entropy_loss /= (self.num_envs * self.num_steps_per_iter) / self.batch_size
            mean_kl_div /= (self.num_envs * self.num_steps_per_iter) / self.batch_size

        writer.add_scalar("Loss", mean_loss, self.global_step)
        writer.add_scalar("Policy Loss", mean_policy_loss, self.global_step)
        writer.add_scalar("Value Loss", mean_value_loss, self.global_step)
        writer.add_scalar("Entropy Loss", mean_entropy_loss, self.global_step)
        writer.add_scalar("Kullback-Liebler Divergence", mean_kl_div, self.global_step)

        self.losses.append(mean_loss)
        self.policy_losses.append(mean_policy_loss)
        self.value_losses.append(mean_value_loss)
        self.entropy_losses.append(mean_entropy_loss)
        self.kl_divs.append(mean_kl_div)

    def train_(self):

        writer = SummaryWriter(os.path.join(self.save_folder, "logs"))

        # First test at initialization
        mean, std = self.test(num_episodes=self.num_test_agents)

        self.next_test += self.test_every

        writer.add_scalar("Test/Mean Reward", mean, self.global_step)
        writer.add_scalar("Test/Std Reward", std, self.global_step)

        self.mean_test_rewards.append(mean)
        self.std_test_rewards.append(std)

        print(f"Step [{self.global_step}/{self.total_timesteps}]: Reward = {mean:.2f} (+-{std:.2f})")

        while self.global_step < self.total_timesteps:

            # Learning Rate Annealing
            if self.lr_annealing:
                new_lr = self.learning_rate * (1 - self.global_step / self.total_timesteps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            # Update the agent with experience
            self.rollout(writer)
            self.update_agent(writer)
            self.buffer.reset()

            # Test the agent
            if self.global_step >= self.next_test:

                mean, std = self.test(num_episodes=self.num_test_agents)

                writer.add_scalar("Test/Mean Reward", mean, self.global_step)
                writer.add_scalar("Test/Std Reward", std, self.global_step)

                self.mean_test_rewards.append(mean)
                self.std_test_rewards.append(std)

                print(f"Step [{self.global_step}/{self.total_timesteps}]: Reward = {mean:.2f} (+-{std:.2f})")

                self.save(os.path.join(self.models_folder, f"iter_{self.global_step}.pt"))

                self.next_test += self.test_every

        # Final test
        mean, std = self.test(num_episodes=self.num_test_agents)
        print(f"Final Test Reward = {mean:.2f} (+-{std:.2f})")

        writer.close()

    def rollout(self, writer):
        """
        Performs one rollout of the agent in the environment.

        The number of stored transitions is equal to `num_steps_per_iter x num_envs`.

        :param writer: Tensorboard writer.
        :type writer: torch.utils.tensorboard.SummaryWriter
        """

        obs, _ = self.env.reset()
        done = False

        for _ in range(self.num_steps_per_iter):
            
            obs = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, _, value = self.current_agent(obs)

            next_obs, reward, next_done, trunc, info = self.env.step(action.detach().numpy())

            self.buffer.store(
                action, obs, reward, done, log_prob.detach(), value.detach().squeeze(-1)
            )

            obs = next_obs
            done = next_done

            self.global_step += self.num_envs

            if "episode" in info and (np.any(done==1) or np.any(trunc==1)) :

                num_episodes = np.sum(done==1) + np.sum(trunc==1)

                writer.add_scalar("Train/Episode Length", info["episode"]["l"].sum() / num_episodes, self.global_step)
                writer.add_scalar("Train/Reward", info["episode"]["r"].sum() / num_episodes, self.global_step)

                if np.any(info['episode']['l'] > 1600):
                    print(info['episode'])

        obs = torch.tensor(obs, dtype=torch.float32)
        next_value = self.current_agent.get_value(obs).detach().squeeze(-1)
        self.buffer.compute_advantages(done, next_value)

    def save(self, path):

        model = {
            "current_agent": self.current_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "videos_folder": self.videos_folder,
            "plots_folder": self.plots_folder,
        }

        running_results = {
            "global_step": self.global_step,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "losses": self.losses,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
            "kl_divs": self.kl_divs,
        }

        data = {
            "kwargs": self.kwargs,
            "model": model,
            "folders": folders,
            "running_results": running_results
        }

        with open(path, "wb") as f:
            torch.save(data, f)

    def load(self, path, verbose=True):

        data = torch.load(path)

        self.__init__(**data["kwargs"])

        # Set back to the correct folders
        for key in data['folders']:
            setattr(self, key, data['folders'][key])

        self.current_agent.load_state_dict(data["model"]["current_agent"])
        self.optimizer.load_state_dict(data["model"]["optimizer"])

        # Get back the results
        for key in data['running_results']:
            setattr(self, key, data['running_results'][key])

        if verbose:

            print("Loaded model from", path)
            print(f"Environment: {self.env_kwargs['id']}")
            print(f"Current iteration: [{self.global_step}/{self.total_timesteps}]")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"Current reward: {self.mean_test_rewards[-1]}")

    def save_plots(self):

        # First save the testing plots

        # We need to find the global step corresponding to the test_every

        testing_steps = [0]  # We always test at the beginning

        step = self.num_steps_per_iter * self.num_envs
        test_step = self.test_every

        # We reproduce what is happening during the training
        while step <= self.total_timesteps:
            if step >= test_step:
                testing_steps.append(step)
                test_step += self.test_every
            step += self.num_steps_per_iter * self.num_envs

        # Now we can plot the results

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(testing_steps, self.mean_test_rewards)
        plt.fill_between(
            testing_steps,
            np.array(self.mean_test_rewards) - np.array(self.std_test_rewards),
            np.array(self.mean_test_rewards) + np.array(self.std_test_rewards),
            alpha=0.3
        )
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(self.plots_folder, "test_rewards.png"), bbox_inches='tight')
        plt.close()

        # Now we save the training plots
        
        x = [i * self.num_steps_per_iter * self.num_envs for i in range(1, len(self.losses) + 1)]

        plt.figure(figsize=(10, 5))
        plt.plot(x, self.losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig(os.path.join(self.plots_folder, "losses.png"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(x, self.policy_losses)
        plt.xlabel("Steps")
        plt.ylabel("Policy Loss")
        plt.savefig(os.path.join(self.plots_folder, "policy_losses.png"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(x, self.value_losses)
        plt.xlabel("Steps")
        plt.ylabel("Value Loss")
        plt.yscale("log")
        plt.savefig(os.path.join(self.plots_folder, "value_losses.png"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(x, self.entropy_losses)
        plt.xlabel("Steps")
        plt.ylabel("Entropy Loss")
        plt.yscale("log")
        plt.savefig(os.path.join(self.plots_folder, "entropy_losses.png"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(x, self.kl_divs)
        plt.xlabel("Steps")
        plt.ylabel("Kullback-Liebler Divergence")
        plt.yscale("log")
        plt.savefig(os.path.join(self.plots_folder, "kl_divs.png"), bbox_inches='tight')
        plt.close()
