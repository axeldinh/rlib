import copy
import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F

from rlib.agents import get_agent
from .base_algorithm import BaseAlgorithm
from .replay_buffer import ReplayBuffer


class DDPGAgent(torch.nn.Module):

    def __init__(self, mu, q, twin_q=False):

        super().__init__()

        self.mu = mu
        self.q = q
        self.twin_q = twin_q
        if self.twin_q:
            self.q2 = copy.deepcopy(self.q)

    def get_action(self, observation):

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        action = self.mu(observation)

        return action.detach().numpy()


class DDPG(BaseAlgorithm):

    def __init__(
            self, env_kwargs, mu_kwargs, q_kwargs,
            max_episode_length=-1,
            max_total_reward=-1,
            save_folder="ddpg",
            q_lr=3e-4,
            mu_lr=3e-4,
            action_noise=0.1,  # Noise added during population of the replay buffer
            target_noise=0.2,  # Noise added to target actions
            delay_policy_update=2,
            twin_q=True,
            discount=0.99,
            num_episodes=1000,
            learning_starts=50_000,  # Number of random samples in the replay buffer before training
            target_update_tau=0.01,  # Percentage of weights to copy from the main model to the target model
            verbose=True,
            test_every=50_000,
            num_test_episodes=10,
            batch_size=64,
            size_replay_buffer=100_000,
            max_grad_norm=10,
            normalize_observation=False
            ):
                        
        super().__init__(env_kwargs, 
                         max_episode_length, max_total_reward, 
                         save_folder, normalize_observation)

        self.mu_kwargs = mu_kwargs
        self.q_kwargs = q_kwargs
        self.q_lr = q_lr
        self.mu_lr = mu_lr
        self.discount = discount
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.delay_policy_update = delay_policy_update
        self.twin_q = twin_q
        self.num_episodes = num_episodes
        self.learning_starts = max(learning_starts, batch_size)
        self.target_update_tau = target_update_tau
        self.verbose = verbose
        self.test_every = test_every
        self.num_test_episodes = num_test_episodes
        self.batch_size = batch_size
        self.size_replay_buffer = size_replay_buffer
        self.max_grad_norm = max_grad_norm
        self.normalize_observation = normalize_observation

        self.current_episode = 0
        self.running_average = []

        if self.obs_shape.__len__() == 1:

            num_obs = self.obs_shape[0]
            num_actions = self.action_shape[0]

            mu_kwargs['input_size'] = num_obs
            mu_kwargs['output_size'] = num_actions
            q_kwargs['input_size'] = num_obs + num_actions
            q_kwargs['output_size'] = 1
            mu_kwargs['requires_grad'] = True
            q_kwargs['requires_grad'] = True
            
            if self.action_space_type == "continuous":
                mu_kwargs['type_actions'] = 'continuous'
                mu_kwargs['action_space'] = self.action_space
            else:
                raise ValueError("Only continuous action spaces are supported for DDPG.")

            mu = get_agent("mlp", **mu_kwargs)
            q = get_agent("mlp", **q_kwargs)

        if self.obs_shape.__len__() in [2, 3]:

            raise NotImplementedError("CNN agents not implemented yet.")
        
        self.current_agent = DDPGAgent(mu, q, twin_q)
        self.target_agent = DDPGAgent(copy.deepcopy(mu), copy.deepcopy(q), twin_q)

        if self.twin_q:
            self.current_agent.q2 = copy.deepcopy(self.current_agent.q)
            self.target_agent.q2 = copy.deepcopy(self.target_agent.q)

        # Copy the parameters of the main model to the target model
        self._update_target_weights(tau=1)
        
        self.losses = []
        self.train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.episodes_lengths = []
        
        # Store s_t, a_t, r_t, s_t+1, done
        self.replay_buffer = ReplayBuffer(self.size_replay_buffer)

        self.q_optimizer = torch.optim.Adam(self.current_agent.q.parameters(), lr=self.q_lr)
        self.mu_optimizer = torch.optim.Adam(self.current_agent.mu.parameters(), lr=self.mu_lr, maximize=True)

        if self.twin_q:
            self.q2_optimizer = torch.optim.Adam(self.current_agent.q2.parameters(), lr=self.q_lr)

        self.best_iteration = 0
        self.best_test_reward = -np.inf
        
    def train_(self):

        env = self.make_env()

        self._populate_replay_buffer(env)

        init_episode = self.current_episode + 1

        pbar = range(init_episode, self.num_episodes + 1)

        times = []

        for _ in pbar:

            start = time.time()
            
            # Check if we need to make tests
            test_progress = (self.current_episode+1) % self.test_every == 0
            test_progress += (self.current_episode+1) == self.num_episodes

            # Run the episode

            state, _ = env.reset()
            done = False
            length_episode = 0
            episode_reward = 0

            while not done:
                
                action = self.current_agent.get_action(state)
                action += np.random.randn(*action.shape) * self.action_noise
                action = np.clip(action,
                                 self.current_agent.mu.action_space.low, 
                                 self.current_agent.mu.action_space.high)

                new_state, reward, done, _, _ = env.step(action)

                episode_reward += reward

                if episode_reward >= self.max_total_reward and self.max_total_reward != -1:
                    done = True

                self.replay_buffer.store(state.copy(), action.copy(), reward, new_state.copy(), done)

                state = new_state
                
                length_episode += 1

                if length_episode >= self.max_episode_length and self.max_episode_length != -1:
                    done = True

            episode_losses = {'q': [], 'mu': []}
            for _ in range(length_episode):
                # When the episode is done, we update the weights
                loss = self.update_weights()
                episode_losses['q'].append(loss['q'])
                episode_losses['mu'].append(loss['mu'])

            loss = {'q': np.mean(episode_losses['q']), 'mu': np.mean(episode_losses['mu'])}
            self.losses.append(loss)
            
            # Save the rewards
            self.episodes_lengths.append(length_episode)
            self.train_rewards.append(episode_reward)

            self.current_episode += 1

            if test_progress:
                mean, std = self.test(self.num_test_episodes)
                self.mean_test_rewards.append(mean)
                self.std_test_rewards.append(std)

                if self.running_average.__len__() == 0:
                    self.running_average.append(mean)
                else:
                    self.running_average.append(0.9 * self.running_average[-1] + 0.1 * mean)

                if mean > self.best_test_reward:
                    self.best_test_reward = mean
                    self.best_iteration = self.current_episode
                    self.save(self.models_folder + "/best.pkl")
                
                self.save(self.models_folder + f"/iter_{self.current_episode}.pkl")

            if self.verbose and test_progress:
                description = f"Episode [{self.current_episode}/{self.num_episodes}]"
                description += ", Test Reward: {:.2f} (+/- {:.2f})".format(self.mean_test_rewards[-1], self.std_test_rewards[-1])
                description += ", Running Average: {:.2f}".format(self.running_average[-1])
                total_time = int(np.mean(times) * self.num_episodes)
                total_time = str(timedelta(seconds=int(total_time)))
                current_time = int(np.sum(times))
                current_time = str(timedelta(seconds=int(current_time)))
                description += ", Time: [{}s/{}s]".format(current_time, total_time)
                print(description)

            run_time = time.time() - start
            times.append(run_time)
                
    def _update_target_weights(self, tau=0.01):
        # Update Q Network weights 
        for target_param, param in zip(self.target_agent.q.parameters(), self.current_agent.q.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if self.twin_q:
            for target_param, param in zip(self.target_agent.q2.parameters(), self.current_agent.q2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if (self.current_episode + 1) % self.delay_policy_update != 0:
            return

        # Update Mu Network weights
        for target_param, param in zip(self.target_agent.mu.parameters(), self.current_agent.mu.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _populate_replay_buffer(self, env):

        obs, _ = env.reset()
        done = False

        while self.replay_buffer.size < self.learning_starts:

            if self.replay_buffer.size >= self.size_replay_buffer:
                break

            action = env.action_space.sample()
            new_obs, reward, done, _, _ = env.step(action)
            self.replay_buffer.store(obs.copy(), action.copy(), reward, new_obs.copy(), done)
            obs = new_obs

            if done:
                obs, _ = env.reset()
                done = False

        if self.verbose:
            print(f"Replay buffer populated with {self.replay_buffer.size} samples.")

    def update_weights(self):

        s_t, a_t, r_t, s_t_1, done_ = self.replay_buffer.sample(self.batch_size)

        s_t = torch.tensor(s_t)
        a_t = torch.tensor(a_t)
        r_t = torch.tensor(r_t).squeeze()
        s_t_1 = torch.tensor(s_t_1)
        done_ = torch.tensor(done_).squeeze()

        # Update the Q function
        target_a_t = self.target_agent.mu(s_t_1)
        target_a_t += torch.randn_like(target_a_t) * self.target_noise
        target_a_t = torch.clamp(target_a_t,
                                 torch.tensor(self.action_space.low).float(), 
                                 torch.tensor(self.action_space.high).float())
        target_q_t = self.target_agent.q(torch.cat([s_t_1, target_a_t], dim=-1)).squeeze().detach()

        if self.twin_q:
            target_q2_t = self.target_agent.q2(torch.cat([s_t_1, target_a_t], dim=-1)).squeeze().detach()
            target_q_t = torch.min(target_q_t, target_q2_t)

        target = r_t + self.discount * target_q_t * (1 - done_)
        target = target.detach()
        
        q = self.current_agent.q(torch.cat([s_t, a_t], dim=-1)).squeeze()

        loss_q = F.mse_loss(q, target)
        
        self.current_agent.q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.current_agent.q.parameters(), self.max_grad_norm)
        self.q_optimizer.step()

        if self.twin_q:
            loss_q2 = F.mse_loss(self.current_agent.q2(torch.cat([s_t, a_t], dim=-1)).squeeze(), target)
            self.current_agent.q2.zero_grad()
            loss_q2.backward()
            torch.nn.utils.clip_grad_norm_(self.current_agent.q2.parameters(), self.max_grad_norm)
            self.q2_optimizer.step()

        # Update the Policy function
        if (self.current_episode + 1) % self.delay_policy_update == 0:
            actions = self.current_agent.mu(s_t)
            loss_mu = self.current_agent.q(torch.cat([s_t, actions], dim=-1)).mean()

            self.current_agent.q.zero_grad()
            if self.twin_q:
                self.current_agent.q2.zero_grad()
            self.current_agent.mu.zero_grad()
            loss_mu.backward()        
            torch.nn.utils.clip_grad_norm_(self.current_agent.mu.parameters(), self.max_grad_norm)
            self.mu_optimizer.step()

            # Update the target weights    
            self._update_target_weights(tau=self.target_update_tau)

        else:
            if self.losses.__len__() == 0:
                loss_mu = torch.tensor(0)
            else:
                loss_mu = torch.tensor(self.losses[-1]['mu'])

        return {'q': loss_q.item(), 'mu': loss_mu.item()}
    
    def save(self, path):

        ddpg_kwargs = {
            "env_kwargs": self.env_kwargs,
            "mu_kwargs": self.mu_kwargs,
            "q_kwargs": self.q_kwargs,
            "max_episode_length": self.max_episode_length,
            "max_total_reward": self.max_total_reward,
            "save_folder": os.path.abspath(os.path.dirname(self.save_folder)),
            "q_lr": self.q_lr,
            "mu_lr": self.mu_lr,
            "discount": self.discount,
            "action_noise": self.action_noise,
            "target_noise": self.target_noise,
            "delay_policy_update": self.delay_policy_update,
            "twin_q": self.twin_q,
            "num_episodes": self.num_episodes,
            "learning_starts": self.learning_starts,
            "target_update_tau": self.target_update_tau,
            "verbose": self.verbose,
            "test_every": self.test_every,
            "num_test_episodes": self.num_test_episodes,
            "batch_size": self.batch_size,
            "size_replay_buffer": self.size_replay_buffer,
            "max_grad_norm": self.max_grad_norm,
            "normalize_observation": self.normalize_observation,
        }

        saving_folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "plots_folder": self.plots_folder,
            "videos_folder": self.videos_folder,
        }

        model_parameters = {
            "current_agent_state_dict": self.current_agent.state_dict(),
            "target_agent_state_dict": self.target_agent.state_dict(),
            "mu_optimizer_state_dict": self.mu_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }

        running_results = {
            "current_episode": self.current_episode,
            "losses": self.losses,
            "train_rewards": self.train_rewards,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "episodes_lengths": self.episodes_lengths,
            "replay_buffer": self.replay_buffer,
            "running_average": self.running_average,
            "best_iteration": self.best_iteration,
            "best_test_reward": self.best_test_reward,
        }

        data = {
            "ddpg_kwargs": ddpg_kwargs,
            "saving_folders": saving_folders,
            "model_parameters": model_parameters,
            "running_results": running_results,
        }

        torch.save(data, path)

    def load(self, path, verbose=True):

        data = torch.load(path)

        # Reinitialize the model using the kwargs
        self.__init__(**data['ddpg_kwargs'])

        # The exact saving folders need to fixed again (their is an increment during init)
        for key in data['saving_folders'].keys():
            setattr(self, key, data['saving_folders'][key])

        # Load the models and optimizers parameters
        self.load_model_parameters(data)

        # Load the running results
        for key in data['running_results'].keys():
            setattr(self, key, data['running_results'][key])

        if verbose:
            print(f"Loaded model from {path}")
            print(f"Current Episode: [{self.current_episode}/{self.num_episodes}]")
            print(f"Maximum Episode Length: {self.max_episode_length}, Maximum Total Reward: {self.max_total_reward}")
            print(f"Q Learning Rate: {self.q_lr}, Mu Learning Rate: {self.mu_lr}")
            print(f"Discount: {self.discount}, Target Update Tau: {self.target_update_tau}")
            print(f"Action Noise: {self.action_noise}, Target Noise: {self.target_noise}")
            print(f"Delay Policy Update: {self.delay_policy_update}, Twin Q: {self.twin_q}")
            print(f"Size Replay Buffer: {self.replay_buffer.size}, Max Size: {self.size_replay_buffer}, Learning Starts: {self.learning_starts}")
            print(f"Normalize Observation: {self.normalize_observation}, Verbose: {self.verbose}")
            print(f"Test Every: {self.test_every}, Number of Test Episodes: {self.num_test_episodes}")
            print(f"Batch Size: {self.batch_size}, Max Grad Norm: {self.max_grad_norm}")

            print("Running Results:")
            print(f"\tMu Loss: {self.losses[-1]['mu']}")
            print(f"\tQ Loss: {self.losses[-1]['q']}")
            print(f"\tRunning Reward: {self.running_average[-1]}")

    def save_plots(self):

        import matplotlib.pyplot as plt

        x_range = np.arange(self.test_every, self.test_every * len(self.mean_test_rewards) + 1, self.test_every)
        if len(x_range) != len(self.mean_test_rewards):
            x_range = np.append(x_range, self.num_time_steps)
        plt.plot(x_range, self.mean_test_rewards, c="red", label="Test Reward")
        plt.fill_between(
            x_range,
            np.array(self.mean_test_rewards) - np.array(self.std_test_rewards),
            np.array(self.mean_test_rewards) + np.array(self.std_test_rewards),
            alpha=0.5, color="red"
        )
        plt.plot(x_range, self.running_average, c="blue", label="Running Average")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(self.plots_folder + "/mean_test_rewards.png", bbox_inches="tight")
        plt.close()

        plt.plot(range(1, len(self.episodes_lengths)+1), self.episodes_lengths)
        plt.xlabel("Number of episodes")
        plt.ylabel("Episode length")
        plt.savefig(self.plots_folder + "/episode_lengths.png", bbox_inches="tight")
        plt.close()

        plt.plot(range(1, len(self.train_rewards)+1), self.train_rewards)
        plt.xlabel("Number of episodes")
        plt.ylabel("Reward")
        plt.savefig(self.plots_folder + "/train_rewards.png", bbox_inches="tight")
        plt.close()

        losses_mu = [loss['mu'] for loss in self.losses]
        plt.plot(range(1, len(self.losses) + 1), losses_mu)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Policy Loss (Higher is better)")
        plt.savefig(self.plots_folder + "/policy_losses.png", bbox_inches="tight")
        plt.close()

        losses_q = [loss['q'] for loss in self.losses]
        plt.plot(range(1, len(losses_q) + 1), losses_q)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Q Loss (Lower is better)")
        plt.yscale("log")
        plt.savefig(self.plots_folder + "/q_losses.png", bbox_inches="tight")
        plt.close()

        print("Figures saved in ", self.plots_folder)

    def load_from_path(path):

        data = torch.load(path)

        model = DDPG(**data['ddpg_kwargs'])
        model.current_agent.load_state_dict(data['model_parameters']['current_agent_state_dict'])
        model.target_agent.load_state_dict(data['model_parameters']['target_agent_state_dict'])
        model.mu_optimizer.load_state_dict(data['model_parameters']['mu_optimizer_state_dict'])
        model.q_optimizer.load_state_dict(data['model_parameters']['q_optimizer_state_dict'])

        for key in data['running_results'].keys():
            setattr(model, key, data['running_results'][key])

        return model
    
    def load_model_parameters(self, data):

        self.current_agent.load_state_dict(data['model_parameters']['current_agent_state_dict'])
        self.target_agent.load_state_dict(data['model_parameters']['target_agent_state_dict'])
        self.mu_optimizer.load_state_dict(data['model_parameters']['mu_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(data['model_parameters']['q_optimizer_state_dict'])
