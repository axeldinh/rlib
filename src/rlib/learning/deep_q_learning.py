from collections import deque
import datetime
import os
import time
from tqdm import trange
import numpy as np
import torch
import torch.nn .functional as F
from .base_algorithm import BaseAlgorithm
from rlib.agents import get_agent
from rlib.learning.replay_buffer import ReplayBuffer

class DeepQLearning(BaseAlgorithm):

    def __init__(
            self, env_kwargs, agent_kwargs,
            max_episode_length=-1,
            max_total_reward=-1,
            save_folder="deep_qlearning",
            lr=3e-4,
            discount=0.99,
            epsilon_greedy=0.1,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            num_time_steps=100_000,
            learning_starts=50_000,
            update_every=4,
            main_target_update=100,
            verbose=True,
            test_every=50_000,
            num_test_episodes=10,
            batch_size=64,
            size_replay_buffer=100_000,
            max_grad_norm=10,
            normalize_observation=False,
            ):
        
        
        super().__init__(env_kwargs, 
                         max_episode_length, max_total_reward, 
                         save_folder, normalize_observation)

        self.agent_kwargs = agent_kwargs
        self.lr = lr
        self.discount = discount
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_time_steps = num_time_steps
        self.learning_starts = learning_starts
        self.update_every = update_every
        self.main_target_update = main_target_update
        self.verbose = verbose
        self.test_every = test_every
        self.num_test_episodes = num_test_episodes
        self.batch_size = batch_size
        self.size_replay_buffer = size_replay_buffer
        self.max_grad_norm = max_grad_norm

        self.current_time_step = 0
        self.running_average = []

        if self.obs_shape.__len__() == 1:

            num_obs = self.obs_shape[0]
            num_actions = self.action_shape[0]

            agent_kwargs['input_size'] = num_obs
            agent_kwargs['output_size'] = num_actions
            agent_kwargs['requires_grad'] = True

            if self.action_space_type != "discrete":

                raise ValueError("The action space must be discrete. Current action space: {}".format(self.action_space_type))
            
            self.current_agent = get_agent("mlp", agent_kwargs)
            self.target_agent = get_agent("mlp", agent_kwargs)

        elif self.obs_shape.__len__() in [2, 3]:
            raise NotImplementedError("CNN not yet implemented, observations must be 1D.")
        
        else:
            raise ValueError("Observations must be 1D, 2D or 3D. Current observations shape: {}".format(self.obs_shape))

        # Copy the parameters of the main model to the target model
        for target_param, param in zip(self.target_agent.parameters(), self.current_agent.parameters()):
            target_param.data.copy_(param.data)
        
        self.losses = []
        self.train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.episodes_lengths = []
        
        # Store s_t, a_t, r_t, s_t+1, done
        self.replay_buffer = ReplayBuffer(max_size=self.size_replay_buffer)

        self.optimizer = torch.optim.Adam(self.current_agent.parameters(), lr=self.lr)

        self.best_iteration = 0
        self.best_test_reward = -np.inf
        
    def train_(self):
        
        env = self.make_env()

        self._populate_replay_buffer(env)

        pbar = range(self.num_time_steps)

        state, _ = env.reset()
        done = False
        length_episode = 0
        episode_reward = 0

        times = []
        time_start = time.time()

        for n in pbar:

            

            test_progress = (n+1) % self.test_every == 0
            test_progress += (n+1) == self.num_time_steps
                
            if np.random.rand() < self.epsilon_greedy:
                action = env.action_space.sample()
            else:
                action = self.target_agent.get_action(state)

            new_state, reward, done, _, _ = env.step(action)

            episode_reward += reward

            if episode_reward >= self.max_total_reward and self.max_total_reward != -1:
                done = True

            self.replay_buffer.store(state.copy(), action, reward, new_state.copy(), done)

            if self.current_time_step % self.update_every == 0:
                loss = self.update_weights()
                self.losses.append(loss.item())
                if self.current_time_step % self.main_target_update == 0:
                    for target_param, param in zip(self.target_agent.parameters(), self.current_agent.parameters()):
                        target_param.data.copy_(param.data)

            state = new_state
            
            self.epsilon_greedy = max(self.epsilon_min, self.epsilon_greedy * self.epsilon_decay)

            length_episode += 1

            if length_episode >= self.max_episode_length and self.max_episode_length != -1:
                done = True

            self.current_time_step += 1

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
                    self.best_iteration = self.current_time_step
                    self.save(self.models_folder + "/best.pkl")

                self.save(self.models_folder + f"/iter_{self.current_time_step}.pkl")

            if self.verbose and test_progress:
                description = f"TimeStep: [{self.current_time_step}/{self.num_time_steps}]"
                description += ", Test Reward: {:.2f} (+/- {:.2f})".format(self.mean_test_rewards[-1], self.std_test_rewards[-1])
                description += ", Running Average: {:.2f}".format(self.running_average[-1])
                times.append(time.time() - time_start)
                total_time = np.mean(times) * self.num_time_steps / self.test_every
                current_time = np.sum(times)
                total_time = str(datetime.timedelta(seconds=int(total_time)))
                current_time = str(datetime.timedelta(seconds=int(current_time)))
                description = f", Time: [{self.current_time_step}/{self.num_time_steps}]"
                print(description)
                time_start = time.time()

            if done:
                state, _ = env.reset()
                done = False
                self.episodes_lengths.append(length_episode)
                self.train_rewards.append(episode_reward)
                length_episode = 0
                episode_reward = 0


    def _populate_replay_buffer(self, env):

        obs, _ = env.reset()
        done = False

        while len(self.replay_buffer) < self.learning_starts:
            if np.random.rand() < self.epsilon_greedy:
                action = env.action_space.sample()
            else:
                action = self.target_agent.get_action(obs)
            new_obs, reward, done, _, _ = env.step(action)
            self.replay_buffer.store(obs.copy(), action, reward, new_obs.copy(), done)
            obs = new_obs

            if done:
                obs, _ = env.reset()
                done = False

            if len(self.replay_buffer) >= self.size_replay_buffer:
                break

        if self.verbose:
            print(f"Replay buffer populated with {len(self.replay_buffer)} samples.")


    def update_weights(self):
        
        s_t, a_t, r_t, s_t_1, done_ = self.replay_buffer.sample(self.batch_size)
        
        s_t = torch.tensor(s_t)
        a_t = torch.tensor(a_t)
        r_t = torch.tensor(r_t).squeeze()
        s_t_1 = torch.tensor(s_t_1)
        done_ = torch.tensor(done_)

        q = self.current_agent(s_t)
        q = q.gather(1, a_t.unsqueeze(1)).squeeze(1)
        next_q = torch.amax(self.target_agent(s_t_1).detach(), dim=-1)
        target = r_t + self.discount * next_q * (1 - done_)

        loss = F.smooth_l1_loss(q, target)

        self.current_agent.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.current_agent.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss
    
    def save(self, path):

        dqn_kwargs = {
            "env_kwargs": self.env_kwargs,
            "agent_kwargs": self.agent_kwargs,
            "max_episode_length": self.max_episode_length,
            "max_total_reward": self.max_total_reward,
            "save_folder": os.path.abspath(os.path.dirname(self.save_folder)),
            "lr": self.lr,
            "discount": self.discount,
            "epsilon_greedy": self.epsilon_greedy,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "num_time_steps": self.num_time_steps,
            "learning_starts": self.learning_starts,
            "update_every": self.update_every,
            "main_target_update": self.main_target_update,
            "verbose": self.verbose,
            "test_every": self.test_every,
            "num_test_episodes": self.num_test_episodes,
            "batch_size": self.batch_size,
            "size_replay_buffer": self.size_replay_buffer,
            "max_grad_norm": self.max_grad_norm,
            "normalize_observation": self.normalize_observation
        }

        saving_folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "plots_folder": self.plots_folder,
            "videos_folder": self.videos_folder
        }

        model_parameters = {
            "current_agent": self.current_agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        running_results = {
            "current_time_step": self.current_time_step,
            "running_average": self.running_average,
            "losses": self.losses,
            "train_rewards": self.train_rewards,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "episodes_lengths": self.episodes_lengths,
            "replay_buffer": self.replay_buffer,
            "best_iteration": self.best_iteration,
            "best_test_reward": self.best_test_reward
        }

        data = {
            "dqn_kwargs": dqn_kwargs,
            "saving_folders": saving_folders,
            "model_parameters": model_parameters,
            "running_results": running_results
        }

        torch.save(data, path)

    def load(self, path, verbose=True):

        data = torch.load(path)

        self.__init__(**data['dqn_kwargs'])

        for key in data['saving_folders'].keys():
            setattr(self, key, data['saving_folders'][key])

        self.load_model_parameters(data)

        for key in data['running_results'].keys():
            setattr(self, key, data['running_results'][key])

        if verbose:
            print("Model loaded from: ", path)
            print(f"Current Iteration: [{self.current_time_step}/{self.num_time_steps}]")
            print(f"Learning Rate: {self.lr}, Discount: {self.discount}")
            print(f"Epsilon Greedy: {self.epsilon_greedy}, Epsilon Decay: {self.epsilon_decay}, Epsilon Min: {self.epsilon_min}")
            print(f"Learning Starts: {self.learning_starts}, Update Every: {self.update_every}, Main Target Update: {self.main_target_update}")
            print(f"Test Every: {self.test_every}, Num Test Episodes: {self.num_test_episodes}")
            print(f"Batch Size: {self.batch_size}, Size Replay Buffer: {self.size_replay_buffer}")
            print(f"Max Grad Norm: {self.max_grad_norm}, Normalize Observation: {self.normalize_observation}")
            print(f"Best Iteration: {self.best_iteration}, Best Test Reward: {self.best_test_reward}")

            print("Running Results:")
            print(f"\tRunning Average: {self.running_average[-1]}")
            print(f"\tTest Rewards: {self.mean_test_rewards[-1]} (+/- {self.std_test_rewards[-1]})")
            print(f"\tLoss: {self.losses[-1]}")

    def load_model_parameters(self, data):

        self.current_agent.load_state_dict(data['model_parameters']['current_agent'])
        self.target_agent.load_state_dict(data['model_parameters']['target_agent'])
        self.optimizer.load_state_dict(data['model_parameters']['optimizer'])
    

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
        plt.xlabel("Number of iterations")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(self.plots_folder + "/mean_test_rewards.png", bbox_inches="tight")
        plt.close()


        x_range = [length for length in self.episodes_lengths]
        x_range = np.cumsum(x_range)
        plt.plot(x_range, self.episodes_lengths)
        plt.xlabel("Number of iterations")
        plt.ylabel("Episode length")
        plt.savefig(self.plots_folder + "/episode_lengths.png", bbox_inches="tight")
        plt.close()

        plt.plot(x_range, self.train_rewards)
        plt.xlabel("Number of iterations")
        plt.ylabel("Reward")
        plt.savefig(self.plots_folder + "/train_rewards.png", bbox_inches="tight")
        plt.close()

        plt.plot(range(1, len(self.losses) * self.batch_size + 1, self.batch_size), self.losses)
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig(self.plots_folder + "/losses.png", bbox_inches="tight")
        plt.close()



        print("Figures saved in ", self.plots_folder)
