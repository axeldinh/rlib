from collections import deque
from tqdm import trange
import numpy as np
import torch
import torch.nn .functional as F
from .base_algorithm import BaseAlgorithm
from rlib.wrappers import NormWrapper

class DeepQLearning(BaseAlgorithm):

    def __init__(
            self, env_fn, agent_fn,
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
            max_grad_norm=10
            ):
        
        norm_env_fn = lambda render_mode=None: NormWrapper(env_fn(render_mode))
        
        super().__init__(norm_env_fn, agent_fn, max_episode_length, max_total_reward, save_folder)

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
        self.max_norm = max_grad_norm

        self.current_time_step = 0
        self.current_agent = self.agent_fn()
        self.target_agent = self.agent_fn()

        # Copy the parameters of the main model to the target model
        for target_param, param in zip(self.target_agent.parameters(), self.current_agent.parameters()):
            target_param.data.copy_(param.data)
        
        self.losses = []
        self.train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.episodes_lengths = []
        
        # Store s_t, a_t, r_t, s_t+1, done
        self.replay_buffer = deque(maxlen=self.size_replay_buffer)

        self.optimizer = torch.optim.Adam(self.current_agent.parameters(), lr=self.lr)
        
    def train_(self):

        env = self.env_fn()

        self._populate_replay_buffer(env)

        if self.verbose:
            pbar = trange(self.num_time_steps)
        else:
            pbar = range(self.num_time_steps)

        state, _ = env.reset()
        done = False
        length_episode = 0
        episode_reward = 0

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

            self.replay_buffer.append((state.copy(), action, reward, new_state.copy(), done))

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
                self.save(self.models_folder + f"/iter_{self.current_time_step}.pkl")

            if self.verbose:
                description = f"Epsilon Greedy: {self.epsilon_greedy:.2f}"
                if len(self.mean_test_rewards) > 0:
                    description += ", Test Reward: {:.2f}".format(self.mean_test_rewards[-1])
                if len(self.losses) > 0:
                    description += ", Loss: {:.2f}".format(self.losses[-1])
                pbar.set_description(description)

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
            self.replay_buffer.append((obs.copy(), action, reward, new_obs.copy(), done))
            obs = new_obs

            if done:
                obs, _ = env.reset()
                done = False

            if len(self.replay_buffer) >= self.size_replay_buffer:
                break

        if self.verbose:
            print(f"Replay buffer populated with {len(self.replay_buffer)} samples.")


    def update_weights(self):

        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch]
        
        s_t = [x[0] for x in batch]
        a_t = [x[1] for x in batch]
        r_t = [x[2] for x in batch]
        s_t_1 = [x[3] for x in batch]
        done_ = [int(x[4]) for x in batch]

        s_t = torch.tensor(np.stack(s_t)).view(self.batch_size, -1)
        a_t = torch.tensor(np.stack(a_t)).view(self.batch_size)
        r_t = torch.tensor(np.stack(r_t)).view(self.batch_size)
        s_t_1 = torch.tensor(np.stack(s_t_1)).view(self.batch_size, -1)
        done_ = torch.tensor(np.stack(done_)).view(self.batch_size)

        q = self.current_agent(s_t)
        q = q.gather(1, a_t.unsqueeze(1)).squeeze(1)
        next_q = torch.amax(self.target_agent(s_t_1).detach(), dim=-1)
        target = r_t + self.discount * next_q * (1 - done_)

        loss = F.smooth_l1_loss(q, target)

        self.current_agent.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.current_agent.parameters(), self.max_norm)

        self.optimizer.step()

        return loss
    
    def save(self, path):

        data = {
            "lr": self.lr,
            "discount": self.discount,
            "epsilon_greedy": self.epsilon_greedy,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "num_time_steps": self.num_time_steps,
            "verbose": self.verbose,
            "test_every": self.test_every,
            "num_test_episodes": self.num_test_episodes,
            "batch_size": self.batch_size,
            "current_time_step": self.current_time_step,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "current_agent": self.current_agent
        }

        torch.save(data, path)

    def load(self, path, verbose=True):

        data = torch.load(path)

        for key in data:
            setattr(self, key, data[key])

        if verbose:
            print(f"Loaded model from {path}")
            print(f"Current time step: {self.current_time_step}")
            print("Discount: {:.2f}".format(self.discount))
            print("Epsilon Greedy: {:.2f}".format(self.epsilon_greedy))
            print("Epsilon Decay: {:.2f}".format(self.epsilon_decay))
            print("Epsilon Min: {:.2f}".format(self.epsilon_min))
            print("Learning Rate: {:.2f}".format(self.lr))
            print("Batch Size: {:.2f}".format(self.batch_size))
            print("Number of time steps: {:.2f}".format(self.num_time_steps))
            print("Test every: {:.2f}".format(self.test_every))
            print("Number of test episodes: {:.2f}".format(self.num_test_episodes))
            print(f"Last test rewards: {self.mean_test_rewards[-1]:.2f} +/- {self.std_test_rewards[-1]:.2f}")

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
