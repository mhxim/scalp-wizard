# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import copy
import datetime
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from market_simulation.market_env import MarketEnvironment
import torch.nn.functional as F
from torch.distributions import Normal

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MarketEnvironment"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 2
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.98
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        # Additional environment wrappers and setup
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

observation_keys = [
    'position', 'weekly_vp', 'daily_vp',
    'vp_vb_500', 'vp_vb_200', 'vp_vb_100'
]


class Agent(nn.Module):
    def __init__(self, envs, lstm_layers=3):
        super().__init__()

        # position, weekly_vp, daily_vp, hours_8_vp, hours_4_vp, hours_2_vp, minutes_1_ohlc, volume_delta_1min, volume_delta_10s = x
        self.lstm_layers = lstm_layers
        # self.feature_sizes_lstm = [[100, 512], [100, 512], [100, 512], [100, 512], [100, 512], [6, 128], [2, 128]]
        self.feature_sizes_lstm = [[65, 512], [65, 512], [49, 512], [33, 512], [25, 512]]

        # Create an LSTM for each feature set
        self.lstms = nn.ModuleList([nn.LSTM(input_size=self.feature_sizes_lstm[i][0], hidden_size=self.feature_sizes_lstm[i][1], batch_first=True, num_layers=lstm_layers) for i in range(len(self.feature_sizes_lstm))])
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        # Calculate the total number of features after LSTM output concatenation
        total_features = 0
        for feature in self.feature_sizes_lstm:
            total_features += feature[1]

        print("total_features", total_features)

        total_features += 2

        # Actor and Critic networks
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(total_features, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, envs.single_action_space.shape[0]))
        )
    
        self.actor_log_std = nn.Parameter(torch.zeros(envs.single_action_space.shape[0]))  # Learnable log std

        self.critic = nn.Sequential(
            layer_init(nn.Linear(total_features, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        # Initialize LSTM weights and biases
        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

    
    def get_states(self, xs, lstm_states, done):
        hidden_states = []
        new_lstm_states = []

        for i, (x, lstm, lstm_state) in enumerate(zip(xs, self.lstms, lstm_states)):
            # Reshape x to [batch_size, sequence_length, feature_size]
            batch_size = x.size(0)
            sequence_length = x.size(1)
            x = x.view(batch_size, sequence_length, -1)

            # Ensure that the input x has the correct shape
            assert x.size(-1) == lstm.input_size, f"Expected input size {lstm.input_size}, got {x.size(-1)}"
            
            # Unpack the lstm_state (hidden and cell states)
            h_n, c_n = lstm_state
            h_n = h_n[:, :batch_size, :].contiguous()  # Make contiguous
            c_n = c_n[:, :batch_size, :].contiguous()  # Make contiguous

            x, (new_h_n, new_c_n) = lstm(x, (h_n, c_n))
            x = x[:, -1, :]  # Use the output of the last timestep
            hidden_states.append(x)
            new_lstm_states.append((new_h_n, new_c_n))

        return torch.cat(hidden_states, dim=1), new_lstm_states

    
    def init_lstm_states(self, batch_size=2, num_envs=1):
        return [(torch.zeros(self.lstm_layers, batch_size, feature_size[1]).to(self.device),
                torch.zeros(self.lstm_layers, batch_size, feature_size[1]).to(self.device)) for feature_size in self.feature_sizes_lstm]


    def get_value(self, obs_dict, lstm_state, done):
        xs = [obs_dict[key] for key in observation_keys if key != 'position']

        hidden, _ = self.get_states(xs, lstm_state, done)
        hidden = torch.cat((obs_dict['position'], hidden), dim=1)
        return self.critic(hidden)
    
    def get_action_and_value(self, obs_dict, lstm_state, done, action=None):
        xs = [obs_dict[key] for key in observation_keys if key != 'position']
        hidden, lstm_state = self.get_states(xs, lstm_state, done)
        hidden = torch.cat((obs_dict['position'], hidden), dim=1)
        
        mean = self.actor_mean(hidden)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        dist = Normal(mean, std)
        
        if action is None:
            action = torch.tanh(dist.sample())
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # Aggregates log probabilities, still keeps an extra dim

        # Correcting the squeezing operation to match dimensions
        log_prob = log_prob.squeeze(-1)  # This should correctly remove the redundant dimension
        

        return action, log_prob, dist.entropy().sum(-1), self.critic(hidden), lstm_state

    
class AlphaWizard:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        env_dates = [
            [datetime.date(2020, 1, 1), datetime.date(2020, 6, 1)],
            [datetime.date(2020, 6, 1), datetime.date(2020, 12, 31)],
            # [datetime.date(2021, 1, 1), datetime.date(2021, 6, 1)],
            # [datetime.date(2021, 6, 1), datetime.date(2021, 12, 31)],
            # [datetime.date(2022, 1, 1), datetime.date(2022, 6, 1)],
            # [datetime.date(2022, 6, 1), datetime.date(2022, 12, 31)]
        ]

        self.envs = gym.vector.SyncVectorEnv(
            [self.make_env(self.args.env_id, date_pair[0], date_pair[1]) for date_pair in env_dates]
        )

        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        self.obs = {key: torch.zeros((args.num_steps, args.num_envs) + space.shape, dtype=torch.float32).to(self.device) for key, space in self.envs.single_observation_space.spaces.items()}
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        self.global_step = 0
        self.iteration = 0

        self.start_time = time.time()
        self.next_obs, _ = self.envs.reset()

        for key in self.next_obs:
            self.next_obs[key] = torch.tensor(self.next_obs[key], dtype=torch.float32).to(self.device)

        self.next_done = torch.zeros(self.args.num_envs).to(self.device)
        self.next_lstm_states = self.agent.init_lstm_states(batch_size=self.args.num_steps, num_envs=self.args.num_envs) 

        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

    def make_env(self, env_id, start_date, end_date):
        def thunk():
            env = gym.make(env_id)
            env.unwrapped.prepare_simulation(start_date, end_date)
            return env
        return thunk

    def train(self):
        for iteration in range(1, args.num_iterations + 1):
            self.next_lstm_states = self.agent.init_lstm_states(batch_size=self.args.num_steps, num_envs=self.args.num_envs) 
            
            self.iteration = iteration
            if self.args.anneal_lr:
                frac = 1.0 - (self.iteration - 1.0) / self.args.num_iterations
                lrnow = max(frac, 0.5) * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            self.get_training_batch()

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(
                    self.next_obs,
                    self.next_lstm_states,
                    self.next_done,
                ).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = {key: torch.cat([self.obs[key][step] for step in range(self.args.num_steps)], dim=0) for key in self.obs}
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.action_space.shape)
            b_dones = self.dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)


            # Optimizing the policy and value network
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
            clipfracs = []
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                mb_obs = {key: b_obs[key][mb_inds] for key in b_obs}
                
                # Process the minibatch
                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                    mb_obs,
                    self.next_lstm_states[mb_inds],
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # Modify policy loss calculations to incorporate dynamic weighting
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Modify value loss calculation similarly 
                # For example, if using a clipped value loss:
                newvalue_weighted = newvalue + b_values[mb_inds]
                v_loss_unclipped = (newvalue_weighted - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue_weighted - b_values[mb_inds],
                    -self.args.clip_coef,
                    self.args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Compute the overall loss
                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()


            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                continue

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            print("SPS:", int(self.global_step / (time.time() - self.start_time)))
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)


    def get_training_batch(self):
        for env in self.envs.envs:
            env.unwrapped.cash = 1000
            env.unwrapped.exposure = 0
            env.unwrapped.previous_exposure = 0
            env.unwrapped.shares_owned = 0

        cumulative_reward = 0
        
        for step in range(0, args.num_steps):
            self.next_lstm_states = self.agent.init_lstm_states(batch_size=self.args.num_steps, num_envs=self.args.num_envs) 
            self.global_step += self.args.num_envs

            for key in self.next_obs:
                self.obs[key][step] = self.next_obs[key]

            self.dones[step] = self.next_done
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = self.agent.get_action_and_value(
                    {key: self.next_obs[key] for key in observation_keys},  # Change here
                    self.next_lstm_states, self.next_done
                )
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            self.next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            print("reward: " + str(reward))

            cumulative_reward += reward
            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)

            for key in self.next_obs:
                self.next_obs[key] = torch.tensor(self.next_obs[key], dtype=torch.float32).to(self.device)

            self.next_done = torch.Tensor(self.next_done).to(self.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
        
        print("episode_reward: " + str(cumulative_reward))


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    trainer = AlphaWizard(args)

    try:
        trainer.train()
    except ValueError as e:
        # Exception handling code for ZeroDivisionError
        print("ValueError:", str(e))
    except TypeError as e:
        # Exception handling code for ZeroDivisionError
        print("TypeError:", str(e))
    except RecursionError as e:
        # Exception handling code for ZeroDivisionError
        print("RecursionError:", str(e))
    except MemoryError as e:
        # Exception handling code for ZeroDivisionError
        print("MemoryError:", str(e))

    trainer.envs.close()
    trainer.writer.close()
