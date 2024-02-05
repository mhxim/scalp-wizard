# Scalp bot in Python using PPO RL with LSTM

## How it works:

### Reward structure:
Rewards are immediate. They are the log returns of the exposure excluding transaction fees from adjustments.
Cumulative rewards can be calculated easily by summing the log() step returns of the eposide.

### How it works:
The RL Agent receives it's current portfolio as MDP observation as well as the normalized volume profiles from different timeframes.
The Agent learns to corralate different timeframes and increase exposure when multiple volume profiles align.
Through Proximinal Policy Optimization with LSTMs the Agent learns to set the optimial continous exposure without making too drastic changes.

### Further enhancements for next project:
- Due to the non-stationary nature of financial markets it's hard for PPO to learn effectively. A solution would be to create specialized policies based on the current dynamics using Graph Neural Networks with flow normalisation and probabilistic graphical models
- Standardize dollar bars into a non stationary representation using Marcos Lopez approach from Advances in Financial Machine Learning
- Extend to meta policy and include further orderflow data, such as tick bars, volume delta
- Include the sortino ratio in the reward function to avoid drawdawns within an episode
- IMPALA


### Credits
- CleanRL PPO implementation baseline  