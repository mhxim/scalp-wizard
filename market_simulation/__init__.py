from gymnasium.envs.registration import register

register(
    id='MarketEnvironment',
    entry_point='market_simulation.market_env:MarketEnvironment',
    disable_env_checker = True
)
