import math
import os
import datetime
import random
import pandas as pd
import torch
import numpy as np
from collections import deque
from order_flow.chart import ChartStandardized
from order_flow.order_flow import OrderFlow
from order_flow.volume_profile import VolumeProfileSimple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarketEnvironment(gym.Env):
    def __init__(self):
        self.current_trade_row = 0
        self.done = False
        self.df = None
        self.trade_time = None

        self.env_start_date = None
        self.env_end_date = None

        # Analyzing the last < 24h
        self.trades = np.empty((0, 4))

        # For weekly VP
        self.aggregated_trades_50 = np.empty((0, 4))
        self.aggregated_volume_50 = 0
        self.aggregated_price_50 = 0
        self.aggregated_trades_count_50 = 0

        # Amount of $ traded per bar
        self.db_size = 1
        self.current_db_size = 0

        self.total_pl = 0
        self.volume = 0

        self.price_low = 0
        self.price_high = 0

        self.spread = 0
        self.latest_buy_price = 0
        self.latest_sell_price = 0

        self.cash = 1000
        self.exposure = 0
        self.previous_exposure = 0
        self.shares_owned = 0
        self.price = 0
        self.previous_price = 0

        self.returns = deque(maxlen=10*24*7)
        self.risk_free_rate = 0

        # self.data_path = "/content/trade_data"
        self.data_path = "C:/dev/crypto-trading/scalp/trade_data"
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'weekly_vp': spaces.Box(low=-np.inf, high=np.inf, shape=(100, 65)),
            'daily_vp': spaces.Box(low=-np.inf, high=np.inf, shape=(100, 65)),
            'vp_vb_500': spaces.Box(low=-np.inf, high=np.inf, shape=(100, 49)),
            'vp_vb_200': spaces.Box(low=-np.inf, high=np.inf, shape=(100, 33)),
            'vp_vb_100': spaces.Box(low=-np.inf, high=np.inf, shape=(100, 25)),
        })


    def step(self, action):
        self.current_trade_row += 1
        reward = self.calculate_reward(action)

        if self.current_trade_row >= len(self.df):
            self.done = True
            next_state = None
        else:
            next_state = self.get_obs(self.current_trade_row)

        if next_state != None:
            for key, value in next_state.items():
                expected_shape = self.observation_space[key].shape
                actual_shape = value.shape
                if expected_shape != actual_shape:
                    print(f"{key}: expected shape {expected_shape}, actual shape {actual_shape}")

        return next_state, reward, self.done, self.current_trade_row == len(self.df), {}


    def reset(self):
        db_bars = ChartStandardized().construct_ohlc(self.trades, self.db_size)

        # Reset the environment state based on the first row of the CSV
        observation = {
            'position': np.array([self.exposure, self.spread]),
            'weekly_vp': np.array(VolumeProfileSimple().get_volume_profile_flow(self.aggregated_trades_50, self.trade_time, db_bars, 32, datetime.timedelta(days=5))),
            'daily_vp': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 32, datetime.timedelta(days=1))),
            'vp_vb_500': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 24, datetime.timedelta(hours=8))),
            'vp_vb_200': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 16, datetime.timedelta(hours=4))),
            'vp_vb_100': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 12, datetime.timedelta(hours=2))),
        }

        return observation, {}
    

    def prepare_simulation(self, start_date, end_date):
        # self.start_date = start_date + datetime.timedelta(minutes=start_minute)
        self.start_date = start_date
        self.env_end_date = end_date
        self.env_start_date = start_date
        prepare_start = start_date - datetime.timedelta(days=5)
        # prepare week prior:
        while prepare_start <= start_date:
            start_datetime = datetime.datetime.combine(start_date, datetime.time(0, 0))  # This combines the date with a default time of 00:00
            file_name = prepare_start.strftime(self.data_path + "/FTMUSDT-trades-%Y-%m-%d.csv")

            print("preparing " + str(prepare_start))

            self.df = pd.read_csv(file_name)
            self.current_trade_row = 0
            self.done = False


            if self.df is None:
                continue

            for index, row in self.df.iloc[self.current_trade_row:].iterrows():
                self.price = float(row.iloc[1])
                volume = float(row.iloc[2])
                is_buyer_maker = row.iloc[5]
                if pd.isna(row.iloc[4]):
                    continue

                self.trade_time = datetime.datetime.fromtimestamp(row.iloc[4] / 1000)
                self.aggregated_volume_50 += volume
                self.aggregated_trades_count_50 += 1

                new_trade = np.array([[self.price, volume, self.trade_time.timestamp(), int(is_buyer_maker)]])
                self.trades = np.append(self.trades, new_trade, axis=0)

                if self.aggregated_trades_count_50 >= 50:
                    new_trade_agg = np.array([[self.aggregated_price_50, self.aggregated_volume_50, self.trade_time.timestamp(), int(0)]])  # Wrap in two sets of brackets to maintain 2D
                    self.aggregated_trades_50 = np.append(self.aggregated_trades_50, new_trade_agg, axis=0)
                    self.aggregated_price_50 = self.price
                    self.aggregated_trades_count_50 = 0

                self.current_trade_row += 1

            prepare_start = prepare_start + datetime.timedelta(days=1)

        self.db_size = ChartStandardized().get_bar_size(self.trades, self.trade_time)
        file_name = start_date.strftime(self.data_path + "/FTMUSDT-trades-%Y-%m-%d.csv")
        self.df = pd.read_csv(file_name)
        self.done = False


    def get_obs(self, step):
        # Process the action and update environment state
        dollar_bar_complete = False 
        while len(self.df.iloc) >= self.current_trade_row and dollar_bar_complete is False:
            for index, row in self.df.iloc[self.current_trade_row:].iterrows():
                self.price = float(row.iloc[1])
                volume = float(row.iloc[2])
                is_buyer_maker = row.iloc[5]
                if pd.isna(row.iloc[4]):
                    continue

                self.trade_time = datetime.datetime.fromtimestamp(row.iloc[4] / 1000)
                self.aggregated_volume_50 += volume
                self.aggregated_trades_count_50 += 1

                if self.aggregated_trades_count_50 >= 50:
                    new_trade_agg = np.array([[self.aggregated_price_50, self.aggregated_volume_50, self.trade_time.timestamp(), int(0)]])  # Wrap in two sets of brackets to maintain 2D
                    self.aggregated_trades_50 = np.append(self.aggregated_trades_50, new_trade_agg, axis=0)
                    self.aggregated_price_50 = self.price
                    self.aggregated_trades_count_50 = 0

                new_trade = np.array([[self.price, volume, self.trade_time.timestamp(), int(is_buyer_maker)]])  # Wrap in two sets of brackets to maintain 2D
                self.trades = np.append(self.trades, new_trade, axis=0)
                self.current_db_size += self.price * volume

                if self.latest_buy_price is None and not is_buyer_maker:
                    self.latest_buy_price = self.price
                if self.latest_sell_price is None and is_buyer_maker:
                    self.latest_sell_price = self.price
                
                self.spread = self.latest_sell_price - self.latest_buy_price

                # Calculate dollar bar size -> volume last 24h/1440
                self.current_trade_row += 1

                if len(self.df.iloc) <= self.current_trade_row:
                    self.start_date = self.start_date + datetime.timedelta(days=1)
                    file_name = self.start_date.strftime(self.data_path + "/FTMUSDT-trades-%Y-%m-%d.csv")
                    self.df = pd.read_csv(file_name)
                    self.current_trade_row = 0

                if self.current_db_size >= self.db_size:
                    self.db_size = ChartStandardized().get_bar_size(self.trades, self.trade_time)
                    self.current_db_size = 0
                    dollar_bar_complete = True

                    threshold_time_day = (self.trade_time - datetime.timedelta(days=1)).timestamp()
                    self.trades = self.trades[self.trades[:, 2] > threshold_time_day]

                    threshold_time_five = (self.trade_time - datetime.timedelta(days=5)).timestamp()
                    self.aggregated_trades_50 = self.aggregated_trades_50[self.aggregated_trades_50[:, 2] > threshold_time_five]
                    break


        db_bars = ChartStandardized().construct_ohlc(self.trades, self.db_size)

        observation = {
            'position': np.array([self.exposure, self.spread]),
            'weekly_vp': np.array(VolumeProfileSimple().get_volume_profile_flow(self.aggregated_trades_50, self.trade_time, db_bars, 32, datetime.timedelta(days=5))),
            'daily_vp': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 32, datetime.timedelta(days=1))),
            'vp_vb_500': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 24, datetime.timedelta(hours=8))),
            'vp_vb_200': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 16, datetime.timedelta(hours=4))),
            'vp_vb_100': np.array(VolumeProfileSimple().get_volume_profile_flow(self.trades, self.trade_time, db_bars, 12, datetime.timedelta(hours=2))),
        }


        return observation
    
    def calculate_reward(self, action):
        exposure = action[0]
        # Calculate the total value of the portfolio at the previous price
        previous_portfolio_value = abs(self.shares_owned) * self.previous_price + self.cash

        if exposure == self.previous_exposure:
            # Calculate the current portfolio value without adjusting shares or cash, just market price changes
            current_portfolio_value = abs(self.shares_owned) * self.price + self.cash
            reward = -np.log(current_portfolio_value / previous_portfolio_value) if exposure < 0 else np.log(current_portfolio_value / previous_portfolio_value)
            self.previous_price = self.price
            return reward  if not np.isnan(reward) and not np.isinf(reward) else 0

        # Calculate total portfolio value without including spread
        total_portfolio_value = self.cash + (self.price * abs(self.shares_owned))
        desired_position = exposure * total_portfolio_value
        
        # Adjust the buy price for new shares to include the spread, simulating a realistic purchase price
        adjusted_buy_price = self.price + self.spread if exposure > self.previous_exposure else self.price
        desired_shares = round(desired_position / adjusted_buy_price)
        share_change = desired_shares - self.shares_owned

        # Calculate transaction fees without spread for selling shares; spread is included in adjusted buy price
        transaction_fee_rate = 0.001
        total_transaction_fee = abs(share_change * adjusted_buy_price) * transaction_fee_rate if share_change > 0 else abs(share_change * self.price) * transaction_fee_rate

        # Update shares and cash to reflect the new position and fees
        if share_change > 0:  # Buying shares
            self.cash -= (share_change * adjusted_buy_price + total_transaction_fee)
        else:  # Selling shares
            self.cash -= (share_change * self.price - total_transaction_fee)  # Gaining cash from selling, pay transaction fee

        self.shares_owned = desired_shares

        # Recalculate the current portfolio value after executing the trade
        current_portfolio_value = abs(self.shares_owned) * self.price + self.cash

        self.previous_price = self.price
        self.previous_exposure = exposure

        # Calculate reward based on the direction of exposure and change in portfolio value
        reward = -np.log(current_portfolio_value / previous_portfolio_value) if exposure < 0 else np.log(current_portfolio_value / previous_portfolio_value)

        return reward if not np.isnan(reward) and not np.isinf(reward) else 0
    
    # def calculate_sortino_ratio(self):
    #     if len(self.returns) < 2:  # Need at least two returns to calculate standard deviation
    #         return random.uniform(-0.5, 0)

    #     # Calculate mean return
    #     mean_return = np.mean(self.returns)

    #     # Calculate downside deviation
    #     negative_returns = [x for x in self.returns if x < 0]
    #     if len(negative_returns) == 0:
    #         return mean_return

    #     downside_deviation = np.std(negative_returns)
    #     if downside_deviation == 0:
    #         return mean_return

    #     # Calculate Sortino ratio
    #     sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation
    #     return sortino_ratio

    # def calculate_return(self):
    #         return 0

    #     transaction_cost_rate = 0.001  # Example: 0.1% transaction cost rate

    #     # Calculate the return without transaction costs

    #     # Calculate transaction costs for both buying and selling
    #     sell_transaction_cost = (self.price - self.spread) * transaction_cost_rate

    #     # Convert transaction costs to percentage of the buying price

    #     # Final return after accounting for transaction costs
    #     final_return = profit_loss_percentage - total_transaction_cost_percentage
    #     return final_return
    
    
