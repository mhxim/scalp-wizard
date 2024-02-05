import datetime
import numpy as np

class OrderFlow:
    def calculate_cvd(self, trades, latest_trade_time, timeframe):
        volume_delta = 0

        timeframe = datetime.timedelta(minutes=timeframe)
        timeframe_before_latest = latest_trade_time - timeframe

        for trade in [trade for trade in trades if trade[2] >= timeframe_before_latest]:
            volume = float(trade[1])
            is_buyer_maker = trade[3]
            order_delta = -volume if is_buyer_maker else volume
            volume_delta += order_delta

        return volume_delta
    
    def get_volume_delta(self, trades, trade_time, seconds, bars):
        self.update_order_flow(trade_time)
        volume_deltas = []

        for i in range(bars):
            bar_end_time = trade_time - datetime.timedelta(seconds=seconds * i)
            bar_start_time = bar_end_time - datetime.timedelta(seconds=seconds)
            bar_trades = [trade for trade in trades if bar_start_time < trade[2] <= bar_end_time]

            if len(bar_trades) == 0:
                volume_deltas.append(0)
                continue

            current_delta = sum(-trade[1] if trade[3] else trade[1] for trade in bar_trades)
            volume_deltas.append(current_delta)

        return volume_deltas
    
    def prepare_volume_delta_array(self, volume_delta_array):
        vol_delta_with_position = []

        for index, volume_delta in enumerate(volume_delta_array):
            has_position_in_bar = 1 if self.position_open_duration > 60 - index else 0
            
            if index == 0 or volume_delta_array[index - 1] < 0 or volume_delta < 0:
                vol_delta_with_position.append([0, has_position_in_bar])
                continue
            
            vol_delta_with_position.append([volume_delta, has_position_in_bar])
        
        return vol_delta_with_position