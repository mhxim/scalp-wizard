from collections import defaultdict
import numpy as np

class ChartStandardized:
    def construct_ohlc(self, trades, dollar_bar_amount):
        ohlc_data = []  # List to hold each OHLC data row
        current_bar_value = 0.0
        bar_open, bar_high, bar_low, bar_close = None, float('-inf'), float('inf'), None

        for trade in trades[::-1]:
            if len(ohlc_data) >= 100:
                break

            price, volume, timestamp, is_buyer_maker = trade
            trade_value = price * volume

            if current_bar_value == 0:  # This means a new bar is starting
                bar_open = price  # First trade price of the bar is the open price

            bar_high = max(bar_high, price)  # Update high price
            bar_low = min(bar_low, price)  # Update low price
            bar_close = price  # Last trade price is the close price

            current_bar_value += trade_value  # Accumulate the dollar value for the current bar

            # Check if the current bar has reached the specified dollar amount
            if current_bar_value >= dollar_bar_amount:
                # Append the OHLC data for the current bar to the list
                ohlc_data.append([bar_open, bar_high, bar_low, bar_close])

                # Reset the variables for the next bar
                current_bar_value = 0.0
                bar_open, bar_high, bar_low, bar_close = None, float('-inf'), float('inf'), None


        # Convert the list of OHLC data to a 2D NumPy array
        ohlc_array = np.array(ohlc_data)

        return ohlc_array


    def get_bar_size(self, trades, trade_time):
        dollars_traded_last_24h = 0
        twenty_four_hours_ago = trade_time.timestamp() - (24 * 60 * 60)  # 24 hours ago in seconds

        # (price, volume, trade_time.timestamp(), is_buyer_maker)
        for trade in trades:
            if trade[2] >= twenty_four_hours_ago:
                dollars_traded_last_24h += trade[0] * trade[1]

        return dollars_traded_last_24h / (1440)