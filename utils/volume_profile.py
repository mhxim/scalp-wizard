from collections import defaultdict
import datetime
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import numpy as np

class VolumeProfileSimple:
    def __init__(self, kde_factor=0.04,):
      self.kde = None
      self.kde_factor = kde_factor
      self.peak_ranges = []


    def get_price_volume_pairs(self, trade_time, trades, timerange, section_size = 32):
        # Filter trades based on the specified timeframe
        filtered_trades = [trade for trade in trades if trade[2] >= (trade_time - timerange).timestamp()]
        
        # Calculate sectioned order flow for the filtered trades
        sectioned_volume, section_prices = self.calculate_sectioned_order_flow(trade_time, filtered_trades, section_size)

        # Find the highest and lowest prices in the timeframe for normalization
        highest_price = max(trade[0] for trade in filtered_trades) if filtered_trades else 0
        lowest_price = min(trade[0] for trade in filtered_trades) if filtered_trades else 0

        # Find the highest volume for normalization
        highest_volume = max(sectioned_volume.values()) if sectioned_volume else 0
        lowest_volume = min(sectioned_volume.values()) if sectioned_volume else 0
        volume_range = highest_volume - lowest_volume

        volume_profile = []

        for section in range(section_size):
            lower_bound, upper_bound = section_prices.get(section, (-1, -1))
            volume = sectioned_volume.get(section, -1)

            if lower_bound != -1 and upper_bound != -1:
                price = (lower_bound + upper_bound) / 2 # Midpoint of the price range
                # Normalize price and volume and scale to 0-100 range
                norm_volume = round(((volume - lowest_volume) / volume_range), 3) if volume_range > 0 else 0
                volume_profile.extend([[price, norm_volume]])
            else:
                # Append default value for missing sections
                volume_profile.append([0, 0])

        # Ensure volume_profile has exactly 68 entries
        while len(volume_profile) < section_size:
            volume_profile.append([0, 0])

        return highest_price, lowest_price, self.calculate_poc(trade_time, trades, timerange, section_size), volume_profile
    
    def calculate_sectioned_order_flow(self, trade_time, trades, section_size):
        if not trades:
            return {}, {}

        # self.update_order_flow(trade_time)

        max_price = max(trade[0] for trade in trades)
        min_price = min(trade[0] for trade in trades)
        price_range = max_price - min_price
        section_size = max(price_range / section_size, 0.0001)

        sectioned_order_flow = defaultdict(int)
        section_prices = {}

        for price, volume, *_ in trades:
            section = int((price - min_price) / section_size)
            sectioned_order_flow[section] += volume
            lower_bound = min_price + section * section_size
            upper_bound = lower_bound + section_size
            section_prices[section] = (lower_bound, upper_bound)

        return sectioned_order_flow, section_prices

    def calculate_poc(self, trade_time, trades, timeframe, section_size):
        last_trades = [trade for trade in trades if trade[2] >= (trade_time - timeframe).timestamp()]
        sectioned_order_flow, section_prices = self.calculate_sectioned_order_flow(trade_time, last_trades, section_size)

        if not sectioned_order_flow:
            return -1

        max_volume_section = max(sectioned_order_flow, key=sectioned_order_flow.get)
        lower_bound, upper_bound = section_prices[max_volume_section]

        # POC is the midpoint of the section with the highest volume
        poc = (lower_bound + upper_bound) / 2
        return poc
    
    def get_volume_profile_flow(self, trades, trade_time, ohlc_bars, section_size, timerange=datetime.timedelta(days=1)):
        vp_timeframe_features = []

        # vp ~ [[price,volume]]
        highest_price, lowest_price, poc, volume_profile_sections = self.get_price_volume_pairs(trade_time, trades, timerange, section_size)
        highest_daily, lowest_daily = max(ohlc[1] for ohlc in ohlc_bars), min(ohlc[2] for ohlc in ohlc_bars)

        highest_price = max(highest_price, highest_daily)
        lowest_price = min(lowest_price, lowest_daily)

        for bar_index, ohlc_bar in enumerate(ohlc_bars):
            _, _, low_price, _ = ohlc_bar[0], ohlc_bar[1], ohlc_bar[2], ohlc_bar[3] 

            vp_range = highest_price - lowest_price
            poc_position_vp = 0

            if vp_range > 0:
                poc_distance_from_vp_low = poc - lowest_price
                poc_position_vp = round(poc_distance_from_vp_low / vp_range, 2)

            volume_profile_sections_norm = []
            for volume_profile_section in volume_profile_sections:
                section_price, section_norm_volume = volume_profile_section[0], volume_profile_section[1]

                if section_price > 0:
                    price_distance_from_vp_section = section_price - low_price

                    section_price_distance_norm = 0
                    if vp_range > 0:
                        section_price_distance_norm = round(price_distance_from_vp_section / vp_range, 3)

                    volume_profile_sections_norm.append(section_price_distance_norm)
                    volume_profile_sections_norm.append(section_norm_volume)
                else:
                    volume_profile_sections_norm.append(0)
                    volume_profile_sections_norm.append(0)

            vp_timeframe_features.append([poc_position_vp] + volume_profile_sections_norm)

        return vp_timeframe_features
    
    # def update_ranges(self, trade_time, trades):
    #     if len(trades) < 5:
    #         return

    #     new_ranges = self.find_volume_nodes(trade_time, trades, self.timerange)

    #     if new_ranges is not None:
    #         updated_ranges = self.filter_ranges(new_ranges, self.peak_ranges)
    #         if updated_ranges is not None:
    #             self.peak_ranges = updated_ranges

    # def filter_ranges(self, new_ranges, old_ranges):
    #     # Check if new_peak_ranges is empty
    #     if not new_ranges:
    #         return
        
    #     filtered_new_peak_ranges = []
    #     for i in range(len(new_ranges)):
    #         x0, x1 = new_ranges[i]
    #         overlap = False
    #         for j in range(len(new_ranges)):
    #             if i != j:
    #                 y0, y1 = new_ranges[j]
    #                 if x0 <= y1 and y0 <= x1:
    #                     overlap = True
    #                     if (x1 - x0) >= (y1 - y0):
    #                         filtered_new_peak_ranges.append((x0, x1))
    #                     break
    #         if not overlap:
    #             filtered_new_peak_ranges.append((x0, x1))

    #     overall_low = min(x0 for x0, _ in filtered_new_peak_ranges)
    #     overall_high = max(x1 for _, x1 in filtered_new_peak_ranges)

    #     # Filter existing peak ranges
    #     filtered_peak_ranges = [r for r in old_ranges if r[1] < overall_low * 1 or r[0] > overall_high * 1]

    #     # Add new peak ranges to the filtered list
    #     return filtered_peak_ranges + filtered_new_peak_ranges

    # def find_volume_nodes(self, trade_time, trades, timerange):
    #     timerange_before_latest = trade_time - timerange
    #     time_range_trades = [trade for trade in trades if trade[2] >= timerange_before_latest]

    #     sectioned_volume, section_prices = self.calculate_sectioned_order_flow(trade_time, time_range_trades)
    #     if not sectioned_volume or not section_prices.keys() or len(section_prices.keys()) < 1:
    #         return []

    #     # Prepare data for KDE
    #     sections = list(section_prices.keys())
    #     volumes = [sectioned_volume[s] for s in sections]

    #     if len(volumes) < 2 or not any(volumes):
    #         # Not enough data or all volumes are zero
    #         return []

    #     # Check if weights sum up to more than zero before applying KDE
    #     if sum(volumes) == 0:
    #         # This avoids the RuntimeWarning when weights sum to zero
    #         return []

    #     # Apply KDE
    #     kde = gaussian_kde(sections, weights=volumes, bw_method=self.kde_factor)
    #     xr = np.linspace(min(sections), max(sections), len(sections))
    #     kdy = kde(xr)

    #     max_price = max(trade[0] for trade in time_range_trades)
    #     min_price = min(trade[0] for trade in time_range_trades)
    #     price_range = max_price - min_price
    #     ticks_per_sample = max(price_range / self.section_size, 0.0001)


    #     # Find peaks representing HVNs
    #     min_prom = kdy.max() * 0.5
    #     width_range=1
    #     peaks, peak_props = find_peaks(kdy, width=width_range, prominence=min_prom, rel_height=0.85)
    #     pkx = xr[peaks]
    #     pky = kdy[peaks]

    #     # Recalculate ticks_per_sample based on the actual price range

    #     left_ips = peak_props['left_ips']
    #     right_ips = peak_props['right_ips']


    #     max_time = max(trade[2] for trade in time_range_trades)

    #     gmt_plus_1_timezone = pytz.timezone('Europe/Zurich')
    #     gmt_plus_1_time = max_time.astimezone(gmt_plus_1_timezone)

    #     # Calculate the price range for each peak
    #     width_x0 = min_price + (left_ips * ticks_per_sample)
    #     width_x1 = min_price + (right_ips * ticks_per_sample)

    #     return [(x0, x1) for x0, x1 in zip(width_x0, width_x1)]
    
    # def get_next_hvn_lvn(self, price):
    #     next_hvn_resistance = -1
    #     next_hvn_support = -1

    #     self.peak_ranges.sort(key=lambda x: x[0])

    #     for lower, upper in self.peak_ranges:
    #         if price >= lower and price <= upper:
    #             return lower, upper 


    #     hvn_ranges_above = [r for r in self.peak_ranges if r[0] > price]
    #     hvn_ranges_below = [r for r in self.peak_ranges if r[1] < price]

    #     if len(hvn_ranges_below) > 0:
    #         next_hvn_support = max(hvn_ranges_below, key=lambda x: x[1])[1]
    #     if len(hvn_ranges_above) > 0:
    #         next_hvn_resistance = min(hvn_ranges_above, key=lambda x: x[0])[0]

    #     return next_hvn_support, next_hvn_resistance
