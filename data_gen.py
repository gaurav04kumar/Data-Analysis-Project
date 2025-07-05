# data_gen.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

def simulate_initial_data(base_price=1518.5, num_time_steps=30, num_price_levels=20):
    """Simulates initial data for 3D order book, volume profile, and bars."""
    prices = np.linspace(base_price - 0.5, base_price + 0.5, num_price_levels).round(2)
    times = [datetime(2024, 7, 5, 19, 30) + timedelta(seconds=30 * i) for i in range(num_time_steps)]

    # Simulate initial heatmap data
    initial_heatmap_data = []
    for t in times:
        for p in prices:
            bid_volume = 0
            ask_volume = 0
            if p <= base_price:
                bid_volume = np.random.randint(100 + int((base_price - p) * 500), 1000 + int((base_price - p) * 1000))
            if p >= base_price:
                 ask_volume = np.random.randint(100 + int((p - base_price) * 500), 1000 + int((p - base_price) * 1000))

            initial_heatmap_data.append({
                "timestamp": t,
                "price": p,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume
            })
    df_heatmap = pd.DataFrame(initial_heatmap_data)

    # Simulate initial volume profile
    df_volume = pd.DataFrame({
        "price": prices,
        "volume": np.random.randint(1000, 5000, len(prices))
    })

    # Simulate initial candlestick + volume bars
    bars = []
    last_close = base_price
    for t in times:
        price_change = np.random.normal(0, 0.1) # Increased volatility
        open_ = last_close + np.random.normal(0, 0.02)
        close = open_ + price_change
        high = max(open_, close) + np.random.uniform(0, 0.05)
        low = min(open_, close) - np.random.uniform(0, 0.05)
        volume = np.random.randint(100, 1500) # Increased volume range

        bar_data = {
            "timestamp": t,
            "open": round(open_, 2),
            "close": round(close, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "volume": volume
        }
        bars.append(bar_data)
        last_close = bar_data["close"] # Update last_close for the next bar

    df_bars = pd.DataFrame(bars)

    return df_heatmap, df_volume, df_bars, prices, base_price


def simulate_new_data(last_timestamp, prices, base_price):
    """Simulates new real-time data for one time step with bid/ask for 3D heatmap."""
    new_timestamp = last_timestamp + timedelta(seconds=30)

    # Simulate new heatmap data - Bid and Ask for one timestamp
    heatmap_data_rows = []
    for p in prices:
        bid_volume = 0
        ask_volume = 0
        if p <= base_price:
            bid_volume = np.random.randint(100 + int((base_price - p) * 500), 1000 + int((base_price - p) * 1000))
        if p >= base_price:
            ask_volume = np.random.randint(100 + int((p - base_price) * 500), 1000 + int((p - base_price) * 1000))

        heatmap_data_rows.append({
            "timestamp": new_timestamp,
            "price": p,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume
        })

    new_heatmap_df = pd.DataFrame(heatmap_data_rows)

    # Simulate new volume profile data (can be updated based on trades, here just resample)
    new_volume_profile_df = pd.DataFrame({
        "price": prices,
        "volume": np.random.randint(1000, 5000, len(prices))
    })

    # Simulate new candlestick bar with more dynamic movement
    # Note: In a real scenario, the previous close would come from the actual last bar
    # For this simulation, we'll assume df_bars is available in the calling scope or passed in
    # Assuming df_bars is available for now.
    # A more robust approach would be to pass the last bar's close price.
    # For this simulation, we'll get the last close from the global df_bars
    # This is not ideal for a standalone function but works within the notebook context.
    # If this function were truly standalone, last_close should be an argument.

    # To make it more robust for extraction, let's pass the last close
    # This requires a change in the calling code later. For now, keep the original logic
    # but acknowledge the dependency on df_bars from the outer scope.

    # Simulating based on a potential last close if df_bars is not empty
    # This part needs refactoring if used truly standalone.
    # For now, let's assume df_bars is accessible as in the notebook.
    try:
        last_close = df_bars.iloc[-1]["close"]
    except (NameError, IndexError):
        # If df_bars is not defined or empty, use base_price
        last_close = base_price

    price_change = np.random.normal(0, 0.1) # Increased volatility
    open_ = last_close + np.random.normal(0, 0.02)
    close = open_ + price_change
    high = max(open_, close) + np.random.uniform(0, 0.05)
    low = min(open_, close) - np.random.uniform(0, 0.05)
    volume = np.random.randint(100, 1500) # Increased volume range

    new_bar_df = pd.DataFrame([{
        "timestamp": new_timestamp,
        "open": round(open_, 2),
        "close": round(close, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "volume": volume
    }])

    return new_heatmap_df, new_volume_profile_df, new_bar_df