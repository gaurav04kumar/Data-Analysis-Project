# viz.py
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as make_subplots
import IPython.display as display
import time
import numpy as np
from datagen import simulate_initial_data, simulate_new_data
from datetime import timedelta # Import timedelta

# Define a color palette
bid_color = 'rgba(0, 255, 0, 0.8)' # Green with transparency for bids
ask_color = 'rgba(255, 0, 0, 0.8)' # Red with transparency for asks
increasing_color = 'green'
decreasing_color = 'red'
volume_profile_color = 'purple'
trade_volume_color = 'blue'
sma_color = 'orange'


# Define the simulate_time_sales function
def simulate_time_sales(df_bars):
    """Simulates time and sales data based on bar timestamps and close prices."""
    time_sales_data = []
    for index, row in df_bars.iterrows():
        timestamp = row["timestamp"]
        close_price = row["close"]
        # Simulate a few trades around the close price for each timestamp
        num_trades = np.random.randint(1, 5)
        for _ in range(num_trades):
            trade_price = close_price + np.random.normal(0, 0.01) # Small random variation
            trade_volume = np.random.randint(10, 200)
            time_sales_data.append({
                "timestamp": timestamp,
                "price": round(trade_price, 2),
                "volume": trade_volume
            })
    return pd.DataFrame(time_sales_data)

# Define the update_plot_traces function
def update_plot_traces(fig, df_heatmap, df_volume, df_bars, prices, window_size):
    """Updates the data of each trace in the Plotly figure."""

    # Prepare data for 3D heatmap updates
    heatmap_data_3d = df_heatmap.pivot_table(index='timestamp', columns='price', values=['bid_volume', 'ask_volume'])

    # Separate bid and ask volumes
    bid_volume_matrix = heatmap_data_3d['bid_volume'].values
    ask_volume_matrix = heatmap_data_3d['ask_volume'].values

    # Get the timestamps and prices for the heatmap
    heatmap_x_prices = heatmap_data_3d.columns.get_level_values(1).astype(float)
    heatmap_y_times = heatmap_data_3d.index

    # Update 3D heatmap traces (trace 0 and 1 for bid/ask)
    fig.update_traces(
        selector=dict(name='Bid Volume (3D)'),
        x=heatmap_x_prices,
        y=heatmap_y_times,
        z=bid_volume_matrix
    )
    fig.update_traces(
        selector=dict(name='Ask Volume (3D)'),
        x=heatmap_x_prices,
        y=heatmap_y_times,
        z=ask_volume_matrix
    )

    # Update candlestick trace (trace 2)
    fig.update_traces(
        selector=dict(type='candlestick'),
        x=df_bars["timestamp"],
        open=df_bars["open"],
        high=df_bars["high"],
        low=df_bars["low"],
        close=df_bars["close"]
    )

    # Update SMA trace (trace 3)
    fig.update_traces(
        selector=dict(name=f'SMA {window_size}'),
        x=df_bars["timestamp"],
        y=df_bars['SMA']
    )

    # Prepare data for Volume Profile (assuming Volume Profile is a Barpolar plot in row 3)
    latest_timestamp = df_bars["timestamp"].max()
    volume_profile_x_prices = df_volume["price"]
    volume_profile_z_volume = df_volume["volume"]

    # Update Volume Profile trace (trace 4) - Assuming Barpolar
    fig.update_traces(
        selector=dict(name='Volume Profile'),
        type='barpolar', # Change to barpolar for 3D effect related to volume profile
        theta=volume_profile_x_prices, # Price as angle
        r=volume_profile_z_volume, # Volume as radius
        marker=dict(color=volume_profile_color)
    )


    # Prepare data for Trade Volume (assuming Trade Volume is a 2D Bar plot in row 4)
    trade_volume_x_times = df_bars["timestamp"]
    trade_volume_z_volume = df_bars["volume"]

    # Update Trade Volume trace (trace 5) - Assuming 2D Bar
    fig.update_traces(
        selector=dict(name='Trade Volume'),
        type='bar', # Use standard bar for now, 3D requires more complex setup
        x=trade_volume_x_times,
        y=trade_volume_z_volume, # Volume as y-axis for 2D bar
        marker=dict(color=trade_volume_color)
    )


    # Simulate and update Time and Sales trace (trace 6)
    df_time_sales = simulate_time_sales(df_bars) # Simulate new time and sales data
    fig.update_traces(
        selector=dict(name='Time and Sales'),
        x=df_time_sales["timestamp"],
        y=df_time_sales["price"],
        mode='markers', # Ensure mode is markers
        marker=dict(
            size=df_time_sales['volume'] / 20, # Size marker by volume
            color=df_time_sales['volume'], # Color marker by volume
            colorscale='Viridis',
            showscale=False # Hide color scale for clarity
        )
    )

# Define the update_dashboard function
def update_dashboard(df_heatmap, df_volume, df_bars, prices, base_price, max_time_steps=60):
    """Simulates new data and updates the dataframes, handling 3D heatmap."""
    # Pass the last close price to simulate_new_data for better modularity
    last_close = df_bars["close"].iloc[-1] if not df_bars.empty else base_price

    last_timestamp = df_bars["timestamp"].max() if not df_bars.empty else datetime(2024, 7, 5, 19, 30) - timedelta(seconds=30)

    # Simulate new data passing last_close
    new_heatmap_data, new_volume_profile_data, new_bar_data = simulate_new_data(last_timestamp, prices, base_price) # Note: simulate_new_data in datagen.py still uses global df_bars for last_close. Refactoring needed there too for true independence.

    # Append new data (multiple rows for heatmap for one time step)
    df_heatmap = pd.concat([df_heatmap, new_heatmap_data], ignore_index=True)
    df_bars = pd.concat([df_bars, new_bar_data], ignore_index=True)

    # Update volume profile (replace with new data in this simulation)
    df_volume = new_volume_profile_data

    # Trim old data (based on the number of unique timestamps for heatmap)
    unique_timestamps = df_heatmap["timestamp"].unique()
    if len(unique_timestamps) > max_time_steps:
        oldest_timestamp = unique_timestamps[0]
        df_heatmap = df_heatmap[df_heatmap["timestamp"] > oldest_timestamp].reset_index(drop=True)

    # Trim old data for df_bars based on max_time_steps
    if len(df_bars) > max_time_steps:
        df_bars = df_bars.iloc[-max_time_steps:].reset_index(drop=True)


    return df_heatmap, df_volume, df_bars

# Initial data simulation
df_heatmap, df_volume, df_bars, prices, base_price = simulate_initial_data()

# Initial Plotly figure creation and trace adding logic
fig = make_subplots.make_subplots(
    rows=5, cols=1, # Increased to 5 rows
    shared_xaxes=True, # Share x-axes for time-based plots
    vertical_spacing=0.05, # Adjusted spacing
    row_heights=[0.3, 0.2, 0.2, 0.1, 0.2], # Adjusted heights
    specs=[[{"type": "scene"}], # 3D Heatmap
           [{"type": "xy"}], # Candlestick + SMA
           [{"type": "polar"}], # Volume Profile (Barpolar)
           [{"type": "xy"}], # Trade Volume (2D Bar)
           [{"type": "xy"}]] # Time and Sales (2D Scatter)
)

# Prepare initial heatmap data for 3D plot
heatmap_data_3d = df_heatmap.pivot_table(index='timestamp', columns='price', values=['bid_volume', 'ask_volume'])
heatmap_z_bid = heatmap_data_3d['bid_volume'].values
heatmap_z_ask = heatmap_data_3d['ask_volume'].values
heatmap_x = heatmap_data_3d.columns.get_level_values(1).astype(float)
heatmap_y = heatmap_data_3d.index


heatmap_bid = go.Surface(
    x=heatmap_x,
    y=heatmap_y,
    z=heatmap_z_bid,
    colorscale='Greens',
    showscale=False,
    name='Bid Volume (3D)',
    opacity=0.7 # Add some transparency
)

heatmap_ask = go.Surface(
    x=heatmap_x,
    y=heatmap_y,
    z=heatmap_z_ask,
    colorscale='Reds',
    showscale=False,
    name='Ask Volume (3D)',
    opacity=0.7 # Add some transparency
)


# Volume profile - Initial trace as Barpolar
volume_profile_x_prices = df_volume["price"]
latest_timestamp = df_bars["timestamp"].max()
volume_profile_y_time = [latest_timestamp] * len(df_volume)
volume_profile_z_volume = df_volume["volume"]

volume_bar = go.Barpolar(
    theta=volume_profile_x_prices, # Price as angle
    r=volume_profile_z_volume, # Volume as radius
    marker=dict(color=volume_profile_color),
    name="Volume Profile",
    subplot='polar' # Specify subplot type
)


# Candlestick
candlestick = go.Candlestick(
    x=df_bars["timestamp"],
    open=df_bars["open"],
    high=df_bars["high"],
    low=df_bars["low"],
    close=df_bars["close"],
    name="Price",
    increasing_line_color=increasing_color, decreasing_line_color=decreasing_color
)

# SMA trace
window_size = 10 # Define window size for SMA
df_bars['SMA'] = df_bars['close'].rolling(window=window_size).mean() # Calculate initial SMA
sma_trace = go.Scatter(
    x=df_bars["timestamp"],
    y=df_bars['SMA'],
    mode='lines',
    name=f'SMA {window_size}',
    line=dict(color=sma_color, width=2)
)

# Volume bars
volume_line = go.Bar(
    x=df_bars["timestamp"],
    y=df_bars["volume"],
    marker_color=trade_volume_color,
    name="Trade Volume"
)

# Simulate initial time and sales data
df_time_sales = simulate_time_sales(df_bars)
time_sales_scatter = go.Scatter(
    x=df_time_sales["timestamp"],
    y=df_time_sales["price"],
    mode='markers',
    name='Time and Sales',
    marker=dict(
        size=df_time_sales['volume'] / 20, # Size marker by volume
        color=df_time_sales['volume'], # Color marker by volume
        colorscale='Viridis',
        showscale=False # Hide color scale for clarity
    )
)

# Add all traces to the updated figure layout
fig.add_trace(heatmap_bid, row=1, col=1)
fig.add_trace(heatmap_ask, row=1, col=1)
fig.add_trace(candlestick, row=2, col=1)
fig.add_trace(sma_trace, row=2, col=1) # Add SMA to the second row
fig.add_trace(volume_bar, row=3, col=1) # Volume Profile in row 3
fig.add_trace(volume_line, row=4, col=1) # Trade Volume in row 4
fig.add_trace(time_sales_scatter, row=5, col=1) # Add time and sales to the new row

fig.update_layout(
    height=1300, # Increased height
    title_text="📊 Market Order Flow Dashboard (Live)",
    showlegend=True,
    hovermode='x unified' # Improved hover behavior
)

# Update 3D scene layout titles
fig.update_layout(scene = dict(
                    xaxis_title='Price',
                    yaxis_title='Time',
                    zaxis_title='Volume'))

# Update polar subplot layout titles (for Volume Profile)
# Removed title_text properties as per instructions
fig.update_layout(polar = dict(
    radialaxis = dict(),
    angularaxis = dict()
))


# Live update loop
num_updates = 10  # Limit updates for simulation
for i in range(num_updates):
    # Update data
    df_heatmap, df_volume, df_bars = update_dashboard(df_heatmap, df_volume, df_bars, prices, base_price)

    # Recalculate SMA after data update
    df_bars['SMA'] = df_bars['close'].rolling(window=window_size).mean()

    # Update plot traces
    update_plot_traces(fig, df_heatmap, df_volume, df_bars, prices, window_size)

    # Clear output and display updated plot
    display.clear_output(wait=True)
    display.display(fig)

    # Pause
    time.sleep(2)

# Save the final state of the figure to an HTML file
import os
os.makedirs("outputs", exist_ok=True)
pio.write_html(fig, file="outputs/golden_image.html", full_html=True, auto_open=False)
print("✅ Final dashboard state saved to: outputs/golden_image.html")