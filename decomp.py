import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# dataloading

file_name = '07_19_2025100k_samples_txp_1551.5_pax_1552.5_polcon_and_fiber_1Hz.mat'

mat_data = scipy.io.loadmat(file_name)

# vars
variables = ['s1_pax', 's2_pax', 's3_pax', 's1_txp', 's2_txp', 's3_txp']

# analysis parameters
subset_size = 100000
period_length = 10000

decompositions = {} 

# generate individual decompositions
for var_name in variables:
    if var_name in mat_data:
        # extract data
        data = mat_data[var_name].flatten()
        ts = pd.Series(data[:subset_size])
        
        # decompose
        decomp = seasonal_decompose(ts, model='additive', period=period_length)
        decompositions[var_name] = decomp
        
        # plot individual decomposition
        fig = decomp.plot()
        fig.set_size_inches(10, 6)
        plt.suptitle(f'Decomposition of {var_name}', y=1.02)
        
        # save figure
        save_name = f'decomposition_{var_name}.png'
        plt.savefig(save_name)
    else:
        print(f"  Warning: {var_name} not found in .mat file.")

# plot combined seasonality and trend
fig_season_all, ax_season_all = plt.subplots(figsize=(12, 6))
fig_trend_all, ax_trend_all = plt.subplots(figsize=(12, 6))

for var_name in variables:
    if var_name in decompositions:
        decomp = decompositions[var_name]
        ax_season_all.plot(decomp.seasonal, label=var_name, alpha=0.7)
        ax_trend_all.plot(decomp.trend, label=var_name, linewidth=2)

# Configuration
bases = ['s1', 's2', 's3']
colors = {'s1': 'blue', 's2': 'orange', 's3': 'green'}
styles = {'pax': '-', 'txp': '--'}  # Solid for Output, Dashed for Input
labels_map = {'pax': 'Output (PAX)', 'txp': 'Input (TXP)'}

# Initialize Figures
fig_season_color, ax_season_color = plt.subplots(figsize=(12, 6))
fig_trend_color, ax_trend_color = plt.subplots(figsize=(12, 6))

for base in bases:
    fig_base_raw, ax_base_raw = plt.subplots(figsize=(12, 6))

    c = colors[base]
    for kind in ['pax', 'txp']:
        var_name = f"{base}_{kind}"
        
        if var_name in decompositions:
            decomp = decompositions[var_name]
            label = f"{base.upper()} {labels_map[kind]}"
            
            # Plot Seasonality
            ax_season_color.plot(decomp.seasonal, label=label, color=c, 
                                    linestyle=styles[kind], alpha=0.8)
            
            # Plot Trend
            ax_trend_color.plot(decomp.trend, label=label, color=c, 
                                linestyle=styles[kind], linewidth=2, alpha=0.8)

            # Plot Raw Data
            ax_base_raw.plot(decomp.observed, label=label, color=c, 
                                linestyle=styles[kind], alpha=0.8)

    # save base-specific raw data plot
    ax_base_raw.set_title(f'Raw Data Comparison for {base.upper()}: Output (Solid) vs Input (Dashed)')
    ax_base_raw.set_xlabel('Sample')
    ax_base_raw.set_ylabel('Value')
    ax_base_raw.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_base_raw.tight_layout()
    fig_base_raw.savefig(f'raw_data_{base}.png')
    plt.close(fig_base_raw)

# Save Seasonality
ax_season_color.set_title('Seasonality Comparison: Output (Solid) vs Input (Dashed)')
ax_season_color.set_xlabel('Sample')
ax_season_color.set_ylabel('Value')
ax_season_color.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig_season_color.tight_layout()
fig_season_color.savefig('combined_seasonality.png')
plt.close(fig_season_color)

# Save Trend
ax_trend_color.set_title('Trend Comparison: Output (Solid) vs Input (Dashed)')
ax_trend_color.set_xlabel('Sample')
ax_trend_color.set_ylabel('Value')
ax_trend_color.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig_trend_color.tight_layout()
fig_trend_color.savefig('combined_trend.png')
plt.close(fig_trend_color)

print("Done.")