import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

np.random.seed(42)
hist_port_mean_daily = 0.000835  
hist_port_std_daily = 0.0061     

initial_investment = 50000000.0
days = 252 
simulations = 99999   



scenarios = {
    "Bear Case (Recession)": {
        "mean": hist_port_mean_daily - (0.12 / days),  # 12% annualized penalty
        "std": hist_port_std_daily * 1.40,             # 40% volatility increase
        "color": "#D62728"                             
    },
    "Base Case (Historical)": {
        "mean": hist_port_mean_daily,                  
        "std": hist_port_std_daily,
        "color": "#7F7F7F"                             
    },
    "Bull Case (Optimistic)": {
        "mean": hist_port_mean_daily + (0.06 / days),  # 6% annualized bonus
        "std": hist_port_std_daily * 0.88,             # 12% volatility decrease
        "color": "#2CA02C"                             
    }
}


fig, ax = plt.subplots(figsize=(14, 8))
percentiles_to_calc = [5, 25, 50, 75, 95]

output_text = []
output_text.append(f"--- STARTING PORTFOLIO VALUE: ${initial_investment:,.2f} ---\n")

for name, params in scenarios.items():
    final_values = np.zeros(simulations)
    
    for i in range(simulations):
        sim_returns = np.random.normal(params["mean"], params["std"], days)
        price_path = initial_investment * np.cumprod(1 + sim_returns)
        final_values[i] = price_path[-1]
    
    pct_values = np.percentile(final_values, percentiles_to_calc)
    
    output_text.append(f"========== {name} ==========")
    for p, val in zip(percentiles_to_calc, pct_values):
        gain_loss = val - initial_investment
        status = "Profit" if gain_loss > 0 else "Loss"
        output_text.append(f"{p:02d}th Percentile: ${val:>15,.2f}  |  {status}: ${abs(gain_loss):>14,.2f}")
    
    var_95 = initial_investment - pct_values[0] 
    if var_95 > 0:
        output_text.append(f">> 95% Confidence VaR (Max Loss): ${var_95:,.2f}\n")
    else:
        output_text.append(f">> 95% Confidence VaR: No Loss (Minimum Profit of ${abs(var_95):,.2f})\n")
    
    ax.hist(final_values, bins=60, alpha=0.45, color=params["color"], label=f"{name}", edgecolor='white', linewidth=0.5)
    
    ax.axvline(pct_values[0], color=params["color"], linestyle='--', linewidth=2.5, alpha=0.9) 
    ax.axvline(pct_values[2], color=params["color"], linestyle='-', linewidth=2.5, alpha=0.9)  
    ax.axvline(pct_values[4], color=params["color"], linestyle=':', linewidth=2.5, alpha=0.9)  


def millions_formatter(x, pos):
    return f'${x/1e6:,.0f}M'
ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

ax.axvline(initial_investment, color='black', linestyle='-', linewidth=3, label="Initial Investment ($50M)")

ax.set_title('1-Year Monte Carlo Scenario Analysis (Starting: $50M)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Ending Portfolio Value (USD)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Frequency (Number of Scenarios)', fontsize=12, fontweight='bold', labelpad=10)

handles, labels = ax.get_legend_handles_labels()
legend1 = ax.legend(handles, labels, loc='upper left', fontsize=11, frameon=True, shadow=True)

line_5th = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.5, label='5th Percentile (Stress)')
line_50th = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2.5, label='50th Percentile (Median)')
line_95th = mlines.Line2D([], [], color='black', linestyle=':', linewidth=2.5, label='95th Percentile (Optimistic)')

legend2 = ax.legend(handles=[line_95th, line_50th, line_5th], loc='upper right', title="Percentile Markers", fontsize=10, shadow=True)
legend2.get_title().set_fontweight('bold')
ax.add_artist(legend1) 

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("\n".join(output_text))

plt.show()