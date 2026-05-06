import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_clean_returns(file_path):
    df = pd.read_excel(file_path)
    data = df.iloc[1:].copy()
    
    return_cols = [col for col in df.columns if 'Log Returns' in col]
    col_mapping = {col: df.columns[df.columns.get_loc(col)-1] for col in return_cols}
    returns_df = data[return_cols].rename(columns=col_mapping)
    
    for col in returns_df.columns:
        returns_df[col] = pd.to_numeric(returns_df[col], errors='coerce')
    
    if 'FXY US Equity' in returns_df.columns:
        returns_df = returns_df.drop(columns=['FXY US Equity'])
        
    return returns_df.dropna()


returns_df = get_clean_returns('Monte Carlo Fix.xlsx')

annual_mean_returns = returns_df.mean() * 252
annual_cov_matrix = returns_df.cov() * 252
num_assets = len(annual_mean_returns)
rf_rate = 0.0421 

num_portfolios = 100000
min_weight = 0.005 # Minimum 0.5% allocation
max_weight = 0.20  # Maximum 20% allocation

rem_weight = 1.0 - (num_assets * min_weight) 

results = np.zeros((3, num_portfolios))
weights_record = np.zeros((num_portfolios, num_assets))

print("Running 100,000 simulations (this may take a few seconds)...")

np.random.seed(42) 

for i in range(num_portfolios):
    while True:
        
        w = np.random.random(num_assets)
        w /= np.sum(w) 
        
        w = min_weight + (w * rem_weight)
        
        
        w /= np.sum(w) 
        
        if np.max(w) <= max_weight:
            break 
            
    weights_record[i, :] = w

    p_ret = np.sum(w * annual_mean_returns)
    p_vol = np.sqrt(np.dot(w.T, np.dot(annual_cov_matrix, w)))

    results[0,i] = p_ret
    results[1,i] = p_vol
    results[2,i] = (p_ret - rf_rate) / p_vol

max_sharpe_idx = np.argmax(results[2])
optimal_return = results[0, max_sharpe_idx]
optimal_volatility = results[1, max_sharpe_idx]
optimal_sharpe = results[2, max_sharpe_idx]

weights_df = pd.DataFrame(
    weights_record[max_sharpe_idx], 
    index=annual_mean_returns.index, 
    columns=['Weight']
).sort_values(by='Weight', ascending=False)

allocations = (weights_df['Weight'] * 100).round(2)

rounding_diff = round(100.0 - allocations.sum(), 2)

if rounding_diff != 0:
    largest_idx = allocations.idxmax()
    allocations[largest_idx] = round(allocations[largest_idx] + rounding_diff, 2)

print(f"\n--- STANDARD MONTE CARLO: MAX SHARPE PORTFOLIO ---")
print(f"Constraints: Min 0.5% | Max 20.0% | Assets: {num_assets}")
print("Allocations:")
print((allocations.astype(str) + '%').to_string())

plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_volatility, optimal_return, marker='*', color='red', s=300, edgecolors='black', label='Maximum Sharpe Ratio')

plt.title(f'Bounded Monte Carlo Simulation ({num_assets} Assets | Min {min_weight*100}% | Max {max_weight*100}%)')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.legend()
plt.tight_layout()
plt.show()