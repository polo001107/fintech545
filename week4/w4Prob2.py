import pandas as pd
import numpy as np
from scipy.stats import norm, t
import statsmodels.api as sm

def return_calculate(prices, method = "DISCRETE", date_column = "Date"):
    vars = list(prices.columns)
    n_vars = len(vars)
    vars = [var for var in vars if var != date_column]

    if n_vars == len(vars):
        raise ValueError("Date column not found")
    
    n_vars -= 1

    p = prices[vars].values
    n, m = p.shape
    rt = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            rt[i,j] = p[i+1,j] / p[i,j]
            r_brown = p[i+1,j] - p[i,j]
    
    if method.upper() == "DISCRETE":
        rt -= 1.0
    elif method.upper() == "LOG":
        rt = np.log(rt)
    elif method.upper() == "BROWNIAN":
        rt = r_brown        
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\",\"BROWNIAN\")")

    dates = prices.iloc[1:n][date_column]
    out = pd.DataFrame({date_column: dates})

    # Create a list to store DataFrames to concatenate
    dfs_to_concat = [pd.DataFrame({vars[i]: rt[:, i]}) for i in range(n_vars)]

    # Concatenate all DataFrames at once
    out = pd.concat([out] + dfs_to_concat, axis=1)
    
    return out

DailyPrices = pd.read_csv('/home/dinlow/fintech510/assignment/DailyPrices.csv')

# Calculate the arithmetic returns for all prices.
returns = return_calculate(DailyPrices, method = "DISCRETE", date_column = "Date")

# Calculate VaR of META
meta_prices = DailyPrices['META']
meta_current_price = meta_prices.iloc[-1]
meta_returns = returns['META']

# Remove the mean from the series
meta_returns -= np.mean(meta_returns)
alpha = 0.05

meta_returns = meta_returns.dropna()


# 1. Normal Distribution
VaR_normal = - norm.ppf(alpha, np.mean(meta_returns), np.std(meta_returns)) * meta_current_price
print("VaR_normal: ", VaR_normal)

# 2.Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)
λ = 0.94
weighted_var = meta_returns.ewm(alpha = 1 - λ).var().iloc[-1]
#print("weighted_var: ", weighted_var)

VaR_ew = - norm.ppf(alpha, np.mean(meta_returns), np.sqrt(weighted_var)) * meta_current_price
print("VaR_ew: ", VaR_ew)

# 3. Using a MLE fitted T distribution.

params = t.fit(meta_returns)
VaR_t = - t.ppf(alpha, *params) * meta_current_price
print("VaR_t: ", VaR_t)

# 4. Using a fitted AR(1) model.
order = (1, 0, 0)
model = sm.tsa.ARIMA(meta_returns, order = order)
results = model.fit()
std_resid = results.resid.std()
VaR_ar1 = - norm.ppf(alpha, np.mean(meta_returns), std_resid) * meta_current_price
print("VaR_ar1: ", VaR_ar1)

# 5. Using a Historic Simulation.
VaR_hist = - np.percentile(meta_returns, alpha * 100) * meta_current_price
print("VaR_hist: ", VaR_hist)



