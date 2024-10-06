import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy.linalg import eigvals

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

def exp_weighted_cov(returns, lambda_=0.94):
    returns = returns.values
    mean_return = np.mean(returns, axis=0)
    normalized_returns = returns - mean_return
    
    n_timesteps = normalized_returns.shape[0]
    var = np.cov(returns, rowvar=False)
    for t in range(1, n_timesteps):
        var = lambda_ * var + (1 - lambda_) * np.outer(normalized_returns[t], normalized_returns[t])
    return var


def calculate_PV(holdings, current_prices, stocks):
    PV = 0
    for stock in stocks:
        PV += holdings[stock] * current_prices[stock]
    return PV

def generate_returns_mn(returns, stocks, n, lamb=0.94):
    mean_returns = np.mean(returns[stocks], axis=0)
    cov_matrix = exp_weighted_cov(returns[stocks], lamb)
    randr_mn = np.random.multivariate_normal(mean_returns, cov_matrix, size=n)
    return randr_mn


def calculate_portfolio_var_multivariate(PV, randr_mn, holdings, current_prices, stocks, alpha):
    simulations = (1 + randr_mn) * holdings.values * current_prices[stocks].values
    VaR = PV - np.percentile(simulations.sum(axis=1), alpha * 100)
    return VaR


def calculate_portfolio_var_delta(PV, holdings, current_prices, stocks, alpha, lamb=0.94):
    delta = np.empty(len(stocks))
    for i, stock in enumerate(stocks):
        delta[i] = holdings[stock] * current_prices[stock] / PV
    
    #sigma = np.cov(returns[stocks].T)
    sigma = exp_weighted_cov(returns[stocks], lamb)
    e = eigvals(sigma)
    p_sigma = np.sqrt(delta @ sigma @ delta) 
    VaR = -PV * norm.ppf(alpha) * p_sigma
    return VaR

def calculate_portfolio_var_historical(PV, holdings, current_prices, returns, stocks, alpha):
    # Simulate prices
    sim_prices = (1 + returns[stocks].values) * current_prices[stocks].values
    vHoldings = np.array([holdings[s] for s in holdings.keys()])
    pVals = np.dot(sim_prices, vHoldings)

    # Sort the simulated portfolio values
    pVals_sorted = np.sort(pVals)

    # Calculate Historical VaR
    n = len(pVals_sorted)
    a = int(np.floor(0.05 * n))
    VaR = PV - pVals_sorted[a]

    return VaR

DailyPrices = pd.read_csv('/home/dinlow/fintech510/assignment/DailyPrices.csv')
returns = return_calculate(DailyPrices, method = "DISCRETE", date_column = "Date")
returns = returns.dropna()

current_prices = DailyPrices.iloc[-1]

stocks = list(returns.columns)[1:]

Portfolio = pd.read_csv('/home/dinlow/fintech510/assignment/portfolio.csv')

ports = ['A', 'B', 'C']
ports_dict = {port: Portfolio[Portfolio['Portfolio'] == port] for port in ports}

lamb = 0.97
alpha = 0.05
n = 10000

weighted_var_dic = {stock: returns[stock].ewm(alpha=1 - lamb).var().iloc[-1] for stock in stocks}
weighted_var = pd.Series(weighted_var_dic)
print(weighted_var)

holdings_dict = {port: ports_dict[port].groupby('Stock')['Holding'].sum() for port in ports}
holdingsA = pd.Series(holdings_dict['A']).reindex(stocks, fill_value=0)
holdingsB = pd.Series(holdings_dict['B']).reindex(stocks, fill_value=0)
holdingsC = pd.Series(holdings_dict['C']).reindex(stocks, fill_value=0)

holdingsTotal = holdingsA + holdingsB + holdingsC

# Calculate the PV
PV_A = calculate_PV(holdingsA, current_prices, stocks)
PV_B = calculate_PV(holdingsB, current_prices, stocks)
PV_C = calculate_PV(holdingsC, current_prices, stocks)
PV_Total = calculate_PV(holdingsTotal, current_prices, stocks)

# Approach: Using Delta-Normal method
VaR_A_delta = calculate_portfolio_var_delta(PV_A, holdingsA, current_prices, stocks, alpha, lamb)
VaR_B_delta = calculate_portfolio_var_delta(PV_B, holdingsB, current_prices, stocks, alpha, lamb)
VaR_C_delta = calculate_portfolio_var_delta(PV_C, holdingsC, current_prices, stocks, alpha, lamb)
VaR_Total_delta = calculate_portfolio_var_delta(PV_Total, holdingsTotal, current_prices, stocks, alpha, lamb)

# Approach: Using Historical method
VaR_A_historical = calculate_portfolio_var_historical(PV_A, holdingsA, current_prices, returns, stocks, alpha)
VaR_B_historical = calculate_portfolio_var_historical(PV_B, holdingsB, current_prices, returns, stocks, alpha)
VaR_C_historical = calculate_portfolio_var_historical(PV_C, holdingsC, current_prices, returns, stocks, alpha)
VaR_Total_historical = calculate_portfolio_var_historical(PV_Total, holdingsTotal, current_prices, returns, stocks, alpha)

print("Portfolio A - PV:", PV_A, "VaR (Delta-Normal):", VaR_A_delta, "VaR (Historical):", VaR_A_historical)  
print("Portfolio B - PV:", PV_B, "VaR (Delta-Normal):", VaR_B_delta, "VaR (Historical):", VaR_B_historical)
print("Portfolio C - PV:", PV_C, "VaR (Delta-Normal):", VaR_C_delta, "VaR (Historical):", VaR_C_historical)
print("Portfolio Total - PV:", PV_Total, "VaR (Delta-Normal):", VaR_Total_delta, "VaR (Historical):", VaR_Total_historical)