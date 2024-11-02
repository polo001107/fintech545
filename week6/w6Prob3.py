import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import fsolve
from scipy.stats import norm
import statsmodels.api as sm
import math

df = pd.read_csv('/home/dinlow/fintech510/assignment/week6/problem3.csv')
portfolios = df['Portfolio'].unique()

S = [50, 100, 150, 200, 250]
r = 0.0425  # Risk-free rate
q = 0.0053  # Dividend rate
b = r - q
underlying = 151.03

# Define the Black-Scholes-Merton function
def gbsm(option_type, stock_price, strike, T, rf, b, ivol):
    d1 = (np.log(stock_price / strike) + (b + ivol**2 / 2) * T) / (ivol * np.sqrt(T))
    d2 = d1 - ivol * np.sqrt(T)
    if option_type == 'Call':
        return stock_price * np.exp((b - rf) * T) * norm.cdf(d1) - strike * np.exp(-rf * T) * norm.cdf(d2)
    else:
        return strike * np.exp(-rf * T) * norm.cdf(-d2) - stock_price * np.exp((b - rf) * T) * norm.cdf(-d1)

# Define the implied volatility function
def implied_volatility(option_type, stock_price, strike, T, rf, b, market_price):
    f = lambda ivol: gbsm(option_type, stock_price, strike, T, rf, b, ivol) - market_price
    return fsolve(f, 0.5)[0]  # Return the first element of the result

# Prepare the dataframe
df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])
df['TTM'] = (df['ExpirationDate'] - datetime(2023, 3, 3)).dt.days / 365
df['ImpliedVol'] = df.apply(
    lambda row: implied_volatility(row['OptionType'], underlying, row['Strike'], row['TTM'], r, b, row['CurrentPrice']) 
    if row['Type'] == 'Option' else 0, 
    axis=1
)

# Function to calculate portfolio value
def portfolio_value(portfolio, stock_price):
    portfolio_df = df[df['Portfolio'] == portfolio]
    total_value = 0
    for _, row in portfolio_df.iterrows():
        if row['Type'] == 'Stock':
            total_value += row['Holding'] * stock_price
        else:
            option_value = gbsm(row['OptionType'], stock_price, row['Strike'], row['TTM'], r, b, row['ImpliedVol'])
            total_value += row['Holding'] * option_value
    return total_value

# Plot portfolio values for different stock prices
for portfolio in portfolios:
    plt.figure(figsize=(5, 5))
    portfolio_values = [portfolio_value(portfolio, stock_price) for stock_price in S]
    plt.plot(S, portfolio_values, label=portfolio)
    plt.title(f'Portfolio Value vs Stock Price: {portfolio}')
    plt.xlabel('Stock Price')
    plt.ylabel('Portfolio Value')
    plt.grid()
    plt.legend()
    plt.show()

# Load daily price data
daily_price = pd.read_csv('/home/dinlow/fintech510/assignment/week6/DailyPrices.csv')
aapl = daily_price['AAPL']
lreturn = np.diff(np.log(aapl))  # Calculate log returns
aapl_return = pd.Series(lreturn).subtract(pd.Series(lreturn).mean())  # Center log returns

# Fit the ARIMA model
model = sm.tsa.ARIMA(aapl_return, order=(1, 0, 0))
result = model.fit()
summary = result.summary()

# Extract parameters from the summary
m = float(summary.tables[1].data[1][1])
a1 = float(summary.tables[1].data[2][1])
s = math.sqrt(float(summary.tables[1].data[3][1]))

# Simulation parameters
num_simulations = 10000
num_days = 10

# Simulate returns using ARIMA parameters
sim = pd.DataFrame(0, index=range(num_simulations), columns=[f"Day {i+1}" for i in range(num_days)])

for i in range(num_days):
    for j in range(num_simulations):
        if i == 0:
            sim.iloc[j, i] = a1 * aapl_return.iloc[-1] + s * np.random.normal() + m
        else:
            sim.iloc[j, i] = a1 * sim.iloc[j, i-1] + s * np.random.normal() + m

# Calculate simulated stock prices over 10 days
sim_p = pd.DataFrame(0, index=range(num_simulations), columns=[f"Day {i+1}" for i in range(num_days)])

for i in range(num_days):
    if i == 0:
        sim_p.iloc[:, i] = np.exp(sim.iloc[:, i]) * underlying
    else:
        sim_p.iloc[:, i] = np.exp(sim.iloc[:, i]) * sim_p.iloc[:, i-1]

# Final stock prices on Day 10
sim_10 = sim_p.iloc[:, -1]

# Prepare to calculate portfolio values based on Day 10 simulated prices
ttm = []  # Time to maturity for each option
for expiration_date in df['ExpirationDate']:
    delta = (expiration_date - datetime(2023, 3, 3)).days
    ttm.append(delta / 365)

# Prepare portfolio value DataFrame
port_value_df = pd.DataFrame(0, index=portfolios, columns=[f'Sim {i+1}' for i in range(num_simulations)])

# Calculate portfolio values for each simulation
for i in range(len(sim_10)):
    for portfolio in portfolios:
        value = portfolio_value(portfolio, sim_10[i])
        port_value_df.at[portfolio, f'Sim {i+1}'] = value

# Calculate Mean, VaR, and ES for each portfolio
confidence_level = 0.95
results = {}

for portfolio in portfolios:
    values = port_value_df.loc[portfolio].values
    mean_value = np.mean(values)
    var_value = np.percentile(values, (1 - confidence_level) * 100)  # VaR
    es_value = values[values <= var_value].mean()  # Expected Shortfall
    results[portfolio] = {
        'Mean': mean_value,
        'VaR': var_value,
        'ES': es_value
    }

# Display results
for portfolio, metrics in results.items():
    print(f"Portfolio: {portfolio}")
    print(f"  Mean: {metrics['Mean']:.2f}")
    print(f"  VaR (95%): {metrics['VaR']:.2f}")
    print(f"  Expected Shortfall (ES): {metrics['ES']:.2f}\n")