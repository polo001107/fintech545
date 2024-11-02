import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from datetime import datetime

# Load the options data
data = pd.read_csv('/home/dinlow/fintech510/assignment/week6/AAPL_Options.csv')

# Set constants
current_price = 170.15
current_date = datetime(2023, 10, 30)
risk_free_rate = 0.0525
dividend_rate = 0.0057

# Calculate the time to maturity for each option
def calculate_ttm(expiration_date):
    exp_date = datetime.strptime(expiration_date, '%m/%d/%Y')
    return (exp_date - current_date).days / 365.0

# Black-Scholes formula for option pricing
def black_scholes_price(underlying, strike, ttm, rf, dividend, ivol, option_type="call"):
    d1 = (np.log(underlying / strike) + (rf - dividend + 0.5 * ivol ** 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)
    
    if option_type == "call":
        price = underlying * np.exp(-dividend * ttm) * norm.cdf(d1) - strike * np.exp(-rf * ttm) * norm.cdf(d2)
    elif option_type == "put":
        price = strike * np.exp(-rf * ttm) * norm.cdf(-d2) - underlying * np.exp(-dividend * ttm) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type")
    
    return price

# Function to calculate implied volatility
def implied_volatility(market_price, underlying, strike, ttm, rf, dividend, option_type):
    # Objective function to match the market price with the calculated option price
    def objective(ivol):
        return black_scholes_price(underlying, strike, ttm, rf, dividend, ivol, option_type) - market_price
    
    try:
        # Use brentq root-finding method within a reasonable range for volatility
        return brentq(objective, 1e-6, 5)
    except ValueError:
        # Return NaN if no solution is found
        return np.nan

# Calculate implied volatility for each option
data['TTM'] = data['Expiration'].apply(calculate_ttm)
data['Implied Volatility'] = data.apply(lambda row: implied_volatility(
    row['Last Price'], current_price, row['Strike'], row['TTM'], risk_free_rate, dividend_rate, row['Type'].lower()
), axis=1)

# Display the calculated implied volatilities
print(data[['Type', 'Strike', 'Last Price', 'Implied Volatility']])

# Split data into calls and puts
calls = data[data['Type'] == 'Call']
puts = data[data['Type'] == 'Put']

# Plot implied volatility vs. strike price
plt.figure(figsize=(12, 6))
plt.plot(calls['Strike'], calls['Implied Volatility'], label='Calls', color='blue', marker='o')
plt.plot(puts['Strike'], puts['Implied Volatility'], label='Puts', color='red', marker='x')
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.legend()
plt.title("Implied Volatility vs. Strike Price for AAPL Options")
plt.grid()
plt.show()