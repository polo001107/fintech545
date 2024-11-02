from datetime import date
from math import log, sqrt, exp, isclose
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Calculate the time to maturity
def calculate_ttm(current_date, expire_date):
    days_to_maturity = (expire_date - current_date).days
    return days_to_maturity / 365

# Define the Black-Scholes-Merton formula for option pricing
def black_scholes(underlying, strike, ttm, rf, b, ivol, option_type="call"):
    d1 = (log(underlying / strike) + (b + 0.5 * ivol ** 2) * ttm) / (ivol * sqrt(ttm))
    d2 = d1 - ivol * sqrt(ttm)

    if option_type == "call":
        return underlying * exp((b - rf) * ttm) * norm.cdf(d1) - strike * exp(-rf * ttm) * norm.cdf(d2)
    elif option_type == "put":
        return strike * exp(-rf * ttm) * norm.cdf(-d2) - underlying * exp((b - rf) * ttm) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

underlying_price = 165
strike_price = 165
risk_free_rate = 0.0525
coupon_rate = 0.0053
b = risk_free_rate - coupon_rate
implied_volatility = np.linspace(0.1, 0.8, 200)

# Calculate time to maturity
current_date = date(2023, 3, 3)
expiration_date = date(2023, 3, 17)
ttm = calculate_ttm(current_date, expiration_date)
print(f"Time to Maturity: {ttm:.4f}")

# Calculate option values
call_values = [black_scholes(underlying_price, strike_price, ttm, risk_free_rate, b, iv, "call") for iv in implied_volatility]
put_values = [black_scholes(underlying_price, strike_price, ttm, risk_free_rate, b, iv, "put") for iv in implied_volatility]

# Verify put-call parity
parity_holds = all(isclose(call + strike_price * exp(-risk_free_rate * ttm), put + underlying_price, abs_tol=0.1)
                   for call, put in zip(call_values, put_values))
print("Put-Call Parity holds:", parity_holds)

# Plot option values for call and put options with same strike
plt.figure()
plt.plot(implied_volatility, call_values, label="Call")
plt.plot(implied_volatility, put_values, label="Put")
plt.xlabel("Implied Volatility")
plt.ylabel("Option Value")
plt.legend()
plt.title("Option Values with Same Strike")
plt.show()

# Calculate option values with different strikes for calls and puts
call_values_diff_strike = [black_scholes(underlying_price, strike_price + 20, ttm, risk_free_rate, b, iv, "call") for iv in implied_volatility]
put_values_diff_strike = [black_scholes(underlying_price, strike_price - 20, ttm, risk_free_rate, b, iv, "put") for iv in implied_volatility]

# Plot option values with different strikes
plt.figure()
plt.plot(implied_volatility, call_values_diff_strike, label="Call with Strike 185")
plt.plot(implied_volatility, put_values_diff_strike, label="Put with Strike 145")
plt.xlabel("Implied Volatility")
plt.ylabel("Option Value")
plt.legend()
plt.title("Option Values with Different Strikes")
plt.show()