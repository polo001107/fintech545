import numpy as np

p, mu, sigma = 100, 0, 0.1
np.random.seed(1000)
r_t = np.random.normal(mu, sigma, 1000)
# Simulate Classical Brownian Motion
p_t_classic = p + r_t
print("Mean of Classical Brownian Motion: ", np.mean(p_t_classic))
print("Standard deviation of Classical Brownian Motion: ", np.std(p_t_classic))

# Simulate Arithmetic Return System
p_t_arithmetic = p*(1+r_t)
print("Mean of Arithmetic Return System: ", np.mean(p_t_arithmetic))
print("Standard deviation of Arithmetic Return System: ", np.std(p_t_arithmetic))

# Simulate Log Return or Geometric Brownian Motion
p_t_log = p*np.exp(r_t)

print("Mean of Log Return or Geometric Brownian Motion: ", np.mean(p_t_log))
print("Standard deviation of Log Return or Geometric Brownian Motion: ", np.std(p_t_log))
print("calculated mean:", p * np.exp(sigma**2/2))
print("calculated std:", p * np.exp(sigma**2/2) * np.sqrt(np.exp(sigma**2) - 1))