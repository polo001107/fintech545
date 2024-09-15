import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import t
import matplotlib.pyplot as plt

df = pd.read_csv('/home/dinlow/fintech510/assignment/problem2.csv')

x = df['x']
y = df['y']

x = sm.add_constant(x)
modelOLS = sm.OLS(y, x).fit()
sigmaOLS = np.std(modelOLS.resid)
print("beta_OLS:", modelOLS.params)
print("sigma_OLS:" , sigmaOLS)

def likelihood(params, x, y):
    beta = params[:-1]
    sigma = params[-1]
    residuals = y - np.dot(x, beta)
    variance = sigma ** 2
    n = len(y)
    log_likelihood = -n / 2 * np.log(variance * 2 * np.pi) - np.sum(residuals ** 2) / (2 * variance)
    return -log_likelihood

initial_params = np.concatenate((np.zeros(x.shape[1]), [1.0]))
results = minimize(likelihood, initial_params, args=(x, y), method='L-BFGS-B')
beta_estimate = results.x[:-1]
sigma_estimate = results.x[-1]

print("beta_MLE:", beta_estimate)
print("sigma_MLE:", sigma_estimate)

#2
def likelihood_t(params, x, y):
    beta = params[:-2]
    df = params[-2]
    sigma = params[-1]
    sigma = y - np.dot(x, beta)
    pdf_values = t.pdf(sigma, df)  
    likelihood = np.sum(np.log(pdf_values))
    
    return -likelihood


initial_params_t = [0, 0, 1, 1] 
results_t = minimize(likelihood_t, initial_params_t, args=(x, y), method='L-BFGS-B')
beta_estimate_t = results_t.x[:-2]
df_estimate = results_t.x[-2]
sigma_estimate_t = results_t.x[-1]

print("beta_MLE_t:", beta_estimate_t)
print("sigma_MLE_t:", sigma_estimate_t)

likelihood = -results.fun  
aic = 2 * (len(initial_params) - 1) - 2 * likelihood

likelihood_t = -results_t.fun  
aic_t = 2 * (len(initial_params_t) - 1) - 2 * likelihood_t

print("AIC:", aic)
print("AIC(T-distribution):", aic_t)

#3
data = pd.read_csv('/home/dinlow/fintech510/assignment/problem2_x.csv')

X1 = data['x1'].values
X2 = data['x2'].values
X = np.column_stack([X1, X2])
mean = np.mean(X, axis=0)
covariance = np.cov(X.T)
mu1, mu2 = mean[0], mean[1]

sigma11 = covariance[0, 0]
sigma22 = covariance[1, 1]
sigma12 = covariance[0, 1]
conditional_mean = mu2 + sigma12 / sigma11 * (X1 - mu1)
conditional_variance = sigma22 - (sigma12**2) / sigma11
conditional_std = np.sqrt(conditional_variance)

z_score = 1.96
lower_bound = conditional_mean - z_score * conditional_std
upper_bound = conditional_mean + z_score * conditional_std

plt.figure(figsize=(10, 6))
plt.plot(X1, X2, 'o', label='Observed X2', alpha=0.5)  # Observed X2 values
plt.plot(X1, conditional_mean, label='Expected X2', color='r')  # Expected value
plt.fill_between(X1, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Conditional Distribution')
plt.legend()
plt.show()
