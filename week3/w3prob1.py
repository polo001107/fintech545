import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('/home/dinlow/fintech510/assignment/DailyReturn.csv')

lambdas = np.linspace(0.1, 0.99, 10)

def exp_weighted_covariance_matrix(data, lamb):
    returns = data.values
    n, m = returns.shape
    weighted_cov = np.zeros((m, m))

    mean_returns = np.mean(returns, axis=0)
    weights = np.array([lamb**(n - i) for i in range(1, n + 1)])
    weights /= weights.sum() 
    for i in range(n):
        diff = returns[i] - mean_returns
        weighted_cov += weights[i] * np.outer(diff, diff)
    
    return weighted_cov

def plot_pca_variance(cov_matrix, lamb):
    pca = PCA()
    pca.fit(cov_matrix)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(explained_variance, label=f'lamb={lamb:.2f}')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained by Each Eigenvalue')
    plt.legend()
plt.figure(figsize=(10, 6))

for lamb in lambdas:
    cov_matrix = exp_weighted_covariance_matrix(data, lamb)
    plot_pca_variance(cov_matrix, lamb)

plt.show()