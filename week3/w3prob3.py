import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA
import time


data = pd.read_csv('/home/dinlow/fintech510/assignment/DailyReturn.csv')
returns = data.values

pearson_corr = np.corrcoef(returns, rowvar=False)
pearson_var = np.var(returns, axis=0)

def exponentially_weighted_covariance(data, lamb=0.97):
    weights = np.array([lamb**i for i in range(len(data))])
    weights = weights[::-1] / weights.sum()  
    weighted_returns = data * weights[:, np.newaxis] 
    return np.cov(weighted_returns, rowvar=False)

ew_cov = exponentially_weighted_covariance(returns)

cov1 = pearson_corr * np.outer(np.sqrt(pearson_var), np.sqrt(pearson_var))
ew_var = np.diag(ew_cov)
cov2 = pearson_corr * np.outer(np.sqrt(ew_var), np.sqrt(ew_var))
cov3 = ew_cov
cov4 = ew_cov * np.outer(np.sqrt(pearson_var), np.sqrt(pearson_var))

cov_matrices = [cov1, cov2, cov3, cov4]

def simulate_data(cov_matrix, n_draws=25000, method='direct', explained_variance=1.0):

    if method == 'direct':
        mean = np.zeros(cov_matrix.shape[0]) 
        return np.random.multivariate_normal(mean, cov_matrix, size=n_draws)
    elif method == 'pca':
        pca = PCA()
        pca.fit(np.linalg.cholesky(cov_matrix))  # Fit on the Cholesky decomposition
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(explained_var, explained_variance) + 1 
        components = pca.components_[:n_components]  # Select principal components
        projected_data = np.random.randn(n_draws, n_components) @ components
        return projected_data

def compare_frobenius_norms(original_cov, simulated_data):

    simulated_cov = np.cov(simulated_data, rowvar=False)
    return norm(original_cov - simulated_cov, 'fro')

n_draws = 25000
methods = ['direct', 'pca_100', 'pca_75', 'pca_50']
explained_variances = [1.0, 0.75, 0.50, 0.50] 

for i, cov_matrix in enumerate(cov_matrices):
    print(f"Covariance Matrix {i + 1}")
    for method, explained_var in zip(methods, explained_variances):
        start_time = time.time()
        
        if method == 'direct':
            simulated_data = simulate_data(cov_matrix, n_draws, method='direct')
        else:
            simulated_data = simulate_data(cov_matrix, n_draws, method='pca', explained_variance=explained_var)
        
        frobenius_norm = compare_frobenius_norms(cov_matrix, simulated_data)
        runtime = time.time() - start_time
        print(f"Method: {method}, Frobenius Norm: {frobenius_norm:.5f}, Runtime: {runtime:.5f} seconds")
   