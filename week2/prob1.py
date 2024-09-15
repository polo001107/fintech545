import pandas as pd
import numpy as np

df = pd.read_csv('/home/dinlow/fintech510/assignment/problem1.csv')
x= df.values.astype(float)
# for first question to calculate moment value using given function
n = len(x)
mean = sum(x) / n
variance = sum((x-mean)**2) / n
sample_variance = sum((x-mean)**2) / (n - 1)
skewness = sum((x-mean)**3) / (n * variance **(3/2))
kurtosis = sum((x-mean)**4) / (n * variance **2) - 3

print('The mean is: ', mean)
print('The variance is: ', sample_variance)
print('The skewness is: ', skewness)
print('The kurtosis is: ', kurtosis)

# for second question to calculate moment value using function in pandas package
S1 = pd.Series(x.flatten())
mean_pd = S1.mean()
variance_pd = S1.var()
skewness_pd = S1.skew()
kurtosis_pd = S1.kurtosis()

print('The mean is: ', mean_pd)
print('The variance is: ', variance_pd)
print('The skewness is: ', skewness_pd)
print('The kurtosis is: ', kurtosis_pd)

