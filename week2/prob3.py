import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/home/dinlow/fintech510/assignment/problem3.csv')
y = df['x']

def fit_ar_model(i):
    model = sm.tsa.ARIMA(y, order=(i, 0, 0))
    result = model.fit()
    return result

def fit_ma_model(i):
    model = sm.tsa.ARIMA(y, order=(0, 0, i))
    result = model.fit()
    return result

for i in range(1,4):
    AR = fit_ar_model(i)
    print("AR(",i,"):",AR.aic)
for i in range(1,4):
    MA = fit_ma_model(i)
    print("MA(",i,"):",MA.aic)

    

