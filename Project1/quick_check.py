import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
X = np.array([[1],[2],[3],[4]])
y = np.array([2,3,5,7])
m = LinearRegression().fit(X,y)
print("coef_", m.coef_, "intercept_", m.intercept_)
print("predict(5) ->", m.predict([[5]])[0])
