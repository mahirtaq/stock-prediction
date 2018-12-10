import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

df = pd.read_csv('data/Google Stock Price Train Set.csv')
y = df['Close']
X = df[['High', 'Low', 'Open']]
lm = LinearRegression()
lm.fit(X, y)
print(lm.score(X, y))

svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
svr_poly.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)
print(svr_lin.score(X, y))
print(svr_poly.score(X, y))
