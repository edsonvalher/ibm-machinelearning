import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

engsize = [2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7, 2.4]
cylinders = [4, 4, 4, 6, 6, 6, 6, 6, 6, 4]
fuel = [8.5, 9.6, 5.9, 11.1, 10.6, 10.0, 10.1, 11.1, 11.6, 9.2]
co2 = [196, 221, 136, 255, 244, 230, 232, 255, 267, 0]

lista = {"engsize": engsize, "cylinders": cylinders, "fuel": fuel, "co2": co2}
df = pd.DataFrame(lista)

cdf = df.head(9)[["engsize", "cylinders", "fuel", "co2"]]
print(cdf)

msk = np.random.rand(len(df.head(9))) < 0.999
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['engsize', 'cylinders', 'fuel']])
train_y = np.asanyarray(train[['co2']])
regr.fit(train_x, train_y)
y_pred = regr.predict(train_x)

print('y= ', regr.coef_, 'x= ', regr.intercept_)
print('The prediction is: ', y_pred[1])
print('Variance score: %.2f' % regr.score(train_x, train_y))
