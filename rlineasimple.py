import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

engsize = [2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7, 2.4]
cylinders = [4, 4, 4, 6, 6, 6, 6, 6, 6, 4]
fuel = [8.5, 9.6, 5.9, 11.1, 10.6, 10.0, 10.1, 11.1, 11.6, 9.2]
co2 = [196, 221, 136, 255, 244, 230, 232, 255, 267, 0]

lista = {"engsize": engsize, "cylinders": cylinders, "fuel": fuel, "co2": co2}

df = pd.DataFrame(lista)
# print(df.head(9).describe())  # sumariza los valores

cdf = df.head(9)[["engsize", "cylinders", "fuel", "co2"]]
print(cdf)


# comparación de estas caracteristicas con el co2 para ver cual lineal es la regresion
def compara_entre_ellos(x, y):
    plt.scatter(x, y,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Co2")
    plt.show()


# dibujar cada una de las caracteristicas
def dibujar_caracteristicas(cdf):
    viz = cdf[["engsize", "cylinders", "fuel", "co2"]]
    viz.hist()
    plt.show()


# compara_entre_ellos(cdf.engsize, cdf.co2)
# dibujar_caracteristicas(cdf)

msk = np.random.rand(len(df.head(9))) < 0.999
train = cdf[msk]
test = cdf[~msk]

#plt.scatter(train.engsize, train.co2,  color='blue')
#plt.xlabel("Engine size")
# plt.ylabel("Co2")
# plt.show()


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['engsize']])
train_y = np.asanyarray(train[['co2']])
regr.fit(train_x, train_y)
y_pred = regr.predict(train_x)
# The coefficients
#print('Coefficients: ', regr.coef_)
#print('Intercept: ', regr.intercept_)
print('y= ', regr.coef_, 'x= ', regr.intercept_)
print('The prediction is: ', y_pred[1])
print('Variance score: %.2f' % regr.score(train_x, train_y))


plt.scatter(train.engsize, train.co2,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Co2")
plt.show()


plt.scatter(cdf.engsize, cdf.co2,  color='blue')
plt.plot(cdf.engsize, cdf.co2, 'ro', label="linear")

plt.xlabel("Engine size")
plt.ylabel("Co2")
plt.show()

# ESTE CODIGO obtiene la predicción de c02 para la tupla 10 teniendo en cuenta el tamaño del motor
