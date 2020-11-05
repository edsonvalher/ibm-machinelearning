import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing

# lib para normalizar
import matplotlib.pyplot as plt

# entrenar y probar los datos
from sklearn.model_selection import train_test_split

# fiting model train libs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# indice de jaccard lib
from sklearn.metrics import jaccard_score

# matriz de confusión lib
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# log loss lib
from sklearn.metrics import log_loss

churn_df = pd.read_csv("ChurnData.csv")

churn_df = churn_df[['tenure', 'age', 'address', 'income',
                     'ed', 'employ', 'equip',   'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# definimos X
X = np.asarray(churn_df[['tenure', 'age', 'address',
                         'income', 'ed', 'employ', 'equip']])

# definimos y
y = np.asarray(churn_df['churn'])

# se debe normalizar la información

X = preprocessing.StandardScaler().fit(X).transform(X)

# Entrenar/Probar el set de datos
# Ahora, dividamos nuestro set de datos en dos sets, entrenamiento y prueba:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
#print('Train set:', X_train.shape,  y_train.shape)
#print('Test set:', X_test.shape,  y_test.shape)

# Now lets fit our model with train set
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# ahora podremos predecir
yhat = LR.predict(X_test)

# predict_proba devuelve estimaciones para todas las clases. La primer columna es la probabilidad de la clase 1, P(Y=1|X),
# y la segunda columna es la probabilidad de la clase 0, P(Y=0|X):
yhat_prob = LR.predict_proba(X_test)

# indice de jaccard
jaccard = jaccard_score(y_test, yhat)

# matriz de confusión
# Otra forma de ver la precisión del clasificador es ver la matriz de confusión.


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función muestra y dibuja la matriz de confusión.
    La normalización se puede aplicar estableciendo el valor `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalización')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')


confusion_matrix(y_test, yhat, labels=[1, 0])

# Calcular la matriz de confusión
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)


# Dibujar la matriz de confusión no normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[
                      'churn=1', 'churn=0'], normalize=False,  title='Matriz de confusión')

# classification report
print(classification_report(y_test, yhat))
# log loss
# Ahora, probemos log loss para la evaluación. En regresión logística, la salida puede ser que la probabilidad de cliente churn sea sí (o su equivalente 1).
# Esta probabilidad es un valor entre 0 y 1. Log loss( pérdida logarítmica) mida el rendimiento de un clasificador donde la salida predicha es una probabilidad de valor entre 0 y 1.


print("Log loss, linear: ", log_loss(y_test, yhat_prob))

LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
