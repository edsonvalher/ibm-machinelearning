# En este ejercicio, utilizarás Máquinas de Soporte Vectorial (SVM) (Support Vector Machines)
# para construir y entrenar un modelo utilizando registros de células humanas para luego clasificarlas en benignas o malignas.
# SVM trabaja enlazando datos con un dimensión espacial de forma tal que los puntos de datos sean categorizados,
# inclusive cuando los datos no son linealmente separables. Un separador entre categorías primero se encuentra, luego los datos se transorman para que el separador pueda dibujarse como un hiperplane. Luego, rasgos de nuevos datos se pueden utilizar para predecir el grupo al cual un nuevo registro debería pertencer.


from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# evaluación libs
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# f1_score lib
from sklearn.metrics import f1_score

# jaccard lib
from sklearn.metrics import jaccard_score

# cargamos los datos
cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()

# miremos la distribución  de las clases basadas en el grosor y uniformidad del tamaño de la celula
ax = cell_df[cell_df['Class'] == 4][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
cell_df[cell_df['Class'] == 2][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)

# este muestra la grafica de clases
# plt.show()

# revisamos el tipo de datos de las columnas
print(cell_df.dtypes)
# convertir en entero la columna BareNuc que es object a int
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                      'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

# Queremos que el modelo prediga el valor de la columna Class (si es benigno (=2), si es maligno (=4)).
# Como este campo puede tener uno de dos valores posibles, necesitaremos cambiar su nivel de medición para reflejar eso.
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print(y[0:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)


# El algoritmo SVM ofrece elegir funciones para realizar su procesamiento. Básicamente, mapear los datos en un espacio dimensional más alto se llama kernelling.
# La función matemática utilizada para la transformación se conoce como la función kernel, y puede ser de distintos tipos, a ser:

# 1.Lineal
# 2.Polimonial
# 3.Función de base Radial (RBF)
# 4.Sigmoide
# Cada una de estas funciones tiene sus características, pros y contras y su ecuación, pero como no hay una forma sencilla de saber la función que mejor funcionaría,
# elegimos utilizar diferentes funciones y comparar los resultados. Utilicemos la función por omisión, RBF (Función Basada en Radio) para este lab.
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Luego que el modelo cuadró, se puede utilizar para predecir nuevos valores:
yhat = clf.predict(X_test)
print("la predicción es ", yhat[0:5])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y marca la matriz de confusión.
    Se puede aplicar Normalización seteando la variable `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Matriz de confusión, sin normalización')

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
    plt.ylabel('Etiqueta True')
    plt.xlabel('Etiqueta predecida')


# Computar la matriz de confusión
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[
                      'Benign(2)', 'Malignant(4)'], normalize=False,  title='Matriz de confusión')

# Computar la matriz de confusión
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Se puede utilizar facilmente el f1_score de la librería sklearn:
f1 = f1_score(y_test, yhat, average='weighted')
print("the f1 score is ", f1)


# Intentemos el índice jaccard para dar precisión:

#jaccard = jaccard_score(y_test, yhat)
