import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import pydot


my_data = pd.read_csv("info.csv", header=0, delimiter=",")
my_data[0:5]

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol']].values

le_Age = preprocessing.LabelEncoder()
le_Age.fit(['Young', 'Middle-age', 'Senior'])
X[:, 0] = le_Age.transform(X[:, 0])


le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['Low', 'Normal', 'High'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['Normal', 'High'])
X[:, 3] = le_Chol.transform(X[:, 3])

y = my_data["Drug"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)
print("Precisión de los Arboles de Decisión: ",
      metrics.accuracy_score(y_testset, predTree))

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:4]
targetNames = my_data["Drug"].unique().tolist()
# ValueError: Length of feature_names, 5 does not match number of features, 4
out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(
    y_trainset), filled=True,  special_characters=True, rotate=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
