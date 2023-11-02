import numpy as np
import pandas as pd
from sklearn import datasets
iris=datasets.load_iris()
list(iris.keys())
X=iris["data"][:,3:]#petal width
y=(iris["target"]==2).astype(int)# 1 if Iris-Virginica, else 0
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
log_reg.predict([[1.7], [1.5]])
