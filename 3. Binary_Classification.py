import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True)
df['data'].head()
df['target'].value_counts()
X = df['data']
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.30, random_state=0)

from sklearn.preprocessing import StandardScaler
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
X_test = ss_train.transform(X_test)

from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()

print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy =  (TP + TN) / (TP + FP + TN + FN)

print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
