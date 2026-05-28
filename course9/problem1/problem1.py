from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd

iris = load_iris()
print(iris.feature_names)
print(iris.data[:5])
print(iris.target_names)

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head())

plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'], c=iris_df['target'], cmap='viridis')
plt.xlabel('petal Length (cm)')
plt.ylabel('petal Width (cm)')
plt.title('Iris Dataset - petal Length vs petal Width')
plt.colorbar()
plt.show()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
print("============== Train/Test Split =============")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("============================================")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) 

knn_predictions = knn.predict(X_test)
print("============== KNN Predictions =============")

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("============================================")
