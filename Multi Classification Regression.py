import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import seaborn as sns
train = pd.read_csv(r"C:\Users\YMALLESH\Desktop\Machine Learning\Logistic Regression\fashion-mnist_train.csv")
test = pd.read_csv(r"C:\Users\YMALLESH\Desktop\Machine Learning\Logistic Regression\fashion-mnist_train.csv")
# print(train.head())
x_train = train.drop(columns="label")
y_train = train["label"]
x_test = test.drop(columns="label")
y_test = test["label"]
print(x_train,x_test,y_train,y_test)
x_train_list = x_train.values.tolist()
print(x_train_list)
plt.imshow(np.reshape(x_train_list[1],(28,28,1)))
plt.show()
model = LogisticRegression()
model.fit(x_train,y_train)
y_test_pred = model.predict(x_test)
print(y_test_pred[0])
x_test_list = x_test.values.tolist()
plt.imshow(np.reshape(x_test_list[1],(28,28,1)))
plt.show()
r2score = r2_score(y_test_pred,y_test)
print(r2score)
cm = confusion_matrix(y_test,y_test_pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True)
plt.show()
print(classification_report(y_test,y_test_pred))
