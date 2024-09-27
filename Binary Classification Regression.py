import  pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv(r"C:\Users\YMALLESH\Desktop\Machine Learning\Logistic Regression\Heart.csv")
data = data.drop(columns="Unnamed: 0")
le = LabelEncoder()
data["ChestPain"] = le.fit_transform(data["ChestPain"])
data["Thal"] = le.fit_transform(data["Thal"])
data["AHD"] = le.fit_transform(data["AHD"])
data = data.dropna()
# print(data.isna().sum())
x = data.drop(columns="AHD")
y = data["AHD"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=7)
# print(x_train)

from sklearn.preprocessing import StandardScaler

# Scale your input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)  # Scale your training data
X_test_scaled = scaler.transform(x_test)
model = LogisticRegression(random_state=0)
model.fit(x_train,y_train)
y_train_Pred = model.predict(X_train_scaled)
print(y_train_Pred)
print(model.score(X_train_scaled,y_train))
