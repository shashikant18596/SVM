import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
lr = load_iris()
df = pd.DataFrame(lr.data,columns= lr.feature_names)
df['target'] = lr.target
df["flower_names"] = df.target.apply(lambda x: lr.target_names[x])
x = df[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]].values
y = df.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
model = SVC()
model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,model.predict(x_test))
plt.xlabel("SEPAL LENGTH")
plt.ylabel("SEPAL WIDTH")
plt.title("Scatter Plot Between Sepal Length V/S Sepal Width")
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],marker = "+",color = "red")
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'],marker = "+",color = "blue")
