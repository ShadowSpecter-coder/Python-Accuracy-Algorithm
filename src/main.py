import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('../assets/music.csv')
input_data = data.drop(columns=['genre'])
output_data = data['genre']
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(prediction)

accuracy_score = accuracy_score(y_test, prediction)
print(accuracy_score)
