import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace = True)
df.drop(['id'],1, inplace = True)#drop ID as this messes up the accuracy of the data

X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y,test_size = 0.3)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_in = np.array([[1,1,1,2,2,2,3,3,3],[33,90,-1,11,36,52,78,5,78]])
example_in = example_in.reshape(len(example_in),-1)#len(example_in) so it automatically reshapes

prediction = clf.predict(example_in)
print(prediction)


