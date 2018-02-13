import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col ='Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))#increasing this yields a higher accuracy
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)#20%percent of data used for training and testing 

clf= LinearRegression()
#clf=svm.SVR(change kernel if you want...kernel = 'poly)This is a different algorithm but yields a much worse accuracy result in this case. SVR means support vector regression
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

#print(accuracy)#accuracy will be the squared error

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy,forecast_out)# NOT a prediction of stock price one month in the figure. Because we are feeding "future" prices as label into algorithm, the machine learning algorithm pretty much found out that we just shift the prices 0.01*len into the past
df['Forecast'] = np.nan 

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:#this nasty for loop is for dates on the axis as what good is a graph without labeled axis!! 
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day 
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()