#Neural network accuracy : 0.8381466201205966

#Logistic regression accuracy : 0.8378645227264713

import pandas as pd
import glob
from sklearn.utils import shuffle
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow import keras

all_files = glob.glob('/home/ishi/anaconda3/dataset' + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))
#print(df.isna().sum())
df.fillna(method='bfill',inplace=True)
df.dropna(inplace=True)
'''df['Location'] = pd.Categorical(df['Location']).codes
df['WindGustDir'] = pd.Categorical(df['WindGustDir']).codes
df['WindDir9am'] = pd.Categorical(df['WindDir9am']).codes
df['WindDir3pm'] = pd.Categorical(df['WindDir3pm']).codes
df['RainToday'] = pd.Categorical(df['RainToday']).codes
df['RainTomorrow'] = pd.Categorical(df['RainTomorrow']).codes
X = array[:,1:23]
Y = array[:,23]
# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
#print(fit.scores_)
for n,s in zip(list(df)[1:24],test.scores_):
 print ('F-score: %3.2ft for feature %s ' % (s,n))
features = fit.transform(X)
#print(features[0:5,:])'''

df = df.drop(['Date','RISK_MM','Evaporation','WindDir9am', 'WindDir3pm','Temp9am',], axis = 1)
df = shuffle(df)
df.reset_index(inplace=True, drop=True) 

df['Location'] = pd.Categorical(df['Location']).codes
df['RainToday'] = pd.Categorical(df['RainToday']).codes
df['Location'] = pd.Categorical(df['Location']).codes
df['WindGustDir'] = pd.Categorical(df['WindGustDir']).codes
df['RainTomorrow'] = pd.Categorical(df['RainTomorrow']).codes
x = df.drop(columns=['RainTomorrow'])
y = df['RainTomorrow']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

#0.3029800613593515
clf=LinearRegression()
clf.fit(xTrain, yTrain)
confidence = clf.score(xTest, yTest)
print(confidence)

#0.8378645227264713 with location
#0.8342677809513734 without
clf=LogisticRegression()
clf.fit(xTrain, yTrain)
confidence = clf.score(xTest, yTest)
print(confidence)

#0.78133925737861
clf = DecisionTreeClassifier(random_state=0)
clf.fit(xTrain,yTrain)
confidence = clf.score(xTest,yTest)
print(confidence)

'''
Testing Accuracy  ----> 0.8380055714235339
Testing Accuracy  ----> 0.8381466201205966
NO location
0.837582425332346
 '''
model = keras.models.Sequential()
model.add(keras.layers.Dense(units = 30,kernel_initializer='uniform',activation = 'relu',input_dim = 17))
model.add(keras.layers.Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))
model.add(keras.layers.Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))
model.add(keras.layers.Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit(xTrain,yTrain,epochs = 5,batch_size=64)

y_pred = model.predict_classes(xTest)
y_train_pred = model.predict_classes(xTrain)
print('Training Accuracy ---->',accuracy_score(yTrain,y_train_pred))
print('Testing Accuracy  ---->',accuracy_score(yTest,y_pred))

#0.8353609083536091
clf=svm.SVC(kernel='linear')
clf.fit(xTrain, yTrain)
confidence = clf.score(xTest, yTest)
print(confidence)

