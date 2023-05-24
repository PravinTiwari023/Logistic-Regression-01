import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

data = sns.load_dataset('iris')

print(data.head())

print(data['species'].unique())

print(data.isnull().sum())

data = data[data['species'] != 'setosa']
sns.pairplot(data, hue='species')

X = data.drop(['species'], axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

encoder = LabelEncoder()

encoder.fit_transform(y_train)
encoder.transform(y_test)

clf = LogisticRegression()

parameters = {'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50], 'max_iter':[100,200,300]}
clfcv = GridSearchCV(clf, parameters, scoring='accuracy', cv= 5) 

clfcv.fit(X_train, y_train)

print(clfcv.best_params_)
print(clfcv.best_score_)

y_pred = clfcv.predict(X_test)

# Accuracy score
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
