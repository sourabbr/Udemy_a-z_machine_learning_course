import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#importing data from xlsx file
dataset=pd.read_excel('complete.xlsx',sheetname='karnataka')
X=dataset.iloc[:,0].values.reshape(11,1)
y=dataset.iloc[:,1].values.reshape(11,1)

#encoding categorical data
from sklearn.preprocessing import LavelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#splitting the dataset into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#feature scaling with train and test
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
y_train=sc_y.fit_transform(y_train)
