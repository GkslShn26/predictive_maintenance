#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:02:22 2018

@author: gkslshn
"""

import pandas as pd
#import data
data = pd.read_csv('maintenance_data.csv',sep=';')

team = data.iloc[:,5:6].values
print(team)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
team[:,0] = le.fit_transform(team[:,0])
print(team[:,0])


provider = data.iloc[:, 6:7].values
print(provider)
le2 = LabelEncoder()
provider[:,0] = le2.fit_transform(provider[:,0])

#onehot encoding 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
team=ohe.fit_transform(team).toarray()
team = team[:,0:2]

ohe2 = OneHotEncoder(categorical_features='all')
provider=ohe.fit_transform(provider).toarray()
provider = provider[:,0:3]

team_d =pd.DataFrame(data = team, index = range(1000), columns=['A','B'] )
provider_d =pd.DataFrame(data = provider, index = range(1000), columns=['Provider1','Provider2','Provider3'] )

lifetime = data.iloc[:,0:1]
broken =data.iloc[:,1:2]
sensor = data.iloc[:,2:5]


#dataframe concat
s1=pd.concat([lifetime,sensor],axis=1)
s2=pd.concat([team_d,provider_d],axis=1)
s3 =pd.concat([s1,s2],axis=1)

#data splitting for training and test
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s3,broken,test_size=0.33, random_state=0)

'''
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#datain olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
'''

def convert_dataframe(a):
    y_pred2 = []
    for i in range(0,330):
        if(a[i] >= 0.5):
            y_pred2.append(1)
        #print('buyuk 0.5---'+y_pred[1])
        else:
            y_pred2.append(0)
    return y_pred2        

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)
y_pred2 = []
for i in range(0,330):
    if(y_pred[i] >= 0.5):
        y_pred2.append(1)
    else:
        y_pred2.append(0)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred2)
print('Linear Regression\n'+str(cm))

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_train,y_train)
svr_pred=svr_reg.predict(x_test)
svr_pred2 = convert_dataframe(svr_pred)
cm = confusion_matrix(y_test,svr_pred2)
print('SVM\n'+str(cm))

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train,y_train)
r_dt_pred = r_dt.predict(x_test)
r_dt_pred2 = convert_dataframe(r_dt_pred)
cm = confusion_matrix(y_test,r_dt_pred2)
print('Decision Tree\n'+str(cm))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(x_train,y_train)
rf_reg_pred = rf_reg.predict(x_test)
rf_reg_pred2 = convert_dataframe(rf_reg_pred)
cm = confusion_matrix(y_test,rf_reg_pred2)
print('Random Forest\n'+str(cm))












