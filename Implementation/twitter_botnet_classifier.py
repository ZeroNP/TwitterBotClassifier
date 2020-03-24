# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import TBDataProcessor

dataset = pd.read_csv('Book1.csv', encoding='latin-1')

#Data Cleaning
#Replacing the NaN values with appropriate values
dataset[['location','description','url']] = dataset[['location','description','url']].fillna(' ')
dataset['status'] = dataset['status'].fillna('{}')
dataset['has_extended_profile']=dataset['has_extended_profile'].fillna(False)
#Drop remaining rows with NaN value
dataset=dataset.dropna()

#Data Exploration
print('Distribution: \n',dataset['bot'].value_counts())
#Distribution:
#0    1476
#1    1321
#Data seems well balanced.
sns.countplot(x='bot',data=dataset,palette='hls')
plt.show()
plt.savefig('Data_Distribution')

observations_y = dataset.groupby('bot').mean()

#Converting names data to useful information
X_scname = TBDataProcessor.filter_screen_name(dataset['screen_name'])
X_name = TBDataProcessor.filter_screen_name(dataset['name'])
X_names_matrix = TBDataProcessor.merge(X_scname,X_name)

for i in range(2797):
    X_names_matrix[i].append(dataset['bot'][i])


dataset_names = pd.DataFrame(X_names_matrix,columns=['scname_chars','scname_digits','scname_spchars','scname_hasbotsubstr',
                                                     'name_chars','name_digits','name_spchars','name_hasbotsubstr','bot'])
X_n = dataset_names.iloc[:,:-1].values
y_n = dataset_names.iloc[:,8].values

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg,20)
rfe = rfe.fit(X_n,y_n)
print(rfe.support_)
print(rfe.ranking_)
#[ True  True  True  True  True  True  True  True]
#[1 1 1 1 1 1 1 1]
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_ntrain,X_ntest,y_ntrain,y_ntest = train_test_split(X_n,y_n)


#Accuracy before scaling=0.6714285714285714

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_ntrain = scaler.fit_transform(X_ntrain)
X_ntest = scaler.transform(X_ntest)


classifier = LogisticRegression()
classifier.fit(X_n,y_n)

'''
y_npred=classifier.predict(X_ntest)

cm = metrics.confusion_matrix(y_ntest, y_npred)
acc = metrics.accuracy_score(y_ntest, y_npred)
print(cm,acc)


from sklearn.svm import SVC
classifier2 = SVC()
classifier2.fit(X_ntrain,y_ntrain)

y_npred2=classifier2.predict(X_ntest)

cm = metrics.confusion_matrix(y_ntest, y_npred2)
acc = metrics.accuracy_score(y_ntest, y_npred2)
print(cm,acc)
'''
y_npred=classifier.predict(X_n)

cm = metrics.confusion_matrix(y_n, y_npred)
acc = metrics.accuracy_score(y_n, y_npred)
print(cm,acc)

y_nprob = classifier.predict_proba(X_n)

y_n1=y_nprob[:,1].tolist()

dataset['name_prob'] = y_n1
dataset['name_prob']=dataset['name_prob']*100

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,19].values

X_selected=dataset[
    ['followers_count','friends_count','listedcount',
     'has_extended_profile','default_profile_image','default_profile',
     'statuses_count','verified','favourites_count']
    ]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_selected,y, test_size=1/6,random_state=2)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_ntrain = scaler.fit_transform(X_train)
X_ntest = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Accuracy before name_prb = 83
#Accuracy after name_prob = 86

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)

X_selected=dataset[
    ['followers_count','friends_count','listedcount',
     'has_extended_profile','default_profile_image','default_profile',
     'statuses_count','verified','favourites_count','name_prob']
    ]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_ntrain = scaler.fit_transform(X_train)
X_ntest = scaler.transform(X_test)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_selected,y, test_size=1/6,random_state=2)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = "entropy")
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)

classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(input_dim=10,units=6,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=3,kernel_initializer='uniform',activation='sigmoid'))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=200)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)
