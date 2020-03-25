# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import TBDataProcessor

dataset = pd.read_csv('Twitter_Dataset.csv', encoding='latin-1')



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



#Recursive Feature Elimination
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

#Logistic regression to predict probability of being a bot
classifier = LogisticRegression()
classifier.fit(X_n,y_n)
y_npred=classifier.predict(X_n)

cm = metrics.confusion_matrix(y_n, y_npred)
acc = metrics.accuracy_score(y_n, y_npred)
print(cm,acc)


'''
#SVC Implementation to predict probability
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

Accuracy ~ 63%
'''

y_nprob = classifier.predict_proba(X_n)

y_n1=y_nprob[:,1].tolist()

#Adding the new column to dataset
dataset['name_prob'] = y_n1
dataset['name_prob']=dataset['name_prob']*100

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,19].values

#Implementing the model without name_pob column
X_selected=dataset[
    ['followers_count','friends_count','listedcount',
     'has_extended_profile','default_profile_image','default_profile',
     'statuses_count','verified','favourites_count']
    ]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_selected,y, test_size=1/6,random_state=2)

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_ntrain = scaler.fit_transform(X_train)
X_ntest = scaler.transform(X_test)

#Decision Tree implementation
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_pred,y_test)
acc = metrics.accuracy_score(y_pred,y_test)

print(cm,acc)


#Accuracy before name_prb = 83
#Accuracy after name_prob = 86

#Implementing the model with name_prob column
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

import seaborn as sns
sns.set(font_scale=1.6)



#Decision Tree Implementation
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_test)

cm = metrics.confusion_matrix(dt_y_pred,y_test)
acc = metrics.accuracy_score(dt_y_pred,y_test)
cr = metrics.classification_report(y_test, dt_y_pred)
print("DTREE",cm,acc,cr)
ax = plt.axes()
sns.heatmap(cm,ax=ax,annot=True,fmt="0.5g")
ax.set_title('Decision Tree')
plt.show()



#Random Forest Implementation
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(criterion = "entropy")
rf_classifier.fit(X_train,y_train)
rf_y_pred = rf_classifier.predict(X_test)

cm = metrics.confusion_matrix(rf_y_pred,y_test)
acc = metrics.accuracy_score(rf_y_pred,y_test)
cr = metrics.classification_report(y_test, rf_y_pred)
print("RF",cm,acc,cr)
ax=plt.axes()
sns.heatmap(cm,ax=ax,annot=True,fmt="0.5g")
ax.set_title('Random Forest')
plt.show()



#Logistic Regression Implementation
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train,y_train)
lr_y_pred = lr_classifier.predict(X_test)

cm = metrics.confusion_matrix(lr_y_pred,y_test)
acc = metrics.accuracy_score(lr_y_pred,y_test)
cr = metrics.classification_report(y_test, lr_y_pred)
print("LR",cm,acc,cr)
ax=plt.axes()
sns.heatmap(cm,ax=ax,annot=True,fmt="0.5g")
ax.set_title('Logistic Regression')
plt.show()



#SVC Implementation
from sklearn.svm import SVC
svc_classifier = SVC(probability=True)
svc_classifier.fit(X_train,y_train)
svc_y_pred = svc_classifier.predict(X_test)

cm = metrics.confusion_matrix(svc_y_pred,y_test)
acc = metrics.accuracy_score(svc_y_pred,y_test)
cr = metrics.classification_report(y_test, svc_y_pred)
print("SVC",cm,acc,cr)
ax=plt.axes()
sns.heatmap(cm,ax=ax,annot=True,fmt="0.5g")
ax.set_title('SVC')
plt.show()



#Gaussian Naive Bayes Implementation
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)
nb_y_pred = nb_classifier.predict(X_test)

cm = metrics.confusion_matrix(nb_y_pred,y_test)
acc = metrics.accuracy_score(nb_y_pred,y_test)
cr = metrics.classification_report(y_test, nb_y_pred)
print("NB",cm,acc,cr)
ax=plt.axes()
sns.heatmap(cm,ax=ax,annot=True,fmt="0.5g")
ax.set_title('Naive Bayes')
plt.show()



#ANN Implementation
import keras
from keras.models import Sequential
from keras.layers import Dense

ann_classifier = Sequential()

ann_classifier.add(Dense(input_dim=10,units=6,kernel_initializer='uniform',activation='relu'))

ann_classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

ann_classifier.add(Dense(units=3,kernel_initializer='uniform',activation='sigmoid'))

ann_classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

ann_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann_classifier.fit(X_train,y_train,batch_size=10,epochs=200)

ann_y_pred = ann_classifier.predict(X_test)
ann_y_pred = (ann_y_pred>0.5)

cm = metrics.confusion_matrix(ann_y_pred,y_test)
acc = metrics.accuracy_score(ann_y_pred,y_test)
cr = metrics.classification_report(y_test, ann_y_pred)
print("ANN:\n",cm,acc,cr)
ax=plt.axes()
sns.heatmap(cm,ax=ax,annot=True,fmt="0.5g")
ax.set_title('ANN')
plt.show()

result_table = pd.DataFrame(columns=['classifier', 'fpr','tpr','auc'])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

lr_roc_auc = roc_auc_score(y_test, lr_y_pred)
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_classifier.predict_proba(X_test)[:,1])

result_table = result_table.append({'classifier':"Logistic Regression",
                     'fpr':lr_fpr,
                     'tpr':lr_tpr,
                     'auc':lr_roc_auc},ignore_index=True)

dt_roc_auc = roc_auc_score(y_test, dt_y_pred)
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_classifier.predict_proba(X_test)[:,1])

result_table = result_table.append({'classifier':"Decision Tree",
                     'fpr':dt_fpr,
                     'tpr':dt_tpr,
                     'auc':dt_roc_auc},ignore_index=True)

rf_roc_auc = roc_auc_score(y_test, rf_y_pred)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:,1])

result_table = result_table.append({'classifier':"Random Forest",
                     'fpr':rf_fpr,
                     'tpr':rf_tpr,
                     'auc':rf_roc_auc},ignore_index=True)

nb_roc_auc = roc_auc_score(y_test, nb_y_pred)
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, nb_classifier.predict_proba(X_test)[:,1])

result_table = result_table.append({'classifier':"Naive Bayes",
                     'fpr':nb_fpr,
                     'tpr':nb_tpr,
                     'auc':nb_roc_auc},ignore_index=True)


svc_roc_auc = roc_auc_score(y_test, svc_y_pred)
svc_fpr, svc_tpr, svc_thresholds = roc_curve(y_test, svc_classifier.predict_proba(X_test)[:,1])

result_table = result_table.append({'classifier':"SVM",
                     'fpr':svc_fpr,
                     'tpr':svc_tpr,
                     'auc':svc_roc_auc},ignore_index=True)

ann_roc_auc = roc_auc_score(y_test, ann_y_pred)
ann_fpr, ann_tpr, ann_thresholds = roc_curve(y_test, ann_classifier.predict(X_test)[:,0])

result_table = result_table.append({'classifier':"ANN",
                     'fpr':ann_fpr,
                     'tpr':ann_tpr,
                     'auc':ann_roc_auc},ignore_index=True)


fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right',labels=result_table.iloc[:,0].values)

plt.show()