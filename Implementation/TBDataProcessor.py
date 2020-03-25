# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:45:02 2020

@author: admin
"""

import pandas as pd;

dataset = pd.read_csv('Twitter_Dataset.csv',encoding='latin-1')

X = dataset.iloc[:,[2,4,18]].values
y = dataset.iloc[:,19].values

X_scname = X[:,0]
X_desc = X[:,1]
X_name = X[:,2]

def calculate_alpha_num(str,index):
    chars = 0
    digits = 0
    special_chars = 0
    hasbotsubstr = 0
    
    if str.lower().find("bot")!=-1:
        hasbotsubstr = 1
    for i in str:
        if i == ' ':
            continue
        if i.isdigit():
            digits += 1
        elif i.isalpha():
            chars += 1
        else:
            special_chars += 1
    return [chars,digits,special_chars,hasbotsubstr]


def filter_screen_name(arr):
    col=[]
    i=0
    for r in arr:
        col.append(calculate_alpha_num(r,i))
        i+=1
    return col

X_scname = filter_screen_name(X_scname)
X_name= filter_screen_name(X_name)

 
def merge(lst1, lst2): 
    for i in range(len(lst1)):
        for j in range(len(lst2[i])):
            lst1[i].append(lst2[i][j])
    return lst1

X_f = merge(X_scname,X_name)

from sklearn.model_selection import train_test_split
X_ftrain,X_ftest,y_ftrain,y_ftest = train_test_split(X_f, y)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_ftrain,y_ftrain)

y_fpred = classifier.predict(X_ftest)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_ftest, y_fpred))


from sklearn.svm import SVC
classifier2 = SVC()
classifier2.fit(X_ftrain,y_ftrain)

y_fpred2=classifier2.predict(X_ftest)

from sklearn import metrics

cm = metrics.confusion_matrix(y_ftest, y_fpred2)
acc = metrics.accuracy_score(y_ftest, y_fpred2)
print(cm,acc)