#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:56:06 2019

@author: mennakamel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from random import randint

# read the files "Users_data_out.xlsx" and "pics_data.xlsx" that contains data of keywords and classes concatenated
worker = pd.read_excel('Users_data_out.xlsx', index_col=None)
task=pd.read_excel('pics_data.xlsx', index_col=None)
workerid=[]
taskid=[]
taskinfo=[]
workerinfo=[]
accept=[]

import os

from gensim.models import KeyedVectors #Word2Vec
if not os.path.exists('GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
    
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

worker = pd.read_excel('Users_data_out.xlsx', index_col=None)
task=pd.read_excel('pics_data.xlsx', index_col=None)
# start from column 2 for both data of users and pics where Words are located
task_word= task.iloc[:,2].values
worker_word= worker.iloc[:,2].values
sim=np.zeros((len(task),len(worker)))


for i in range(len(task)):
    for j in range(len(worker)):
         sim[i,j]= model.wmdistance(task_word[i], worker_word[j])

for i in range(len(task)):
    #ranomly take 25 users as accepted for pic history and other 25 as rejected for pics
    noofworkers_ac=randint(1,20)
    noofworkers_rj=randint(1,20)
    print(i)
    similar_max=sim[i,:]
    similar_min=sim[i,:]
    # Put accept for smallest pics and users distance
    for j in range(noofworkers_ac):
        smallest=min(similar_min)
        arr = np.array(similar_min)
            # get index of the smallest value
        idx=np.where(arr == smallest)
        menna=idx[0]
        idx=menna[0]
        similar_min[idx]=9999999
        workerid.append(idx)    
        taskid.append(i)    
        taskinfo.append(task.iloc[i,2:].values)
        workerinfo.append(worker.iloc[idx,2:].values)
        accept.append(1)
        
          # Put reject for biggest pics and users distance
    for j in range(noofworkers_rj):
        largest=max(similar_max)
        arr = np.array(similar_max)
            # get index of the smallest value
        idx=np.where(arr == largest)
        menna=idx[0]
        idx=menna[0]
        similar_max[idx]=0
        workerid.append(idx)    
        taskid.append(i)    
        taskinfo.append(task.iloc[i,2:].values)
        workerinfo.append(worker.iloc[idx,2:].values)
        accept.append(0)
        
df1 = pd.DataFrame(taskid, columns = ['Tasks ID'])
df2 = pd.DataFrame(workerid,columns=['worker Id'])
df3 = pd.DataFrame(taskinfo)
df4 = pd.DataFrame(workerinfo)
df5=  pd.DataFrame(accept,columns=['A/R'])
data=pd.concat([df1,df2,df3,df4,df5], axis=1)
data=data.dropna()
data.to_excel("Pic User Sim wordvector.xlsx")  
# the word2 vec generated history is stored in "Pic User Sim wordvector.xlsx" 

#read for run instant
data = pd.read_excel('Pic User Sim wordvector.xlsx', index_col=None)

df1 = pd.DataFrame(data.iloc [:,1].values)
df2 = pd.DataFrame(data.iloc [:,2].values)
df5=  pd.DataFrame(data.iloc [:,5].values)


#Start BoW for the generated pictures
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 300)
XB = cv.fit_transform(data.iloc[:,3].values).toarray()
    

df6=pd.DataFrame(XB)

#Seperate bag of words for the users
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 300)
XBB = cv.fit_transform(data.iloc[:,4].values).toarray()



df7=pd.DataFrame(XBB)
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 400)
#XBB = cv.fit_transform(data.iloc[:,6].values).toarray()



#df8=pd.DataFrame(XBB)
#data2=pd.concat([df1, df2,df6,df7,df8,df5], axis=1)
data2=pd.concat([df1, df2,df6,df7,df5], axis=1)
data2=data2.dropna()
data2.to_excel("Task Worker Sim wordvector BoW.xlsx")  

#read for run instant
data2 = pd.read_excel('Task Worker Sim wordvector BoW.xlsx', index_col=None)

#classification Process 

# Remove A/R column from data2 in X training data
X = data2.iloc[:, 3:502].values

# consider the A/R column for data2 in X training data 
#X = data2.iloc[:, 2:].values
# y change with BoW - TF size check data2 DataFrane size 
y = data2.iloc[:, 502].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#scaling the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#this is to make normalization
from sklearn import preprocessing
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#Computing the score 10 consecutive times with different splits each time
scores = cross_val_score(classifier, X, y, cv=10)
scores_mean= scores.mean()

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


#Fitting using Neural Network to the training set
from sklearn.neural_network import MLPClassifier

#68%
classifier = MLPClassifier(activation='logistic', alpha=1e-4, solver='adam')

#lbfgs is better with small dataset 83% -- (200 -- 90.3) -- (300 -- 89.9) -- (400 -- 91.3)
classifier = MLPClassifier(hidden_layer_sizes=(200, ), activation='logistic', alpha=1e-4, solver='lbfgs')

#(92% default) -- (300 - 96%) -- (200 - 95%) -- (400 - 96%))
classifier = MLPClassifier(hidden_layer_sizes=(400, ), activation='identity', alpha=1e-4, solver='lbfgs')

#(default -- 87) -- (300 -- 89.6) 
classifier = MLPClassifier(hidden_layer_sizes=(400, ), activation='relu', alpha=1e-4, solver='lbfgs')

#(default -- 82%) -- (300 -- 90.6%) -- (400 -- 92.3%)
classifier = MLPClassifier(hidden_layer_sizes=(400, ), activation='tanh', alpha=1e-4, solver='lbfgs')


from sklearn.neural_network import BernoulliRBM

classifier = BernoulliRBM(n_components=128, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
scores = cross_val_score(classifier, X, y, cv=10)


#cross validation

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#calculate accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)

#calculate percision and recall
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))

#Plot percision and recall
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

precision, recall, _ = precision_recall_curve(y_test, y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Neural Network (tanh-lbfgs) Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='macro')
f1_score(y_test, y_pred, average='micro')
f1_score(y_test, y_pred, average='weighted')

