#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
import pandas as pd
import numpy as np


# In[8]:


data = pd.read_csv('tweets.160k.random.csv', encoding = 'utf-8')
data.head()


# In[16]:


labels = data['label']
x_train, x_test , y_train, y_test = train_test_split(data,labels,test_size = 0.2)
getTweetCol = FunctionTransformer(lambda x: x['text'],validate = False)#EXTACRT TWEETS
tfVect = CountVectorizer(stop_words='english', lowercase=True,ngram_range=(1,2))
mnbClf = MultinomialNB()


# In[18]:


clf_tf = Pipeline([('getTweets',getTweetCol),('vect',tfVect),('clf',mnbClf)])
clf_tf.fit(x_train,y_train)
predicted = clf_tf.predict(x_test)
print(accuracy_score(y_test,predicted))


# In[25]:


feaSelect = SelectPercentile(chi2, percentile = 5)
clf_tf = Pipeline([('getTweets',getTweetCol),('vect',tfVect),('feaSelect',feaSelect),('clf',mnbClf)])
clf_tf.fit(x_train,y_train)
predicted = clf_tf.predict(x_test)
print(accuracy_score(y_test,predicted))


# In[32]:


feaSelect = SelectKBest(chi2,k=15)
clf_tf = Pipeline([('getTweets',getTweetCol),('vect',tfVect),('feaSelect',feaSelect),('clf',mnbClf)])
clf_tf.fit(x_train,y_train)
predicted = clf_tf.predict(x_test)
print(accuracy_score(y_test,predicted))


# In[30]:


feaSelect = SelectPercentile(chi2, percentile = 10)
clf_tf = Pipeline([('getTweets',getTweetCol),('vect',tfVect),('feaSelect',feaSelect),('clf',mnbClf)])
clf_tf.fit(x_train,y_train)
predicted = clf_tf.predict(x_test)
print(accuracy_score(y_test,predicted))


# In[ ]:




