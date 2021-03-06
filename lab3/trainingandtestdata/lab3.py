# -*- coding: utf-8 -*-
"""lab3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Kr4CjmDdws99IYx9iJrE4ny964vmEw3H
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

colnames = ['label', 'id', 'date', 'query','user','text']
df_train = pd.read_csv('training.1600000.processed.noemoticon.csv',
                      header=None, names=colnames, encoding='windows-1252')
df_test = pd.read_csv('testdata.manual.2009.06.14.csv',
                      header=None, names=colnames, encoding='windows-1252')
df_train.shape
df_train.head()
df_train['label'].value_counts()

bowVect = CountVectorizer()
trainBow = bowVect.fit_transform(df_train.text)
trainBow.shape

testBow = bowVect.transform(df_test[df_test.label!=2].text)
mnbClf = MultinomialNB().fit(trainBow, df_train['label'])
predicted = mnbClf.predict(testBow)

y_test = df_test[df_test.label != 2]

print(metrics.classification_report(y_test['label'],predicted))
print(accuracy_score(y_test['label'],predicted))

x_train, x_test, y_train, y_test = train_test_split(df_train, df_train['label'], test_size=0.4)
getTweetCol = FunctionTransformer(lambda x: x['text'], validate = False)
tfVect= CountVectorizer(ngram_range=(1, 3), stop_words = "english")
mnb = MultinomialNB()
clf_tf = Pipeline([('getTweets',getTweetCol),('vect',tfVect),('clf',mnbClf)])
clf_tf.fit(x_train,y_train)
predicted = clf_tf.predict(x_test)

print(metrics.classification_report(y_test, predicted))
print(accuracy_score(y_test, predicted))

