from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np



colnames=['label','id','date','query','user','text']

df_train = pd.read('/Users/zenger/FileDoc/ISTD 2018 T6/ComputationalData/lab3/trainingandtestdata/training.1600000.processed.noemoticon.csv',header =None, names = colnames, encoding = 'windows 1252')
df_test = pd.read('/Users/zenger/FileDoc/ISTD 2018 T6/ComputationalData/lab3/trainingandtestdata/testdata.manual.2009.06.14.csv',header =None, names = colnames, encoding = 'windows 1252')

df_train.shape
df_train.head()
df_train['label'].value_counts()