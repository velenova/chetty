
# coding: utf-8

# In[132]:

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[133]:

def loadData(): #load data
    leaf = pd.read_csv('train.csv')
    y = pd.DataFrame(leaf, columns=['species']).as_matrix().ravel()
    x = leaf.drop(['id', 'species'], 1).as_matrix()
    print(x.shape)
    print(y.shape)
    
    #transform data
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)


# In[134]:

def buildModel(): #build model
    model = RandomForestClassifier(2)
    score = np.mean(cross_val_score(model, x, y, cv=10))
    print(score)
    model.fit(x, y)


# In[135]:

def fitModel(): #fit model
    leaf_test = pd.read_csv('test.csv')
    test_ids = pd.DataFrame(leaf_test, columns = ['id']).as_matrix().ravel()
    leaf_test = leaf_test.drop('id', 1).as_matrix()
    answer = model.predict_proba(leaf_test)
    print(answer)


# In[136]:

def createSubmission(): #create our submission
    sub = pd.DataFrame(answer, index = test_ids, columns = le.classes_)
    sub.index.names = ['id']
    sub.to_csv('output.csv')


# In[137]:

#call each method
loadData()
buildModel()
fitModel()
createSubmission()

