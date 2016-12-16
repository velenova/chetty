
# coding: utf-8

# In[238]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np


# In[239]:

#read in the data
train_data = pd.read_csv('train.csv')


# In[240]:

#split the data into inputs and outputs
y = pd.DataFrame(train_data, columns=['label']).as_matrix()
x = train_data.drop('label', 1).as_matrix()


# In[241]:

#convert output to categorical data
#y = to_categorical(y, 10)


# In[242]:

#split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)


# In[243]:

#create the model
model = Sequential()


# In[244]:

#build the layers of the model
model.add(Dense(output_dim=200, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dropout(0.2))

model.add(Dense(output_dim=10))
model.add(Activation("softmax"))


# In[245]:

#compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])


# In[246]:

model.fit(x_train, y_train, nb_epoch=10)


# In[247]:

loss_and_metrics = model.evaluate(x_test, y_test)
print(loss_and_metrics)


# In[248]:

import numpy as np
df = pd.read_csv('test.csv').as_matrix()
print(df.shape)
y = model.predict(df)

list_val = range(1, df.shape[0]+1)
array_val = np.asarray(list_val)
col1 = array_val.reshape(len(list_val), 1)

col_x = np.argmax(y, axis=1)
col2 = col_x.reshape(col_x.shape[0], 1)

x = np.concatenate((col1, col2), axis=1)
col_headers = ['ImageId', 'Label']

new_df = pd.DataFrame(x, columns=col_headers)
new_df.set_index('ImageId', inplace=True)
new_df.to_csv('output.csv')


# In[ ]:



