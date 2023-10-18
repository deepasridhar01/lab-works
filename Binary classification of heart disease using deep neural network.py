#!/usr/bin/env python
# coding: utf-8

# Name : Deepa S
# 
# Roll No : 225229107

# In[7]:


import tensorflow as tf


# In[2]:


import pandas as pd
import numpy as np


# ##### Step-1

# In[3]:


f=pd.read_csv("heart_data.csv")


# In[4]:


f.head()


# In[5]:


f.isnull()


# In[6]:


f.info()


# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[23]:


from sklearn.model_selection import train_test_split


# ##### Step-2

# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[26]:


X.columns


# In[15]:


y=f["target"]


# In[18]:


X=f.drop("target",axis=1)


# ##### Step-3

# In[25]:


model=Sequential()


# In[28]:


model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1, activation='sigmoid'))


# In[29]:


from tensorflow.keras.optimizers import RMSprop


# ##### Step-4

# In[30]:


optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])


# ##### Step-5

# In[31]:


model.summary()


# ##### Step-6

# In[33]:


model.fit(X_train,y_train,epochs=200,batch_size=10,validation_data=(X_test,y_test))


# ##### Step-7

# In[34]:


history=model.fit(X_train,y_train,validation_split=0.2,epochs=100,batch_size=10,verbose=1)


# ##### Step-8

# In[35]:


loss,accuracy=model.evaluate(X_test,y_test)
print("Test loss:",loss)
print("Test accuracy:",accuracy)


# ##### Step-9

# In[37]:


import matplotlib.pyplot as plt


# In[40]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Validation'])
plt.show()


# In[41]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Validation'])
plt.show()


# ##### Step-10(a)

# In[42]:


model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[43]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()


# ##### Step-10(b)

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Create the model
model = Sequential()

# Add the first hidden layer
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))

# Add the second hidden layer
model.add(Dense(16, activation='relu'))

# Add the third hidden layer
model.add(Dense(8, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot the accuracy and loss charts
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




