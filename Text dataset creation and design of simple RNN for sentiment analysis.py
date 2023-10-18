#!/usr/bin/env python
# coding: utf-8

# ## Lab15: Text dataset creation and design of Simple RNN for Sentiment Analysis

# Name : Deepa S
#     
# RollNo : 225229107

# ### 1. Import libraries

# In[1]:


import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Embedding,SimpleRNN


# In[3]:


import nltk
nltk.download('stopwords')


# In[4]:


from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# ### 2. Creation of data

# In[6]:


df = pd.read_csv("data.csv",encoding="ISO-8859-1")


# In[7]:


df.head()


# ### 3. Opening your CSV file

# In[8]:


import csv


# In[9]:


file = open('data.csv')
type(file)


# In[10]:


csvreader = csv.reader(file)


# In[11]:


header = []
header = next(csvreader)
header


# In[24]:


with open('data.csv', 'r') as file:
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
row


# In[25]:


file.close()


# In[26]:


df.info()


# ### 4. Pre-processing the text

# In[27]:


y = df['Labels']
X = df['Quotes']


# ### 5. Dataset Preparation

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)


# In[29]:


print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[30]:


# 4th step to be continue
train_token = Tokenizer(num_words=100,oov_token='<oov>')
train_token.fit_on_texts(X_train)
word_index = train_token.word_index
train_sequence = train_token.texts_to_sequences(X_train)
dict(list(word_index.items())[0:10])


# In[31]:


vocab = len(train_token.word_index) + 1
vocab


# In[32]:


train_sequence[3]


# In[33]:


train_padded = pad_sequences(train_sequence,maxlen=100,padding='post')
train_padded[5]


# In[34]:


train_padded.shape


# In[35]:


val_token = Tokenizer(num_words=500,oov_token='<oov>')
val_token.fit_on_texts(X_val)
val_index = val_token.word_index
val_sequence = val_token.texts_to_sequences(X_val)


# In[37]:


if len(val_sequence) > 4:
    # Access the element at index 4
    value = val_sequence[4]
else:
    # Handle the case where the list doesn't have enough elements
    print("The list doesn't have enough elements.")


# In[38]:


val_padded = pad_sequences(val_sequence,maxlen=100,padding='post')


# ### 6. Model Creation

# In[40]:


model = Sequential()
# Embedding layer
model.add(Embedding(300,70,input_length=100))
model.add(SimpleRNN(70,activation='relu'))
model.add(Dense('1',activation='sigmoid'))


# In[41]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[42]:


model.summary()


# In[43]:


history=model.fit(train_padded,y_train,epochs=10,verbose=2,batch_size=15)


# In[44]:


model.evaluate(val_padded,y_val)


# In[45]:


plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[46]:


plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[47]:


text = df["Quotes"]


# In[48]:


#sent = [w.lower() for w in text.split() if not w in STOPWORDS]
trail_token = Tokenizer()
trail_token.fit_on_texts(text)
#word_index = trail_token.word_index
trail_seq = trail_token.texts_to_sequences(text)
#dict(list(word_index.items())[0:10])
trail_pad = pad_sequences(trail_seq,maxlen=100,padding='post')


# In[49]:


trail_pad


# ### Step-7:

# In[50]:


res = model.predict(trail_pad)
label = ['positive','negative']
print(res,label[np.argmax(trail_pad)>50])


# ### Step-8:
# 
# ### Model 2

# In[51]:


model1 = Sequential()
# Embedding layer
model1.add(Embedding(5000,64,input_length=100))
model1.add(SimpleRNN(32,activation='tanh'))
model1.add(Embedding(5000,32,input_length=100))
model1.add(SimpleRNN(32,activation='tanh' ))
model1.add(Dense('1',activation='sigmoid'))


# In[52]:


model1.summary()


# In[53]:


model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[54]:


history1=model1.fit(train_padded,y_train,epochs=10,verbose=2,batch_size=15)


# In[55]:


model1.evaluate(val_padded,y_val)


# In[56]:


plt.plot(history1.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[57]:


plt.plot(history1.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[58]:


res = model1.predict(trail_pad)
label = ['positive','negative']
print(res,label[np.argmax(trail_pad)>50])


# ### Model 3

# In[59]:


model2 = Sequential()
# Embedding layer
model2.add(Embedding(4000,128,input_length=100))
model2.add(SimpleRNN(64,activation='tanh'))
model2.add(Embedding(4000,128,input_length=100))
model2.add(SimpleRNN(64,activation='relu' ))
model2.add(Embedding(4000,128,input_length=100))
model2.add(SimpleRNN(64,activation='tanh' ))
model2.add(Dense('1',activation='sigmoid'))


# In[60]:


model2.summary()


# In[61]:


model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[62]:


history2=model2.fit(train_padded,y_train,epochs=10,verbose=2,batch_size=15)


# In[63]:


model2.evaluate(val_padded,y_val)


# In[64]:


plt.plot(history2.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[65]:


plt.plot(history2.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[66]:


res = model2.predict(trail_pad)
label = ['positive','negative']
print(res,label[np.argmax(trail_pad)>50])

