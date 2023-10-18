#!/usr/bin/env python
# coding: utf-8

# ## PDL LAB : 17

# Name : Deepa S
#     
# RollNo : 225229107

# In[1]:


import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import nltk
import sklearn
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.optimizers import RMSprop , Adam
from keras.models import Sequential
from keras.layers import *
from nltk.corpus import stopwords 


# In[2]:


data = pd.read_csv("news20.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


english_stops = set(stopwords.words('english'))


# In[6]:


y =data['category'] 
X=[] 
for review in data['text']: 
    filtered_sentence = [w.lower() for w in review.split() if not w in english_stops ] 
    X.append(filtered_sentence)
X = pd.Series(X) 


# In[7]:


y_tokenizer = Tokenizer() 
y_tokenizer.fit_on_texts(y) 
y_seq = np.array(y_tokenizer.texts_to_sequences (y))
X_token = Tokenizer(num_words=5000,oov_token='<oov>') 
X_token.fit_on_texts(X) 
word_index = X_token.word_index
X_sequence = X_token.texts_to_sequences(X) 
dict(list(word_index.items())[0:15])


# In[8]:


X_padding= pad_sequences(X_sequence, maxlen=200, padding='post') 
print(y_seq.shape) 
print(X_padding.shape)


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(X_padding, y_seq, test_size = 0.2)


# In[10]:


vocab_size = 5000 
embedding_dim = 64 
max_length = 200
model = Sequential() 
model.add(Embedding(vocab_size, embedding_dim)) 
model.add(LSTM(embedding_dim))
model.add(Dense(embedding_dim, activation='tanh'))
model.add(Dense(6,activation='softmax'))
model. summary()


# In[11]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train, epochs=20, verbose=2, validation_split=0.2)


# In[12]:


model1 = Sequential() 
model1.add(Embedding(vocab_size, embedding_dim)) 
model1.add(Conv1D(filters=32, kernel_size=5, strides=1, activation='relu'))
model1.add(MaxPooling1D((2))) 
model1.add(LSTM(embedding_dim)) 
model1.add(Dense(128, activation= 'relu'))
model1.add(Dense(6, activation='softmax')) 
model1.summary()


# In[13]:


model1.compile (optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history1 = model1.fit(x_train,y_train, epochs=20,validation_split=0.2, verbose=2) 


# In[21]:


from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# In[22]:


glove_file = "glove.6B.100d.txt"
glove_word2vec_file = "glove.6B.100d.txt.word2vec"
glove2word2vec(glove_file, glove_word2vec_file)
glove_embeddings = KeyedVectors.load_word2vec_format(glove_word2vec_file, binary=False)


# In[26]:


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in X_token.word_index.items():
    try:
        embedding_vector = glove_embeddings[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector 
    except:
        pass


# In[ ]:


model1 = Sequential() 
model1.add(Embedding(vocab_size, embedding_dim)) 
model1.add(Conv1D(filters=32, kernel_size=5, strides=1, activation='relu'))
model1.add(MaxPooling1D((2))) 
model1.add(LSTM(embedding_dim)) 
model1.add(Dense(128, activation= 'relu'))
model1.add(Dense(6, activation='softmax')) 
model1.summary()


# In[27]:


model2 = Sequential()
model2.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))
model2.add(Conv1D(filters=32, kernel_size=5, strides=1, activation='relu'))
model2.add(MaxPooling1D((2))) 
model2.add(LSTM(embedding_dim)) 
model2.add(Dense(128, activation= 'relu'))
model2.add(Dense(6, activation='softmax')) 
model2.summary()


# In[30]:


model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(x_train,y_train, epochs=5, verbose=2, validation_split=0.2)


# In[ ]:




