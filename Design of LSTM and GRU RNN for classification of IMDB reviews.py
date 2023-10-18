#!/usr/bin/env python
# coding: utf-8

# ## PDL LAB- 16

# Name : Deepa S
#     
# RollNo : 225229107

# ##### Step 1: Imports

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU, Dropout
from tensorflow.keras.optimizers import Adam


# ##### Step 2: Load and preprocess the dataset
# 

# In[2]:


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)


# ##### Step 3: Dataset Preparation
# 

# In[3]:


max_sequence_length = 500 
x_train = pad_sequences(x_train, maxlen=max_sequence_length)
x_test = pad_sequences(x_test, maxlen=max_sequence_length)


# In[4]:


split_ratio = 0.6
split_index = int(len(x_train) * split_ratio)
x_train_split, y_train_split = x_train[:split_index], y_train[:split_index]
x_val_split, y_val_split = x_train[split_index:], y_train[split_index:]


# ##### Step 4: Model Creation
# 

# In[5]:


embedding_dim = 128
hidden_units = 64


# In[6]:


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=3)


# ##### Step 5: Run the models with different LSTM hidden layers (2, 3, 4)

# In[7]:


lstm_hidden_layers = [2, 3, 4]
for num_layers in lstm_hidden_layers:
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length))
    for _ in range(num_layers):
        model.add(LSTM(hidden_units, return_sequences=True)) 
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(f"Training LSTM model with {num_layers} hidden layers")
    model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=3)


# ##### Step 6: Variations
# 

# In[8]:


from tensorflow.keras.layers import Bidirectional
bidirectional_model = Sequential()
bidirectional_model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length))
bidirectional_model.add(Bidirectional(LSTM(hidden_units)))
bidirectional_model.add(Dense(1, activation='sigmoid'))
bidirectional_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
bidirectional_model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=3)


# In[9]:


max_sequence_length = 400
hidden_units = 128


# In[10]:


from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_sequence_length = 400 
x_train_split = pad_sequences(x_train_split, maxlen=max_sequence_length)
x_val_split = pad_sequences(x_val_split, maxlen=max_sequence_length)


# In[12]:


from tensorflow.keras.layers import Bidirectional, Flatten
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(hidden_units, return_sequences=True))
model.add(LSTM(hidden_units, return_sequences=True))  
model.add(Flatten())  
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=3)


# ###### Step 7: Run the model with different RNN layers and uits
# 

# In[13]:


rnn_layers = ["LSTM", "GRU"]
units_list = [16, 32, 64]
for rnn_layer in rnn_layers:
    for units in units_list:
        model = Sequential()
        model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length))
        if rnn_layer == "LSTM":
            model.add(LSTM(units))
        elif rnn_layer == "GRU":
            model.add(GRU(units))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(f"Training {rnn_layer} model with {units} units")
        model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=3)


# ##### Step 8: Predict the class for a sample text
# 

# In[16]:


from tensorflow.keras.preprocessing.text import Tokenizer
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
tokenizer = Tokenizer(num_words=10000)
x_train_text = [" ".join([word_index.get(i - 3, "") for i in sequence]) for sequence in x_train]
tokenizer.fit_on_texts(x_train_text)
word_index = tokenizer.word_index


# In[19]:


sample_text = "TCS, Wipro, HCL Technologies and Infosys have ramped up their hiring projections and have added over 50,000 people in the second quarter of FY22, taking the hiring number to more than one lakh (1,02,517) in the first six months of the fiscal year. These four firms employ more than one fourth of India's total workforce."
sample_text = [word_index[word.lower()] if word.lower() in word_index else 2 for word in sample_text.split()]
sample_text = pad_sequences([sample_text], maxlen=max_sequence_length)
prediction = model.predict(sample_text)
if prediction >= 0.5:
    print("Positive Sentiment")
else:
    print("Negative Sentiment")


# In[19]:




