#!/usr/bin/env python
# coding: utf-8

# Name : Deepa S
# 
# Roll NO : 225229107

# ### Lab2. Design of Logic Gates using Perceptron and Keras

# ### Part-I: Design OR gate using the concept of Perceptron

# In[2]:


import numpy as np
def logic_gate(w1, w2, b):
    return lambda x1, x2: sigmoid(w1 * x1 + w2 * x2 + b)
def test(gate):
      for a, b in (0, 0), (0, 1), (1, 0), (1, 1):
            print("{}, {}: {}".format(a, b, np.round(gate(a,b))))


# In[3]:


def sigmoid(x):
    return 1/(1+np.exp(-x))
or_gate = logic_gate(20, 20, -10)
test(or_gate)


# ### Part-II: Implement the operations of AND, NOR and NAND gates

# In[4]:


def logic_gate(w1, w2, b):
    return lambda x1, x2: sigmoid(w1 * x1 + w2 * x2 + b)

def test(gate):
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            print("{}, {}: {}".format(a, b, np.round(gate(a, b))))

#step1:
# AND gate
and_gate = logic_gate(1, 1, -1.5)
print("AND gate:")
test(and_gate)
print()

#step2:
# NOR gate
nor_gate = logic_gate(-1, -1, 0.5)
print("NOR gate:")
test(nor_gate)
print()

#step3:
# NAND gate
nand_gate = logic_gate(-1, -1, 1.5)
print("NAND gate:")
test(nand_gate)


# ### Part-III Limitations of single neuron for XOR operation

# In[6]:


def xor_gate(x1, x2):
    
    w1 = np.array([[20, -20], [-20, 20]])
    b1 = np.array([-10, 30])
    w2 = np.array([[-20], [-20]])
    b2 = np.array([30])

    
    a1 = sigmoid(np.dot(np.array([x1, x2]), w1) + b1)

    
    a2 = sigmoid(np.dot(a1, w2) + b2)

    return a2[0]


print("0 XOR 0 =", xor_gate(0, 0))
print("0 XOR 1 =", xor_gate(0, 1))
print("1 XOR 0 =", xor_gate(1, 0))
print("1 XOR 1 =", xor_gate(1, 1))


# ### Part-IV Logic Gates using Keras library

# In[8]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [0], [0], [1]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input_data, output_data, epochs=100)

predictions = model.predict(input_data)
print(predictions)


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [1]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input_data, output_data, epochs=100)

predictions = model.predict(input_data)
print(predictions)


input_data = np.array([[0], [1]])
output_data = np.array([[1], [0]])

model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input_data, output_data, epochs=100)

predictions = model.predict(input_data)
print(predictions)


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[1], [1], [1], [0]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input_data, output_data, epochs=100)

predictions = model.predict(input_data)
print(predictions)


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[1], [0], [0], [0]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input_data, output_data, epochs=100)

predictions = model.predict(input_data)
print(predictions)


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
model.fit(input_data, output_data, epochs=100)

predictions = model.predict(input_data)
print(predictions)


# In[ ]:




