#!/usr/bin/env python
# coding: utf-8

# # Pizza Liking Prediction using kNN

# In[ ]:


Name:S.Deepa
Rollno:225229107
lab02


# # step 1: Prepare your dataset

# In[5]:


import pandas as pd
df=pd.read_csv('pizza.csv')
print(df)


# In[6]:


import pandas as pd
df1=pd.read_csv('pizza_test.csv')
print(df1)


# # step 2:import dataset

# In[7]:


df.head()


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df1.shape


# In[11]:


df.columns


# In[12]:


df1.columns


# In[13]:


df.info


# In[14]:


df1.info


# # step 3:visualize relationships

# In[16]:


import seaborn as sns
sns.relplot(x='age',y='weight',hue='likePizza',data=df)


# # step 4:Prepare x matrix and y vector

# In[17]:


X=pd.DataFrame(df)
cols=[0,1]
X=X[X.columns[cols]]


# In[20]:


y=df['likePizza'].values


# # step 5: Examine x and y

# In[21]:


X


# In[22]:


type(X)


# In[23]:


y


# In[24]:


type(y)


# # step 6:Model building

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X,y)


# # step 7:Model testing

# In[28]:


knn.predict(X)


# In[32]:


a=[25,50]
knn.predict([a])


# In[43]:


b=[60,60]
knn.predict([b])


# # step 8:Change n_neighbors = 3

# In[33]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)


# In[34]:


c=[25,50]
knn.predict([c])


# In[35]:


d=[60,60]
knn.predict([d])


# # step 9:predict on entire dataset

# In[47]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)


# In[48]:



y_pred=knn.predict(X)
y_pred


# # step 10:Accuracy function

# In[49]:


def accuracy(actual,pred):
    return sum(actual==pred)/float(actual.shape[0])


# # step 11:Find accuracy

# In[50]:


accuracy_score=accuracy(y,y_pred)
accuracy_score


# # step 12:Prediction on Test set

# In[53]:


import pandas as pd
df1=pd.read_csv('pizza_test.csv')
print(df1)


# In[54]:


df1.head()


# In[55]:


df1.shape


# In[56]:


df1.columns


# In[57]:


df1.info


# In[59]:


x=pd.DataFrame(df1)
cols=[0,1]
x=x[x.columns[cols]]


# In[60]:


x


# In[63]:


Y=df1['likePizza'].values
Y


# In[64]:


from sklearn.neighbors import KNeighborsClassifier
test=KNeighborsClassifier(n_neighbors=2)
test.fit(x,Y)


# In[65]:


Y_pred=test.predict(x)
Y_pred


# In[67]:


import numpy as np
Y=np.array([1,1,0,0])
Y


# In[69]:


Y_test=accuracy(Y,Y_pred)
Y_test


# # step 13: Find best value for k

# In[74]:


scores=[]
for k in range(1,4):
    kn=KNeighborsClassifier(n_neighbors=k)
    kn.fit(x,Y)
    kn.predict(x)
    y_test=kn.predict(x)
    a=accuracy(Y,Y_pred)
    scores.append((k,a))
    
print(scores)


# # step 14: Accuracy_score function

# In[75]:


from sklearn.metrics import accuracy_score


# In[77]:


accuracy_score(Y,Y_pred)


# In[ ]:




