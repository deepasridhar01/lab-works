#!/usr/bin/env python
# coding: utf-8

# Name : S.Deepa
#     
# RollNO : 225229107

# ## Lab06 : Predictive Analytics for Hospitals

# #### Step 1 : Import dataset

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("diabetes.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.info


# In[11]:


df["Glucose"].value_counts()


# #### step 2 : Identify relationships between feature

# In[23]:


import seaborn as sns
import matplotlib as plt
sns.heatmap(df.head(),annot=True)


# #### Step 3 : Prediction Using One Feature

# In[25]:


X=df[['Age']]
y=df[['Outcome']]


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42)
LOR=LogisticRegression()
LOR.fit(X_train,y_train)


# In[27]:


LOR.predict(X_test)


# In[29]:


#model parameter values
print("coef_ : ",LOR.coef_)
print("intercept_ : ",LOR.intercept_)


# In[30]:


LOR.predict([[60]])


# In[31]:


lrf=LOR.coef_*60+LOR.intercept_
from scipy.special import expit
a=expit(lrf)
a


# In[32]:


if a>.5:
    print("yes,he will become diabetic!")
else:
    print("No,he will not be diabetic!")


# #### Step 4:Prediction Using Many Features

# In[33]:


X1=df[['Glucose','BMI','Age']]
y1=df[['Outcome']]
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=.25,random_state=42)
LOR1=LogisticRegression()
LOR1.fit(X1_train,y1_train)
LOR1.predict(X1_test)


# In[36]:


#model parameter values
print("coef_ : ",LOR1.coef_)
print("intercept_ : ",LOR1.intercept_)


# In[37]:


lrf=LOR1.coef_*150*30*40+LOR1.intercept_
from scipy.special import expit
expit(lrf)


# In[38]:


LOR1.predict([[150,30,40]])


# In[39]:


LOR1.predict_proba([[150,30,40]])


# #### Step 5 : Build LoR model with all features

# In[40]:


X2=df.drop(['Outcome'],axis=1)
y2=df[['Outcome']]
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=.25,random_state=42)
LOR2=LogisticRegression()
LOR2.fit(X2_train,y2_train)
LOR2.predict(X2_test)


# In[41]:


y2_pred=LOR2.predict(X2_test)
y2_pred


# In[42]:


from sklearn.metrics import roc_auc_score
lor_auc3=roc_auc_score(y2_test,y2_pred)
lor_auc3


# #### Step 6 : Forward Selection Procedure

# In[44]:


def get_auc(var,tar,df):
    fX=df[var]
    fy=df[tar]
    LOR4=LogisticRegression()
    LOR4.fit(fX,fy)
    pred=LOR4.predict_proba(fX)[:,1]
    auc_val=roc_auc_score(y2,pred)
    return auc_val
get_auc(['Glucose','BMI'],['Outcome'],df)


# In[45]:


get_auc(['Pregnancies','BloodPressure','SkinThickness'],['Outcome'],df)


# In[48]:


def best_next(current,cand,tar,df): 
    best_auc=-1
    best_var=None
    for i in cand:
        auc_v=get_auc(current+[i],tar,df)
    if auc_v>=best_auc:
        best_auc=auc_v
        best_var=i
    return best_var


# In[49]:


tar=['Outcome']
current=['Insulin','BMI','DiabetesPedigreeFunction','Age']
cand=['Pregnancies','Glucose','BloodPressure','SkinThickness']
next_var=best_next(current,cand,tar,df)
print(next_var)


# In[51]:


tar = ["Outcome"]
current =[]
cand =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
max_num = 7
num_it = min(max_num, len(cand))
for i in range(0, num_it):
    next_var = best_next(current, cand, tar, df)
    current += [next_var]
    cand.remove(next_var)
    print("Variable added in Step "+str(i+1) +' is ' + next_var +".")


# In[52]:


print(current)


# In[53]:


Step : 7 


# In[54]:


X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,stratify=y2,test_size=0.5,random_state=42)


# In[55]:


prediction = LOR2.predict_proba(X2_test)


# In[57]:


train = pd.concat([X2_train, y2_train], axis =1)
test = pd.concat([X2_test, y2_test], axis =1)
def auc_train_test (variables, target, train, test):
    X_train = train [variables]
    X_test = test[variables]
    Y_train = train[target]
    Y_test = test[target]
    LR3 = LogisticRegression()
    LR3.fit(X_train, Y_train)
    predictions_train = LR3.predict_proba(X_train)[:,1]
    predictions_test = LR3.predict_proba(X_test)[:,1]
    auc_train = roc_auc_score(Y_train, predictions_train)
    auc_test = roc_auc_score(Y_train, predictions_test)
    return (auc_train, auc_test)
auc_values_train =[]
auc_values_test =[]
variables_evaluate=[]
for v in X2.columns:
    variables_evaluate.append(v)
    auc_train, auc_test = auc_train_test(variables_evaluate, ['Outcome'],train,test)
    auc_values_train.append(auc_train)
    auc_values_test.append(auc_test)


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
x = np.array(range(0,len(auc_values_train)))
my_train = np.array(auc_values_train)
my_test = np.array(auc_values_test)
plt.xticks(x, X2.columns, rotation =90)

plt.plot(x, my_train)
plt.plot(x, my_test)
plt.ylim(0.5, 1)
plt.show()


# In[61]:


step : 8


# In[68]:


get_ipython().system('pip install scikit-plot')
from scikitplot.estimators import plot_feature_importances
from scikitplot.metrics import plot_confusion_matrix, plot_roc


# In[69]:


import scikitplot as skplt
skplt.metrics.plot_cumulative_gain(y2_test, prediction)
plt.show()
plt.figure(figsize=(7,7))
skplt.metrics.plot_lift_curve(y2_test, prediction)
plt.show()


# In[ ]:




