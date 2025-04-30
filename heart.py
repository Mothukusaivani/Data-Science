#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Load the data
df=pd.read_csv("heart.csv")
df


# In[5]:


df.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[7]:


from sklearn.metrics import accuracy_score


# In[8]:


print(df.isnull().sum()) #No null values


# In[9]:


df.dropna(inplace=True) # to drop null values


# In[10]:


df.head()


# In[11]:


X=df.drop('target',axis=1)
Y=df['target']


# In[12]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)


# In[13]:


# Models
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}


# In[14]:


# before normalization
acc_before = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_before[name] = accuracy_score(y_test, y_pred)
print(acc_before)


# In[15]:


#Normalization
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)


# In[16]:


#After Normalization
acc_after={}
for name ,model in models.items():
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    acc_after[name] = accuracy_score(y_test, y_pred)
print(acc_after)


# In[22]:


# Plot Before Normalization
plt.figure(figsize=(8, 5))
plt.bar(acc_before.keys(), acc_before.values(), color='blue')
plt.title('Accuracy Before Normalization')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.show()


# In[21]:


# Plotting: After Normalization
plt.figure(figsize=(8, 6))
plt.bar(acc_before.keys(), acc_before.values(), color='green')
plt.title('Accuracy After Normalization')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.show()


# In[ ]:




