#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import the dataset
students_scores=pd.read_csv('students_scores.csv')


# In[3]:


#X = students_scores.iloc[:,-1].values
#y = students_scores.iloc[:, 1].values


# In[4]:


students_scores.describe()


# In[5]:


X= students_scores[['Student Score']].values
y= students_scores['Intelligence Score']


# In[6]:


students_scores.head()


# In[7]:


sns.barplot(y= 'Student Score', x= 'Intelligence Score', data = students_scores)
plt.show()


# In[8]:


#split dataset for testing and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=1/3)


# In[9]:


X_train.shape, y_train.shape


# In[10]:


X_test.shape, y_test.shape


# In[11]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train, y_train)


# In[12]:


y_pred= lr.predict(X_test)
y_pred


# In[13]:


plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color = 'red')
plt.show()


# In[14]:


plt.scatter(X_test, y_test)
plt.plot(X_test, lr.predict(X_test), color = 'red')
plt.show()


# In[15]:


from sklearn import metrics
#Calculate residuals
print('MAE', metrics.mean_absolute_error(y_test, y_pred))
print('MSE', metrics.mean_squared_error(y_test, y_pred))
print('RMSE',np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))


# In[ ]:




