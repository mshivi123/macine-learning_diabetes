
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


mydata=datasets.load_diabetes()


# In[3]:


mydata


# In[4]:


mydata.keys()


# In[5]:


mydata.feature_names


# In[6]:


mydata.DESCR


# In[7]:


x_input=mydata.data
y_target=mydata.target
mydata.target


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x_input,y_target,test_size=.3)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


myobj=LinearRegression()


# mymodel=myobj.fit(x_train,y_train)

# In[13]:


mymodel=myobj.fit(x_train,y_train)


# In[14]:


Yp=mymodel.predict(x_test)


# In[15]:


Ya=y_test


# In[16]:


from sklearn import metrics


# In[17]:


import numpy as np


# In[18]:


error=metrics.mean_squared_error(Ya,Yp)


# In[20]:


np.sqrt(error)

