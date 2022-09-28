#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np


# In[84]:


df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
               names=["sepal length","sepal widht","petal length","petal width","class"])


# In[85]:


df.head()


# In[86]:


df.tail()


# In[87]:


import seaborn as sns
import matplotlib.pyplot as plt
  
# loading dataset
data = sns.load_dataset("iris")
  
plot = sns.FacetGrid(data, col="species")
plot.map(plt.plot, "sepal_width")
  
plt.show()


# In[88]:


labels = df['class']
iris_data = df.drop(['class'], axis=1)
print(labels)


# In[89]:


labels = np.where(labels == 'Iris-setosa',1,-1)


# In[90]:


dp= np.array(df)
dp.shape


# In[91]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dp,labels,test_size=0.2)
print(x_train.shape)
x_test.shape


# In[92]:


def perceptron(c,X,d,w,iter):           #Function for perceptron
    for n in range(1,iter):
        print("---Iteration :---",n)    #Iteration given is 20
        for i, x in enumerate(X):
            net = np.dot(X[i],w)        
            if net > 0:                #Bipolar discrete
                out = 1
            else:
                out = -1
            r = c*(d[i] - out)        #calculate weight
            del_w = r*x
            w = del_w+w
            print ("Weight in iteration",n,w)
    return w

X = data.iloc[:,0:4].values            #Input define with .iloc is used when dataset is not numeric 
print ("Inputs:", X)                   #or if user doen't knows the index lable
d1=data.iloc[:,4].values
d=np.where(d1=='Iris-setosa',1,-1)
print ("Teacher values", d)
                                                                          
w= ([0,0,0,0])                                         #Initial weight as zero
print ("Initial weights:", w)
c = 2                                               #Fix the learning rate
iterations = 2                                        #define iteration is 20
print ("----Training-----")
final_weight = perceptron(c,X_train,y_train,w,iterations)      #Print the final weight
print ("Final weights: ", final_weight)


# In[93]:


def test_perceptron(final_out,X,w):         #Testing function
    for i,x in enumerate(X):
        net = np.dot(X[i],w)
        if net>0:
            out = 1
        else:
            out = -1
        final_out = final_out+[out]
    return final_out
final_out = []
print ("------Testing-----")
output = test_perceptron(final_out,X_test,final_weight)   #calculate and print final output
print ("Final output: ", output)


# In[124]:


for i,x in enumerate(output):
    if output[i]== y_train[i]:
        print("true")
    else:
        print("false")


# In[96]:


plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


# ![image.png](attachment:image.png)
