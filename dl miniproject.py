#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
               names=["sepal length","sepal widht","petal length","petal width","class"])


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


import numpy as np
import pandas as pd
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
               names=["sepal length","sepal widht","petal length","petal width","class"])        #Import iris dataset               
from sklearn.model_selection import train_test_split   #Import train_test_split is used to split dataset into test and train set

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
            w = w+del_w
            print ("Weight in iteration",n,w)
    return w


def test_perceptron(final_out,X,w):         #Testing function
    for i,x in enumerate(X):
        net = np.dot(X[i],w)
        if net>0:
            out = 1
        else:
            out = -1
        final_out = final_out+[out]
    return final_out


X = data.iloc[:,0:4].values            #Input define with .iloc is used when dataset is not numeric 
print ("Inputs:", X)                   #or if user doen't knows the index lable
d1=data.iloc[:,4].values
d=np.where(d1=='Iris-setosa',1,-1)
print ("Teacher values", d)
                                                                          
X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33)     #traing and testing 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print ("Original Teacher values:",y_test)
w= ([0,0,0,0])                                         #Initial weight as zero
print ("Initial weights:", w)
c = 0.5                                                #Fix the learning rate
iterations = 20                                        #define iteration is 20
print ("----Training-----")
final_weight = perceptron(c,X_train,y_train,w,iterations)      #Print the final weight
print ("Final weights: ", final_weight)

final_out = []
print ("------Testing-----")
output = test_perceptron(final_out,X_test,final_weight)   #calculate and print final output
print ("Final output: ", output)


# In[ ]:




