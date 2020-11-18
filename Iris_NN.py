#!/usr/bin/env python
# coding: utf-8

# **Author: Samar Shaikh**

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Load Data**

# In[ ]:


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
df = pd.read_csv('iris.csv', names=names)
df.head()


# **Mapping**

# In[ ]:


mapping = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}
df['species'] = df['species'].apply(lambda x: mapping[x])
df.head()


# In[ ]:


df[10:]


# **Splitting Data**

# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop('species', axis=1).values
y = df['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **Converting Data to Tensors**

# In[ ]:


X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# In[ ]:


X_train


# **Create Model**

# In[ ]:


class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(4, 16)
    self.fc2 = nn.Linear(16, 12)
    self.out = nn.Linear(12, 3)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)
    return x

model = Network()
model


# **Training Neural Net**

# In[ ]:


#Set Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[ ]:


#Training Loop
%%time
epochs = 100
losses = []

for i in range(epochs):
  y_hat = model.forward(X_train)
  loss = criterion(y_hat, y_train)
  losses.append(loss)
  if i%10 == 0:
    print(f'Epochs: {i} Loss: {loss}')
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


# In[ ]:


#visualizing Loss
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss');


# **Validating The Model**

# In[ ]:


preds = []
with torch.no_grad():
  for val in X_test:
    y_hat = model.forward(val)
    preds.append(y_hat.argmax().item())
df2 = pd.DataFrame({'Y': y_test, 'YHat': preds})
df2


# **Calculating Accuracy**

# In[ ]:


df2['correct'] = [1 if corr == preds else 0 for corr, preds in zip(df2['Y'], df2['YHat'])]
acc = df2['correct'].sum()/len(df2)
print(f'Accuracy is: {acc*100}%')


# In[ ]:


#Prediction on Unknown Data
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
data = torch.Tensor([4.0, 3.3, 1.7, 0.5])
data


# In[ ]:


with torch.no_grad():
  print(model(data))
  print(labels[model(data).argmax()])


# In[ ]:




