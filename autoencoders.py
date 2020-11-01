"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


"""## training set and the test set"""

training_set_1 = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set_1 = np.array(training_set_1, dtype = 'int')
test_set_1 = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set_1 = np.array(test_set_1, dtype = 'int')

training_set_2 = pd.read_csv('ml-100k/u2.base', delimiter = '\t')
training_set_2 = np.array(training_set_1, dtype = 'int')
test_set_2 = pd.read_csv('ml-100k/u2.test', delimiter = '\t')
test_set_2 = np.array(test_set_1, dtype = 'int')

training_set_3 = pd.read_csv('ml-100k/u3.base', delimiter = '\t')
training_set_3 = np.array(training_set_1, dtype = 'int')
test_set_3 = pd.read_csv('ml-100k/u3.test', delimiter = '\t')
test_set_3 = np.array(test_set_1, dtype = 'int')

training_set_4 = pd.read_csv('ml-100k/u4.base', delimiter = '\t')
training_set_4 = np.array(training_set_1, dtype = 'int')
test_set_4 = pd.read_csv('ml-100k/u4.test', delimiter = '\t')
test_set_4 = np.array(test_set_1, dtype = 'int')

training_set_5 = pd.read_csv('ml-100k/u5.base', delimiter = '\t')
training_set_5 = np.array(training_set_1, dtype = 'int')
test_set_5 = pd.read_csv('ml-100k/u5.test', delimiter = '\t')
test_set_5 = np.array(test_set_1, dtype = 'int')

"""## Getting the number of users and movies"""

user_count_training_set = int(max(training_set_1[:, 0]))
user_count_test_set = int(max(test_set_1[:, 0]))
user_count = int(max(user_count_test_set, user_count_training_set))

movies_count_training_set = int(max(training_set_1[:, 1]))
movies_count_test_set = int(max(test_set_1[:, 1]))
movies_count = int(max(movies_count_test_set, movies_count_training_set))

"""## Converting the data into an array with users in rows and movies in columns"""

def modify(data):
  listoflist = []
  for ui in range(1, user_count+1):
    mi =data[:, 1][data[:, 0]==ui]
    ri =data[:, 2][data[:, 0]==ui]
    listofratings =np.zeros(movies_count)
    listofratings[mi-1] =ri
    listoflist.append(list(listofratings))
  return listoflist


training_set_1 = modify(training_set_1)
test_set_1 = modify(test_set_1)

training_set_2 = modify(training_set_2)
test_set_2 = modify(test_set_2)

training_set_3 = modify(training_set_3)
test_set_3 = modify(test_set_3)

training_set_4 = modify(training_set_4)
test_set_4 = modify(test_set_4)

training_set_5 = modify(training_set_5)
test_set_5 = modify(test_set_5)
"""## Converting the data into Torch tensors"""

training_set_1=torch.FloatTensor(training_set_1)
test_set_1=torch.FloatTensor(test_set_1)

training_set_2=torch.FloatTensor(training_set_2)
test_set_2=torch.FloatTensor(test_set_2)

training_set_3=torch.FloatTensor(training_set_3)
test_set_3=torch.FloatTensor(test_set_3)

training_set_4=torch.FloatTensor(training_set_4)
test_set_4=torch.FloatTensor(test_set_4)

training_set_5=torch.FloatTensor(training_set_5)
test_set_5=torch.FloatTensor(test_set_5)

"""## Creating the architecture of the Neural Network"""

class MODELL(nn.Module):
    def __init__(self, ):
        super(MODELL, self).__init__()
        self.layer1 =nn.Linear(movies_count, 20)
        self.layer2 =nn.Linear(20, 10)
        self.layer3 =nn.Linear(10, 20)
        self.layer4 =nn.Linear(20, movies_count)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x =self.activation(self.layer1(x))
        x =self.activation(self.layer2(x))
        x =self.activation(self.layer3(x))
        x =self.layer4(x)
        return x

archi = MODELL()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(archi.parameters(), lr = 0.01, weight_decay = 0.5)

"""## Training the autoencoder : part 1"""

e_counts = 100
for epoch in range(e_counts):
  train_loss1 , s1 =0 , 0.                                    
  for ui in range(user_count):
    input_info = Variable(training_set_1[ui]).unsqueeze(0)
    target_info = input_info.clone()
    if torch.sum(target_info.data > 0) > 0:
      output_info =archi(input_info)
      target_info.require_grad = False
      output_info[target_info ==0] =0
      l =criterion(output_info, target_info)
      adj_cons = movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
      l.backward()
      train_loss1 +=np.sqrt(l.data*adj_cons)
      s1 +=1.
      optimizer.step()
  print('train_1 loss at epoch number '+ str(epoch+1)+ ' is ' +str(train_loss1/s1))
print()

"""## Testing the autoencoder : part 1"""

test_loss1 ,s1  = 0 , 0.
 
for ui in range(user_count):
  input_info= Variable(training_set_1[ui]).unsqueeze(0)
  target_info =Variable(test_set_1[ui]).unsqueeze(0)
  if torch.sum(target_info.data>0)> 0:
    output_info =archi(input_info)
    target_info.require_grad= False
    output_info[target_info== 0] = 0
    l= criterion(output_info, target_info)
    adj_cons =movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
    test_loss1 += np.sqrt(l.data*adj_cons)
    s1+= 1.
print('test_1 loss: '+str(test_loss1/s1))
print()


archi = MODELL()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(archi.parameters(), lr = 0.01, weight_decay = 0.5)

"""## Training the autoencoder : part 2"""

e_counts = 100
for epoch in range(e_counts):
  train_loss2 , s2 =0 , 0.                                    
  for ui in range(user_count):
    input_info = Variable(training_set_2[ui]).unsqueeze(0)
    target_info = input_info.clone()
    if torch.sum(target_info.data > 0) > 0:
      output_info =archi(input_info)
      target_info.require_grad = False
      output_info[target_info ==0] =0
      l =criterion(output_info, target_info)
      adj_cons = movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
      l.backward()
      train_loss2 +=np.sqrt(l.data*adj_cons)
      s2 +=1.
      optimizer.step()
  print('train_2 loss at epoch number '+ str(epoch+1)+ ' is ' +str(train_loss2/s2))
print()

"""## Testing the autoencoder : part 2"""

test_loss2 ,s2  = 0 , 0.
 
for ui in range(user_count):
  input_info= Variable(training_set_2[ui]).unsqueeze(0)
  target_info =Variable(test_set_2[ui]).unsqueeze(0)
  if torch.sum(target_info.data>0)> 0:
    output_info =archi(input_info)
    target_info.require_grad= False
    output_info[target_info== 0] = 0
    l= criterion(output_info, target_info)
    adj_cons =movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
    test_loss2 += np.sqrt(l.data*adj_cons)
    s2+= 1.
print('test_2 loss: '+str(test_loss2/s2))
print()


archi = MODELL()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(archi.parameters(), lr = 0.01, weight_decay = 0.5)

"""## Training the autoencoder : part 3"""

e_counts = 100
for epoch in range(e_counts):
  train_loss3 , s3 =0 , 0.                                    
  for ui in range(user_count):
    input_info = Variable(training_set_3[ui]).unsqueeze(0)
    target_info = input_info.clone()
    if torch.sum(target_info.data > 0) > 0:
      output_info =archi(input_info)
      target_info.require_grad = False
      output_info[target_info ==0] =0
      l =criterion(output_info, target_info)
      adj_cons = movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
      l.backward()
      train_loss3 +=np.sqrt(l.data*adj_cons)
      s3 +=1.
      optimizer.step()
  print('train_3 loss at epoch number '+ str(epoch+1)+ ' is ' +str(train_loss3/s3))
print()

"""## Testing the autoencoder : part 3"""

test_loss3 ,s3  = 0 , 0.
 
for ui in range(user_count):
  input_info= Variable(training_set_3[ui]).unsqueeze(0)
  target_info =Variable(test_set_3[ui]).unsqueeze(0)
  if torch.sum(target_info.data>0)> 0:
    output_info =archi(input_info)
    target_info.require_grad= False
    output_info[target_info== 0] = 0
    l= criterion(output_info, target_info)
    adj_cons =movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
    test_loss3 += np.sqrt(l.data*adj_cons)
    s3+= 1.
print('test_3 loss: '+str(test_loss3/s3))
print()

archi = MODELL()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(archi.parameters(), lr = 0.01, weight_decay = 0.5)

"""## Training the autoencoder : part 4"""

e_counts = 100
for epoch in range(e_counts):
  train_loss4 , s4 =0 , 0.                                    
  for ui in range(user_count):
    input_info = Variable(training_set_4[ui]).unsqueeze(0)
    target_info = input_info.clone()
    if torch.sum(target_info.data > 0) > 0:
      output_info =archi(input_info)
      target_info.require_grad = False
      output_info[target_info ==0] =0
      l =criterion(output_info, target_info)
      adj_cons = movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
      l.backward()
      train_loss4 +=np.sqrt(l.data*adj_cons)
      s4 +=1.
      optimizer.step()
  print('train_4 loss at epoch number '+ str(epoch+1)+ ' is ' +str(train_loss4/s4))
print()

"""## Testing the autoencoder : part 4"""

test_loss4 ,s4  = 0 , 0.
 
for ui in range(user_count):
  input_info= Variable(training_set_4[ui]).unsqueeze(0)
  target_info =Variable(test_set_4[ui]).unsqueeze(0)
  if torch.sum(target_info.data>0)> 0:
    output_info =archi(input_info)
    target_info.require_grad= False
    output_info[target_info== 0] = 0
    l= criterion(output_info, target_info)
    adj_cons =movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
    test_loss4 += np.sqrt(l.data*adj_cons)
    s4+= 1.
print('test_4 loss: '+str(test_loss4/s4))
print()

archi = MODELL()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(archi.parameters(), lr = 0.01, weight_decay = 0.5)

"""## Training the autoencoder : part 5"""

e_counts = 100
for epoch in range(e_counts):
  train_loss5 , s5 =0 , 0.                                    
  for ui in range(user_count):
    input_info = Variable(training_set_5[ui]).unsqueeze(0)
    target_info = input_info.clone()
    if torch.sum(target_info.data > 0) > 0:
      output_info =archi(input_info)
      target_info.require_grad = False
      output_info[target_info ==0] =0
      l =criterion(output_info, target_info)
      adj_cons = movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
      l.backward()
      train_loss5 +=np.sqrt(l.data*adj_cons)
      s5 +=1.
      optimizer.step()
  print('train_5 loss at epoch number '+ str(epoch+1)+ ' is ' +str(train_loss5/s5))
print()

"""## Testing the autoencoder : part 5"""

test_loss5 ,s5  = 0 , 0.
 
for ui in range(user_count):
  input_info = Variable(training_set_5[ui]).unsqueeze(0)
  target_info = Variable(test_set_5[ui]).unsqueeze(0)
  if torch.sum(target_info.data>0)> 0:
    output_info =archi(input_info)
    target_info.require_grad= False
    output_info[target_info== 0] = 0
    l= criterion(output_info, target_info)
    adj_cons =movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
    test_loss5 += np.sqrt(l.data*adj_cons)
    s5+= 1.
print('test_5 loss: '+str(test_loss5/s5))
print()

total_error = ((test_loss1)/s1) + ((test_loss2)/s2) + ((test_loss3)/s3) + ((test_loss4)/s4) + ((test_loss5)/s5)
total_error /= 5.0
print("Total average Loss after 5-fold cross Validation : " + str(total_error))

