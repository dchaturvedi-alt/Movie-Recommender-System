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

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

"""## Getting the number of users and movies"""

user_count_training_set = int(max(training_set[:, 0]))
user_count_test_set = int(max(test_set[:, 0]))
user_count = int(max(user_count_test_set, user_count_training_set))

movies_count_training_set = int(max(training_set[:, 1]))
movies_count_test_set = int(max(test_set[:, 1]))
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


training_set = modify(training_set)
test_set = modify(test_set)

"""## Converting the data into Torch tensors"""

training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

"""## Creating the architecture of the Neural Network"""

class MODELL(nn.Module):
    def __init__(self, ):
        super(MODELL, self).__init__()
        self.layer1 =nn.Linear(movies_count, 10)
        self.layer2 =nn.Linear(10, 8)
        self.layer3 =nn.Linear(8, 10)
        self.layer4 =nn.Linear(10, movies_count)
        self.activation = nn.ReLU()
    def forward(self, x):
        x =self.activation(self.layer1(x))
        x =self.activation(self.layer2(x))
        x =self.activation(self.layer3(x))
        x =self.layer4(x)
        return x

archi =MODELL()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(archi.parameters(), lr = 0.1, weight_decay = 0.5)

"""## Training the autoencoder"""

e_counts = 100
for epoch in range(e_counts):
  train_loss , s =0 , 0.                                    
  for ui in range(user_count):
    input_info = Variable(training_set[ui]).unsqueeze(0)
    target_info = input_info.clone()
    if torch.sum(target_info.data > 0) > 0:
      output_info =archi(input_info)
      target_info.require_grad = False
      output_info[target_info ==0] =0
      l =criterion(output_info, target_info)
      adj_cons = movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
      l.backward()
      train_loss +=np.sqrt(l.data*adj_cons)
      s +=1.
      optimizer.step()
  print('loss at epoch number '+ str(epoch+1)+ ' is ' +str(train_loss/s))

"""## Testing the autoencoder"""

test_loss ,s  = 0 , 0.
 
for ui in range(user_count):
  input_info= Variable(training_set[ui]).unsqueeze(0)
  target_info =Variable(test_set[ui]).unsqueeze(0)
  if torch.sum(target_info.data>0)> 0:
    output_info =archi(input_info)
    target_info.require_grad= False
    output_info[target_info== 0] = 0
    l= criterion(output_info, target_info)
    adj_cons =movies_count/float(torch.sum(target_info.data > 0) + 1e-10)
    test_loss += np.sqrt(l.data*adj_cons)
    s+= 1.
print('test loss: '+str(test_loss/s))


















