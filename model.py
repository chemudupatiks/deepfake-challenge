# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:43:48 2020

@author: ckris
"""


from data_loader import data
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


faces = 'Faces/'  
videos = 'deepfake-detection-challenge/train_sample_videos/'
df = data(videos, faces)
train = df[:int(df.shape[0]*0.8)]
train_labels = np.zeros(len(train))
train_labels[train['label']=='FAKE'] = 1
train_X = np.array([np.array(a) for a in train['frame']])



test = df[int(df.shape[0]*0.8):]
test_labels = np.zeros(len(test))
test_labels[test['label']=='FAKE'] = 1
test_X = np.array([np.array(a) for a in test['frame']])


tensor_train_data = torch.utils.data.TensorDataset(torch.reshape(torch.tensor(train_X, dtype=torch.float), (3367, 3, 160, 160)), torch.tensor(train_labels, dtype=torch.long))
trainloader = torch.utils.data.DataLoader(tensor_train_data, batch_size=4,shuffle=True, num_workers=0)

tensor_test_data = torch.utils.data.TensorDataset(torch.reshape(torch.tensor(test_X, dtype=torch.float), (842, 3, 160, 160)), torch.tensor(test_labels, dtype=torch.long))
testloader = torch.utils.data.DataLoader(tensor_test_data, batch_size=4,shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # print('in')
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # print(inputs)
        # print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        # print(i)
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print('Accuracy of the network on Epoch: ',epoch,'= %.2f %%' % (
    100 * correct / total))

print('Finished Training')

PATH = './deepfake_net.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 850 test images: %.2f %%' % (
    100 * correct / total))






