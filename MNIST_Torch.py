# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:55:39 2023

@author: Nautilus
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Set the working directory
path='F:/Semester 1 Spring 2023/Machine Learning for CVEN/Project_4'
os.chdir(path)

# Load train data
training_data = datasets.MNIST( 
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Load test data
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

batch_size = 64 #iterable will return a batch of 64 features and labels.

# Create data loaders.  This wraps an iterable over our dataset, and supports automatic batching, sampling, 
# shuffling and multiprocess data loading
train_dataloader = DataLoader(training_data,  batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define the structure of model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #convert 2D image into a contiguous array 
        self.linear_relu_stack = nn.Sequential( #processed orderly
            nn.Linear(28*28, 128), #a linear transformation on the input using its stored weights and biases
            nn.ReLU(), #activation function         
            nn.Linear(128, 128), #Hidden layer
            nn.ReLU(),
            nn.Linear(128, 10) #output layer
        )

    def forward(self, x): #feed forward function
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_list=[]
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_list.append(float(loss.data.mean()))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_list

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 50
time_list=[]
train_loss_list = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    start_time=time.time()   
    model=model.train()
    train_loss=train(train_dataloader,model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    train_loss_list.append(np.mean(train_loss))
    total_time=time.time()-start_time
    time_list.append(total_time)
print("Done!")

print('Total training time:', sum(time_list))

x = np.linspace(0.0, epochs-1, epochs)
plt.figure(figsize=(6,4))
plt.xlabel('Epoch')
plt.ylabel('Training time per epoch (s)')
plt.plot(x, time_list)
plt.grid()

x = np.linspace(0.0, epochs-1, epochs)

plt.figure(figsize=(6,4))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x, train_loss_list)
plt.grid()

torch.save(model.state_dict(), "Torch_model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("Torch_model.pth"))

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()

#calculate accuracy using testing data
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cpu')
            y = y.to(device='cpu')
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()
check_accuracy(test_dataloader, model)

net=model()
for epoch in range(epochs):

    correct = 0
    for i, (inputs,labels) in enumerate (train_dataloader):
        ...
        output = net(inputs)
        ...
        optimizer.step()

        correct += (output == labels).float().sum()

    accuracy = 100 * correct / len(training_data)
    # trainset, not train_loader
    # probably x in your case

    print("Accuracy = {}".format(accuracy))
