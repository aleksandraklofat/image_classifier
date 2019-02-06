
# coding: utf-8

# In[ ]:


# Imports here
from __future__ import print_function, division
import time
import numpy as np
import os 
import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.optim import lr_scheduler
from PIL import Image
import argparse

# Setting argparse module

parser = argparse.ArgumentParser(description='Train an image classifier')

parser.add_argument('--data_dir', type=str, default='flowers', help='dataset files')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--arch', type=str, default ='vgg13', help='model architecture')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
parser.add_argument('--epochs', type=int, default=20,  help='epochs')
parser.add_argument('--foo', type=int, choices=['vgg16', 'densenet121'])
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='saving checkpoint')

args = parser.parse_args()




data_dir = args.data_dir
print("data_dir: {}".format(args.data_dir))


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Transforming datasets
data_transforms = {
    'train' : transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    'validation' : transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),                                        
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    
    'test' : transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),                                        
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
}



# Load the datasets with ImageFolder

train_data = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
validation_data = datasets.ImageFolder(valid_dir, transform = data_transforms['validation'])
test_data = datasets.ImageFolder(test_dir ,transform = data_transforms['test'])
       

# Using the image datasets and the trainforms, define the dataloaders
    
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=30, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)



# GPU if it's available
device = torch.device("cuda" if args.gpu else "cpu")

# Choosing the preterained model

if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    input_size = model.classifier[0].in_features
    
elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = model.classifier[1].in_features
else:
    print('You can only select these two models')


  
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
# Building the classifier    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.1)), 
                          ('hidden', nn.Linear(args.hidden_units, 1000)),
                          ('fc2', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    


    
if args.arch == 'alexnet':
    model.classifier = classifier
else:    
    model.classifier = classifier

criterion = nn.NLLLoss()
    
if args.arch == 'alexnet':
    optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)



# Training the model

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        model.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    model.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validation_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validation_loader):.3f}")
            running_loss = 0
            model.train()



# TODO: Do validation on the test set
def testing_my_network(test_loader):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the test set: %d %%' % (100 * correct / total))
    
testing_my_network(test_loader)

# TODO: Save the checkpoint 
    
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch':args.arch,
              'input':input_size,
              'output':102,
              'epochs':args.epochs,
              'learning_rate':args.learning_rate,
              'dropout':0.1,
              'classifier':classifier,
              'state_dict':model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, args.save_dir)
    
    



