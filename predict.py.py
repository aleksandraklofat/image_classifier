
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from torch import optim,nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,models,transforms

import argparse
import os
from PIL import Image


# Setting argparse module
parser = argparse.ArgumentParser(description='predicting images')
parser.add_argument('--image', type=str, default = 'flowers/test/18/image_04256.jpg',
                    help='image to be classified')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                    help='load the checkpoint')
parser.add_argument('--gpu', type=bool, default=False,
                    help='use GPU')
parser.add_argument('--top_k', action='store', type=int, default=5,
                    help='top probabilities')
parser.add_argument('--json1', action='store',type=str, default = 'cat_to_name.json',
                    help='json file with flower names')

args = parser.parse_args()

# Setting GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Lodading checkpoint
def load_checkpoint(file_path):
    '''Function that loads and rebuilds the model'''
    checkpoint = torch.load(file_path)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)        
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True) 
    else:
        print("Your input is not a valid model. Please try again")
     
    model.to(device)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_path=Image.open(image)
    img_tranforms=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img_tranforms(image_path)



 

def predict(image_dir, model, top_k=5):
    '''Predicts probabilities using image and model as inputs'''
    
    # Opening the json file
    with open(args.json1, 'r') as f:
        cat_to_name = json.load(f)
    
    image = process_image(image_dir)
    image = image.unsqueeze(0).float()
    
    device = torch.device('cuda:0' if args.gpu else 'cpu')
    
   
    
    model = load_checkpoint(model)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(Variable(image.cuda()))
        ps = torch.exp(output)
        
    prob, index = torch.topk(ps, top_k)
    top_pb = np.array(prob.data[0])
    Index = np.array(index.data[0])
    
    
        
    idx_to_class = {idx:clas for clas,idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in Index]
    labels = [cat_to_name[clas] for clas in classes]

    return top_pb,labels

# Printing classes and probabilities
probability, classes = predict(args.image, args.checkpoint, args.top_k)
print('Results:')
for top_pb, clas in zip(probability, classes):
    print(f"The probability of flower being {clas} is {top_pb}.")
    








