# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
from collections import OrderedDict
from process import process_image
import json
from torch.autograd import Variable

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])    
    return model, checkpoint['class_to_idx']

def predict(image_path, model, topk, device, class_to_idx, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    # TODO: Implement the code to predict the class from an image file    
    image= process_image(image_path)
    image = Variable(torch.cuda.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        fc_out = model.forward(image)
    
    probs = torch.nn.functional.softmax(fc_out.data,dim=1)
    probs, idx = probs.topk(topk)
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    top_class = [idx_to_class[x] for x in idx.tolist()[0]]
    names = [cat_to_name[k] for k in top_class]
    return probs.tolist()[0], names

parser = argparse.ArgumentParser(description='Predict the class')
parser.add_argument('--image_path', type=str, metavar='', default='flowers/test/28/image_05230.jpg', help='Path of image')
parser.add_argument('--checkpoint_path', type=str, metavar='', default='my_checkpoint.pt', help='Save trained model checkpoint to file')
parser.add_argument('--topk', type=int, metavar='', default=5, help='Top K most likely classes')
parser.add_argument('--category_names', type=str, metavar='', default='cat_to_name.json', help='Filename which contains real names to map categories')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU if available')

args = parser.parse_args()
print(args)

# load model
model, class_to_idx = load_checkpoint(args.checkpoint_path)

# make predictions
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

if args.gpu and torch.cuda.is_available():
    print('Using GPU for training')
    device = torch.device("cuda:0")
else:
    print('Using CPU for training')
    device = torch.device("cpu") 
    
print(predict(args.image_path, model, args.topk, device, class_to_idx, cat_to_name))