# Imports here
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch
from torch import nn,optim
import torch.nn.functional as F
from PIL import Image
#from workspace_utils import active_session
#from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.autograd import Variable
import argparse
from load import load_data

#load pre-trained model
def pre_train_model(arch,hidden_units,output_units):
    if arch=='vgg19':
        model = models.vgg19(pretrained=True)
    elif arch=='densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Supported archs: vgg, densenet.")
        
    for param in model.parameters():
        param.requires_grad = False
        
#features = list(model.classifier.children())[:-1]
#input_size = model.classifier[len(features)].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
                              ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.5)),
                             ('fc2', nn.Linear(hidden_units, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(hidden_units, output_units)),
                            ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

# Define training function. 
def do_deep_learning(model, trainloader, epochs, print_every, device, learning_rate,criterion, optimizer):        
                        
    model.to(device)
        
    steps = 0   
    running_loss = 0
    
    for e in range(epochs):
        
        model.train()
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()                
                val_loss = 0
                correct = 0
                
                with torch.no_grad():
                    for images, labels in dataloaders['validate']:
                        images, labels = images.to(device), labels.to(device)
                    
                        optimizer.zero_grad()
                        outputs =  model(images)
                        _, predicted = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        val_loss +=loss.item() * inputs.size(0)                 
                        correct += torch.sum(predicted == labels.data).double()                       
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Val loss: {:.4f}".format(val_loss/len(image_datasets['validate'])),
                      "Val accuracy: {:.4f}".format(correct/len(image_datasets['validate'])))
                running_loss = 0
                
                model.train()
    return model

# TODO: Define testing function.
def test(model, testloader, criterion, device):
    model.to(device)
    test_loss = 0
    test_correct = 0 
    with torch.no_grad():        
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs =  model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss +=loss.item() * images.size(0)                 
            test_correct += torch.sum(predicted == labels.data).double()
    
    return test_loss, test_correct 

# TODO: Save the checkpoint 

def save_checkpoint(model, arch, checkpoint_filename, epochs, optimizer):
    model.class_to_idx = image_datasets['train'].class_to_idx
    classifier_input_units = model.classifier[0].in_features
    classifier_hidden_units = [_.out_features for _ in model.classifier if type(_) is nn.Linear]
    classifier_hidden_units = classifier_hidden_units[:-1]        
    
    checkpoint = {'model':model,
                  'arch':arch,
                  'hidden_units': classifier_hidden_units,
                  'input_units':classifier_input_units,
                  'epochs':epochs,
                  'optimizer_state':optimizer.state_dict,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, checkpoint_filename)


parser = argparse.ArgumentParser(description='Train your image classifier')
parser.add_argument('--data_dir', type=str,  default='flowers', help='Data directory of filepath')
parser.add_argument('--checkpoint_filename', type=str,  default='my_checkpoint.pt', help='Save trained model checkpoint to file')
parser.add_argument('--arch', type=str, default='vgg19', help='Model architecture(supported archs: vgg19, densenet121).')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
parser.add_argument('--output_units', type=int, default=102, help='Number of hidden units')
parser.add_argument('--epochs', type=int,  default=10, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU if available')

args = parser.parse_args()
print(args)

image_datasets, dataloaders = load_data(args.data_dir)

if args.gpu and torch.cuda.is_available():
    print('Using GPU for training')
    device = torch.device("cuda:0")
else:
    print('Using CPU for training')
    device = torch.device("cpu") 
    
pre_model = pre_train_model(args.arch,args.hidden_units,args.output_units)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pre_model.classifier.parameters(), args.learning_rate)
done_model = do_deep_learning(pre_model, dataloaders['train'], args.epochs, 100, device, args.learning_rate,criterion,optimizer)
test(done_model, dataloaders['test'], criterion, device)
save_checkpoint(done_model, args.arch, args.checkpoint_filename, args.epochs, optimizer)    