import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
# TODO: Load document directory
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets

    img_normal_means = [0.485, 0.456, 0.406]
    img_normal_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(img_normal_means, img_normal_std)
        ]),
        'validate': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(img_normal_means, img_normal_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(img_normal_means, img_normal_std)
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'validate': datasets.ImageFolder(valid_dir, transform=data_transforms['validate']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'validate': DataLoader(image_datasets['validate'], batch_size=32, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }
    return image_datasets,dataloaders
