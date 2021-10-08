"""
Problems 1.4 and 1.5.
"""

from pathlib import Path
from pprint import pprint
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class FacesDataset(Dataset):
    """
    Custom dataset for processing faces.
    """

    def __init__(self, annotations_file:Path, img_dir:Path, 
                    transform:transforms.Compose):
        """Constructor for the faces dataset.

        Args:
            annotations_file (Path): location of .csv file of labels
            img_dir (Path): location of .jpg images
            transform (transforms.Compose): series of transformations
                to perform on the images
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx:int):
        img_path = self.img_dir / str(self.img_labels.iloc[idx, 0])
        image = read_image(img_path.as_posix()).float()
        if self.transform:
            image = self.transform(image)
        label = self.img_labels.iloc[idx, 1]
        label = self.target_transform(label)
        return image, label

    def target_transform(self, label:str):
        """Maps the labels from labels csv file to numeric values.
        """
        if label.upper() == "TAYLOR":
            return 0
        elif label.upper() == "ROCK":
            return 1
        elif label.upper() == "QUENTIN":
            return 2

class FaceCNN(nn.Module):
    """Convolutional Neural Network for classifying faces.
    """
    def __init__(self, n_classes:int):
        """Constructor for the faces CNN.

        Args:
            n_classes (int): number of output neurons and classes in the
                faces dataset
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x:torch.Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(dataloader, model, loss_fn, optimizer):
    """Training function.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    """Testing function.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    tracker = {
        0 : {0 : 0, 1 : 0, 2 : 0}, 
        1 : {0 : 0, 1 : 0, 2 : 0}, 
        2 : {0 : 0, 1 : 0, 2 : 0}, 
    }
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            tracker[y.item()][pred.argmax(1).item()] += 1
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return tracker

def resize_images_in_dir(raw_data_path:Path, width:int, height:int):
    """Will resize all images in the given directory to the new size.
    """
    for img_path in raw_data_path.iterdir():
        img = Image.open(img_path)
        img = img.resize((width, height), Image.ANTIALIAS)
        img.save(img_path)

def preprocess_faces():
    """Performs preprocessing of faces into three image sizes.
    """
    root_data_path = Path("/home/hpc/kurlanl1/CSC-380/CSC380-"
                            "Artificial-Intelligence/")
    raw_data_path = root_data_path / "faces_raw"
    processed_data_path = root_data_path / "faces_processed"
    for size in (16, 32, 48, 64):
        path = processed_data_path / str(size)
        if not path.exists():
            shutil.copytree(raw_data_path, path)
            resize_images_in_dir(path / "train" / "X", size, size)
            resize_images_in_dir(path / "test" / "X", size, size)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_loaders_faces(size):
    """Get the train and test DatasetLoaders for faces dataset
    """
    root_data_path = Path("/home/hpc/kurlanl1/CSC-380/CSC380-Artificial"
                            "-Intelligence/faces_processed") / str(size)
    train_data_path = root_data_path / "train"
    test_data_path = root_data_path / "test"

    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = FacesDataset(
        train_data_path / "y" / "y.csv", train_data_path / "X", 
        transform)
    testset = FacesDataset(
        test_data_path / "y" / "y.csv", test_data_path / "X", 
        transform)

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    return trainloader, testloader

def get_loaders_cifar():
    """Get the train and test DatasetLoaders for a test dataset
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                    batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                    batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def main():

    # Get the 32x32 faces datasets
    preprocess_faces()
    trainloader, testloader = get_loaders_faces(size=32)

    # The CNN model, optimizer, and loss function
    model = FaceCNN(n_output=3)
    optimizer = torch.optim.SGD(model.parameters(), 
        lr=1e-3, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Tracks the predictions for each class.
    tracker = {
        0 : {0 : 0, 1 : 0, 2 : 0}, 
        1 : {0 : 0, 1 : 0, 2 : 0}, 
        2 : {0 : 0, 1 : 0, 2 : 0}, 
    }

    # Perform the training
    epochs = 100
    for e in range(epochs):
        if e % 10 == 0:
            print("fEpoch: {e}\n", "-"*20)
        train(trainloader, model, loss_fn, optimizer)
        results_tracker = test(testloader, model, loss_fn)
        for k1 in tracker:
            for k2 in tracker[k1]:
                tracker[k1][k2] += results_tracker[k1][k2]
    
    # Print
    pprint(tracker)

if __name__ == "__main__":
    main()