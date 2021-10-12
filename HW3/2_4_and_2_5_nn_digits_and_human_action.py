from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, SubsetRandomSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class CharacterRecognitionDataset(Dataset):
    def __init__(self, X, y):
        X, y = datasets.load_digits(return_X_y=True)
        self.input = np.float32(X)
        self.target = np.int64(y)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.input[idx, :], self.target[idx]

class MyCharRecogNetwork(nn.Module):
    def __init__(self):
        super(MyCharRecogNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 128),  # input layer -> hidden
            nn.ReLU(),
            nn.Linear(128, 128),  # hidden -> output
            nn.ReLU(),
            nn.Linear(128, 10)  # because we expect 10 possible output in one-hot encoding
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(data_loader, model, loss_fn, optimizer, verbose=False):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if verbose:
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(data_loader, model, loss_fn, verbose=False):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f}\n")
    return 100*correct

def run_experiment(X, y):
    k_folds = 5
    epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    verbose = False
    
    # Model
    model = MyCharRecogNetwork().to(device)
    # Define the K-fold Cross Validator
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Perform learning in a cross-folded setup.
    accuracies_over_folds = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        digit_data_train = CharacterRecognitionDataset(X_train, y_train)
        digit_data_test = CharacterRecognitionDataset(X_test, y_test)
    
        # Define data loaders for training and testing data in the fold.
        data_loader_train = DataLoader(digit_data_train, 
            batch_size=batch_size, shuffle=True)
        data_loader_test = DataLoader(digit_data_test, 
        batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer.
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), 
            lr=learning_rate)

        # Perform the learning.
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n", "-"*25)
            train(data_loader_train, model, loss_fn, optimizer, verbose)
            acc = test(data_loader_test, model, loss_fn, verbose)
        
        # Collect the accuracies from the final epoch in a list.
        accuracies_over_folds.append(acc)
        print(f"Accuracy for fold {fold} : {acc}")

    # Print experiment results.
    print(f"Five-folded accuracies: {accuracies_over_folds}")
    print(f"Average over five-folds: {mean(accuracies_over_folds)}")

def digits():
    # Dataset
    X, y = datasets.load_digits(return_X_y=True)
    run_experiment(X, y)

def human_activity_recognition():
    path_to_dataset = Path("/home/hpc/kurlanl1/CSC-380/CSC380-Artificial-Intelligence/UCIHARDataset/")

    # Get the data from the files.
    X_train = np.genfromtxt(path_to_dataset / "train/X_train_clean.txt", 
        delimiter=',')
    y_train = np.genfromtxt(path_to_dataset / "train/y_train.txt", 
        delimiter=',')
    X_test = np.genfromtxt(path_to_dataset / "test/X_test_clean.txt", 
        delimiter=',')
    y_test = np.genfromtxt(path_to_dataset / "test/y_test.txt", 
        delimiter=',')
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test, axis=0)
    run_experiment(X, y)

if __name__ == "__main__":
    #digits()
    human_activity_recognition()

