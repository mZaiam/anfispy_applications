import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import optuna
import numpy as np
import matplotlib.pyplot as plt

# Loading and separating data

df = pd.read_csv('brazil_covid19_macro.csv')
cases = df['cases'].values

total_len = len(cases)  
train_size = int(0.8 * total_len)
val_size = int(0.1 * total_len)
test_size = total_len - train_size - val_size

cases_train = cases[:train_size]
cases_val = cases[train_size:train_size + val_size]
cases_test = cases[train_size + val_size:]

norm_min, norm_max = cases_train.min(), cases_train.max()

cases_train = (cases_train - norm_min) / (norm_max - norm_min)
cases_val = (cases_val - norm_min) / (norm_max - norm_min)
cases_test = (cases_test - norm_min) / (norm_max - norm_min)

def create_data(data):
    x = []
    y = []
    for i in range(len(data) - 1):
        x.append([data[i]])        
        y.append(data[i + 1])      
    return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(-1)

X_train, y_train = create_data(cases_train)
X_test, y_test = create_data(cases_test)
X_val, y_val = create_data(cases_val)

# Setting Optuna objective

class NN(nn.Module):
    def __init__(
        self,
        size_layers,
        activation,
    ):
        super(NN, self).__init__()
        
        layers = []
        
        for i in range(len(size_layers) - 1):
            layers.append(
                nn.Linear(in_features=size_layers[i], out_features=size_layers[i + 1]),
            )
            layers.append(activation)
            
        self.model = nn.Sequential(*layers)
                    
    def forward(self, x):
        return self.model(x)

def objective(trial):
    params = {
        'size_layers': trial.suggest_int('size_layers', 8, 128),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'activation': trial.suggest_categorical('activation', ['relu', 'identity']),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'shuffle': trial.suggest_categorical('shuffle', [True, False]),
    }
    
    layers = [1]
    for n in range(params['num_layers']):
        layers.append(params['size_layers'])
    layers.append(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=params['shuffle'])
    
    if params['activation'] == 'relu':
        activation = nn.ReLU()
        
    if params['activation'] == 'identity':
        activation = nn.Identity()

    model = NN(
        size_layers=layers,
        activation=activation,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    epochs = 1000
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss_train = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * X_batch.size(0) 
    
        epoch_loss_train /= len(train_loader.dataset)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            loss_val = criterion(outputs, y_val)
            
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
    
    return best_loss

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=500)

# Saving DataFrame

df_trials = study.trials_dataframe()
df_trials.to_csv("mlp_optuna.csv", index=False)
