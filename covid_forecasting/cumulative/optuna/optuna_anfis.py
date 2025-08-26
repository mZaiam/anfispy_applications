import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import optuna
import numpy as np
import matplotlib.pyplot as plt

from ANFISpy import ANFIS

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

def objective(trial):
    params = {
        'shape': trial.suggest_categorical('shape', ['gaussian', 'bell', 'sigmoid']),
        'activation': trial.suggest_categorical('activation', ['relu', 'identity']),
        'n_sets': trial.suggest_int('n_sets', 2, 7),
        'and_operator': trial.suggest_categorical('and_operator', ['prod', 'min']),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'shuffle': trial.suggest_categorical('shuffle', [True, False]),
    }
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=params['shuffle'])
    
    n_vars = 1
    variables = {
        'inputs': {
            'n_sets': n_vars * [params['n_sets']],
            'uod': n_vars * [(0, 1.5)],
            'var_names': None,
            'mf_names': None,
        },
        'output': {
            'var_names': None,
            'n_classes': 1,
        },
    }
    
    if params['activation'] == 'relu':
        activation = nn.ReLU()
        
    if params['activation'] == 'identity':
        activation = nn.Identity()

    # Instantiate ANFIS
    if params['and_operator'] == 'prod':
        anfis = ANFIS(
            variables,
            params['shape'],
            and_operator=torch.prod,
            output_activation=activation,
            mean_rule_activation=False
        )
    else:
        anfis = ANFIS(
            variables,
            params['shape'],
            and_operator=torch.min,
            output_activation=activation,
            mean_rule_activation=False
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(anfis.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    epochs = 1000
    best_loss = float('inf')
    
    for epoch in range(epochs):
        anfis.train()
        epoch_loss_train = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = anfis(X_batch).reshape(y_batch.shape)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * X_batch.size(0) 
    
        epoch_loss_train /= len(train_loader.dataset)
        
        anfis.eval()
        with torch.no_grad():
            outputs = anfis(X_val).unsqueeze(-1)
            loss_val = criterion(outputs, y_val)
            
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
    
    return best_loss

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=500)

# Saving DataFrame

df_trials = study.trials_dataframe()
df_trials.to_csv("anfis_optuna.csv", index=False)
