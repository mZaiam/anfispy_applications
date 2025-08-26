import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import optuna
import numpy as np
import matplotlib.pyplot as plt

from ANFISpy import LSTMANFIS

# Loading and separating data

df = pd.read_csv('brazil_covid19_macro.csv')
cases = df['cases'].values
cases = np.diff(cases)
cases = pd.Series(cases).rolling(window=20, center=True, min_periods=1).mean().values

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

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])     
        y.append(data[i + 1 : i + seq_length + 1])  
    x = torch.FloatTensor(np.array(x)).unsqueeze(-1) 
    y = torch.FloatTensor(np.array(y)).unsqueeze(-1)
    return x, y

# Setting Optuna objective

def objective(trial):
    params = {
        'shape': trial.suggest_categorical('shape', ['gaussian', 'bell', 'sigmoid']),
        'n_sets': trial.suggest_int('n_sets', 5, 9),
        'seq_length': trial.suggest_int('seq_length', 5, 25),
        'and_operator': trial.suggest_categorical('and_operator', ['prod', 'min']),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
    }

    seq_len = params['seq_length'] 

    x_train, y_train = create_sequences(cases_train, seq_len)
    x_val, y_val = create_sequences(cases_val, seq_len)
    x_test, y_test = create_sequences(cases_test, seq_len)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
 
    n_vars = 1
    variables = {
        'inputs': {
            'n_sets': [params['n_sets']],
            'uod': n_vars * [(0, 1.5)],
            'var_names': None,
            'mf_names': None,
        },
        'output': {
            'var_names': None,
            'n_classes': 1,
        },
    }

    # Instantiate ANFIS
    if params['and_operator'] == 'prod':
        lstmanfis = LSTMANFIS(
            variables,
            params['shape'],
            params['seq_length'],
            and_operator=torch.prod,
            output_activation=nn.Identity(),
            bidirectional=True,
            mean_rule_activation=False
        )
    else:
        lstmanfis = LSTMANFIS(
            variables,
            params['shape'],
            params['seq_length'],
            and_operator=torch.min,
            output_activation=nn.Identity(),
            bidirectional=True,
            mean_rule_activation=False
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstmanfis.parameters(), lr=params['lr'])

    epochs = 1000
    best_loss = float('inf')
    
    for epoch in range(epochs):
        lstmanfis.train()
        epoch_loss_train = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = lstmanfis(X_batch)[0]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * X_batch.size(0) 
    
        epoch_loss_train /= len(train_loader.dataset)
        
        lstmanfis.eval()
        with torch.no_grad():
            outputs = lstmanfis(x_val)[0]
            loss_val = criterion(outputs, y_val)
        
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
    
    return best_loss

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=500)

# Saving DataFrame

df_trials = study.trials_dataframe()
df_trials.to_csv("bilstmanfis_daily_optuna.csv", index=False)
