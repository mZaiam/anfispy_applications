import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import optuna
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ANFISpy import ANFIS

# Loading data

SEED = 42
torch.manual_seed(SEED)

df = pd.read_excel('protein_data.xlsx')
df = df.dropna()

FEATURES = ["Concentração de proteína (em mg/mL)", "Concentração de sal (em mg/mL)"]
TARGET = ["Definicação do cristal"]

x = df[FEATURES].values.astype('float16')
y = df[TARGET].values.astype('float16')
y = np.clip(np.rint(y), 0, 2)

# Remove duplicates keeping the most frequent label
unique_x, inverse_indices = np.unique(x, axis=0, return_inverse=True)

most_frequent_y = []
for idx in range(len(unique_x)):
    group_y = y[inverse_indices == idx]
    most_common = mode(group_y, keepdims=False).mode
    most_frequent_y.append(most_common)

most_frequent_y = np.array(most_frequent_y).astype(int).squeeze()

x = torch.tensor(unique_x, dtype=torch.float32)
y = torch.tensor(most_frequent_y, dtype=torch.long)

x_kfold, x_test, y_kfold, y_test = train_test_split(
    x, y,
    test_size=0.2,      
    random_state=SEED,
    stratify=y    
)

# Optuna ANFIS
def objective_anfis(trial):
    params = {
        'shape': trial.suggest_categorical('shape', ['gaussian', 'bell', 'sigmoid']),
        'n_sets_protein': trial.suggest_int('n_sets_protein', 2, 7),
        'n_sets_salt': trial.suggest_int('n_sets_salt', 2, 7),
        'and_operator': trial.suggest_categorical('and_operator', ['prod', 'min']),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accuracies = []

    for train_idx, val_idx in kf.split(x_kfold, y_kfold):
        x_train, y_train = x_kfold[train_idx], y_kfold[train_idx]
        x_val, y_val = x_kfold[val_idx], y_kfold[val_idx]

        x_max = x_train.max(dim=0, keepdim=True)[0]
        x_train = x_train / x_max
        x_val = x_val / x_max

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        # Variables for ANFIS
        n_vars = 2
        variables = {
            'inputs': {
                'n_sets': [params['n_sets_protein'], params['n_sets_salt']],
                'uod': n_vars * [(0, 1)],
                'var_names': None,
                'mf_names': None,
            },
            'output': {
                'var_names': None,
                'n_classes': 3,
            },
        }

        # Instantiate ANFIS
        if params['and_operator'] == 'prod':
            anfis = ANFIS(
                variables,
                params['shape'],
                and_operator=torch.prod,
                output_activation=nn.Identity(),
                mean_rule_activation=False
            )
        else:
            anfis = ANFIS(
                variables,
                params['shape'],
                and_operator=torch.min,
                output_activation=nn.Identity(),
                mean_rule_activation=False
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(anfis.parameters(), lr=params['lr'])

        # Training
        epochs = 100
        for epoch in range(epochs):
            anfis.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = anfis(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        anfis.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                preds = anfis(xb)
                preds = preds.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(yb)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    return mean_acc

study_anfis = optuna.create_study(direction='maximize')
study_anfis.optimize(objective_anfis, n_trials=500) 
anfis_results_df = study_anfis.trials_dataframe()
anfis_results_df.to_csv('anfis_optuna.csv', index=False)

# Optuna MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, activation):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective_mlp(trial):
    params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 8, 64),
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accuracies = []

    for train_idx, val_idx in kf.split(x_kfold, y_kfold):
        x_train, y_train = x_kfold[train_idx], y_kfold[train_idx]
        x_val, y_val = x_kfold[val_idx], y_kfold[val_idx]

        x_max = x_train.max(dim=0, keepdim=True)[0]
        x_train = x_train / x_max
        x_val = x_val / x_max

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        input_dim = 2
        output_dim = 3

        if params['activation'] == 'relu':
            model = MLP(
                input_dim=input_dim,
                hidden_dim=params['hidden_dim'],
                n_layers=params['n_layers'],
                output_dim=output_dim,
                activation=nn.ReLU(),
            )

        if params['activation'] == 'tanh':
            model = MLP(
                input_dim=input_dim,
                hidden_dim=params['hidden_dim'],
                n_layers=params['n_layers'],
                output_dim=output_dim,
                activation=nn.Tanh(),
            )
            
        elif params['activation'] == 'sigmoid':
            model = MLP(
                input_dim=input_dim,
                hidden_dim=params['hidden_dim'],
                n_layers=params['n_layers'],
                output_dim=output_dim,
                activation=nn.Sigmoid(),
            )
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        epochs = 100

        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(x_val)
            y_pred = torch.argmax(y_pred, dim=1)
            acc = accuracy_score(y_val.numpy(), y_pred.numpy())
            accuracies.append(acc)

    return np.mean(accuracies)

study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=500) 
mlp_results_df = study_mlp.trials_dataframe()
mlp_results_df.to_csv('mlp_optuna.csv', index=False)