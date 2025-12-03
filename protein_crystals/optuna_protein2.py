import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import pandas as pd
import optuna
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ANFISpy import ANFIS
from ANFISpy import HamacherAND, MinAND, FrankAND, ProdAND, LukasiewiczAND

# Loading data

SEED = 42
torch.manual_seed(SEED)

df = pd.read_excel('protein_data2.xlsx')
df = df.dropna()

FEATURES = ["Lisozima (mg/mL)", "Cloreto de sódio (M)"]
TARGET = ["Gota 1", "Gota 2", "Gota 3"]

x = df[FEATURES].values.astype('float16')
y = df[TARGET].values.astype('float16')
y = stats.mode(y, axis=1)[0]

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
y[-7] = 1

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
        'n_sets_protein': trial.suggest_int('n_sets_protein', 2, 5),
        'n_sets_salt': trial.suggest_int('n_sets_salt', 2, 5),
        'and_operator': trial.suggest_categorical('and_operator', ['prod', 'min', 'frank', 'luka', 'hamacher']),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accuracies = []

    for train_idx, val_idx in kf.split(x_kfold, y_kfold):
        x_train, y_train = x_kfold[train_idx], y_kfold[train_idx]
        x_val, y_val = x_kfold[val_idx], y_kfold[val_idx]

        x_max = x_train.max(dim=0, keepdim=True)[0]
        x_train = x_train / x_max
        x_val = x_val / x_max

        class_counts = torch.bincount(y_train)
        weights = 1.0 / class_counts.float()
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=2*len(y_train), replacement=True)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

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
                'n_classes': 2,
            },
        }

        match params['and_operator']:
            case 'prod':
                and_op = ProdAND()
            case 'min':
                and_op = MinAND()
            case 'hamacher':
                and_op = HamacherAND()
            case 'frank':
                and_op = FrankAND()
            case 'luka':
                and_op = LukasiewiczAND()

        anfis = ANFIS(
            variables,
            params['shape'],
            and_operator=and_op,
            output_activation=nn.Identity()
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(anfis.parameters(), lr=params['lr'])

        # Training
        epochs = 100
        for epoch in range(epochs):
            anfis.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = anfis(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        anfis.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = anfis(x_batch)
                preds = preds.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(y_batch)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    return mean_acc

study_anfis = optuna.create_study(direction='maximize')
study_anfis.optimize(objective_anfis, n_trials=500)
anfis_results_df = study_anfis.trials_dataframe()
anfis_results_df.to_csv('anfis_optuna2.csv', index=False)

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

        class_counts = torch.bincount(y_train)
        weights = 1.0 / class_counts.float()
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=2*len(y_train), replacement=True)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        input_dim = 2
        output_dim = 2

        match params['activation']:
            case 'relu':
                act = nn.ReLU()
            case 'tanh':
                act = nn.Tanh()
            case 'sigmoid':
                act = nn.Sigmoid()

        model = MLP(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            n_layers=params['n_layers'],
            output_dim=output_dim,
            activation=act,
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

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                y_pred = torch.argmax(y_pred, dim=1)
                all_preds.append(y_pred)
                all_labels.append(y_batch)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        accuracies.append(acc)

    return np.mean(accuracies)

study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=500)
mlp_results_df = study_mlp.trials_dataframe()
mlp_results_df.to_csv('mlp_optuna2.csv', index=False)