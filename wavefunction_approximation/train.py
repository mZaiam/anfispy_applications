import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

from ANFISpy import ANFIS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

mfshape = 'gaussian'
gridsize = 100

with open(f"{gridsize}_data.pkl", "rb") as f:
    data = pickle.load(f)

gx, gy = data['gx'].ravel(), data['gy'].ravel()

x = np.stack((gx, gy), axis=1)
y = data['ps'].ravel()

SEED = 42

x, index = np.unique(x, axis=0, return_index=True)
y = y[index]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.15, random_state=SEED)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.1, random_state=SEED)

x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

x_train_mean = x_train.mean(dim=0)
x_test_mean = x_test.mean(dim=0)
x_val_mean = x_val.mean(dim=0)
x_train_std = x_train.std(dim=0)
x_test_std = x_test.std(dim=0)
x_val_std = x_val.std(dim=0)

x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_test_mean) / x_test_std
x_val = (x_val - x_val_mean) / x_val_std

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f'Data processing for {gridsize} done.\n')

n_vars = 2
mf_names = [['Low', 'Medium', 'High']]

variables = {
    'inputs': {
        'n_sets': [3, 3],
        'uod': n_vars * [(-3, 3)],
        'var_names': ['X', 'Y'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'P',
        'n_classes': 1,
    },
}

anfis = ANFIS(
    variables,
    mfshape,
    and_operator=torch.prod,
    output_activation=nn.Identity(),
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(anfis.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1e-1, patience=10)

epochs = 200
train_losses, val_losses = [], []

print(f'{mfshape} ({gridsize})\n')

for epoch in range(epochs):
    anfis.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = anfis(x_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    anfis.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = anfis(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if epoch % 10 == 9:
        print(f'Epoch {epoch+1}/{epochs}: train_loss: {train_loss:.5f} validation_loss {val_loss:.5f}')

np.save(f'data_{gridsize}/train_{mfshape}_{gridsize}.npy', train_losses)
np.save(f'data_{gridsize}/val_{mfshape}_{gridsize}.npy', val_losses)

with torch.no_grad():
    y_pred = anfis(x_test).squeeze()

rmse = torch.sqrt(torch.sum((y_pred - y_test) ** 2) / y_test.shape[0])
mae = torch.sum(torch.abs(y_pred - y_test)) / y_test.shape[0]
r2 = 1 - (torch.sum((y_pred - y_test) ** 2)) / (torch.sum((y_test - torch.mean(y_test)) ** 2))

print(f'Metrics\n')
print(f'RMSE: {rmse.item():.4f}')
print(f'MAE: {mae.item():.4f}')
print(f'R2: {r2.item():.4f}\n')

torch.save(anfis.state_dict(), f'data_{gridsize}/{mfshape}_{gridsize}.pth')