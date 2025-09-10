import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sanfis import SANFIS
from ANFISpy import ANFIS
import timeit
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pandas as pd

from skanfis import scikit_anfis
from skanfis.fs import *
from skanfis import *

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = torch.load('x_train.pth')
x_test = torch.load('x_test.pth')
y_train = torch.load('y_train.pth')
y_test = torch.load('y_test.pth')

## ANFISpy

anfispy_time = []
anfispy_preds = []

for i in range(10):
    n_vars = 2
    mf_names = [['L', 'M', 'H']]

    variables = {
        'inputs': {
            'n_sets': [3, 3],
            'uod': n_vars * [(-5, 5)],
            'var_names': ['var1', 'var2'],
            'mf_names': n_vars * mf_names,
        },
        'output': {
            'var_names': 'out',
            'n_classes': 1,
        },
    }

    anfis = ANFIS(variables, 'gaussian')
    anfis.to(device)
    
    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(anfis.parameters())
    
    epochs = 100

    start = timeit.default_timer()

    for epoch in range(epochs):
        anfis.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            y_pred = anfis(x_batch).to(device)
            loss = criterion(y_pred, y_batch.flatten())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
    end = timeit.default_timer()
    
    y_pred = anfis(x_test.to(device))
    
    anfispy_time.append(end - start)
    anfispy_preds.append(y_pred)
    
    print(end - start)
    
np.save('anfispy_time.npy', np.array(anfispy_time))
np.save('anfispy_preds.npy', torch.stack(anfispy_preds).detach().cpu().numpy().squeeze())
    
## SKANFIS

x_train = torch.load('x_train.pth')
x_test = torch.load('x_test.pth')
y_train = torch.load('y_train.pth')
y_test = torch.load('y_test.pth')

skanfis_time = []
skanfis_preds = []

for i in range(10):
    fs = FS()

    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x0", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 0"))

    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x1", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 1"))

    R1 = "IF (x0 IS mf0) AND (x1 IS mf0) THEN (y0 IS sinc_x_y0)"
    R2 = "IF (x0 IS mf0) AND (x1 IS mf1) THEN (y0 IS sinc_x_y1)"
    R3 = "IF (x0 IS mf0) AND (x1 IS mf2) THEN (y0 IS sinc_x_y2)"
    R4 = "IF (x0 IS mf1) AND (x1 IS mf0) THEN (y0 IS sinc_x_y3)"
    R5 = "IF (x0 IS mf1) AND (x1 IS mf1) THEN (y0 IS sinc_x_y4)"
    R6 = "IF (x0 IS mf1) AND (x1 IS mf2) THEN (y0 IS sinc_x_y5)"
    R7 = "IF (x0 IS mf2) AND (x1 IS mf0) THEN (y0 IS sinc_x_y6)"
    R8 = "IF (x0 IS mf2) AND (x1 IS mf1) THEN (y0 IS sinc_x_y7)"
    R9 = "IF (x0 IS mf2) AND (x1 IS mf2) THEN (y0 IS sinc_x_y8)"
    fs.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])

    model = scikit_anfis(fs, description="Sinc", epoch=100, hybrid=True)

    start = timeit.default_timer()

    model.fit(x_train.numpy(), y_train.numpy().squeeze()) 

    end = timeit.default_timer()

    y_pred = model.predict(x_test.numpy())

    skanfis_time.append(end - start)
    skanfis_preds.append(y_pred)

    print(end - start)
    
np.save('skanfis_time.npy', np.array(skanfis_time))
np.save('skanfis_preds.npy', np.array(skanfis_preds).squeeze())

## SANFIS

x_train = torch.load('x_train.pth')
x_test = torch.load('x_test.pth')
y_train = torch.load('y_train.pth')
y_test = torch.load('y_test.pth')

sanfis_preds = []
sanfis_time = []

for i in range(10):
    MEMBFUNCS = [
        {'function': 'gaussian',
         'n_memb': 3,
         'params': {'mu': {'value': [-5., 0.0, 5.],
                           'trainable': True},
                    'sigma': {'value': [2.5, 2.5, 2.5],
                              'trainable': True}}},

        {'function': 'gaussian',
         'n_memb': 3,
         'params': {'mu': {'value': [-5., 0.0, 5.],
                           'trainable': True},
                    'sigma': {'value': [2.5, 2.5, 2.5],
                              'trainable': True}}},

    ]
    
    model = SANFIS(membfuncs=MEMBFUNCS, n_input=2, scale='None').to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    loss_functions = torch.nn.MSELoss()
    
    start = timeit.default_timer()

    history = model.fit(
        train_data=[x_train, y_train],
        valid_data=[x_train, y_train],
        optimizer=optimizer,
        loss_function=loss_functions,
        epochs=100,
        batch_size=32    
    )

    end = timeit.default_timer()
    
    y_pred = model.predict(x_test)
    
    sanfis_time.append(end - start)
    sanfis_preds.append(y_pred)
    
    print(i, end - start)
    
np.save('sanfis_time.npy', np.array(sanfis_time))
np.save('sanfis_preds.npy', torch.stack(sanfis_preds).squeeze().detach().numpy())    