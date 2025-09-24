import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import timeit
import itertools

from ANFISpy import ANFIS
from skanfis import scikit_anfis
from skanfis.fs import *

np.random.seed(42)
torch.manual_seed(42)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url)
df = df.dropna()

FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
TARGET = ['species']

x = df[FEATURES].values
y = df[TARGET].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.ravel())

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).long()

def accuracy(y_pred, y_true):
    predicted_classes = torch.argmax(y_pred, dim=1)
    return accuracy_score(y_true.numpy(), predicted_classes.numpy())

def f1(y_pred, y_true):
    predicted_classes = torch.argmax(y_pred, dim=1)
    return f1_score(y_true.numpy(), predicted_classes.numpy(), average='weighted')

N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

anfispy_time, anfispy_inference, anfispy_acc, anfispy_f1 = [], [], [], []
skanfis_time, skanfis_inference, skanfis_acc, skanfis_f1 = [], [], [], []
skanfis_online_time, skanfis_online_inference, skanfis_online_acc, skanfis_online_f1 = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
    print(f"FOLD {fold + 1}")

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    
    train_mean = x_train.mean(dim=0, keepdim=True)
    train_std = x_train.std(dim=0, keepdim=True)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std
    
    print('ANFISpy')
    
    n_vars = 4
    mf_names = [['L', 'M', 'H']]

    variables = {
        'inputs': {
            'n_sets': [3, 3, 3, 3],
            'uod': n_vars * [(-5, 5)],
            'var_names': ['var1', 'var2', 'var3', 'var4'],
            'mf_names': n_vars * mf_names,
        },
        'output': {
            'var_names': 'out',
            'n_classes': 3,
        },
    }

    anfis = ANFIS(variables, 'gaussian')
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(anfis.parameters())
    
    epochs = 100

    start = timeit.default_timer()

    for epoch in range(epochs):
        anfis.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = anfis(x_batch)
            loss = criterion(y_pred, y_batch) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
    end = timeit.default_timer()
    
    start_inference = timeit.default_timer()
    with torch.no_grad():
        anfis.eval()
        y_pred = anfis(x_test)
    end_inference = timeit.default_timer()
    
    anfispy_time.append(end - start)
    anfispy_inference.append(end_inference - start_inference) 
    anfispy_acc.append(accuracy(y_pred, y_test))
    anfispy_f1.append(f1(y_pred, y_test))
    
    print('SkANFIS Hybrid')
    
    fs = FS()
    
    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x0", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 0"))

    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x1", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 1"))
    
    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x2", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 2"))

    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x3", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 3"))

    mf_indices = list(itertools.product(range(3), repeat=4))
    rules = []
    for i, p in enumerate(mf_indices):
        antecedent = " AND ".join([f"(x{j} IS mf{p[j]})" for j in range(4)])
        rules.append(f"IF {antecedent} THEN (y0 IS term_{i})")
    fs.add_rules(rules)

    model = scikit_anfis(fs, epoch=100, hybrid=True, label='c')

    start = timeit.default_timer()
    model.fit(x_train.numpy(), y_train.numpy()) 
    end = timeit.default_timer()

    start_inference = timeit.default_timer()
    y_pred = model.predict(x_test.numpy())
    end_inference = timeit.default_timer()

    skanfis_time.append(end - start)
    skanfis_inference.append(end_inference - start_inference) 
    skanfis_acc.append(accuracy_score(y_test.numpy(), y_pred))
    skanfis_f1.append(f1_score(y_test.numpy(), y_pred, average='weighted'))
    
    print('SkANFIS Online')
    
    fs = FS()
    
    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x0", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 0"))

    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x1", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 1"))
    
    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x2", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 2"))

    S_1 = GaussianFuzzySet(mu=-5, sigma=2.5, term="mf0")
    S_2 = GaussianFuzzySet(mu=0, sigma=2.5, term="mf1")
    S_3 = GaussianFuzzySet(mu=5, sigma=2.5, term="mf2")
    fs.add_linguistic_variable("x3", LinguisticVariable([S_1, S_2, S_3], concept="Input Variable 3"))

    mf_indices = list(itertools.product(range(3), repeat=4))
    rules = []
    for i, p in enumerate(mf_indices):
        antecedent = " AND ".join([f"(x{j} IS mf{p[j]})" for j in range(4)])
        rules.append(f"IF {antecedent} THEN (y0 IS term_{i})")
    fs.add_rules(rules)

    model = scikit_anfis(fs, epoch=100, hybrid=False, label='c')

    start = timeit.default_timer()
    model.fit(x_train.numpy(), y_train.numpy()) 
    end = timeit.default_timer()

    start_inference = timeit.default_timer()
    y_pred = model.predict(x_test.numpy())
    end_inference = timeit.default_timer()

    skanfis_online_time.append(end - start)
    skanfis_online_inference.append(end_inference - start_inference) 
    skanfis_online_acc.append(accuracy_score(y_test.numpy(), y_pred))
    skanfis_online_f1.append(f1_score(y_test.numpy(), y_pred, average='weighted'))

np.save('anfispy_time_iris.npy', np.array(anfispy_time))
np.save('anfispy_inference_iris.npy', np.array(anfispy_inference))
np.save('anfispy_acc_iris.npy', np.array(anfispy_acc))
np.save('anfispy_f1_iris.npy', np.array(anfispy_f1))

np.save('skanfis_hybrid_time_iris.npy', np.array(skanfis_time))
np.save('skanfis_hybrid_inference_iris.npy', np.array(skanfis_inference))
np.save('skanfis_hybrid_acc_iris.npy', np.array(skanfis_acc))
np.save('skanfis_hybrid_f1_iris.npy', np.array(skanfis_f1))

np.save('skanfis_online_time_iris.npy', np.array(skanfis_online_time))
np.save('skanfis_online_inference_iris.npy', np.array(skanfis_online_inference))
np.save('skanfis_online_acc_iris.npy', np.array(skanfis_online_acc))
np.save('skanfis_online_f1_iris.npy', np.array(skanfis_online_f1))