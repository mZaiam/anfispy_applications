import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import timeit

from ANFISpy import ANFIS
from skanfis import scikit_anfis
from skanfis.fs import *

np.random.seed(42)
torch.manual_seed(42)

x_train_tensor = torch.load('x_train.pth')
x_test_tensor = torch.load('x_test.pth')
y_train_tensor = torch.load('y_train.pth')
y_test_tensor = torch.load('y_test.pth')

X_full = torch.cat((x_train_tensor, x_test_tensor), dim=0)
Y_full = torch.cat((y_train_tensor, y_test_tensor), dim=0)

def rmse(x, y):
    y = y.to(x.device, dtype=x.dtype)
    return torch.sqrt(((x - y)**2).mean())

def mae(x, y):
    y = y.to(x.device, dtype=x.dtype)
    return torch.abs(x - y).mean()

N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

anfispy_time, anfispy_inference, anfispy_rmse, anfispy_mae = [], [], [], []
skanfis_time, skanfis_inference, skanfis_rmse, skanfis_mae = [], [], [], []
skanfis_online_time, skanfis_online_inference, skanfis_online_rmse, skanfis_online_mae = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
    print(f"FOLD {fold + 1}")

    x_train, y_train = X_full[train_idx], Y_full[train_idx]
    x_test, y_test = X_full[test_idx], Y_full[test_idx]

    print('ANFISpy')
    
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
            optimizer.zero_grad()
            y_pred = anfis(x_batch)
            loss = criterion(y_pred, y_batch.flatten())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
    end = timeit.default_timer()
    
    start_inference = timeit.default_timer()
    with torch.no_grad():
        anfis.eval()
        y_pred = anfis(x_test).squeeze()
    end_inference = timeit.default_timer()
    
    print(end - start)
    
    anfispy_time.append(end - start)
    anfispy_inference.append(end_inference - start_inference) 
    anfispy_rmse.append(rmse(y_pred.cpu(), y_test.squeeze()).item())
    anfispy_mae.append(mae(y_pred.cpu(), y_test.squeeze()).item())
    
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

    rules = [f"IF (x0 IS mf{i}) AND (x1 IS mf{j}) THEN (y0 IS term_{i}_{j})" for i in range(3) for j in range(3)]
    fs.add_rules(rules)

    model = scikit_anfis(fs, description="Sinc", epoch=100, hybrid=True)

    start = timeit.default_timer()

    model.fit(x_train.numpy(), y_train.numpy().squeeze()) 

    end = timeit.default_timer()

    start_inference = timeit.default_timer()
    y_pred = model.predict(x_test.numpy())
    end_inference = timeit.default_timer()

    skanfis_time.append(end - start)
    skanfis_inference.append(end_inference - start_inference) 
    y_pred_tensor = torch.from_numpy(y_pred)
    skanfis_rmse.append(rmse(y_pred_tensor, y_test).item())
    skanfis_mae.append(mae(y_pred_tensor, y_test).item())

    print(end - start)
    
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

    rules = [f"IF (x0 IS mf{i}) AND (x1 IS mf{j}) THEN (y0 IS term_{i}_{j})" for i in range(3) for j in range(3)]
    fs.add_rules(rules)

    model = scikit_anfis(fs, description="Sinc", epoch=100, hybrid=False)

    start = timeit.default_timer()

    model.fit(x_train.numpy(), y_train.numpy().squeeze()) 

    end = timeit.default_timer()

    start_inference = timeit.default_timer()
    y_pred = model.predict(x_test.numpy())
    end_inference = timeit.default_timer()

    skanfis_online_time.append(end - start)
    skanfis_online_inference.append(end_inference - start_inference) 
    y_pred_tensor = torch.from_numpy(y_pred)
    skanfis_online_rmse.append(rmse(y_pred_tensor, y_test).item())
    skanfis_online_mae.append(mae(y_pred_tensor, y_test).item())

    print(end - start)
    
np.save('anfispy_time_sinc.npy', np.array(anfispy_time))
np.save('anfispy_inference_sinc.npy', np.array(anfispy_inference))
np.save('anfispy_rmse_sinc.npy', np.array(anfispy_rmse))
np.save('anfispy_mae_sinc.npy', np.array(anfispy_mae))

np.save('skanfis_time_sinc.npy', np.array(skanfis_time))
np.save('skanfis_inference_sinc.npy', np.array(skanfis_inference))
np.save('skanfis_rmse_sinc.npy', np.array(skanfis_rmse))
np.save('skanfis_mae_sinc.npy', np.array(skanfis_mae))

np.save('skanfis_online_time_sinc.npy', np.array(skanfis_online_time))
np.save('skanfis_online_inference_sinc.npy', np.array(skanfis_online_inference))
np.save('skanfis_online_rmse_sinc.npy', np.array(skanfis_online_rmse))
np.save('skanfis_online_mae_sinc.npy', np.array(skanfis_online_mae))