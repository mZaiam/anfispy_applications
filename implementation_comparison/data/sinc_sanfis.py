import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import timeit

from sanfis import SANFIS

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

sanfis_time, sanfis_inference, sanfis_rmse, sanfis_mae = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
    print(f"FOLD {fold + 1}")

    x_train, y_train = X_full[train_idx], Y_full[train_idx]
    x_test, y_test = X_full[test_idx], Y_full[test_idx]

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
    
    model = SANFIS(membfuncs=MEMBFUNCS, n_input=2, scale='None', to_device='cpu')
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
    
    start_inference = timeit.default_timer()
    with torch.no_grad():
        model.eval()
        y_pred = model.predict(x_test)
    end_inference = timeit.default_timer()
    
    print(end - start)
    
    sanfis_time.append(end - start)
    sanfis_inference.append(end_inference - start_inference) 
    sanfis_rmse.append(rmse(y_pred.squeeze(), y_test.squeeze()).item())
    sanfis_mae.append(mae(y_pred.squeeze(), y_test.squeeze()).item())
    
np.save('sanfis_time_sinc.npy', np.array(sanfis_time))
np.save('sanfis_inference_sinc.npy', np.array(sanfis_inference))
np.save('sanfis_rmse_sinc.npy', np.array(sanfis_rmse))
np.save('sanfis_mae_sinc.npy', np.array(sanfis_mae))
