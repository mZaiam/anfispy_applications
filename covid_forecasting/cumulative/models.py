import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        size_layers,
        activation,
    ):
        super(MLP, self).__init__()
        
        layers = []
        
        for i in range(len(size_layers) - 1):
            layers.append(
                nn.Linear(in_features=size_layers[i], out_features=size_layers[i + 1]),
            )
            layers.append(activation)
            
        self.model = nn.Sequential(*layers)
                    
    def forward(self, x):
        return self.model(x)
    
class RNN(nn.Module):
    def __init__(self, h=64, activation=nn.Sigmoid()):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,      
            hidden_size=h,    
            batch_first=True   
        )
        self.fc = nn.Linear(h, 1)
        self.activation = activation
    
    def forward(self, x, h=None):
        out, h = self.rnn(x, h)         
        return self.activation(self.fc(out)), h
    
class GRU(nn.Module):
    def __init__(self, h=64, activation=nn.Sigmoid()):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,      
            hidden_size=h,    
            batch_first=True   
        )
        self.fc = nn.Linear(h, 1)
        self.activation = activation
    
    def forward(self, x, h=None):
        out, h = self.gru(x, h)         
        return self.activation(self.fc(out)), h

class LSTM(nn.Module):
    def __init__(self, h=64, activation=nn.Sigmoid()):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=h,
            batch_first=True
        )
        self.fc = nn.Linear(h, 1)
        self.activation = activation
    
    def forward(self, x, h=None):
        out, (h, c) = self.lstm(x, h)
        out = self.fc(out)
        return self.activation(out), (h, c)
    
class BiRNN(nn.Module):
    def __init__(self, h=64, activation=nn.Sigmoid()):
        super().__init__()
        self.h = h
        self.bidirectional = True
        self.rnn = nn.RNN(
            input_size=1,      
            hidden_size=h,   
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.fc = nn.Linear(h, 1)
        self.activation = activation
    
    def forward(self, x, h=None):
        out, h = self.rnn(x, h)  
        if self.bidirectional:
            fwd = out[:, :, :self.h]     
            bwd = out[:, :, self.h:]      
            out = 0.5 * (fwd + bwd)
        out = self.fc(out)
        return self.activation(out), h
    
class BiGRU(nn.Module):
    def __init__(self, h=64, activation=nn.Sigmoid()):
        super().__init__()
        self.h = h
        self.bidirectional = True
        self.gru = nn.GRU(
            input_size=1,      
            hidden_size=h,    
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear(h, 1)
        self.activation = activation
    
    def forward(self, x, h=None):
        out, h = self.gru(x, h)  
        if self.bidirectional:
            fwd = out[:, :, :self.h]     
            bwd = out[:, :, self.h:]      
            out = 0.5 * (fwd + bwd)
        out = self.fc(out)
        return self.activation(out), h
    
class BiLSTM(nn.Module):
    def __init__(self, h=64, activation=nn.Sigmoid()):
        super().__init__()
        self.bidirectional = True
        self.h = h
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=h,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear(h, 1)
        self.activation = activation
    
    def forward(self, x, h=None):
        out, (h, c) = self.lstm(x, h)
        if self.bidirectional:
            fwd = out[:, :, :self.h]     
            bwd = out[:, :, self.h:]      
            out = 0.5 * (fwd + bwd)
        out = self.fc(out)
        return self.activation(out), (h, c)