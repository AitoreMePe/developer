```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loader import match_data
from preprocessing import preprocessed_data

class GraphicallyStructuredDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphicallyStructuredDiffusionModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def train(model, data, target, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validate(model, data, target):
    model.eval()
    with torch.no_grad():
        output = model(data)
        valid_loss = ((output - target) ** 2).sum().item()
    return valid_loss

def test(model, data, target):
    model.eval()
    with torch.no_grad():
        output = model(data)
        test_loss = ((output - target) ** 2).sum().item()
    return test_loss

def run_model():
    input_dim = preprocessed_data.shape[1]
    hidden_dim = 100
    output_dim = 1
    model = GraphicallyStructuredDiffusionModel(input_dim, hidden_dim, output_dim)

    train_data = torch.from_numpy(preprocessed_data.values).float()
    target = torch.from_numpy(match_data['target'].values).float()

    train(model, train_data, target, epochs=100, learning_rate=0.01)
    valid_loss = validate(model, train_data, target)
    test_loss = test(model, train_data, target)

    model_parameters = model.state_dict()
    model_performance_metrics = {'valid_loss': valid_loss, 'test_loss': test_loss}

    return model, model_parameters, model_performance_metrics

model, model_parameters, model_performance_metrics = run_model()
```