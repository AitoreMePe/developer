```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocessed_data

class VC_Dimension_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VC_Dimension_GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
    data = preprocessed_data
    target = torch.tensor(data['target'].values, dtype=torch.float32)
    data = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)
    model = VC_Dimension_GNN(input_dim=data.shape[1], hidden_dim=50, output_dim=1)
    train(model, train_data, train_target, epochs=100, learning_rate=0.01)
    valid_loss = validate(model, test_data, test_target)
    test_loss = test(model, test_data, test_target)
    model_parameters = model.state_dict()
    model_performance_metrics = {'valid_loss': valid_loss, 'test_loss': test_loss}
    return model, model_parameters, model_performance_metrics

model, model_parameters, model_performance_metrics = run_model()
```