```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

# Load the shared dependencies
from data_loader import load_data
from preprocessing import preprocess_data

# Load the data
match_data = load_data()

# Preprocess the data
preprocessed_data = preprocess_data(match_data)

class OverSquashingEffect(nn.Module):
    def __init__(self, num_features, num_classes):
        super(OverSquashingEffect, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data).max(1)[1]
    return (preds[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data).max(1)[1]
    return (preds[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

def run_model(data):
    model = OverSquashingEffect(data.num_features, data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion)
        acc = validate(model, data)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')

    return model, model.parameters(), {'loss': loss, 'val_acc': acc, 'test_acc': test_acc}

model, model_parameters, model_performance_metrics = run_model(preprocessed_data)
```