```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
from data_loader import load_data
match_data = load_data()

# Preprocess data
from preprocessing import preprocess_data
preprocessed_data = preprocess_data(match_data)

class GCANLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCANLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        return F.relu(x)

class GCAN(nn.Module):
    def __init__(self, n_features, n_classes):
        super(GCAN, self).__init__()
        self.layer1 = GCANLayer(n_features, 16)
        self.layer2 = GCANLayer(16, n_classes)

    def forward(self, data):
        x = self.layer1(data)
        x = self.layer2(x)
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
        preds = model(data).max(dim=1)[1]
    return accuracy_score(data.y[data.val_mask].cpu(), preds[data.val_mask].cpu())

def test(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data).max(dim=1)[1]
    return accuracy_score(data.y[data.test_mask].cpu(), preds[data.test_mask].cpu())

def run_model(data, n_features, n_classes):
    model = GCAN(n_features, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion)
        val_acc = validate(model, data)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')

    return model, model.parameters()

model, model_parameters = run_model(preprocessed_data, n_features=preprocessed_data.num_node_features, n_classes=preprocessed_data.y.max().item()+1)

model_performance_metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
}
```