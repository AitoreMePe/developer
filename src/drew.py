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

class DRew(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DRew, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(self.classifier(x), dim=1)

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

def run_model(data):
    model = DRew(data.num_features, data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion)
        val_acc = validate(model, data)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')

run_model(preprocessed_data)
```