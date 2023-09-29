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

# Load the preprocessed data
match_data = load_data()
preprocessed_data = preprocess_data(match_data)

class EDGE(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EDGE, self).__init__()
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

def validate(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data).max(dim=1)[1]
    return (preds[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data).max(dim=1)[1]
    return (preds[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

def run_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EDGE(num_features=preprocessed_data.num_features, num_classes=preprocessed_data.num_classes).to(device)
    data = preprocessed_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        train(model, data, optimizer, criterion)
        val_acc = validate(model, data)
        print(f'Epoch: {epoch+1}, Validation Accuracy: {val_acc}')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc}')

model = EDGE
model_parameters = model.parameters()
run_model()
```