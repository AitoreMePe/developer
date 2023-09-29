```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

# Load data
from data_loader import load_data
match_data = load_data()

# Preprocess data
from preprocessing import preprocess_data
preprocessed_data = preprocess_data(match_data)

class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transformer, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels)
        self.conv2 = GATConv(out_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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

def run_model(data):
    model = Transformer(data.num_node_features, data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train(model, data, optimizer)
        val_acc = validate(model, data)
        print('Epoch: {:03d}, Val Acc: {:.5f}'.format(epoch, val_acc))

    test_acc = test(model, data)
    print('Test Acc: {:.5f}'.format(test_acc))

    return model, model.parameters(), {'val_acc': val_acc, 'test_acc': test_acc}

model, model_parameters, model_performance_metrics = run_model(preprocessed_data)
```