```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class SubgraphGNNs(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SubgraphGNNs, self).__init__()

        self.conv1 = GraphConv(num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(self.linear(x), dim=-1)

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
    correct = float(pred.eq(data.y).sum().item())
    return correct / len(data.y)

def test(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
    correct = float(pred.eq(data.y).sum().item())
    return correct / len(data.y)

def run_model(data, num_features, num_classes):
    model = SubgraphGNNs(num_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 201):
        loss = train(model, optimizer, data)
        if epoch % 10 == 0:
            val_acc = validate(model, data)
            print('Epoch: {:03d}, Loss: {:.5f}, Validation Accuracy: {:.5f}'.format(epoch, loss, val_acc))

    test_acc = test(model, data)
    print('Test Accuracy: {:.5f}'.format(test_acc))

    return model, model.parameters(), {'loss': loss, 'val_acc': val_acc, 'test_acc': test_acc}
```