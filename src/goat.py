```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from preprocessing import preprocessed_data

class GOAT(nn.Module):
    def __init__(self):
        super(GOAT, self).__init__()

        # Graph convolution layers
        self.conv1 = GraphConv(preprocessed_data.num_node_features, 128)
        self.conv2 = GraphConv(128, 64)

        # TopKPooling
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.pool2 = TopKPooling(64, ratio=0.8)

        # Dense layers
        self.fc1 = torch.nn.Linear(64*2, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

model = GOAT()

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for data in preprocessed_data:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def validate():
    model.eval()
    correct = 0

    for data in preprocessed_data:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(preprocessed_data)

def test():
    model.eval()
    correct = 0

    for data in preprocessed_data:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(preprocessed_data)

model_parameters = model.parameters()
model_performance_metrics = {'Train': train(), 'Validation': validate(), 'Test': test()}
```