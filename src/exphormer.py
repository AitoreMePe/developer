```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class ExphormerConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ExphormerConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        weight = 1 / degree(edge_index[0], size[0], dtype=x_j.dtype)
        return x_j * weight.view(-1, 1)

    def update(self, aggr_out):
        return self.lin_self(aggr_out)

class Exphormer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Exphormer, self).__init__()
        self.conv1 = ExphormerConv(in_channels, 128)
        self.conv2 = ExphormerConv(128, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
        loss = criterion(pred[data.val_mask], data.y[data.val_mask])
        correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        acc = correct / data.val_mask.sum().item()
    return loss.item(), acc

def test(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
    return acc

model = Exphormer(1, 1)
model_parameters = model.parameters()
```