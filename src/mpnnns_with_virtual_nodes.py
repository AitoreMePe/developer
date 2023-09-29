```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MPNNWithVirtualNodes(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNWithVirtualNodes, self).__init__(aggr='add') 
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.lin_virtualnode = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        aggr_out = self.lin_virtualnode(aggr_out)
        return F.relu(aggr_out)

def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def test(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in data_loader:
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)

def run_model(preprocessed_data):
    model = MPNNWithVirtualNodes(32, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(100):
        train_loss = train(model, preprocessed_data, optimizer, criterion)
        val_loss = validate(model, preprocessed_data, criterion)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    test_acc = test(model, preprocessed_data)
    print(f'Test Accuracy: {test_acc:.4f}')

model = MPNNWithVirtualNodes
model_parameters = model.parameters()
run_model = run_model
```