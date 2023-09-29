```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class EGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(EGNNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

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
        return F.relu(aggr_out)


class EGNN(nn.Module):
    def __init__(self, data, nhid, nclass, dropout):
        super(EGNN, self).__init__()

        self.conv1 = EGNNConv(data.num_node_features, nhid)
        self.conv2 = EGNNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    preds = logits[data.val_mask].max(1)[1]
    correct = preds.eq(data.y[data.val_mask]).sum().item()
    return correct / data.val_mask.sum().item()


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    preds = logits[data.test_mask].max(1)[1]
    correct = preds.eq(data.y[data.test_mask]).sum().item()
    return correct / data.test_mask.sum().item()


def run_model(data, nhid=16, nclass=7, dropout=0.5, lr=0.01, weight_decay=5e-4, epochs=200):
    model = EGNN(data, nhid, nclass, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        loss = train(model, data, optimizer)
        acc = validate(model, data)
        print('Epoch: {:03d}, Loss: {:.5f}, Validation Accuracy: {:.5f}'.format(epoch, loss, acc))

    test_acc = test(model, data)
    print('Test Accuracy: {:.5f}'.format(test_acc))

    return model
```