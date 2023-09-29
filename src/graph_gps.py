```python
import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv

class GraphGPS(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphGPS, self).__init__()
        self.conv1 = GraphConv(num_features, 128)
        self.conv2 = GraphConv(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = torch.mean(x, dim=0)
        x = self.classifier(x)

        return x

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
        preds = model(data)
    return criterion(preds[data.val_mask], data.y[data.val_mask]).item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data).max(dim=1)[1]
    return (preds[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

def run_model(data):
    model = GraphGPS(data.num_features, data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        train_loss = train(model, data, optimizer, criterion)
        val_loss = validate(model, data, criterion)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc}')

    model_parameters = model.state_dict()
    model_performance_metrics = {'Train Loss': train_loss, 'Val Loss': val_loss, 'Test Accuracy': test_acc}

    return model, model_parameters, model_performance_metrics
```