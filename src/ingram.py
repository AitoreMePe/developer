```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from src.preprocessing import preprocessed_data

class InGram(nn.Module):
    def __init__(self, num_features, num_classes):
        super(InGram, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
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
        predictions = model(data).max(dim=1)[1]
    return accuracy_score(data.y[data.val_mask].cpu(), predictions[data.val_mask].cpu())

def test(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data).max(dim=1)[1]
    return accuracy_score(data.y[data.test_mask].cpu(), predictions[data.test_mask].cpu())

def run_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InGram(preprocessed_data.num_features, preprocessed_data.num_classes).to(device)
    data = preprocessed_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train(model, data, optimizer)
        val_acc = validate(model, data)
        print('Epoch: {:03d}, Val Acc: {:.4f}'.format(epoch, val_acc))

    test_acc = test(model, data)
    print('Test Acc: {:.4f}'.format(test_acc))

model = run_model()
model_parameters = model.parameters()
model_performance_metrics = {'Validation Accuracy': validate(model, preprocessed_data), 'Test Accuracy': test(model, preprocessed_data)}
```