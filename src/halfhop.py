```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class HalfHop(nn.Module):
    def __init__(self, num_features, num_classes):
        super(HalfHop, self).__init__()
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

def test(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data).max(dim=1)[1]
    return predictions

def validate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    return logits

def run_model(data, num_features, num_classes):
    model = HalfHop(num_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train(model, data, optimizer)

    logits = validate(model, data)
    predictions = test(model, data)

    accuracy = accuracy_score(data.y[data.test_mask], predictions[data.test_mask])
    precision = precision_score(data.y[data.test_mask], predictions[data.test_mask], average='micro')
    recall = recall_score(data.y[data.test_mask], predictions[data.test_mask], average='micro')
    f1 = f1_score(data.y[data.test_mask], predictions[data.test_mask], average='micro')

    model_performance_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return model, model.parameters(), model_performance_metrics
```