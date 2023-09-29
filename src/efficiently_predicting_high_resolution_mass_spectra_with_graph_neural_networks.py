```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess_data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(preprocessed_data.num_node_features, 16)
        self.conv2 = GCNConv(16, preprocessed_data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(preprocessed_data), preprocessed_data.y).backward()
    optimizer.step()

def test():
    model.eval()
    correct = 0
    for data in preprocessed_data:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(preprocessed_data)

def run_model():
    for epoch in range(1, 201):
        train()
        test_acc = test()
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))

match_data = load_data()
preprocessed_data = preprocess_data(match_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

run_model()

model_parameters = model.parameters()
model_performance_metrics = {'Test Accuracy': test()}
```