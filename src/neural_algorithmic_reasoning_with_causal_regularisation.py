```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import networkx as nx
from src.data_loader import load_data
from src.preprocessing import preprocess_data

class NeuralAlgorithmicReasoning(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralAlgorithmicReasoning, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_model():
    match_data = load_data()
    preprocessed_data = preprocess_data(match_data)
    X_train, X_val, y_train, y_val = train_test_split(preprocessed_data.drop('win', axis=1), preprocessed_data['win'], test_size=0.2, random_state=42)
    train_data = DataLoader(list(zip(X_train.values, y_train.values)), batch_size=32, shuffle=True)
    val_data = DataLoader(list(zip(X_val.values, y_val.values)), batch_size=32, shuffle=True)
    model = NeuralAlgorithmicReasoning(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        train_loss = train(model, train_data, criterion, optimizer)
        val_loss = validate(model, val_data, criterion)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    test_accuracy = test(model, val_data)
    print(f'Test Accuracy: {test_accuracy}%')

model = run_model()
```