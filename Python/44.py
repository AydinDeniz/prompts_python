
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from collections import OrderedDict

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define client class
class FederatedClient:
    def __init__(self, model, dataset, lr=0.01, epochs=3):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        self.model.train()
        train_loader = data.DataLoader(self.dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.model.state_dict()

# Federated Learning Aggregation
def federated_aggregation(client_weights):
    avg_weights = OrderedDict()
    
    for key in client_weights[0].keys():
        avg_weights[key] = torch.stack([client_weights[i][key] for i in range(len(client_weights))]).mean(dim=0)

    return avg_weights

# Simulate multiple clients
def simulate_federated_learning(num_clients=5, rounds=3):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    client_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
    
    global_model = SimpleNN()
    for round in range(rounds):
        print(f"Round {round + 1} - Federated Training")
        client_weights = []
        for i in range(num_clients):
            client = FederatedClient(SimpleNN(), client_data[i])
            client_weights.append(client.train())

        aggregated_weights = federated_aggregation(client_weights)
        global_model.load_state_dict(aggregated_weights)
    
    print("Federated Learning Complete. Global Model Trained.")
    torch.save(global_model.state_dict(), "federated_model.pth")

if __name__ == "__main__":
    simulate_federated_learning()
