
import scapy.all as scapy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque
import time
import threading

# Define Deep Learning Model for Intrusion Detection
class NIDSModel(nn.Module):
    def __init__(self, input_dim):
        super(NIDSModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Feature Extraction from Network Packets
def extract_features(packet):
    if packet.haslayer(scapy.IP):
        ip_layer = packet.getlayer(scapy.IP)
        features = [
            ip_layer.src, ip_layer.dst, ip_layer.len, ip_layer.ttl
        ]
    else:
        features = [0, 0, 0, 0]

    if packet.haslayer(scapy.TCP):
        tcp_layer = packet.getlayer(scapy.TCP)
        features.extend([
            tcp_layer.sport, tcp_layer.dport, tcp_layer.seq, tcp_layer.ack
        ])
    else:
        features.extend([0, 0, 0, 0])

    return features

# Preprocess Data for Model
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Live Network Monitoring
def network_monitor(model, interface="eth0"):
    print("Starting real-time network monitoring...")
    packet_queue = deque(maxlen=100)

    def process_packets(packet):
        features = extract_features(packet)
        packet_queue.append(features)
        if len(packet_queue) == 100:
            input_data = preprocess_data(np.array(packet_queue))
            predictions = model(torch.tensor(input_data, dtype=torch.float32))
            if predictions[:, 1].mean() > 0.5:  # Intrusion detected
                print("ALERT! Possible Intrusion Detected!")

    scapy.sniff(iface=interface, prn=process_packets, store=False)

# Train Model on KDD Cup 99 Dataset
def train_nids_model():
    print("Training Intrusion Detection Model...")
    dataset = pd.read_csv("kddcup.data_10_percent.csv", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1].apply(lambda x: 1 if x != "normal." else 0)  # Binary Classification

    X = preprocess_data(X)
    X_train, y_train = torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)

    model = NIDSModel(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "nids_model.pth")
    print("Model Training Complete.")

if __name__ == "__main__":
    train_nids_model()
    model = NIDSModel(41)  # Adjust input dimension based on dataset
    model.load_state_dict(torch.load("nids_model.pth"))
    model.eval()

    monitoring_thread = threading.Thread(target=network_monitor, args=(model,))
    monitoring_thread.start()
