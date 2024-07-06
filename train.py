import torch
from torch_geometric.data import DataLoader, Dataset
from model import TransAm

# Hyperparameters
hidden_channels = 100
output_channels = 1
num_heads = 1

# Dataset loading
train_data = Dataset()
train_subset = tensor_dataset0.data_list[:int(len(tensor_dataset0.data_list) * 0.8)]
train_data.add_data(train_subset)

val_data = Dataset()
val_subset = tensor_dataset0.data_list[int(len(tensor_dataset0.data_list) * 0.8):]
val_data.add_data(val_subset)

# Training settings
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransAm().to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for data in train_data.data_list:
        data = data.to(device)
        optimizer.zero_grad()
        x, y, edge_index = data.x, data.y, data.edge_index
        output = model(x)
        loss = criterion(output, y[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_data.data_list)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')
