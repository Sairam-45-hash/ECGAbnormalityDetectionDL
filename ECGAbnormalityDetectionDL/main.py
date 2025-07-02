import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from preprocess import load_and_segment_ecg_data
from model import RhythmTCN_GRUAttNet
from train import train_model
from evaluate import evaluate_model, plot_confusion, print_summary
from explainability import extract_attention_weights, plot_ecg_with_attention
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from preprocess import load_and_segment_ecg_data
from model import RhythmTCN_GRUAttNet
from train import train_model

from preprocess import load_and_segment_ecg_data
from model import RhythmTCN_GRUAttNet
from train import train_model

# Step 1: Load and preprocess data
data_dir = r'C:\Users\krake\Downloads\mizoram\mit-bih-arrhythmia-database-1.0.0'
segments, labels = load_and_segment_ecg_data(data_dir)

# Step 2: Encode string labels to integers
unique_classes = sorted(set(labels))
class_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
labels = np.array([class_to_idx[label] for label in labels])

# Step 3: Prepare dataset and dataloaders
X_tensor = torch.tensor(segments, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Step 4: Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RhythmTCN_GRUAttNet(input_size=360, num_classes=len(unique_classes)).to(device)

# Step 5: Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10)

# Step 6: Evaluate the model
class_names = unique_classes
report, conf_matrix, roc_auc = evaluate_model(model, val_loader, class_names)
print_summary(report, roc_auc)
plot_confusion(conf_matrix, class_names)

# Step 7: Visualize explainability for a sample
sample_index = 0
sample_ecg = X_tensor[sample_index].unsqueeze(0).to(device)  # Add batch dim
att_weights = extract_attention_weights(model, sample_ecg)
plot_ecg_with_attention(sample_ecg.squeeze().cpu().numpy(), att_weights, title="Attention Map on ECG Signal")
