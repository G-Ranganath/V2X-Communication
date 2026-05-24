import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle

print("Loading Intersection Dataset...")
df = pd.read_csv('intersection_training_data.csv')

# --- 1. Data Preprocessing ---
# We now have 6 features for the AI to study
features = ['x', 'y', 'speed', 'dist_to_center', 'dist_to_closest_veh', 'time_to_brake']
X_raw = df[features].values
y_beam_raw = df['best_beam'].values
y_action_raw = df['action_label'].values

# Scale the physics data between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Save the scaler so we can use it later in the live simulation!
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# --- 2. Create Sequences (The "Memory" of the LSTM) ---
SEQUENCE_LENGTH = 5

def create_sequences(X, y_b, y_a, seq_length, veh_ids):
    xs, ys_b, ys_a = [], [], []
    unique_vehicles = np.unique(veh_ids)
    
    for vid in unique_vehicles:
        idx = np.where(veh_ids == vid)[0]
        v_X = X[idx]
        v_y_b = y_b[idx]
        v_y_a = y_a[idx]
        
        for i in range(len(v_X) - seq_length):
            xs.append(v_X[i:(i + seq_length)])
            ys_b.append(v_y_b[i + seq_length])
            ys_a.append(v_y_a[i + seq_length])
            
    return np.array(xs), np.array(ys_b), np.array(ys_a)

veh_ids = df['veh_id'].values
X_seq, y_beam_seq, y_action_seq = create_sequences(X_scaled, y_beam_raw, y_action_raw, SEQUENCE_LENGTH, veh_ids)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_beam_tensor = torch.tensor(y_beam_seq, dtype=torch.long)
y_action_tensor = torch.tensor(y_action_seq, dtype=torch.float32).unsqueeze(1) # Action needs shape [batch, 1]

# Create PyTorch Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_b, y_a):
        self.X, self.y_b, self.y_a = X, y_b, y_a
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_b[idx], self.y_a[idx]

dataloader = DataLoader(MultiTaskDataset(X_tensor, y_beam_tensor, y_action_tensor), batch_size=64, shuffle=True)

# --- 3. Build the Two-Headed LSTM Architecture ---
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_beams):
        super(MultiTaskLSTM, self).__init__()
        # The Main Brain
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Head 1: Telecom (Predicts 1 out of 32 Beams)
        self.fc_beam = nn.Linear(hidden_size, num_beams)
        
        # Head 2: Driving (Predicts a probability between 0 and 1 for STOP/GO)
        self.fc_action = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] # Grab the last time step
        
        # Branch out to both heads
        beam_pred = self.fc_beam(last_step)
        action_pred = self.fc_action(last_step)
        return beam_pred, action_pred

# Initialize model (6 input features)
model = MultiTaskLSTM(input_size=6, hidden_size=64, num_beams=32)

# --- 4. Define Losses and Optimizer ---
# We need TWO loss functions because we have two different types of tasks
criterion_beam = nn.CrossEntropyLoss() # For multi-class (0-31)
criterion_action = nn.BCEWithLogitsLoss() # For binary (0 or 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# --- 5. The Training Loop ---
num_epochs = 80
print("\nStarting Multi-Task Training...")

for epoch in range(num_epochs):
    epoch_loss = 0
    correct_beams, correct_actions, total = 0, 0, 0
    
    for batch_X, batch_y_beam, batch_y_action in dataloader:
        optimizer.zero_grad()
        
        # Get both predictions
        pred_beam, pred_action = model(batch_X)
        
        # Calculate loss for both tasks and add them together!
        loss_beam = criterion_beam(pred_beam, batch_y_beam)
        loss_action = criterion_action(pred_action, batch_y_action)
        total_loss = loss_beam + loss_action
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        
        # --- Calculate Accuracies ---
        total += batch_y_beam.size(0)
        
        # Beam Accuracy
        _, predicted_b = torch.max(pred_beam.data, 1)
        correct_beams += (predicted_b == batch_y_beam).sum().item()
        
        # Action Accuracy (Convert logits to 0 or 1)
        predicted_a = (torch.sigmoid(pred_action) > 0.5).float()
        correct_actions += (predicted_a == batch_y_action).sum().item()
        
    if (epoch+1) % 10 == 0:
        avg_loss = epoch_loss / len(dataloader)
        acc_beam = (correct_beams / total) * 100
        acc_action = (correct_actions / total) * 100
        print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Beam Acc: {acc_beam:.1f}% | Action Acc: {acc_action:.1f}%')

print("\nTraining Complete!")
torch.save(model.state_dict(), 'intersection_multitask_lstm.pth')
print("Model saved as 'intersection_multitask_lstm.pth'")
