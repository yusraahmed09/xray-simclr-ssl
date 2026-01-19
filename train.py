import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import CheXpertPretrainDataset
from model import SimCLRModel                
from torch.amp import autocast, GradScaler
import os
import time

# NT-XENT LOSS FUNCTION (Normalized Temperature-scaled Cross-Entropy)

class NTXentLoss(nn.Module):
    """
    Standard, robust implementation of NT-Xent Loss that uses the full 2N x 2N 
    similarity matrix for stability.
    """
    def __init__(self, device, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        
        N = z_i.size(0) # Physical batch size
        z = torch.cat((z_i, z_j), dim=0) # Concatenate views into [2N, D] tensor

        
        # Calculate pairwise cosine similarity S[i, j] = cos(z_i, z_j) / tau
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # Create target indices (labels) for the CrossEntropyLoss
        # For row i, the positive pair is at index N+i, and vice versa.
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(self.device)

        # Mask out the diagonal (self-comparison) for numerical stability
        # The values are set to -inf so they do not contribute to the softmax
        logits = sim.masked_fill(torch.eye(2 * N).bool().to(self.device), float('-inf'))
        
        
        # Calculate Cross-Entropy Loss
        loss = self.criterion(logits, labels)
        
        # Average Loss over the 2N inputs
        loss /= (2 * N)
        
        return loss

# --- 2. TRAINING FUNCTION ---

def train(model, train_loader, criterion, optimizer, device, scaler, log_interval=100):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (view_1, view_2) in enumerate(train_loader):
        
        # Zero gradients before the forward pass
        optimizer.zero_grad()
        
        # Automatic Mixed Precision (AMP) context
        with autocast(device_type = 'cuda'):
            # --- CRITICAL SIMCLR STEP ---
            # Concatenate the two views along the batch dimension
            inputs = torch.cat([view_1, view_2], dim=0).to(device) 
            
            # Forward pass: Encode images and project features (z)
            projections = model(inputs) 
            
            # Split features back into two BATCH_SIZE groups
            z_i = projections[:projections.size(0) // 2]
            z_j = projections[projections.size(0) // 2:]
            
            # Calculate NT-Xent Loss
            loss = criterion(z_i, z_j)
            
            
        # Scale the loss and call backward() for mixed precision stability
        scaler.scale(loss).backward()
        
        # Update the model parameters
        scaler.step(optimizer)
        
        # Update the scale factor for the next iteration
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} \tLoss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    epoch_duration = time.time() - start_time
    return avg_loss, epoch_duration

# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    # --- HYPERPARAMETERS ---
    CSV_PATH = "./train.csv"
    DATA_ROOT = "."
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.5
    NUM_WORKERS = 8 # High number for HPC parallelism
    
    scaler = GradScaler('cuda') # Initialize gradient scaler for AMP

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data Loading
    dataset = CheXpertPretrainDataset(csv_file=CSV_PATH, root_dir=DATA_ROOT)
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True 
    )

    # 3. Model, Loss, and Optimizer
    model = SimCLRModel(out_dim=128).to(device)

    # NT-Xent Loss requires the effective batch size (2 * physical batch size)
    criterion = NTXentLoss(device, BATCH_SIZE, TEMPERATURE) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting Pre-training...")
    for epoch in range(1, EPOCHS + 1):
        loss, duration = train(model, train_loader, criterion, optimizer, device, scaler)
        
        print(f"\n===== Epoch {epoch}/{EPOCHS} Done | Loss: {loss:.4f} | Time: {duration:.2f}s =====\n")
        
        # --- CHECKPOINTING (Crucial for HPC) ---
        
        # Save the ENTIRE model state (including epoch, optimizer, loss)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.encoder.state_dict(), # We only save the encoder!
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Overwrite the latest checkpoint file for easy resumption
        torch.save(checkpoint, "latest_checkpoint.pth")
        print(f"--- Saved Checkpoint at Epoch {epoch} ---")
        
        # Save a permanent milestone every 50 epochs
        if epoch%50 == 0:
          torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
          print(f"--- Saved Checkpoint separately at Epoch {epoch} ---")