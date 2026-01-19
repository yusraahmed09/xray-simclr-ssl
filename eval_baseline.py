import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# --- CONFIGURATION ---
FEW_SHOT_CSV = "ecm_train.csv" # CSV file containing 20 training examples (few-shot set)
TEST_CSV = "ecm_test.csv" # CSV file containing the larger, balanced test set
DATA_ROOT = "."                             # Root directory for image files
EPOCHS = 50                                 # Number of epochs to train the linear head
BATCH_SIZE = 10                             # Small batch size for few-shot training
LEARNING_RATE = 1e-3                        # High learning rate typical for linear heads
NUM_CLASSES = 2                             # Binary classification (e.g., Cardiomegaly vs. Not Cardiomegaly)

# --- 1. DOWNSTREAM DATASET (Reads the 20 examples and their labels) ---

class DownstreamDataset(Dataset):
    """Dataset class for loading image paths and corresponding labels from a CSV."""
    def __init__(self, csv_file, root_dir, transform=None):
        # Load the CSV file containing image paths and labels
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx] # Fetch the row data by numerical index
        
        img_path = row['Path'] # Retrieve the image file path
        
        # Load the image and convert it to 3-channel RGB (as required by ResNet)
        image = Image.open(img_path).convert('RGB')
        
        # Get the label value (assuming the second column is the binary label)
        label = row[row.index[1]] # Accesses the value of the second column by its name
        
        if self.transform:
            image = self.transform(image)
            
        # Returns the processed image tensor and the integer label (implicitly cast by DataLoader)
        return image, label

# --- 2. THE MAIN EVALUATION LOGIC ---

def run_evaluation(model, train_loader, test_loader, device):
    """Main training loop for linear probing (training only the classification head)."""
    criterion = nn.CrossEntropyLoss()
    # Optimizer targets ONLY the new 'fc' layer parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    print("--- Starting Linear Probing ---")
    for epoch in range(1, EPOCHS + 1):
        model.train() # Set model to training mode
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # Clear gradients from the previous step
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Quick validation check every 10 epochs (Early Stopping check)
        if epoch % 10 == 0 or epoch == EPOCHS:
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

def evaluate(model, test_loader, device):
    """Calculates accuracy on the test set."""
    model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculations to save memory and speed up inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Get the class with the highest probability
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- IMAGE TRANSFORMS (Simple for Evaluation) ---
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize images to standard ResNet input size
        transforms.ToTensor(),
        # Normalize using standard ImageNet statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 1. MODEL SETUP (The SimCLR Hypothesis Test) ---
    
    # Load ResNet-50 structure with no weights (starting from random initialization for this test)
    model = models.resnet50(weights=None) 
    
    # Freeze ALL layers in the encoder (Linear Probing)
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the number of features entering the final layer (2048 for ResNet-50)
    num_ftrs = model.fc.in_features
    # Replace the final classification layer with a new, trainable one (2 classes)
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Only the new model.fc parameters are trainable!
    model = model.to(device)

    # --- 2. DATA LOADING ---
    train_ds = DownstreamDataset(FEW_SHOT_CSV, DATA_ROOT, eval_transform)
    test_ds = DownstreamDataset(TEST_CSV, DATA_ROOT, eval_transform) 

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # --- 3. RUN EVALUATION ---
    run_evaluation(model, train_loader, test_loader, device)