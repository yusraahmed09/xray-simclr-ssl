import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple

# THE AUGMENTATION PIPELINE 

class SimCLRDataTransform:
    """
    Applies the medical-safe augmentation pipeline twice to a single input image, 
    creating the positive pair (View A, View B).
    """
    def __init__(self, input_height: int = 224):
        # Defines the sequence of random transformations
        self.transform = transforms.Compose([
            # 1. Random Resized Crop: Forces local-to-global learning. 
            # Scale 0.2 to 1.0 is safer for anatomical context.
            transforms.RandomResizedCrop(input_height, scale=(0.2, 1.0)),
            
            # 2. Random Horizontal Flip: Feature learning.
            transforms.RandomHorizontalFlip(),
            
            # 3. Color Jitter: Applies random exposure changes.
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, 
                    saturation=0.0, hue=0.0         
                )
            ], p=0.8), # 80% chance of applying this
            
            # 4. Gaussian Blur: Removes high-frequency noise.
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9)
            ], p=0.5), # 50% chance of applying this
            
            transforms.ToTensor(),
            # 5. Normalize: Standardizes pixel values for the neural network.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply the defined transform twice independently
        x_i = self.transform(sample)
        x_j = self.transform(sample)
        return x_i, x_j


# THE DATASET CLASS ---

class CheXpertPretrainDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # Use the dual-transform if none is provided
        self.transform = transform if transform else SimCLRDataTransform()

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the original path from the CSV
        raw_path = self.data_frame.iloc[idx, 0]

        # Remove the CheXpert prefix which doesn't exist in our folder structure
        relative_path = raw_path.replace("CheXpert-v1.0-small/", "")
        relative_path = relative_path.replace("CheXpert-v1.0/", "")
        
        img_path = os.path.join(self.root_dir, relative_path)

        try:
            # Load the single original image and convert to 3-channel RGB
            image = Image.open(img_path).convert('RGB') 
        except (FileNotFoundError, OSError):
            # If the file is missing/corrupted, skip it and load the next one
            return self.__getitem__((idx + 1) % len(self.data_frame))

        # Apply the SimCLR dual-transform and return the two views
        view_1, view_2 = self.transform(image)
        return view_1, view_2