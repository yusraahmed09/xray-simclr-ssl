import torch.nn as nn
import torchvision.models as models

class SimCLRModel(nn.Module):
    """
    Implements the two-part architecture required for SimCLR:
    1. Encoder (f): The ResNet backbone.
    2. Projection Head (g): The MLP used to compute the contrastive loss.
    """
    def __init__(self, base_model_name: str = 'resnet50', out_dim: int = 128):
      """
      Args:
          base_model_name: Name of the torchvision model (e.g., 'resnet50').
          out_dim: The dimensionality of the final contrastive vector (z).
      """
      super().__init__()

      if base_model_name == 'resnet50':
          # Start from random weights (weights=None) as this is self-supervised
          self.encoder = models.resnet50(weights=None)
      else:
          raise NotImplementedError("Only ResNet-50 is implemented in this example.")

      # Get the feature dimension BEFORE the ResNet's default classification layer
      feat_dim = self.encoder.fc.in_features  # 2048 for ResNet-50
      
      # Remove the default classifier layer (nn.Linear)
      self.encoder.fc = nn.Identity() # Replaces the classification layer with a pass-through

      # This small MLP maps the features (h) to the contrastive space (z)
      self.projector = nn.Sequential(
          nn.Linear(feat_dim, feat_dim),  # Input: 2048 features
          nn.ReLU(),
          nn.Linear(feat_dim, out_dim)    # Output: 128 features (the final contrastive vector)
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # 1. Pass input through the encoder to get representations (h)
      features = self.encoder(x)
      
      # 2. Pass representations through the projector to get projection (z)
      projection = self.projector(features)
      
      # Return z for the contrastive loss calculation
      return projection