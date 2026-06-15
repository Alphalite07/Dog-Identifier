import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DogEmbedder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load the lightweight ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Strip the final classification layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Attach the metric learning embedding head
        self.embedder = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedder(features)
        # L2 Normalize the vectors for cosine similarity
        return nn.functional.normalize(embeddings, p=2, dim=1)