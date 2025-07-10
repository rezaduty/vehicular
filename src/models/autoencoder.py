"""
Autoencoder architectures for compressing image and LiDAR data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class ImageAutoencoder(nn.Module):
    """Autoencoder for image compression"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 256,
                 compression_ratio: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_ratio = compression_ratio
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1: 384x1280 -> 192x640
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2: 192x640 -> 96x320
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: 96x320 -> 48x160
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: 48x160 -> 24x80
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 5: 24x80 -> 12x40
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 20)),  # 512 x 6 x 20
            nn.Flatten(),
            nn.Linear(512 * 6 * 20, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 6 * 20),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            # Block 1: 12x40 -> 24x80
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 2: 24x80 -> 48x160
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 3: 48x160 -> 96x320
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 96x320 -> 192x640
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 5: 192x640 -> 384x1280
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        """Encode image to latent representation"""
        features = self.encoder(x)
        latent = self.bottleneck(features)
        return latent
    
    def decode(self, latent):
        """Decode latent representation to image"""
        x = self.decoder_fc(latent)
        x = x.view(-1, 512, 6, 20)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class LiDARAutoencoder(nn.Module):
    """Autoencoder for LiDAR point cloud compression"""
    
    def __init__(self,
                 max_points: int = 16384,
                 point_features: int = 4,
                 latent_dim: int = 256):
        super().__init__()
        self.max_points = max_points
        self.point_features = point_features
        self.latent_dim = latent_dim
        
        # Point-wise feature extraction
        self.point_encoder = nn.Sequential(
            nn.Conv1d(point_features, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global feature aggregation
        self.global_encoder = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, max_points * point_features)
        )
        
    def encode(self, points):
        """Encode point cloud to latent representation"""
        # points: [B, N, 4] -> [B, 4, N]
        points = points.transpose(1, 2)
        
        # Point-wise features
        point_features = self.point_encoder(points)
        
        # Global features
        global_features = self.global_encoder(point_features)
        
        return global_features
    
    def decode(self, latent):
        """Decode latent representation to point cloud"""
        points = self.decoder(latent)
        points = points.view(-1, self.max_points, self.point_features)
        return points
    
    def forward(self, points):
        """Forward pass through autoencoder"""
        latent = self.encode(points)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for probabilistic compression"""
    
    def __init__(self,
                 input_channels: int = 3,
                 latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 6 * 20)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        """Encode to latent distribution parameters"""
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space"""
        x = self.decoder_input(z)
        x = x.view(-1, 512, 6, 20)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, target, mu, logvar, beta=1.0):
        """VAE loss function"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss


class ConditionalAutoencoder(nn.Module):
    """Conditional autoencoder that takes class information"""
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 latent_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # Encoder
        self.encoder = ImageAutoencoder(input_channels, latent_dim).encoder
        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 20)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 20 + 64, latent_dim),  # +64 for class embedding
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 64, 512 * 6 * 20),  # +64 for class embedding
            nn.ReLU(inplace=True)
        )
        self.decoder = ImageAutoencoder(input_channels, latent_dim).decoder
        
    def encode(self, x, class_labels):
        """Encode with class conditioning"""
        # Extract features
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # Get class embeddings
        class_emb = self.class_embedding(class_labels)
        
        # Combine features and class information
        combined = torch.cat([features, class_emb], dim=1)
        latent = self.encoder_fc(combined)
        
        return latent
    
    def decode(self, latent, class_labels):
        """Decode with class conditioning"""
        # Get class embeddings
        class_emb = self.class_embedding(class_labels)
        
        # Combine latent and class information
        combined = torch.cat([latent, class_emb], dim=1)
        
        # Decode
        x = self.decoder_fc(combined)
        x = x.view(-1, 512, 6, 20)
        x = self.decoder(x)
        
        return x
    
    def forward(self, x, class_labels):
        """Forward pass with class conditioning"""
        latent = self.encode(x, class_labels)
        reconstruction = self.decode(latent, class_labels)
        return reconstruction, latent 