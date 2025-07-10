"""
Semantic segmentation models for autonomous driving
Includes DeepLabV3+, UNet, and other architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ implementation for semantic segmentation"""
    
    def __init__(self, 
                 num_classes: int = 19,
                 backbone: str = 'resnet50',
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(backbone_channels, 256)
        
        # Decoder
        self.decoder = Decoder(num_classes, 256)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Extract features
        features = self.backbone(x)
        
        # ASPP
        aspp_out = self.aspp(features)
        
        # Decoder
        output = self.decoder(aspp_out, input_shape)
        
        return output


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different atrous rates
        atrous_rates = [6, 12, 18]
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0], 
                              dilation=atrous_rates[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1],
                              dilation=atrous_rates[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2],
                              dilation=atrous_rates[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Different branches
        feat1 = F.relu(self.bn1(self.conv1(x)))
        feat2 = F.relu(self.bn2(self.conv2(x)))
        feat3 = F.relu(self.bn3(self.conv3(x)))
        feat4 = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling branch
        feat5 = self.global_avg_pool(x)
        feat5 = F.relu(self.bn5(self.conv5(feat5)))
        feat5 = F.interpolate(feat5, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        # Concatenate all features
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)
        
        return out


class Decoder(nn.Module):
    """DeepLabV3+ Decoder"""
    
    def __init__(self, num_classes: int, low_level_channels: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x, input_shape):
        # Upsample ASPP output
        x = F.interpolate(x, size=(input_shape[0]//4, input_shape[1]//4), 
                         mode='bilinear', align_corners=True)
        
        # Since we don't have low-level features in this simplified version,
        # we'll just process the upsampled features
        low_level_feat = F.relu(self.bn1(self.conv1(x)))
        
        # Concatenate (in full implementation, this would be with backbone low-level features)
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Final convolutions
        x = self.last_conv(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x


class UNet(nn.Module):
    """U-Net implementation for semantic segmentation"""
    
    def __init__(self, num_classes: int = 19, in_channels: int = 3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._make_decoder_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._make_decoder_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._make_decoder_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._make_decoder_block(128, 64)
        
        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final classification
        out = self.classifier(dec1)
        
        return out 