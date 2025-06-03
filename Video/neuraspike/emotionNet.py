"""
Custom Convolutional Neural Network for Emotion Recognition

This module implements a lightweight CNN architecture specifically designed for
facial emotion recognition. The network uses modern deep learning techniques
including ELU activation functions, batch normalization, and dropout for
regularization.

Architecture Overview:
- Input: Grayscale images (1 channel) or RGB images (3 channels)
- Convolutional layers with increasing filter sizes: 32 → 64 → 128
- MaxPooling layers for spatial dimension reduction
- Fully connected classifier with dropout for regularization
- Output: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)

Author: CS55100 Student
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionNet(nn.Module):
    """
    Custom CNN architecture for emotion recognition.
    
    This network is designed to be lightweight yet effective for real-time
    emotion detection applications. It uses a VGG-like structure with
    modern activation functions and regularization techniques.
    
    Attributes:
        network_config (list): Configuration for convolutional layers
        features (nn.Sequential): Convolutional feature extractor
        classifier (nn.Sequential): Fully connected classifier
    """
    
    # Network architecture configuration
    # Numbers represent filter sizes, 'M' represents MaxPooling
    network_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

    def __init__(self, num_of_channels, num_of_classes):
        """
        Initialize the EmotionNet architecture.
        
        Args:
            num_of_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_of_classes (int): Number of output emotion classes (typically 7)
            
        Note:
            The network automatically adjusts the first convolutional layer
            to accept the specified number of input channels.
        """
        super(EmotionNet, self).__init__()
        
        # Build the convolutional feature extractor
        self.features = self._make_layers(num_of_channels, self.network_config)
        
        # Build the classifier (fully connected layers)
        # Input size: 6x6x128 (after 3 max pooling operations on 48x48 input)
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 128, 64),  # Flatten and reduce to 64 features
            nn.ELU(True),                 # Exponential Linear Unit activation
            nn.Dropout(p=0.5),            # Dropout for regularization
            nn.Linear(64, num_of_classes) # Final classification layer
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
            
        Note:
            The forward pass includes:
            1. Feature extraction through convolutional layers
            2. Spatial flattening for the classifier
            3. Dropout during training for regularization
            4. Final classification output
        """
        # Extract features through convolutional layers
        out = self.features(x)
        
        # Flatten the spatial dimensions for the classifier
        out = out.view(out.size(0), -1)
        
        # Apply dropout during training (regularization)
        out = F.dropout(out, p=0.5, training=self.training)
        
        # Pass through the classifier
        out = self.classifier(out)
        
        return out

    def _make_layers(self, in_channels, cfg):
        """
        Dynamically construct convolutional layers based on configuration.
        
        This method creates a sequence of convolutional layers with the
        specified configuration. It automatically handles the transition
        between layers and applies appropriate activation functions.
        
        Args:
            in_channels (int): Number of input channels for the first layer
            cfg (list): Configuration list defining layer structure
            
        Returns:
            nn.Sequential: Sequential container of all convolutional layers
            
        Architecture Details:
            - Convolutional layers: 3x3 kernels with padding=1 to maintain spatial dimensions
            - Batch Normalization: Applied after each convolution for training stability
            - ELU Activation: Exponential Linear Unit for better gradient flow
            - MaxPooling: 2x2 pooling with stride 2 for dimension reduction
        """
        layers = []
        
        for x in cfg:
            if x == 'M':
                # Add MaxPooling layer for spatial dimension reduction
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # Add convolutional block: Conv2d + BatchNorm + ELU
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ELU(inplace=True)
                ]
                # Update input channels for next layer
                in_channels = x
                
        return nn.Sequential(*layers)

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization and analysis.
        
        This method is useful for:
        - Understanding what the network learns at different layers
        - Debugging network behavior
        - Feature visualization and interpretability
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            dict: Dictionary containing feature maps from different layers
        """
        feature_maps = {}
        
        # Extract features from convolutional layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps[f'conv_{i}'] = x
            elif isinstance(layer, nn.MaxPool2d):
                feature_maps[f'pool_{i}'] = x
                
        return feature_maps

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the network.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_emotion_net(num_channels=1, num_classes=7, pretrained=False):
    """
    Factory function to create an EmotionNet instance.
    
    Args:
        num_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        num_classes (int): Number of emotion classes
        pretrained (bool): Whether to load pretrained weights (if available)
        
    Returns:
        EmotionNet: Configured emotion recognition network
        
    Example:
        >>> model = create_emotion_net(num_channels=1, num_classes=7)
        >>> print(f"Total parameters: {model.count_parameters():,}")
        >>> print(f"Model architecture:\n{model}")
    """
    model = EmotionNet(num_channels, num_classes)
    
    if pretrained:
        # Load pretrained weights if available
        # This would typically load weights from a previous training run
        pass
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("EmotionNet Architecture Test")
    print("=" * 40)
    
    # Create model instances for different configurations
    model_grayscale = create_emotion_net(num_channels=1, num_classes=7)
    model_rgb = create_emotion_net(num_channels=3, num_classes=7)
    
    # Test with sample input
    batch_size = 4
    input_grayscale = torch.randn(batch_size, 1, 48, 48)
    input_rgb = torch.randn(batch_size, 3, 48, 48)
    
    # Forward pass
    with torch.no_grad():
        output_grayscale = model_grayscale(input_grayscale)
        output_rgb = model_rgb(input_rgb)
    
    print(f"Grayscale model parameters: {model_grayscale.count_parameters():,}")
    print(f"RGB model parameters: {model_rgb.count_parameters():,}")
    print(f"Grayscale output shape: {output_grayscale.shape}")
    print(f"RGB output shape: {output_rgb.shape}")
    
    # Test feature extraction
    feature_maps = model_grayscale.get_feature_maps(input_grayscale)
    print(f"Number of feature map layers: {len(feature_maps)}")