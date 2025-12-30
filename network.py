import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class GoRankResNet(nn.Module):
    """
    A ResNet-based model for ranking Go positions.
    Input: (B, 500, 19, 19)
    Output: (B, 20) - scores for 20 classes
    """
    def __init__(self, in_channels=500, num_channels=256, num_blocks=20, num_classes=20):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.conv_in2 = nn.Conv2d(num_channels, 6, kernel_size=1)
        self.bn_in2 = nn.BatchNorm2d(6)
        self.fc = nn.Linear(6 * 19 * 19, 2 * 19 * 19)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(2 * 19 * 19, num_classes)

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        out = F.relu(self.bn_in2(self.conv_in2(out))) 
        out = out.view(out.size(0), -1) # Flatten
        out = F.relu(self.fc(out))
        out = self.dropout(out)  # Apply dropout before final layer
        out = self.fc_2(out)

        return out

class GoPlayerResNet(nn.Module):
    """
    A ResNet-based model to generate player embeddings.
    Input: (B, 17, 19, 19)
    Output: (B, 128) L2-normalized embedding
    """
    def __init__(self, in_channels=17, num_channels=128, num_blocks=16, embed_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Linear(num_channels // 2, embed_dim)
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        out = self.avg_pool(out) # Pooling        
        out = out.view(out.size(0), -1) # Flatten
        out = self.mlp_head(out) # MLP Head
        out = F.normalize(out, p=2, dim=1) # L2 Normalize the embedding (crucial for metric learning)
        return out