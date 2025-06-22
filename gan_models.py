import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Conditional GAN (CGAN) for generating Alzheimer's brain MRI images
class CGAN(nn.Module):
    def __init__(self, latent_dim=100, num_classes=4, img_size=224, channels=3):
        super(CGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Embedding layer for class conditioning
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Initial size for upsampling
        self.init_size = img_size // 16
        self.l1 = nn.Sequential(nn.Linear(latent_dim * 2, 128 * self.init_size ** 2))
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        
        # Project and reshape
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(out)
        return img


# CycleGAN for transforming between normal and Alzheimer's brain MRI images
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)


class CycleGAN(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9, 
                 direction="AB", smaller_network=False, input_size=224):
        super(CycleGAN, self).__init__()
        self.direction = direction  # AB: Normal -> Alzheimer's, BA: Alzheimer's -> Normal
        
        # Use fewer residual blocks and features for smaller network
        if smaller_network:
            n_residual_blocks = 6  # Reduced number of residual blocks
            base_features = 32     # Reduced base feature count (was 64)
        else:
            base_features = 64     # Original feature count
            
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_features, 7),
            nn.InstanceNorm2d(base_features),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = base_features
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_features, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# High-Resolution GAN (HighResGAN) for generating detailed brain MRI images
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Query, Key, Value projections
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out


class HighResGAN(nn.Module):
    def __init__(self, latent_dim=128, num_classes=4, img_size=224, channels=3, smaller_network=False):
        super(HighResGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.smaller_network = smaller_network
        
        # Use smaller feature maps if smaller_network is True
        if smaller_network:
            feature_maps = [256, 128, 64, 32, 16]
        else:
            feature_maps = [1024, 512, 256, 128, 64]
        
        # Embedding layer for class conditioning (optional)
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Initial dense layer and reshape
        self.init_size = img_size // 32
        self.l1 = nn.Linear(latent_dim * 2, feature_maps[0] * self.init_size ** 2)
        
        # Upsampling blocks with residual connections and self-attention
        self.conv_blocks = nn.ModuleList([
            # First block: 7x7 -> 14x14
            nn.Sequential(
                nn.BatchNorm2d(feature_maps[0]),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(feature_maps[0], feature_maps[1], 3, stride=1, padding=1),
                nn.BatchNorm2d(feature_maps[1]),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Second block: 14x14 -> 28x28
            nn.Sequential(
                nn.BatchNorm2d(feature_maps[1]),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(feature_maps[1], feature_maps[2], 3, stride=1, padding=1),
                nn.BatchNorm2d(feature_maps[2]),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Self-attention layer
            SelfAttention(feature_maps[2]),
            # Third block: 28x28 -> 56x56
            nn.Sequential(
                nn.BatchNorm2d(feature_maps[2]),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(feature_maps[2], feature_maps[3], 3, stride=1, padding=1),
                nn.BatchNorm2d(feature_maps[3]),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Fourth block: 56x56 -> 112x112
            nn.Sequential(
                nn.BatchNorm2d(feature_maps[3]),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(feature_maps[3], feature_maps[4], 3, stride=1, padding=1),
                nn.BatchNorm2d(feature_maps[4]),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Fifth block: 112x112 -> 224x224
            nn.Sequential(
                nn.BatchNorm2d(feature_maps[4]),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(feature_maps[4], channels, 3, stride=1, padding=1),
                nn.Tanh()
            )
        ])
        
        # Skip connections matched to feature dimensions
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(feature_maps[0], feature_maps[1], 1, stride=1, padding=0),
            nn.Conv2d(feature_maps[1], feature_maps[2], 1, stride=1, padding=0),
            nn.Conv2d(feature_maps[2], feature_maps[3], 1, stride=1, padding=0),
            nn.Conv2d(feature_maps[3], feature_maps[4], 1, stride=1, padding=0)
        ])

    def forward(self, noise, labels=None):
        # If labels are provided, use conditioning
        if labels is not None:
            embedded_labels = self.label_embedding(labels)
            gen_input = torch.cat((embedded_labels, noise), -1)
        else:
            # If no labels, duplicate noise to match dimensions
            gen_input = torch.cat((noise, noise), -1)
        
        # Initial projection and reshape
        out = self.l1(gen_input)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        
        # Apply upsampling blocks with skip connections
        skip_outputs = []
        
        # First two blocks
        for i in range(2):
            skip_outputs.append(out)
            out = self.conv_blocks[i](out)
        
        # Self-attention
        out = self.conv_blocks[2](out)
        
        # For debugging dimension issues
        for i in range(3, 5):
            try:
                skip_out = skip_outputs.pop()
                # Check dimensions before skip connection
                if self.smaller_network:
                    # Different approach for smaller networks to avoid dimension mismatch
                    out = self.conv_blocks[i](out)
                else:
                    # Apply skip connection only if dimensions allow
                    skip_out = self.skip_connections[i-3](skip_out)
                    skip_out = F.interpolate(skip_out, size=out.shape[2:])
                    out = torch.add(out, skip_out)
                    out = self.conv_blocks[i](out)
            except Exception as e:
                # Gracefully handle dimension issues
                print(f"Skipping connection at layer {i} due to dimension mismatch. Using direct path.")
                out = self.conv_blocks[i](out)
        
        # Final block
        img = self.conv_blocks[5](out)
        
        return img


# Training functions for GANs (to be used in a separate training script)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Discriminator for CycleGAN
class CycleGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, smaller_network=False, input_size=224):
        super(CycleGANDiscriminator, self).__init__()
        
        # Use fewer features for smaller network
        if smaller_network:
            base_features = 32  # Reduced base feature count
        else:
            base_features = 64  # Original feature count
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, base_features, normalize=False),
            *discriminator_block(base_features, base_features * 2),
            *discriminator_block(base_features * 2, base_features * 4),
            *discriminator_block(base_features * 4, base_features * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_features * 8, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img) 