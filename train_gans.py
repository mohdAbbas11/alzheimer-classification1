import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import argparse
import time
from pathlib import Path

# Import GAN models
from gan_models import CGAN, CycleGAN, HighResGAN, CycleGANDiscriminator, weights_init_normal
from test import AlzheimerDataset, categories, device, get_device

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set environment variable to force CPU usage if needed
force_cpu = os.environ.get('FORCE_CPU', '0') == '1'
device = get_device(force_cpu)
print(f"Using device: {device}")

# Create directories for saving models and samples
os.makedirs("models", exist_ok=True)
os.makedirs("samples", exist_ok=True)

# Define transforms for GAN training
gan_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}

# Discriminator for CGAN
class CGANDiscriminator(nn.Module):
    def __init__(self, img_size=224, channels=3, num_classes=4):
        super(CGANDiscriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        self.model = nn.Sequential(
            # Input: (channels) x 224 x 224
            nn.Conv2d(channels + 1, 16, 4, 2, 1),  # 112 x 112
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(16, 32, 4, 2, 1),  # 56 x 56
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 28 x 28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 14 x 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 7 x 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embed labels and reshape to image dimensions
        batch_size, _, h, w = img.shape
        label_embedding = self.label_embedding(labels).view(batch_size, 1, h, w)
        
        # Concatenate image and label embedding
        x = torch.cat((img, label_embedding), dim=1)
        
        # Pass through model
        return self.model(x)



# Discriminator for HighResGAN
class HighResGANDiscriminator(nn.Module):
    def __init__(self, img_size=224, channels=3, num_classes=4):
        super(HighResGANDiscriminator, self).__init__()
        
        # Optional label embedding for conditional discrimination
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1),  # 112 x 112
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsampling layers
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1),  # 56 x 56
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1),  # 28 x 28
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1),  # 14 x 14
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ),
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, 1),  # 7 x 7
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            )
        ])
        
        # Output layers
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels=None):
        batch_size, _, h, w = img.shape
        
        # If labels are provided, use them for conditioning
        if labels is not None:
            label_embedding = self.label_embedding(labels).view(batch_size, 1, h, w)
            x = torch.cat((img, label_embedding), dim=1)
        else:
            # If no labels, add a zero channel
            x = torch.cat((img, torch.zeros(batch_size, 1, h, w, device=img.device)), dim=1)
        
        # Initial convolution
        x = self.conv1(x)
        
        # Downsampling blocks
        for block in self.down_blocks:
            x = block(x)
        
        # Output
        return self.output(x)

# Train CGAN
def train_cgan(data_dir, batch_size=16, n_epochs=200, latent_dim=100, sample_interval=200):
    print("Starting CGAN training...")
    
    # Create dataset and dataloader
    dataset = AlzheimerDataset(data_dir, transform=gan_transforms['train'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize generator and discriminator
    generator = CGAN(latent_dim=latent_dim, num_classes=len(categories)).to(device)
    discriminator = CGANDiscriminator(num_classes=len(categories)).to(device)
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Create labels for real and fake images
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and labels as generator input
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_labels = torch.randint(0, len(categories), (imgs.size(0),), device=device)
            
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Print progress
            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
            
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                # Save sample images
                with torch.no_grad():
                    # Generate images for each class
                    n_row = len(categories)
                    z = torch.randn(n_row, latent_dim, device=device)
                    fixed_labels = torch.tensor(list(range(len(categories))), device=device)
                    gen_imgs = generator(z, fixed_labels)
                    
                    # Save images
                    vutils.save_image(gen_imgs.data, f"samples/cgan_{batches_done}.png", 
                                     nrow=n_row, normalize=True)
    
    # Save the final model
    torch.save(generator.state_dict(), "models/cgan_generator.pth")
    torch.save(discriminator.state_dict(), "models/cgan_discriminator.pth")
    print("CGAN training complete!")

# Train CycleGAN
def train_cyclegan(data_dir, batch_size=4, n_epochs=50, sample_interval=200, save_interval=10, smaller_network=True, reduced_resolution=True):
    print("Starting CycleGAN training with optimized settings...")
    
    # Reduce batch size for memory efficiency
    if torch.device.type == 'cpu':
        print("Running on CPU - using optimized settings for memory efficiency")
        batch_size = min(batch_size, 2)  # Smaller batch size on CPU
        
    # Create datasets for domain A (Normal) and domain B (Alzheimer's)
    # For simplicity, we'll use NonDemented as domain A and all other categories as domain B
    
    # Create transforms with optionally reduced resolution to save memory
    res_size = 128 if reduced_resolution else 224
    
    cyclegan_transforms = {
        'train': transforms.Compose([
            transforms.Resize(res_size + 32),
            transforms.CenterCrop(res_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    }
    
    # Create full dataset with optimized transforms
    full_dataset = AlzheimerDataset(data_dir, transform=cyclegan_transforms['train'])
    
    # Split into domain A and domain B
    domain_A_indices = [i for i, label in enumerate(full_dataset.labels) if label == 0]  # NonDemented
    domain_B_indices = [i for i, label in enumerate(full_dataset.labels) if label > 0]   # All Demented types
    
    # Limit dataset size for faster training if requested
    max_samples_per_domain = 500  # Limit the number of samples to speed up training
    if len(domain_A_indices) > max_samples_per_domain:
        import random
        random.shuffle(domain_A_indices)
        domain_A_indices = domain_A_indices[:max_samples_per_domain]
    
    if len(domain_B_indices) > max_samples_per_domain:
        import random
        random.shuffle(domain_B_indices)
        domain_B_indices = domain_B_indices[:max_samples_per_domain]
    
    print(f"Using {len(domain_A_indices)} domain A samples and {len(domain_B_indices)} domain B samples")
    
    # Create subset datasets
    domain_A_dataset = torch.utils.data.Subset(full_dataset, domain_A_indices)
    domain_B_dataset = torch.utils.data.Subset(full_dataset, domain_B_indices)
    
    # Create dataloaders with smaller batch size and efficient settings
    num_workers = 0 if device.type == 'cpu' else 2  # Reduced workers for CPU
    dataloader_A = DataLoader(domain_A_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=device.type=='cuda')
    dataloader_B = DataLoader(domain_B_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=device.type=='cuda')
    
    # Initialize generators and discriminators with optimized architectures
    G_AB = CycleGAN(direction="AB", smaller_network=smaller_network, 
                    input_size=res_size).to(device)  # Generator: A -> B (Normal to Alzheimer's)
    G_BA = CycleGAN(direction="BA", smaller_network=smaller_network, 
                    input_size=res_size).to(device)  # Generator: B -> A (Alzheimer's to Normal)
    D_A = CycleGANDiscriminator(smaller_network=smaller_network, 
                                input_size=res_size).to(device)    # Discriminator for domain A
    D_B = CycleGANDiscriminator(smaller_network=smaller_network, 
                                input_size=res_size).to(device)    # Discriminator for domain B
    
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers with reduced learning rate for stability
    lr = 0.0001 if device.type == 'cpu' else 0.0002  # Lower learning rate on CPU
    
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - n_epochs // 2) / float(n_epochs // 2)
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - n_epochs // 2) / float(n_epochs // 2)
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - n_epochs // 2) / float(n_epochs // 2)
    )
    
    # Calculate discriminator output size based on input resolution
    disc_output_size = res_size // 16  # For a typical discriminator with 4 downsampling layers
    
    # Function to sample images with less frequency to save time
    def sample_images(batches_done):
        """Save sample images for visualization"""
        imgs = next(iter(dataloader_A))
        real_A = imgs[0].to(device)
        fake_B = G_AB(real_A)
        
        imgs = next(iter(dataloader_B))
        real_B = imgs[0].to(device)
        fake_A = G_BA(real_B)
        
        # Arrange images in a grid
        img_sample = torch.cat([real_A, fake_B, real_B, fake_A], 0)
        vutils.save_image(img_sample, f"samples/cyclegan_{batches_done}.png", 
                         nrow=batch_size, normalize=True)
    
    # Training loop with progress tracking
    from tqdm import tqdm
    import gc
    
    # Keep track of losses for monitoring
    G_losses = []
    D_losses = []
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Initialize running losses for the epoch
        running_loss_G = 0.0
        running_loss_D = 0.0
        batch_count = 0
        
        # Use the shorter of the two dataloaders to avoid issues
        max_batches = min(len(dataloader_A), len(dataloader_B))
        
        # Create zip of the two dataloaders with a progress bar
        data_iterator = zip(dataloader_A, dataloader_B)
        progress_bar = tqdm(data_iterator, total=max_batches, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for i, ((real_A, _), (real_B, _)) in enumerate(progress_bar):
            if i >= max_batches:
                break
                
            # Configure input
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Adversarial ground truths - adjusted for resolution
            valid = torch.ones((real_A.size(0), 1, disc_output_size, disc_output_size), device=device)
            fake = torch.zeros((real_A.size(0), 1, disc_output_size, disc_output_size), device=device)
            
            # Clear memory before backward pass
            if device.type == 'cpu' and i % 5 == 0:
                gc.collect()
            
            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # Total loss with adjusted weights for better stability
            # Reduce identity weight on CPU for faster training
            identity_weight = 1.0 if device.type == 'cpu' else 5.0
            loss_G = loss_GAN + 10.0 * loss_cycle + identity_weight * loss_identity
            
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss
            fake_A_ = fake_A.detach()
            loss_fake = criterion_GAN(D_A(fake_A_), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss
            fake_B_ = fake_B.detach()
            loss_fake = criterion_GAN(D_B(fake_B_), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            
            loss_D_B.backward()
            optimizer_D_B.step()
            
            loss_D = (loss_D_A + loss_D_B) / 2
            
            # Track losses
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': loss_G.item(), 
                'D_loss': loss_D.item(),
                'cycle': loss_cycle.item()
            })
            
            # Sample less frequently to save time
            sampling_freq = sample_interval * 2 if device.type == 'cpu' else sample_interval
            batches_done = epoch * max_batches + i
            if batches_done % sampling_freq == 0:
                sample_images(batches_done)
        
        # Calculate average epoch losses
        avg_loss_G = running_loss_G / batch_count
        avg_loss_D = running_loss_D / batch_count
        
        # Store for plotting
        G_losses.append(avg_loss_G)
        D_losses.append(avg_loss_D)
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{n_epochs} completed in {epoch_time:.1f}s - "
              f"Avg G loss: {avg_loss_G:.4f}, Avg D loss: {avg_loss_D:.4f}")
        
        # Save models periodically to avoid losing progress
        if epoch % save_interval == 0 or epoch == n_epochs - 1:
            torch.save(G_AB.state_dict(), f"models/cyclegan_G_AB_epoch_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"models/cyclegan_G_BA_epoch_{epoch}.pth")
    
    # Save the final models
    torch.save(G_AB.state_dict(), "models/cyclegan_G_AB.pth")
    torch.save(G_BA.state_dict(), "models/cyclegan_G_BA.pth")
    torch.save(D_A.state_dict(), "models/cyclegan_D_A.pth")
    torch.save(D_B.state_dict(), "models/cyclegan_D_B.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.title("CycleGAN Training Losses")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("samples/cyclegan_training_curve.png")
    
    print("CycleGAN training complete!")

# Train HighResGAN
def train_highres_gan(data_dir, batch_size=8, n_epochs=150, latent_dim=128, sample_interval=200, smaller_network=False):
    print("Starting HighResGAN training...")
    
    # Create dataset and dataloader
    dataset = AlzheimerDataset(data_dir, transform=gan_transforms['train'])
    
    # Set appropriate workers based on device
    num_workers = 0 if device.type == 'cpu' else 2
    pin_memory = device.type == 'cuda'
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=pin_memory)
    
    # Initialize generator and discriminator with smaller_network option
    generator = HighResGAN(latent_dim=latent_dim, num_classes=len(categories), 
                          smaller_network=smaller_network).to(device)
    discriminator = HighResGANDiscriminator(num_classes=len(categories)).to(device)
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    
    # Optimizers - adjust learning rate based on device
    lr = 0.0001 if device.type == 'cpu' else 0.0002
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training loop
    print(f"Training on {device} with batch size {batch_size}")
    from tqdm import tqdm
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        running_loss_G = 0.0
        running_loss_D = 0.0
        batch_count = 0
        
        # Create progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for i, (imgs, labels) in enumerate(progress_bar):
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and labels as generator input
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_labels = labels
            
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Track losses
            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': g_loss.item(), 
                'D_loss': d_loss.item()
            })
            
            if i % sample_interval == 0:
                # Save sample images
                with torch.no_grad():
                    # Generate images for each class
                    n_row = len(categories)
                    z = torch.randn(n_row, latent_dim, device=device)
                    fixed_labels = torch.tensor(list(range(len(categories))), device=device)
                    gen_imgs = generator(z, fixed_labels)
                    
                    # Save images
                    vutils.save_image(gen_imgs.data, f"samples/highres_{epoch}_{i}.png", 
                                     nrow=n_row, normalize=True)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss_G = running_loss_G / batch_count
        avg_loss_D = running_loss_D / batch_count
        
        print(f"Epoch {epoch+1}/{n_epochs} completed in {epoch_time:.1f}s - "
              f"Avg G loss: {avg_loss_G:.4f}, Avg D loss: {avg_loss_D:.4f}")
        
        # Save model periodically
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            torch.save(generator.state_dict(), f"models/highres_gan_generator_epoch_{epoch}.pth")
    
    # Save the final model
    torch.save(generator.state_dict(), "models/highres_gan_generator.pth")
    torch.save(discriminator.state_dict(), "models/highres_gan_discriminator.pth")
    print("HighResGAN training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\mohdr\OneDrive\Desktop\python\alzimer2\segmented_images\train", 
                        help="Path to the dataset directory")
    parser.add_argument("--model", type=str, default="all", choices=["cgan", "cyclegan", "highres", "all"], 
                        help="Which GAN model to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--sample_interval", type=int, default=200, help="Interval between image sampling")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--smaller_network", action="store_true", help="Use smaller network for faster training with less memory")
    args = parser.parse_args()
    
    # Force CPU if requested via command line
    if args.force_cpu:
        os.environ['FORCE_CPU'] = '1'
        force_cpu = True
        device = get_device(force_cpu)
        print(f"Forced CPU usage. Using device: {device}")
    else:
        # Empty CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"Using device: {device}")
    
    # Train the selected model(s)
    if args.model == "cgan" or args.model == "all":
        train_cgan(args.data_dir, batch_size=args.batch_size, n_epochs=args.epochs, 
                  sample_interval=args.sample_interval)
    
    if args.model == "cyclegan" or args.model == "all":
        train_cyclegan(args.data_dir, batch_size=args.batch_size, n_epochs=args.epochs, 
                      sample_interval=args.sample_interval, smaller_network=args.smaller_network)
    
    if args.model == "highres" or args.model == "all":
        train_highres_gan(args.data_dir, batch_size=args.batch_size, n_epochs=args.epochs, 
                         sample_interval=args.sample_interval, smaller_network=args.smaller_network) 