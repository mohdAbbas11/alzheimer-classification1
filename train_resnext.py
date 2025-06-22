import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import time
import copy
import random
import sys

# Import necessary classes and functions from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test import (get_device, categories, data_transforms, tta_transforms, 
                  AlzheimerDataset, EnhancedAlzheimerNet, EarlyStopping, 
                  train_model, evaluate_model)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available and use it by default
force_cpu = os.environ.get('FORCE_CPU', '0') == '1'
device = get_device(force_cpu)
print(f"Selected device: {device}")

def train_resnext():
    # Data paths - using the specified training path
    data_dir = r"C:\Users\mohdr\OneDrive\Desktop\python\alzimer2\segmented_images\train"
    
    # Hyperparameters
    batch_size = 12
    num_epochs = 10  # Train for 10 epochs as requested
    learning_rate = 0.0001
    weight_decay = 2e-4
    gradient_clip = 1.0
    patience = 5
    use_amp = True
    model_name = 'resnext101_32x8d'  # Specifically using resnext101_32x8d
    
    print(f"Loading data from: {data_dir}")
    
    try:
        # Create dataset
        full_dataset = AlzheimerDataset(data_dir, transform=data_transforms['val'], balance_classes=True)
        
        # Print class distribution
        print("\nClass distribution:")
        for category, count in full_dataset.class_counts.items():
            print(f"{category}: {count} images")
        
        # Calculate total dataset size for splitting
        dataset_size = len(full_dataset)
        
        # Split into train, validation and test sets (70/15/15)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply appropriate transforms
        train_dataset = torch.utils.data.Subset(AlzheimerDataset(data_dir, transform=data_transforms['train']), 
                                               train_dataset.indices)
        val_dataset = torch.utils.data.Subset(AlzheimerDataset(data_dir, transform=data_transforms['val']), 
                                             val_dataset.indices)
        test_dataset = torch.utils.data.Subset(AlzheimerDataset(data_dir, transform=data_transforms['test']), 
                                              test_dataset.indices)
        
        # Calculate class weights for sampler
        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
        class_sample_counts = np.bincount(train_labels)
        weight_per_class = 1. / class_sample_counts
        weights = [weight_per_class[t] for t in train_labels]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                                num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=2, pin_memory=True)
        
        # Define model - specifically using resnext101_32x8d
        print(f"Training {model_name} for {num_epochs} epochs")
        model = EnhancedAlzheimerNet(model_name=model_name, num_classes=len(categories))
        model = model.to(device)
        
        # Define loss function with class weights
        class_weights = torch.FloatTensor(weight_per_class).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Define optimizer - AdamW with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler with cosine annealing
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Train the model
        print(f"Starting training for {model_name}...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            num_epochs=num_epochs, patience=patience, model_path=f'best_{model_name}.pth',
            gradient_clipping=gradient_clip, use_amp=use_amp
        )
        
        # Save training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Loss Curves')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} - Accuracy Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_curves.png')
        plt.close()
        
        # Load the best model
        model.load_state_dict(torch.load(f'best_{model_name}.pth'))
        
        # Evaluate the model
        print("\nEvaluating model on test set...")
        all_preds, all_labels, all_probs, report = evaluate_model(model, test_loader, categories, use_tta=True)
        
        # Print final results
        print(f"\nFinal Test Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        print(f"\n{model_name} training and evaluation complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_resnext() 