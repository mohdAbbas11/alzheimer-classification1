import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from PIL import Image

# Import models and utilities
from gan_models import CGAN, CycleGAN, HighResGAN
from test import AlzheimerDataset, EnhancedAlzheimerNet, categories, get_device

# Set device
device = get_device()
print(f"Using device: {device}")

# Define transforms for testing
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define classifier model for evaluation
def load_classifier():
    model = EnhancedAlzheimerNet(model_name='resnet101', num_classes=len(categories))
    model.load_state_dict(torch.load('best_resnet101_fold1.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to evaluate real images
def evaluate_real_images(data_dir, batch_size=16):
    print("\nEvaluating real images...")
    
    # Load test dataset
    test_dataset = AlzheimerDataset(data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load classifier
    classifier = load_classifier()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Real Images Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Real Images Confusion Matrix')
    plt.savefig('real_images_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report for Real Images:")
    print(classification_report(all_labels, all_preds, target_names=categories))
    
    return accuracy, all_preds, all_labels

# Function to evaluate CGAN generated images
def evaluate_cgan(num_samples=100, batch_size=16, latent_dim=100):
    print("\nEvaluating CGAN generated images...")
    
    # Load models
    generator = CGAN(latent_dim=latent_dim, num_classes=len(categories)).to(device)
    generator.load_state_dict(torch.load('models/cgan_generator.pth', map_location=device))
    generator.eval()
    
    classifier = load_classifier()
    
    # Generate images and evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for class_idx in range(len(categories)):
            # Generate images for this class
            samples_per_class = num_samples // len(categories)
            
            # Generate in batches
            for i in range(0, samples_per_class, batch_size):
                curr_batch_size = min(batch_size, samples_per_class - i)
                
                # Generate noise and labels
                z = torch.randn(curr_batch_size, latent_dim).to(device)
                labels = torch.full((curr_batch_size,), class_idx, dtype=torch.long).to(device)
                
                # Generate images
                gen_imgs = generator(z, labels)
                
                # Normalize from [-1, 1] to [0, 1] range
                gen_imgs = (gen_imgs + 1) / 2
                
                # Normalize for classifier input
                gen_imgs = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(gen_imgs)
                
                # Classify generated images
                outputs = classifier(gen_imgs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"CGAN Generated Images Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CGAN Generated Images Confusion Matrix')
    plt.savefig('cgan_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report for CGAN Generated Images:")
    print(classification_report(all_labels, all_preds, target_names=categories))
    
    return accuracy, all_preds, all_labels

# Function to evaluate CycleGAN generated images
def evaluate_cyclegan(data_dir, batch_size=16, direction="AB"):
    print(f"\nEvaluating CycleGAN generated images (direction: {direction})...")
    
    # Load models
    generator = CycleGAN(input_channels=3, output_channels=3, direction=direction).to(device)
    
    # Load the appropriate generator model
    if direction == "AB":  # Normal to Alzheimer's
        generator.load_state_dict(torch.load('models/cyclegan_G_AB.pth', map_location=device))
    else:  # Alzheimer's to Normal
        generator.load_state_dict(torch.load('models/cyclegan_G_BA.pth', map_location=device))
    
    generator.eval()
    classifier = load_classifier()
    
    # Load test dataset
    test_dataset = AlzheimerDataset(data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Generate images and evaluate
    all_preds = []
    all_labels = []
    all_orig_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Generate transformed images
            gen_imgs = generator(inputs)
            
            # Normalize from [-1, 1] to [0, 1] range if needed
            if gen_imgs.min() < 0:
                gen_imgs = (gen_imgs + 1) / 2
            
            # Normalize for classifier input
            gen_imgs = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(gen_imgs)
            
            # Classify original and generated images
            orig_outputs = classifier(inputs)
            gen_outputs = classifier(gen_imgs)
            
            _, orig_preds = torch.max(orig_outputs, 1)
            _, gen_preds = torch.max(gen_outputs, 1)
            
            all_orig_preds.extend(orig_preds.cpu().numpy())
            all_preds.extend(gen_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    orig_accuracy = accuracy_score(all_labels, all_orig_preds)
    
    print(f"Original Images Accuracy: {orig_accuracy:.4f}")
    print(f"CycleGAN Generated Images Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'CycleGAN Generated Images Confusion Matrix (Direction: {direction})')
    plt.savefig(f'cyclegan_{direction}_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print(f"\nClassification Report for CycleGAN Generated Images (Direction: {direction}):")
    print(classification_report(all_labels, all_preds, target_names=categories))
    
    return accuracy, all_preds, all_labels

# Function to evaluate HighResGAN generated images
def evaluate_highres_gan(num_samples=100, batch_size=16, latent_dim=128):
    print("\nEvaluating HighResGAN generated images...")
    
    # Load models
    generator = HighResGAN(latent_dim=latent_dim, num_classes=len(categories)).to(device)
    generator.load_state_dict(torch.load('models/highres_gan_generator.pth', map_location=device))
    generator.eval()
    
    classifier = load_classifier()
    
    # Generate images and evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for class_idx in range(len(categories)):
            # Generate images for this class
            samples_per_class = num_samples // len(categories)
            
            # Generate in batches
            for i in range(0, samples_per_class, batch_size):
                curr_batch_size = min(batch_size, samples_per_class - i)
                
                # Generate noise and labels
                z = torch.randn(curr_batch_size, latent_dim).to(device)
                labels = torch.full((curr_batch_size,), class_idx, dtype=torch.long).to(device)
                
                # Generate images
                gen_imgs = generator(z, labels)
                
                # Normalize from [-1, 1] to [0, 1] range
                gen_imgs = (gen_imgs + 1) / 2
                
                # Normalize for classifier input
                gen_imgs = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(gen_imgs)
                
                # Classify generated images
                outputs = classifier(gen_imgs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"HighResGAN Generated Images Accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('HighResGAN Generated Images Confusion Matrix')
    plt.savefig('highres_gan_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report for HighResGAN Generated Images:")
    print(classification_report(all_labels, all_preds, target_names=categories))
    
    return accuracy, all_preds, all_labels

# Compare all models
def compare_models(results):
    # Create bar chart comparing accuracies
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of GAN Models')
    plt.ylim(0, 1.1)
    plt.savefig('gan_models_accuracy_comparison.png')
    plt.close()
    
    # Print summary table
    print("\n--- GAN Models Accuracy Comparison ---")
    for model, data in results.items():
        print(f"{model}: {data['accuracy']:.4f}")

def main():
    # Test directory
    test_dir = "alzimer2\OriginalDataset"
    
    # Dictionary to store results
    results = {}
    
    try:
        # Evaluate real images first
        real_acc, real_preds, real_labels = evaluate_real_images(test_dir)
        results['Real Images'] = {
            'accuracy': real_acc,
            'predictions': real_preds,
            'labels': real_labels
        }
        
        # Evaluate CGAN
        cgan_acc, cgan_preds, cgan_labels = evaluate_cgan()
        results['CGAN'] = {
            'accuracy': cgan_acc,
            'predictions': cgan_preds,
            'labels': cgan_labels
        }
        
        # Evaluate CycleGAN (Normal to Alzheimer's)
        cyclegan_ab_acc, cyclegan_ab_preds, cyclegan_ab_labels = evaluate_cyclegan(test_dir, direction="AB")
        results['CycleGAN (N→A)'] = {
            'accuracy': cyclegan_ab_acc,
            'predictions': cyclegan_ab_preds,
            'labels': cyclegan_ab_labels
        }
        
        # Evaluate HighResGAN
        highres_acc, highres_preds, highres_labels = evaluate_highres_gan()
        results['HighResGAN'] = {
            'accuracy': highres_acc,
            'predictions': highres_preds,
            'labels': highres_labels
        }
        
        # Compare all models
        compare_models(results)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 