import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import time
import copy
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Function to determine device
def get_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Configure PyTorch to use more efficient memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        return device
    else:
        print("Using CPU for computation")
        return torch.device("cpu")

# Check if CUDA is available and use it by default
force_cpu = os.environ.get('FORCE_CPU', '0') == '1'
device = get_device(force_cpu)
print(f"Selected device: {device}")

# Define the categories
categories = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Enhanced data augmentation with standard PyTorch transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Test Time Augmentation transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=[90, 90]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

# Custom Dataset class with enhanced features
class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None, balance_classes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_counts = {category: 0 for category in categories}
        self.balance_classes = balance_classes
        
        # Iterate through category folders
        for category_idx, category in enumerate(categories):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in files:
                    self.image_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(category_idx)
                    self.class_counts[category] += 1
                print(f"Found {len(files)} images in category '{category}'")
        
        # Calculate class weights for balanced sampling
        if balance_classes and len(self.labels) > 0:
            class_weights = [1.0/self.class_counts[categories[i]] for i in range(len(categories))]
            self.sample_weights = [class_weights[label] for label in self.labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the same label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, self.labels[idx]

# Ensemble Model - Using standard torchvision models
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=4):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList()
        
        # Initialize models for ensemble
        # Model 1: ResNet50
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        self.models.append(resnet)
        
        # Model 2: ResNeXt50
        resnext = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        num_ftrs = resnext.fc.in_features
        resnext.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        self.models.append(resnext)
        
        # Model 3: DenseNet121
        densenet = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        self.models.append(densenet)
    
    def forward(self, x):
        # Average the predictions from all models
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average the logits
        return torch.mean(torch.stack(outputs), dim=0)

# Enhanced single model using standard torchvision models
class EnhancedAlzheimerNet(nn.Module):
    def __init__(self, model_name='resnet101', num_classes=4, pretrained=True):
        super(EnhancedAlzheimerNet, self).__init__()
        
        self.model_name = model_name
        weights_param = None
        if pretrained:
            if model_name == 'resnet101':
                weights_param = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
            elif model_name == 'resnext101_32x8d':
                weights_param = torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
            elif model_name == 'densenet161':
                weights_param = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
            elif model_name == 'wide_resnet101_2':
                weights_param = torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
            else:  # Default to ResNet50
                weights_param = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        
        if model_name == 'resnet101':
            self.model = torchvision.models.resnet101(weights=weights_param)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == 'resnext101_32x8d':
            self.model = torchvision.models.resnext101_32x8d(weights=weights_param)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == 'densenet161':
            self.model = torchvision.models.densenet161(weights=weights_param)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == 'wide_resnet101_2':
            self.model = torchvision.models.wide_resnet101_2(weights=weights_param)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        else:  # Default to ResNet50
            self.model = torchvision.models.resnet50(weights=weights_param)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
    
    def forward(self, x):
        return self.model(x)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Function to train the model with advanced techniques
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=50, patience=15, model_path='best_alzheimer_model.pth', 
                gradient_clipping=1.0, use_amp=True):
    since = time.time()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Initialize the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Use AMP for faster training when available
                    if phase == 'train' and use_amp and torch.cuda.is_available():
                        with autocast(device_type='cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # Scales loss and performs backward pass, producing scaled gradients
                        scaler.scale(loss).backward()
                        
                        # Unscales gradients and clips as usual
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        
                        # Step optimizer and update scaler
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            # Gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                            optimizer.step()
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    # Will handle after validation phase
                    pass
                else:
                    scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
                # Step the scheduler if it's ReduceLROnPlateau
                if scheduler is not None and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                
                # Early stopping
                early_stopping(epoch_loss, model)
                
                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
        print()
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

# Test-time augmentation function
def apply_tta(model, image_tensor, transforms_list):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        original_output = model(image_tensor.unsqueeze(0).to(device))
        predictions.append(original_output)
        
        # Create PIL image from tensor for transforms
        img = transforms.ToPILImage()(image_tensor)
        
        # Apply each TTA transform
        for transform in transforms_list:
            augmented = transform(img).unsqueeze(0).to(device)
            output = model(augmented)
            predictions.append(output)
    
    # Average predictions
    avg_preds = torch.mean(torch.stack(predictions), dim=0)
    return avg_preds

# Function to evaluate the model with additional metrics
def evaluate_model(model, test_loader, class_names, use_tta=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if use_tta:
                batch_probs = []
                for i in range(inputs.size(0)):
                    # Apply TTA to each image in batch
                    tta_output = apply_tta(model, inputs[i], tta_transforms)
                    prob = torch.nn.functional.softmax(tta_output, dim=1)
                    batch_probs.append(prob)
                
                # Stack batch probabilities
                probs = torch.cat(batch_probs, dim=0)
                _, preds = torch.max(probs, 1)
            else:
                # Standard evaluation
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC curve and AUC for multi-class
    n_classes = len(class_names)
    
    # Binarize the labels for ROC curve
    y_bin = label_binarize(all_labels, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, color, cls in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {cls} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()
    
    # Calculate and print the average accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    for i, cls_name in enumerate(class_names):
        cls_indices = all_labels == i
        if np.any(cls_indices):
            cls_acc = np.mean(all_preds[cls_indices] == all_labels[cls_indices])
            print(f"Accuracy for {cls_name}: {cls_acc:.4f}")
    
    return all_preds, all_labels, all_probs, report

def main():
    # Data paths - using the specified training path
    data_dir = r"C:\Users\mohdr\OneDrive\Desktop\python\alzimer2\segmented_images\train"
    
    # Hyperparameters optimized for high accuracy
    batch_size = 12  # Smaller batch size for better generalization
    num_epochs = 10  # Reduced from 100 to 10 epochs
    learning_rate = 0.0001  # Lower initial learning rate
    weight_decay = 2e-4  # Increased L2 regularization
    gradient_clip = 1.0  # Gradient clipping value
    patience = 5  # Reduced early stopping patience to match fewer epochs
    use_amp = True  # Use mixed precision training if available
    
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
        
        # Do an 80/20 split first to isolate the test set
        train_val_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_val_size
        
        train_val_dataset, test_dataset = random_split(
            full_dataset, [train_val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply test transforms
        test_dataset = torch.utils.data.Subset(AlzheimerDataset(data_dir, transform=data_transforms['test']), 
                                              test_dataset.indices)
        
        # Convert to numpy array for stratification
        labels_array = np.array([full_dataset.labels[i] for i in train_val_dataset.indices])
        
        # Implement stratified K-fold cross-validation
        k_folds = 5
        fold_results = []
        best_models = []
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # K-fold cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_val_dataset)), labels_array)):
            print(f"\n{'='*20} Fold {fold+1}/{k_folds} {'='*20}")
            
            # Get indices relative to the full dataset
            train_indices = [train_val_dataset.indices[i] for i in train_idx]
            val_indices = [train_val_dataset.indices[i] for i in val_idx]
            
            # Create datasets with appropriate transforms
            train_dataset = torch.utils.data.Subset(AlzheimerDataset(data_dir, transform=data_transforms['train']), 
                                                   train_indices)
            val_dataset = torch.utils.data.Subset(AlzheimerDataset(data_dir, transform=data_transforms['val']), 
                                                 val_indices)
            
            # Calculate class weights for sampler
            train_labels = [full_dataset.labels[i] for i in train_indices]
            class_sample_counts = np.bincount(train_labels)
            weight_per_class = 1. / class_sample_counts
            weights = [weight_per_class[t] for t in train_labels]
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                                    num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=2, pin_memory=True)
            
            # Define model architecture for this fold
            if fold % 3 == 0:
                model_name = 'resnet101'
            elif fold % 3 == 1:
                model_name = 'resnext101_32x8d'
            else:
                model_name = 'densenet161'
                
            print(f"Training {model_name} for fold {fold+1}")
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
                T_0=10,  # Restart every 10 epochs
                T_mult=2,  # Double the restart interval after each restart
                eta_min=1e-6  # Minimum learning rate
            )
            
            # Train the model
            print(f"Starting training for fold {fold+1}...")
            model, train_losses, val_losses, train_accs, val_accs = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=num_epochs, patience=patience, model_path=f'best_{model_name}_fold{fold+1}.pth',
                gradient_clipping=gradient_clip, use_amp=use_amp
            )
            
            # Save training curves
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Fold {fold+1} - Loss Curves')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(val_accs, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Fold {fold+1} - Accuracy Curves')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'fold{fold+1}_training_curves.png')
            plt.close()
            
            # Record validation accuracy and save model
            best_val_acc = max(val_accs)
            fold_results.append(best_val_acc)
            
            # Load the best model for this fold
            model.load_state_dict(torch.load(f'best_{model_name}_fold{fold+1}.pth'))
            best_models.append((model_name, model))
            
            print(f"Best validation accuracy for fold {fold+1}: {best_val_acc:.4f}")
        
        # Print cross-validation results
        print("\nCross-validation results:")
        for fold, acc in enumerate(fold_results):
            print(f"Fold {fold+1}: {acc:.4f}")
        print(f"Mean accuracy: {np.mean(fold_results):.4f}")
        print(f"Standard deviation: {np.std(fold_results):.4f}")
        
        # Create test dataloader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Create ensemble from best models
        print("\nCreating ensemble model for final evaluation...")
        ensemble = EnsembleModel(num_classes=len(categories))
        
        # Copy weights from trained models to ensemble
        for i, (model_name, trained_model) in enumerate(best_models[:3]):  # Use top 3 models
            if model_name == 'resnet101' or model_name == 'resnext101_32x8d':
                ensemble.models[i].load_state_dict(trained_model.model.state_dict())
            elif model_name == 'densenet161':
                # Use densenet from the ensemble
                ensemble.models[2].load_state_dict(trained_model.model.state_dict())
        
        ensemble = ensemble.to(device)
        
        # Evaluate the ensemble model
        print("\nEvaluating ensemble model on test set...")
        all_preds, all_labels, all_probs, report = evaluate_model(ensemble, test_loader, categories, use_tta=True)
        
        # Print final results
        print(f"\nFinal Test Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        print("\nAlzheimer's detection model training and evaluation complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 