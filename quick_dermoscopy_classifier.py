"""
Quick Dermoscopy Classifier - Simplified version for demonstration
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleDermoscopyDataset(Dataset):
    """Simple Dermoscopy Dataset"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleClassifier(nn.Module):
    """Simple ResNet18-based classifier"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use pre-trained ResNet18
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def get_simple_transforms(is_train=True):
    """Simple data transforms"""
    
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_dataset():
    """Load dataset"""
    
    base_path = "/Users/iinuma/Desktop/ダーモ"
    
    # Malignant classes
    malignant_folders = ['AK', 'BCC', 'Bowen病', 'MM']
    
    image_paths = []
    labels = []
    
    # Load malignant images (label: 1)
    for folder in malignant_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.endswith('.JPG'):
                    image_paths.append(os.path.join(folder_path, img_file))
                    labels.append(1)  # Malignant
    
    print(f"Malignant images: {len(image_paths)}")
    
    # Load benign images (label: 0)
    benign_folder = os.path.join(base_path, 'benign')
    if os.path.exists(benign_folder):
        benign_count = 0
        for img_file in os.listdir(benign_folder):
            if img_file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG')):
                image_paths.append(os.path.join(benign_folder, img_file))
                labels.append(0)  # Benign
                benign_count += 1
        print(f"Benign images: {benign_count}")
    
    print(f"Total images: {len(image_paths)}")
    print(f"Malignant: {sum(labels)} Benign: {len(labels) - sum(labels)}")
    
    return image_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=3):
    """Train the model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'quick_dermoscopy_model.pth')
            print(f'Best model saved with accuracy: {val_acc:.2f}%')
    
    return history

def evaluate_model(model, test_loader):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    return all_preds, all_labels

def visualize_results(history):
    """Visualize training results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('quick_training_history.png')
    plt.show()

def main():
    """Main execution function"""
    
    print("Quick Dermoscopy Classifier")
    print("=" * 50)
    
    # Load dataset
    image_paths, labels = load_dataset()
    
    if len(set(labels)) < 2:
        print("\nError: Need both benign and malignant data.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SimpleDermoscopyDataset(
        X_train, y_train, transform=get_simple_transforms(is_train=True)
    )
    
    test_dataset = SimpleDermoscopyDataset(
        X_test, y_test, transform=get_simple_transforms(is_train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create model
    model = SimpleClassifier(num_classes=2).to(device)
    
    print(f"\nDataset Info:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Testing: {len(test_dataset)} images")
    print(f"Device: {device}")
    
    # Train model
    print("\nStarting training...")
    history = train_model(model, train_loader, test_loader, num_epochs=3)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader)
    
    # Visualize results
    visualize_results(history)
    
    print("\nTraining completed!")
    print("Model saved as 'quick_dermoscopy_model.pth'")

if __name__ == "__main__":
    main()