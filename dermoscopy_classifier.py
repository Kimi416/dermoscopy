"""
2段階ダーモスコピー画像分類モデル
Stage 1: 画像品質改善・前処理
Stage 2: 良悪性分類
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_b, convnext_base
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class Stage1_ImageEnhancer:
    """Stage 1: 画像品質改善・前処理"""
    
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def remove_hair(self, img):
        """髪の毛除去（Inpaintingベース）"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Black Hat Transform で髪の毛検出
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        
        # 閾値処理でマスク作成
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpainting で髪の毛を除去
        result = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
        return result
    
    def enhance_contrast(self, img):
        """コントラスト強調（CLAHE）"""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def normalize_color(self, img):
        """色彩正規化"""
        # 各チャンネルの正規化
        normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            channel = img[:, :, i].astype(np.float32)
            mean = np.mean(channel)
            std = np.std(channel)
            normalized[:, :, i] = (channel - mean) / (std + 1e-8)
        
        # 0-255の範囲に戻す
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)
        return normalized
    
    def process(self, img):
        """全前処理を適用"""
        img = self.remove_hair(img)
        img = self.enhance_contrast(img)
        img = self.normalize_color(img)
        return img

class DermoscopyDataset(Dataset):
    """ダーモスコピー画像データセット"""
    
    def __init__(self, image_paths, labels, transform=None, stage1_enhancer=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.stage1_enhancer = stage1_enhancer
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 画像読み込み
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Stage 1: 前処理
        if self.stage1_enhancer:
            image = self.stage1_enhancer.process(image)
        
        # データ拡張
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class ClinicalFeatureExtractor(nn.Module):
    """臨床的特徴（ABCD基準）の自動抽出"""
    
    def __init__(self):
        super().__init__()
        # Asymmetry（非対称性）検出
        self.asymmetry_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Border（境界）不整検出
        self.border_conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Color（色彩）多様性検出
        self.color_conv = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Diameter/Differential structures（構造）検出
        self.structure_conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        asymmetry = self.asymmetry_conv(x).flatten(1)
        border = self.border_conv(x).flatten(1)
        color = self.color_conv(x).flatten(1)
        structure = self.structure_conv(x).flatten(1)
        
        # ABCD特徴を結合
        clinical_features = torch.cat([asymmetry, border, color, structure], dim=1)
        return clinical_features

class HierarchicalClassifier(nn.Module):
    """階層的分類モデル（高精度版）"""
    
    def __init__(self, num_classes=9):
        super().__init__()
        
        # 臨床的特徴抽出
        self.clinical_extractor = ClinicalFeatureExtractor()
        
        # Swin Transformer（深層特徴）
        self.swin = swin_b(weights='IMAGENET1K_V1')
        self.swin.head = nn.Linear(self.swin.head.in_features, 512)
        
        # ConvNeXt（テクスチャ特徴）
        self.convnext = convnext_base(weights='IMAGENET1K_V1')
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, 512)
        
        # Attention機構
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        
        # Stage 1: 良性/悪性分類
        self.malignancy_classifier = nn.Sequential(
            nn.Linear(1024 + 80, 512),  # 深層特徴 + ABCD特徴
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 良性/悪性
        )
        
        # Stage 2: 詳細分類
        self.detail_classifier = nn.Sequential(
            nn.Linear(1024 + 80 + 2, 512),  # 特徴 + 良悪性確率
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 臨床的特徴抽出
        clinical_features = self.clinical_extractor(x)
        
        # 深層特徴抽出
        swin_features = self.swin(x)
        convnext_features = self.convnext(x)
        deep_features = torch.cat([swin_features, convnext_features], dim=1)
        
        # Self-Attention
        deep_features_reshaped = deep_features.unsqueeze(0)
        attended_features, _ = self.attention(
            deep_features_reshaped, 
            deep_features_reshaped, 
            deep_features_reshaped
        )
        attended_features = attended_features.squeeze(0)
        
        # 全特徴を結合
        all_features = torch.cat([attended_features, clinical_features], dim=1)
        
        # Stage 1: 良悪性判定
        malignancy_logits = self.malignancy_classifier(all_features)
        malignancy_probs = torch.softmax(malignancy_logits, dim=1)
        
        # Stage 2: 詳細分類（良悪性確率も入力）
        combined_input = torch.cat([all_features, malignancy_probs], dim=1)
        detail_logits = self.detail_classifier(combined_input)
        
        return {
            'malignancy': malignancy_logits,
            'detail': detail_logits,
            'clinical_features': clinical_features
        }

class Stage2_Classifier(nn.Module):
    """Stage 2: アンサンブル分類モデル"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Swin Transformer
        self.swin = swin_b(weights='IMAGENET1K_V1')
        self.swin.head = nn.Linear(self.swin.head.in_features, 256)
        
        # ConvNeXt
        self.convnext = convnext_base(weights='IMAGENET1K_V1')
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, 256)
        
        # アンサンブル層
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 両モデルで特徴抽出
        swin_features = self.swin(x)
        convnext_features = self.convnext(x)
        
        # 特徴を結合
        combined = torch.cat([swin_features, convnext_features], dim=1)
        
        # 最終分類
        output = self.fusion(combined)
        return output

def get_augmentation_pipeline(is_train=True):
    """データ拡張パイプライン"""
    
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.ElasticTransform(p=1),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(p=1),
                A.RandomBrightnessContrast(p=1),
                A.ColorJitter(p=1),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def load_dataset(use_detailed_labels=False):
    """データセットの読み込み
    
    Args:
        use_detailed_labels: True で詳細な病変分類、False で良悪性のみ
    """
    
    base_path = "/Users/iinuma/Desktop/ダーモ"
    
    # 病変クラスの定義
    class_mapping = {
        # 悪性病変
        'AK': {'malignant': 1, 'detail': 0, 'name': '日光角化症'},
        'BCC': {'malignant': 1, 'detail': 1, 'name': '基底細胞癌'},
        'Bowen病': {'malignant': 1, 'detail': 2, 'name': 'Bowen病'},
        'MM': {'malignant': 1, 'detail': 3, 'name': '悪性黒色腫'},
        
        # 良性病変
        'SK': {'malignant': 0, 'detail': 4, 'name': '脂漏性角化症'},
        'nevus': {'malignant': 0, 'detail': 5, 'name': '母斑'},
        'DF': {'malignant': 0, 'detail': 6, 'name': '皮膚線維腫'},
        'VASC': {'malignant': 0, 'detail': 7, 'name': '血管病変'},
        'benign': {'malignant': 0, 'detail': 8, 'name': 'その他良性'}
    }
    
    image_paths = []
    labels = []
    detailed_labels = []
    class_counts = {}
    
    # 各フォルダから画像を読み込み
    for folder_name, class_info in class_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        if os.path.exists(folder_path):
            count = 0
            for img_file in os.listdir(folder_path):
                if img_file.endswith(('.JPG', '.jpg', '.png')):
                    image_paths.append(os.path.join(folder_path, img_file))
                    labels.append(class_info['malignant'])
                    detailed_labels.append(class_info['detail'])
                    count += 1
            if count > 0:
                class_counts[class_info['name']] = count
                print(f"{class_info['name']}: {count}枚")
    
    print(f"\n合計画像数: {len(image_paths)}")
    print(f"悪性: {sum(1 for l in labels if l == 1)}枚")
    print(f"良性: {sum(1 for l in labels if l == 0)}枚")
    
    if use_detailed_labels:
        return image_paths, detailed_labels, class_mapping
    else:
        return image_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=30):
    """モデルの学習"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
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
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Val'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 統計記録
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # ベストモデルの保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_dermoscopy_model.pth')
            print(f'Best model saved with accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    return history

def visualize_results(history):
    """学習結果の可視化"""
    
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
    plt.savefig('training_history.png')
    plt.show()

def main():
    """メイン実行関数"""
    
    print("🔬 2段階ダーモスコピー分類モデルの構築")
    print("=" * 50)
    
    # データセット読み込み
    image_paths, labels = load_dataset()
    
    if len(set(labels)) < 2:
        print("\n⚠️ エラー: 良性データが必要です。")
        print("以下のいずれかの方法で良性データを追加してください：")
        print("1. ISICアーカイブからダウンロード")
        print("2. 'benign'フォルダを作成して良性画像を配置")
        return
    
    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Stage 1: 前処理器
    stage1_enhancer = Stage1_ImageEnhancer()
    
    # データセット作成
    train_dataset = DermoscopyDataset(
        X_train, y_train,
        transform=get_augmentation_pipeline(is_train=True),
        stage1_enhancer=stage1_enhancer
    )
    
    val_dataset = DermoscopyDataset(
        X_val, y_val,
        transform=get_augmentation_pipeline(is_train=False),
        stage1_enhancer=stage1_enhancer
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Stage 2: 分類モデル
    model = Stage2_Classifier(num_classes=2).to(device)
    
    print(f"\n📊 データセット情報:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Device: {device}")
    
    # モデル学習
    print("\n🚀 学習開始...")
    history = train_model(model, train_loader, val_loader, num_epochs=5)
    
    # 結果可視化
    visualize_results(history)
    
    print("\n✅ 学習完了!")
    print("モデルは 'best_dermoscopy_model.pth' に保存されました。")

if __name__ == "__main__":
    main()