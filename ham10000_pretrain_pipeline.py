"""
HAM10000データでの事前学習 → ユーザーデータでのファインチューニング・評価パイプライン
ISIC版からHAM10000版への移行
"""

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class HAM10000Loader:
    """HAM10000データセットローダー"""
    
    def __init__(self, data_dir="ham10000_data"):
        self.data_dir = data_dir
        
        # 分類マップ読み込み
        map_file = os.path.join(data_dir, "binary_classification_map.json")
        if os.path.exists(map_file):
            with open(map_file, 'r', encoding='utf-8') as f:
                self.classification_map = json.load(f)
        else:
            print(f"⚠️ 分類マップが見つかりません: {map_file}")
            self.classification_map = self._create_default_map()
    
    def _create_default_map(self):
        """デフォルト分類マップ作成"""
        return {
            'bkl': {'binary_label': 0, 'category': 'benign'},
            'df': {'binary_label': 0, 'category': 'benign'},
            'nv': {'binary_label': 0, 'category': 'benign'},
            'vasc': {'binary_label': 0, 'category': 'benign'},
            'akiec': {'binary_label': 1, 'category': 'malignant'},
            'bcc': {'binary_label': 1, 'category': 'malignant'},
            'mel': {'binary_label': 1, 'category': 'malignant'}
        }
    
    def load_metadata(self):
        """HAM10000メタデータ読み込み"""
        
        metadata_file = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        
        if not os.path.exists(metadata_file):
            print(f"⚠️ メタデータファイルが見つかりません: {metadata_file}")
            print("手動でHAM10000データセットをダウンロードしてください")
            return None
        
        df = pd.read_csv(metadata_file)
        print(f"📋 メタデータ読み込み完了: {len(df)}件")
        
        return df
    
    def prepare_binary_dataset(self, test_size=0.2):
        """HAM10000を良性・悪性の2クラスデータセットに変換"""
        
        df = self.load_metadata()
        if df is None:
            return [], [], [], []
        
        image_paths = []
        labels = []
        
        # 各画像のパスとラベルを準備
        for _, row in df.iterrows():
            image_id = row['image_id']
            diagnosis = row['dx']
            
            # 画像ファイルパス（part1またはpart2から検索）
            image_file = f"{image_id}.jpg"
            image_path = None
            
            # part1, part2のディレクトリから検索
            for part_dir in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                potential_path = os.path.join(self.data_dir, part_dir, image_file)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path and diagnosis in self.classification_map:
                image_paths.append(image_path)
                labels.append(self.classification_map[diagnosis]['binary_label'])
        
        print(f"📊 有効な画像数: {len(image_paths)}")
        
        if len(image_paths) == 0:
            print("❌ 有効な画像が見つかりません")
            return [], [], [], []
        
        # 学習・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 統計表示
        train_benign = sum(1 for label in y_train if label == 0)
        train_malignant = sum(1 for label in y_train if label == 1)
        test_benign = sum(1 for label in y_test if label == 0)
        test_malignant = sum(1 for label in y_test if label == 1)
        
        print(f"\\n📈 HAM10000データセット分割結果:")
        print(f"学習データ: {len(X_train)}枚 (良性: {train_benign}, 悪性: {train_malignant})")
        print(f"テストデータ: {len(X_test)}枚 (良性: {test_benign}, 悪性: {test_malignant})")
        
        return X_train, X_test, y_train, y_test

class PretrainedModel(nn.Module):
    """事前学習用モデル（ISIC版と同じ）"""
    
    def __init__(self, num_classes=2, model_type='efficientnet'):
        super().__init__()
        
        if model_type == 'efficientnet':
            self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:  # resnet
            self.backbone = resnet50(weights='IMAGENET1K_V2')
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

class DermoscopyDataset(Dataset):
    """ダーモスコピー画像データセット（ISIC版と同じ）"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_train=True):
    """データ変換パイプライン（ISIC版と同じ）"""
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def pretrain_on_ham10000(model, train_loader, val_loader, epochs=10):
    """HAM10000データで事前学習"""
    
    print("\\n🔬 HAM10000データで事前学習中...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ham10000_pretrained_model.pth')
        
        scheduler.step()
    
    print(f"✅ HAM10000事前学習完了。最高精度: {best_val_acc:.2f}%")
    return model

def load_user_data():
    """ユーザーデータの読み込み（ISIC版と同じ、SKは除外）"""
    
    base_path = "/Users/iinuma/Desktop/ダーモ"
    
    # 病変タイプ別のデータ（SKは除外）
    disease_data = {
        'AK': {'paths': [], 'label': 1, 'name': '日光角化症'},
        'BCC': {'paths': [], 'label': 1, 'name': '基底細胞癌'},
        'Bowen病': {'paths': [], 'label': 1, 'name': 'Bowen病'},
        'MM': {'paths': [], 'label': 1, 'name': '悪性黒色腫'}
    }
    
    # 各フォルダから画像パスを収集
    for folder_name, info in disease_data.items():
        folder_path = os.path.join(base_path, folder_name)
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.endswith('.JPG'):
                    info['paths'].append(os.path.join(folder_path, img_file))
    
    # 学習用とテスト用に分割（各疾患から20%をテストに）
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    test_diseases = []
    
    for folder_name, info in disease_data.items():
        if len(info['paths']) > 0:
            disease_train, disease_test = train_test_split(
                info['paths'], test_size=0.2, random_state=42
            )
            
            train_paths.extend(disease_train)
            train_labels.extend([info['label']] * len(disease_train))
            
            test_paths.extend(disease_test)
            test_labels.extend([info['label']] * len(disease_test))
            test_diseases.extend([info['name']] * len(disease_test))
            
            print(f"{info['name']}: 学習 {len(disease_train)}枚, テスト {len(disease_test)}枚")
    
    return train_paths, train_labels, test_paths, test_labels, test_diseases

def load_sk_data_only():
    """SKデータ（良性）のみを読み込み（ISIC版と同じ）"""
    
    base_path = "/Users/iinuma/Desktop/ダーモ"
    sk_folder = os.path.join(base_path, "SK")
    
    sk_paths = []
    sk_labels = []
    
    if os.path.exists(sk_folder):
        for img_file in os.listdir(sk_folder):
            if img_file.endswith('.JPG'):
                sk_paths.append(os.path.join(sk_folder, img_file))
                sk_labels.append(0)  # 良性
    
    print(f"SK（脂漏性角化症）データ: {len(sk_paths)}枚（良性）")
    
    # 学習用とテスト用に分割（8:2）
    if len(sk_paths) > 0:
        sk_train, sk_test, sk_train_labels, sk_test_labels = train_test_split(
            sk_paths, sk_labels, test_size=0.2, random_state=42
        )
        
        sk_test_diseases = ['脂漏性角化症（良性）'] * len(sk_test)
        
        print(f"SK学習用: {len(sk_train)}枚, テスト用: {len(sk_test)}枚")
        
        return sk_train, sk_train_labels, sk_test, sk_test_labels, sk_test_diseases
    else:
        return [], [], [], [], []

def finetune_on_user_data(model, train_loader, epochs=10):
    """ユーザーデータでファインチューニング（ISIC版と同じ）"""
    
    print("\\n🎯 ユーザーデータでファインチューニング中...")
    
    params = [
        {'params': model.backbone.parameters(), 'lr': 1e-5},
    ]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
        
        train_acc = 100. * train_correct / train_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%')
    
    torch.save(model.state_dict(), 'ham10000_finetuned_model.pth')
    print("✅ ファインチューニング完了")
    return model

def finetune_with_sk_data(model, sk_train_loader, epochs=5):
    """SKデータ（良性）で3段階目のファインチューニング"""
    
    print("\\n🌟 SKデータ（良性）で3段階目ファインチューニング中...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)  # 非常に低い学習率
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(sk_train_loader, desc=f'SK Epoch {epoch+1}/{epochs}'):
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
        
        train_acc = 100. * train_correct / train_total
        print(f'SK Epoch {epoch+1}: Train Acc: {train_acc:.2f}%')
    
    torch.save(model.state_dict(), 'ham10000_balanced_finetuned_model.pth')
    print("✅ HAM10000ベースのバランス調整済みモデルを保存")
    return model

def evaluate_on_test_data(model, test_loader, test_labels, test_diseases):
    """テストデータで評価（ISIC版と同じ）"""
    
    print("\\n📊 テストデータで評価中...")
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='評価中'):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # 全体の精度
    accuracy = np.mean(np.array(all_predictions) == np.array(test_labels))
    print(f"\\n全体精度: {accuracy:.2%}")
    
    # 疾患別の精度
    disease_results = {}
    unique_diseases = list(set(test_diseases))
    
    for disease in unique_diseases:
        disease_indices = [i for i, d in enumerate(test_diseases) if d == disease]
        disease_preds = [all_predictions[i] for i in disease_indices]
        disease_labels = [test_labels[i] for i in disease_indices]
        disease_acc = np.mean(np.array(disease_preds) == np.array(disease_labels))
        disease_results[disease] = {
            'accuracy': disease_acc,
            'total': len(disease_indices),
            'correct': sum(p == l for p, l in zip(disease_preds, disease_labels))
        }
    
    print("\\n疾患別精度:")
    for disease, results in disease_results.items():
        print(f"  {disease}: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # 混同行列
    cm = confusion_matrix(test_labels, all_predictions)
    
    # AUC計算
    if len(set(test_labels)) == 2:
        auc = roc_auc_score(test_labels, all_probabilities)
        print(f"\\nAUC: {auc:.4f}")
    
    # 分類レポート
    print("\\n分類レポート:")
    unique_labels = sorted(list(set(test_labels)))
    if len(unique_labels) == 2:
        target_names = ['良性', '悪性']
    else:
        target_names = [f'クラス{i}' for i in unique_labels]
    
    print(classification_report(test_labels, all_predictions, 
                              target_names=target_names[:len(unique_labels)]))
    
    return cm, disease_results

def main():
    """HAM10000パイプラインのメイン関数"""
    
    print("=" * 60)
    print("🔬 HAM10000ベース 3段階ファインチューニングパイプライン")
    print("   Stage 1: ImageNet → HAM10000")
    print("   Stage 2: HAM10000 → ユーザー悪性データ")
    print("   Stage 3: 悪性学習済み → SK良性データでバランス調整")
    print("=" * 60)
    
    # 1. HAM10000データの準備
    ham_loader = HAM10000Loader()
    X_train_ham, X_test_ham, y_train_ham, y_test_ham = ham_loader.prepare_binary_dataset()
    
    if len(X_train_ham) == 0:
        print("❌ HAM10000データが見つかりません。手動ダウンロードしてください。")
        print("デモ用にISICパイプラインを実行します...")
        
        # ISIC版にフォールバック
        from isic_pretrain_pipeline import main as isic_main
        isic_main()
        return
    
    # HAM10000データのDataLoader
    train_dataset_ham = DermoscopyDataset(X_train_ham, y_train_ham, get_transforms(True))
    val_dataset_ham = DermoscopyDataset(X_test_ham, y_test_ham, get_transforms(False))
    
    train_loader_ham = DataLoader(train_dataset_ham, batch_size=32, shuffle=True, num_workers=2)
    val_loader_ham = DataLoader(val_dataset_ham, batch_size=32, shuffle=False, num_workers=2)
    
    # 2. ユーザーデータ読み込み（悪性のみ、SKは除外）
    user_train_paths, user_train_labels, user_test_paths, user_test_labels, test_diseases = load_user_data()
    
    print(f"\\nユーザー悪性データ統計:")
    print(f"  学習用: {len(user_train_paths)}枚")
    print(f"  テスト用: {len(user_test_paths)}枚")
    
    # ユーザーデータのDataLoader
    train_dataset_user = DermoscopyDataset(user_train_paths, user_train_labels, get_transforms(True))
    train_loader_user = DataLoader(train_dataset_user, batch_size=16, shuffle=True, num_workers=2)
    
    # 3. モデル初期化
    model = PretrainedModel(num_classes=2, model_type='efficientnet').to(device)
    
    # 4. Stage 1: HAM10000データで事前学習
    model = pretrain_on_ham10000(model, train_loader_ham, val_loader_ham, epochs=5)
    
    # 5. Stage 2: ユーザーデータでファインチューニング
    if os.path.exists('ham10000_pretrained_model.pth'):
        model.load_state_dict(torch.load('ham10000_pretrained_model.pth', map_location=device))
        print("✅ HAM10000モデルを再読み込みしました")
    
    model = finetune_on_user_data(model, train_loader_user, epochs=10)
    
    # 6. Stage 3: SKデータでバランス調整
    sk_train_paths, sk_train_labels, sk_test_paths, sk_test_labels, sk_test_diseases = load_sk_data_only()
    
    if len(sk_train_paths) > 0:
        # SKデータのDataLoader
        sk_train_dataset = DermoscopyDataset(sk_train_paths, sk_train_labels, get_transforms(True))
        sk_train_loader = DataLoader(sk_train_dataset, batch_size=16, shuffle=True, num_workers=2)
        
        # HAM10000モデルを再読み込みしてSKでファインチューニング
        if os.path.exists('ham10000_finetuned_model.pth'):
            model.load_state_dict(torch.load('ham10000_finetuned_model.pth', map_location=device))
            print("✅ HAM10000ファインチューニング済みモデルを再読み込み")
        
        model = finetune_with_sk_data(model, sk_train_loader, epochs=5)
        
        # 7. 拡張テストデータで評価（ユーザー悪性 + SK良性）
        print("\\n📊 拡張テストデータセットでの評価...")
        
        combined_test_paths = user_test_paths + sk_test_paths
        combined_test_labels = user_test_labels + sk_test_labels
        combined_test_diseases = test_diseases + sk_test_diseases
        
        combined_test_dataset = DermoscopyDataset(combined_test_paths, combined_test_labels, get_transforms(False))
        combined_test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        cm, disease_results = evaluate_on_test_data(
            model, combined_test_loader, combined_test_labels, combined_test_diseases
        )
        
        print("\\n🎯 HAM10000ベースモデルの改善効果:")
        print(f"   悪性テストデータ: {len([l for l in combined_test_labels if l == 1])}枚")
        print(f"   良性テストデータ: {len([l for l in combined_test_labels if l == 0])}枚")
        
    else:
        print("⚠️ SKデータが見つかりません。従来の評価を実行...")
        
        # 従来のユーザーデータのみで評価
        test_dataset_user = DermoscopyDataset(user_test_paths, user_test_labels, get_transforms(False))
        test_loader_user = DataLoader(test_dataset_user, batch_size=16, shuffle=False, num_workers=2)
        
        cm, disease_results = evaluate_on_test_data(
            model, test_loader_user, user_test_labels, test_diseases
        )
    
    print("\\n✅ HAM10000ベース3段階ファインチューニングパイプライン完了!")
    print("\\n📁 生成されたモデル:")
    print("   • ham10000_pretrained_model.pth: Stage 1完了（HAM10000のみ）")
    print("   • ham10000_finetuned_model.pth: Stage 2完了（悪性データ追加）")
    print("   • ham10000_balanced_finetuned_model.pth: Stage 3完了（良性データでバランス調整）")

if __name__ == "__main__":
    main()