"""
過学習対策版 HAM10000ダーモスコピー分類パイプライン
Data Augmentation + 正則化 + Early Stopping + Cross-Validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import time

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️ 使用デバイス: {device}")

class ImprovedDermoscopyModel(nn.Module):
    """改善版ダーモスコピー分類モデル（過学習対策済み）"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        # EfficientNet-v2-S バックボーン
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        
        # 改善されたヘッド（より強い正則化）
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),  # 最終層は少し弱め
            nn.Linear(128, num_classes)
        )
        
        # Backbone の一部を凍結（過学習防止）
        self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """初期層を凍結して過学習を防止"""
        # 最初の2つのブロックを凍結
        for i, (name, param) in enumerate(self.backbone.features.named_parameters()):
            if i < 10:  # 最初の10層を凍結
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

class AugmentedDermoscopyDataset(Dataset):
    """強化されたデータ拡張付きデータセット"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training
        
        if transform is None:
            if is_training:
                # 訓練時: 強力なデータ拡張
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=30),
                    transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1
                    ),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
                ])
            else:
                # 検証時: 基本的な前処理のみ
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

class EarlyStopping:
    """Early Stopping実装"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        return False

def create_learning_rate_scheduler(optimizer, mode='cosine'):
    """学習率スケジューラー作成"""
    if mode == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    elif mode == 'reduce':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def load_ham10000_data():
    """HAM10000データの読み込み"""
    print("📊 HAM10000データを読み込み中...")
    
    # メタデータ読み込み
    metadata_path = "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"メタデータが見つかりません: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    
    # 画像パス構築
    image_dirs = [
        "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_images_part_1",
        "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_images_part_2"
    ]
    
    image_paths = []
    labels = []
    
    # 7クラスから2クラスへのマッピング
    benign_classes = ['bkl', 'df', 'nv', 'vasc']  # 良性
    malignant_classes = ['akiec', 'bcc', 'mel']   # 悪性
    
    for _, row in df.iterrows():
        image_id = row['image_id']
        diagnosis = row['dx']
        
        # 画像ファイル探索
        image_path = None
        for img_dir in image_dirs:
            potential_path = os.path.join(img_dir, f"{image_id}.jpg")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path and diagnosis in (benign_classes + malignant_classes):
            image_paths.append(image_path)
            # ラベル設定: 0=良性, 1=悪性
            labels.append(0 if diagnosis in benign_classes else 1)
    
    print(f"✅ HAM10000データ読み込み完了: {len(image_paths)}枚")
    print(f"   良性: {labels.count(0)}枚, 悪性: {labels.count(1)}枚")
    
    return image_paths, labels

def load_user_data():
    """ユーザーデータの読み込み"""
    print("📊 ユーザーデータを読み込み中...")
    
    # 悪性データ
    malignant_dirs = ['AK', 'BCC', 'Bowen病', 'MM']
    malignant_paths = []
    
    for disease in malignant_dirs:
        disease_dir = f"/Users/iinuma/Desktop/ダーモ/{disease}"
        if os.path.exists(disease_dir):
            disease_images = glob.glob(os.path.join(disease_dir, "*.jpg")) + \
                           glob.glob(os.path.join(disease_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(disease_dir, "*.JPG")) + \
                           glob.glob(os.path.join(disease_dir, "*.JPEG"))
            malignant_paths.extend(disease_images)
    
    # 良性データ (SK)
    benign_paths = []
    sk_dir = "/Users/iinuma/Desktop/ダーモ/SK"
    if os.path.exists(sk_dir):
        benign_paths = glob.glob(os.path.join(sk_dir, "*.jpg")) + \
                      glob.glob(os.path.join(sk_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(sk_dir, "*.JPG")) + \
                      glob.glob(os.path.join(sk_dir, "*.JPEG"))
    
    # パスとラベルを結合
    user_paths = malignant_paths + benign_paths
    user_labels = [1] * len(malignant_paths) + [0] * len(benign_paths)
    
    print(f"✅ ユーザーデータ読み込み完了: {len(user_paths)}枚")
    print(f"   良性(SK): {len(benign_paths)}枚, 悪性: {len(malignant_paths)}枚")
    
    return user_paths, user_labels

def train_epoch(model, train_loader, criterion, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 勾配クリッピング（過学習防止）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """1エポックの検証"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 確率とラベルを保存（AUC計算用）
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # 悪性の確率
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    
    # AUC計算
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    return avg_loss, accuracy, auc, all_targets, all_preds, all_probs

def train_with_cross_validation(ham_paths, ham_labels, user_paths, user_labels, n_folds=3):
    """Cross-Validationを使った訓練"""
    print(f"\n🔄 {n_folds}-Fold Cross-Validation開始")
    
    # データを結合
    all_paths = ham_paths + user_paths
    all_labels = ham_labels + user_labels
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"\n📊 Fold {fold + 1}/{n_folds}")
        print("-" * 50)
        
        # データ分割
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        print(f"訓練データ: {len(train_paths)}枚 (良性:{train_labels.count(0)}, 悪性:{train_labels.count(1)})")
        print(f"検証データ: {len(val_paths)}枚 (良性:{val_labels.count(0)}, 悪性:{val_labels.count(1)})")
        
        # データセット作成
        train_dataset = AugmentedDermoscopyDataset(train_paths, train_labels, is_training=True)
        val_dataset = AugmentedDermoscopyDataset(val_paths, val_labels, is_training=False)
        
        # データローダー作成
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # モデル初期化
        model = ImprovedDermoscopyModel(num_classes=2, dropout_rate=0.5).to(device)
        
        # 損失関数とオプティマイザー
        # クラス重み調整
        class_counts = [train_labels.count(0), train_labels.count(1)]
        class_weights = [len(train_labels) / (2 * count) for count in class_counts]
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = create_learning_rate_scheduler(optimizer, mode='cosine')
        
        # Early Stopping
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # 訓練ループ
        best_auc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(50):  # 最大50エポック
            start_time = time.time()
            
            # 訓練
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # 検証
            val_loss, val_acc, val_auc, val_targets, val_preds, val_probs = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # 学習率更新
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()
            
            # 記録
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            if (epoch + 1) % 5 == 0:  # 5エポックごとに表示
                print(f"Epoch {epoch+1:2d}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%, Val AUC: {val_auc:.3f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # ベストモデル保存
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
            
            # Early Stopping チェック
            if early_stopping(val_auc, model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Fold結果保存
        fold_result = {
            'fold': fold + 1,
            'best_auc': best_auc,
            'final_val_acc': val_acc,
            'final_val_auc': val_auc,
            'val_targets': val_targets,
            'val_preds': val_preds,
            'val_probs': val_probs
        }
        fold_results.append(fold_result)
        
        print(f"Fold {fold + 1} 完了: Best AUC = {best_auc:.3f}")
        
        # Foldごとのモデル保存
        fold_model_path = f"/Users/iinuma/Desktop/ダーモ/improved_model_fold_{fold+1}.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'fold': fold + 1,
            'best_auc': best_auc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, fold_model_path)
        print(f"モデル保存: {fold_model_path}")
    
    return fold_results

def evaluate_cross_validation_results(fold_results):
    """Cross-Validation結果の評価"""
    print("\n" + "="*60)
    print("📊 Cross-Validation 結果総括")
    print("="*60)
    
    aucs = [result['best_auc'] for result in fold_results]
    accs = [result['final_val_acc'] for result in fold_results]
    
    print(f"平均AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"平均精度: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")
    print(f"AUC範囲: {min(aucs):.3f} - {max(aucs):.3f}")
    print(f"精度範囲: {min(accs):.1f}% - {max(accs):.1f}%")
    
    # 各Foldの詳細
    for i, result in enumerate(fold_results):
        print(f"\nFold {i+1}: AUC={result['best_auc']:.3f}, Acc={result['final_val_acc']:.1f}%")
    
    # 最良のFoldを特定
    best_fold = max(fold_results, key=lambda x: x['best_auc'])
    print(f"\n🏆 最良Fold: Fold {best_fold['fold']} (AUC: {best_fold['best_auc']:.3f})")
    
    return best_fold

def main():
    """メイン実行関数"""
    print("🔬 過学習対策版 ダーモスコピー分類システム")
    print("   Data Augmentation + 正則化 + Early Stopping + Cross-Validation")
    print("="*80)
    
    try:
        # データ読み込み
        print("\n📂 データ読み込み...")
        ham_paths, ham_labels = load_ham10000_data()
        user_paths, user_labels = load_user_data()
        
        print(f"\n📊 データ統計:")
        print(f"HAM10000: {len(ham_paths)}枚 (良性:{ham_labels.count(0)}, 悪性:{ham_labels.count(1)})")
        print(f"ユーザー: {len(user_paths)}枚 (良性:{user_labels.count(0)}, 悪性:{user_labels.count(1)})")
        print(f"総計: {len(ham_paths) + len(user_paths)}枚")
        
        # Cross-Validation訓練実行
        fold_results = train_with_cross_validation(ham_paths, ham_labels, user_paths, user_labels, n_folds=3)
        
        # 結果評価
        best_fold = evaluate_cross_validation_results(fold_results)
        
        print(f"\n✅ 改善版モデル訓練完了!")
        print(f"🎯 最良モデル: improved_model_fold_{best_fold['fold']}.pth")
        print(f"🏆 最良AUC: {best_fold['best_auc']:.3f}")
        
        print(f"\n🚀 次のステップ:")
        print(f"   python3 predict_improved_model.py")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()