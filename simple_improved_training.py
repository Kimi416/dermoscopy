"""
シンプル改善版過学習対策パイプライン
効率的な実装で確実に完了
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
import copy

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️ 使用デバイス: {device}")

class ImprovedModel(nn.Module):
    """改善版モデル（過学習対策）"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        
        # 強化された分類ヘッド（BatchNorm除去でバッチサイズ問題回避）
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 高いDropout率
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # バックボーンの一部を凍結
        self._freeze_layers()
    
    def _freeze_layers(self):
        """初期層を凍結して過学習を防止"""
        for i, (name, param) in enumerate(self.backbone.features.named_parameters()):
            if i < 20:  # 最初の20層を凍結
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

class ImprovedDataset(Dataset):
    """改善版データセット（強力なデータ拡張）"""
    
    def __init__(self, image_paths, labels, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        
        if is_training:
            # 訓練時：強力なデータ拡張
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
            ])
        else:
            # 検証時：基本前処理のみ
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"画像読み込みエラー: {self.image_paths[idx]}")
            # エラー時はダミーデータ返す
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, self.labels[idx]

def load_balanced_subset():
    """バランスの取れたサブセットデータを読み込み"""
    print("📊 バランス調整データセット作成中...")
    
    # HAM10000データ（小さなサブセット）
    metadata_path = "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_metadata.csv"
    df = pd.read_csv(metadata_path)
    
    # 画像ディレクトリ
    image_dirs = [
        "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_images_part_1",
        "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_images_part_2"
    ]
    
    # クラス定義
    benign_classes = ['bkl', 'df', 'nv', 'vasc']
    malignant_classes = ['akiec', 'bcc', 'mel']
    
    # HAM10000データ収集
    ham_benign_paths = []
    ham_malignant_paths = []
    
    for _, row in df.iterrows():
        image_id = row['image_id']
        diagnosis = row['dx']
        
        # 画像パス探索
        image_path = None
        for img_dir in image_dirs:
            potential_path = os.path.join(img_dir, f"{image_id}.jpg")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            if diagnosis in benign_classes:
                ham_benign_paths.append(image_path)
            elif diagnosis in malignant_classes:
                ham_malignant_paths.append(image_path)
    
    # 効率化のため各クラス800枚に制限
    np.random.seed(42)
    if len(ham_benign_paths) > 800:
        ham_benign_paths = np.random.choice(ham_benign_paths, 800, replace=False).tolist()
    if len(ham_malignant_paths) > 800:
        ham_malignant_paths = np.random.choice(ham_malignant_paths, 800, replace=False).tolist()
    
    # ユーザーデータ追加
    user_malignant_paths = []
    for disease in ['AK', 'BCC', 'Bowen病', 'MM']:
        disease_dir = f"/Users/iinuma/Desktop/ダーモ/{disease}"
        if os.path.exists(disease_dir):
            images = glob.glob(os.path.join(disease_dir, "*.jpg")) + \
                    glob.glob(os.path.join(disease_dir, "*.JPG"))
            user_malignant_paths.extend(images)
    
    user_benign_paths = []
    sk_dir = "/Users/iinuma/Desktop/ダーモ/SK"
    if os.path.exists(sk_dir):
        user_benign_paths = glob.glob(os.path.join(sk_dir, "*.jpg")) + \
                           glob.glob(os.path.join(sk_dir, "*.JPG"))
    
    # データ統合
    all_benign = ham_benign_paths + user_benign_paths
    all_malignant = ham_malignant_paths + user_malignant_paths
    
    # さらにバランス調整
    min_class_size = min(len(all_benign), len(all_malignant))
    if len(all_benign) > min_class_size:
        all_benign = np.random.choice(all_benign, min_class_size, replace=False).tolist()
    if len(all_malignant) > min_class_size:
        all_malignant = np.random.choice(all_malignant, min_class_size, replace=False).tolist()
    
    # 最終データセット
    all_paths = all_benign + all_malignant
    all_labels = [0] * len(all_benign) + [1] * len(all_malignant)
    
    # シャッフル
    combined = list(zip(all_paths, all_labels))
    np.random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    all_paths, all_labels = list(all_paths), list(all_labels)
    
    print(f"✅ バランス調整完了: {len(all_paths)}枚")
    print(f"   良性: {all_labels.count(0)}枚")
    print(f"   悪性: {all_labels.count(1)}枚")
    
    return all_paths, all_labels

class EarlyStopping:
    """Early Stopping実装"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
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
                model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        return False

def train_improved_model():
    """改善版モデル訓練"""
    print("\n🚀 改善版モデル訓練開始")
    print("-" * 50)
    
    # データ準備
    all_paths, all_labels = load_balanced_subset()
    
    # 訓練・検証分割
    split_idx = int(0.8 * len(all_paths))
    train_paths = all_paths[:split_idx]
    train_labels = all_labels[:split_idx]
    val_paths = all_paths[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"訓練データ: {len(train_paths)}枚 (良性:{train_labels.count(0)}, 悪性:{train_labels.count(1)})")
    print(f"検証データ: {len(val_paths)}枚 (良性:{val_labels.count(0)}, 悪性:{val_labels.count(1)})")
    
    # データセット・ローダー作成
    train_dataset = ImprovedDataset(train_paths, train_labels, is_training=True)
    val_dataset = ImprovedDataset(val_paths, val_labels, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # モデル初期化
    model = ImprovedModel(num_classes=2).to(device)
    
    # クラス重み計算
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = [len(train_labels) / (2 * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"クラス重み: 良性={class_weights[0]:.2f}, 悪性={class_weights[1]:.2f}")
    
    # 損失関数・オプティマイザー
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)  # 強いWeight Decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    # 訓練ループ
    best_auc = 0
    print("\n🔄 訓練開始...")
    
    for epoch in range(25):  # 最大25エポック
        # 訓練フェーズ
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        val_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                # AUC計算用
                probs = torch.softmax(output, dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # AUC計算
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except:
            val_auc = 0.5
        
        # 学習率スケジューラー
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 結果表示
        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%, Val AUC: {val_auc:.3f} | "
              f"LR: {current_lr:.2e}")
        
        # ベストモデル保存
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"   ✅ 新しいベストAUC: {best_auc:.3f}")
        
        # Early Stopping チェック
        if early_stopping(val_auc, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ 訓練完了! 最良AUC: {best_auc:.3f}")
    
    # モデル保存
    model_path = "/Users/iinuma/Desktop/ダーモ/quick_improved_model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'best_auc': best_auc,
        'final_val_targets': val_targets,
        'final_val_preds': val_preds,
        'final_val_probs': val_probs,
        'model_type': 'improved'
    }, model_path)
    
    print(f"モデル保存: {model_path}")
    
    # 最終評価
    if len(set(val_targets)) > 1:  # 両クラスが存在する場合のみ
        cm = confusion_matrix(val_targets, val_preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, sum(val_targets))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n📊 最終評価結果:")
        print(f"感度 (Sensitivity): {sensitivity:.1%}")
        print(f"特異度 (Specificity): {specificity:.1%}")
        print(f"AUC: {best_auc:.3f}")
        print(f"混同行列:")
        print(f"  実際\\予測   良性  悪性")
        print(f"  良性        {tn:4d}  {fp:4d}")
        print(f"  悪性        {fn:4d}  {tp:4d}")
    
    return model_path

def main():
    """メイン実行関数"""
    print("🔬 シンプル改善版ダーモスコピー分類システム")
    print("   効率的な過学習対策実装")
    print("="*60)
    
    try:
        model_path = train_improved_model()
        
        print(f"\n🚀 次のステップ:")
        print(f"   python3 predict_quick_improved.py")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()