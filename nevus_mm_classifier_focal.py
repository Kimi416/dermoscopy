"""
Nevus vs Melanoma分類器（FocalLoss版）
母斑と悪性黒色腫の高精度分類
"""
import torch, numpy as np, os, glob
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from utils_training import make_weighted_sampler, FocalLoss
import json

device = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')

class NevusMMDataset(Dataset):
    def __init__(self, image_paths, labels, is_training=True, img_size=320):
        self.image_paths = image_paths
        self.labels = labels
        if is_training:
            self.t = transforms.Compose([
                transforms.Resize((img_size+32, img_size+32)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.t = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, i):
        try:
            x = Image.open(self.image_paths[i]).convert('RGB')
            return self.t(x), self.labels[i]
        except Exception as e:
            print(f"画像エラー: {self.image_paths[i]} - {e}")
            return torch.zeros(3, 320, 320), self.labels[i]

class NevusMMNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        from torchvision.models import efficientnet_v2_s
        self.model = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        return self.model(x)

def collect_data(base_path='/Users/iinuma/Desktop/ダーモ'):
    """データ収集"""
    print("📁 Nevus vs MM データ収集中...")
    
    image_paths = []
    labels = []
    
    # MM_combinedフォルダから母斑とMM画像を収集
    mm_combined_dir = os.path.join(base_path, 'MM_combined')
    if os.path.exists(mm_combined_dir):
        for img_file in glob.glob(os.path.join(mm_combined_dir, 'ham_*.jpg')):
            image_paths.append(img_file)
            # ファイル名からラベル判定
            if 'ham_mm_' in os.path.basename(img_file):
                labels.append(1)  # MM
            else:
                labels.append(0)  # Nevus
    
    # 従来のMMフォルダも追加
    mm_dir = os.path.join(base_path, 'MM')
    if os.path.exists(mm_dir):
        for img_file in glob.glob(os.path.join(mm_dir, '*.JPG')):
            image_paths.append(img_file)
            labels.append(1)  # MM
    
    print(f"✅ 合計: {len(image_paths)}枚 (MM: {sum(labels)}, Nevus: {len(labels)-sum(labels)})")
    return image_paths, labels

def train_nevus_mm_focal(output_dir='/Users/iinuma/Desktop/ダーモ/nevusmm_focal_weights', 
                         n_folds=5, epochs=8):
    """Nevus vs MM分類器の学習（FocalLoss版）"""
    print("🚀 Nevus vs MM分類器学習開始（FocalLoss版）")
    print("=" * 60)
    
    # データ収集
    image_paths, labels = collect_data()
    X = np.array(image_paths)
    y = np.array(labels)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 層化K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n📂 Fold {fold + 1}/{n_folds}")
        print("-" * 40)
        
        # データ分割
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  訓練: {len(X_train)}枚 (MM: {sum(y_train)})")
        print(f"  検証: {len(X_val)}枚 (MM: {sum(y_val)})")
        
        # データセット作成
        train_dataset = NevusMMDataset(X_train, y_train, is_training=True)
        val_dataset = NevusMMDataset(X_val, y_val, is_training=False)
        
        # WeightedSamplerで不均衡対策
        sampler = make_weighted_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # モデル作成
        model = NevusMMNet().to(device)
        
        # FocalLossでMM（悪性）を重視
        alpha = torch.tensor([0.3, 0.7])  # MM側を重視
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_auc = 0
        best_sens = 0
        
        for epoch in range(epochs):
            # 訓練
            model.train()
            train_loss = 0
            
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 検証
            model.eval()
            val_probs = []
            val_true = []
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(targets.numpy())
            
            # メトリクス計算
            val_probs = np.array(val_probs)
            val_true = np.array(val_true)
            
            auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0
            
            # MM感度重視の閾値探索
            best_threshold = 0.5
            for t in np.linspace(0.1, 0.9, 81):
                pred = (val_probs >= t).astype(int)
                tp = ((val_true == 1) & (pred == 1)).sum()
                fn = ((val_true == 1) & (pred == 0)).sum()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if sens > best_sens:
                    best_sens = sens
                    best_threshold = t
            
            scheduler.step()
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
                      f"AUC={auc:.4f}, MM感度={best_sens:.3f} (閾値={best_threshold:.2f})")
            
            # モデル保存
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'auc': auc,
                    'sensitivity': best_sens,
                    'threshold': best_threshold,
                    'fold': fold
                }, f"{output_dir}/nevusmm_focal_fold{fold}.pth")
        
        fold_results.append({
            'fold': fold,
            'auc': best_auc,
            'sensitivity': best_sens
        })
        
        print(f"✅ Fold {fold+1} 完了: AUC={best_auc:.4f}, MM感度={best_sens:.3f}")
    
    # 結果保存
    with open(f"{output_dir}/training_results.json", 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    print(f"\n🎉 Nevus vs MM分類器学習完了（FocalLoss版）！")
    print(f"  平均AUC: {np.mean([r['auc'] for r in fold_results]):.4f}")
    print(f"  平均MM感度: {np.mean([r['sensitivity'] for r in fold_results]):.3f}")

def predict_mm_prob_focal(image_paths, weights_dir='/Users/iinuma/Desktop/ダーモ/nevusmm_focal_weights'):
    """MM確率予測（FocalLoss版アンサンブル）"""
    models = []
    
    # 全foldモデル読み込み
    for fold in range(5):
        model_path = f"{weights_dir}/nevusmm_focal_fold{fold}.pth"
        if os.path.exists(model_path):
            model = NevusMMNet().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
    
    if not models:
        print("⚠️ FocalLoss版Nevus-MMモデルが見つかりません")
        return np.array([0.5] * len(image_paths))
    
    # 推論
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    all_probs = []
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # アンサンブル予測
            probs = []
            with torch.no_grad():
                for model in models:
                    output = model(image_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].item()
                    probs.append(prob)
            
            all_probs.append(np.mean(probs))
        except Exception as e:
            print(f"⚠️ MM予測エラー {img_path}: {e}")
            all_probs.append(0.5)
    
    return np.array(all_probs)

if __name__ == "__main__":
    train_nevus_mm_focal()