# nevus_mm_classifier.py
import torch, numpy as np, os, glob
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
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
            # ダミー画像を返す
            return torch.zeros(3, 320, 320), self.labels[i]

class NevusMMNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        from torchvision.models import efficientnet_v2_s
        m = efficientnet_v2_s(weights='IMAGENET1K_V1')
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 2)
        )
        self.backbone = m
    def forward(self, x): return self.backbone(x)

def load_nevus_mm_data(nevus_dir='/Users/iinuma/Desktop/ダーモ/nevus', 
                       mm_dir='/Users/iinuma/Desktop/ダーモ/MM_combined'):
    """Nevus vs Melanoma データ読み込み"""
    print("📊 Nevus vs Melanoma データ読み込み")
    
    paths, labels = [], []
    
    # Nevus (0)
    nevus_patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    nevus_files = []
    for pattern in nevus_patterns:
        nevus_files.extend(glob.glob(os.path.join(nevus_dir, pattern)))
    
    for path in nevus_files:
        paths.append(path)
        labels.append(0)  # Nevus = 0
    
    print(f"   Nevus: {len(nevus_files)}枚")
    
    # Melanoma (1)
    mm_patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    mm_files = []
    for pattern in mm_patterns:
        mm_files.extend(glob.glob(os.path.join(mm_dir, pattern)))
    
    for path in mm_files:
        paths.append(path)
        labels.append(1)  # Melanoma = 1
    
    print(f"   Melanoma: {len(mm_files)}枚")
    print(f"   合計: {len(paths)}枚")
    
    return np.array(paths), np.array(labels)

def train_kfold_save(nevus_dir, mm_dir, out_dir, n_folds=5, epochs=10):
    """K-fold 交差検証で学習・保存"""
    print("🚀 Nevus vs Melanoma 分類器訓練開始")
    print("=" * 60)
    
    os.makedirs(out_dir, exist_ok=True)
    X, y = load_nevus_mm_data(nevus_dir, mm_dir)
    
    # 患者ID層化（より細かい粒度で）
    patient_ids = []
    for path in X:
        filename = os.path.basename(path)
        # より細かい患者ID生成（レシオンIDベース）
        if 'ham_nevus' in filename or 'ham_mm' in filename:
            # HAMデータ: ISIC_xxxxxxx部分を抽出
            parts = filename.split('_')
            if len(parts) >= 4 and 'ISIC' in parts[3]:
                patient_id = parts[3]  # ISIC_xxxxxxx
            else:
                patient_id = filename.split('.')[0]
        elif 'user_' in filename:
            # ユーザーデータ: より粗い粒度
            parts = filename.split('_')
            if len(parts) >= 4:
                patient_id = '_'.join(parts[:4])  # user_type_xxxx_name
            else:
                patient_id = filename.split('.')[0]
        else:
            # その他: ファイル名ベース
            patient_id = filename.split('.')[0]
        patient_ids.append(patient_id)
    
    # 患者ID別でグループ化
    unique_patients = list(set(patient_ids))
    patient_labels = []
    for patient in unique_patients:
        indices = [i for i, pid in enumerate(patient_ids) if pid == patient]
        patient_label = y[indices[0]]  # その患者の最初の画像のラベル
        patient_labels.append(patient_label)
    
    print(f"👥 患者統計:")
    print(f"   総患者数: {len(unique_patients)}")
    print(f"   Nevus患者: {sum([1 for l in patient_labels if l == 0])}")
    print(f"   Melanoma患者: {sum([1 for l in patient_labels if l == 1])}")
    
    # 患者分布が極端に偏っている場合は画像レベル分割を使用
    nevus_patients = sum([1 for l in patient_labels if l == 0])
    melanoma_patients = sum([1 for l in patient_labels if l == 1])
    
    if len(unique_patients) < n_folds or min(nevus_patients, melanoma_patients) < 2:
        print(f"⚠️ 患者分布が不適切（Nevus:{nevus_patients}, Melanoma:{melanoma_patients}）")
        print("   画像レベルで分割します")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_iterator = skf.split(X, y)
        use_patient_split = False
    else:
        print(f"✅ 患者レベル分割を使用")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_iterator = skf.split(unique_patients, patient_labels)
        use_patient_split = True
    oof = np.zeros(len(X))
    cv_scores = []
    
    # 既に完了したfoldをスキップ
    completed_folds = []
    for i in range(n_folds):
        if os.path.exists(os.path.join(out_dir, f'nevusmm_fold{i}.pth')):
            completed_folds.append(i)
    
    for f, (train_indices, val_indices) in enumerate(cv_iterator):
        if f in completed_folds:
            print(f"\\n📂 Fold {f+1}/{n_folds} - 既に完了済み、スキップ")
            continue
            
        print(f"\\n📂 Fold {f+1}/{n_folds}")
        print("-" * 40)
        
        if use_patient_split:
            # 患者ベースで分割
            train_patients = [unique_patients[i] for i in train_indices]
            val_patients = [unique_patients[i] for i in val_indices]
            
            # 画像インデックスを取得
            tr_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
            va_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
            
            print(f"   訓練患者: {len(train_patients)}, 画像: {len(tr_indices)}枚")
            print(f"   検証患者: {len(val_patients)}, 画像: {len(va_indices)}枚")
        else:
            # 画像レベルで分割
            tr_indices = train_indices
            va_indices = val_indices
            
            print(f"   訓練画像: {len(tr_indices)}枚")
            print(f"   検証画像: {len(va_indices)}枚")
        
        # データセット・モデル準備
        net = NevusMMNet().to(device)
        tr_ds = NevusMMDataset(X[tr_indices], y[tr_indices], is_training=True)
        va_ds = NevusMMDataset(X[va_indices], y[va_indices], is_training=False)
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=2)
        va_ld = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=2)
        
        # 損失関数・オプティマイザー
        crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
        best_auc, best_state = 0, None
        
        for e in range(epochs):
            # 訓練
            net.train()
            train_loss = 0
            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(net(xb), yb)
                loss.backward()
                opt.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # 検証
            net.eval()
            probs = []
            with torch.no_grad():
                for xb, yb in va_ld:
                    p = torch.softmax(net(xb.to(device)), dim=1)[:,1].cpu().numpy()
                    probs.extend(p)
            
            try:
                auc = roc_auc_score(y[va_indices], probs)
                if auc > best_auc:
                    best_auc, best_state = auc, net.state_dict()
                
                if e % 2 == 0:
                    print(f"   Epoch {e+1:2d}: Loss {train_loss/len(tr_ld):.4f}, AUC {auc:.4f}")
            except ValueError:
                # AUC計算不可の場合（単一クラスのみ）
                if e % 2 == 0:
                    print(f"   Epoch {e+1:2d}: Loss {train_loss/len(tr_ld):.4f}, AUC 計算不可")
        
        # 最良モデル保存
        if best_state is not None:
            net.load_state_dict(best_state)
            torch.save({'state_dict': best_state, 'auc': best_auc}, 
                      os.path.join(out_dir, f'nevusmm_fold{f}.pth'))
        else:
            # 学習失敗の場合は現在の状態を保存
            torch.save({'state_dict': net.state_dict(), 'auc': 0.5}, 
                      os.path.join(out_dir, f'nevusmm_fold{f}.pth'))
            best_auc = 0.5
        
        # OOF予測
        net.eval()
        probs = []
        with torch.no_grad():
            for xb, yb in va_ld:
                p = torch.softmax(net(xb.to(device)), dim=1)[:,1].cpu().numpy()
                probs.extend(p)
        oof[va_indices] = probs
        cv_scores.append(best_auc)
        
        print(f"   ✅ Fold {f+1} 完了: AUC {best_auc:.4f}")
    
    # OOF全体評価
    overall_auc = roc_auc_score(y, oof)
    mean_cv_auc = np.mean(cv_scores)
    std_cv_auc = np.std(cv_scores)
    
    print(f"\\n🎯 交差検証結果:")
    print(f"   平均AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")
    print(f"   OOF AUC: {overall_auc:.4f}")
    
    # 混同行列（閾値0.5）
    oof_pred = (oof > 0.5).astype(int)
    cm = confusion_matrix(y, oof_pred)
    acc = accuracy_score(y, oof_pred)
    
    print(f"\\n📊 混同行列 (閾値0.5):")
    print(f"   TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]} | TP: {cm[1,1]}")
    print(f"   精度: {acc:.3f}")
    
    if cm[1,1] + cm[1,0] > 0:
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
        print(f"   感度: {sensitivity:.3f}")
    
    if cm[0,0] + cm[0,1] > 0:
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
        print(f"   特異度: {specificity:.3f}")
    
    # 結果保存
    np.save(os.path.join(out_dir, 'nevusmm_oof.npy'), oof)
    
    results = {
        'cv_scores': [float(x) for x in cv_scores],
        'mean_auc': float(mean_cv_auc),
        'std_auc': float(std_cv_auc),
        'oof_auc': float(overall_auc),
        'confusion_matrix': cm.tolist(),
        'accuracy': float(acc),
        'n_folds': n_folds,
        'total_samples': len(y),
        'nevus_samples': int(sum(y == 0)),
        'melanoma_samples': int(sum(y == 1))
    }
    
    with open(os.path.join(out_dir, 'nevusmm_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 結果保存: {out_dir}")
    return overall_auc
    
def predict_mm_prob(image_paths, weights_dir):
    """全foldモデル平均でp(MM)を返す"""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    
    # 全foldモデル読み込み
    fold_paths = sorted([p for p in glob.glob(os.path.join(weights_dir, 'nevusmm_fold*.pth'))])
    if not fold_paths:
        print(f"❌ モデルファイルが見つかりません: {weights_dir}")
        return np.zeros(len(image_paths))
    
    nets = []
    for p in fold_paths:
        net = NevusMMNet().to(device)
        checkpoint = torch.load(p, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        nets.append(net)
    
    print(f"📁 {len(nets)}個のfoldモデル読み込み完了")
    
    probs_all = []
    for img_path in image_paths:
        try:
            x = t(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            ps = []
            with torch.no_grad():
                for net in nets:
                    p = torch.softmax(net(x), dim=1)[:,1].item()
                    ps.append(p)
            probs_all.append(float(np.mean(ps)))
        except Exception as e:
            print(f"⚠️ 予測エラー {img_path}: {e}")
            probs_all.append(0.5)  # デフォルト値
    
    return np.array(probs_all)

def main():
    """メイン実行"""
    print("🔬 Nevus vs Melanoma 分類システム")
    print("=" * 80)
    
    # データディレクトリ
    nevus_dir = '/Users/iinuma/Desktop/ダーモ/nevus'
    mm_dir = '/Users/iinuma/Desktop/ダーモ/MM_combined'
    weights_dir = '/Users/iinuma/Desktop/ダーモ/nevusmm_weights'
    
    # 学習実行
    auc = train_kfold_save(nevus_dir, mm_dir, weights_dir, n_folds=5, epochs=15)
    
    print(f"\\n🎉 Nevus vs Melanoma 分類器完成!")
    print(f"   最終AUC: {auc:.4f}")
    print(f"   重みファイル: {weights_dir}")
    print("\\n💡 使用方法:")
    print("   from nevus_mm_classifier import predict_mm_prob")
    print("   probs = predict_mm_prob(image_paths, weights_dir)")

if __name__ == "__main__":
    main()