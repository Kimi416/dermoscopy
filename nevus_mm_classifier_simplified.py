# nevus_mm_classifier_simplified.py - 簡略化版
import torch, numpy as np, os, glob
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import json

device = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')

class NevusMMDataset(Dataset):
    def __init__(self, image_paths, labels, is_training=True, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        if is_training:
            self.t = transforms.Compose([
                transforms.Resize((img_size+32, img_size+32)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
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
            return torch.zeros(3, 224, 224), self.labels[i]

class NevusMMNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        from torchvision.models import efficientnet_v2_s
        m = efficientnet_v2_s(weights='IMAGENET1K_V1')
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 2)
        )
        self.backbone = m
    def forward(self, x): return self.backbone(x)

def load_nevus_mm_data(nevus_dir='/Users/iinuma/Desktop/ダーモ/nevus', 
                       mm_dir='/Users/iinuma/Desktop/ダーモ/MM_combined'):
    """Nevus vs Melanoma データ読み込み"""
    print("📊 Nevus vs Melanoma データ読み込み")
    
    paths, labels = [], []
    
    # Nevus (0) - サンプリングして500枚に削減
    nevus_files = []
    for pattern in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        nevus_files.extend(glob.glob(os.path.join(nevus_dir, pattern)))
    
    # ランダムサンプリング
    np.random.seed(42)
    if len(nevus_files) > 500:
        nevus_files = np.random.choice(nevus_files, 500, replace=False).tolist()
    
    for path in nevus_files:
        paths.append(path)
        labels.append(0)
    
    print(f"   Nevus: {len(nevus_files)}枚（サンプリング済み）")
    
    # Melanoma (1) - 同じく500枚にサンプリング
    mm_files = []
    for pattern in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        mm_files.extend(glob.glob(os.path.join(mm_dir, pattern)))
    
    if len(mm_files) > 500:
        mm_files = np.random.choice(mm_files, 500, replace=False).tolist()
    
    for path in mm_files:
        paths.append(path)
        labels.append(1)
    
    print(f"   Melanoma: {len(mm_files)}枚（サンプリング済み）")
    print(f"   合計: {len(paths)}枚")
    
    return np.array(paths), np.array(labels)

def train_quick_nevus_mm(nevus_dir, mm_dir, out_dir, n_folds=3, epochs=8):
    """軽量Nevus vs Melanoma分類器"""
    print("🚀 軽量Nevus vs Melanoma 分類器訓練")
    print("=" * 60)
    
    os.makedirs(out_dir, exist_ok=True)
    X, y = load_nevus_mm_data(nevus_dir, mm_dir)
    
    # 画像レベルでの層化K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    cv_scores = []
    
    for f, (tr_indices, va_indices) in enumerate(skf.split(X, y)):
        print(f"\\n📂 Fold {f+1}/{n_folds}")
        print("-" * 40)
        print(f"   訓練: {len(tr_indices)}枚, 検証: {len(va_indices)}枚")
        
        # データセット・モデル準備
        net = NevusMMNet().to(device)
        tr_ds = NevusMMDataset(X[tr_indices], y[tr_indices], is_training=True)
        va_ds = NevusMMDataset(X[va_indices], y[va_indices], is_training=False)
        tr_ld = DataLoader(tr_ds, batch_size=24, shuffle=True, num_workers=1)
        va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=1)
        
        # 損失関数・オプティマイザー
        crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
        
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
            except:
                print(f"   Epoch {e+1:2d}: Loss {train_loss/len(tr_ld):.4f}, AUC 計算不可")
        
        # 最良モデル保存
        if best_state is not None:
            net.load_state_dict(best_state)
            torch.save({'state_dict': best_state, 'auc': best_auc}, 
                      os.path.join(out_dir, f'nevusmm_fold{f}.pth'))
            
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
        else:
            print(f"   ❌ Fold {f+1} 失敗")
    
    # 全体評価
    if len(cv_scores) > 0:
        try:
            overall_auc = roc_auc_score(y, oof)
            mean_cv_auc = np.mean(cv_scores)
            std_cv_auc = np.std(cv_scores)
            
            print(f"\\n🎯 交差検証結果:")
            print(f"   平均AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")
            print(f"   OOF AUC: {overall_auc:.4f}")
            
            # 混同行列
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
        except Exception as e:
            print(f"❌ 評価エラー: {e}")
            return 0.5
    else:
        print("❌ 有効なfoldがありませんでした")
        return 0.5

def predict_mm_prob(image_paths, weights_dir):
    """全foldモデル平均でp(MM)を返す（軽量版）"""
    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    
    # 全foldモデル読み込み
    fold_paths = sorted([p for p in glob.glob(os.path.join(weights_dir, 'nevusmm_fold*.pth'))])
    if not fold_paths:
        print(f"❌ モデルファイルが見つかりません: {weights_dir}")
        return np.array([0.5] * len(image_paths))
    
    nets = []
    for p in fold_paths:
        try:
            net = NevusMMNet().to(device)
            checkpoint = torch.load(p, map_location=device)
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            nets.append(net)
        except Exception as e:
            print(f"⚠️ モデル読み込みエラー {p}: {e}")
    
    if not nets:
        print(f"❌ 有効なモデルがありません")
        return np.array([0.5] * len(image_paths))
    
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
            probs_all.append(0.5)
    
    return np.array(probs_all)

def main():
    """メイン実行（軽量版）"""
    print("🔬 軽量 Nevus vs Melanoma 分類システム")
    print("=" * 80)
    
    # データディレクトリ
    nevus_dir = '/Users/iinuma/Desktop/ダーモ/nevus'
    mm_dir = '/Users/iinuma/Desktop/ダーモ/MM_combined'
    weights_dir = '/Users/iinuma/Desktop/ダーモ/nevusmm_weights'
    
    # 軽量学習実行
    auc = train_quick_nevus_mm(nevus_dir, mm_dir, weights_dir, n_folds=3, epochs=8)
    
    print(f"\\n🎉 軽量 Nevus vs Melanoma 分類器完成!")
    print(f"   最終AUC: {auc:.4f}")
    print(f"   重みファイル: {weights_dir}")
    print("\\n💡 使用方法:")
    print("   from nevus_mm_classifier_simplified import predict_mm_prob")
    print("   probs = predict_mm_prob(image_paths, weights_dir)")

if __name__ == "__main__":
    main()