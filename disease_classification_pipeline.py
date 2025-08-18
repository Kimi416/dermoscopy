"""
疾患分類パイプライン
HAM10000事前学習 → ユーザーデータファインチューニング → 疾患別良性・悪性判定
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
import copy
import json
from collections import defaultdict

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用デバイス: {device}")

# 疾患分類定義
DISEASE_MAPPING = {
    'AK': {'type': 'malignant', 'full_name': 'Actinic Keratosis'},
    'BCC': {'type': 'malignant', 'full_name': 'Basal Cell Carcinoma'}, 
    'Bowen病': {'type': 'malignant', 'full_name': 'Bowen Disease'},
    'MM': {'type': 'malignant', 'full_name': 'Malignant Melanoma'},
    'SK': {'type': 'benign', 'full_name': 'Seborrheic Keratosis'}
}

class DiseaseClassificationModel(nn.Module):
    """疾患分類モデル（良性・悪性分類）"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        
        # 分類ヘッド
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class DiseaseDataset(Dataset):
    """疾患データセット"""
    
    def __init__(self, image_paths, labels, disease_names, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.disease_names = disease_names
        
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
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
            disease = self.disease_names[idx]
            return image, label, disease
        except Exception as e:
            print(f"画像読み込みエラー: {self.image_paths[idx]} - {e}")
            # エラー時はダミーデータ
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, self.labels[idx], self.disease_names[idx]

def load_ham10000_data():
    """HAM10000データの読み込み"""
    print("📊 HAM10000データを読み込み中...")
    
    metadata_path = "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"HAM10000メタデータが見つかりません: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    
    image_dirs = [
        "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_images_part_1",
        "/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_images_part_2"
    ]
    
    # HAM10000クラス定義
    benign_classes = ['bkl', 'df', 'nv', 'vasc']  # 良性
    malignant_classes = ['akiec', 'bcc', 'mel']   # 悪性
    
    image_paths = []
    labels = []
    diseases = []
    
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
            # ラベル: 0=良性, 1=悪性
            labels.append(0 if diagnosis in benign_classes else 1)
            diseases.append(f"HAM_{diagnosis}")
    
    print(f"✅ HAM10000データ読み込み完了: {len(image_paths)}枚")
    print(f"   良性: {labels.count(0)}枚, 悪性: {labels.count(1)}枚")
    
    return image_paths, labels, diseases

def load_user_disease_data():
    """ユーザー疾患データの読み込み"""
    print("📊 ユーザー疾患データを読み込み中...")
    
    all_paths = []
    all_labels = []
    all_diseases = []
    
    disease_stats = {}
    
    for disease, info in DISEASE_MAPPING.items():
        disease_dir = f"/Users/iinuma/Desktop/ダーモ/{disease}"
        if not os.path.exists(disease_dir):
            print(f"⚠️ {disease}ディレクトリが見つかりません: {disease_dir}")
            continue
        
        # 画像ファイル取得
        image_files = glob.glob(os.path.join(disease_dir, "*.jpg")) + \
                     glob.glob(os.path.join(disease_dir, "*.JPG")) + \
                     glob.glob(os.path.join(disease_dir, "*.jpeg"))
        
        disease_stats[disease] = {
            'count': len(image_files),
            'type': info['type'],
            'full_name': info['full_name']
        }
        
        for img_path in image_files:
            all_paths.append(img_path)
            all_labels.append(0 if info['type'] == 'benign' else 1)
            all_diseases.append(disease)
    
    print(f"✅ ユーザー疾患データ読み込み完了: {len(all_paths)}枚")
    for disease, stats in disease_stats.items():
        print(f"   {disease} ({stats['full_name']}): {stats['count']}枚 [{stats['type']}]")
    
    return all_paths, all_labels, all_diseases, disease_stats

def pretrain_on_ham10000(model, ham_paths, ham_labels, ham_diseases, epochs=10):
    """HAM10000での事前学習"""
    print(f"\n🔄 HAM10000事前学習開始 ({epochs}エポック)")
    print("-" * 50)
    
    # データセット分割
    np.random.seed(42)
    indices = np.random.permutation(len(ham_paths))
    split_idx = int(0.8 * len(indices))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_paths = [ham_paths[i] for i in train_indices]
    train_labels = [ham_labels[i] for i in train_indices]
    train_diseases = [ham_diseases[i] for i in train_indices]
    
    val_paths = [ham_paths[i] for i in val_indices]
    val_labels = [ham_labels[i] for i in val_indices]
    val_diseases = [ham_diseases[i] for i in val_indices]
    
    # データローダー
    train_dataset = DiseaseDataset(train_paths, train_labels, train_diseases, is_training=True)
    val_dataset = DiseaseDataset(val_paths, val_labels, val_diseases, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 最適化設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
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
            for data, target, _ in val_loader:
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
        
        scheduler.step(val_auc)
        
        print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%, Val AUC: {val_auc:.3f}")
        
        # ベストモデル保存
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    
    # ベストモデル復元
    model.load_state_dict(best_state)
    
    print(f"✅ HAM10000事前学習完了! 最良AUC: {best_auc:.3f}")
    return model

def finetune_on_user_data(model, user_paths, user_labels, user_diseases, epochs=15):
    """ユーザーデータでのファインチューニング"""
    print(f"\n🔄 ユーザーデータファインチューニング開始 ({epochs}エポック)")
    print("-" * 50)
    
    # 全データを訓練に使用（小さなデータセットのため）
    # データローダー
    train_dataset = DiseaseDataset(user_paths, user_labels, user_diseases, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # クラス重み調整
    class_counts = [user_labels.count(0), user_labels.count(1)]
    class_weights = [len(user_labels) / (2 * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"クラス重み: 良性={class_weights[0]:.2f}, 悪性={class_weights[1]:.2f}")
    
    # 最適化設定（低い学習率でファインチューニング）
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
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
        
        print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}%")
    
    print(f"✅ ユーザーデータファインチューニング完了!")
    return model

def evaluate_disease_classification(model, user_paths, user_labels, user_diseases, disease_stats):
    """疾患分類性能評価"""
    print(f"\n📊 疾患分類性能評価")
    print("=" * 60)
    
    # データセット作成
    eval_dataset = DiseaseDataset(user_paths, user_labels, user_diseases, is_training=False)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_diseases = []
    
    with torch.no_grad():
        for data, target, disease in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            probs = torch.softmax(output, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 悪性の確率
            all_diseases.extend(disease)
    
    # 全体評価
    overall_accuracy = accuracy_score(all_targets, all_preds)
    overall_auc = roc_auc_score(all_targets, all_probs)
    
    # 混同行列
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # 感度・特異度計算
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 悪性を正しく検出
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 良性を正しく識別
    
    print(f"🎯 全体性能:")
    print(f"   精度 (Accuracy): {overall_accuracy:.1%}")
    print(f"   AUC: {overall_auc:.3f}")
    print(f"   感度 (Sensitivity): {sensitivity:.1%}")
    print(f"   特異度 (Specificity): {specificity:.1%}")
    
    print(f"\n📋 混同行列:")
    print(f"   実際\\\\予測   良性  悪性")
    print(f"   良性        {tn:4d}  {fp:4d}")
    print(f"   悪性        {fn:4d}  {tp:4d}")
    
    # 疾患別評価
    print(f"\n🔬 疾患別詳細評価:")
    print("-" * 60)
    
    disease_results = {}
    
    for disease in DISEASE_MAPPING.keys():
        if disease not in disease_stats:
            continue
            
        # 疾患別データ抽出
        disease_indices = [i for i, d in enumerate(all_diseases) if d == disease]
        if not disease_indices:
            continue
            
        disease_targets = [all_targets[i] for i in disease_indices]
        disease_preds = [all_preds[i] for i in disease_indices]
        disease_probs = [all_probs[i] for i in disease_indices]
        
        # 疾患別メトリクス
        disease_accuracy = accuracy_score(disease_targets, disease_preds)
        correct_count = sum(1 for t, p in zip(disease_targets, disease_preds) if t == p)
        total_count = len(disease_targets)
        
        # 平均確信度
        avg_confidence = np.mean([max(1-p, p) for p in disease_probs])
        
        disease_results[disease] = {
            'accuracy': disease_accuracy,
            'correct': correct_count,
            'total': total_count,
            'confidence': avg_confidence,
            'true_type': DISEASE_MAPPING[disease]['type']
        }
        
        print(f"{disease} ({DISEASE_MAPPING[disease]['full_name']}):")
        print(f"   正解率: {correct_count}/{total_count} ({disease_accuracy:.1%})")
        print(f"   平均確信度: {avg_confidence:.1%}")
        print(f"   実際のタイプ: {DISEASE_MAPPING[disease]['type']}")
    
    # 良性・悪性別評価
    print(f"\n📈 良性・悪性別評価:")
    print("-" * 40)
    
    benign_diseases = [d for d in DISEASE_MAPPING.keys() if DISEASE_MAPPING[d]['type'] == 'benign']
    malignant_diseases = [d for d in DISEASE_MAPPING.keys() if DISEASE_MAPPING[d]['type'] == 'malignant']
    
    for disease_type, diseases in [('良性', benign_diseases), ('悪性', malignant_diseases)]:
        type_indices = [i for i, d in enumerate(all_diseases) if d in diseases]
        if type_indices:
            type_targets = [all_targets[i] for i in type_indices]
            type_preds = [all_preds[i] for i in type_indices]
            type_accuracy = accuracy_score(type_targets, type_preds)
            correct = sum(1 for t, p in zip(type_targets, type_preds) if t == p)
            total = len(type_targets)
            
            print(f"{disease_type}疾患:")
            print(f"   正解率: {correct}/{total} ({type_accuracy:.1%})")
    
    # 結果保存（型変換してJSON対応）
    def convert_to_json_serializable(obj):
        """NumPy型をJSON対応型に変換"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    results = {
        'overall': {
            'accuracy': float(overall_accuracy),
            'auc': float(overall_auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist()
        },
        'diseases': convert_to_json_serializable(disease_results)
    }
    
    with open('/Users/iinuma/Desktop/ダーモ/disease_classification_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 評価結果を保存しました: disease_classification_results.json")
    
    return results

def main():
    """メイン実行関数"""
    print("🔬 疾患分類パイプライン")
    print("   HAM10000事前学習 → ユーザーデータファインチューニング → 疾患別評価")
    print("=" * 80)
    
    try:
        # 1. データ読み込み
        ham_paths, ham_labels, ham_diseases = load_ham10000_data()
        user_paths, user_labels, user_diseases, disease_stats = load_user_disease_data()
        
        # 2. モデル初期化
        model = DiseaseClassificationModel(num_classes=2, dropout_rate=0.3).to(device)
        
        # 3. HAM10000事前学習
        model = pretrain_on_ham10000(model, ham_paths, ham_labels, ham_diseases, epochs=10)
        
        # 事前学習モデル保存
        torch.save({
            'model_state_dict': model.state_dict(),
            'disease_mapping': DISEASE_MAPPING
        }, '/Users/iinuma/Desktop/ダーモ/ham10000_pretrained_disease_model.pth')
        print("💾 HAM10000事前学習モデルを保存しました")
        
        # 4. ユーザーデータファインチューニング
        model = finetune_on_user_data(model, user_paths, user_labels, user_diseases, epochs=15)
        
        # ファインチューニング済みモデル保存
        torch.save({
            'model_state_dict': model.state_dict(),
            'disease_mapping': DISEASE_MAPPING
        }, '/Users/iinuma/Desktop/ダーモ/disease_classification_model.pth')
        print("💾 疾患分類モデルを保存しました")
        
        # 5. 疾患分類性能評価
        results = evaluate_disease_classification(model, user_paths, user_labels, user_diseases, disease_stats)
        
        print(f"\n🎉 疾患分類パイプライン完了!")
        print(f"🎯 全体精度: {results['overall']['accuracy']:.1%}")
        print(f"🎯 感度: {results['overall']['sensitivity']:.1%}")
        print(f"🎯 特異度: {results['overall']['specificity']:.1%}")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()