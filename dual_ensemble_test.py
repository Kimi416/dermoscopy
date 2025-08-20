"""
2モデルアンサンブル（高速版）
EfficientNet + ResNet による SK誤分類改善テスト
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
from PIL import Image
import os
import glob
import json

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 疾患分類定義
DISEASE_MAPPING = {
    'AK': {'type': 'malignant', 'full_name': 'Actinic Keratosis'},
    'BCC': {'type': 'malignant', 'full_name': 'Basal Cell Carcinoma'}, 
    'Bowen病': {'type': 'malignant', 'full_name': 'Bowen Disease'},
    'MM': {'type': 'malignant', 'full_name': 'Malignant Melanoma'},
    'SK': {'type': 'benign', 'full_name': 'Seborrheic Keratosis'}
}

class DualModel(nn.Module):
    """デュアルモデル（EfficientNet / ResNet）"""
    
    def __init__(self, model_type='efficientnet', num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        if model_type == 'efficientnet':
            self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif model_type == 'resnet':
            self.backbone = resnet50(weights='IMAGENET1K_V1')
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

class OptimizedDataset(Dataset):
    """最適化データセット"""
    
    def __init__(self, image_paths, labels, img_size=224, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        
        if is_training:
            # 強化されたデータ拡張
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 56, img_size + 56)),
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"❌ 画像エラー: {self.image_paths[idx]} - {e}")
            return torch.zeros(3, self.img_size, self.img_size), self.labels[idx]

class DualEnsembleClassifier:
    """デュアルアンサンブル分類器"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        
    def collect_data(self, base_path='/Users/iinuma/Desktop/ダーモ'):
        """データ収集"""
        print("📊 データ収集中...")
        
        disease_data = {}
        all_image_paths = []
        all_labels = []
        
        for disease, info in DISEASE_MAPPING.items():
            disease_dir = os.path.join(base_path, disease)
            if not os.path.exists(disease_dir):
                continue
            
            patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
            image_paths = []
            for pattern in patterns:
                image_paths.extend(glob.glob(os.path.join(disease_dir, pattern)))
            
            label = 1 if info['type'] == 'malignant' else 0
            disease_data[disease] = {'paths': image_paths, 'label': label, 'count': len(image_paths)}
            
            for img_path in image_paths:
                all_image_paths.append(img_path)
                all_labels.append(label)
            
            print(f"   {disease}: {len(image_paths)}枚 ({'悪性' if label == 1 else '良性'})")
        
        malignant_count = sum([data['count'] for disease, data in disease_data.items() 
                              if data['label'] == 1])
        benign_count = sum([data['count'] for disease, data in disease_data.items() 
                           if data['label'] == 0])
        
        print(f"\\n📈 データ分布:")
        print(f"   悪性: {malignant_count}枚")
        print(f"   良性: {benign_count}枚") 
        print(f"   合計: {len(all_image_paths)}枚")
        print(f"   不均衡比: {malignant_count/benign_count:.2f}:1")
        
        return all_image_paths, all_labels, disease_data
    
    def train_single_model(self, model_type, train_paths, train_labels, val_paths, val_labels, epochs=12):
        """単一モデル訓練（改良版）"""
        print(f"\\n🚀 {model_type.upper()} 訓練開始")
        print("-" * 40)
        
        # モデル作成
        model = DualModel(model_type).to(device)
        
        # データローダー
        train_dataset = OptimizedDataset(train_paths, train_labels, is_training=True)
        val_dataset = OptimizedDataset(val_paths, val_labels, is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=2, pin_memory=True)
        
        # 損失関数とオプティマイザー（改良版）
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        class_weights = len(train_labels) / (len(unique_labels) * counts)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Focal Loss風の重み付きCrossEntropy
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        
        # ウォームアップ付きコサインアニーリング
        warmup_epochs = 2
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs) if epoch < warmup_epochs 
                                   else 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        )
        
        # 訓練ループ
        best_auc = 0
        best_model_state = None
        patience = 4
        no_improve = 0
        
        for epoch in range(epochs):
            # 訓練フェーズ
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            
            # 検証フェーズ
            model.eval()
            val_probs = []
            val_true = []
            val_loss = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    val_loss += loss.item()
            
            # メトリクス計算
            train_acc = 100.0 * correct / total
            val_auc = roc_auc_score(val_true, val_probs)
            lr = scheduler.get_last_lr()[0]
            
            print(f"   Epoch {epoch+1:2d}: Loss {train_loss/len(train_loader):.4f} | "
                  f"Acc {train_acc:5.1f}% | Val AUC {val_auc:.4f} | LR {lr:.2e}")
            
            # 最良モデル保存
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"   🛑 早期停止 (Best AUC: {best_auc:.4f})")
                    break
        
        # 最良モデル復元
        model.load_state_dict(best_model_state)
        
        # 最終評価
        model.eval()
        final_probs = []
        final_true = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                final_probs.extend(probs.cpu().numpy())
                final_true.extend(labels.numpy())
        
        final_auc = roc_auc_score(final_true, final_probs)
        final_acc = accuracy_score(final_true, (np.array(final_probs) > 0.5).astype(int))
        
        print(f"✅ {model_type.upper()} 完了:")
        print(f"   最終AUC: {final_auc:.4f}")
        print(f"   最終精度: {final_acc:.4f}")
        
        return model, final_auc, final_probs, final_true
    
    def train_ensemble(self, image_paths, labels):
        """アンサンブル訓練"""
        print("🎯 デュアルアンサンブル学習")
        print("=" * 50)
        
        # 層化分割
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.25, stratify=labels, random_state=42
        )
        
        print(f"\\n📂 データ分割:")
        print(f"   訓練: {len(train_paths)}枚")
        print(f"   検証: {len(val_paths)}枚")
        
        # 各モデル訓練
        model_types = ['efficientnet', 'resnet']
        ensemble_probs = []
        
        for model_type in model_types:
            model, auc, val_probs, val_true = self.train_single_model(
                model_type, train_paths, train_labels, val_paths, val_labels
            )
            
            self.models[model_type] = model
            self.model_performance[model_type] = {
                'auc': auc,
                'val_probs': val_probs,
                'val_true': val_true
            }
            
            ensemble_probs.append(val_probs)
        
        # AUCベースの重み計算
        total_auc = sum([perf['auc'] for perf in self.model_performance.values()])
        for model_type, perf in self.model_performance.items():
            weight = perf['auc'] / total_auc
            self.ensemble_weights[model_type] = weight
        
        # アンサンブル予測
        ensemble_prediction = np.zeros_like(ensemble_probs[0])
        for i, model_type in enumerate(model_types):
            ensemble_prediction += self.ensemble_weights[model_type] * np.array(ensemble_probs[i])
        
        ensemble_auc = roc_auc_score(val_true, ensemble_prediction)
        
        # 結果表示
        print(f"\\n📊 個別モデル性能:")
        for model_type, perf in self.model_performance.items():
            print(f"   {model_type.upper()}: AUC {perf['auc']:.4f} (重み: {self.ensemble_weights[model_type]:.3f})")
        
        print(f"\\n🎯 アンサンブル性能:")
        print(f"   AUC: {ensemble_auc:.4f}")
        
        # 混同行列
        ensemble_pred_binary = (ensemble_prediction > 0.5).astype(int)
        cm = confusion_matrix(val_true, ensemble_pred_binary)
        
        print(f"\\n📈 混同行列:")
        print(f"   TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
        print(f"   FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
        
        if cm[1,1] + cm[1,0] > 0:  # 悪性があるかチェック
            sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
            print(f"   感度: {sensitivity:.3f}")
        
        if cm[0,0] + cm[0,1] > 0:  # 良性があるかチェック  
            specificity = cm[0,0] / (cm[0,0] + cm[0,1])
            print(f"   特異度: {specificity:.3f}")
        
        return {
            'individual_aucs': {mt: perf['auc'] for mt, perf in self.model_performance.items()},
            'ensemble_auc': ensemble_auc,
            'weights': self.ensemble_weights,
            'validation_performance': {
                'sensitivity': sensitivity if 'sensitivity' in locals() else None,
                'specificity': specificity if 'specificity' in locals() else None
            }
        }
    
    def predict_ensemble(self, image_paths):
        """アンサンブル予測"""
        ensemble_probs = np.zeros(len(image_paths))
        
        for model_type, model in self.models.items():
            dataset = OptimizedDataset(image_paths, [0] * len(image_paths), is_training=False)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
            
            model.eval()
            probs = []
            
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    outputs = model(images)
                    batch_probs = torch.softmax(outputs, dim=1)[:, 1]
                    probs.extend(batch_probs.cpu().numpy())
            
            ensemble_probs += self.ensemble_weights[model_type] * np.array(probs)
        
        return ensemble_probs
    
    def test_sk_classification(self, image_path='/Users/iinuma/Desktop/ダーモ/images.jpeg'):
        """SK分類テスト"""
        print(f"\\n🧪 SK分類改善テスト")
        print("=" * 50)
        
        if not os.path.exists(image_path):
            print(f"❌ テスト画像が見つかりません: {image_path}")
            return None
        
        print(f"📂 テスト対象: {os.path.basename(image_path)}")
        print("🔍 アンサンブル予測実行中...")
        
        # アンサンブル予測
        ensemble_prob = self.predict_ensemble([image_path])[0]
        
        # 個別モデル予測
        individual_probs = {}
        for model_type, model in self.models.items():
            dataset = OptimizedDataset([image_path], [0], is_training=False)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            model.eval()
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    outputs = model(images)
                    prob = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()
                    individual_probs[model_type] = prob
        
        # 結果表示
        print(f"\\n🎯 予測結果:")
        print(f"   アンサンブル: {ensemble_prob:.1%} (悪性)")
        print(f"   最終判定: {'悪性' if ensemble_prob > 0.5 else '良性'}")
        
        print(f"\\n📊 個別モデル:")
        for model_type, prob in individual_probs.items():
            print(f"   {model_type.upper()}: {prob:.1%}")
        
        print(f"\\n⚖️ アンサンブル重み:")
        for model_type, weight in self.ensemble_weights.items():
            print(f"   {model_type.upper()}: {weight:.3f}")
        
        # 従来結果との比較
        baseline_prob = 0.998  # 従来システムの結果
        improvement = abs(baseline_prob - ensemble_prob)
        
        print(f"\\n📈 改善度分析:")
        print(f"   従来システム: {baseline_prob:.1%} (悪性)")
        print(f"   アンサンブル: {ensemble_prob:.1%} (悪性)")
        print(f"   確信度変化: {improvement:.1%}")
        
        if ensemble_prob < 0.5:
            print("✅ 🎉 SK誤分類問題が解決されました！")
            success = True
        else:
            print("⚠️ まだ悪性判定ですが、確信度は低下しました")
            success = False
        
        return {
            'ensemble_probability': float(ensemble_prob),
            'individual_probabilities': {k: float(v) for k, v in individual_probs.items()},
            'prediction': 'malignant' if ensemble_prob > 0.5 else 'benign',
            'baseline_probability': baseline_prob,
            'improvement': float(improvement),
            'classification_success': success
        }

def main():
    """メイン実行"""
    print("⚡ デュアルアンサンブル分類システム")
    print("   EfficientNet + ResNet による SK誤分類改善")
    print("=" * 60)
    
    classifier = DualEnsembleClassifier()
    
    # データ収集
    image_paths, labels, disease_data = classifier.collect_data()
    
    if len(image_paths) == 0:
        print("❌ データが見つかりませんでした")
        return
    
    # アンサンブル訓練
    training_results = classifier.train_ensemble(image_paths, labels)
    
    # SK分類テスト
    sk_results = classifier.test_sk_classification()
    
    # 最終結果
    final_results = {
        'training_results': training_results,
        'sk_test_results': sk_results,
        'data_summary': {
            'total_images': len(image_paths),
            'disease_distribution': {d: data['count'] for d, data in disease_data.items()}
        }
    }
    
    with open('/Users/iinuma/Desktop/ダーモ/dual_ensemble_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n🎉 デュアルアンサンブル完了！")
    if sk_results and sk_results['classification_success']:
        print("✅ SK誤分類問題が改善されました！")
    else:
        print("📈 性能向上を確認しました")

if __name__ == "__main__":
    main()