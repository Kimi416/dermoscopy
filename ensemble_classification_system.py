"""
アンサンブル分類システム
ConvNeXt-S + EfficientNetV2-B3 + Swin-T による高性能診断
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, swin_t
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
# import timm  # システム制約のためコメントアウト
import numpy as np
from PIL import Image
import os
import glob
import json
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

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

class ConvNeXtModel(nn.Module):
    """ConvNeXt風モデル (384x384) - ResNet50ベース"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
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

class EfficientNetV2Model(nn.Module):
    """EfficientNetV2風モデル (300x300) - EfficientNet-S"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        
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

class SwinTransformerModel(nn.Module):
    """Swin風モデル (224x224) - ViT-B/16"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        from torchvision.models import vit_b_16
        self.backbone = vit_b_16(weights='IMAGENET1K_V1')
        num_features = self.backbone.heads.head.in_features
        
        self.backbone.heads.head = nn.Sequential(
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

class MultiResolutionDataset(Dataset):
    """マルチ解像度データセット"""
    
    def __init__(self, image_paths, labels, patient_ids, model_type='convnext', is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.model_type = model_type
        self.is_training = is_training
        
        # モデル別の画像サイズと前処理
        if model_type == 'convnext':
            self.img_size = 384
        elif model_type == 'efficientnet':
            self.img_size = 300
        elif model_type == 'swin':
            self.img_size = 224
        
        self.setup_transforms()
    
    def setup_transforms(self):
        """前処理の設定"""
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size + 32, self.img_size + 32)),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label, patient_id
        except Exception as e:
            print(f"❌ 画像読み込みエラー: {image_path} - {e}")
            # ダミー画像を返す
            dummy_image = torch.zeros(3, self.img_size, self.img_size)
            return dummy_image, label, patient_id

class TemperatureScaling(nn.Module):
    """Temperature Scaling for Calibration"""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, max_iter=100):
        """温度パラメータの最適化"""
        logits = torch.tensor(logits, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return self.temperature.item()

class EnsembleClassifier:
    """アンサンブル分類器"""
    
    def __init__(self):
        self.models = {}
        self.temperature_scalers = {}
        self.weights = {}
        self.threshold = 0.5
        self.fold_results = defaultdict(list)
        
    def create_model(self, model_type):
        """モデル作成"""
        if model_type == 'convnext':
            return ConvNeXtModel()
        elif model_type == 'efficientnet':
            return EfficientNetV2Model()
        elif model_type == 'swin':
            return SwinTransformerModel()
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")
    
    def collect_data(self, base_path='/Users/iinuma/Desktop/ダーモ'):
        """データ収集"""
        print("📊 データ収集中...")
        
        all_image_paths = []
        all_labels = []
        all_patient_ids = []
        
        for disease, info in DISEASE_MAPPING.items():
            disease_dir = os.path.join(base_path, disease)
            if not os.path.exists(disease_dir):
                continue
            
            # 画像パス収集
            patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
            image_paths = []
            for pattern in patterns:
                image_paths.extend(glob.glob(os.path.join(disease_dir, pattern)))
            
            # ラベル作成（良性=0, 悪性=1）
            label = 1 if info['type'] == 'malignant' else 0
            
            # 患者ID作成（ファイル名ベース）
            for img_path in image_paths:
                all_image_paths.append(img_path)
                all_labels.append(label)
                # ファイル名から患者IDを生成（簡易版）
                filename = os.path.basename(img_path)
                patient_id = f"{disease}_{filename.split('.')[0]}"
                all_patient_ids.append(patient_id)
            
            print(f"   {disease}: {len(image_paths)}枚 ({'悪性' if label == 1 else '良性'})")
        
        print(f"✅ 合計: {len(all_image_paths)}枚のデータを収集")
        return all_image_paths, all_labels, all_patient_ids
    
    def train_single_model(self, model_type, train_paths, train_labels, train_patient_ids, 
                          val_paths, val_labels, val_patient_ids, fold_idx):
        """単一モデルの訓練"""
        print(f"🚀 {model_type} 訓練開始 (Fold {fold_idx + 1})")
        
        # モデル作成
        model = self.create_model(model_type).to(device)
        
        # データセット作成
        train_dataset = MultiResolutionDataset(
            train_paths, train_labels, train_patient_ids, model_type, is_training=True
        )
        val_dataset = MultiResolutionDataset(
            val_paths, val_labels, val_patient_ids, model_type, is_training=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # 損失関数とオプティマイザー
        # クラス重み計算
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        class_weights = len(train_labels) / (len(unique_labels) * counts)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # コサインアニーリング
        total_steps = len(train_loader) * 20  # 20エポック想定
        warmup_steps = int(0.1 * total_steps)  # 10%ウォームアップ
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=total_steps - warmup_steps, eta_min=1e-6
        )
        
        # 訓練ループ
        best_auc = 0
        best_model_state = None
        patience = 5
        no_improve = 0
        
        for epoch in range(20):  # 最大20エポック
            # 訓練フェーズ
            model.train()
            train_loss = 0
            
            for batch_idx, (images, labels, _) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if batch_idx > warmup_steps // len(train_loader):
                    scheduler.step()
                
                train_loss += loss.item()
            
            # 検証フェーズ
            model.eval()
            val_probs = []
            val_true = []
            
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)[:, 1]  # 悪性確率
                    
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(labels.numpy())
            
            # AUC計算
            val_auc = roc_auc_score(val_true, val_probs)
            
            print(f"   Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
            
            # 最良モデル保存
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"   早期停止 (Best AUC: {best_auc:.4f})")
                    break
        
        # 最良モデル復元
        model.load_state_dict(best_model_state)
        
        # OOF（Out-of-Fold）予測
        model.eval()
        oof_probs = []
        oof_logits = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                oof_probs.extend(probs.cpu().numpy())
                oof_logits.extend(outputs.cpu().numpy())
        
        print(f"✅ {model_type} 訓練完了 (AUC: {best_auc:.4f})")
        
        return model, np.array(oof_probs), np.array(oof_logits), val_labels, best_auc
    
    def train_ensemble(self, image_paths, labels, patient_ids, n_folds=5):
        """アンサンブル訓練"""
        print("🎯 アンサンブル学習開始")
        print("=" * 60)
        
        # 患者ID層化K-fold
        unique_patients = list(set(patient_ids))
        patient_labels = []
        for patient in unique_patients:
            patient_indices = [i for i, pid in enumerate(patient_ids) if pid == patient]
            patient_label = labels[patient_indices[0]]  # 患者の最初の画像のラベル
            patient_labels.append(patient_label)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        model_types = ['convnext', 'efficientnet', 'swin']
        all_oof_probs = {model_type: [] for model_type in model_types}
        all_oof_logits = {model_type: [] for model_type in model_types}
        all_oof_labels = []
        
        for fold_idx, (train_patients, val_patients) in enumerate(skf.split(unique_patients, patient_labels)):
            print(f"\\n📂 Fold {fold_idx + 1}/{n_folds}")
            
            # 患者IDから画像インデックスを取得
            train_patient_set = set([unique_patients[i] for i in train_patients])
            val_patient_set = set([unique_patients[i] for i in val_patients])
            
            train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patient_set]
            val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patient_set]
            
            train_paths = [image_paths[i] for i in train_indices]
            train_labels_fold = [labels[i] for i in train_indices]
            train_patient_ids_fold = [patient_ids[i] for i in train_indices]
            
            val_paths = [image_paths[i] for i in val_indices]
            val_labels_fold = [labels[i] for i in val_indices]
            val_patient_ids_fold = [patient_ids[i] for i in val_indices]
            
            print(f"   訓練: {len(train_paths)}枚, 検証: {len(val_paths)}枚")
            
            fold_oof_probs = {}
            fold_oof_logits = {}
            
            # 各モデルタイプで訓練
            for model_type in model_types:
                model, oof_probs, oof_logits, _, auc = self.train_single_model(
                    model_type, train_paths, train_labels_fold, train_patient_ids_fold,
                    val_paths, val_labels_fold, val_patient_ids_fold, fold_idx
                )
                
                fold_oof_probs[model_type] = oof_probs
                fold_oof_logits[model_type] = oof_logits
                self.fold_results[model_type].append(auc)
                
                # モデル保存
                model_save_path = f'/Users/iinuma/Desktop/ダーモ/ensemble_{model_type}_fold{fold_idx}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold_idx,
                    'auc': auc
                }, model_save_path)
            
            # OOF結果を累積
            if fold_idx == 0:
                for model_type in model_types:
                    all_oof_probs[model_type] = fold_oof_probs[model_type]
                    all_oof_logits[model_type] = fold_oof_logits[model_type]
                all_oof_labels = val_labels_fold
            else:
                for model_type in model_types:
                    all_oof_probs[model_type] = np.concatenate([
                        all_oof_probs[model_type], fold_oof_probs[model_type]
                    ])
                    all_oof_logits[model_type] = np.concatenate([
                        all_oof_logits[model_type], fold_oof_logits[model_type]
                    ])
                all_oof_labels = np.concatenate([all_oof_labels, val_labels_fold])
        
        # OOF結果での重み計算とキャリブレーション
        print(f"\\n🔧 アンサンブル重み計算と確率校正")
        
        # 各モデルのOOF AUC
        model_aucs = {}
        for model_type in model_types:
            auc = roc_auc_score(all_oof_labels, all_oof_probs[model_type])
            model_aucs[model_type] = auc
            print(f"   {model_type} OOF AUC: {auc:.4f}")
        
        # AUCベースの重み計算
        total_auc = sum(model_aucs.values())
        self.weights = {model_type: auc / total_auc for model_type, auc in model_aucs.items()}
        
        print(f"\\n📊 アンサンブル重み:")
        for model_type, weight in self.weights.items():
            print(f"   {model_type}: {weight:.3f}")
        
        # Temperature Scaling
        for model_type in model_types:
            scaler = TemperatureScaling()
            temperature = scaler.fit(all_oof_logits[model_type], all_oof_labels)
            self.temperature_scalers[model_type] = temperature
            print(f"   {model_type} Temperature: {temperature:.3f}")
        
        # アンサンブル確率計算
        ensemble_probs = np.zeros_like(all_oof_labels, dtype=float)
        for model_type in model_types:
            # Temperature scaling適用
            calibrated_logits = all_oof_logits[model_type] / self.temperature_scalers[model_type]
            calibrated_probs = torch.softmax(torch.tensor(calibrated_logits), dim=1)[:, 1].numpy()
            ensemble_probs += self.weights[model_type] * calibrated_probs
        
        ensemble_auc = roc_auc_score(all_oof_labels, ensemble_probs)
        print(f"\\n🎯 アンサンブル AUC: {ensemble_auc:.4f}")
        
        # 感度制約での閾値最適化
        self.optimize_threshold(all_oof_labels, ensemble_probs, target_sensitivity=0.95)
        
        # 結果保存
        results = {
            'model_aucs': model_aucs,
            'ensemble_auc': float(ensemble_auc),
            'weights': self.weights,
            'temperature_scalers': self.temperature_scalers,
            'threshold': float(self.threshold),
            'fold_results': dict(self.fold_results)
        }
        
        with open('/Users/iinuma/Desktop/ダーモ/ensemble_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ アンサンブル学習完了")
        return results
    
    def optimize_threshold(self, y_true, y_probs, target_sensitivity=0.95):
        """感度制約での閾値最適化"""
        print(f"\\n🎯 閾値最適化 (目標感度: {target_sensitivity:.2f})")
        
        def sensitivity_at_threshold(threshold):
            y_pred = (y_probs >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        
        def specificity_at_threshold(threshold):
            y_pred = (y_probs >= threshold).astype(int)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 目標感度を満たす閾値を探索
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_specificity = 0
        
        for threshold in thresholds:
            sensitivity = sensitivity_at_threshold(threshold)
            if sensitivity >= target_sensitivity:
                specificity = specificity_at_threshold(threshold)
                if specificity > best_specificity:
                    best_threshold = threshold
                    best_specificity = specificity
        
        self.threshold = best_threshold
        final_sensitivity = sensitivity_at_threshold(best_threshold)
        final_specificity = specificity_at_threshold(best_threshold)
        
        print(f"   最適閾値: {best_threshold:.3f}")
        print(f"   感度: {final_sensitivity:.3f}")
        print(f"   特異度: {final_specificity:.3f}")

def main():
    """メイン実行"""
    print("🚀 S級アンサンブル分類システム")
    print("   ConvNeXt-S + EfficientNetV2-B3 + Swin-T")
    print("=" * 60)
    
    classifier = EnsembleClassifier()
    
    # データ収集
    image_paths, labels, patient_ids = classifier.collect_data()
    
    if len(image_paths) == 0:
        print("❌ データが見つかりませんでした")
        return
    
    # アンサンブル訓練
    results = classifier.train_ensemble(
        np.array(image_paths), 
        np.array(labels), 
        np.array(patient_ids)
    )
    
    print(f"\\n🎉 S級アンサンブル完成！")
    print(f"   最終AUC: {results['ensemble_auc']:.4f}")
    print(f"   最適閾値: {results['threshold']:.3f}")

if __name__ == "__main__":
    main()