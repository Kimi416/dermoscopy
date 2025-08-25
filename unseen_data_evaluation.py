"""
未見データ評価システム
完全に学習に使用していないデータでの評価
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
import numpy as np
from PIL import Image
import os
import glob
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

class UnseenDataEvaluator:
    """未見データ評価システム"""
    
    def __init__(self):
        self.results = {}
        
    def collect_all_data(self, base_path='/Users/iinuma/Desktop/ダーモ'):
        """全データ収集と分割"""
        print("📊 データ収集と未見データ分離")
        print("=" * 60)
        
        all_data = {
            'paths': [],
            'labels': [],
            'diseases': [],
            'patient_ids': []
        }
        
        for disease, info in DISEASE_MAPPING.items():
            disease_dir = os.path.join(base_path, disease)
            if not os.path.exists(disease_dir):
                continue
            
            patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
            image_paths = []
            for pattern in patterns:
                image_paths.extend(glob.glob(os.path.join(disease_dir, pattern)))
            
            label = 1 if info['type'] == 'malignant' else 0
            
            # 各画像のデータを収集
            for i, img_path in enumerate(image_paths):
                all_data['paths'].append(img_path)
                all_data['labels'].append(label)
                all_data['diseases'].append(disease)
                # 患者IDを生成（ファイル名ベース）
                filename = os.path.basename(img_path)
                patient_id = f"{disease}_{filename.split('.')[0]}"
                all_data['patient_ids'].append(patient_id)
            
            print(f"   {disease}: {len(image_paths)}枚 ({'悪性' if label == 1 else '良性'})")
        
        total_images = len(all_data['paths'])
        print(f"\n📈 データ総数: {total_images}枚")
        
        return all_data
    
    def create_unseen_test_set(self, all_data, test_ratio=0.2):
        """完全に独立したテストセットの作成"""
        print(f"\n🔄 未見テストセット作成（{test_ratio:.0%}をホールドアウト）")
        print("-" * 40)
        
        # 疾患ごとに層化してテストセットを作成
        train_indices = []
        test_indices = []
        
        for disease in DISEASE_MAPPING.keys():
            # 該当疾患のインデックスを取得
            disease_indices = [i for i, d in enumerate(all_data['diseases']) if d == disease]
            
            if len(disease_indices) < 2:
                # データが少なすぎる場合は全て訓練用に
                train_indices.extend(disease_indices)
                continue
            
            # 層化分割
            n_test = max(1, int(len(disease_indices) * test_ratio))
            np.random.seed(42)  # 再現性のため
            np.random.shuffle(disease_indices)
            
            test_indices.extend(disease_indices[:n_test])
            train_indices.extend(disease_indices[n_test:])
            
            print(f"   {disease}: 訓練 {len(disease_indices) - n_test}枚, テスト {n_test}枚")
        
        # データを分割
        train_data = {
            'paths': [all_data['paths'][i] for i in train_indices],
            'labels': [all_data['labels'][i] for i in train_indices],
            'diseases': [all_data['diseases'][i] for i in train_indices],
            'patient_ids': [all_data['patient_ids'][i] for i in train_indices]
        }
        
        test_data = {
            'paths': [all_data['paths'][i] for i in test_indices],
            'labels': [all_data['labels'][i] for i in test_indices],
            'diseases': [all_data['diseases'][i] for i in test_indices],
            'patient_ids': [all_data['patient_ids'][i] for i in test_indices]
        }
        
        print(f"\n📊 分割結果:")
        print(f"   訓練セット: {len(train_data['paths'])}枚")
        print(f"   テストセット: {len(test_data['paths'])}枚（完全に未見）")
        
        # テストセットの分布確認
        test_malignant = sum(test_data['labels'])
        test_benign = len(test_data['labels']) - test_malignant
        print(f"   テストセット内訳: 悪性 {test_malignant}枚, 良性 {test_benign}枚")
        
        return train_data, test_data
    
    def train_model_on_subset(self, train_data, model_type='efficientnet'):
        """訓練セットのみでモデル訓練"""
        print(f"\n🚀 {model_type.upper()} 訓練（訓練セットのみ使用）")
        
        # モデル作成
        if model_type == 'efficientnet':
            model = self.create_efficientnet()
        else:
            model = self.create_resnet()
        
        model = model.to(device)
        
        # データセット作成
        train_dataset = SimpleDataset(train_data['paths'], train_data['labels'])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 訓練
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        model.train()
        for epoch in range(10):  # 簡略化のため10エポック
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 3 == 0:
                print(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
        print(f"✅ 訓練完了")
        return model
    
    def create_efficientnet(self):
        """EfficientNetモデル作成"""
        model = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        return model
    
    def create_resnet(self):
        """ResNetモデル作成"""
        model = resnet50(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        return model
    
    def evaluate_on_unseen_data(self, model, test_data):
        """未見データでの評価"""
        print("\n🧪 未見データでの評価")
        
        model.eval()
        
        # テストデータセット作成
        test_dataset = SimpleDataset(test_data['paths'], test_data['labels'])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 悪性確率
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # メトリクス計算
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # 混同行列
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        
        # 各種メトリクス
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)  # 感度
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # AUC（両クラスが存在する場合のみ）
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall_sensitivity': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': {
                'TP': int(tp), 'TN': int(tn), 
                'FP': int(fp), 'FN': int(fn)
            },
            'total_samples': len(all_labels),
            'malignant_samples': int(sum(all_labels)),
            'benign_samples': len(all_labels) - int(sum(all_labels))
        }
        
        return results
    
    def generate_report(self, results):
        """評価レポート生成"""
        print("\n" + "=" * 80)
        print("📋 未見データ評価レポート")
        print("=" * 80)
        
        print("\n🎯 性能メトリクス（完全に未見のデータ）:")
        print("-" * 50)
        print(f"   精度 (Accuracy): {results['accuracy']:.3f}")
        print(f"   適合率 (Precision): {results['precision']:.3f}")
        print(f"   感度 (Sensitivity/Recall): {results['recall_sensitivity']:.3f}")
        print(f"   特異度 (Specificity): {results['specificity']:.3f}")
        print(f"   F1スコア: {results['f1_score']:.3f}")
        if results['auc'] is not None:
            print(f"   AUC: {results['auc']:.3f}")
        
        print(f"\n📊 混同行列:")
        cm = results['confusion_matrix']
        print(f"   真陽性 (TP): {cm['TP']} | 偽陽性 (FP): {cm['FP']}")
        print(f"   偽陰性 (FN): {cm['FN']} | 真陰性 (TN): {cm['TN']}")
        
        print(f"\n📈 データ分布:")
        print(f"   テスト総数: {results['total_samples']}枚")
        print(f"   悪性: {results['malignant_samples']}枚")
        print(f"   良性: {results['benign_samples']}枚")
        
        # 臨床的解釈
        print(f"\n🏥 臨床的解釈:")
        if results['recall_sensitivity'] >= 0.9:
            print("   ✅ 高い感度 - 悪性の見逃しが少ない")
        else:
            print("   ⚠️ 感度に改善の余地あり")
        
        if results['specificity'] >= 0.9:
            print("   ✅ 高い特異度 - 良性を正しく判定")
        else:
            print("   ⚠️ 特異度に改善の余地あり")
        
        return results

class SimpleDataset(Dataset):
    """シンプルなデータセット"""
    
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

def check_for_new_images(base_path='/Users/iinuma/Desktop/ダーモ'):
    """新しい画像の確認"""
    print("\n🔍 新規画像の確認")
    print("-" * 40)
    
    # 特別なテスト画像の確認
    special_images = []
    
    # test.JPG
    test_jpg = os.path.join(base_path, 'test.JPG')
    if os.path.exists(test_jpg):
        special_images.append(test_jpg)
        print(f"   ✅ test.JPG 発見")
    
    # images.jpeg
    images_jpeg = os.path.join(base_path, 'images.jpeg')
    if os.path.exists(images_jpeg):
        special_images.append(images_jpeg)
        print(f"   ✅ images.jpeg 発見")
    
    # 新規追加画像の検索
    recent_images = []
    import time
    current_time = time.time()
    
    for root, dirs, files in os.walk(base_path):
        # 疾患フォルダ以外は除外
        if any(disease in root for disease in DISEASE_MAPPING.keys()):
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                # 最近追加されたファイル（24時間以内）
                if current_time - os.path.getmtime(file_path) < 86400:
                    recent_images.append(file_path)
    
    if recent_images:
        print(f"   📁 最近追加された画像: {len(recent_images)}枚")
        for img in recent_images[:5]:  # 最初の5枚表示
            print(f"      - {os.path.basename(img)}")
    
    return special_images, recent_images

def main():
    """メイン実行"""
    print("🔬 未見データ評価システム")
    print("   完全に学習に使用していないデータでの真の性能評価")
    print("=" * 80)
    
    evaluator = UnseenDataEvaluator()
    
    # 新規画像の確認
    special_images, recent_images = check_for_new_images()
    
    if special_images or recent_images:
        print("\n💡 追加画像を使用した評価が可能です")
    
    # 全データ収集
    all_data = evaluator.collect_all_data()
    
    # 訓練・テストセット分割（80:20）
    train_data, test_data = evaluator.create_unseen_test_set(all_data, test_ratio=0.2)
    
    # モデル訓練（訓練セットのみ使用）
    model = evaluator.train_model_on_subset(train_data, model_type='efficientnet')
    
    # 未見データでの評価
    results = evaluator.evaluate_on_unseen_data(model, test_data)
    
    # レポート生成
    final_results = evaluator.generate_report(results)
    
    # 結果保存
    with open('/Users/iinuma/Desktop/ダーモ/unseen_data_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 結果を保存: unseen_data_evaluation_results.json")
    print(f"\n🎉 未見データ評価完了！")
    
    # ユーザーへの提案
    print(f"\n" + "=" * 80)
    print("💡 より正確な評価のために：")
    print("=" * 80)
    print("1. 新しい診断画像を追加してください")
    print("2. 各疾患につき10-20枚の新規画像があると理想的です")
    print("3. 特にSKの追加画像があると誤分類改善の検証に有効です")

if __name__ == "__main__":
    main()