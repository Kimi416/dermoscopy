"""
ISICデータでの事前学習 → ユーザーデータでのファインチューニング・評価パイプライン
"""

import os
import json
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import shutil
import warnings
warnings.filterwarnings('ignore')

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class ISICDownloader:
    """ISICアーカイブから画像をダウンロード（v2 API使用）"""
    
    def __init__(self, output_dir="isic_data"):
        self.output_dir = output_dir
        self.api_base = "https://api.isic-archive.com/api/v2"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/benign", exist_ok=True)
        os.makedirs(f"{output_dir}/malignant", exist_ok=True)
    
    def download_images(self, benign_count=1000, malignant_count=1000):
        """良性・悪性画像をダウンロード"""
        
        print("📥 ISIC v2 APIから画像をダウンロード中...")
        
        # 良性画像（母斑、脂漏性角化症など）
        benign_downloaded = self._download_by_diagnosis(
            ["Nevus", "Solar lentigo", "Seborrheic keratosis"], 
            "benign", 
            benign_count
        )
        
        # 悪性画像（メラノーマ、基底細胞癌など）
        malignant_downloaded = self._download_by_diagnosis(
            ["Melanoma", "Basal cell carcinoma", "Squamous cell carcinoma"], 
            "malignant", 
            malignant_count
        )
        
        print(f"✅ ダウンロード完了: 良性 {benign_downloaded}枚, 悪性 {malignant_downloaded}枚")
    
    def _download_by_diagnosis(self, diagnoses, category, target_count):
        """診断名別にダウンロード（v2 API使用）"""
        
        downloaded = 0
        limit = 50
        cursor = None
        
        with tqdm(total=target_count, desc=f"{category}画像") as pbar:
            
            while downloaded < target_count:
                try:
                    # API リクエスト
                    params = {"limit": limit}
                    if cursor:
                        params["cursor"] = cursor
                    
                    response = requests.get(
                        f"{self.api_base}/images/",
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        print(f"API Error: {response.status_code}")
                        break
                    
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        break
                    
                    # 各画像を処理
                    for item in results:
                        if downloaded >= target_count:
                            break
                        
                        # 診断名をチェック
                        metadata = item.get("metadata", {})
                        clinical = metadata.get("clinical", {})
                        
                        # 診断名の階層をチェック
                        diagnosis_found = False
                        for diag_key in ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5"]:
                            diagnosis = clinical.get(diag_key, "")
                            if any(target_diag.lower() in diagnosis.lower() for target_diag in diagnoses):
                                diagnosis_found = True
                                break
                        
                        if not diagnosis_found:
                            continue
                        
                        # 画像をダウンロード
                        isic_id = item.get("isic_id")
                        if not isic_id:
                            continue
                        
                        img_path = f"{self.output_dir}/{category}/{isic_id}.jpg"
                        
                        # 既存ファイルをスキップ
                        if os.path.exists(img_path):
                            downloaded += 1
                            pbar.update(1)
                            continue
                        
                        # 画像URL取得
                        files = item.get("files", {})
                        full_img = files.get("full", {})
                        img_url = full_img.get("url")
                        
                        if not img_url:
                            continue
                        
                        # 画像ダウンロード
                        img_response = requests.get(img_url, stream=True, timeout=30)
                        
                        if img_response.status_code == 200:
                            try:
                                img = Image.open(BytesIO(img_response.content))
                                # リサイズして保存
                                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                                img.save(img_path, "JPEG", quality=95)
                                downloaded += 1
                                pbar.update(1)
                            except Exception as e:
                                print(f"画像保存エラー {isic_id}: {e}")
                        
                        time.sleep(0.1)  # API制限対策
                    
                    # 次のページへ
                    next_url = data.get("next")
                    if next_url and "cursor=" in next_url:
                        cursor = next_url.split("cursor=")[1].split("&")[0]
                    else:
                        break
                
                except Exception as e:
                    print(f"\nエラー: {e}")
                    time.sleep(5)
                    continue
        
        return downloaded

class PretrainedModel(nn.Module):
    """事前学習用モデル"""
    
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
    """ダーモスコピー画像データセット"""
    
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
            # 画像読み込みエラー時は黒画像を返す
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_train=True):
    """データ変換パイプライン"""
    
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

def load_isic_data():
    """ISICデータの読み込み"""
    
    isic_dir = "isic_data"
    image_paths = []
    labels = []
    
    # 良性画像
    benign_dir = f"{isic_dir}/benign"
    if os.path.exists(benign_dir):
        for img_file in os.listdir(benign_dir):
            if img_file.endswith('.jpg'):
                image_paths.append(os.path.join(benign_dir, img_file))
                labels.append(0)
    
    # 悪性画像
    malignant_dir = f"{isic_dir}/malignant"
    if os.path.exists(malignant_dir):
        for img_file in os.listdir(malignant_dir):
            if img_file.endswith('.jpg'):
                image_paths.append(os.path.join(malignant_dir, img_file))
                labels.append(1)
    
    print(f"ISICデータ: 良性 {labels.count(0)}枚, 悪性 {labels.count(1)}枚")
    return image_paths, labels

def load_user_data():
    """ユーザーデータの読み込みと分割"""
    
    base_path = "/Users/iinuma/Desktop/ダーモ"
    
    # 病変タイプ別のデータ
    disease_data = {
        'AK': {'paths': [], 'label': 1, 'name': '日光角化症'},
        'BCC': {'paths': [], 'label': 1, 'name': '基底細胞癌'},
        'Bowen病': {'paths': [], 'label': 1, 'name': 'Bowen病'},
        'MM': {'paths': [], 'label': 1, 'name': '悪性黒色腫'},
        'SK': {'paths': [], 'label': 0, 'name': '脂漏性角化症（良性）'}
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
    test_diseases = []  # テストデータの疾患名
    
    for folder_name, info in disease_data.items():
        if len(info['paths']) > 0:
            # 各疾患を8:2で分割
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
    """SKデータ（良性）のみを読み込み"""
    
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
        
        # テスト用の疾患名
        sk_test_diseases = ['脂漏性角化症（良性）'] * len(sk_test)
        
        print(f"SK学習用: {len(sk_train)}枚, テスト用: {len(sk_test)}枚")
        
        return sk_train, sk_train_labels, sk_test, sk_test_labels, sk_test_diseases
    else:
        return [], [], [], [], []

def pretrain_on_isic(model, train_loader, val_loader, epochs=10):
    """ISICデータで事前学習"""
    
    print("\n🔬 ISICデータで事前学習中...")
    
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
            torch.save(model.state_dict(), 'isic_pretrained_model.pth')
        
        scheduler.step()
    
    print(f"✅ 事前学習完了。最高精度: {best_val_acc:.2f}%")
    return model

def finetune_on_user_data(model, train_loader, val_loader, epochs=10):
    """ユーザーデータでファインチューニング"""
    
    print("\n🎯 ユーザーデータでファインチューニング中...")
    
    # 最後の層のみ学習率を高く設定
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
    
    torch.save(model.state_dict(), 'finetuned_model.pth')
    print("✅ ファインチューニング完了")
    return model

def evaluate_on_test_data(model, test_loader, test_labels, test_diseases):
    """テストデータで評価"""
    
    print("\n📊 テストデータで評価中...")
    
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
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 悪性の確率
    
    # 全体の精度
    accuracy = np.mean(np.array(all_predictions) == np.array(test_labels))
    print(f"\n全体精度: {accuracy:.2%}")
    
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
    
    print("\n疾患別精度:")
    for disease, results in disease_results.items():
        print(f"  {disease}: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    
    # 混同行列
    cm = confusion_matrix(test_labels, all_predictions)
    
    # AUC計算
    if len(set(test_labels)) == 2:
        auc = roc_auc_score(test_labels, all_probabilities)
        print(f"\nAUC: {auc:.4f}")
    
    # 分類レポート
    print("\n分類レポート:")
    unique_labels = sorted(list(set(test_labels)))
    if len(unique_labels) == 2:
        target_names = ['良性', '悪性']
    else:
        target_names = [f'クラス{i}' for i in unique_labels]
    
    print(classification_report(test_labels, all_predictions, 
                              target_names=target_names[:len(unique_labels)]))
    
    return cm, disease_results

def plot_results(cm, disease_results):
    """結果の可視化"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 混同行列
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['良性', '悪性'],
                yticklabels=['良性', '悪性'],
                ax=axes[0])
    axes[0].set_title('混同行列')
    axes[0].set_xlabel('予測')
    axes[0].set_ylabel('実際')
    
    # 疾患別精度
    diseases = list(disease_results.keys())
    accuracies = [disease_results[d]['accuracy'] for d in diseases]
    
    axes[1].barh(diseases, accuracies)
    axes[1].set_xlabel('精度')
    axes[1].set_title('疾患別精度')
    axes[1].set_xlim([0, 1])
    
    for i, (disease, acc) in enumerate(zip(diseases, accuracies)):
        axes[1].text(acc + 0.01, i, f'{acc:.1%}', va='center')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    plt.show()
    
    print("\n📈 結果を 'evaluation_results.png' に保存しました")

def finetune_with_sk_data(model, sk_train_loader, epochs=5):
    """SKデータ（良性）で3段階目のファインチューニング"""
    
    print("\n🌟 SKデータ（良性）で3段階目ファインチューニング中...")
    
    # 非常に低い学習率で既存知識を保持
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
    
    # バランス調整済みモデルを保存
    torch.save(model.state_dict(), 'balanced_finetuned_model.pth')
    print("✅ バランス調整済みモデルを 'balanced_finetuned_model.pth' に保存")
    return model

def main():
    """メインパイプライン（3段階ファインチューニング対応）"""
    
    print("=" * 60)
    print("🔬 3段階ファインチューニングパイプライン")
    print("   Stage 1: ImageNet → ISIC")
    print("   Stage 2: ISIC → ユーザー悪性データ")
    print("   Stage 3: 悪性学習済み → SK良性データでバランス調整")
    print("=" * 60)
    
    # 1. ISICデータのダウンロード
    downloader = ISICDownloader()
    
    # 既存データをチェック
    isic_paths, isic_labels = load_isic_data()
    
    if len(isic_paths) < 200:
        print("ISICデータが不足しています。ダウンロードを開始します...")
        downloader.download_images(benign_count=100, malignant_count=100)
        isic_paths, isic_labels = load_isic_data()
    
    if len(isic_paths) == 0:
        print("⚠️ ISICデータのダウンロードに失敗しました。")
        print("代替案: ローカルデータのみで学習します。")
        
    # 2. データ準備
    # ISICデータ分割
    if len(isic_paths) > 0:
        X_train_isic, X_val_isic, y_train_isic, y_val_isic = train_test_split(
            isic_paths, isic_labels, test_size=0.2, random_state=42, stratify=isic_labels
        )
        
        train_dataset_isic = DermoscopyDataset(X_train_isic, y_train_isic, get_transforms(True))
        val_dataset_isic = DermoscopyDataset(X_val_isic, y_val_isic, get_transforms(False))
        
        train_loader_isic = DataLoader(train_dataset_isic, batch_size=32, shuffle=True, num_workers=2)
        val_loader_isic = DataLoader(val_dataset_isic, batch_size=32, shuffle=False, num_workers=2)
    
    # ユーザーデータ読み込み（悪性のみ、SKは除外）
    user_train_paths, user_train_labels, user_test_paths, user_test_labels, test_diseases = load_user_data()
    
    # SKデータは除外してユーザーデータを再構成
    filtered_train_paths = []
    filtered_train_labels = []
    filtered_test_paths = []
    filtered_test_labels = []
    filtered_test_diseases = []
    
    for i, path in enumerate(user_train_paths):
        if 'SK' not in path:
            filtered_train_paths.append(path)
            filtered_train_labels.append(user_train_labels[i])
    
    for i, path in enumerate(user_test_paths):
        if 'SK' not in path:
            filtered_test_paths.append(path)
            filtered_test_labels.append(user_test_labels[i])
            filtered_test_diseases.append(test_diseases[i])
    
    user_train_paths = filtered_train_paths
    user_train_labels = filtered_train_labels
    user_test_paths = filtered_test_paths
    user_test_labels = filtered_test_labels
    test_diseases = filtered_test_diseases
    
    print(f"\n悪性データのみ統計:")
    print(f"  学習用: {len(user_train_paths)}枚")
    print(f"  テスト用: {len(user_test_paths)}枚")
    
    
    # ユーザーデータのDataLoader
    train_dataset_user = DermoscopyDataset(user_train_paths, user_train_labels, get_transforms(True))
    test_dataset_user = DermoscopyDataset(user_test_paths, user_test_labels, get_transforms(False))
    
    train_loader_user = DataLoader(train_dataset_user, batch_size=16, shuffle=True, num_workers=2)
    test_loader_user = DataLoader(test_dataset_user, batch_size=16, shuffle=False, num_workers=2)
    
    # 3. 既存のfinetuned_modelを読み込み（Stage 2完了済み）
    print("\n🔄 既存のfinetuned_model.pthを読み込み...")
    model = PretrainedModel(num_classes=2, model_type='efficientnet').to(device)
    
    if os.path.exists('finetuned_model.pth'):
        model.load_state_dict(torch.load('finetuned_model.pth', map_location=device))
        print("✅ Stage 2完了済みモデル（finetuned_model.pth）を読み込みました")
    else:
        print("⚠️ finetuned_model.pth が見つかりません。通常のパイプラインを実行...")
        # 通常のパイプライン実行
        if len(isic_paths) > 0:
            model = pretrain_on_isic(model, train_loader_isic, val_loader_isic, epochs=5)
        
        if os.path.exists('isic_pretrained_model.pth'):
            model.load_state_dict(torch.load('isic_pretrained_model.pth', map_location=device))
            print("✅ ISICモデルを再読み込みしました")
        
        model = finetune_on_user_data(model, train_loader_user, train_loader_user, epochs=10)
    
    # 4. SKデータ（良性）の準備
    sk_train_paths, sk_train_labels, sk_test_paths, sk_test_labels, sk_test_diseases = load_sk_data_only()
    
    if len(sk_train_paths) > 0:
        # SKデータのDataLoader
        sk_train_dataset = DermoscopyDataset(sk_train_paths, sk_train_labels, get_transforms(True))
        sk_train_loader = DataLoader(sk_train_dataset, batch_size=16, shuffle=True, num_workers=2)
        
        # 5. Stage 3: SKデータでバランス調整
        model = finetune_with_sk_data(model, sk_train_loader, epochs=5)
        
        # 6. 拡張テストデータで評価（ユーザー悪性 + SK良性）
        print("\n📊 拡張テストデータセットでの評価...")
        
        # テストデータを結合
        combined_test_paths = user_test_paths + sk_test_paths
        combined_test_labels = user_test_labels + sk_test_labels
        combined_test_diseases = test_diseases + sk_test_diseases
        
        # 拡張テストデータのDataLoader
        combined_test_dataset = DermoscopyDataset(combined_test_paths, combined_test_labels, get_transforms(False))
        combined_test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # 評価実行
        cm, disease_results = evaluate_on_test_data(
            model, combined_test_loader, combined_test_labels, combined_test_diseases
        )
        
        # 7. 結果の可視化
        plot_results(cm, disease_results)
        
        print("\n🎯 データバランス改善結果:")
        print(f"   悪性テストデータ: {len([l for l in combined_test_labels if l == 1])}枚")
        print(f"   良性テストデータ: {len([l for l in combined_test_labels if l == 0])}枚")
        
    else:
        print("⚠️ SKデータが見つかりません。従来の評価を実行...")
        # 従来の評価
        cm, disease_results = evaluate_on_test_data(
            model, test_loader_user, user_test_labels, test_diseases
        )
        plot_results(cm, disease_results)
    
    print("\n✅ 3段階ファインチューニングパイプライン完了!")
    print("\n📁 生成されたモデル:")
    print("   • isic_pretrained_model.pth: Stage 1完了（ISICのみ）")
    print("   • finetuned_model.pth: Stage 2完了（悪性データ追加）")
    print("   • balanced_finetuned_model.pth: Stage 3完了（良性データでバランス調整）")

def run_sk_only_finetuning():
    """SKデータのみで既存モデルを改善（簡易版）"""
    
    print("=" * 50)
    print("🌟 SKデータ追加ファインチューニング（簡易版）")
    print("=" * 50)
    
    # 既存モデル読み込み
    model = PretrainedModel(num_classes=2, model_type='efficientnet').to(device)
    
    if os.path.exists('finetuned_model.pth'):
        model.load_state_dict(torch.load('finetuned_model.pth', map_location=device))
        print("✅ 既存のfinetuned_model.pthを読み込みました")
    else:
        print("❌ finetuned_model.pth が見つかりません")
        return
    
    # SKデータ準備
    sk_train_paths, sk_train_labels, sk_test_paths, sk_test_labels, sk_test_diseases = load_sk_data_only()
    
    if len(sk_train_paths) == 0:
        print("❌ SKデータが見つかりません")
        return
    
    # SKデータのDataLoader
    sk_train_dataset = DermoscopyDataset(sk_train_paths, sk_train_labels, get_transforms(True))
    sk_train_loader = DataLoader(sk_train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # SKデータでファインチューニング
    model = finetune_with_sk_data(model, sk_train_loader, epochs=5)
    
    print("\n✅ SKデータ追加ファインチューニング完了!")
    print("📁 balanced_finetuned_model.pth を生成しました")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sk-only":
        run_sk_only_finetuning()
    else:
        main()