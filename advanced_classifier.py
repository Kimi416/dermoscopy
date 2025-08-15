"""
高精度ダーモスコピー分類システム
脂漏性角化症など詳細な病変タイプを識別
"""

import os
import sys
sys.path.append('/Users/iinuma/Desktop/ダーモ')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from dermoscopy_classifier import (
    HierarchicalClassifier,
    DermoscopyDataset,
    Stage1_ImageEnhancer,
    get_augmentation_pipeline,
    load_dataset
)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def train_hierarchical_model(model, train_loader, val_loader, num_epochs=20):
    """階層的モデルの学習"""
    
    # 損失関数（クラス不均衡対応）
    malignancy_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))
    detail_criterion = nn.CrossEntropyLoss()
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_val_acc = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mal_acc': [], 'val_mal_acc': [],
        'train_det_acc': [], 'val_det_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_mal_correct = 0
        train_det_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 階層的損失計算
            mal_loss = malignancy_criterion(outputs['malignancy'], (labels > 3).long())
            det_loss = detail_criterion(outputs['detail'], labels)
            
            # 重み付き総損失
            total_loss = 0.4 * mal_loss + 0.6 * det_loss
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # 精度計算
            _, mal_pred = outputs['malignancy'].max(1)
            _, det_pred = outputs['detail'].max(1)
            
            train_mal_correct += mal_pred.eq((labels > 3).long()).sum().item()
            train_det_correct += det_pred.eq(labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_mal_correct = 0
        val_det_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Val'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                mal_loss = malignancy_criterion(outputs['malignancy'], (labels > 3).long())
                det_loss = detail_criterion(outputs['detail'], labels)
                total_loss = 0.4 * mal_loss + 0.6 * det_loss
                
                val_loss += total_loss.item()
                
                _, mal_pred = outputs['malignancy'].max(1)
                _, det_pred = outputs['detail'].max(1)
                
                val_mal_correct += mal_pred.eq((labels > 3).long()).sum().item()
                val_det_correct += det_pred.eq(labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(det_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 統計記録
        train_mal_acc = 100. * train_mal_correct / train_total
        train_det_acc = 100. * train_det_correct / train_total
        val_mal_acc = 100. * val_mal_correct / val_total
        val_det_acc = 100. * val_det_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_mal_acc'].append(train_mal_acc)
        history['val_mal_acc'].append(val_mal_acc)
        history['train_det_acc'].append(train_det_acc)
        history['val_det_acc'].append(val_det_acc)
        
        print(f'Epoch {epoch+1}:')
        print(f'  良悪性分類 - Train: {train_mal_acc:.2f}%, Val: {val_mal_acc:.2f}%')
        print(f'  詳細分類   - Train: {train_det_acc:.2f}%, Val: {val_det_acc:.2f}%')
        
        # ベストモデル保存
        if val_det_acc > best_val_acc:
            best_val_acc = val_det_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_val_acc,
                'confusion_matrix': confusion_matrix(all_labels, all_preds)
            }, 'advanced_dermoscopy_model.pth')
            print(f'  ✅ Best model saved: {val_det_acc:.2f}%')
        
        scheduler.step()
    
    return history, confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, class_names):
    """混同行列の可視化"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - 病変タイプ分類')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_training(history):
    """学習過程の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 良悪性分類精度
    axes[0, 1].plot(history['train_mal_acc'], label='Train')
    axes[0, 1].plot(history['val_mal_acc'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('良悪性分類精度')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 詳細分類精度
    axes[1, 0].plot(history['train_det_acc'], label='Train')
    axes[1, 0].plot(history['val_det_acc'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('詳細分類精度')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 精度比較
    epochs = range(1, len(history['train_loss']) + 1)
    axes[1, 1].plot(epochs, history['val_mal_acc'], 'b-', label='良悪性')
    axes[1, 1].plot(epochs, history['val_det_acc'], 'r-', label='詳細')
    axes[1, 1].fill_between(epochs, history['val_mal_acc'], history['val_det_acc'], 
                            alpha=0.3, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy (%)')
    axes[1, 1].set_title('階層的分類の精度差')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('advanced_training_history.png')
    plt.show()

def main():
    """メイン実行関数"""
    
    print("🔬 高精度ダーモスコピー分類システム")
    print("=" * 50)
    
    # データセット読み込み（詳細ラベル使用）
    try:
        image_paths, labels, class_mapping = load_dataset(use_detailed_labels=True)
    except:
        # class_mappingが返されない場合は通常のラベルで実行
        print("\n⚠️ 詳細ラベルが利用できません。基本的な良悪性分類で実行します。")
        image_paths, labels = load_dataset(use_detailed_labels=False)
        # 仮のクラスマッピング
        class_mapping = {
            'benign': {'name': '良性', 'detail': 0},
            'malignant': {'name': '悪性', 'detail': 1}
        }
    
    if len(image_paths) == 0:
        print("エラー: 画像が見つかりません。")
        return
    
    # クラス名のリスト作成
    class_names = [info['name'] for _, info in sorted(class_mapping.items(), 
                                                      key=lambda x: x[1].get('detail', 0))]
    
    print(f"\nクラス構成: {class_names}")
    
    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, 
        stratify=labels if len(set(labels)) > 1 else None
    )
    
    # Stage 1: 前処理器
    stage1_enhancer = Stage1_ImageEnhancer()
    
    # データセット作成
    train_dataset = DermoscopyDataset(
        X_train, y_train,
        transform=get_augmentation_pipeline(is_train=True),
        stage1_enhancer=stage1_enhancer
    )
    
    val_dataset = DermoscopyDataset(
        X_val, y_val,
        transform=get_augmentation_pipeline(is_train=False),
        stage1_enhancer=stage1_enhancer
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # モデル初期化
    num_classes = len(set(labels))
    model = HierarchicalClassifier(num_classes=num_classes).to(device)
    
    print(f"\n📊 データセット情報:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Classes: {num_classes}")
    print(f"Device: {device}")
    
    # モデル学習
    print("\n🚀 階層的学習開始...")
    history, cm = train_hierarchical_model(model, train_loader, val_loader, num_epochs=20)
    
    # 結果可視化
    visualize_training(history)
    if len(class_names) == num_classes:
        plot_confusion_matrix(cm, class_names)
    
    print("\n✅ 学習完了!")
    print("モデルは 'advanced_dermoscopy_model.pth' に保存されました。")
    print(f"最終精度:")
    print(f"  良悪性分類: {history['val_mal_acc'][-1]:.2f}%")
    print(f"  詳細分類: {history['val_det_acc'][-1]:.2f}%")

if __name__ == "__main__":
    main()