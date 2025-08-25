"""
包括的検証システム
全データセットを使用した交差検証
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
import numpy as np
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import json
from collections import defaultdict

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

def collect_all_images(base_path='/Users/iinuma/Desktop/ダーモ'):
    """全画像データ収集"""
    print("📁 全データ収集中...")
    
    all_images = defaultdict(list)
    total_count = 0
    
    for disease in DISEASE_MAPPING.keys():
        disease_dir = os.path.join(base_path, disease)
        if not os.path.exists(disease_dir):
            continue
        
        patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
        disease_images = []
        for pattern in patterns:
            disease_images.extend(glob.glob(os.path.join(disease_dir, pattern)))
        
        all_images[disease] = disease_images
        total_count += len(disease_images)
        print(f"  {disease}: {len(disease_images)}枚")
    
    print(f"✅ 合計: {total_count}枚")
    return all_images

def split_data_for_validation(all_images, test_size=0.5, random_state=42):
    """データを訓練用とテスト用に分割"""
    train_images = []
    test_images = []
    
    print("\n📊 データ分割中...")
    for disease, images in all_images.items():
        if len(images) > 1:
            # 各疾患から一定割合をテスト用に分割
            train, test = train_test_split(
                images, 
                test_size=test_size, 
                random_state=random_state
            )
            train_images.extend(train)
            test_images.extend(test)
            print(f"  {disease}: 訓練{len(train)}枚, テスト{len(test)}枚")
        else:
            # データが1枚しかない場合はテストに含める
            test_images.extend(images)
            print(f"  {disease}: テスト{len(images)}枚のみ")
    
    print(f"\n総計: 訓練{len(train_images)}枚, テスト{len(test_images)}枚")
    return train_images, test_images

def run_comprehensive_validation():
    """包括的検証実行"""
    print("="*80)
    print("🔬 四段階統合診断システム - 包括的検証")
    print("="*80)
    
    # データ収集
    all_images = collect_all_images()
    
    # データ分割
    train_images, test_images = split_data_for_validation(all_images)
    
    # 既存のテストシステムを使用
    from test_integrated_three_stage_system import ThreeStageIntegratedSystem
    
    # システム初期化
    system = ThreeStageIntegratedSystem()
    system.load_base_models()
    
    # テスト実行
    print("\n🚀 テストデータで検証開始...")
    results = system.diagnose_four_stage(test_images)
    
    # 詳細な分析
    print("\n📈 詳細分析結果")
    print("-"*60)
    
    # 疾患別の詳細統計
    disease_stats = defaultdict(lambda: {
        'total': 0, 'correct': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
        'probs': [], 'confidences': []
    })
    
    for result in results:
        disease = result['disease_folder']
        stats = disease_stats[disease]
        
        stats['total'] += 1
        stats['probs'].append(result['final_prob'])
        stats['confidences'].append(result['confidence'])
        
        actual = result['actual_label']
        predicted = result['predicted_label']
        
        if actual == predicted:
            stats['correct'] += 1
            if actual == 1:
                stats['tp'] += 1
            else:
                stats['tn'] += 1
        else:
            if actual == 1:
                stats['fn'] += 1
            else:
                stats['fp'] += 1
    
    # 各疾患の詳細メトリクス
    print("\n【疾患別詳細性能】")
    overall_correct = 0
    overall_total = 0
    
    for disease, stats in disease_stats.items():
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        tp = stats['tp']
        tn = stats['tn']
        fp = stats['fp']
        fn = stats['fn']
        
        # 感度と特異度
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0
        
        # 平均確率と信頼度
        avg_prob = np.mean(stats['probs']) * 100 if stats['probs'] else 0
        avg_conf = np.mean(stats['confidences']) * 100 if stats['confidences'] else 0
        
        print(f"\n{disease} ({DISEASE_MAPPING[disease]['type']}):")
        print(f"  精度: {accuracy:.1f}% ({correct}/{total})")
        print(f"  感度: {sensitivity:.1f}%, 特異度: {specificity:.1f}%")
        print(f"  TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}")
        print(f"  平均確率: {avg_prob:.1f}%, 平均信頼度: {avg_conf:.1f}%")
        
        overall_correct += correct
        overall_total += total
    
    # 全体性能
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    
    print("\n" + "="*60)
    print("【全体性能サマリー】")
    print(f"全体精度: {overall_accuracy:.1f}% ({overall_correct}/{overall_total})")
    
    # 悪性・良性別の統計
    malignant_correct = 0
    malignant_total = 0
    benign_correct = 0
    benign_total = 0
    
    for result in results:
        if result['actual_label'] == 1:  # 悪性
            malignant_total += 1
            if result['predicted_label'] == 1:
                malignant_correct += 1
        else:  # 良性
            benign_total += 1
            if result['predicted_label'] == 0:
                benign_correct += 1
    
    malignant_sensitivity = (malignant_correct / malignant_total * 100) if malignant_total > 0 else 0
    benign_specificity = (benign_correct / benign_total * 100) if benign_total > 0 else 0
    
    print(f"悪性疾患感度: {malignant_sensitivity:.1f}% ({malignant_correct}/{malignant_total})")
    print(f"良性疾患特異度: {benign_specificity:.1f}% ({benign_correct}/{benign_total})")
    
    # AUC計算
    if len(results) > 0:
        actual_labels = [r['actual_label'] for r in results]
        predicted_probs = [r['final_prob'] for r in results]
        
        if len(set(actual_labels)) > 1:
            auc = roc_auc_score(actual_labels, predicted_probs)
            print(f"AUC: {auc:.4f}")
    
    # 結果保存
    validation_results = {
        'test_size': len(test_images),
        'overall_accuracy': overall_accuracy,
        'malignant_sensitivity': malignant_sensitivity,
        'benign_specificity': benign_specificity,
        'disease_stats': {
            disease: {
                'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'total': stats['total'],
                'correct': stats['correct']
            }
            for disease, stats in disease_stats.items()
        }
    }
    
    with open('/Users/iinuma/Desktop/ダーモ/comprehensive_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    print("\n💾 検証結果保存: comprehensive_validation_results.json")
    print("="*80)
    print("✅ 包括的検証完了")
    
    return validation_results

if __name__ == "__main__":
    run_comprehensive_validation()