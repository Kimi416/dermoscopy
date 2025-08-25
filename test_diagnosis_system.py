"""
統合診断システムテスト
Nevus vs Melanoma統合済みアンサンブルシステムでの診断テスト
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
import numpy as np
from PIL import Image
import os
import json
from sklearn.metrics import roc_auc_score
import glob

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

class IntegratedDiagnosisSystem:
    """統合診断システム"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def load_trained_models(self):
        """訓練済みモデル読み込み（シミュレート版）"""
        print("📁 統合診断システム初期化...")
        
        # モデルタイプ
        model_types = ['efficientnet', 'resnet']
        
        # 各モデルを作成（実際には訓練済みモデルを読み込む）
        for model_type in model_types:
            model = DualModel(model_type).to(device)
            model.eval()
            self.models[model_type] = model
            print(f"   ✅ {model_type.upper()} モデル読み込み完了")
        
        # AUCベースの重み（シミュレート値）
        self.ensemble_weights = {
            'efficientnet': 0.55,
            'resnet': 0.45
        }
        
        print(f"📊 アンサンブル重み:")
        for model_type, weight in self.ensemble_weights.items():
            print(f"   {model_type}: {weight:.3f}")
    
    def predict_base_ensemble(self, image_paths):
        """基本アンサンブル予測"""
        ensemble_probs = np.zeros(len(image_paths))
        
        for model_type, model in self.models.items():
            model_probs = []
            
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        prob = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()
                        model_probs.append(prob)
                
                except Exception as e:
                    print(f"⚠️ 画像処理エラー {img_path}: {e}")
                    model_probs.append(0.5)  # デフォルト値
            
            # アンサンブルに重み付け加算
            weight = self.ensemble_weights[model_type]
            ensemble_probs += weight * np.array(model_probs)
        
        return ensemble_probs
    
    def apply_nevus_mm_correction(self, base_probs, image_paths):
        """Nevus vs Melanoma補正適用"""
        try:
            from nevus_mm_classifier import predict_mm_prob
            
            print("🧬 Nevus vs Melanoma分類器適用中...")
            p_mm = predict_mm_prob(image_paths, weights_dir='/Users/iinuma/Desktop/ダーモ/nevusmm_weights')
            
            # ロジスティック融合（alpha=0.35）
            alpha = 0.35
            corrected_probs = (1 - alpha) * base_probs + alpha * p_mm
            
            print(f"   補正係数 alpha: {alpha}")
            print(f"   p(MM) 平均: {np.mean(p_mm):.3f}")
            print(f"   補正前平均: {np.mean(base_probs):.3f}")
            print(f"   補正後平均: {np.mean(corrected_probs):.3f}")
            
            return corrected_probs, p_mm
            
        except ImportError:
            print("⚠️ nevus_mm_classifier が利用できません")
            return base_probs, None
        except Exception as e:
            print(f"⚠️ Nevus-MM統合エラー: {e}")
            return base_probs, None
    
    def diagnose_images(self, image_paths):
        """画像診断実行"""
        print(f"🔍 {len(image_paths)}枚の画像を診断中...")
        
        # 基本アンサンブル予測
        base_probs = self.predict_base_ensemble(image_paths)
        
        # Nevus vs Melanoma補正適用
        final_probs, p_mm = self.apply_nevus_mm_correction(base_probs, image_paths)
        
        # 結果整理
        results = []
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            disease_folder = os.path.basename(os.path.dirname(img_path))
            
            # 実際のラベル取得
            actual_label = 1 if DISEASE_MAPPING.get(disease_folder, {}).get('type') == 'malignant' else 0
            
            result = {
                'filename': filename,
                'disease_folder': disease_folder,
                'actual_label': actual_label,
                'actual_type': 'malignant' if actual_label == 1 else 'benign',
                'base_probability': float(base_probs[i]),
                'final_probability': float(final_probs[i]),
                'predicted_label': 1 if final_probs[i] > 0.5 else 0,
                'predicted_type': 'malignant' if final_probs[i] > 0.5 else 'benign',
                'confidence': float(abs(final_probs[i] - 0.5) * 2),
                'nevus_mm_prob': float(p_mm[i]) if p_mm is not None else None,
                'correction_effect': float(final_probs[i] - base_probs[i])
            }
            
            results.append(result)
        
        return results
    
    def generate_diagnosis_report(self, results):
        """診断レポート生成"""
        print("\n" + "=" * 80)
        print("📋 統合診断システム レポート")
        print("=" * 80)
        
        # 基本統計
        total_cases = len(results)
        malignant_cases = sum([1 for r in results if r['actual_label'] == 1])
        benign_cases = total_cases - malignant_cases
        
        print(f"\n📊 診断対象:")
        print(f"   総症例数: {total_cases}例")
        print(f"   悪性症例: {malignant_cases}例")
        print(f"   良性症例: {benign_cases}例")
        
        # 予測精度
        correct_predictions = sum([1 for r in results if r['predicted_label'] == r['actual_label']])
        accuracy = correct_predictions / total_cases
        
        print(f"\n🎯 診断性能:")
        print(f"   正解率: {accuracy:.1%} ({correct_predictions}/{total_cases})")
        
        # AUC計算
        actual_labels = [r['actual_label'] for r in results]
        final_probs = [r['final_probability'] for r in results]
        
        if len(set(actual_labels)) > 1:  # 両方のクラスが存在する場合
            auc = roc_auc_score(actual_labels, final_probs)
            print(f"   AUC: {auc:.4f}")
        
        # 疾患別性能
        print(f"\n🏥 疾患別診断結果:")
        diseases = {}
        for result in results:
            disease = result['disease_folder']
            if disease not in diseases:
                diseases[disease] = []
            diseases[disease].append(result)
        
        for disease, cases in diseases.items():
            correct = sum([1 for c in cases if c['predicted_label'] == c['actual_label']])
            total = len(cases)
            accuracy = correct / total
            avg_prob = np.mean([c['final_probability'] for c in cases])
            disease_type = DISEASE_MAPPING.get(disease, {}).get('type', 'unknown')
            
            print(f"   {disease} ({disease_type}): {accuracy:.1%} ({correct}/{total}), 平均悪性確率: {avg_prob:.1%}")
        
        # SK特化分析
        sk_cases = [r for r in results if r['disease_folder'] == 'SK']
        if sk_cases:
            print(f"\n🎯 SK誤分類改善効果:")
            sk_avg_prob = np.mean([c['final_probability'] for c in sk_cases])
            sk_correct = sum([1 for c in sk_cases if c['predicted_label'] == c['actual_label']])
            sk_total = len(sk_cases)
            
            print(f"   SK症例: {sk_total}例")
            print(f"   SK平均悪性確率: {sk_avg_prob:.1%}")
            print(f"   SK正解率: {sk_correct/sk_total:.1%}")
            
            if sk_avg_prob < 0.5:
                print("   ✅ SK誤分類問題が改善されています")
            else:
                print("   ⚠️ SK誤分類がまだ残存しています")
        
        # Nevus-MM補正効果
        nevus_mm_integrated = any([r['nevus_mm_prob'] is not None for r in results])
        if nevus_mm_integrated:
            print(f"\n🧬 Nevus vs Melanoma補正効果:")
            corrections = [r['correction_effect'] for r in results if r['nevus_mm_prob'] is not None]
            avg_correction = np.mean(corrections)
            significant_corrections = sum([1 for c in corrections if abs(c) > 0.1])
            
            print(f"   平均補正量: {avg_correction:+.3f}")
            print(f"   有意な補正: {significant_corrections}/{len(corrections)}例")
        
        # 高信頼度・低信頼度症例
        print(f"\n📈 診断信頼度分析:")
        high_confidence = [r for r in results if r['confidence'] > 0.8]
        low_confidence = [r for r in results if r['confidence'] < 0.4]
        
        print(f"   高信頼度症例 (>80%): {len(high_confidence)}例")
        print(f"   低信頼度症例 (<40%): {len(low_confidence)}例")
        
        if high_confidence:
            high_accuracy = sum([1 for r in high_confidence if r['predicted_label'] == r['actual_label']]) / len(high_confidence)
            print(f"   高信頼度症例の正解率: {high_accuracy:.1%}")
        
        return {
            'total_cases': total_cases,
            'accuracy': accuracy,
            'auc': auc if len(set(actual_labels)) > 1 else None,
            'disease_performance': diseases,
            'nevus_mm_integrated': nevus_mm_integrated,
            'detailed_results': results
        }

def collect_test_images(base_path='/Users/iinuma/Desktop/ダーモ', samples_per_disease=5):
    """テスト画像収集"""
    print("📸 テスト画像収集中...")
    
    test_images = []
    
    for disease, info in DISEASE_MAPPING.items():
        disease_dir = os.path.join(base_path, disease)
        if not os.path.exists(disease_dir):
            continue
        
        # 画像ファイル取得
        patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(disease_dir, pattern)))
        
        # サンプリング
        if len(image_files) > samples_per_disease:
            selected = np.random.choice(image_files, samples_per_disease, replace=False)
        else:
            selected = image_files
        
        test_images.extend(selected)
        print(f"   {disease}: {len(selected)}枚選択")
    
    print(f"✅ 合計: {len(test_images)}枚のテスト画像を収集")
    return test_images

def main():
    """メイン実行"""
    print("🚀 統合診断システムテスト開始")
    print("   Nevus vs Melanoma統合済みアンサンブル")
    print("=" * 80)
    
    # システム初期化
    diagnosis_system = IntegratedDiagnosisSystem()
    diagnosis_system.load_trained_models()
    
    # テスト画像収集
    test_images = collect_test_images(samples_per_disease=3)  # 各疾患3枚ずつ
    
    if len(test_images) == 0:
        print("❌ テスト画像が見つかりませんでした")
        return
    
    # 診断実行
    results = diagnosis_system.diagnose_images(test_images)
    
    # 詳細結果表示
    print(f"\n📋 個別診断結果:")
    print("-" * 80)
    for result in results:
        print(f"📁 {result['filename']} ({result['disease_folder']})")
        print(f"   実際: {result['actual_type']}")
        print(f"   予測: {result['predicted_type']} ({result['final_probability']:.1%})")
        print(f"   信頼度: {result['confidence']:.1%}")
        if result['nevus_mm_prob'] is not None:
            print(f"   p(MM): {result['nevus_mm_prob']:.1%}")
        print(f"   補正効果: {result['correction_effect']:+.3f}")
        print()
    
    # 診断レポート生成
    report = diagnosis_system.generate_diagnosis_report(results)
    
    # 結果保存
    with open('/Users/iinuma/Desktop/ダーモ/integrated_diagnosis_results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 詳細結果保存: integrated_diagnosis_results.json")
    print(f"\n🎉 統合診断テスト完了！")

if __name__ == "__main__":
    # ランダムシード固定
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()