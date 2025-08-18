"""
改善版モデルでダーモスコピー画像を判定
過学習対策済み Cross-Validation訓練モデル
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class ImprovedDermoscopyModel(nn.Module):
    """改善版ダーモスコピー分類モデル（推論用）"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_improved_model(model_path):
    """改善版モデルの読み込み"""
    model = ImprovedDermoscopyModel(num_classes=2, dropout_rate=0.5)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            fold_info = f" (Fold {checkpoint.get('fold', '?')}, AUC: {checkpoint.get('best_auc', 0):.3f})"
        else:
            model.load_state_dict(checkpoint)
            fold_info = ""
        
        model.to(device)
        model.eval()
        print(f"✅ 改善版モデルを読み込みました: {model_path}{fold_info}")
        return model
    
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None

def preprocess_image(image_path):
    """画像の前処理（検証時と同じ処理）"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"❌ 画像読み込みエラー: {e}")
        return None

def predict_with_ensemble(models, image_tensor):
    """アンサンブル予測（複数モデルの平均）"""
    if not models:
        return None
    
    all_probs = []
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        for model in models:
            model.eval()
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy())
    
    # アンサンブル（平均）
    ensemble_probs = np.mean(all_probs, axis=0)
    benign_prob = ensemble_probs[0][0]
    malignant_prob = ensemble_probs[0][1]
    
    predicted_class = 1 if malignant_prob > benign_prob else 0
    confidence = max(benign_prob, malignant_prob)
    
    # 個別モデルの予測も返す
    individual_results = []
    for i, prob in enumerate(all_probs):
        b_prob = prob[0][0]
        m_prob = prob[0][1]
        pred_class = 1 if m_prob > b_prob else 0
        individual_results.append({
            'model_idx': i,
            'predicted_class': pred_class,
            'benign_probability': b_prob,
            'malignant_probability': m_prob,
            'confidence': max(b_prob, m_prob)
        })
    
    return {
        'ensemble_prediction': {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob
        },
        'individual_predictions': individual_results,
        'prediction_variance': np.std([r['malignant_probability'] for r in individual_results])
    }

def compare_with_previous_models(image_path):
    """従来モデルとの比較"""
    print(f"\n🔍 従来モデルとの比較分析")
    print("-" * 50)
    
    # 従来モデルのパス
    previous_models = [
        ("/Users/iinuma/Desktop/ダーモ/ham10000_balanced_finetuned_model.pth", "HAM10000バランス調整"),
        ("/Users/iinuma/Desktop/ダーモ/balanced_finetuned_model.pth", "ISICバランス調整"),
    ]
    
    # 改善版モデル用のクラス（簡易版）
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)
    
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    for model_path, model_desc in previous_models:
        if not os.path.exists(model_path):
            continue
            
        try:
            # 従来モデル読み込み
            old_model = SimpleModel(num_classes=2)
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                old_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                old_model.load_state_dict(checkpoint)
            
            old_model.to(device)
            old_model.eval()
            
            # 予測実行
            with torch.no_grad():
                image_tensor_device = image_tensor.to(device)
                outputs = old_model(image_tensor_device)
                probabilities = torch.softmax(outputs, dim=1)
                
                benign_prob = probabilities[0][0].item()
                malignant_prob = probabilities[0][1].item()
                predicted_class = 1 if malignant_prob > benign_prob else 0
                confidence = max(benign_prob, malignant_prob)
            
            class_name = "悪性" if predicted_class == 1 else "良性"
            print(f"📊 {model_desc}: {class_name} ({confidence:.1%})")
            print(f"   良性: {benign_prob:.1%}, 悪性: {malignant_prob:.1%}")
            
        except Exception as e:
            print(f"❌ {model_desc}の読み込みエラー: {e}")

def main():
    """メイン実行関数"""
    print("🔬 改善版ダーモスコピー診断システム")
    print("   過学習対策済み Cross-Validation アンサンブルモデル")
    print("="*70)
    
    # test.JPGのパス
    image_path = "/Users/iinuma/Desktop/ダーモ/test.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ 画像が見つかりません: {image_path}")
        return
    
    # 改善版モデルを読み込み
    improved_models = []
    model_paths = glob.glob("/Users/iinuma/Desktop/ダーモ/improved_model_fold_*.pth")
    
    if not model_paths:
        print("❌ 改善版モデルが見つかりません。先にtraining pipelineを実行してください。")
        return
    
    print(f"\n📂 改善版モデル読み込み ({len(model_paths)}個)")
    print("-" * 50)
    
    for model_path in sorted(model_paths):
        model = load_improved_model(model_path)
        if model is not None:
            improved_models.append(model)
    
    if not improved_models:
        print("❌ 改善版モデルの読み込みに失敗しました")
        return
    
    # 画像前処理
    print(f"\n📸 画像分析: {image_path}")
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    # アンサンブル予測実行
    print(f"\n🧠 改善版アンサンブル予測 ({len(improved_models)}モデル)")
    print("="*50)
    
    result = predict_with_ensemble(improved_models, image_tensor)
    if result is None:
        print("❌ 予測に失敗しました")
        return
    
    # アンサンブル結果表示
    ensemble = result['ensemble_prediction']
    final_prediction = "悪性" if ensemble['predicted_class'] == 1 else "良性"
    
    print(f"🎯 アンサンブル最終判定: {final_prediction}")
    print(f"🎯 総合確信度: {ensemble['confidence']:.1%}")
    print(f"📊 良性確率: {ensemble['benign_probability']:.1%}")
    print(f"📊 悪性確率: {ensemble['malignant_probability']:.1%}")
    print(f"📈 予測分散: {result['prediction_variance']:.3f}")
    
    if result['prediction_variance'] > 0.1:
        print("⚠️ モデル間で予測に大きなばらつきがあります")
    else:
        print("✅ モデル間で予測が一致しています")
    
    # 個別モデル結果
    print(f"\n📋 個別モデル予測:")
    for i, pred in enumerate(result['individual_predictions']):
        class_name = "悪性" if pred['predicted_class'] == 1 else "良性"
        print(f"  Fold {i+1}: {class_name} ({pred['confidence']:.1%}) "
              f"[良性:{pred['benign_probability']:.1%}, 悪性:{pred['malignant_probability']:.1%}]")
    
    # 従来モデルとの比較
    compare_with_previous_models(image_path)
    
    # 改善効果の分析
    print(f"\n📈 改善効果分析")
    print("="*50)
    print(f"✅ 過学習対策:")
    print(f"  • データ拡張による汎化性能向上")
    print(f"  • 強化された正則化 (Dropout 0.5, Weight Decay)")
    print(f"  • Early Stopping による過学習防止")
    print(f"  • Cross-Validation による頑健性確保")
    print(f"  • アンサンブル学習による予測安定性向上")
    
    print(f"\n💡 信頼性指標:")
    if result['prediction_variance'] < 0.05:
        reliability = "非常に高い"
    elif result['prediction_variance'] < 0.1:
        reliability = "高い"
    else:
        reliability = "中程度"
    
    print(f"  予測の信頼性: {reliability}")
    print(f"  モデル間一致度: {(1 - result['prediction_variance']):.1%}")
    
    if ensemble['predicted_class'] == 0:  # 良性予測の場合
        print(f"\n🎉 改善版モデルは test.JPG を良性と正しく判定する可能性が高まりました!")
    else:
        print(f"\n⚠️ 改善版モデルでも悪性判定ですが、確信度と信頼性を確認してください。")

if __name__ == "__main__":
    main()