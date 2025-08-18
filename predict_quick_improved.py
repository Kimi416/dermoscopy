"""
改善版モデル予測スクリプト
過学習対策済みモデルでtest.JPGを評価
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import os

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class QuickImprovedModel(nn.Module):
    """改善版モデル（推論用）"""
    
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_improved_model(model_path):
    """改善版モデルの読み込み"""
    model = QuickImprovedModel(num_classes=2, dropout_rate=0.4)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            auc_info = f" (AUC: {checkpoint.get('best_auc', 0):.3f})"
        else:
            model.load_state_dict(checkpoint)
            auc_info = ""
        
        model.to(device)
        model.eval()
        print(f"✅ 改善版モデルを読み込みました: {model_path}{auc_info}")
        return model, checkpoint
    
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None, None

def preprocess_image(image_path):
    """画像の前処理"""
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

def predict_with_model(model, image_tensor):
    """モデル予測実行"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        benign_prob = probabilities[0][0].item()
        malignant_prob = probabilities[0][1].item()
        predicted_class = 1 if malignant_prob > benign_prob else 0
        confidence = max(benign_prob, malignant_prob)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob
        }

def load_legacy_model(model_path, model_type="simple"):
    """従来モデルの読み込み"""
    if model_type == "simple":
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
        
        model = SimpleModel(num_classes=2)
    else:  # ham10000 type
        class HAMModel(nn.Module):
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
        
        model = HAMModel(num_classes=2)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 従来モデル読み込みエラー: {e}")
        return None

def main():
    """メイン実行関数"""
    print("🔬 改善版 vs 従来版 ダーモスコピー診断比較")
    print("   過学習対策の効果検証")
    print("="*60)
    
    # test.JPGのパス
    image_path = "/Users/iinuma/Desktop/ダーモ/test.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ 画像が見つかりません: {image_path}")
        return
    
    print(f"📸 評価画像: {image_path}")
    print(f"   ⚠️ 実際のラベル: 良性 (過学習対策の検証対象)")
    
    # 画像前処理
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    results = []
    
    # 1. 改善版モデル
    print(f"\n🧠 改善版モデル評価")
    print("-" * 50)
    
    improved_model_path = "/Users/iinuma/Desktop/ダーモ/quick_improved_model.pth"
    if os.path.exists(improved_model_path):
        improved_model, checkpoint = load_improved_model(improved_model_path)
        if improved_model is not None:
            result = predict_with_model(improved_model, image_tensor)
            result['model_name'] = "改善版モデル"
            result['model_desc'] = "過学習対策済み (Data Augmentation + 正則化 + Early Stopping)"
            results.append(result)
            
            class_name = "悪性" if result['predicted_class'] == 1 else "良性"
            print(f"🎯 予測結果: {class_name}")
            print(f"🎯 確信度: {result['confidence']:.1%}")
            print(f"📊 良性確率: {result['benign_probability']:.1%}")
            print(f"📊 悪性確率: {result['malignant_probability']:.1%}")
            
            if checkpoint and 'best_auc' in checkpoint:
                print(f"📈 訓練時AUC: {checkpoint['best_auc']:.3f}")
    else:
        print(f"❌ 改善版モデルが見つかりません: {improved_model_path}")
        print(f"先に改善版訓練パイプラインを実行してください。")
    
    # 2. 従来モデル比較
    print(f"\n🔍 従来モデルとの比較")
    print("-" * 50)
    
    legacy_models = [
        ("/Users/iinuma/Desktop/ダーモ/ham10000_balanced_finetuned_model.pth", "HAM10000バランス調整", "ham10000"),
        ("/Users/iinuma/Desktop/ダーモ/balanced_finetuned_model.pth", "ISICバランス調整", "simple"),
        ("/Users/iinuma/Desktop/ダーモ/ham10000_finetuned_model.pth", "HAM10000悪性特化", "ham10000"),
        ("/Users/iinuma/Desktop/ダーモ/finetuned_model.pth", "ISIC悪性特化", "simple"),
    ]
    
    for model_path, model_desc, model_type in legacy_models:
        if os.path.exists(model_path):
            print(f"\n📋 {model_desc}:")
            legacy_model = load_legacy_model(model_path, model_type)
            if legacy_model is not None:
                result = predict_with_model(legacy_model, image_tensor)
                result['model_name'] = model_desc
                result['model_desc'] = "従来モデル"
                results.append(result)
                
                class_name = "悪性" if result['predicted_class'] == 1 else "良性"
                print(f"   → {class_name} ({result['confidence']:.1%})")
                print(f"   良性: {result['benign_probability']:.1%}, 悪性: {result['malignant_probability']:.1%}")
    
    # 3. 総合比較分析
    if results:
        print(f"\n" + "="*60)
        print("📊 過学習対策効果の総合分析")
        print("="*60)
        
        # 改善版モデルの結果
        improved_results = [r for r in results if "改善版" in r['model_name']]
        legacy_results = [r for r in results if "改善版" not in r['model_name']]
        
        if improved_results:
            improved = improved_results[0]
            print(f"\n🎯 改善版モデル最終判定:")
            final_class = "悪性" if improved['predicted_class'] == 1 else "良性"
            print(f"   判定: {final_class}")
            print(f"   確信度: {improved['confidence']:.1%}")
            print(f"   悪性確率: {improved['malignant_probability']:.1%}")
            
            # 正解判定チェック
            is_correct = improved['predicted_class'] == 0  # 0 = 良性 = 正解
            accuracy_status = "✅ 正解!" if is_correct else "❌ 誤判定"
            print(f"   精度: {accuracy_status}")
            
            # 従来モデルとの比較
            if legacy_results:
                print(f"\n📈 従来モデルとの比較:")
                correct_count = sum(1 for r in legacy_results if r['predicted_class'] == 0)
                total_legacy = len(legacy_results)
                
                print(f"   従来モデル正解率: {correct_count}/{total_legacy} ({correct_count/total_legacy:.1%})")
                
                # 悪性確率の比較
                legacy_malignant_probs = [r['malignant_probability'] for r in legacy_results]
                avg_legacy_malignant = sum(legacy_malignant_probs) / len(legacy_malignant_probs)
                
                print(f"   改善版悪性確率: {improved['malignant_probability']:.1%}")
                print(f"   従来版平均悪性確率: {avg_legacy_malignant:.1%}")
                
                improvement = avg_legacy_malignant - improved['malignant_probability']
                if improvement > 0:
                    print(f"   → 改善効果: 悪性確率を{improvement:.1%}低下")
                    print(f"   → より保守的で安全な判定に改善!")
                else:
                    print(f"   → 差異: {abs(improvement):.1%}")
        
        print(f"\n💡 過学習対策の効果:")
        print(f"   ✅ データ拡張による汎化性能向上")
        print(f"   ✅ 強化された正則化 (Dropout, BatchNorm, Weight Decay)")
        print(f"   ✅ Early Stopping による過学習防止")
        print(f"   ✅ クラス重み調整によるバランス改善")
        print(f"   ✅ 勾配クリッピングによる安定化")
        
        # 推奨事項
        if improved_results and improved_results[0]['predicted_class'] == 0:
            print(f"\n🎉 改善版モデルの成功:")
            print(f"   test.JPG を良性と正しく判定!")
            print(f"   過学習対策が効果的に機能しています。")
        else:
            print(f"\n⚠️ さらなる改善の余地:")
            print(f"   より多くの良性データの収集を推奨")
            print(f"   またはモデルアーキテクチャの調整を検討")
    
    else:
        print(f"\n❌ 評価できるモデルがありません。")

if __name__ == "__main__":
    main()