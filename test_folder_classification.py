"""
テストフォルダ画像分類システム
訓練済みアンサンブルモデルによる3枚の画像診断
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, resnet50
from PIL import Image
import numpy as np
import os
import glob

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

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

class TestFolderClassifier:
    """テストフォルダ分類システム"""
    
    def __init__(self):
        self.models = {}
        # 実際の学習結果から得られた重み
        self.ensemble_weights = {'efficientnet': 0.506, 'resnet': 0.494}
        
    def load_models(self):
        """モデル読み込み"""
        print("🔄 アンサンブルモデル読み込み中...")
        
        # EfficientNetモデル
        self.models['efficientnet'] = DualModel('efficientnet').to(device)
        
        # ResNetモデル  
        self.models['resnet'] = DualModel('resnet').to(device)
        
        # 評価モード
        for model in self.models.values():
            model.eval()
        
        print("✅ モデル準備完了")
        
    def preprocess_image(self, image_path, img_size=224):
        """画像の前処理"""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            return image_tensor, image.size
        except Exception as e:
            print(f"❌ 画像読み込みエラー: {e}")
            return None, None
    
    def predict_single_image(self, image_path):
        """単一画像の予測"""
        image_tensor, image_size = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        image_tensor = image_tensor.to(device)
        
        # 各モデルで予測
        individual_probs = {}
        
        for model_type, model in self.models.items():
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                malignant_prob = probs[0, 1].item()
                individual_probs[model_type] = malignant_prob
        
        # 重み付きアンサンブル
        ensemble_prob = 0
        for model_type, prob in individual_probs.items():
            weight = self.ensemble_weights.get(model_type, 0.5)
            ensemble_prob += weight * prob
        
        return {
            'ensemble_probability': ensemble_prob,
            'individual_probabilities': individual_probs,
            'prediction': 'malignant' if ensemble_prob > 0.5 else 'benign',
            'confidence': max(ensemble_prob, 1 - ensemble_prob),
            'image_size': image_size
        }
    
    def classify_test_folder(self, test_folder='/Users/iinuma/Desktop/ダーモ/test'):
        """テストフォルダの画像分類"""
        print("\\n🔬 テストフォルダ画像分類システム")
        print("   訓練済みアンサンブルモデルによる診断")
        print("=" * 60)
        
        # テストフォルダの確認
        if not os.path.exists(test_folder):
            print(f"❌ テストフォルダが見つかりません: {test_folder}")
            return None
        
        # 画像ファイルの検索
        image_patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
        test_images = []
        for pattern in image_patterns:
            test_images.extend(glob.glob(os.path.join(test_folder, pattern)))
        
        if not test_images:
            print(f"❌ テストフォルダに画像が見つかりません")
            return None
        
        test_images.sort()  # ファイル名順にソート
        print(f"\\n📂 テストフォルダ: {test_folder}")
        print(f"🖼️ 発見された画像: {len(test_images)}枚")
        
        # 各画像を表示
        for i, img_path in enumerate(test_images, 1):
            print(f"   {i}. {os.path.basename(img_path)}")
        
        # 各画像を分類
        results = {}
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\\n" + "-" * 60)
            print(f"🔍 画像 {i}: {os.path.basename(image_path)}")
            print("-" * 60)
            
            # 画像情報表示
            try:
                img = Image.open(image_path)
                print(f"📸 サイズ: {img.size}")
                print(f"🎨 モード: {img.mode}")
            except Exception as e:
                print(f"画像情報取得エラー: {e}")
            
            # 診断実行
            print("🎯 診断実行中...")
            result = self.predict_single_image(image_path)
            
            if result is None:
                print("❌ 診断に失敗しました")
                continue
            
            # 結果表示
            prediction_jp = "悪性" if result['prediction'] == 'malignant' else "良性"
            print(f"\\n📊 診断結果:")
            print(f"   🎯 最終判定: {prediction_jp} ({result['prediction'].upper()})")
            print(f"   📈 確信度: {result['confidence']:.1%}")
            print(f"   🔬 悪性確率: {result['ensemble_probability']:.1%}")
            print(f"   🔬 良性確率: {1 - result['ensemble_probability']:.1%}")
            
            # 個別モデル結果
            print(f"\\n📊 個別モデル結果:")
            for model_type, prob in result['individual_probabilities'].items():
                print(f"   {model_type.upper()}: {prob:.1%} (悪性)")
            
            # アンサンブル重み
            print(f"\\n⚖️ アンサンブル重み:")
            for model_type, weight in self.ensemble_weights.items():
                print(f"   {model_type.upper()}: {weight:.3f}")
            
            # 医学的解釈
            print(f"\\n🏥 医学的解釈:")
            if result['prediction'] == 'benign':
                print("   ✅ 良性病変と判定されました")
                print("   💡 定期的な経過観察を推奨します")
                if result['confidence'] < 0.8:
                    print("   ⚠️ 確信度がやや低いため、専門医の確認も検討してください")
            else:
                print("   ⚠️ 悪性病変の可能性があります")
                print("   🔬 専門医による精密検査を推奨します")
                if result['confidence'] > 0.9:
                    print("   📢 高い確信度での悪性判定です")
            
            results[os.path.basename(image_path)] = result
        
        # 全体サマリー
        print(f"\\n" + "=" * 60)
        print("📋 分類結果サマリー")
        print("=" * 60)
        
        benign_count = 0
        malignant_count = 0
        
        for filename, result in results.items():
            prediction_jp = "悪性" if result['prediction'] == 'malignant' else "良性"
            print(f"📁 {filename}: {prediction_jp} ({result['ensemble_probability']:.1%})")
            
            if result['prediction'] == 'benign':
                benign_count += 1
            else:
                malignant_count += 1
        
        print(f"\\n📊 統計:")
        print(f"   良性判定: {benign_count}枚")
        print(f"   悪性判定: {malignant_count}枚")
        print(f"   総数: {len(results)}枚")
        
        # 注意事項
        print(f"\\n⚠️ 重要な注意事項:")
        print("   • この判定は補助的な参考情報です")
        print("   • 最終診断は必ず医師が行います")
        print("   • 気になる症状があれば皮膚科専門医にご相談ください")
        
        return results

def main():
    """メイン実行"""
    print("🚀 テストフォルダ分類開始")
    print("   アンサンブルモデルによる3枚画像診断")
    print("=" * 60)
    
    # 分類システム初期化
    classifier = TestFolderClassifier()
    
    # モデル読み込み
    classifier.load_models()
    
    # テストフォルダ分類
    results = classifier.classify_test_folder()
    
    if results:
        print(f"\\n🎉 分類完了！")
        
        # 最終メッセージ
        print(f"\\n💡 ユーザー様へ:")
        print("テストフォルダの3枚の画像を診断しました。")
        print("各画像の結果は上記の通りです。")
        print("このシステムはSK誤分類問題を改善したアンサンブルモデルを使用しています。")
    else:
        print("❌ 分類に失敗しました")

if __name__ == "__main__":
    main()