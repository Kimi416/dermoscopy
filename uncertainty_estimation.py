"""
不確実性推定システム
SK誤分類対策のためのモンテカルロドロップアウト実装
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class UncertaintyModel(nn.Module):
    """不確実性推定対応モデル"""
    
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
    
    def enable_dropout(self):
        """推論時にもドロップアウトを有効化"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def monte_carlo_predict(self, x, n_samples=50):
        """モンテカルロドロップアウト予測"""
        self.eval()
        self.enable_dropout()  # ドロップアウトのみ有効化
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self(x)
                probabilities = torch.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        predictions = np.array(predictions)  # [n_samples, batch_size, n_classes]
        return predictions

def load_model():
    """訓練済みモデルの読み込み"""
    model_path = '/Users/iinuma/Desktop/ダーモ/disease_classification_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return None
    
    model = UncertaintyModel(num_classes=2, dropout_rate=0.3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ 不確実性推定モデルを読み込みました")
    return model

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

def calculate_uncertainty_metrics(predictions):
    """不確実性メトリクスの計算"""
    # predictions: [n_samples, batch_size, n_classes]
    predictions = predictions[:, 0, :]  # バッチサイズ1を仮定
    
    # 平均予測
    mean_prediction = np.mean(predictions, axis=0)
    
    # 予測分散 (Predictive Variance)
    prediction_variance = np.var(predictions, axis=0)
    
    # エントロピー (Predictive Entropy)
    entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-8))
    
    # 相互情報量 (Mutual Information)
    # E[H[y|x,θ]] - H[E[y|x,D]]
    individual_entropies = [-np.sum(pred * np.log(pred + 1e-8)) for pred in predictions]
    expected_entropy = np.mean(individual_entropies)
    mutual_information = entropy - expected_entropy
    
    # 分散比 (Variation Ratio) - 最頻値以外の予測頻度
    predicted_classes = np.argmax(predictions, axis=1)
    mode_count = np.max(np.bincount(predicted_classes))
    variation_ratio = 1 - (mode_count / len(predictions))
    
    return {
        'mean_prediction': mean_prediction,
        'prediction_variance': prediction_variance,
        'entropy': entropy,
        'mutual_information': mutual_information,
        'variation_ratio': variation_ratio,
        'predictions': predictions
    }

def predict_with_uncertainty(model, image_path, n_samples=50):
    """不確実性を含む予測"""
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None
    
    image_tensor = image_tensor.to(device)
    
    # モンテカルロ予測
    print(f"🎲 モンテカルロサンプリング実行中... (n={n_samples})")
    predictions = model.monte_carlo_predict(image_tensor, n_samples)
    
    # 不確実性メトリクス計算
    uncertainty_metrics = calculate_uncertainty_metrics(predictions)
    
    # 基本的な結果
    mean_pred = uncertainty_metrics['mean_prediction']
    benign_prob = mean_pred[0]
    malignant_prob = mean_pred[1]
    predicted_class = 1 if malignant_prob > benign_prob else 0
    confidence = max(benign_prob, malignant_prob)
    
    # 信頼性評価
    high_uncertainty = uncertainty_metrics['entropy'] > 0.5  # 閾値は調整可能
    high_variance = np.max(uncertainty_metrics['prediction_variance']) > 0.1
    low_consensus = uncertainty_metrics['variation_ratio'] > 0.3
    
    reliability_flags = {
        'high_uncertainty': high_uncertainty,
        'high_variance': high_variance, 
        'low_consensus': low_consensus
    }
    
    # 総合的な信頼性スコア
    reliability_issues = sum(reliability_flags.values())
    reliability_score = max(0, 1 - (reliability_issues / 3))  # 0-1スケール
    
    return {
        'predicted_class': predicted_class,
        'predicted_type': 'malignant' if predicted_class == 1 else 'benign',
        'confidence': confidence,
        'benign_probability': benign_prob,
        'malignant_probability': malignant_prob,
        'uncertainty_metrics': uncertainty_metrics,
        'reliability_flags': reliability_flags,
        'reliability_score': reliability_score,
        'n_samples': n_samples
    }

def analyze_prediction_distribution(result):
    """予測分布の分析"""
    predictions = result['uncertainty_metrics']['predictions']
    
    print(f"\n📊 予測分布分析:")
    print("-" * 40)
    
    # クラス別予測確率の統計
    benign_probs = predictions[:, 0]
    malignant_probs = predictions[:, 1]
    
    print(f"良性確率:")
    print(f"  平均: {np.mean(benign_probs):.3f}")
    print(f"  標準偏差: {np.std(benign_probs):.3f}")
    print(f"  範囲: [{np.min(benign_probs):.3f}, {np.max(benign_probs):.3f}]")
    
    print(f"\n悪性確率:")
    print(f"  平均: {np.mean(malignant_probs):.3f}")
    print(f"  標準偏差: {np.std(malignant_probs):.3f}")
    print(f"  範囲: [{np.min(malignant_probs):.3f}, {np.max(malignant_probs):.3f}]")
    
    # 予測の一貫性
    predicted_classes = np.argmax(predictions, axis=1)
    class_counts = np.bincount(predicted_classes)
    consensus_ratio = np.max(class_counts) / len(predictions)
    
    print(f"\n予測一貫性:")
    print(f"  良性予測: {class_counts[0]}回 ({class_counts[0]/len(predictions):.1%})")
    print(f"  悪性予測: {class_counts[1]}回 ({class_counts[1]/len(predictions):.1%})")
    print(f"  合意率: {consensus_ratio:.1%}")

def generate_reliability_assessment(result):
    """信頼性評価の生成"""
    metrics = result['uncertainty_metrics']
    flags = result['reliability_flags']
    score = result['reliability_score']
    
    print(f"\n🔍 信頼性評価:")
    print("-" * 40)
    print(f"信頼性スコア: {score:.1%}")
    
    # フラグ別の解釈
    if flags['high_uncertainty']:
        print("⚠️ 高い不確実性: モデルが判定に迷っています")
    
    if flags['high_variance']:
        print("⚠️ 高い分散: 予測が安定していません")
    
    if flags['low_consensus']:
        print("⚠️ 低い合意: サンプル間で予測が分散しています")
    
    # 不確実性メトリクスの表示
    print(f"\n📈 不確実性メトリクス:")
    print(f"  エントロピー: {metrics['entropy']:.3f}")
    print(f"  相互情報量: {metrics['mutual_information']:.3f}")
    print(f"  分散比: {metrics['variation_ratio']:.3f}")
    print(f"  予測分散 (良性): {metrics['prediction_variance'][0]:.3f}")
    print(f"  予測分散 (悪性): {metrics['prediction_variance'][1]:.3f}")

def recommend_action(result):
    """推奨アクションの生成"""
    score = result['reliability_score']
    predicted_type = result['predicted_type']
    confidence = result['confidence']
    
    print(f"\n💡 推奨アクション:")
    print("-" * 40)
    
    if score >= 0.8:
        print("✅ 高い信頼性 - 判定結果を信頼できます")
        if predicted_type == 'malignant':
            print("🔬 悪性判定: 専門医による確認を推奨")
        else:
            print("👀 良性判定: 定期的な経過観察を推奨")
    
    elif score >= 0.5:
        print("⚠️ 中程度の信頼性 - 追加検査を検討")
        print("🔄 別角度からの撮影画像での再判定を推奨")
        if predicted_type == 'malignant':
            print("🚨 悪性の可能性があるため、速やかな専門医受診を推奨")
    
    else:
        print("❌ 低い信頼性 - 判定結果は参考程度")
        print("🏥 専門医による直接診断を強く推奨")
        print("📸 高品質な画像での再撮影を検討")
    
    # 特殊ケース: SK誤分類対策
    if (predicted_type == 'malignant' and 
        confidence > 0.95 and 
        result['uncertainty_metrics']['entropy'] > 0.3):
        print("\n🎯 SK誤分類の可能性:")
        print("   高い確信度で悪性判定されていますが、")
        print("   不確実性も高いため、脂漏性角化症(SK)の")
        print("   可能性も考慮して専門医にご相談ください")

def main():
    """メイン実行関数"""
    print("🎲 不確実性推定による診断システム")
    print("   SK誤分類対策版")
    print("=" * 60)
    
    # モデル読み込み
    model = load_model()
    if model is None:
        return
    
    # 診断対象画像
    image_path = '/Users/iinuma/Desktop/ダーモ/images.jpeg'
    
    if not os.path.exists(image_path):
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return
    
    print(f"\n📂 診断対象: {os.path.basename(image_path)}")
    
    # 不確実性推定付き予測
    result = predict_with_uncertainty(model, image_path, n_samples=100)
    
    if result is None:
        print("❌ 診断に失敗しました")
        return
    
    # 結果表示
    print(f"\n" + "=" * 60)
    print("🎯 診断結果（不確実性推定付き）")
    print("=" * 60)
    
    prediction_jp = "悪性" if result['predicted_type'] == 'malignant' else "良性"
    print(f"📊 判定: {prediction_jp} ({result['predicted_type'].upper()})")
    print(f"🎯 確信度: {result['confidence']:.1%}")
    print(f"🔄 サンプル数: {result['n_samples']}回")
    
    print(f"\n📈 平均確率:")
    print(f"   良性: {result['benign_probability']:.1%}")
    print(f"   悪性: {result['malignant_probability']:.1%}")
    
    # 詳細分析
    analyze_prediction_distribution(result)
    generate_reliability_assessment(result)
    recommend_action(result)
    
    # 前回の結果と比較
    print(f"\n📋 従来システムとの比較:")
    print("-" * 40)
    print("前回: 悪性 99.8%確信度（不確実性推定なし）")
    print(f"今回: {prediction_jp} {result['confidence']:.1%}確信度")
    print(f"信頼性スコア: {result['reliability_score']:.1%}")
    
    if result['reliability_score'] < 0.8:
        print("💡 不確実性推定により、判定の信頼性に")
        print("   問題があることが検出されました！")

if __name__ == "__main__":
    main()