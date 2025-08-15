"""
学習済みモデルでダーモスコピー画像を判定
2段階ファインチューニングモデル対応版
感度・特異度計算機能付き
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class PretrainedModel(nn.Module):
    """事前学習用モデル（推論用）"""
    
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

def load_model(model_path):
    """学習済みモデルの読み込み"""
    model = PretrainedModel(num_classes=2)
    
    try:
        # モデルの重みを読み込み
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✅ モデルを読み込みました: {model_path}")
        return model
    
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None

def preprocess_image(image_path):
    """画像の前処理"""
    
    # 画像変換パイプライン
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # 画像読み込み
        image = Image.open(image_path).convert('RGB')
        print(f"📸 画像を読み込みました: {image_path}")
        print(f"    サイズ: {image.size}")
        
        # 前処理適用
        image_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加
        return image_tensor
    
    except Exception as e:
        print(f"❌ 画像読み込みエラー: {e}")
        return None

def predict_image(model, image_tensor):
    """画像の予測"""
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 予測実行
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # 予測結果
        benign_prob = probabilities[0][0].item()    # 良性の確率
        malignant_prob = probabilities[0][1].item() # 悪性の確率
        
        predicted_class = 1 if malignant_prob > benign_prob else 0
        confidence = max(benign_prob, malignant_prob)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob
        }

def evaluate_batch(model, image_paths, true_labels=None):
    """複数画像のバッチ評価と感度・特異度計算
    
    Args:
        model: 学習済みモデル
        image_paths: 画像パスのリスト
        true_labels: 正解ラベルのリスト（0: 良性, 1: 悪性）
    
    Returns:
        predictions: 予測結果のリスト
        metrics: 感度・特異度を含む評価指標（true_labelsが提供された場合）
    """
    
    predictions = []
    pred_labels = []
    pred_probs = []
    
    for img_path in image_paths:
        # 画像前処理
        image_tensor = preprocess_image(img_path)
        if image_tensor is None:
            continue
        
        # 予測実行
        result = predict_image(model, image_tensor)
        predictions.append(result)
        pred_labels.append(result['predicted_class'])
        pred_probs.append(result['malignant_probability'])
    
    # 評価指標の計算（正解ラベルがある場合）
    metrics = None
    if true_labels is not None and len(true_labels) == len(pred_labels):
        # 混同行列
        cm = confusion_matrix(true_labels, pred_labels)
        
        # 感度（Sensitivity）: TP / (TP + FN)
        # 悪性を正しく悪性と判定する割合
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 特異度（Specificity）: TN / (TN + FP)
        # 良性を正しく良性と判定する割合
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 精度（Accuracy）
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # 陽性的中率（PPV: Positive Predictive Value）
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # 陰性的中率（NPV: Negative Predictive Value）
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # F1スコア
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        # AUC計算
        try:
            auc = roc_auc_score(true_labels, pred_probs)
        except:
            auc = None
        
        metrics = {
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'ppv': ppv,
            'npv': npv,
            'f1_score': f1,
            'auc': auc,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        # 詳細レポート出力
        print("\n" + "="*50)
        print("📊 評価指標")
        print("="*50)
        print(f"感度 (Sensitivity/Recall): {sensitivity:.1%}")
        print(f"  → 悪性病変を正しく検出する能力")
        print(f"特異度 (Specificity): {specificity:.1%}")
        print(f"  → 良性病変を正しく識別する能力")
        print(f"精度 (Accuracy): {accuracy:.1%}")
        print(f"陽性的中率 (PPV/Precision): {ppv:.1%}")
        print(f"陰性的中率 (NPV): {npv:.1%}")
        print(f"F1スコア: {f1:.3f}")
        if auc is not None:
            print(f"AUC: {auc:.3f}")
        
        print(f"\n混同行列:")
        print(f"  実際\\予測   良性  悪性")
        print(f"  良性        {tn:4d}  {fp:4d}")
        print(f"  悪性        {fn:4d}  {tp:4d}")
    
    return predictions, metrics

def plot_roc_curve(true_labels, pred_probs, save_path='roc_curve.png'):
    """ROC曲線の描画
    
    Args:
        true_labels: 正解ラベル
        pred_probs: 予測確率（悪性の確率）
        save_path: 保存先パス
    """
    
    try:
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        auc = roc_auc_score(true_labels, pred_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve for Dermoscopy Classification')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 ROC曲線を '{save_path}' に保存しました")
    except Exception as e:
        print(f"ROC曲線の描画エラー: {e}")

def main():
    """メイン実行関数"""
    
    print("🔬 ダーモスコピー画像診断システム")
    print("   2段階ファインチューニングモデル")
    print("   (ImageNet → ISIC → ユーザーデータ)")
    print("=" * 50)
    
    # 画像パス
    image_path = "/Users/iinuma/Desktop/ダーモ/test.JPG"
    
    # 2段階ファインチューニングモデルの説明
    print("\n📚 モデル説明:")
    print("1. balanced_finetuned_model.pth:")
    print("   - 3段階学習: ImageNet → ISIC → 悪性データ → SK良性データ")
    print("   - データバランス調整済み（良性・悪性バランス改善）")
    print("   - 最新の改善モデル（推奨）")
    print("\n2. finetuned_model.pth:")
    print("   - 2段階学習: ImageNet → ISIC → 悪性データのみ")
    print("   - 悪性に偏った判定傾向")
    print("\n3. isic_pretrained_model.pth:")
    print("   - ISICデータのみで学習（中間段階）")
    print("\n4. quick_dermoscopy_model.pth:")
    print("   - 簡易版モデル（初期テスト用）")
    
    # 利用可能なモデルを確認
    model_paths = [
        "/Users/iinuma/Desktop/ダーモ/balanced_finetuned_model.pth", # 3段階バランス調整済み（最新）
        "/Users/iinuma/Desktop/ダーモ/finetuned_model.pth",        # 2段階ファインチューニング済み
        "/Users/iinuma/Desktop/ダーモ/isic_pretrained_model.pth",  # ISIC事前学習のみ
        "/Users/iinuma/Desktop/ダーモ/quick_dermoscopy_model.pth", # 簡易モデル
    ]
    
    results = []
    
    for i, model_path in enumerate(model_paths):
        model_name = model_path.split('/')[-1]
        print(f"\n🧠 モデル {i+1}: {model_name}")
        print("-" * 40)
        
        # モデル読み込み
        model = load_model(model_path)
        if model is None:
            continue
        
        # 画像前処理
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            continue
        
        # 予測実行
        result = predict_image(model, image_tensor)
        result['model_name'] = model_name
        results.append(result)
        
        # 結果表示
        class_name = "悪性" if result['predicted_class'] == 1 else "良性"
        print(f"📊 予測結果: {class_name}")
        print(f"🎯 確信度: {result['confidence']:.1%}")
        print(f"📈 良性確率: {result['benign_probability']:.1%}")
        print(f"📈 悪性確率: {result['malignant_probability']:.1%}")
    
    # 総合判定
    if results:
        print("\n" + "=" * 50)
        print("🏆 総合判定")
        print("=" * 50)
        
        # 各モデルの予測を集計
        malignant_votes = sum(1 for r in results if r['predicted_class'] == 1)
        benign_votes = len(results) - malignant_votes
        
        # 平均確率計算
        avg_malignant_prob = np.mean([r['malignant_probability'] for r in results])
        avg_benign_prob = np.mean([r['benign_probability'] for r in results])
        
        final_prediction = "悪性" if avg_malignant_prob > avg_benign_prob else "良性"
        final_confidence = max(avg_malignant_prob, avg_benign_prob)
        
        print(f"📷 画像: test.JPG")
        print(f"🎯 最終判定: {final_prediction}")
        print(f"🎯 総合確信度: {final_confidence:.1%}")
        print(f"📊 モデル合意: {malignant_votes}/{len(results)} が悪性と予測")
        print(f"📈 平均良性確率: {avg_benign_prob:.1%}")
        print(f"📈 平均悪性確率: {avg_malignant_prob:.1%}")
        
        # 詳細結果
        print(f"\n📋 各モデルの詳細結果:")
        for result in results:
            class_name = "悪性" if result['predicted_class'] == 1 else "良性"
            print(f"  • {result['model_name']}: {class_name} ({result['confidence']:.1%})")
    
        
        # テストデータセットでの評価（オプション）
        print("\n" + "="*50)
        print("🧪 テストデータセットでの評価（デモ）")
        print("="*50)
        
        # デモ用: 複数画像での評価例
        # 実際の使用時は適切な画像パスと正解ラベルを設定
        test_image_paths = [image_path]  # デモでは1枚のみ
        test_true_labels = None  # 正解ラベルが不明な場合はNone
        
        # 最良モデル（finetuned_model）で評価
        best_model_path = "/Users/iinuma/Desktop/ダーモ/finetuned_model.pth"
        if os.path.exists(best_model_path):
            print(f"\n🎯 最良モデルでバッチ評価: {best_model_path.split('/')[-1]}")
            best_model = load_model(best_model_path)
            if best_model:
                batch_predictions, batch_metrics = evaluate_batch(
                    best_model, test_image_paths, test_true_labels
                )
                
                if batch_metrics:
                    print("\n📊 バッチ評価完了")
                    print(f"感度: {batch_metrics['sensitivity']:.1%}")
                    print(f"特異度: {batch_metrics['specificity']:.1%}")
        
        print("\n💡 ヒント: 感度・特異度を計算するには、正解ラベル付きの")
        print("   テストデータセットでevaluate_batch()関数を使用してください。")
    
    else:
        print("\n❌ 予測を実行できませんでした。")

if __name__ == "__main__":
    main()