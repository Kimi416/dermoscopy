"""
SK特化分類システム
脂漏性角化症の特徴に基づく段階的判定
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class SKClassifier:
    """SK特化分類器"""
    
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.sk_features = {
            'color_clusters': None,
            'texture_patterns': None,
            'shape_features': None
        }
    
    def load_model(self, model_path):
        """モデル読み込み（柔軟な形式対応）"""
        if not os.path.exists(model_path):
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return None
        
        model = self.create_model()
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 複数の形式を試行
            if isinstance(checkpoint, dict):
                # 辞書形式の場合
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # 辞書自体がstate_dictの場合
                    state_dict = checkpoint
            else:
                # 直接state_dictの場合
                state_dict = checkpoint
            
            # state_dictの読み込みを試行
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"⚠️ Strict読み込み失敗: {str(e)[:100]}...")
                # より緩い読み込みを試行
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ 部分的なstate_dict読み込み成功")
                except RuntimeError as e2:
                    print(f"⚠️ 部分読み込みも失敗: {str(e2)[:100]}...")
                    # 基本モデルのままSK特徴分析のみ使用
                    print("🔧 基本モデルでSK特徴分析のみ実行")
            
            model.to(device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {str(e)[:100]}...")
            return None
    
    def create_model(self):
        """モデル作成"""
        model = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = model.classifier[1].in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        return model
    
    def extract_sk_features(self, image_path):
        """SK特有の特徴抽出"""
        try:
            # PIL画像読み込み
            pil_image = Image.open(image_path).convert('RGB')
            
            # OpenCV用にnumpy配列に変換
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            features = {}
            
            # 1. 色彩特徴解析（SK特有の褐色調）
            features.update(self._analyze_color_features(image_rgb))
            
            # 2. テクスチャ特徴解析（表面の粗さ）
            features.update(self._analyze_texture_features(image))
            
            # 3. 形状特徴解析（境界の明瞭性）
            features.update(self._analyze_shape_features(image))
            
            # 4. SK特有パターン検出
            features.update(self._detect_sk_patterns(image_rgb))
            
            return features
            
        except Exception as e:
            print(f"⚠️ 特徴抽出エラー: {e}")
            return {}
    
    def _analyze_color_features(self, image_rgb):
        """色彩特徴の解析"""
        # HSV変換
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # SK特有の褐色範囲 (Hue: 10-30, Saturation: 50-255, Value: 50-200)
        sk_brown_lower = np.array([5, 50, 50])
        sk_brown_upper = np.array([35, 255, 200])
        brown_mask = cv2.inRange(hsv, sk_brown_lower, sk_brown_upper)
        brown_ratio = np.sum(brown_mask > 0) / (image_rgb.shape[0] * image_rgb.shape[1])
        
        # 色彩の均一性（SKは比較的均一）
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        color_variance = np.var(gray)
        
        # 平均色調
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        return {
            'brown_ratio': brown_ratio,
            'color_variance': color_variance,
            'mean_hue': mean_hue,
            'mean_saturation': mean_saturation,
            'mean_value': mean_value
        }
    
    def _analyze_texture_features(self, image):
        """テクスチャ特徴の解析"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 画像サイズを制限してパフォーマンス向上（機能は保持）
        if gray.shape[0] > 512 or gray.shape[1] > 512:
            gray = cv2.resize(gray, (512, 512))
        
        # LBP (Local Binary Pattern) - テクスチャの局所パターン
        def calculate_lbp(image, radius=1, neighbors=8):
            lbp = np.zeros_like(image, dtype=np.uint8)
            height, width = image.shape
            
            # 境界を考慮した効率的な実装
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center = image[i, j]
                    binary_val = 0
                    
                    for k in range(neighbors):
                        angle = 2 * np.pi * k / neighbors
                        x = int(round(i + radius * np.cos(angle)))
                        y = int(round(j + radius * np.sin(angle)))
                        
                        # 境界チェック
                        x = max(0, min(x, height - 1))
                        y = max(0, min(y, width - 1))
                        
                        if image[x, y] >= center:
                            binary_val |= (1 << k)
                    
                    lbp[i, j] = binary_val
            return lbp
        
        lbp = calculate_lbp(gray)
        lbp_variance = np.var(lbp)
        
        # エッジ密度（SKは比較的滑らかなエッジ）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 表面の粗さ（標準偏差）
        surface_roughness = np.std(gray)
        
        return {
            'lbp_variance': lbp_variance,
            'edge_density': edge_density,
            'surface_roughness': surface_roughness
        }
    
    def _analyze_shape_features(self, image):
        """形状特徴の解析"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 輪郭検出
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 最大輪郭
            max_contour = max(contours, key=cv2.contourArea)
            
            # 輪郭の滑らかさ（周囲長と面積の比）
            perimeter = cv2.arcLength(max_contour, True)
            area = cv2.contourArea(max_contour)
            roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 境界の明瞭性
            boundary_sharpness = np.mean(edges) / 255.0
            
            return {
                'roundness': roundness,
                'boundary_sharpness': boundary_sharpness,
                'contour_area_ratio': area / (gray.shape[0] * gray.shape[1])
            }
        
        return {
            'roundness': 0,
            'boundary_sharpness': 0,
            'contour_area_ratio': 0
        }
    
    def _detect_sk_patterns(self, image_rgb):
        """SK特有パターンの検出"""
        # 毛嚢開口部様構造の検出
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 円形構造の検出（ハフ変換）
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=1, maxRadius=10
        )
        
        circle_count = len(circles[0]) if circles is not None else 0
        
        # コメド様構造（暗い点状構造）
        kernel = np.ones((3, 3), np.uint8)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        dark_spots = np.sum(blackhat > 30)
        
        return {
            'circle_count': circle_count,
            'dark_spots': dark_spots,
            'comedonal_pattern': dark_spots / (gray.shape[0] * gray.shape[1])
        }
    
    def calculate_sk_score(self, features):
        """SK尤度スコアの計算（強化版）"""
        score = 0.0
        confidence_factors = []
        
        # 色彩スコア（褐色調の強さ）- 重み増加
        if 'brown_ratio' in features:
            brown_score = min(features['brown_ratio'] * 2.5, 1.0)  # より敏感に
            score += brown_score * 0.4  # 重み増加 0.3 → 0.4
            confidence_factors.append(f"褐色調: {brown_score:.2f}")
        
        # テクスチャスコア（表面の滑らかさ）
        if 'edge_density' in features:
            # SKは比較的滑らかなので、エッジ密度が低い方が高スコア
            smoothness_score = max(0, 1 - features['edge_density'] * 8)  # より緩い閾値
            score += smoothness_score * 0.25
            confidence_factors.append(f"滑らかさ: {smoothness_score:.2f}")
        
        # 境界明瞭性スコア - 重み増加
        if 'boundary_sharpness' in features:
            # SKは境界が比較的明瞭
            boundary_score = min(features['boundary_sharpness'] * 1.2, 1.0)  # 強化
            score += boundary_score * 0.25  # 重み増加 0.2 → 0.25
            confidence_factors.append(f"境界明瞭性: {boundary_score:.2f}")
        
        # 色彩均一性スコア（SK特有の特徴）
        if 'color_variance' in features:
            # 正規化された色彩分散（低い方が良い）
            uniformity_score = max(0, 1 - features['color_variance'] / 8000)  # より厳格
            score += uniformity_score * 0.1
            confidence_factors.append(f"色彩均一性: {uniformity_score:.2f}")
        
        # SK特有の追加特徴量
        if 'mean_hue' in features and 'mean_saturation' in features:
            # SK典型的な色相・彩度範囲
            hue_match = 1.0 if 10 <= features['mean_hue'] <= 30 else 0.0
            saturation_match = 1.0 if 100 <= features['mean_saturation'] <= 200 else 0.5
            color_profile_score = (hue_match + saturation_match) / 2
            score += color_profile_score * 0.15
            confidence_factors.append(f"色相プロファイル: {color_profile_score:.2f}")
        
        # 表面粗さ（SK特徴）
        if 'surface_roughness' in features:
            # SKは中程度の表面粗さ
            roughness_score = 1.0 - abs(features['surface_roughness'] - 30) / 30
            roughness_score = max(0, roughness_score)
            score += roughness_score * 0.1
            confidence_factors.append(f"表面粗さ適合: {roughness_score:.2f}")
        
        # 円形構造（毛嚢開口部様）
        if 'circle_count' in features:
            circle_score = min(features['circle_count'] / 10, 1.0)  # 10個以上で最大
            score += circle_score * 0.05
            confidence_factors.append(f"円形構造: {circle_score:.2f}")
        
        return score, confidence_factors
    
    def predict_with_sk_analysis(self, image_path):
        """SK分析を含む予測（モデル読み込み失敗時も対応）"""
        # SK特徴分析（必須）
        sk_features = self.extract_sk_features(image_path)
        sk_score, confidence_factors = self.calculate_sk_score(sk_features)
        
        # 基本予測（モデルが利用可能な場合のみ）
        base_benign_prob = 0.5  # デフォルト値
        base_malignant_prob = 0.5
        model_prediction_available = False
        
        if self.model is not None:
            image_tensor = self._preprocess_image(image_path)
            if image_tensor is not None:
                try:
                    with torch.no_grad():
                        image_tensor = image_tensor.to(device)
                        output = self.model(image_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        
                        base_benign_prob = probabilities[0][0].item()
                        base_malignant_prob = probabilities[0][1].item()
                        model_prediction_available = True
                except Exception as e:
                    print(f"⚠️ モデル予測エラー: {str(e)[:50]}...")
        
        # SK補正の適用（バランス調整版）
        sk_threshold = 0.45  # SK尤度閾値を調整（0.3→0.45で悪性見逃し防止）
        
        if sk_score > sk_threshold:
            if model_prediction_available:
                # モデル予測がある場合：強化された補正
                correction_factor = (sk_score - sk_threshold) / (1 - sk_threshold)
                # SK確信度が高い場合はより強い補正
                correction_strength = 0.8 if sk_score > 0.6 else 0.6
                corrected_benign_prob = base_benign_prob + (1 - base_benign_prob) * correction_factor * correction_strength
                corrected_malignant_prob = 1 - corrected_benign_prob
            else:
                # モデル予測がない場合：SK特徴のみで判定（強化）
                sk_strength = (sk_score - sk_threshold) / (1 - sk_threshold)
                corrected_benign_prob = 0.65 + sk_strength * 0.3  # 0.65-0.95の範囲
                corrected_malignant_prob = 1 - corrected_benign_prob
            
            applied_correction = True
        else:
            corrected_benign_prob = base_benign_prob
            corrected_malignant_prob = base_malignant_prob
            applied_correction = False
        
        # 最終判定
        predicted_class = 0 if corrected_benign_prob > corrected_malignant_prob else 1
        confidence = max(corrected_benign_prob, corrected_malignant_prob)
        
        return {
            'predicted_class': predicted_class,
            'predicted_type': 'benign' if predicted_class == 0 else 'malignant',
            'confidence': confidence,
            'benign_probability': corrected_benign_prob,
            'malignant_probability': corrected_malignant_prob,
            'base_benign_probability': base_benign_prob,
            'base_malignant_probability': base_malignant_prob,
            'sk_score': sk_score,
            'sk_features': sk_features,
            'confidence_factors': confidence_factors,
            'correction_applied': applied_correction,
            'sk_threshold': sk_threshold,
            'model_prediction_available': model_prediction_available
        }
    
    def _preprocess_image(self, image_path):
        """画像前処理"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            return transform(image).unsqueeze(0)
        except Exception as e:
            print(f"❌ 画像読み込みエラー: {e}")
            return None

def main():
    """メイン実行"""
    print("🔬 SK特化分類システム")
    print("   脂漏性角化症対応版")
    print("=" * 60)
    
    # モデル読み込み
    model_path = '/Users/iinuma/Desktop/ダーモ/disease_classification_model.pth'
    classifier = SKClassifier(model_path)
    
    if classifier.model is None:
        return
    
    # 診断実行
    image_path = '/Users/iinuma/Desktop/ダーモ/images.jpeg'
    
    if not os.path.exists(image_path):
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return
    
    print(f"\n📂 診断対象: {os.path.basename(image_path)}")
    print("🔍 SK特徴分析実行中...")
    
    result = classifier.predict_with_sk_analysis(image_path)
    
    if result is None:
        print("❌ 診断に失敗しました")
        return
    
    # 結果表示
    print(f"\n" + "=" * 60)
    print("🎯 SK特化診断結果")
    print("=" * 60)
    
    prediction_jp = "良性" if result['predicted_type'] == 'benign' else "悪性"
    print(f"📊 最終判定: {prediction_jp} ({result['predicted_type'].upper()})")
    print(f"🎯 確信度: {result['confidence']:.1%}")
    
    print(f"\n📈 確率詳細:")
    print(f"   良性: {result['benign_probability']:.1%}")
    print(f"   悪性: {result['malignant_probability']:.1%}")
    
    print(f"\n🔧 補正前の結果:")
    print(f"   良性: {result['base_benign_probability']:.1%}")
    print(f"   悪性: {result['base_malignant_probability']:.1%}")
    
    print(f"\n🎯 SK分析:")
    print(f"   SK尤度スコア: {result['sk_score']:.3f}")
    print(f"   SK閾値: {result['sk_threshold']:.3f}")
    print(f"   補正適用: {'✅' if result['correction_applied'] else '❌'}")
    
    print(f"\n📊 SK特徴詳細:")
    features = result['sk_features']
    for key, value in features.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n💡 信頼性要因:")
    for factor in result['confidence_factors']:
        print(f"   • {factor}")
    
    # 医学的解釈
    print(f"\n🏥 医学的解釈:")
    if result['predicted_type'] == 'benign':
        print("✅ 良性病変と判定されました")
        if result['sk_score'] > result['sk_threshold']:
            print("🔍 SK（脂漏性角化症）の特徴を強く示しています")
        print("👀 定期的な経過観察を推奨")
    else:
        print("⚠️ 悪性病変の可能性があります")
        print("🔬 専門医による精密検査を推奨")
        
        if result['sk_score'] > result['sk_threshold'] * 0.8:
            print("💡 ただし、SK様の特徴も認められるため、")
            print("   専門医による鑑別診断をお勧めします")
    
    # 改善提案
    if result['correction_applied']:
        print(f"\n✨ 改善点:")
        print("   SK特化補正により、より適切な判定が可能になりました")
        
        improvement = abs(result['base_malignant_probability'] - result['malignant_probability'])
        print(f"   悪性確率を{improvement:.1%}補正しました")

if __name__ == "__main__":
    main()