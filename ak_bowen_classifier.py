"""
AK・Bowen病特化分類システム
光線性角化症・ボーエン病の特徴に基づく診断支援
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

class AKBowenClassifier:
    """AK・Bowen病特化分類器"""
    
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.ak_bowen_features = {
            'scale_patterns': None,
            'keratinization': None,
            'surface_texture': None,
            'color_heterogeneity': None
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
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            try:
                model.load_state_dict(state_dict, strict=False)
                print("✅ AK・Bowen病分類器読み込み成功")
            except RuntimeError as e:
                print(f"⚠️ モデル読み込み警告: {str(e)[:50]}...")
                print("🔧 特徴分析のみ実行")
            
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
    
    def extract_ak_bowen_features(self, image_path):
        """AK・Bowen病特有の特徴抽出"""
        try:
            # PIL画像読み込み
            pil_image = Image.open(image_path).convert('RGB')
            
            # OpenCV用にnumpy配列に変換
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            features = {}
            
            # 1. スケール・クラスト特徴（AK特有）
            features.update(self._analyze_scale_features(image_rgb))
            
            # 2. 角化特徴（AK・Bowen病共通）
            features.update(self._analyze_keratinization_features(image))
            
            # 3. 表面テクスチャ（Bowen病特有）
            features.update(self._analyze_surface_texture(image_rgb))
            
            # 4. 色彩の不均一性（Bowen病特有）
            features.update(self._analyze_color_heterogeneity(image_rgb))
            
            # 5. 血管パターン
            features.update(self._analyze_vascular_patterns(image_rgb))
            
            return features
            
        except Exception as e:
            print(f"⚠️ AK・Bowen病特徴抽出エラー: {e}")
            return {}
    
    def _analyze_scale_features(self, image_rgb):
        """スケール・クラスト特徴の解析（AK特有）"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 高周波成分の検出（スケール・クラスト）
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        scale_intensity = np.mean(high_freq[high_freq > 0])
        
        # スケールの分布パターン
        # ガボールフィルターによるテクスチャ解析
        angles = [0, 45, 90, 135]
        gabor_responses = []
        
        for angle in angles:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(np.mean(filtered))
        
        scale_directionality = np.std(gabor_responses)
        
        # 表面の粗さ（スケール特徴）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        surface_roughness = np.var(laplacian)
        
        return {
            'scale_intensity': scale_intensity,
            'scale_directionality': scale_directionality,
            'surface_roughness': surface_roughness
        }
    
    def _analyze_keratinization_features(self, image):
        """角化特徴の解析"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 角化部位の検出（高輝度領域）
        keratinized_mask = gray > np.percentile(gray, 80)
        keratinization_ratio = np.sum(keratinized_mask) / gray.size
        
        # 角化の不均一性
        if np.sum(keratinized_mask) > 0:
            keratinized_regions = gray[keratinized_mask]
            keratinization_variance = np.var(keratinized_regions)
        else:
            keratinization_variance = 0
        
        # 表面の層状構造検出
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        layer_patterns = np.mean(np.abs(sobel_y))
        
        return {
            'keratinization_ratio': keratinization_ratio,
            'keratinization_variance': keratinization_variance,
            'layer_patterns': layer_patterns
        }
    
    def _analyze_surface_texture(self, image_rgb):
        """表面テクスチャの解析（Bowen病特有）"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # LBP (Local Binary Pattern) - より詳細な実装
        def enhanced_lbp(image, radius=2, neighbors=16):
            lbp = np.zeros_like(image, dtype=np.uint8)
            height, width = image.shape
            
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center = image[i, j]
                    binary_val = 0
                    
                    for k in range(neighbors):
                        angle = 2 * np.pi * k / neighbors
                        x = int(round(i + radius * np.cos(angle)))
                        y = int(round(j + radius * np.sin(angle)))
                        
                        x = max(0, min(x, height - 1))
                        y = max(0, min(y, width - 1))
                        
                        if image[x, y] >= center:
                            binary_val |= (1 << k)
                    
                    lbp[i, j] = binary_val
            return lbp
        
        lbp = enhanced_lbp(gray)
        lbp_uniformity = len(np.unique(lbp))
        
        # 表面の不規則性（Bowen病特徴）
        # フラクタル次元の近似
        def box_counting_dimension(image, scales=None):
            if scales is None:
                scales = [2, 3, 4, 5, 8, 10, 15, 20]
            
            counts = []
            for scale in scales:
                # 画像をスケールサイズのボックスに分割
                h, w = image.shape
                h_boxes = h // scale
                w_boxes = w // scale
                
                count = 0
                for i in range(h_boxes):
                    for j in range(w_boxes):
                        box = image[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
                        if np.std(box) > 10:  # 変化があるボックス
                            count += 1
                counts.append(count)
            
            # フラクタル次元の計算
            scales = np.array(scales)
            counts = np.array(counts)
            if len(counts) > 1 and np.sum(counts) > 0:
                coeffs = np.polyfit(np.log(scales), np.log(counts + 1), 1)
                return -coeffs[0]
            return 0
        
        fractal_dimension = box_counting_dimension(gray)
        
        return {
            'lbp_uniformity': lbp_uniformity,
            'fractal_dimension': fractal_dimension,
            'texture_complexity': np.std(lbp)
        }
    
    def _analyze_color_heterogeneity(self, image_rgb):
        """色彩の不均一性解析（Bowen病特有）"""
        # HSV変換
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # 色相の分散（不均一性）
        hue_variance = np.var(hsv[:, :, 0])
        saturation_variance = np.var(hsv[:, :, 1])
        
        # 色彩クラスタリング
        pixels = image_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # クラスター間の距離（色彩多様性）
        centers = kmeans.cluster_centers_
        color_diversity = 0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                color_diversity += np.linalg.norm(centers[i] - centers[j])
        color_diversity /= (len(centers) * (len(centers) - 1) / 2)
        
        # 紅斑様変化の検出（Bowen病特徴）
        red_channel = image_rgb[:, :, 0]
        erythema_ratio = np.sum(red_channel > np.percentile(red_channel, 70)) / red_channel.size
        
        return {
            'hue_variance': hue_variance,
            'saturation_variance': saturation_variance,
            'color_diversity': color_diversity,
            'erythema_ratio': erythema_ratio
        }
    
    def _analyze_vascular_patterns(self, image_rgb):
        """血管パターン解析"""
        # 赤色チャンネルの強調
        red_enhanced = image_rgb[:, :, 0] - (image_rgb[:, :, 1] + image_rgb[:, :, 2]) / 2
        red_enhanced = np.clip(red_enhanced, 0, 255)
        
        # 血管様構造の検出
        kernel = np.ones((3, 3), np.uint8)
        tophat = cv2.morphologyEx(red_enhanced.astype(np.uint8), cv2.MORPH_TOPHAT, kernel)
        vascular_density = np.sum(tophat > 20) / tophat.size
        
        # 線状構造の検出
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
        line_count = len(lines) if lines is not None else 0
        
        return {
            'vascular_density': vascular_density,
            'linear_structures': line_count
        }
    
    def calculate_ak_bowen_score(self, features, disease_type='unknown'):
        """AK・Bowen病尤度スコアの計算"""
        ak_score = 0.0
        bowen_score = 0.0
        confidence_factors = []
        
        # AK特徴スコア
        if 'scale_intensity' in features:
            # AKはスケール・クラストが特徴的
            ak_scale_score = min(features['scale_intensity'] / 50, 1.0)
            ak_score += ak_scale_score * 0.4
            confidence_factors.append(f"AK スケール特徴: {ak_scale_score:.2f}")
        
        if 'keratinization_ratio' in features:
            # AKは角化が顕著
            ak_keratinization_score = min(features['keratinization_ratio'] * 3, 1.0)
            ak_score += ak_keratinization_score * 0.3
            confidence_factors.append(f"AK 角化: {ak_keratinization_score:.2f}")
        
        if 'surface_roughness' in features:
            # AKは表面が粗い
            roughness_score = min(features['surface_roughness'] / 1000, 1.0)
            ak_score += roughness_score * 0.3
            confidence_factors.append(f"AK 表面粗さ: {roughness_score:.2f}")
        
        # Bowen病特徴スコア
        if 'color_diversity' in features:
            # Bowen病は色彩多様性が高い
            bowen_color_score = min(features['color_diversity'] / 100, 1.0)
            bowen_score += bowen_color_score * 0.4
            confidence_factors.append(f"Bowen 色彩多様性: {bowen_color_score:.2f}")
        
        if 'fractal_dimension' in features:
            # Bowen病は複雑な表面テクスチャ
            texture_score = min(features['fractal_dimension'] / 2, 1.0)
            bowen_score += texture_score * 0.3
            confidence_factors.append(f"Bowen テクスチャ: {texture_score:.2f}")
        
        if 'erythema_ratio' in features:
            # Bowen病は紅斑様変化
            erythema_score = min(features['erythema_ratio'] * 2, 1.0)
            bowen_score += erythema_score * 0.3
            confidence_factors.append(f"Bowen 紅斑: {erythema_score:.2f}")
        
        # 疾患タイプに応じてスコア調整
        if disease_type == 'AK':
            final_score = ak_score
            disease_confidence = "AK特徴重視"
        elif disease_type == 'Bowen病':
            final_score = bowen_score  
            disease_confidence = "Bowen病特徴重視"
        else:
            # 両方を考慮
            final_score = max(ak_score, bowen_score)
            disease_confidence = f"AK: {ak_score:.2f}, Bowen: {bowen_score:.2f}"
        
        confidence_factors.append(disease_confidence)
        
        return final_score, confidence_factors, {'ak_score': ak_score, 'bowen_score': bowen_score}
    
    def predict_with_ak_bowen_analysis(self, image_path, disease_type='unknown'):
        """AK・Bowen病分析を含む予測"""
        # AK・Bowen病特徴分析（必須）
        features = self.extract_ak_bowen_features(image_path)
        final_score, confidence_factors, detailed_scores = self.calculate_ak_bowen_score(features, disease_type)
        
        # 基本予測（モデルが利用可能な場合のみ）
        base_benign_prob = 0.5
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
        
        # AK・Bowen病補正の適用
        ak_bowen_threshold = 0.4
        
        if final_score > ak_bowen_threshold:
            if model_prediction_available:
                # モデル予測がある場合：悪性側に補正
                correction_factor = (final_score - ak_bowen_threshold) / (1 - ak_bowen_threshold)
                correction_strength = 0.6 if final_score > 0.7 else 0.4
                corrected_malignant_prob = base_malignant_prob + (1 - base_malignant_prob) * correction_factor * correction_strength
                corrected_benign_prob = 1 - corrected_malignant_prob
            else:
                # モデル予測がない場合：AK・Bowen病特徴のみで判定
                ak_bowen_strength = (final_score - ak_bowen_threshold) / (1 - ak_bowen_threshold)
                corrected_malignant_prob = 0.55 + ak_bowen_strength * 0.35  # 0.55-0.9の範囲
                corrected_benign_prob = 1 - corrected_malignant_prob
            
            applied_correction = True
        else:
            corrected_benign_prob = base_benign_prob
            corrected_malignant_prob = base_malignant_prob
            applied_correction = False
        
        # 最終判定
        predicted_class = 1 if corrected_malignant_prob > corrected_benign_prob else 0
        confidence = max(corrected_benign_prob, corrected_malignant_prob)
        
        return {
            'predicted_class': predicted_class,
            'predicted_type': 'malignant' if predicted_class == 1 else 'benign',
            'confidence': confidence,
            'benign_probability': corrected_benign_prob,
            'malignant_probability': corrected_malignant_prob,
            'base_benign_probability': base_benign_prob,
            'base_malignant_probability': base_malignant_prob,
            'ak_bowen_score': final_score,
            'detailed_scores': detailed_scores,
            'features': features,
            'confidence_factors': confidence_factors,
            'correction_applied': applied_correction,
            'ak_bowen_threshold': ak_bowen_threshold,
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
    print("🔬 AK・Bowen病特化分類システム")
    print("   光線性角化症・ボーエン病対応版")
    print("=" * 60)
    
    # モデル読み込み
    model_path = '/Users/iinuma/Desktop/ダーモ/disease_classification_model.pth'
    classifier = AKBowenClassifier(model_path)
    
    # 診断実行例
    test_cases = [
        ('/Users/iinuma/Desktop/ダーモ/AK/CIMG8780.JPG', 'AK'),
        ('/Users/iinuma/Desktop/ダーモ/Bowen病/CIMG9291.JPG', 'Bowen病')
    ]
    
    for image_path, disease_type in test_cases:
        if os.path.exists(image_path):
            print(f"\n📂 診断対象: {os.path.basename(image_path)} ({disease_type})")
            result = classifier.predict_with_ak_bowen_analysis(image_path, disease_type)
            
            print(f"🎯 最終判定: {result['predicted_type']}")
            print(f"📊 AK・Bowen病スコア: {result['ak_bowen_score']:.3f}")
            print(f"📈 詳細スコア: AK={result['detailed_scores']['ak_score']:.3f}, Bowen={result['detailed_scores']['bowen_score']:.3f}")

if __name__ == "__main__":
    main()