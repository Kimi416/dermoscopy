"""
一クラス分類システム
悪性画像のみで学習し、悪性らしさで良性・悪性を判別
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import os
import glob
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import json

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 疾患分類定義（悪性のみ）
MALIGNANT_DISEASES = {
    'AK': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma', 
    'Bowen病': 'Bowen Disease',
    'MM': 'Malignant Melanoma'
}

class FeatureExtractor(nn.Module):
    """特徴抽出器（悪性画像専用）"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1' if pretrained else None)
        
        # 特徴抽出用（分類層を除去）
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        features = self.features(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        return features

class OneClassMalignancyDetector:
    """一クラス悪性検出器"""
    
    def __init__(self):
        self.feature_extractor = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.malignancy_threshold = 0.0
        self.training_stats = {}
        
    def extract_features_from_image(self, image_path):
        """単一画像から特徴抽出"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"❌ 特徴抽出エラー {image_path}: {e}")
            return None
    
    def collect_malignant_features(self, base_path='/Users/iinuma/Desktop/ダーモ'):
        """悪性画像から特徴収集"""
        print("🔍 悪性画像から特徴抽出中...")
        
        all_features = []
        image_paths = []
        disease_labels = []
        
        for disease in MALIGNANT_DISEASES.keys():
            disease_dir = os.path.join(base_path, disease)
            if not os.path.exists(disease_dir):
                print(f"⚠️ ディレクトリが見つかりません: {disease_dir}")
                continue
            
            # 画像ファイル収集
            patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
            disease_images = []
            
            for pattern in patterns:
                disease_images.extend(glob.glob(os.path.join(disease_dir, pattern)))
            
            print(f"   {disease}: {len(disease_images)}枚")
            
            # 特徴抽出
            for img_path in disease_images:
                features = self.extract_features_from_image(img_path)
                if features is not None:
                    all_features.append(features)
                    image_paths.append(img_path)
                    disease_labels.append(disease)
        
        self.training_stats = {
            'total_images': len(all_features),
            'diseases': {disease: disease_labels.count(disease) for disease in MALIGNANT_DISEASES.keys()},
            'feature_dimension': len(all_features[0]) if all_features else 0
        }
        
        print(f"✅ 特徴抽出完了: {len(all_features)}枚の悪性画像")
        return np.array(all_features), image_paths, disease_labels
    
    def train_anomaly_detector(self, features, method='isolation_forest'):
        """異常検知器の訓練"""
        print(f"🧠 異常検知器訓練中... (手法: {method})")
        
        # 特徴量の正規化
        features_scaled = self.scaler.fit_transform(features)
        
        if method == 'isolation_forest':
            # Isolation Forest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # 10%の異常を想定
                random_state=42,
                n_estimators=100
            )
        elif method == 'one_class_svm':
            # One-Class SVM
            self.anomaly_detector = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1  # 10%の異常を想定
            )
        else:
            raise ValueError(f"未対応の手法: {method}")
        
        # 訓練
        self.anomaly_detector.fit(features_scaled)
        
        # 訓練データでの性能評価
        train_scores = self.anomaly_detector.decision_function(features_scaled)
        self.malignancy_threshold = np.percentile(train_scores, 10)  # 下位10%を閾値に
        
        print(f"✅ 異常検知器訓練完了")
        print(f"   悪性らしさ閾値: {self.malignancy_threshold:.3f}")
        print(f"   訓練スコア範囲: [{np.min(train_scores):.3f}, {np.max(train_scores):.3f}]")
        
        return train_scores
    
    def predict_malignancy(self, image_path):
        """悪性らしさ予測"""
        # 特徴抽出
        features = self.extract_features_from_image(image_path)
        if features is None:
            return None
        
        # 正規化
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 悪性らしさスコア計算
        malignancy_score = self.anomaly_detector.decision_function(features_scaled)[0]
        
        # 異常検知結果（-1: 異常/良性, 1: 正常/悪性）
        anomaly_prediction = self.anomaly_detector.predict(features_scaled)[0]
        
        # 確率的解釈（0-1スケール）
        # スコアが高いほど悪性らしい
        normalized_score = (malignancy_score - self.malignancy_threshold) / \
                          (np.abs(self.malignancy_threshold) + 1e-8)
        malignancy_probability = 1 / (1 + np.exp(-normalized_score))  # シグモイド関数
        
        # 最終判定
        is_malignant = malignancy_score > self.malignancy_threshold
        confidence = malignancy_probability if is_malignant else (1 - malignancy_probability)
        
        return {
            'malignancy_score': malignancy_score,
            'malignancy_probability': malignancy_probability,
            'benign_probability': 1 - malignancy_probability,
            'predicted_type': 'malignant' if is_malignant else 'benign',
            'predicted_class': 1 if is_malignant else 0,
            'confidence': confidence,
            'anomaly_prediction': anomaly_prediction,
            'threshold': self.malignancy_threshold
        }
    
    def save_model(self, save_path):
        """モデル保存"""
        model_data = {
            'feature_extractor_state': self.feature_extractor.state_dict(),
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler,
            'malignancy_threshold': self.malignancy_threshold,
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ モデル保存完了: {save_path}")
    
    def load_model(self, load_path):
        """モデル読み込み"""
        if not os.path.exists(load_path):
            print(f"❌ モデルファイルが見つかりません: {load_path}")
            return False
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 特徴抽出器の復元
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load_state_dict(model_data['feature_extractor_state'])
        self.feature_extractor.to(device)
        self.feature_extractor.eval()
        
        # その他のコンポーネント復元
        self.anomaly_detector = model_data['anomaly_detector']
        self.scaler = model_data['scaler']
        self.malignancy_threshold = model_data['malignancy_threshold']
        self.training_stats = model_data['training_stats']
        
        print(f"✅ モデル読み込み完了: {load_path}")
        print(f"   訓練画像数: {self.training_stats['total_images']}")
        print(f"   特徴次元: {self.training_stats['feature_dimension']}")
        
        return True

def train_one_class_system():
    """一クラス分類システムの訓練"""
    print("🚀 一クラス悪性検出システム訓練開始")
    print("=" * 60)
    
    detector = OneClassMalignancyDetector()
    
    # 特徴抽出器の初期化
    detector.feature_extractor = FeatureExtractor()
    detector.feature_extractor.to(device)
    detector.feature_extractor.eval()
    
    # 悪性画像から特徴収集
    features, image_paths, disease_labels = detector.collect_malignant_features()
    
    if len(features) == 0:
        print("❌ 悪性画像が見つかりませんでした")
        return None
    
    # 異常検知器の訓練
    train_scores = detector.train_anomaly_detector(features, method='isolation_forest')
    
    # モデル保存
    model_path = '/Users/iinuma/Desktop/ダーモ/one_class_malignancy_detector.pkl'
    detector.save_model(model_path)
    
    # 訓練統計の保存
    stats_path = '/Users/iinuma/Desktop/ダーモ/one_class_training_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_stats': detector.training_stats,
            'threshold': float(detector.malignancy_threshold),
            'score_statistics': {
                'mean': float(np.mean(train_scores)),
                'std': float(np.std(train_scores)),
                'min': float(np.min(train_scores)),
                'max': float(np.max(train_scores))
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 訓練完了統計:")
    for disease, count in detector.training_stats['diseases'].items():
        print(f"   {disease}: {count}枚")
    
    return detector

def test_one_class_system():
    """一クラス分類システムのテスト"""
    print("\n🧪 一クラス分類システムテスト")
    print("=" * 60)
    
    # モデル読み込み
    detector = OneClassMalignancyDetector()
    model_path = '/Users/iinuma/Desktop/ダーモ/one_class_malignancy_detector.pkl'
    
    if not detector.load_model(model_path):
        return
    
    # テスト画像
    test_image = '/Users/iinuma/Desktop/ダーモ/images.jpeg'
    
    if not os.path.exists(test_image):
        print(f"❌ テスト画像が見つかりません: {test_image}")
        return
    
    print(f"\n📂 テスト対象: {os.path.basename(test_image)}")
    print("🔍 一クラス分類実行中...")
    
    # 予測実行
    result = detector.predict_malignancy(test_image)
    
    if result is None:
        print("❌ 予測に失敗しました")
        return
    
    # 結果表示
    print(f"\n" + "=" * 60)
    print("🎯 一クラス分類結果")
    print("=" * 60)
    
    prediction_jp = "悪性" if result['predicted_type'] == 'malignant' else "良性"
    print(f"📊 判定: {prediction_jp} ({result['predicted_type'].upper()})")
    print(f"🎯 確信度: {result['confidence']:.1%}")
    
    print(f"\n📈 詳細スコア:")
    print(f"   悪性らしさスコア: {result['malignancy_score']:.3f}")
    print(f"   悪性確率: {result['malignancy_probability']:.1%}")
    print(f"   良性確率: {result['benign_probability']:.1%}")
    print(f"   判定閾値: {result['threshold']:.3f}")
    
    # 解釈
    print(f"\n💡 結果解釈:")
    if result['malignancy_score'] > result['threshold']:
        margin = result['malignancy_score'] - result['threshold']
        print(f"   ✅ 悪性らしさが閾値を{margin:.3f}上回っています")
        print(f"   🔬 悪性の特徴パターンと類似しています")
    else:
        margin = result['threshold'] - result['malignancy_score']
        print(f"   ✅ 悪性らしさが閾値を{margin:.3f}下回っています")
        print(f"   🌿 悪性の特徴パターンとは異なります")
    
    # 従来手法との比較用データ保存
    comparison_data = {
        'one_class_result': result,
        'test_image': test_image,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('/Users/iinuma/Desktop/ダーモ/one_class_test_result.json', 'w', encoding='utf-8') as f:
        # numpy型を通常のPython型に変換
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=convert_numpy)

def main():
    """メイン実行"""
    print("🔬 一クラス悪性検出システム")
    print("   悪性画像のみで学習する良性・悪性判別")
    print("=" * 60)
    
    # 訓練実行
    detector = train_one_class_system()
    
    if detector is None:
        return
    
    # テスト実行
    test_one_class_system()

if __name__ == "__main__":
    main()