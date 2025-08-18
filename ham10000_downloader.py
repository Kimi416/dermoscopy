"""
HAM10000データセットダウンローダー
Human Against Machine with 10,000 training images

HAM10000は皮膚科医が検証済みの高品質データセット：
- 総数: 10,015枚
- 7クラス分類
- 皮膚科専門医による診断確認済み
"""

import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import time

class HAM10000Downloader:
    """HAM10000データセットのダウンローダー"""
    
    def __init__(self, output_dir="ham10000_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # HAM10000の7クラス分類
        self.classes = {
            'akiec': 'Actinic keratoses',           # 日光角化症（悪性前駆病変）
            'bcc': 'Basal cell carcinoma',          # 基底細胞癌（悪性）
            'bkl': 'Benign keratosis-like lesions', # 良性角化症様病変（良性）
            'df': 'Dermatofibroma',                 # 皮膚線維腫（良性）
            'mel': 'Melanoma',                      # メラノーマ（悪性）
            'nv': 'Melanocytic nevi',               # 色素性母斑（良性）
            'vasc': 'Vascular lesions'              # 血管病変（良性）
        }
        
        # 良性・悪性分類
        self.benign_classes = ['bkl', 'df', 'nv', 'vasc']
        self.malignant_classes = ['akiec', 'bcc', 'mel']
        
        print("🔬 HAM10000データセット初期化")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        print(f"📊 クラス数: {len(self.classes)}")
        print(f"✅ 良性クラス: {len(self.benign_classes)}個")
        print(f"❌ 悪性クラス: {len(self.malignant_classes)}個")
    
    def download_dataset(self):
        """HAM10000データセットをダウンロード"""
        
        print("\\n📥 HAM10000データセットダウンロード開始...")
        
        # Kaggle HAM10000のURL（要認証）
        urls = {
            'metadata': 'https://dataverse.harvard.edu/api/access/datafile/3450625',
            'images_part1': 'https://dataverse.harvard.edu/api/access/datafile/3450626',
            'images_part2': 'https://dataverse.harvard.edu/api/access/datafile/3450627'
        }
        
        print("⚠️ HAM10000は認証が必要なデータセットです。")
        print("📋 手動ダウンロード手順:")
        print("1. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("2. HAM10000_metadata.csv をダウンロード")
        print("3. HAM10000_images_part_1.zip をダウンロード") 
        print("4. HAM10000_images_part_2.zip をダウンロード")
        print(f"5. ファイルを {self.output_dir} に配置")
        
        return False
    
    def setup_demo_structure(self):
        """デモ用のディレクトリ構造作成"""
        
        print("\\n🏗️ HAM10000ディレクトリ構造作成...")
        
        # 7クラス用ディレクトリ
        for class_code, class_name in self.classes.items():
            class_dir = self.output_dir / class_code
            class_dir.mkdir(exist_ok=True)
            print(f"📁 {class_code}: {class_name}")
        
        # 良性・悪性用ディレクトリ  
        (self.output_dir / "benign").mkdir(exist_ok=True)
        (self.output_dir / "malignant").mkdir(exist_ok=True)
        
        print(f"✅ ディレクトリ構造作成完了: {self.output_dir}")
        
        return True
    
    def create_binary_classification_map(self):
        """7クラス→2クラス（良性・悪性）マッピング作成"""
        
        classification_map = {}
        
        # 良性クラス
        for class_code in self.benign_classes:
            classification_map[class_code] = {
                'binary_label': 0,  # 良性
                'category': 'benign',
                'description': self.classes[class_code]
            }
        
        # 悪性クラス
        for class_code in self.malignant_classes:
            classification_map[class_code] = {
                'binary_label': 1,  # 悪性 
                'category': 'malignant',
                'description': self.classes[class_code]
            }
        
        # マッピング保存
        import json
        map_file = self.output_dir / "binary_classification_map.json"
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(classification_map, f, indent=2, ensure_ascii=False)
        
        print(f"\\n📋 2クラス分類マップ作成: {map_file}")
        print("\\n🏷️ クラス分類:")
        print("良性 (benign = 0):")
        for code in self.benign_classes:
            print(f"  • {code}: {self.classes[code]}")
        print("\\n悪性 (malignant = 1):")
        for code in self.malignant_classes:
            print(f"  • {code}: {self.classes[code]}")
        
        return classification_map
    
    def analyze_dataset_balance(self):
        """データセットのバランス分析（理論値）"""
        
        # HAM10000の理論的な分布
        theoretical_distribution = {
            'nv': 6705,      # 色素性母斑（良性）
            'mel': 1113,     # メラノーマ（悪性）
            'bkl': 1099,     # 良性角化症（良性）
            'bcc': 514,      # 基底細胞癌（悪性）
            'akiec': 327,    # 日光角化症（悪性前駆）
            'vasc': 142,     # 血管病変（良性）
            'df': 115        # 皮膚線維腫（良性）
        }
        
        benign_count = sum(theoretical_distribution[code] for code in self.benign_classes)
        malignant_count = sum(theoretical_distribution[code] for code in self.malignant_classes)
        total_count = benign_count + malignant_count
        
        print(f"\\n📊 HAM10000データセット分布（理論値）:")
        print(f"総数: {total_count:,}枚")
        print(f"\\n良性: {benign_count:,}枚 ({benign_count/total_count:.1%})")
        for code in self.benign_classes:
            count = theoretical_distribution[code]
            print(f"  • {code}: {count:,}枚 ({count/total_count:.1%}) - {self.classes[code]}")
        
        print(f"\\n悪性: {malignant_count:,}枚 ({malignant_count/total_count:.1%})")
        for code in self.malignant_classes:
            count = theoretical_distribution[code]
            print(f"  • {code}: {count:,}枚 ({count/total_count:.1%}) - {self.classes[code]}")
        
        print(f"\\n⚖️ 良性:悪性比率 = {benign_count/malignant_count:.1f}:1")
        
        return theoretical_distribution

def main():
    """HAM10000セットアップのメイン関数"""
    
    print("=" * 60)
    print("🔬 HAM10000データセットセットアップ")
    print("=" * 60)
    
    # ダウンローダー初期化
    downloader = HAM10000Downloader()
    
    # 1. ディレクトリ構造作成
    downloader.setup_demo_structure()
    
    # 2. 2クラス分類マップ作成
    downloader.create_binary_classification_map()
    
    # 3. データセット分布分析
    downloader.analyze_dataset_balance()
    
    # 4. ダウンロード案内
    downloader.download_dataset()
    
    print("\\n✅ HAM10000セットアップ完了!")
    print("\\n🚀 次のステップ:")
    print("1. 手動でHAM10000データをダウンロード")
    print("2. ham10000_pretrain_pipeline.py で学習開始")
    print("3. ISIC版と性能比較")

if __name__ == "__main__":
    main()