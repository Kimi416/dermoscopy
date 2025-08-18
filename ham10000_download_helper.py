"""
HAM10000ダウンロード支援ツール
ダウンロード状況確認・ファイル検証・自動展開
"""

import os
import zipfile
import hashlib
from pathlib import Path

class HAM10000DownloadHelper:
    """HAM10000ダウンロード支援クラス"""
    
    def __init__(self, data_dir="/Users/iinuma/Desktop/ダーモ/ham10000_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 必要ファイルの定義
        self.required_files = {
            'metadata': {
                'filename': 'HAM10000_metadata.csv',
                'expected_size_mb': 1,
                'description': 'メタデータ（診断情報）'
            },
            'images_part1': {
                'filename': 'HAM10000_images_part_1.zip',
                'expected_size_mb': 2500,
                'description': '画像データ Part 1'
            },
            'images_part2': {
                'filename': 'HAM10000_images_part_2.zip', 
                'expected_size_mb': 2500,
                'description': '画像データ Part 2'
            }
        }
    
    def check_download_status(self):
        """ダウンロード状況をチェック"""
        
        print("=" * 60)
        print("🔍 HAM10000ダウンロード状況確認")
        print("=" * 60)
        
        status = {}
        total_files = len(self.required_files)
        downloaded_files = 0
        
        for key, file_info in self.required_files.items():
            file_path = self.data_dir / file_info['filename']
            
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                expected_mb = file_info['expected_size_mb']
                
                if file_size_mb >= expected_mb * 0.95:  # 95%以上なら完了とみなす
                    status[key] = {
                        'status': '✅ 完了',
                        'size_mb': file_size_mb,
                        'path': str(file_path)
                    }
                    downloaded_files += 1
                else:
                    status[key] = {
                        'status': '⚠️ 不完全',
                        'size_mb': file_size_mb,
                        'expected_mb': expected_mb,
                        'path': str(file_path)
                    }
            else:
                status[key] = {
                    'status': '❌ 未ダウンロード',
                    'expected_mb': file_info['expected_size_mb'],
                    'path': str(file_path)
                }
        
        # 状況表示
        for key, file_info in self.required_files.items():
            file_status = status[key]
            print(f"\\n📁 {file_info['description']}:")
            print(f"   ファイル名: {file_info['filename']}")
            print(f"   状況: {file_status['status']}")
            
            if 'size_mb' in file_status:
                print(f"   サイズ: {file_status['size_mb']:.1f} MB")
                if 'expected_mb' in file_status:
                    print(f"   期待サイズ: {file_status['expected_mb']} MB")
            
            print(f"   パス: {file_status['path']}")
        
        print(f"\\n📊 進行状況: {downloaded_files}/{total_files} ファイル完了")
        print(f"完了率: {downloaded_files/total_files:.1%}")
        
        return status, downloaded_files == total_files
    
    def extract_zip_files(self):
        """ZIPファイルを展開"""
        
        print("\\n📦 ZIPファイル展開中...")
        
        zip_files = ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']
        
        for zip_filename in zip_files:
            zip_path = self.data_dir / zip_filename
            extract_dir = self.data_dir / zip_filename.replace('.zip', '')
            
            if not zip_path.exists():
                print(f"⚠️ {zip_filename} が見つかりません")
                continue
            
            if extract_dir.exists() and any(extract_dir.iterdir()):
                print(f"✅ {zip_filename} は既に展開済み")
                continue
            
            print(f"📦 {zip_filename} を展開中...")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print(f"✅ {zip_filename} 展開完了")
                
                # 展開後のファイル数確認
                if extract_dir.exists():
                    file_count = len([f for f in extract_dir.iterdir() if f.is_file()])
                    print(f"   展開されたファイル数: {file_count}")
                
            except Exception as e:
                print(f"❌ {zip_filename} 展開エラー: {e}")
    
    def verify_dataset_integrity(self):
        """データセットの整合性確認"""
        
        print("\\n🔍 データセット整合性確認...")
        
        # メタデータ確認
        metadata_path = self.data_dir / 'HAM10000_metadata.csv'
        if metadata_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(metadata_path)
                print(f"✅ メタデータ: {len(df)} 件のレコード")
                print(f"   列: {', '.join(df.columns.tolist())}")
                
                # 診断分布確認
                if 'dx' in df.columns:
                    diagnosis_counts = df['dx'].value_counts()
                    print("\\n📊 診断分布:")
                    for dx, count in diagnosis_counts.items():
                        print(f"   {dx}: {count}件")
                
            except Exception as e:
                print(f"❌ メタデータ読み込みエラー: {e}")
        
        # 画像ファイル確認
        image_dirs = [
            self.data_dir / 'HAM10000_images_part_1',
            self.data_dir / 'HAM10000_images_part_2'
        ]
        
        total_images = 0
        for img_dir in image_dirs:
            if img_dir.exists():
                jpg_files = list(img_dir.glob('*.jpg'))
                total_images += len(jpg_files)
                print(f"✅ {img_dir.name}: {len(jpg_files)} 枚の画像")
        
        print(f"\\n📊 総画像数: {total_images} 枚")
        print(f"期待値: 10,015 枚")
        
        if total_images >= 10000:
            print("✅ 画像データセット完了!")
            return True
        else:
            print("⚠️ 画像データが不完全です")
            return False
    
    def create_download_urls(self):
        """ダウンロードURLと手順を表示"""
        
        print("\\n🔗 HAM10000ダウンロードリンク")
        print("=" * 60)
        
        print("📋 公式ダウンロードページ:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        
        print("\\n📁 必要ファイル:")
        for key, file_info in self.required_files.items():
            print(f"  • {file_info['filename']} ({file_info['expected_size_mb']} MB)")
            print(f"    → {file_info['description']}")
        
        print("\\n🚀 ダウンロード後の手順:")
        print("1. ダウンロードしたファイルをすべてこのディレクトリに配置:")
        print(f"   {self.data_dir}")
        print("2. このスクリプトを再実行して整合性確認:")
        print("   python3 ham10000_download_helper.py")
        print("3. HAM10000学習パイプライン実行:")
        print("   python3 ham10000_pretrain_pipeline.py")
    
    def setup_directory_structure(self):
        """ディレクトリ構造をセットアップ"""
        
        print(f"\\n🏗️ ディレクトリ構造セットアップ...")
        
        # 必要ディレクトリ作成
        directories = [
            'HAM10000_images_part_1',
            'HAM10000_images_part_2', 
            'processed',
            'benign',
            'malignant'
        ]
        
        for dir_name in directories:
            dir_path = self.data_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"📁 {dir_name}")
        
        print("✅ ディレクトリ構造準備完了")

def main():
    """メイン関数"""
    
    print("🔬 HAM10000ダウンロード支援ツール")
    print("=" * 60)
    
    helper = HAM10000DownloadHelper()
    
    # 1. ディレクトリ構造セットアップ
    helper.setup_directory_structure()
    
    # 2. ダウンロード状況確認
    status, is_complete = helper.check_download_status()
    
    if is_complete:
        print("\\n🎉 すべてのファイルがダウンロード済みです!")
        
        # 3. ZIPファイル展開
        helper.extract_zip_files()
        
        # 4. データセット整合性確認
        if helper.verify_dataset_integrity():
            print("\\n✅ HAM10000データセット準備完了!")
            print("\\n🚀 次のステップ:")
            print("   python3 ham10000_pretrain_pipeline.py")
        else:
            print("\\n⚠️ データセットに問題があります。再ダウンロードを推奨。")
    
    else:
        print("\\n📥 まだダウンロードが必要です")
        helper.create_download_urls()
        
        print("\\n💡 ダウンロードのコツ:")
        print("• 安定したインターネット接続を使用")
        print("• 夜間など、サーバー負荷が少ない時間帯")
        print("• ファイルを1つずつダウンロード")
        print("• ダウンロード完了後は別の場所にバックアップ")

if __name__ == "__main__":
    main()