"""
HAM10000からNevusとMelanomaをバランスよく抽出
既存のユーザーMMデータも保持
"""

import pandas as pd
import os
import shutil
import glob
from PIL import Image
import numpy as np

def extract_melanoma_from_ham10000(
    ham_metadata_path='/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_metadata.csv',
    ham_images_dir='/Users/iinuma/Desktop/ダーモ/ham10000_data',
    user_mm_dir='/Users/iinuma/Desktop/ダーモ/MM',
    output_mm_dir='/Users/iinuma/Desktop/ダーモ/MM_combined',
    nevus_count=1000  # 母斑数に合わせる
):
    """HAM10000からメラノーマを抽出してユーザーデータと統合"""
    
    print("🔬 HAM10000からメラノーマ抽出・統合")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    os.makedirs(output_mm_dir, exist_ok=True)
    
    # メタデータ読み込み
    print("📊 メタデータ読み込み中...")
    df = pd.read_csv(ham_metadata_path)
    
    # メラノーマ（mel）フィルタリング
    melanoma_df = df[df['dx'] == 'mel'].copy()
    print(f"   HAM10000メラノーマレコード数: {len(melanoma_df):,}")
    
    # 既存ユーザーMM数確認
    user_mm_files = []
    if os.path.exists(user_mm_dir):
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
            user_mm_files.extend(glob.glob(os.path.join(user_mm_dir, ext)))
    
    print(f"   既存ユーザーMM: {len(user_mm_files)}枚")
    
    # 必要なHAM10000メラノーマ数を計算
    ham_mm_needed = max(0, nevus_count - len(user_mm_files))
    print(f"   必要HAM10000 MM: {ham_mm_needed}枚")
    
    # HAM10000画像ファイルマッピング
    print("📂 HAM10000画像ファイル検索中...")
    ham_image_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        ham_image_files.extend(glob.glob(os.path.join(ham_images_dir, ext)))
    
    image_id_to_file = {}
    for file_path in ham_image_files:
        filename = os.path.basename(file_path)
        image_id = filename.split('.')[0]
        image_id_to_file[image_id] = file_path
    
    # 既存ユーザーMMコピー
    print("\\n👤 既存ユーザーMM統合中...")
    user_copied = 0
    for user_mm_path in user_mm_files:
        try:
            filename = os.path.basename(user_mm_path)
            name_base = filename.split('.')[0]
            
            output_filename = f"user_mm_{user_copied:04d}_{name_base}.jpg"
            output_path = os.path.join(output_mm_dir, output_filename)
            
            img = Image.open(user_mm_path).convert('RGB')
            img = img.resize((320, 320), Image.Resampling.LANCZOS)
            img.save(output_path, 'JPEG', quality=95)
            user_copied += 1
            
        except Exception as e:
            print(f"   ⚠️ ユーザーMM エラー {filename}: {e}")
    
    print(f"   ユーザーMM統合完了: {user_copied}枚")
    
    # HAM10000メラノーマをサンプリング・コピー
    if ham_mm_needed > 0:
        print("\\n📋 HAM10000メラノーマコピー中...")
        
        # バランスよくサンプリング
        if len(melanoma_df) > ham_mm_needed:
            melanoma_df = melanoma_df.groupby(['sex', 'localization']).apply(
                lambda x: x.sample(min(len(x), ham_mm_needed // 8), random_state=42)
            ).reset_index(drop=True)
            melanoma_df = melanoma_df.sample(n=min(len(melanoma_df), ham_mm_needed), random_state=42)
        
        ham_copied = 0
        for idx, row in melanoma_df.iterrows():
            if ham_copied >= ham_mm_needed:
                break
                
            image_id = row['image_id']
            age = row.get('age', 'unknown')
            sex = row.get('sex', 'unknown')
            location = row.get('localization', 'unknown')
            
            if image_id in image_id_to_file:
                source_path = image_id_to_file[image_id]
                output_filename = f"ham_mm_{ham_copied:04d}_{image_id}_{sex}_{age}y_{location}.jpg"
                output_path = os.path.join(output_mm_dir, output_filename)
                
                try:
                    img = Image.open(source_path).convert('RGB')
                    img = img.resize((320, 320), Image.Resampling.LANCZOS)
                    img.save(output_path, 'JPEG', quality=95)
                    ham_copied += 1
                    
                    if ham_copied % 50 == 0:
                        print(f"   進捗: {ham_copied}/{ham_mm_needed}")
                        
                except Exception as e:
                    print(f"   ⚠️ HAM MM エラー {image_id}: {e}")
        
        print(f"   HAM10000 MM統合完了: {ham_copied}枚")
    else:
        ham_copied = 0
    
    total_mm = user_copied + ham_copied
    print(f"\\n✅ メラノーマデータ統合完了:")
    print(f"   ユーザーMM: {user_copied}枚")
    print(f"   HAM10000 MM: {ham_copied}枚")
    print(f"   総MM数: {total_mm}枚")
    
    return total_mm

def main():
    """データバランス調整実行"""
    print("⚖️ Nevus vs Melanoma データバランス調整")
    print("=" * 80)
    
    # 現在の母斑数確認
    nevus_dir = '/Users/iinuma/Desktop/ダーモ/nevus'
    nevus_files = glob.glob(os.path.join(nevus_dir, "*.jpg"))
    nevus_count = len(nevus_files)
    
    print(f"現在の母斑数: {nevus_count}枚")
    
    # メラノーマデータ調整
    mm_count = extract_melanoma_from_ham10000(nevus_count=nevus_count)
    
    print(f"\\n🎯 最終データバランス:")
    print(f"   Nevus (母斑): {nevus_count}枚")
    print(f"   Melanoma (メラノーマ): {mm_count}枚")
    
    if mm_count > 0:
        ratio = nevus_count / mm_count
        print(f"   データ比率: {ratio:.1f}:1")
        
        if 0.8 <= ratio <= 1.5:
            print("   ✅ バランスの良いデータセットです")
        else:
            print("   ⚠️ 若干の不均衡がありますが実用的です")
    
    print(f"\\n💡 準備完了:")
    print("   nevus_mm_classifier.py でトレーニング開始可能です")

if __name__ == "__main__":
    main()