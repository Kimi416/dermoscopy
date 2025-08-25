"""
HAM10000から母斑（nevus）を抽出してnevusフォルダを作成
ユーザーの追加画像も統合可能な仕様
"""

import pandas as pd
import os
import shutil
import glob
from PIL import Image

def extract_nevus_from_ham10000(
    ham_metadata_path='/Users/iinuma/Desktop/ダーモ/ham10000_data/HAM10000_metadata.csv',
    ham_images_dir='/Users/iinuma/Desktop/ダーモ/ham10000_data',
    output_nevus_dir='/Users/iinuma/Desktop/ダーモ/nevus',
    user_nevus_dir=None,  # ユーザーの母斑画像フォルダ（オプション）
    max_nevus_samples=1000  # 抽出する母斑の最大数
):
    """HAM10000から母斑を抽出し、ユーザー画像も統合"""
    
    print("🔍 HAM10000から母斑（nevus）抽出開始")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    os.makedirs(output_nevus_dir, exist_ok=True)
    
    # メタデータ読み込み
    print("📊 メタデータ読み込み中...")
    try:
        df = pd.read_csv(ham_metadata_path)
        print(f"   総レコード数: {len(df):,}")
    except Exception as e:
        print(f"❌ メタデータ読み込みエラー: {e}")
        return
    
    # 母斑（nv）フィルタリング
    nevus_df = df[df['dx'] == 'nv'].copy()
    print(f"   母斑（nv）レコード数: {len(nevus_df):,}")
    
    # サンプリング（指定数まで削減）
    if len(nevus_df) > max_nevus_samples:
        # 年齢・性別・部位でバランスよくサンプリング
        nevus_df = nevus_df.groupby(['sex', 'localization']).apply(
            lambda x: x.sample(min(len(x), max_nevus_samples // 10), random_state=42)
        ).reset_index(drop=True)
        nevus_df = nevus_df.sample(n=min(len(nevus_df), max_nevus_samples), random_state=42)
        print(f"   サンプリング後: {len(nevus_df)}枚")
    
    # HAM10000画像ファイルの検索
    print(f"\\n📂 HAM10000画像ファイル検索中...")
    ham_image_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        ham_image_files.extend(glob.glob(os.path.join(ham_images_dir, ext)))
    
    # image_idから実際のファイル名マッピングを作成
    image_id_to_file = {}
    for file_path in ham_image_files:
        filename = os.path.basename(file_path)
        # ISIC_xxxxxxx.jpg -> ISIC_xxxxxxx
        image_id = filename.split('.')[0]
        image_id_to_file[image_id] = file_path
    
    print(f"   HAM10000画像ファイル数: {len(image_id_to_file):,}")
    
    # 母斑画像をコピー
    print(f"\\n📋 母斑画像コピー中...")
    copied_count = 0
    error_count = 0
    
    for idx, row in nevus_df.iterrows():
        image_id = row['image_id']
        lesion_id = row['lesion_id']
        age = row.get('age', 'unknown')
        sex = row.get('sex', 'unknown')
        location = row.get('localization', 'unknown')
        
        if image_id in image_id_to_file:
            source_path = image_id_to_file[image_id]
            
            # 出力ファイル名（情報を含む）
            output_filename = f"ham_nevus_{copied_count:04d}_{image_id}_{sex}_{age}y_{location}.jpg"
            output_path = os.path.join(output_nevus_dir, output_filename)
            
            try:
                # 画像をコピー（リサイズして統一）
                img = Image.open(source_path).convert('RGB')
                # 高解像度で保存（320x320）
                img = img.resize((320, 320), Image.Resampling.LANCZOS)
                img.save(output_path, 'JPEG', quality=95)
                copied_count += 1
                
                if copied_count % 100 == 0:
                    print(f"   進捗: {copied_count}/{len(nevus_df)}")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # 最初の5個のエラーのみ表示
                    print(f"   ⚠️ コピーエラー {image_id}: {e}")
        else:
            error_count += 1
    
    print(f"\\n✅ HAM10000母斑抽出完了:")
    print(f"   成功: {copied_count}枚")
    print(f"   エラー: {error_count}枚")
    
    # ユーザー画像の統合（オプション）
    user_copied = 0
    if user_nevus_dir and os.path.exists(user_nevus_dir):
        print(f"\\n👤 ユーザー母斑画像統合中...")
        print(f"   ソース: {user_nevus_dir}")
        
        user_image_patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
        user_images = []
        for pattern in user_image_patterns:
            user_images.extend(glob.glob(os.path.join(user_nevus_dir, pattern)))
        
        print(f"   ユーザー画像数: {len(user_images)}枚")
        
        for user_img_path in user_images:
            try:
                filename = os.path.basename(user_img_path)
                name_base = filename.split('.')[0]
                
                # ユーザー画像のファイル名
                output_filename = f"user_nevus_{user_copied:04d}_{name_base}.jpg"
                output_path = os.path.join(output_nevus_dir, output_filename)
                
                # 画像をコピー（同じく320x320にリサイズ）
                img = Image.open(user_img_path).convert('RGB')
                img = img.resize((320, 320), Image.Resampling.LANCZOS)
                img.save(output_path, 'JPEG', quality=95)
                user_copied += 1
                
            except Exception as e:
                print(f"   ⚠️ ユーザー画像エラー {filename}: {e}")
        
        print(f"   ユーザー画像統合完了: {user_copied}枚")
    
    # 統計情報
    total_nevus = copied_count + user_copied
    print(f"\\n📊 最終統計:")
    print(f"   HAM10000母斑: {copied_count}枚")
    print(f"   ユーザー母斑: {user_copied}枚")
    print(f"   総母斑数: {total_nevus}枚")
    print(f"   保存先: {output_nevus_dir}")
    
    # 品質チェック
    print(f"\\n🔍 品質チェック:")
    saved_files = glob.glob(os.path.join(output_nevus_dir, "*.jpg"))
    print(f"   実際保存ファイル数: {len(saved_files)}枚")
    
    if len(saved_files) > 0:
        # サンプル画像の情報表示
        sample_img = Image.open(saved_files[0])
        print(f"   サンプル画像サイズ: {sample_img.size}")
        print(f"   サンプルファイル: {os.path.basename(saved_files[0])}")
    
    return total_nevus

def check_mm_data(mm_dir='/Users/iinuma/Desktop/ダーモ/MM'):
    """既存のメラノーマデータ確認"""
    print(f"\\n🔬 既存メラノーマデータ確認:")
    
    if not os.path.exists(mm_dir):
        print(f"   ❌ メラノーマフォルダが見つかりません: {mm_dir}")
        return 0
    
    mm_patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    mm_images = []
    for pattern in mm_patterns:
        mm_images.extend(glob.glob(os.path.join(mm_dir, pattern)))
    
    print(f"   メラノーマ画像数: {len(mm_images)}枚")
    print(f"   フォルダ: {mm_dir}")
    
    if len(mm_images) > 0:
        sample_img = Image.open(mm_images[0])
        print(f"   サンプル画像サイズ: {sample_img.size}")
    
    return len(mm_images)

def main():
    """メイン実行"""
    print("🚀 HAM10000母斑抽出・統合システム")
    print("   Nevus vs Melanoma分類用データ準備")
    print("=" * 80)
    
    # 既存のメラノーマデータ確認
    mm_count = check_mm_data()
    
    # HAM10000から母斑抽出
    nevus_count = extract_nevus_from_ham10000(
        max_nevus_samples=1000,  # バランスのため1000枚に制限
        user_nevus_dir=None  # ユーザー画像パスを指定する場合はここを変更
    )
    
    print(f"\\n🎯 データ準備完了:")
    print(f"   Nevus (母斑): {nevus_count}枚")
    print(f"   Melanoma (メラノーマ): {mm_count}枚")
    
    if nevus_count > 0 and mm_count > 0:
        ratio = nevus_count / mm_count
        print(f"   データ比率: {ratio:.1f}:1 (Nevus:Melanoma)")
        
        if 0.5 <= ratio <= 3.0:
            print("   ✅ バランスの良いデータセットです")
        else:
            print("   ⚠️ データ不均衡があります（調整推奨）")
    
    # 使用方法の案内
    print(f"\\n💡 次のステップ:")
    print("1. 今後ユーザー様の母斑画像を追加する場合:")
    print("   - 専用フォルダに画像を配置")
    print("   - user_nevus_dir='/path/to/user/nevus' を指定して再実行")
    print("2. nevus_mm_classifier.py でトレーニング開始")
    print("3. 既存アンサンブルシステムとの統合")

if __name__ == "__main__":
    main()