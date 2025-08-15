"""
良性データのダウンロードスクリプト
ISICアーカイブから良性病変のダーモスコピー画像を取得
"""

import os
import requests
import json
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm

def download_isic_benign_images(output_dir="benign", max_images=500):
    """
    ISICアーカイブから良性画像をダウンロード
    
    Args:
        output_dir: 保存先ディレクトリ
        max_images: ダウンロードする最大画像数
    """
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    print("🌐 ISICアーカイブから良性画像をダウンロード中...")
    print("=" * 50)
    
    # ISIC API エンドポイント
    api_base = "https://isic-archive.com/api/v1"
    
    # 良性病変のクエリパラメータ
    params = {
        "limit": 50,  # 一度に取得する数
        "sort": "name",
        "sortdir": 1,
        "filter": json.dumps({
            "benign_malignant": "benign"  # 良性のみ
        })
    }
    
    downloaded_count = 0
    offset = 0
    
    try:
        while downloaded_count < max_images:
            params["offset"] = offset
            
            # メタデータ取得
            response = requests.get(f"{api_base}/image", params=params)
            
            if response.status_code != 200:
                print(f"⚠️ APIエラー: {response.status_code}")
                break
            
            images_metadata = response.json()
            
            if not images_metadata:
                print("これ以上画像がありません。")
                break
            
            # 各画像をダウンロード
            for img_meta in tqdm(images_metadata, desc=f"Batch {offset//50 + 1}"):
                if downloaded_count >= max_images:
                    break
                
                image_id = img_meta["_id"]
                
                # 画像ダウンロード
                img_response = requests.get(
                    f"{api_base}/image/{image_id}/download",
                    stream=True
                )
                
                if img_response.status_code == 200:
                    # 画像保存
                    img = Image.open(BytesIO(img_response.content))
                    img_path = os.path.join(output_dir, f"ISIC_{image_id}.jpg")
                    img.save(img_path, "JPEG")
                    downloaded_count += 1
                
                # APIレート制限対策
                time.sleep(0.1)
            
            offset += 50
            
            print(f"ダウンロード済み: {downloaded_count}/{max_images} 画像")
    
    except Exception as e:
        print(f"エラー発生: {e}")
    
    print(f"\n✅ 完了: {downloaded_count}枚の良性画像をダウンロードしました。")
    print(f"保存先: {os.path.abspath(output_dir)}")
    
    return downloaded_count

def create_dummy_benign_data(output_dir="benign", num_images=100):
    """
    テスト用のダミー良性データを作成（実際のAPIが使えない場合）
    """
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 テスト用ダミー良性データを生成中...")
    
    for i in range(num_images):
        # ランダムな肌色っぽい画像を生成
        img_array = np.random.randint(150, 220, (256, 256, 3), dtype=np.uint8)
        
        # 中央に茶色い円形の模様を追加（母斑を模倣）
        center = (128, 128)
        radius = np.random.randint(30, 80)
        
        y, x = np.ogrid[:256, :256]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # 茶色っぽい色
        brown = [np.random.randint(100, 150), 
                 np.random.randint(70, 100), 
                 np.random.randint(50, 70)]
        
        img_array[mask] = brown
        
        # ガウシアンフィルタでぼかし
        from PIL import ImageFilter
        img = Image.fromarray(img_array)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        img_path = os.path.join(output_dir, f"benign_{i:04d}.jpg")
        img.save(img_path, "JPEG")
    
    print(f"✅ {num_images}枚のダミー良性画像を生成しました。")
    print(f"保存先: {os.path.abspath(output_dir)}")

def main():
    """メイン実行関数"""
    
    print("良性ダーモスコピー画像の準備")
    print("=" * 50)
    
    choice = input("""
選択してください:
1. ISICアーカイブから実際の良性画像をダウンロード（推奨）
2. テスト用ダミーデータを生成
選択 (1/2): """).strip()
    
    if choice == "1":
        num_images = int(input("ダウンロードする画像数 (推奨: 400-500): ") or "450")
        download_isic_benign_images(max_images=num_images)
    elif choice == "2":
        num_images = int(input("生成する画像数: ") or "100")
        create_dummy_benign_data(num_images=num_images)
    else:
        print("無効な選択です。")

if __name__ == "__main__":
    main()