"""
ISIC API v2を使用したダウンロードスクリプト
"""

import os
import requests
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm

class ISICv2Downloader:
    """ISIC API v2を使用したダウンローダー"""
    
    def __init__(self, output_dir="isic_v2_data"):
        self.output_dir = output_dir
        self.api_base = "https://api.isic-archive.com/api/v2"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/benign", exist_ok=True)
        os.makedirs(f"{output_dir}/malignant", exist_ok=True)
    
    def download_images(self, benign_count=500, malignant_count=500):
        """良性・悪性画像をダウンロード"""
        
        print("📥 ISIC v2 APIから画像をダウンロード中...")
        
        # 良性画像（母斑など）
        benign_downloaded = self._download_by_diagnosis(
            ["Nevus", "Solar lentigo", "Seborrheic keratosis"], 
            "benign", 
            benign_count
        )
        
        # 悪性画像（メラノーマ、基底細胞癌など）
        malignant_downloaded = self._download_by_diagnosis(
            ["Melanoma", "Basal cell carcinoma", "Squamous cell carcinoma"], 
            "malignant", 
            malignant_count
        )
        
        print(f"✅ ダウンロード完了: 良性 {benign_downloaded}枚, 悪性 {malignant_downloaded}枚")
        return benign_downloaded, malignant_downloaded
    
    def _download_by_diagnosis(self, diagnoses, category, target_count):
        """診断名別にダウンロード"""
        
        downloaded = 0
        limit = 50
        cursor = None
        
        with tqdm(total=target_count, desc=f"{category}画像") as pbar:
            
            while downloaded < target_count:
                try:
                    # API リクエスト
                    params = {"limit": limit}
                    if cursor:
                        params["cursor"] = cursor
                    
                    response = requests.get(
                        f"{self.api_base}/images/",
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        print(f"API Error: {response.status_code}")
                        break
                    
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        break
                    
                    # 各画像を処理
                    for item in results:
                        if downloaded >= target_count:
                            break
                        
                        # 診断名をチェック
                        metadata = item.get("metadata", {})
                        clinical = metadata.get("clinical", {})
                        
                        # 診断名の階層をチェック
                        diagnosis_found = False
                        for diag_key in ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5"]:
                            diagnosis = clinical.get(diag_key, "")
                            if any(target_diag.lower() in diagnosis.lower() for target_diag in diagnoses):
                                diagnosis_found = True
                                break
                        
                        if not diagnosis_found:
                            continue
                        
                        # 画像をダウンロード
                        isic_id = item.get("isic_id")
                        if not isic_id:
                            continue
                        
                        img_path = f"{self.output_dir}/{category}/{isic_id}.jpg"
                        
                        # 既存ファイルをスキップ
                        if os.path.exists(img_path):
                            downloaded += 1
                            pbar.update(1)
                            continue
                        
                        # 画像URL取得
                        files = item.get("files", {})
                        full_img = files.get("full", {})
                        img_url = full_img.get("url")
                        
                        if not img_url:
                            continue
                        
                        # 画像ダウンロード
                        img_response = requests.get(img_url, stream=True, timeout=30)
                        
                        if img_response.status_code == 200:
                            try:
                                img = Image.open(BytesIO(img_response.content))
                                # リサイズして保存
                                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                                img.save(img_path, "JPEG", quality=95)
                                downloaded += 1
                                pbar.update(1)
                            except Exception as e:
                                print(f"画像保存エラー {isic_id}: {e}")
                        
                        time.sleep(0.1)  # API制限対策
                    
                    # 次のページへ
                    cursor = data.get("next")
                    if cursor:
                        # URLからcursorパラメータを抽出
                        if "cursor=" in cursor:
                            cursor = cursor.split("cursor=")[1].split("&")[0]
                    else:
                        break
                
                except Exception as e:
                    print(f"\nエラー: {e}")
                    time.sleep(5)
                    continue
        
        return downloaded

def test_download():
    """テスト用ダウンロード（少数枚）"""
    
    downloader = ISICv2Downloader()
    benign_count, malignant_count = downloader.download_images(
        benign_count=50, 
        malignant_count=50
    )
    
    print(f"\nテストダウンロード結果:")
    print(f"  良性: {benign_count}枚")
    print(f"  悪性: {malignant_count}枚")

if __name__ == "__main__":
    test_download()