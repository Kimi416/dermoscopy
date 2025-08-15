"""
ISIC APIの接続テスト
"""

import requests
import json

def test_isic_api():
    """ISIC APIの動作確認"""
    
    print("ISIC API接続テスト中...")
    print("=" * 50)
    
    # 新しいISIC API (v2)
    api_v2 = "https://api.isic-archive.com/api/v2"
    
    # 旧ISIC API (v1)
    api_v1 = "https://isic-archive.com/api/v1"
    
    # 1. V1 APIテスト
    print("\n1. V1 API テスト:")
    try:
        response = requests.get(f"{api_v1}/image", params={"limit": 1}, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {len(data)} items")
            if data:
                print(f"   Sample ID: {data[0].get('_id', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. V2 APIテスト
    print("\n2. V2 API テスト:")
    try:
        response = requests.get(f"{api_v2}/images", params={"limit": 1}, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response type: {type(data)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. 代替API - ISIC Gallery
    print("\n3. ISIC Gallery API テスト:")
    gallery_api = "https://api.isic-archive.com/api/v2/images"
    try:
        headers = {
            'Accept': 'application/json',
        }
        response = requests.get(gallery_api, headers=headers, params={"limit": 1}, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. 直接ダウンロードテスト
    print("\n4. 画像直接ダウンロードテスト:")
    test_image_id = "5436e3a4bae478396759f0b9"  # 既知のID
    try:
        download_url = f"{api_v1}/image/{test_image_id}/download"
        response = requests.head(download_url, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Metadata APIテスト
    print("\n5. Metadata API テスト:")
    try:
        metadata_url = f"{api_v1}/image"
        params = {
            "limit": 10,
            "offset": 0,
            "sort": "name",
            "sortdir": 1
        }
        response = requests.get(metadata_url, params=params, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Images found: {len(data)}")
            if data:
                # 最初の画像の情報を表示
                first_image = data[0]
                print(f"   First image keys: {list(first_image.keys())[:5]}...")
                print(f"   Diagnosis: {first_image.get('meta', {}).get('clinical', {}).get('diagnosis', 'N/A')}")
                print(f"   Benign/Malignant: {first_image.get('meta', {}).get('clinical', {}).get('benign_malignant', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_isic_api()