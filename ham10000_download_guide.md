# HAM10000ダウンロード完全ガイド

## 🔗 必須ダウンロードリンク

### 公式データセットページ
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

## 📋 必要ファイル（必須3ファイル）

### 1. メタデータ（診断情報）
- **ファイル名**: `HAM10000_metadata.csv`
- **サイズ**: 約1MB
- **内容**: 10,015件の診断情報、年齢、性別、部位情報

### 2. 画像データ Part 1
- **ファイル名**: `HAM10000_images_part_1.zip`
- **サイズ**: 約2.5GB
- **内容**: 5,000枚の皮膚病変画像

### 3. 画像データ Part 2  
- **ファイル名**: `HAM10000_images_part_2.zip`
- **サイズ**: 約2.5GB
- **内容**: 5,015枚の皮膚病変画像

## 🚀 ダウンロード手順

### Step 1: アカウント作成
1. Harvard Dataverseにアクセス
2. 「Log In / Sign Up」をクリック
3. 「Create Account」を選択
4. 必要情報を入力（研究機関メール推奨）

### Step 2: データセットアクセス
1. データセットページにアクセス
2. 「Access Dataset」ボタンをクリック
3. 利用規約に同意
4. ダウンロード開始

### Step 3: ファイル保存
ダウンロードしたファイルを以下の場所に保存：

```
/Users/iinuma/Desktop/ダーモ/ham10000_data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1.zip
└── HAM10000_images_part_2.zip
```

### Step 4: 展開
```bash
cd "/Users/iinuma/Desktop/ダーモ/ham10000_data"
unzip HAM10000_images_part_1.zip
unzip HAM10000_images_part_2.zip
```

## 🔧 自動セットアップスクリプト

```bash
# ダウンロード完了後、以下のコマンドで自動セットアップ
python3 ham10000_downloader.py
python3 ham10000_pretrain_pipeline.py
```

## ⚠️ 注意事項

### ライセンス
- **研究目的のみ**: 商用利用禁止
- **データ共有禁止**: 第三者への配布不可
- **プライバシー保護**: 患者データの適切な取り扱い

### 技術要件
- **容量**: 約5GB以上の空き容量
- **回線**: 安定したインターネット接続
- **時間**: ダウンロード完了まで30分-2時間

## 🆘 トラブルシューティング

### ダウンロードできない場合
1. **アカウント認証確認**: メール認証完了確認
2. **ブラウザ変更**: Chrome, Firefox, Safari試行
3. **VPN解除**: VPN接続を一時無効化
4. **時間変更**: アクセス集中時間を避ける

### ファイル破損の場合
1. **再ダウンロード**: 完全なファイルサイズ確認
2. **チェックサム確認**: ファイル整合性検証
3. **解凍エラー**: 異なる解凍ソフト試行

## 📞 サポート

### Harvard Dataverse サポート
- Email: support@dataverse.org
- 問い合わせフォーム: https://dataverse.org/contact

### 本プロジェクトサポート
- GitHub Issues: https://github.com/Kimi416/dermoscopy/issues
- 実装サポート: Claude Code

## 🎯 ダウンロード完了後の次のステップ

1. ✅ ファイル配置確認
2. ✅ `python3 ham10000_downloader.py` 実行
3. ✅ `python3 ham10000_pretrain_pipeline.py` 実行
4. ✅ `python3 predict_image_ham10000.py` でtest.JPG評価
5. ✅ ISIC vs HAM10000 性能比較分析

## 💪 成功のコツ

- **夜間ダウンロード**: サーバー負荷が少ない時間帯
- **有線接続**: Wi-Fiより安定した接続
- **分割ダウンロード**: 1ファイルずつダウンロード
- **バックアップ**: ダウンロード完了後は別ドライブにコピー