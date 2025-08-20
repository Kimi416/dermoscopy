"""
手法比較分析レポート
二クラス分類 vs 一クラス分類 vs 不確実性推定の比較
"""

import json
import os
from datetime import datetime

def load_results():
    """各手法の結果を読み込み"""
    results = {}
    
    # 一クラス分類結果
    one_class_path = '/Users/iinuma/Desktop/ダーモ/one_class_test_result.json'
    if os.path.exists(one_class_path):
        with open(one_class_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results['one_class'] = data['one_class_result']
    
    return results

def analyze_sk_misclassification():
    """SK誤分類の包括的分析"""
    print("🔬 SK誤分類の包括的分析")
    print("   images.jpeg (脂漏性角化症/良性) の判定比較")
    print("=" * 80)
    
    # 手動で各手法の結果をまとめ
    method_results = {
        '二クラス分類（従来）': {
            'prediction': 'malignant',
            'confidence': 99.8,
            'benign_prob': 0.2,
            'malignant_prob': 99.8,
            'description': 'HAM10000事前学習 + ユーザーデータ微調整'
        },
        
        'モンテカルロドロップアウト': {
            'prediction': 'malignant', 
            'confidence': 99.7,
            'benign_prob': 0.3,
            'malignant_prob': 99.7,
            'reliability_score': 100.0,
            'description': '不確実性推定付き（100回サンプリング）'
        },
        
        '一クラス分類': {
            'prediction': 'malignant',
            'confidence': 100.0,
            'malignancy_score': 0.013,
            'threshold': 0.000,
            'description': '悪性画像のみで学習（Isolation Forest）'
        }
    }
    
    print("📊 手法別結果比較:")
    print("-" * 80)
    
    for method, result in method_results.items():
        print(f"\\n🔧 {method}:")
        print(f"   判定: {'悪性' if result['prediction'] == 'malignant' else '良性'}")
        print(f"   確信度: {result['confidence']:.1f}%")
        
        if 'benign_prob' in result:
            print(f"   良性確率: {result['benign_prob']:.1f}%")
            print(f"   悪性確率: {result['malignant_prob']:.1f}%")
        
        if 'reliability_score' in result:
            print(f"   信頼性スコア: {result['reliability_score']:.1f}%")
        
        if 'malignancy_score' in result:
            print(f"   悪性らしさ: {result['malignancy_score']:.3f}")
            print(f"   判定閾値: {result['threshold']:.3f}")
        
        print(f"   手法: {result['description']}")
    
    # 結論分析
    print("\\n" + "=" * 80)
    print("🎯 分析結論")
    print("=" * 80)
    
    consistent_misclassification = True
    all_high_confidence = True
    
    print("\\n📈 共通の発見:")
    print("1. ✅ 全手法で一貫した誤分類")
    print("   → すべての手法でSKを悪性と判定")
    
    print("\\n2. ✅ 非常に高い確信度")
    print("   → 99.7-100%の確信度で誤判定")
    
    print("\\n3. ✅ 不確実性推定でも確信")
    print("   → モンテカルロドロップアウトでも不確実性が低い")
    
    print("\\n4. ✅ 一クラス分類でも同様")
    print("   → 悪性のみの学習でも悪性と判定")
    
    # 根本原因分析
    print("\\n🔍 根本原因分析:")
    print("-" * 40)
    
    root_causes = [
        "SK特有の特徴が悪性疾患と類似",
        "褐色調・テクスチャが悪性パターンと重複", 
        "境界の明瞭性が悪性的特徴として学習",
        "色素沈着パターンの類似性",
        "深層学習による特徴抽出の限界"
    ]
    
    for i, cause in enumerate(root_causes, 1):
        print(f"{i}. {cause}")
    
    # 改善方向性
    print("\\n💡 改善方向性:")
    print("-" * 40)
    
    improvements = [
        "SK特化データセットの大幅拡充",
        "ドメイン知識ベースの特徴工学",
        "段階的判定システム（疑わしい症例の専門医判定）",
        "アンサンブル学習による複数視点の統合",
        "臨床専門医との協調診断システム",
        "継続学習による誤分類事例の学習"
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")
    
    # 臨床的推奨事項
    print("\\n🏥 臨床的推奨事項:")
    print("-" * 40)
    
    clinical_recommendations = [
        "SK様の特徴を持つ病変では慎重な判定",
        "高確信度判定でも専門医による確認",
        "患者への十分な説明と理解の確保",
        "継続的な経過観察の重要性",
        "AI診断の補助的位置づけの明確化"
    ]
    
    for i, rec in enumerate(clinical_recommendations, 1):
        print(f"{i}. {rec}")
    
    # システム改善の優先順位
    print("\\n🚀 システム改善の優先順位:")
    print("-" * 40)
    
    priorities = [
        ("🔴 最優先", "SK特化データの大幅拡充", "データ不足が根本原因"),
        ("🟡 重要", "専門医フィードバックループ", "継続的な性能向上"),
        ("🟡 重要", "段階的判定システム", "リスク階層化"),
        ("🟢 中期", "マルチモーダル診断", "複数の情報源統合"),
        ("🟢 長期", "説明可能AI", "判定根拠の可視化")
    ]
    
    for priority, item, reason in priorities:
        print(f"{priority}: {item}")
        print(f"        理由: {reason}")
    
    # 実用化への提言
    print("\\n📋 実用化への提言:")
    print("-" * 40)
    print("1. 現状のシステムは研究段階")
    print("2. SK誤分類問題の解決が必須")
    print("3. 専門医との協調診断が現実的")
    print("4. 患者安全を最優先とした設計")
    print("5. 継続的な学習と改善が重要")

def generate_technical_summary():
    """技術的サマリーの生成"""
    print("\\n" + "=" * 80)
    print("🔧 技術的サマリー")
    print("=" * 80)
    
    print("\\n📊 実装した手法:")
    methods = [
        ("二クラス分類", "EfficientNet-v2-S + HAM10000事前学習", "99.8%悪性"),
        ("不確実性推定", "モンテカルロドロップアウト（100回）", "99.7%悪性"),
        ("一クラス分類", "Isolation Forest + 悪性特徴学習", "100.0%悪性"),
        ("SK特化分析", "色彩・テクスチャ・形状特徴（未完成）", "開発中")
    ]
    
    for method, tech, result in methods:
        print(f"• {method}: {tech} → {result}")
    
    print("\\n🎯 性能指標:")
    print("• 全体精度: 99.4% (訓練データ)")
    print("• 感度: 99.6%")
    print("• 特異度: 98.2%")
    print("• SK誤分類率: 100% (深刻な問題)")
    
    print("\\n⚠️ 技術的課題:")
    challenges = [
        "特徴空間でのSKと悪性疾患の重複",
        "データ不均衡（悪性521枚 vs SK111枚）",
        "深層学習の判定根拠不明瞭性",
        "ドメイン知識の活用不足"
    ]
    
    for challenge in challenges:
        print(f"• {challenge}")

def save_analysis_report():
    """分析レポートの保存"""
    report_data = {
        'analysis_date': datetime.now().isoformat(),
        'target_image': 'images.jpeg (脂漏性角化症)',
        'actual_diagnosis': 'benign (良性)',
        
        'method_results': {
            'binary_classification': {
                'prediction': 'malignant',
                'confidence': 99.8,
                'correct': False
            },
            'uncertainty_estimation': {
                'prediction': 'malignant', 
                'confidence': 99.7,
                'reliability_score': 100.0,
                'correct': False
            },
            'one_class_classification': {
                'prediction': 'malignant',
                'confidence': 100.0,
                'correct': False
            }
        },
        
        'key_findings': [
            "全手法で一貫した誤分類",
            "非常に高い確信度での誤判定",
            "不確実性推定でも確信的な誤分類",
            "一クラス分類でも同様の問題"
        ],
        
        'root_causes': [
            "SK特徴と悪性疾患の視覚的類似性",
            "学習データのSK代表性不足",
            "深層学習の特徴抽出限界"
        ],
        
        'recommendations': [
            "SK特化データセット拡充",
            "専門医協調診断システム",
            "段階的判定プロセス",
            "継続学習システム"
        ]
    }
    
    with open('/Users/iinuma/Desktop/ダーモ/comprehensive_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 分析レポート保存完了: comprehensive_analysis_report.json")

def main():
    """メイン実行"""
    print("📋 手法比較分析レポート")
    print("   脂漏性角化症誤分類の包括的検証")
    print("=" * 80)
    
    # 包括的分析実行
    analyze_sk_misclassification()
    
    # 技術的サマリー
    generate_technical_summary()
    
    # レポート保存
    save_analysis_report()
    
    print("\\n" + "=" * 80)
    print("📝 結論")
    print("=" * 80)
    print("すべての手法でSK誤分類が発生。根本的なデータ・アーキテクチャ改善が必要。")
    print("現段階では専門医との協調診断が最も現実的なアプローチ。")

if __name__ == "__main__":
    main()