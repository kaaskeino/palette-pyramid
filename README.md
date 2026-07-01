# 🎨 Palette Pyramid
 
絵画や写真をアップロードすると、k-meansで色を抽出し、正三角形ピラミッド配色図(PNG)を生成するWebアプリ。
 
![Python](https://img.shields.io/badge/Python-3.x-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Community_Cloud-red)
 
## 使い方
 
1. アプリにアクセス → **(StreamlitのURLをここに貼る)**
2. 画像(絵画・写真など)をアップロード
3. タイトル・サブタイトルを入力
4. 「Generate pyramid」をクリック
5. 生成されたピラミッド画像をダウンロード
## 機能
 
- **色抽出**: CIE Lab色空間でk-meansクラスタリング(9色 / 16色 / 25色から選択)
- **色名ラベリング**: 顔料風の色名辞書(132色)との最近傍探索で自動命名
- **PNG出力**: タイトル・サブタイトル付きのピラミッド画像をダウンロード可能
## 技術スタック
 
- **Python**
- **Streamlit** — UI / デプロイ
- **scikit-learn** — k-meansクラスタリング
- **scikit-image** — Lab色空間変換
- **Pillow** — 画像描画
## ファイル構成
 
```
palette-pyramid/
├── app.py                      # Streamlit UI
├── image_to_palette_pyramid.py # 色抽出・ピラミッド描画ロジック
└── requirements.txt            # 依存パッケージ
```
