"""
image_to_palette_pyramid.py

画像を1枚渡すと、k-meansで9色を抽出し、正三角形ピラミッド配色図(PNG)を出力する。

使い方:
    python3 image_to_palette_pyramid.py 入力画像.jpg 出力.png \
        --title "The Sick Child" --subtitle "Edvard Munch, 1896"

依存パッケージ:
    pip install pillow numpy scikit-learn scikit-image --break-system-packages
    (cairosvg等の外部ライブラリは不要。描画は全てPillowで完結)
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

N = 3  # 三角形の一辺の分割数。9色なら3、16色なら4、25色なら5。

# 顔料(絵具)風の色名リスト。Lab距離で一番近いものを選んでラベルに使う。
PIGMENT_NAMES = {
    # --- 白・アイボリー系 ---
    "Titanium White": "#F5F1E8", "Lead White": "#EDE6D6", "Ivory": "#FFFFF0",
    "Chalk White": "#F0EAD6", "Bone White": "#E3DAC9",
 
    # --- 黄系 ---
    "Naples Yellow": "#FADA5E", "Cadmium Yellow": "#FFF143", "Yellow Ochre": "#CB9D06",
    "Lemon Yellow": "#FAFA33", "Mustard": "#C2A028", "Straw": "#E4D96F",
    "Old Gold": "#CFB53B",
 
    # --- オレンジ・アプリコット系 ---
    "Cadmium Orange": "#E48454", "Apricot": "#E4B478", "Amber": "#D9A441",
    "Marigold": "#E8A33D", "Persimmon": "#D9763A",
 
    # --- 茶系(シエナ・アンバー・セピアなど) ---
    "Raw Sienna": "#C56E29", "Burnt Sienna": "#8A3324", "Burnt Umber": "#5C3317",
    "Raw Umber": "#826644", "Tobacco": "#807050", "Terra Rosa": "#B4543C",
    "Sepia": "#704214", "Bistre": "#3D2B1F", "Caput Mortuum": "#592720",
    "Walnut": "#5B4636", "Chestnut": "#954535", "Mahogany": "#4E2A1E",
    "Cinnamon": "#7B3F00", "Taupe": "#483C32", "Umber Grey": "#5E5248",
 
    # --- 赤・ピンク系 ---
    "Vermilion": "#E34234", "Cadmium Red": "#E32227", "Venetian Red": "#B4786C",
    "Crimson Lake": "#7D1935", "Madder Rose": "#9E5E6F", "Alizarin Crimson": "#8E3A45",
    "Indian Red": "#A0522D", "Mars Red": "#963430", "Brick Red": "#8B4A3C",
    "Rose Dust": "#C08081", "Dusty Pink": "#D8A7B1", "Blush": "#E3B5A4",
    "Mauve": "#915F6D", "Wine": "#5A2A35",
 
    # --- 紫系 ---
    "Dioxazine Purple": "#4B0082", "Heather": "#9C8AA5", "Plum": "#5D3954",
    "Lavender Grey": "#A6A0B8", "Thistle": "#7E6A8F", "Aubergine": "#3B1F2B",
    "Lilac Mist": "#B9A8C9",
 
    # --- 緑系 ---
    "Olive Green": "#708238", "Sap Green": "#507D2A", "Malachite": "#407060",
    "Viridian": "#40826D", "Emerald Green": "#246B57", "Pine": "#3C4830",
    "Forest Green": "#228B22", "Hooker's Green": "#2C4A33", "Phthalo Green": "#123524",
    "Dark Gold": "#606030", "Moss Green": "#5C6B47", "Sage Green": "#8A9A7A",
    "Juniper": "#3E4F3E", "Spruce": "#2E4034", "Fern Green": "#4F7942",
    "Celadon": "#9CB89A", "Laurel Green": "#6E8B6E", "Bottle Green": "#15463B",
 
    # --- 青緑(ティール)系 ---
    "Teal": "#367588", "Deep Teal": "#1B4D4F", "Verdigris": "#43B3AE",
    "Sea Green": "#2E8B7E", "Patina": "#6E9985",
 
    # --- 青系 ---
    "Prussian Blue": "#003153", "Ultramarine": "#3F00FF", "Cerulean Blue": "#2A52BE",
    "Cobalt Blue": "#0047AB", "Indigo": "#3C3D8C", "Steel Blue": "#46647A",
    "Storm Blue": "#5C6B73", "Midnight Blue": "#1A2238", "Denim Blue": "#3B5B7E",
    "Powder Blue": "#9DB5C9", "Glacier Blue": "#7FA8B8", "Fjord Blue": "#4A6B7A",
    "Dusk Blue": "#39506B", "Navy": "#1B2A4A",
 
    # --- 青灰色・緑灰色(グレイ系の細分化) ---
    "Payne's Grey": "#536878", "Slate Grey": "#708090", "Davy's Grey": "#555555",
    "Charcoal": "#36454F", "Lamp Black": "#2B2B2B", "Ivory Black": "#231F20",
    "Mist Grey": "#9DAAB0", "Stone Grey": "#857F72", "Pewter": "#8A8C8E",
    "Ash Grey": "#A0A29C", "Smoke Grey": "#75787B", "Granite": "#5E6063",
    "Fog Grey": "#C6CBCE", "Driftwood Grey": "#A89F91", "Cool Grey": "#8C92AC",
    "Warm Grey": "#9B9482", "Graphite": "#41424C", "Basalt": "#3A3B3C",
}


def _pigment_lab_table():
    names, labs = [], []
    for name, hexcode in PIGMENT_NAMES.items():
        rgb = np.array([[[int(hexcode[i:i+2], 16) / 255 for i in (1, 3, 5)]]])
        labs.append(rgb2lab(rgb)[0, 0])
        names.append(name)
    return names, np.array(labs)


def _nearest_pigment_name(lab_color, names, lab_table):
    dists = np.linalg.norm(lab_table - lab_color, axis=1)
    return names[int(np.argmin(dists))]


def extract_palette(image_path, n_colors=N * N, sample_size=20000, seed=42):
    """画像からLab空間でk-meansし、明るい順に並んだ [(RGBタプル, Lab値), ...] を返す"""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).reshape(-1, 3) / 255.0

    rng = np.random.default_rng(seed)
    if len(arr) > sample_size:
        arr = arr[rng.choice(len(arr), sample_size, replace=False)]

    lab = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    km = KMeans(n_clusters=n_colors, random_state=seed, n_init="auto").fit(lab)
    centers_lab = km.cluster_centers_[np.argsort(-km.cluster_centers_[:, 0])]  # 明るい順
    centers_rgb = np.clip(lab2rgb(centers_lab.reshape(-1, 1, 3)).reshape(-1, 3) * 255, 0, 255).astype(int)

    return list(zip(map(tuple, centers_rgb), centers_lab))


def _hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _luminance(rgb):
    r, g, b = [c / 255 for c in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _load_font(size, bold=False, italic=False):
    """環境によって入っているフォントが違うので、いくつか候補を試して見つかったものを使う。
    どれも無ければPillow付属のデフォルトフォントにフォールバックする。"""
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/Library/Fonts/Georgia Bold.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
        ]
    elif italic:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
            "/Library/Fonts/Georgia Italic.ttf",
            "C:/Windows/Fonts/georgiai.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/Library/Fonts/Georgia.ttf",
            "C:/Windows/Fonts/georgia.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_pyramid_image(palette, n=N, title="", subtitle="", tri_size=900, margin=40):
    """palette(長さ n^2)から三角形タイリングのPillow Imageを直接描画する。
    cairosvg等の外部ライブラリ(ネイティブ依存あり)を使わないので環境を選ばない。
    margin: 三角形の上下左右に入れる余白(px)。タイトル用にさらに下が広がる。"""
    assert len(palette) == n * n, f"n={n}には{n*n}色必要です"
    names, lab_table = _pigment_lab_table()

    title_area = 70 if (title or subtitle) else margin
    tri_W, tri_H = tri_size, int(tri_size * (3 ** 0.5) / 2)
    W = tri_W + margin * 2
    H = tri_H + margin + title_area
    img = Image.new("RGB", (W, H), "#F7F4EC")
    draw = ImageDraw.Draw(img)

    # 三角形をキャンバス中央寄り、上にmarginぶん下げて配置
    apex = (W / 2, float(margin))
    base_l = (float(margin), float(margin + tri_H))
    base_r = (float(margin + tri_W), float(margin + tri_H))

    def grid_point(i, j):
        if i == 0:
            return apex
        u = i / n
        left = (apex[0] + u * (base_l[0] - apex[0]), apex[1] + u * (base_l[1] - apex[1]))
        right = (apex[0] + u * (base_r[0] - apex[0]), apex[1] + u * (base_r[1] - apex[1]))
        t = j / i
        return (left[0] + t * (right[0] - left[0]), left[1] + t * (right[1] - left[1]))

    font_hex = _load_font(max(11, tri_size // 75), bold=True)
    font_name = _load_font(max(11, tri_size // 75), italic=True)
    font_title = _load_font(max(16, tri_size // 45))
    font_subtitle = _load_font(max(12, tri_size // 65))

    idx = 0
    for row in range(n):
        cells = []
        for j in range(row + 1):
            cells.append((row, j, row + 1, j, row + 1, j + 1))       # 上向き三角形
            if j < row:
                cells.append((row, j, row, j + 1, row + 1, j + 1))    # 下向き三角形

        for (i1, j1, i2, j2, i3, j3) in cells:
            p1, p2, p3 = grid_point(i1, j1), grid_point(i2, j2), grid_point(i3, j3)
            rgb, lab = palette[idx]
            hexcode = _hex(rgb)
            pname = _nearest_pigment_name(lab, names, lab_table)
            cx, cy = (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3

            draw.polygon([p1, p2, p3], fill=rgb)
            # draw.polygon([p1, p2, p3], fill=rgb, outline="#F7F4EC", width=2) # 境界線あり

            text_color = "#FFFFFF" if _luminance(rgb) < 0.55 else "#222222"
            draw.text((cx, cy - 10), hexcode, font=font_hex, fill=text_color, anchor="mm")
            draw.text((cx, cy + 10), pname, font=font_name, fill=text_color, anchor="mm")
            idx += 1

    if title:
        draw.text((margin, margin + tri_H + 12), title, font=font_title, fill="#333333")
    if subtitle:
        draw.text((margin, margin + tri_H + 40), subtitle, font=font_subtitle, fill="#777777")

    return img


def image_to_pyramid_png(input_path, output_path, n=N, title="", subtitle="", tri_size=900, margin=40):
    """画像パスを受け取り、ピラミッドPNGを書き出すまでの一括処理"""
    palette = extract_palette(input_path, n_colors=n * n)
    img = draw_pyramid_image(palette, n=n, title=title, subtitle=subtitle, tri_size=tri_size, margin=margin)
    img.save(output_path)
    print(f"Saved -> {output_path}")
    for rgb, _ in palette:
        print(f"  {_hex(rgb)}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="画像からk-meansでパレットを抽出し、三角形ピラミッド画像を出力する")
    ap.add_argument("input", help="入力画像のパス")
    ap.add_argument("output", help="出力PNGのパス")
    ap.add_argument("--n", type=int, default=N, help="三角形の一辺の分割数 (色数 = n^2)。デフォルト3")
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    image_to_pyramid_png(args.input, args.output, n=args.n, title=args.title, subtitle=args.subtitle)
