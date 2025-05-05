import streamlit as st
import pandas as pd

# --- ページ設定 ---
st.set_page_config(page_title="ライン競輪スコア計算（完全統一版）", layout="wide")

st.title("⭐ ライン競輪スコア計算（7車ライン＋政春補正＋欠番対応）⭐")

wind_coefficients = {
    "左上": +0.7,
    "上": +1.0,
    "右上": +0.7,
    "左": -0.2,   # 左風はやや不利（ホーム直線で外から押される）
    "右": +0.2,   # 右風はやや有利（バック直線で外に膨らみやすい）
    "左下": -0.7,
    "下": -1.0,
    "右下": -0.7
}

position_multipliers = {
    0: 1.2,  # 単騎
    1: 1.0,  # 先頭
    2: 0.3,
    3: 0.1,
    4: 0.05  # 4番手
}


# --- 基本スコア（脚質ごとの基準値） ---
base_score = {'逃': 6, '両': 5.5, '追': 5}
symbol_bonus = {'◎': 0.6, '〇': 0.4, '▲': 0.3, '△': 0.2, '×': 0.1, '無': 0.0}

# --- 状態保持 ---
if "selected_wind" not in st.session_state:
    st.session_state.selected_wind = "無風"

# --- バンク・風条件セクション ---
st.header("【バンク・風条件】")

cols_top = st.columns(3)
cols_mid = st.columns(3)
cols_bot = st.columns(3)

with cols_top[0]:
    if st.button("左上"):
        st.session_state.selected_wind = "左上"
with cols_top[1]:
    if st.button("上"):
        st.session_state.selected_wind = "上"
with cols_top[2]:
    if st.button("右上"):
        st.session_state.selected_wind = "右上"
with cols_mid[0]:
    if st.button("左"):
        st.session_state.selected_wind = "左"
with cols_mid[1]:
    st.markdown("""
    <div style='text-align:center; font-size:16px; line-height:1.6em;'>
        ↑<br>［上］<br>
        ← 左　　　右 →<br>
        ［下］<br>↓<br>
        □ ホーム→（ ゴール）
    </div>
    """, unsafe_allow_html=True)
with cols_mid[2]:
    if st.button("右"):
        st.session_state.selected_wind = "右"
with cols_bot[0]:
    if st.button("左下"):
        st.session_state.selected_wind = "左下"
with cols_bot[1]:
    if st.button("下"):
        st.session_state.selected_wind = "下"
with cols_bot[2]:
    if st.button("右下"):
        st.session_state.selected_wind = "右下"

st.subheader(f"✅ 選択中の風向き：{st.session_state.selected_wind}")

# ▼ 競輪場選択による自動入力
keirin_data = {
    "函館": {"bank_angle": 30.6, "straight_length": 51.3, "bank_length": 400},
    "青森": {"bank_angle": 32.3, "straight_length": 58.9, "bank_length": 400},
    "いわき平": {"bank_angle": 32.9, "straight_length": 62.7, "bank_length": 400},
    "弥彦": {"bank_angle": 32.4, "straight_length": 63.1, "bank_length": 400},
    "前橋": {"bank_angle": 36.0, "straight_length": 46.7, "bank_length": 335},
    "取手": {"bank_angle": 31.5, "straight_length": 54.8, "bank_length": 400},
    "宇都宮": {"bank_angle": 25.8, "straight_length": 63.3, "bank_length": 500},
    "大宮": {"bank_angle": 26.3, "straight_length": 66.7, "bank_length": 500},
    "西武園": {"bank_angle": 29.4, "straight_length": 47.6, "bank_length": 400},
    "京王閣": {"bank_angle": 32.2, "straight_length": 51.5, "bank_length": 400},
    "立川": {"bank_angle": 31.2, "straight_length": 58.0, "bank_length": 400},
    "松戸": {"bank_angle": 29.8, "straight_length": 38.2, "bank_length": 333},
    "川崎": {"bank_angle": 32.2, "straight_length": 58.0, "bank_length": 400},
    "平塚": {"bank_angle": 31.5, "straight_length": 54.2, "bank_length": 400},
    "小田原": {"bank_angle": 35.6, "straight_length": 36.1, "bank_length": 333},
    "伊東": {"bank_angle": 34.7, "straight_length": 46.6, "bank_length": 333},
    "静岡": {"bank_angle": 30.7, "straight_length": 56.4, "bank_length": 400},
    "名古屋": {"bank_angle": 34.0, "straight_length": 58.8, "bank_length": 400},
    "岐阜": {"bank_angle": 32.3, "straight_length": 59.3, "bank_length": 400},
    "大垣": {"bank_angle": 30.6, "straight_length": 56.0, "bank_length": 400},
    "豊橋": {"bank_angle": 33.8, "straight_length": 60.3, "bank_length": 400},
    "富山": {"bank_angle": 33.7, "straight_length": 43.0, "bank_length": 333},
    "松坂": {"bank_angle": 34.4, "straight_length": 61.5, "bank_length": 400},
    "四日市": {"bank_angle": 32.3, "straight_length": 62.4, "bank_length": 400},
    "福井": {"bank_angle": 31.5, "straight_length": 52.8, "bank_length": 400},
    "奈良": {"bank_angle": 33.4, "straight_length": 38.0, "bank_length": 333},
    "向日町": {"bank_angle": 30.5, "straight_length": 47.3, "bank_length": 400},
    "和歌山": {"bank_angle": 32.3, "straight_length": 59.9, "bank_length": 400},
    "岸和田": {"bank_angle": 30.9, "straight_length": 56.7, "bank_length": 400},
    "玉野": {"bank_angle": 30.6, "straight_length": 47.9, "bank_length": 400},
    "広島": {"bank_angle": 30.8, "straight_length": 57.9, "bank_length": 400},
    "防府": {"bank_angle": 34.7, "straight_length": 42.5, "bank_length": 333},
    "高松": {"bank_angle": 33.3, "straight_length": 54.8, "bank_length": 400},
    "小松島": {"bank_angle": 29.8, "straight_length": 55.5, "bank_length": 400},
    "高知": {"bank_angle": 24.5, "straight_length": 52.0, "bank_length": 500},
    "松山": {"bank_angle": 34.0, "straight_length": 58.6, "bank_length": 400},
    "小倉": {"bank_angle": 34.0, "straight_length": 56.9, "bank_length": 400},
    "久留米": {"bank_angle": 31.5, "straight_length": 50.7, "bank_length": 400},
    "武雄": {"bank_angle": 32.0, "straight_length": 64.4, "bank_length": 400},
    "佐世保": {"bank_angle": 31.5, "straight_length": 40.2, "bank_length": 400},
    "別府": {"bank_angle": 33.7, "straight_length": 59.9, "bank_length": 400},
    "熊本": {"bank_angle": 34.3, "straight_length": 60.3, "bank_length": 400},
    "手入力": {"bank_angle": 30.0, "straight_length": 52.0, "bank_length": 400}
}


selected_track = st.selectbox("▼ 競輪場選択（自動入力）", list(keirin_data.keys()))
selected_info = keirin_data[selected_track]

# ▼ 風速入力（手動）
wind_speed = st.number_input("風速(m/s)", min_value=0.0, max_value=30.0, step=0.1, value=3.0)

# ▼ 自動反映される直線長さ・バンク角・周長
straight_length = st.number_input("みなし直線(m)", min_value=30.0, max_value=80.0, step=0.1,
                                  value=float(selected_info["straight_length"]))

bank_angle = st.number_input("バンク角(°)", min_value=20.0, max_value=45.0, step=0.1,
                             value=float(selected_info["bank_angle"]))

bank_length = st.number_input("バンク周長(m)", min_value=300.0, max_value=500.0, step=0.1,
                              value=float(selected_info["bank_length"]))


# ▼ 雨チェック（最後に）
rain = st.checkbox("雨（滑走・慎重傾向あり）")

# --- 【選手データ入力】 ---
st.header("【選手データ入力】")

symbol_input_options = ['◎', '〇', '▲', '△', '×', '無']

st.subheader("▼ 脚質入力（逃・両・追：車番を半角数字で入力）")

kakushitsu_keys = ['逃', '両', '追']
kakushitsu_inputs = {}
cols = st.columns(3)
for i, k in enumerate(kakushitsu_keys):
    with cols[i]:
        st.markdown(f"**{k}**")
        kakushitsu_inputs[k] = st.text_input("", key=f"kaku_{k}", max_chars=14)

# 車番 → 脚質の辞書を構築
car_to_kakushitsu = {}
for k, val in kakushitsu_inputs.items():
    for c in val:
        if c.isdigit():
            n = int(c)
            if 1 <= n <= 7:
                car_to_kakushitsu[n] = k

# kakushitsu[0] = 1番選手の脚質、など
kakushitsu = [car_to_kakushitsu.get(i + 1, '追') for i in range(7)]  # 未指定は「追」で補完

st.subheader("▼ 前走着順入力（1〜7着）")
chaku = [st.number_input(f"{i+1}番着順", min_value=1, max_value=7, value=5, step=1, key=f"chaku_{i}") for i in range(7)]


st.subheader("▼ 競争得点入力")
rating = [st.number_input(f"{i+1}番得点", value=55.0, step=0.1, key=f"rate_{i}") for i in range(7)]

st.subheader("▼ 予想隊列入力（数字、欠の場合は空欄）")
tairetsu = [st.text_input(f"{i+1}番隊列順位", key=f"tai_{i}") for i in range(7)]



st.subheader("▼ 政春印入力（各記号ごとに該当車番を入力）")

# 使用する記号と評価値
symbol_input_options = ['◎', '〇', '▲', '△', '×', '無', 'ム']
symbol_bonus = {
    '◎': 0.6, '〇': 0.4, '▲': 0.3, '△': 0.2, '×': 0.1,
    '無': 0.0, 'ム': 0.0
}

# 入力欄を記号ごとに表示（タブ入力可能）
symbol_inputs = {}
cols = st.columns(len(symbol_input_options))
for i, sym in enumerate(symbol_input_options):
    with cols[i]:
        st.markdown(f"**{sym}**")
        symbol_inputs[sym] = st.text_input("", key=f"symbol_{sym}", max_chars=14)

# 車番→記号 の辞書を構築
car_to_symbol = {}
for sym, input_str in symbol_inputs.items():
    for c in input_str:
        if c.isdigit():
            car_to_symbol[int(c)] = sym
    
# 使用記号リストと評価値
symbol_keys = ['◎', '〇', '▲', '△', '×', '無', 'ム']
symbol_bonus = {
    '◎': 0.6, '〇': 0.4, '▲': 0.3, '△': 0.2, '×': 0.1,
    '無': 0.0, 'ム': 0.0  # 同じ評価
}

# --- スコア計算処理（入力UIのline_inputから処理） ---
import pandas as pd

# ▼ ライン構成入力（カンマ区切りで記入）
st.subheader("▼ ライン構成入力（カンマ区切りで記入）")
line_input = st.text_input("例）137,25,4,6（A=137, B=25, C=4, 単騎=6）")

# ▼ スコア計算ボタン
st.subheader("▼ スコア計算")
if st.button("スコア計算実行"):

    # --- ラインマップ構築 ---
    def build_line_order_map(line_input):
        car_to_line = {}
        line_order_map = {}
        lines = [line.strip() for line in line_input.split(',') if line.strip()]
        for idx, line_str in enumerate(lines):
            cars = [int(c) for c in line_str if c.isdigit()]
            for i, car in enumerate(cars):
                line_order_map[car] = i + 1 if len(cars) > 1 else 0
                car_to_line[car] = chr(65 + idx) if len(cars) <= 3 else 'D'
        return line_order_map, car_to_line

    # --- グループ補正 ---
    def compute_group_bonus(score_parts, car_to_line):
        group_scores = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        group_counts = {'A': 0, 'B': 0, 'C': 0}
        for entry in score_parts:
            car_no = entry[0]
            group = car_to_line.get(car_no)
            if group in group_scores:
                group_scores[group] += entry[-1]
                group_counts[group] += 1
        group_avg = {
            k: (group_scores[k] / group_counts[k]) if group_counts[k] > 0 else 0.0
            for k in group_scores
        }
        sorted_lines = sorted(group_avg.items(), key=lambda x: x[1], reverse=True)
        bonus_values = [0.15, 0.08, 0.03]
        bonus_map = {group: bonus_values[i] if i < len(bonus_values) else 0.0 for i, (group, _) in enumerate(sorted_lines)}
        return bonus_map

    def get_group_bonus(car_no, car_to_line, group_bonus_map):
        group = car_to_line.get(car_no)
        return group_bonus_map.get(group, 0.0)

    # --- 計算用補助関数 ---
    def score_from_tenscore_list(tenscore_list):
        sorted_unique = sorted(set(tenscore_list), reverse=True)
        score_to_rank = {score: rank + 1 for rank, score in enumerate(sorted_unique)}
        result = []
        for score in tenscore_list:
            rank = score_to_rank[score]
            if rank == 1:
                correction = -0.6
            elif rank == 2:
                correction = -0.4
            elif rank == 3:
                correction = -0.2
            elif rank == 4:
                correction = 0.0
            elif rank == 5:
                correction = +0.2
            elif rank == 6:
                correction = +0.1
            else:
                correction = 0.0
            result.append(correction)
        return result

    # --- ここで必要変数を受け取る（例として仮定。あなたの環境に合わせて調整） ---
    # kakushitsu, chaku, rating, tairetsu, symbol_bonus, car_to_symbol などは事前に定義済みとする

    # ▼ 前提：line_inputが定義済み
    line_order_map, car_to_line = build_line_order_map(line_input)
    line_order = [line_order_map.get(i + 1, 0) for i in range(7)]

    # ▼ 得点スコア
    tenscore_score = score_from_tenscore_list(rating)

    score_parts = []
    for i in range(7):
        if not tairetsu[i].isdigit():
            continue
        num = i + 1
        base = base_score[kakushitsu[i]]
        wind = wind_straight_combo_adjust(kakushitsu[i], st.session_state.selected_wind, wind_speed, straight_length, line_order[i])
        kasai = score_from_chakujun(chaku[i])
        rating_score = tenscore_score[i]
        rain_corr = rain_adjust(kakushitsu[i])
        symbol_bonus_score = symbol_bonus.get(car_to_symbol.get(num, '無'), 0.0)
        line_bonus = line_member_bonus(line_order[i])
        bank_bonus = bank_character_bonus(kakushitsu[i], bank_angle, straight_length)
        length_bonus = bank_length_adjust(kakushitsu[i], bank_length)
        total = base + wind + kasai + rating_score + rain_corr + symbol_bonus_score + line_bonus + bank_bonus + length_bonus
        score_parts.append((
            num, kakushitsu[i], base, wind, kasai, rating_score,
            rain_corr, symbol_bonus_score, line_bonus, bank_bonus, length_bonus, total
        ))

    group_bonus_map = compute_group_bonus(score_parts, car_to_line)
    final_score_parts = []
    for row in score_parts:
        car_no = row[0]
        group_corr = get_group_bonus(car_no, car_to_line, group_bonus_map)
        new_total = row[-1] + group_corr
        final_score_parts.append(row[:-1] + [group_corr, new_total])

    # ▼ 表示
    df = pd.DataFrame(final_score_parts, columns=[
        '車番', '脚質', '基本', '風補正', '着順補正', '得点補正',
        '雨補正', '政春印補正', 'ライン補正', 'バンク補正', '周長補正',
        'グループ補正', '合計スコア'
    ])
    st.dataframe(df.sort_values(by='合計スコア', ascending=False).reset_index(drop=True))
