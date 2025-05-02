import streamlit as st
import pandas as pd

# --- ページ設定 ---
st.set_page_config(page_title="ライン競輪スコア計算（完全統一版）", layout="wide")

st.title("⭐ ライン競輪スコア計算（7車ライン＋政春補正＋欠番対応）⭐")

# --- 定義部分 ---
wind_coefficients = {"左上": +0.7, "上": +1.0, "右上": +0.7, "左": 0.0, "右": 0.0, "左下": -0.7, "下": -1.0, "右下": -0.7}
position_multipliers = {1: 1.0, 2: 0.3, 3: 0.1, 0: 1.2}
base_score = {'逃': 8, '両': 6, '追': 5}
symbol_bonus = {'◎': 2.0, '〇': 1.5, '▲': 1.0, '△': 0.5, '×': 0.2, '無': 0.0}

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
    "函館": {"bank_angle": 30.6, "straight_length": 51.3},
    "青森": {"bank_angle": 32.3, "straight_length": 58.9},
    "いわき平": {"bank_angle": 32.9, "straight_length": 62.7},
    "弥彦": {"bank_angle": 32.4, "straight_length": 63.1},
    "前橋": {"bank_angle": 36.0, "straight_length": 46.7},
    "取手": {"bank_angle": 31.5, "straight_length": 54.8},
    "宇都宮": {"bank_angle": 25.8, "straight_length": 63.3},
    "大宮": {"bank_angle": 26.3, "straight_length": 66.7},
    "西武園": {"bank_angle": 29.4, "straight_length": 47.6},
    "京王閣": {"bank_angle": 32.2, "straight_length": 51.5},
    "立川": {"bank_angle": 31.2, "straight_length": 58.0},
    "松戸": {"bank_angle": 29.8, "straight_length": 38.2},
    "川崎": {"bank_angle": 32.2, "straight_length": 58.0},
    "平塚": {"bank_angle": 31.5, "straight_length": 54.2},
    "小田原": {"bank_angle": 35.6, "straight_length": 36.1},
    "伊東": {"bank_angle": 34.7, "straight_length": 46.6},
    "静岡": {"bank_angle": 30.7, "straight_length": 56.4},
    "名古屋": {"bank_angle": 34.0, "straight_length": 58.8},
    "岐阜": {"bank_angle": 32.3, "straight_length": 59.3},
    "大垣": {"bank_angle": 30.6, "straight_length": 56.0},
    "豊橋": {"bank_angle": 33.8, "straight_length": 60.3},
    "富山": {"bank_angle": 33.7, "straight_length": 43},
    "松坂": {"bank_angle": 34.4, "straight_length": 61.5},
    "四日市": {"bank_angle": 32.3, "straight_length": 62.4},
    "福井": {"bank_angle": 31.5, "straight_length": 52.8},
    "奈良": {"bank_angle": 33.4, "straight_length": 38.0},
    "向日町": {"bank_angle": 30.5, "straight_length": 47.3},
    "和歌山": {"bank_angle": 32.3, "straight_length": 59.9},
    "岸和田": {"bank_angle": 30.9, "straight_length": 56.7},
    "玉野": {"bank_angle": 30.6, "straight_length": 47.9},
    "広島": {"bank_angle": 30.8, "straight_length": 57.9},
    "防府": {"bank_angle": 34.7, "straight_length": 42.5},
    "高松": {"bank_angle": 33.3, "straight_length": 54.8},
    "小松島": {"bank_angle": 29.8, "straight_length": 55.5},
    "高知": {"bank_angle": 24.5, "straight_length": 52},
    "松山": {"bank_angle": 34.0, "straight_length": 58.6},
    "小倉": {"bank_angle": 34.0, "straight_length": 56.9},
    "久留米": {"bank_angle": 31.5, "straight_length": 50.7},
    "武雄": {"bank_angle": 32.0, "straight_length": 64.4},
    "佐世保": {"bank_angle": 31.5, "straight_length": 40.2},
    "別府": {"bank_angle": 33.7, "straight_length": 59.9},
    "熊本": {"bank_angle": 34.3, "straight_length": 60.3},
    "手入力": {"bank_angle": 30.0, "straight_length": 52}
}

selected_track = st.selectbox("▼ 競輪場選択（自動入力）", list(keirin_data.keys()))
selected_info = keirin_data[selected_track]

# ▼ 風速入力（手動）
wind_speed = st.number_input("風速(m/s)", min_value=0.0, max_value=30.0, step=0.1, value=3.0)

# ▼ 自動反映される直線長さ・バンク角
straight_length = st.number_input("みなし直線(m)", min_value=30.0, max_value=80.0, step=0.1,
                                  value=float(selected_info["straight_length"]))

bank_angle = st.number_input("バンク角(°)", min_value=20.0, max_value=45.0, step=0.1,
                             value=float(selected_info["bank_angle"]))


# ▼ 雨チェック（最後に）
rain = st.checkbox("雨（滑走・慎重傾向あり）")

# --- 【選手データ入力】 ---
st.header("【選手データ入力】")

kakushitsu_options = ['逃', '両', '追']
symbol_input_options = ['◎', '〇', '▲', '△', '×', '無']

st.subheader("▼ 脚質入力")
kakushitsu = [st.selectbox(f"{i+1}番脚質", kakushitsu_options, key=f"kaku_{i}") for i in range(7)]

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


st.subheader("▼ 競争得点入力")
rating = [st.number_input(f"{i+1}番得点", value=55.0, step=0.1, key=f"rate_{i}") for i in range(7)]

st.subheader("▼ 予想隊列入力（数字、欠の場合は空欄）")
tairetsu = [st.text_input(f"{i+1}番隊列順位", key=f"tai_{i}") for i in range(7)]

st.subheader("▼ ラインポジション入力（0単騎 1先頭 2番手 3三番手 4四番手）")
line_order = [
    st.number_input(f"{i+1}番ラインポジション", min_value=0, max_value=4, step=1, value=0, key=f"line_{i}")
    for i in range(7)
]

st.subheader("▼ 政春印入力（各記号ごとに該当車番を入力）")

# 使用する記号と評価値
symbol_input_options = ['◎', '〇', '▲', '△', '×', '無', 'ム']
symbol_bonus = {
    '◎': 2.0, '〇': 1.5, '▲': 1.0, '△': 0.5, '×': 0.2,
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
    '◎': 2.0, '〇': 1.5, '▲': 1.0, '△': 0.5, '×': 0.2,
    '無': 0.0, 'ム': 0.0  # 同じ評価
}


# --- スコア計算 ---
if st.button("スコア計算実行"):

    def wind_straight_combo_adjust(kaku, direction, speed, straight, pos):
        if direction == "無風" or speed < 0.5:
            return 0
        basic = wind_coefficients.get(direction, 0.0) * speed * position_multipliers[pos]
        coeff = {'逃': 1.2, '両': 1.0, '追': 0.8}.get(kaku, 1.0)
        return round(basic * coeff, 2)

    def tairyetsu_adjust(num, tairetsu_list):
        pos = tairetsu_list.index(num)
        base = max(0, round(3.0 - 0.5 * pos, 1))
        if kakushitsu[num - 1] == '追':
            if 2 <= pos <= 4:
                return base + 0.5 + 1.5
            else:
                return base + 0.5
        return base

    def score_from_chakujun(pos):
        if pos == 1: return 3.0
        elif pos == 2: return 2.5
        elif pos == 3: return 2.0
        elif pos <= 6: return 1.0
        else: return 0.0

    def rain_adjust(kaku):
        if not rain:
            return 0
        return {'逃': +2.5, '両': +0.5, '追': -2.5}[kaku]

def line_member_bonus(pos):
    if pos == 0:
        return -1.0
    elif pos == 1:
        return 2.0
    elif pos == 2:
        return 1.5
    elif pos == 3:
        return 1.0
    elif pos == 4:
        return 0.5  # ← ここを追加！
    return 0.0

    def bank_character_bonus(kaku, angle, straight):
        base_straight = 50.0
        base_angle = 30.0

        straight_factor = (straight - base_straight) / 10.0
        angle_factor = (angle - base_angle) / 5.0

        total_factor = -0.8 * straight_factor + 0.6 * angle_factor

        bonus = {
            '逃': +total_factor,
            '追': -total_factor,
            '両': 0.0
        }.get(kaku, 0.0)

        return round(bonus, 2)



    tairetsu_list = [i+1 for i, v in enumerate(tairetsu) if v.isdigit()]

    score_parts = []
    for i in range(7):
        if not tairetsu[i].isdigit():
            continue
        num = i + 1
        base = base_score[kakushitsu[i]]
        wind = wind_straight_combo_adjust(kakushitsu[i], st.session_state.selected_wind, wind_speed, straight_length, line_order[i])
        tai = tairyetsu_adjust(num, tairetsu_list)
        kasai = score_from_chakujun(chaku[i])
        rating_score = max(0, round((sum(rating)/7 - rating[i]) * 0.2, 1))
        rain_corr = rain_adjust(kakushitsu[i])
        symbol_bonus_score = symbol_bonus.get(car_to_symbol.get(num, '無'), 0.0)
        line_bonus = line_member_bonus(line_order[i])
        bank_bonus = bank_character_bonus(kakushitsu[i], bank_angle, straight_length)
        total = base + wind + tai + kasai + rating_score + rain_corr + symbol_bonus_score + line_bonus + bank_bonus

        score_parts.append((num, kakushitsu[i], base, wind, tai, kasai, rating_score, rain_corr, symbol_bonus_score, line_bonus, bank_bonus, total))

    df = pd.DataFrame(score_parts, columns=[
        '車番', '脚質', '基本', '風補正', '隊列補正', '着順補正', '得点補正', '雨補正', '政春印補正', 'ライン補正', 'バンク補正', '合計スコア'
    ])
    st.dataframe(df.sort_values(by='合計スコア', ascending=False).reset_index(drop=True))

