import streamlit as st
import pandas as pd

# --- ページ設定 ---
st.set_page_config(page_title="ライン競輪スコア計算（完全統一版）", layout="wide")

st.title("⭐ ライン競輪スコア計算（7車ライン＋欠番対応）⭐")

wind_coefficients = {
    "左上": -0.07,   # ホーム寄りからの風 → 差し有利（逃げやや不利）
    "上":   -0.10,   # バック向かい風 → 逃げ最大不利
    "右上": -0.07,   # 差しやや有利

    "左":   +0.10,   # ホーム向かい風 → 差し不利、逃げ有利
    "右":   -0.10,   # バック追い風 → 差し不利、逃げ有利

    "左下": +0.07,   # ゴール寄り追い風 → 差しやや有利
    "下":   +0.10,   # ゴール強追い風 → 差し最大有利（逃げ最大不利）
    "右下": +0.07    # 差しやや有利
}
position_multipliers = {
    0: 0.6,  # 単騎
    1: 0.65,  # 先頭
    2: 0.6,
    3: 0.5,
    4: 0.4  # 4番手
}


# --- 基本スコア（脚質ごとの基準値） ---
base_score = {'逃': 5.0, '両': 4.9, '追': 4.7}

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


# ▼ 周回数の入力（通常は4、高松などは5）
laps = st.number_input("周回数（通常は4、高松などは5）", min_value=1, max_value=10, value=4, step=1)

# --- 【選手データ入力】 ---
st.header("【選手データ入力】")

st.subheader("▼ 位置入力（逃＝先頭・両＝番手・追＝３番手以降&単騎：車番を半角数字で入力）")

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

st.subheader("▼ 前々走・前走の着順入力（1〜9着 または 0＝落車）")

# 7選手 × 2走分
chaku_inputs = []  # [[前々走, 前走], ..., [前々走, 前走]]

for i in range(7):
    col1, col2 = st.columns(2)
    with col1:
        chaku1 = st.text_input(f"{i+1}番【前々走】", value="", key=f"chaku1_{i}")
    with col2:
        chaku2 = st.text_input(f"{i+1}番【前走】", value="", key=f"chaku2_{i}")
    chaku_inputs.append([chaku1, chaku2])



st.subheader("▼ 競争得点入力")
rating = [st.number_input(f"{i+1}番得点", value=55.0, step=0.1, key=f"rate_{i}") for i in range(7)]

st.subheader("▼ 予想隊列入力（数字、欠の場合は空欄）")
tairetsu = [st.text_input(f"{i+1}番隊列順位", key=f"tai_{i}") for i in range(7)]


# --- S・B 入力（回数を数値で入力） ---
st.subheader("▼ S・B 入力（各選手のS・B回数を入力）")

for i in range(7):
    st.markdown(f"**{i+1}番**")
    s_val = st.number_input("S回数", min_value=0, max_value=99, value=0, step=1, key=f"s_point_{i+1}")
    b_val = st.number_input("B回数", min_value=0, max_value=99, value=0, step=1, key=f"b_point_{i+1}")

# --- 年齢と級の入力（代謝補正用） ---
st.subheader("▼ 年齢・級（代謝補正用）")

race_class = st.selectbox("級を選択", ["S級", "A級", "チャレンジ"])

ages = [st.number_input(f"{i+1}番 年齢", min_value=16, max_value=60, value=40, key=f"age_{i+1}") for i in range(7)]


# --- ライン構成入力欄（A〜Cライン＋単騎） ---
st.subheader("▼ ライン構成入力（A〜Cライン＋単騎）")
a_line = st.text_input("Aライン（例：137）", max_chars=7)
b_line = st.text_input("Bライン（例：25）", max_chars=7)
c_line = st.text_input("Cライン（例：4）", max_chars=7)
solo_line = st.text_input("単騎枠（例：6）", max_chars=7)

# --- ライン構成入力に必要な補助関数 ---
def extract_car_list(input_str):
    return [int(c) for c in input_str if c.isdigit()]

def build_line_position_map():
    result = {}
    for line, name in zip([a_line, b_line, c_line, solo_line], ['A', 'B', 'C', 'D']):
        cars = extract_car_list(line)
        for i, car in enumerate(cars):
            if name == 'D':
                result[car] = 0
            else:
                result[car] = i + 1
    return result

# --- 代謝補正関数（スコア計算の直前に定義） ---
    def get_metabolism_score(age, class_type):
        if class_type == "チャレンジ":
            if age >= 45:
                return 0.15
            elif age >= 38:
                return 0.07
        elif class_type == "A級":
            if age >= 46:
                return 0.10
            elif age >= 40:
                return 0.05
        elif class_type == "S級":
            if age >= 48:
                return 0.05
        return 0.0

# --- 代謝スコアのリストを作成 ---
try:
    metabolism_scores = [get_metabolism_score(ages[i], race_class) for i in range(7)]
except Exception:
    metabolism_scores = [0.0] * 7


# --- スコア計算処理 ---
st.subheader("▼ スコア計算")
if st.button("スコア計算実行"):

    def score_from_tenscore_list(tenscore_list):
        import pandas as pd
    
        df = pd.DataFrame({"得点": tenscore_list})
        df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    
        # 2〜6位の平均得点を基準
        baseline = df[df["順位"].between(2, 6)]["得点"].mean()
        df["元の補正値"] = ((baseline - df["得点"]) / 5).round(3)
    
        # 6位の補正値を取得
        sixth = df[df["順位"] == 6]["元の補正値"].values[0]
    
        # 補正適用関数
        def apply_limit(row):
            if row["順位"] <= 4:
                return 0.0  # 上位1〜4位は補正なし
            elif row["順位"] == 7:
                # 6位との差を最大+0.1に制限
                diff = row["元の補正値"] - sixth
                return round(sixth + min(diff, 0.1), 3)
            else:
                return row["元の補正値"]
    
        df["最終補正値"] = df.apply(apply_limit, axis=1)
        return df["最終補正値"].tolist()


    def wind_straight_combo_adjust(kaku, direction, speed, straight, pos):
        if direction == "無風" or speed < 0.5:
            return 0
    
        base = wind_coefficients.get(direction, 0.0)  # e.g. 上=+0.10
        pos_mult = position_multipliers.get(pos, 0.0)  # e.g. 先頭=1.0, 番手=0.6
    
        # 強化された脚質補正係数（±1.0スケールに）
        kaku_coeff = {
            '逃': +1.0,
            '両':  0.5,
            '追': -1.0
        }.get(kaku, 0.0)
    
        total = base * speed * pos_mult * kaku_coeff  # 例: +0.1×10×1×1 = +1.0
        return round(total, 2)

    def convert_chaku_to_score(values):
        scores = []
        for i, v in enumerate(values):
            v = v.strip()
            try:
                chaku = int(v)
                if chaku == 0:
                    score = 0.0
                elif 1 <= chaku <= 9:
                    score = round(1.0 / chaku, 2)
                    if i == 1:  # 2番目（前々走）だけ過小評価
                        score *= 0.7
                    scores.append(score)
            except ValueError:
                continue

        if not scores:
            return None
        else:
            return round(sum(scores) / len(scores), 2)


    def lap_adjust(kaku, laps):
        delta = max(laps - 4, 0)
        return {
            '逃': round(-0.2 * delta, 2),
            '追': round(+0.1 * delta, 2),
            '両': 0.0
        }.get(kaku, 0.0)

    def line_member_bonus(pos):
        return {
            0: 0.5,  # 単騎
            1: 0.5,  # 先頭（ライン1番手）
            2: 0.6,  # 2番手（番手）
            3: 0.4,  # 3番手（最後尾）
            4: 0.3   # 4番手（9車用：評価不要レベル）
        }.get(pos, 0.0)


    def bank_character_bonus(kaku, angle, straight):
        """
        カント角と直線長による脚質補正（スケール緩和済み）
        """
        straight_factor = (straight - 40.0) / 10.0
        angle_factor = (angle - 25.0) / 5.0
        total_factor = -0.2 * straight_factor + 0.2 * angle_factor
        return round({'逃': +total_factor, '追': -total_factor, '両': +0.5 * total_factor}.get(kaku, 0.0), 2)
    
    
    def bank_length_adjust(kaku, length):
        """
        バンク周長による補正（400基準を完全維持しつつ、±0.15に制限）
        """
        delta = (length - 411) / 100
        delta = max(min(delta, 0.075), -0.075)
        return round({'逃': 2.0 * delta, '両': 4.0 * delta, '追': 6.0 * delta}.get(kaku, 0.0), 2)

    def compute_group_bonus(score_parts, line_def):
        # 初期化
        group_scores = {k: 0.0 for k in ['A', 'B', 'C']}
        group_counts = {k: 0 for k in ['A', 'B', 'C']}
    
        # 各ラインの合計スコアと人数を集計
        for entry in score_parts:
            car_no, score = entry[0], entry[-1]
            for group in ['A', 'B', 'C']:
                if car_no in line_def[group]:
                    group_scores[group] += score
                    group_counts[group] += 1
                    break
    
    
        # 合計スコアで順位を決定（平均ではない）
        sorted_lines = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    
        # 上位グループから順に 0.5 → 0.4 → 0.3 のボーナスを付与
        bonus_map = {group: [0.5, 0.4, 0.3][idx] for idx, (group, _) in enumerate(sorted_lines)}
    
        return bonus_map
        
    def get_group_bonus(car_no, line_def, group_bonus_map):
        for group in ['A', 'B', 'C']:
            if car_no in line_def[group]:
                base_bonus = group_bonus_map.get(group, 0.0)
                s_bonus = 0.3 if group == 'A' else 0.0  # ← 無条件でAだけに+0.3
                return base_bonus + s_bonus
        if '単騎' in line_def and car_no in line_def['単騎']:
            return 0.3
        return 0.0

    # ライン構成取得
    line_def = {
        'A': extract_car_list(a_line),
        'B': extract_car_list(b_line),
        'C': extract_car_list(c_line),
        '単騎': extract_car_list(solo_line)  # tanki → solo_line に合わせて
        }

    line_order_map = build_line_position_map()
    line_order = [line_order_map.get(i + 1, 0) for i in range(7)]

    
    # --- 1. 年齢補正の関数 ---
    def get_age_correction(age, base_age=35, step=0.03):
        return max(0.0, (age - base_age) * step)
    
    # --- 2. 級別係数の定義 ---
    correction_factor = {
        "チャレンジ": 1.2,
        "A級":        1.0,
        "S級":        0.6
    }

    # --- 3. 補正スコアの生成（7選手分） ---
    metabolism_scores = [
        get_age_correction(ages[i]) * correction_factor.get(race_class, 1.0)
        if isinstance(ages[i], (int, float)) else 0.0
        for i in range(7)
    ]


    # 2. スコア計算ループ
    tenscore_score = score_from_tenscore_list(rating)
    score_parts = []

    for i in range(7):
        try:
            if i >= len(tairetsu) or not tairetsu[i].isdigit():
                continue

            num = i + 1
            kaku = car_to_kakushitsu.get(num, "追")
            base = base_score[kaku]

            wind = wind_straight_combo_adjust(
                kaku,
                st.session_state.selected_wind,
                wind_speed,
                straight_length,
                line_order[i]
            )

            kasai = convert_chaku_to_score(chaku_inputs[i]) or 0.0
            rating_score = tenscore_score[i]
            rain_corr = lap_adjust(kaku, laps)
            s_bonus = 0.05 * st.session_state.get(f"s_point_{num}", 0)
            b_bonus = 0.05 * st.session_state.get(f"b_point_{num}", 0)
            symbol_score = s_bonus + b_bonus
            line_bonus = line_member_bonus(line_order[i])
            bank_bonus = bank_character_bonus(kaku, bank_angle, straight_length)
            length_bonus = bank_length_adjust(kaku, bank_length)
            meta_score = metabolism_scores[i]

            total = base + wind + kasai + rating_score + rain_corr + symbol_score + line_bonus + bank_bonus + length_bonus + meta_score

            score_parts.append([
                num, kaku, base, wind, kasai, rating_score,
                rain_corr, symbol_score, line_bonus, bank_bonus, length_bonus,
                meta_score,
                total
            ])
        except Exception as e:
            st.warning(f"{num}番のスコア計算エラー: {e}")

    # 3. グループ補正
    group_bonus_map = compute_group_bonus(score_parts, line_def)
    final_score_parts = []
    for row in score_parts:
        group_corr = get_group_bonus(row[0], line_def, group_bonus_map)
        new_total = row[-1] + group_corr
        final_score_parts.append(row[:-1] + [group_corr, new_total])

    # 4. 結果表示
    df = pd.DataFrame(final_score_parts, columns=[
        '車番', '脚質', '基本', '風補正', '着順補正', '得点補正',
        '周回補正', 'SB印補正', 'ライン補正', 'バンク補正', '周長補正',
        '代謝補正', 'グループ補正', '合計スコア'
    ])
    st.dataframe(df.sort_values(by='合計スコア', ascending=False).reset_index(drop=True))
