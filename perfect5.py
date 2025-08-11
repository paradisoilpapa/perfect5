import streamlit as st
import pandas as pd

"""
ヴェロビ（欠車対応・統一版perfect5ver）
- 目的：7車UIを維持しつつ、欠車（隊列空欄）でも安全に計算が通るように全面整理
- 主な変更点：
  1) active_idx（有効車番）で全ループを駆動（range(7)固定の解消）
  2) 競争得点補正のbaselineレンジを動的化（2〜min(n, upper_k)）
  3) グループ補正は line_def の実在キーのみ、ボーナス配列は本数に合わせてスライス
  4) 重複関数の排除・定義を1本化
  5) 安全ガード（NaN/0除去、列存在チェック）

※ 地方競馬転用時は、ライン→通過順/枠・馬場/指数へ写像予定
"""

# =========================================================
# 定数・共通テーブル
# =========================================================

WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035
}

# ライン順：0=単騎, 1=先頭, 2=番手, 3=3番手, 4=4番手
POS_MULTI = {0: 0.3, 1: 0.32, 2: 0.30, 3: 0.25, 4: 0.20}

# 脚質基準値（KAPP3ベース）
BASE_SCORE = {'逃': 1.577, '両': 1.628, '追': 1.796}

# 競輪場プリセット
KEIRIN_DATA = {
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
    "手入力": {"bank_angle": 30.0, "straight_length": 52.0, "bank_length": 400},
}

# =========================================================
# 補助関数
# =========================================================

def convert_chaku_to_score(values):
    """前々走/前走の着順を[0..1]に正規化し平均。欠は無視。"""
    scores = []
    for i, v in enumerate(values):
        v = str(v).strip()
        try:
            chaku = int(v)
            if 1 <= chaku <= 9:
                score = (10 - chaku) / 9
                if i == 1:
                    score *= 0.35  # 前走比重（現行仕様踏襲）
                scores.append(score)
        except ValueError:
            continue
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def wind_straight_combo_adjust(kakushitsu, wind_direction, wind_speed, straight_length, line_order):
    """風×ライン順の補正（±0.05制限／係数:逃1.0 両0.7 追0.4）"""
    if wind_direction == "無風" or wind_speed == 0:
        return 0.0
    wind_adj = WIND_COEFF.get(wind_direction, 0.0)
    pos_multi = POS_MULTI.get(line_order, 0.3)
    coeff = {'逃': 1.0, '両': 0.7, '追': 0.4}.get(kakushitsu, 0.5)
    total = wind_speed * wind_adj * coeff * pos_multi
    total = max(min(total, 0.05), -0.05)
    return round(total, 3)


def lap_adjust(kaku, laps):
    delta = max(int(laps) - 2, 0)
    return {
        '逃': round(-0.1 * delta, 1),
        '追': round(+0.05 * delta, 1),
        '両': 0.0
    }.get(kaku, 0.0)


def line_member_bonus(line_order):
    """ライン位置補正（0:単騎, 1:先頭, 2:番手, 3:3番手）"""
    return {0: 0.03, 1: 0.05, 2: 0.04, 3: 0.03}.get(line_order, 0.0)


def bank_character_bonus(kakushitsu, bank_angle, straight_length):
    """バンク性格補正（±0.05）"""
    straight_factor = (float(straight_length) - 40.0) / 10.0
    angle_factor = (float(bank_angle) - 25.0) / 5.0
    total_factor = -0.1 * straight_factor + 0.1 * angle_factor
    total_factor = max(min(total_factor, 0.05), -0.05)
    return round({'逃': +total_factor, '追': -total_factor, '両': +0.25 * total_factor}.get(kakushitsu, 0.0), 2)


def bank_length_adjust(kakushitsu, bank_length):
    """周長補正（±0.05）"""
    delta = (float(bank_length) - 411.0) / 100.0
    delta = max(min(delta, 0.05), -0.05)  # 強制制限
    return round({'逃': 1.0 * delta, '両': 2.0 * delta, '追': 3.0 * delta}.get(kakushitsu, 0.0), 2)


def extract_car_list(input_data):
    if isinstance(input_data, str):
        return [int(c) for c in input_data if c.isdigit()]
    elif isinstance(input_data, list):
        return [int(c) for c in input_data if isinstance(c, (str, int)) and str(c).isdigit()]
    else:
        return []


def build_line_position_map(lines):
    """各車番→(ライン内の順番: 1.. / 単騎:0) を返す"""
    line_order_map = {}
    for idx, line in enumerate(lines):
        if not line:
            continue
        if len(line) == 1:  # 単騎
            line_order_map[line[0]] = 0
        else:
            for pos, car in enumerate(line, start=1):
                line_order_map[car] = pos
    return line_order_map


def score_from_tenscore_list_dynamic(tenscore_list, upper_k=8):
    """競争得点補正：2〜min(n, upper_k)の平均を基準に、2〜4位へ差分×3%加点。
    欠車時も安全。n<=2 は全員0。
    """
    n_local = len(tenscore_list)
    if n_local <= 2:
        return [0.0] * n_local
    df = pd.DataFrame({"得点": tenscore_list})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n_local, int(upper_k))
    baseline = df[df["順位"].between(2, hi)]["得点"].mean()
    def corr(row):
        return round(abs(baseline - row["得点"]) * 0.03, 3) if row["順位"] in [2, 3, 4] else 0.0
    return (df.apply(corr, axis=1)).tolist()


def compute_group_bonus(score_parts, line_def):
    """ライン合計スコアで順位→ボーナス値を割当。実在キーのみ対象。"""
    if not line_def:
        return {}
    # 車番→グループ逆引き
    car_to_group = {car: g for g, members in line_def.items() for car in members}
    # 合計集計
    group_scores = {g: 0.0 for g in line_def}
    for row in score_parts:
        car_no, total = row[0], row[-1]
        g = car_to_group.get(car_no)
        if g:
            group_scores[g] += total
    # 順位化
    sorted_lines = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    base_vals = [0.125, 0.10, 0.075, 0.05, 0.04, 0.02, 0.01][:len(sorted_lines)]
    return {g: base_vals[idx] for idx, (g, _) in enumerate(sorted_lines)}


def get_group_bonus(car_no, line_def, bonus_map, a_head_bonus=True):
    for g, members in line_def.items():
        if car_no in members:
            add = 0.15 if (a_head_bonus and g == 'A') else 0.0
            return bonus_map.get(g, 0.0) + add
    return 0.0

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="ライン競輪スコア計算（欠車対応・統一版KAPP3）", layout="wide")
st.title("⭐ ライン競輪スコア計算（欠車対応・統一版KAPP3）⭐")

# 風向選択（ボタン）
if "selected_wind" not in st.session_state:
    st.session_state.selected_wind = "無風"

st.header("【バンク・風条件】")
cols_top = st.columns(3)
cols_mid = st.columns(3)
cols_bot = st.columns(3)
with cols_top[0]:
    if st.button("左上"): st.session_state.selected_wind = "左上"
with cols_top[1]:
    if st.button("上"): st.session_state.selected_wind = "上"
with cols_top[2]:
    if st.button("右上"): st.session_state.selected_wind = "右上"
with cols_mid[0]:
    if st.button("左"): st.session_state.selected_wind = "左"
with cols_mid[1]:
    st.markdown(
        """
        <div style='text-align:center; font-size:16px; line-height:1.6em;'>
            ↑<br>［上］<br>
            ← 左　　　右 →<br>
            ［下］<br>↓<br>
            □ ホーム→（ ゴール）
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols_mid[2]:
    if st.button("右"): st.session_state.selected_wind = "右"
with cols_bot[0]:
    if st.button("左下"): st.session_state.selected_wind = "左下"
with cols_bot[1]:
    if st.button("下"): st.session_state.selected_wind = "下"
with cols_bot[2]:
    if st.button("右下"): st.session_state.selected_wind = "右下"

st.subheader(f"✅ 選択中の風向き：{st.session_state.selected_wind}")

# 競輪場選択
selected_track = st.selectbox("▼ 競輪場選択（自動入力）", list(KEIRIN_DATA.keys()))
info = KEIRIN_DATA[selected_track]

# 風速・コース諸元
wind_speed = st.number_input("風速(m/s)", min_value=0.0, max_value=30.0, step=0.1, value=3.0)
straight_length = st.number_input("みなし直線(m)", min_value=30.0, max_value=80.0, step=0.05, value=float(info["straight_length"]))
bank_angle = st.number_input("バンク角(°)", min_value=20.0, max_value=45.0, step=0.05, value=float(info["bank_angle"]))
bank_length = st.number_input("バンク周長(m)", min_value=300.0, max_value=500.0, step=0.05, value=float(info["bank_length"]))

# 周回数
laps = st.number_input("周回数（通常は4、高松などは5）", min_value=1, max_value=10, value=4, step=1)

# 選手データ入力
st.header("【選手データ入力】")
st.subheader("▼ 位置入力（逃＝先頭・両＝番手・追＝３番手以降&単騎：車番を半角数字で入力）")

kakushitsu_inputs = {}
cols = st.columns(3)
for i, k in enumerate(['逃', '両', '追']):
    with cols[i]:
        st.markdown(f"**{k}**")
        kakushitsu_inputs[k] = st.text_input("", key=f"kaku_{k}", max_chars=14)

# 車番→脚質
car_to_kakushitsu = {}
for k, val in kakushitsu_inputs.items():
    for c in val:
        if c.isdigit():
            n = int(c)
            if 1 <= n <= 9:
                car_to_kakushitsu[n] = k

st.subheader("▼ 前々走・前走の着順入力（1〜9着 または 0＝落車）")
chaku_inputs = []
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

st.subheader("▼ S・B 入力（各選手のS・B回数を入力）")
for i in range(7):
    st.markdown(f"**{i+1}番**")
    st.number_input("S回数", min_value=0, max_value=99, value=0, step=1, key=f"s_point_{i+1}")
    st.number_input("B回数", min_value=0, max_value=99, value=0, step=1, key=f"b_point_{i+1}")

st.subheader("▼ ライン構成入力（最大7ライン：単騎も1ラインとして扱う）")
line_inputs = [
    st.text_input("ライン1（例：4）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：12）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：35）", key="line_3", max_chars=9),
    st.text_input("ライン4（例：7）", key="line_4", max_chars=9),
    st.text_input("ライン5（例：6）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
]

# ライン配列
lines = [extract_car_list(x) for x in line_inputs if str(x).strip()]
line_order_map = build_line_position_map(lines)
line_order = [line_order_map.get(i + 1, 0) for i in range(7)]

# 有効車番のみ採用
active_idx = [i for i in range(7) if str(tairetsu[i]).isdigit()]

# 競争得点補正（有効車番だけ計算→7枠に戻す）
ratings_active = [rating[i] for i in active_idx]
# upper_k は 6/8 のいずれかで好みに合わせて調整可
corr_active = score_from_tenscore_list_dynamic(ratings_active, upper_k=8)

tenscore_score = [0.0] * 7
for j, k in enumerate(active_idx):
    tenscore_score[k] = corr_active[j]

# スコア計算本体（activeのみ）
score_parts = []
for i in active_idx:
    num = i + 1
    kaku = car_to_kakushitsu.get(num, "追")
    base = BASE_SCORE.get(kaku, 0.0)

    wind = wind_straight_combo_adjust(
        kaku, st.session_state.selected_wind, wind_speed, straight_length, line_order[i]
    )
    kasai = convert_chaku_to_score(chaku_inputs[i]) or 0.0
    rating_score = tenscore_score[i]
    rain_corr = lap_adjust(kaku, laps)
    s_bonus = min(0.1 * st.session_state.get(f"s_point_{num}", 0), 0.5)
    b_bonus = min(0.1 * st.session_state.get(f"b_point_{num}", 0), 0.5)
    symbol_score = s_bonus + b_bonus
    line_bonus = line_member_bonus(line_order[i])
    bank_bonus = bank_character_bonus(kaku, bank_angle, straight_length)
    length_bonus = bank_length_adjust(kaku, bank_length)

    total = base + wind + kasai + rating_score + rain_corr + symbol_score + line_bonus + bank_bonus + length_bonus

    score_parts.append([
        num, kaku, base, wind, kasai, rating_score, rain_corr, symbol_score, line_bonus, bank_bonus, length_bonus, total
    ])

# line_def 構築（存在するものだけ）
labels = ["A", "B", "C", "D", "E", "F", "G"]
line_def = {labels[idx]: line for idx, line in enumerate(lines) if line}

# グループ補正
group_bonus_map = compute_group_bonus(score_parts, line_def)
final_score_parts = []
for row in score_parts:
    group_corr = get_group_bonus(row[0], line_def, group_bonus_map, a_head_bonus=True)
    final_score_parts.append(row[:-1] + [group_corr, row[-1] + group_corr])

# DataFrame 化
columns = ['車番', '脚質', '基本', '風補正', '着順補正', '得点補正', '周回補正', 'SB印補正', 'ライン補正', 'バンク補正', '周長補正', 'グループ補正', '合計スコア']
df = pd.DataFrame(final_score_parts, columns=columns)

# 競争得点の列（元の素点）を併記したい場合
try:
    if len(rating) >= len(df):
        # df の車番に合わせて並べる
        rating_map = {i + 1: rating[i] for i in range(7)}
        df['競争得点'] = df['車番'].map(rating_map)
except Exception:
    pass

# 表示とエラーチェック
st.markdown("### 📊 合計スコア順スコア表（欠車対応）")
if df.empty:
    st.error("データが空です。入力を確認してください。")
else:
    st.dataframe(df.sort_values(by='合計スコア', ascending=False).reset_index(drop=True))
