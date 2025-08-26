import streamlit as st
import pandas as pd

"""
ヴェロビ（欠車対応・統一版 完全版 / 男子・ガールズ分岐 + note出力）
- スコア計算は従来どおり（欠車対応・5〜9車・風/周長/開催日＝疲労補正 など）
- 印ルール（男子）：
  ◎：競争得点1〜4位の中でヴェロビ合計スコア最上位
  ○：A=◎同ラインの次点 vs B=競争得点1〜4位の次点（◎除外） → スコア高い方
  ▲：○で選ばれなかった方（A or B）
  ※◎が単騎のとき：まず○=B、その後「○基準」で A'=○同ライン次点 vs B'=競争得点1〜4位の次点（○除外） → ▲は高い方、△は残り
  △×αβ：残りをヴェロビ合計スコア順
- 印ルール（ガールズ＝全員単騎）：
  ◎：競争得点1〜4位の中でスコア最上位
  ○：◎以外の競争得点1〜4位の中でスコア最上位
  ▲以下：残りをヴェロビ合計スコア順
- 最後に note 記事向けの上下2行（ライン構成／印の最終順）を出力（コピー可）
"""

# =========================================================
# 定数・共通テーブル
# =========================================================

WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035
}
POS_MULTI = {0: 0.3, 1: 0.32, 2: 0.30, 3: 0.25, 4: 0.20}  # 8-9車時は動的拡張
BASE_SCORE = {'逃': 1.577, '両': 1.628, '追': 1.796}

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

DAY_DELTA = {1: 1, 2: 2, 3: 3}  # 開催日＝疲労補正（＋方向固定）

# =========================================================
# 補助関数（スコア計算）
# =========================================================

def convert_chaku_to_score(values):
    scores = []
    for i, v in enumerate(values):
        v = str(v).strip()
        try:
            chaku = int(v)
            if 1 <= chaku <= 9:
                score = (10 - chaku) / 9
                if i == 1:
                    score *= 0.35
                scores.append(score)
        except ValueError:
            continue
    return round(sum(scores) / len(scores), 2) if scores else 0.0

def wind_straight_combo_adjust(kakushitsu, wind_direction, wind_speed, straight_length, line_order, pos_multi_map):
    if wind_direction == "無風" or wind_speed == 0:
        return 0.0
    wind_adj = WIND_COEFF.get(wind_direction, 0.0)
    pos_multi = pos_multi_map.get(line_order, 0.3)
    coeff = {'逃': 1.0, '両': 0.7, '追': 0.4}.get(kakushitsu, 0.5)
    total = wind_speed * wind_adj * coeff * pos_multi
    return round(max(min(total, 0.05), -0.05), 3)

def lap_adjust(kaku, laps):
    delta = max(int(laps) - 2, 0)
    return {'逃': round(-0.1 * delta, 1), '追': round(+0.05 * delta, 1), '両': 0.0}.get(kaku, 0.0)

def line_member_bonus(line_order, bonus_map):
    return bonus_map.get(line_order, 0.0)

def bank_character_bonus(kakushitsu, bank_angle, straight_length):
    straight_factor = (float(straight_length) - 40.0) / 10.0
    angle_factor = (float(bank_angle) - 25.0) / 5.0
    total_factor = -0.1 * straight_factor + 0.1 * angle_factor
    total_factor = max(min(total_factor, 0.05), -0.05)
    return round({'逃': +total_factor, '追': -total_factor, '両': +0.25 * total_factor}.get(kakushitsu, 0.0), 2)

def bank_length_adjust(kakushitsu, bank_length):
    delta = (float(bank_length) - 411.0) / 100.0
    delta = max(min(delta, 0.05), -0.05)
    return round({'逃': 1.0 * delta, '両': 2.0 * delta, '追': 3.0 * delta}.get(kakushitsu, 0.0), 2)

def extract_car_list(input_data):
    if isinstance(input_data, str):
        return [int(c) for c in input_data if c.isdigit()]
    elif isinstance(input_data, list):
        return [int(c) for c in input_data if isinstance(c, (str, int)) and str(c).isdigit()]
    return []

def build_line_position_map(lines):
    line_order_map = {}
    for line in lines:
        if not line: continue
        if len(line) == 1:
            line_order_map[line[0]] = 0
        else:
            for pos, car in enumerate(line, start=1):
                line_order_map[car] = pos
    return line_order_map

def score_from_tenscore_list_dynamic(tenscore_list, upper_k=8):
    n_local = len(tenscore_list)
    if n_local <= 2:
        return [0.0] * n_local
    df = pd.DataFrame({"得点": tenscore_list})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n_local, int(upper_k))
    baseline = df[df["順位"].between(2, hi)]["得点"].mean()
    def corr(row):
        return round(abs(baseline - row["得点"]) * 0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return (df.apply(corr, axis=1)).tolist()

def compute_group_bonus(score_parts, line_def, n):
    if not line_def: return {}
    alpha = 0.0 if n <= 7 else (0.25 if n == 8 else 0.5)
    total_budget = 0.42 * ((max(n,1) / 7.0) ** 0.5)
    car_to_group = {car: g for g, members in line_def.items() for car in members}
    sums, sizes = {}, {}
    for g, members in line_def.items():
        sums[g], sizes[g] = 0.0, max(len(members), 1)
    for row in score_parts:
        car_no, total = row[0], row[-1]
        g = car_to_group.get(car_no)
        if g: sums[g] += total
    adj = {g: (sums[g] / (sizes[g] ** alpha)) for g in line_def.keys()}
    sorted_lines = sorted(adj.items(), key=lambda x: x[1], reverse=True)
    r = 0.80
    weights = [r**i for i in range(len(sorted_lines))]
    sw = sum(weights) if weights else 1.0
    bonuses = [(w / sw) * total_budget for w in weights]
    return {g: bonuses[i] for i, (g, _) in enumerate(sorted_lines)}

def get_group_bonus(car_no, line_def, bonus_map, a_head_bonus=True):
    for g, members in line_def.items():
        if car_no in members:
            add = 0.15 if (a_head_bonus and g == 'A') else 0.0
            return bonus_map.get(g, 0.0) + add
    return 0.0

# =========================================================
# 印選定補助（ここが今回の新ロジック）
# =========================================================

def pick_anchor(velobi_sorted, comp_points_rank):
    for no, sc in velobi_sorted:
        if comp_points_rank.get(no, 99) <= 4:
            return no, sc
    return velobi_sorted[0]  # フォールバック

def pick_A_B_for_anchor(anchor_no, velobi_sorted, comp_points_rank, car_to_group):
    # A: ◎同ラインの次点
    A = None
    anchor_group = car_to_group.get(anchor_no, None)
    for no, sc in velobi_sorted:
        if no == anchor_no: continue
        if anchor_group and car_to_group.get(no, None) == anchor_group:
            A = (no, sc, "同ライン")
            break
    # B: 競争得点1〜4位の次点（◎除外）
    B = None
    for no, sc in velobi_sorted:
        if no == anchor_no: continue
        if comp_points_rank.get(no, 99) <= 4:
            B = (no, sc, "得点上位")
            break
    return A, B

def pick_for_single_anchor(anchor_no, velobi_sorted, comp_points_rank, car_to_group):
    # ◎が単騎のとき：まず○=B（得点1〜4位次点）
    O = None
    for no, sc in velobi_sorted:
        if no == anchor_no: continue
        if comp_points_rank.get(no, 99) <= 4:
            O = (no, sc, "得点上位")
            break
    if not O:
        return None, None, None  # ○も置けないケース（まれ）
    # ○基準で A'/B' 比較 → ▲/△
    o_no, o_sc, _ = O
    o_group = car_to_group.get(o_no, None)
    A2 = B2 = None
    for no, sc in velobi_sorted:
        if no in [anchor_no, o_no]: continue
        if o_group and car_to_group.get(no, None) == o_group and not A2:
            A2 = (no, sc, "○同ライン")
        if comp_points_rank.get(no, 99) <= 4 and not B2:
            B2 = (no, sc, "得点上位")
    # ▲/△
    if A2 and B2:
        if A2[1] >= B2[1]:
            return O, A2, B2
        else:
            return O, B2, A2
    elif A2:
        return O, A2, None
    elif B2:
        return O, B2, None
    else:
        return O, None, None

def pick_girls_anchor_second(velobi_sorted, comp_points_rank):
    anchor = second = None
    for no, sc in velobi_sorted:
        if comp_points_rank.get(no, 99) <= 4:
            if not anchor:
                anchor = (no, sc, "本命")
            elif not second:
                second = (no, sc, "対抗")
                break
    return anchor, second

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="ヴェロビ 完全版（note記事用出力つき）", layout="wide")
st.title("⭐ ヴェロビ 完全版（note記事用出力つき）⭐")

mode = st.radio("開催種別を選択", ["男子", "ガールズ"], horizontal=True)

# 風・バンク
if "selected_wind" not in st.session_state:
    st.session_state.selected_wind = "無風"

st.header("【バンク・風条件】")
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("左上"): st.session_state.selected_wind = "左上"
with c2:
    if st.button("上"): st.session_state.selected_wind = "上"
with c3:
    if st.button("右上"): st.session_state.selected_wind = "右上"
c4,c5,c6 = st.columns(3)
with c4:
    if st.button("左"): st.session_state.selected_wind = "左"
with c5:
    st.write(f"✅ 風向：{st.session_state.selected_wind}")
with c6:
    if st.button("右"): st.session_state.selected_wind = "右"
c7,c8,c9 = st.columns(3)
with c7:
    if st.button("左下"): st.session_state.selected_wind = "左下"
with c8:
    if st.button("下"): st.session_state.selected_wind = "下"
with c9:
    if st.button("右下"): st.session_state.selected_wind = "右下"

selected_track = st.selectbox("競輪場（自動プリセット）", list(KEIRIN_DATA.keys()))
info = KEIRIN_DATA[selected_track]
wind_speed = st.number_input("風速(m/s)", 0.0, 30.0, 3.0, 0.1)
straight_length = st.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle = st.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length = st.number_input("バンク周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)

# 周回・開催日（疲労）
base_laps = st.number_input("周回数（通常4、高松など5）", 1, 10, 4, 1)
day_label_to_idx = {"初日":1, "2日目":2, "最終日":3}
day_label = st.selectbox("開催日（疲労補正：初日+1 / 2日目+2 / 最終日+3）", list(day_label_to_idx.keys()))
day_idx = day_label_to_idx[day_label]
eff_laps = int(base_laps) + DAY_DELTA.get(day_idx, 1)

# 入力
N_MAX = 9
st.header("【選手データ入力】")
st.subheader("▼ 位置（脚質）：逃＝先頭／両＝番手／追＝3番手以降＆単騎（車番を半角数字で入力）")
kakushitsu_inputs = {}
c = st.columns(3)
for i, k in enumerate(['逃','両','追']):
    with c[i]:
        kakushitsu_inputs[k] = st.text_input(f"{k}", key=f"kaku_{k}", max_chars=14)

car_to_kakushitsu = {}
for k, val in kakushitsu_inputs.items():
    for ch in val:
        if ch.isdigit():
            n = int(ch)
            if 1 <= n <= 9:
                car_to_kakushitsu[n] = k

st.subheader("▼ 前々走・前走の着順（1〜9、0=落車 可）")
chaku_inputs = []
for i in range(N_MAX):
    col1,col2 = st.columns(2)
    with col1: ch1 = st.text_input(f"{i+1}番【前々走】", key=f"ch1_{i}")
    with col2: ch2 = st.text_input(f"{i+1}番【前走】", key=f"ch2_{i}")
    chaku_inputs.append([ch1,ch2])

st.subheader("▼ 競争得点")
rating = [st.number_input(f"{i+1}番得点", value=55.0, step=0.1, key=f"rate_{i}") for i in range(N_MAX)]

st.subheader("▼ 予想隊列（数字、欠は空欄）")
tairetsu = [st.text_input(f"{i+1}番隊列順位", key=f"tai_{i}") for i in range(N_MAX)]

st.subheader("▼ S・B 回数")
for i in range(N_MAX):
    st.number_input(f"{i+1}番 S回数", 0, 99, 0, key=f"s_{i+1}")
    st.number_input(f"{i+1}番 B回数", 0, 99, 0, key=f"b_{i+1}")

st.subheader("▼ ライン構成（最大7：単騎も1ライン）")
line_inputs = [
    st.text_input("ライン1（例：4）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：12）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：35）", key="line_3", max_chars=9),
    st.text_input("ライン4（例：7）", key="line_4", max_chars=9),
    st.text_input("ライン5（例：6）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
]
lines = [extract_car_list(x) for x in line_inputs if str(x).strip()]
line_order_map = build_line_position_map(lines)
line_order = [line_order_map.get(i+1, 0) for i in range(N_MAX)]

# 人数と動的パラメータ
active_idx = [i for i in range(N_MAX) if str(tairetsu[i]).isdigit()]
n_cars = len(active_idx)

def choose_upper_k(n:int)->int:
    if n <= 3: return 0
    if n == 4: return 4
    if n == 5: return 5
    if n == 6: return 6
    if n == 7: return 6
    return 8

def dynamic_params(n:int):
    if n <= 7:
        line_bonus = {0:0.03, 1:0.05, 2:0.04, 3:0.03}
        pos_multi_map = {0:0.30, 1:0.32, 2:0.30, 3:0.25, 4:0.20}
    else:
        line_bonus = {0:0.03, 1:0.05, 2:0.04, 3:0.03, 4:0.02, 5:0.015}
        pos_multi_map = {0:0.30, 1:0.32, 2:0.30, 3:0.25, 4:0.20, 5:0.18}
    return line_bonus, pos_multi_map, choose_upper_k(n)

LINE_BONUS, POS_MULTI_MAP, UPPER_K = dynamic_params(n_cars)

# スコア計算（activeのみ）
ratings_active = [rating[i] for i in active_idx]
corr_active = score_from_tenscore_list_dynamic(ratings_active, upper_k=UPPER_K)
tenscore_score = [0.0] * N_MAX
for j,k in enumerate(active_idx):
    tenscore_score[k] = corr_active[j]

score_parts = []
for i in active_idx:
    num = i+1
    kaku = car_to_kakushitsu.get(num, "追")
    base = BASE_SCORE.get(kaku, 0.0)
    wind = wind_straight_combo_adjust(kaku, st.session_state.selected_wind, wind_speed, straight_length, line_order[i], POS_MULTI_MAP)
    kasai = convert_chaku_to_score(chaku_inputs[i]) or 0.0
    rating_score = tenscore_score[i]
    rain_corr = lap_adjust(kaku, eff_laps)
    s_bonus = min(0.1*st.session_state.get(f"s_{num}",0), 0.5)
    b_bonus = min(0.1*st.session_state.get(f"b_{num}",0), 0.5)
    symbol_score = s_bonus + b_bonus
    line_b = line_member_bonus(line_order[i], LINE_BONUS)
    bank_b = bank_character_bonus(kaku, bank_angle, straight_length)
    length_b = bank_length_adjust(kaku, bank_length)
    total = base + wind + kasai + rating_score + rain_corr + symbol_score + line_b + bank_b + length_b
    score_parts.append([num, kaku, base, wind, kasai, rating_score, rain_corr, symbol_score, line_b, bank_b, length_b, total])

labels = ["A","B","C","D","E","F","G"]
line_def = {labels[idx]: line for idx, line in enumerate(lines) if line}
car_to_group = {car: g for g, members in line_def.items() for car in members}

group_bonus_map = compute_group_bonus(score_parts, line_def, n_cars)
final_score_parts = []
for row in score_parts:
    group_corr = get_group_bonus(row[0], line_def, group_bonus_map, a_head_bonus=True)
    final_score_parts.append(row[:-1] + [group_corr, row[-1] + group_corr])

columns = ['車番','脚質','基本','風補正','着順補正','得点補正','周回補正','SB印補正','ライン補正','バンク補正','周長補正','グループ補正','合計スコア']
df = pd.DataFrame(final_score_parts, columns=columns)

# 競争得点の列を併記
try:
    if len(rating) >= len(df):
        rating_map = {i+1: rating[i] for i in range(N_MAX)}
        df['競争得点'] = df['車番'].map(rating_map)
except Exception:
    pass

# =========================================================
# 印決定（男子/ガールズ分岐）
# =========================================================
st.markdown("### 📊 合計スコア順（印・スコア・競争得点・理由）")
if df.empty:
    st.error("データが空です。入力を確認してください。")
else:
    df_rank = df.sort_values(by='合計スコア', ascending=False).reset_index(drop=True)
    velobi_sorted = list(zip(df_rank['車番'].tolist(), df_rank['合計スコア'].round(1).tolist()))

    # 競争得点順位（active対象）
    points_df = pd.DataFrame({"車番":[i+1 for i in active_idx], "得点":[rating[i] for i in active_idx]})
    if not points_df.empty:
        points_df["順位"] = points_df["得点"].rank(ascending=False, method="min").astype(int)
        comp_points_rank = dict(zip(points_df["車番"], points_df["順位"]))
    else:
        comp_points_rank = {}

    marks_order = ["◎","〇","▲","△","×","α","β"]
    result_marks = {}
    reasons = {}

    if mode == "ガールズ":
        # ◎/○ は得点1-4縛り、以降はスコア順
        a, b = pick_girls_anchor_second(velobi_sorted, comp_points_rank)
        if a:
            result_marks["◎"] = a[0]; reasons[a[0]] = "本命(得点1-4)"
        if b:
            result_marks["〇"] = b[0]; reasons[b[0]] = "対抗(得点1-4)"
        used = set(result_marks.values())
        rest = [no for no,_ in velobi_sorted if no not in used]
        fill_marks = [m for m in marks_order if m not in result_marks]
        for m, n in zip(fill_marks, rest):
            result_marks[m] = n
    else:
        # 男子（ラインあり／単騎あり）
        anchor_no, _ = pick_anchor(velobi_sorted, comp_points_rank)
        result_marks["◎"] = anchor_no; reasons[anchor_no] = "本命(得点1-4内最高スコア)"
        anchor_group = car_to_group.get(anchor_no, None)
        same_line_exists = anchor_group and any((car_to_group.get(no)==anchor_group and no!=anchor_no) for no,_ in velobi_sorted)

        if same_line_exists:
            A,B = pick_A_B_for_anchor(anchor_no, velobi_sorted, comp_points_rank, car_to_group)
            # ○/▲ 決定
            if A and B:
                if A[1] >= B[1]:
                    result_marks["〇"] = A[0]; reasons[A[0]] = "同ライン"
                    result_marks["▲"] = B[0]; reasons[B[0]] = "得点上位"
                else:
                    result_marks["〇"] = B[0]; reasons[B[0]] = "得点上位"
                    result_marks["▲"] = A[0]; reasons[A[0]] = "同ライン"
            elif A:
                result_marks["〇"] = A[0]; reasons[A[0]] = "同ライン"
            elif B:
                result_marks["〇"] = B[0]; reasons[B[0]] = "得点上位"
        else:
            # ◎単騎：○=B、その後○基準で A'/B' 比較 → ▲/△
            O, A2, B2 = pick_for_single_anchor(anchor_no, velobi_sorted, comp_points_rank, car_to_group)
            if O:
                result_marks["〇"] = O[0]; reasons[O[0]] = "得点上位"
            if A2 and B2:
                # スコア高い方を▲、残りを△
                if A2[1] >= B2[1]:
                    result_marks["▲"] = A2[0]; reasons[A2[0]] = A2[2]
                    result_marks["△"] = B2[0]; reasons[B2[0]] = B2[2]
                else:
                    result_marks["▲"] = B2[0]; reasons[B2[0]] = B2[2]
                    result_marks["△"] = A2[0]; reasons[A2[0]] = A2[2]
            elif A2:
                result_marks["▲"] = A2[0]; reasons[A2[0]] = A2[2]
            elif B2:
                result_marks["▲"] = B2[0]; reasons[B2[0]] = B2[2]

        # 残りをスコア順で補完
        used = set(result_marks.values())
        rest = [no for no,_ in velobi_sorted if no not in used]
        for m,n in zip([m for m in marks_order if m not in result_marks], rest):
            result_marks[m] = n

    # 表示（印入りランキング＋詳細内訳）
    rows = []
    for r,(no,sc) in enumerate(velobi_sorted, start=1):
        mark = [m for m,v in result_marks.items() if v==no]
        reason = reasons.get(no,"")
        pt = df.loc[df['車番']==no, '競争得点'].iloc[0] if '競争得点' in df.columns else None
        rows.append({"順":r,"印":"".join(mark),"車":no,"合計スコア":sc,"競争得点":pt,"理由":reason})
    view_df = pd.DataFrame(rows)
    st.dataframe(view_df, use_container_width=True)

    st.markdown("### 🧩 補正内訳（合計スコア高い順）")
    st.dataframe(df_rank, use_container_width=True)

    # タグ表示
    tag = f"開催日補正 +{DAY_DELTA.get(day_idx,1)}（有効周回={eff_laps}） / 風向:{st.session_state.selected_wind}"
    st.caption(tag)

    # =========================================================
    # ✅ note記事用（上下2行）— 必ず最後に表示
    # =========================================================
    st.markdown("### 📋 note記事用（コピー可 / 上下2行）")
    line_text = "　".join([x for x in line_inputs if str(x).strip()])
    marks_line = " ".join([f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks])
    note_text = f"ライン　{line_text}\n{marks_line}"
    st.text_area("note貼り付け用（この枠の内容をそのままコピー）", note_text, height=90)
