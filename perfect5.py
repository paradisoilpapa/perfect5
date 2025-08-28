# app.py
# ヴェロビ 完全版（5〜9車対応 / 2連対率・3連対率 / 欠車対応 / 男子・ガールズ分岐）
# ◎：Δ≤5母集団のスコア首位
# ○▲：同ライン最上位と他ライン最上位を直接比較して上位を○、もう一方を▲
#      かつ「○が同ラインなら▲は他ライン」「○が他ラインなら▲は同ライン」を徹底
# note出力：3行のみ（手動コピー）

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import re, unicodedata

st.set_page_config(page_title="ヴェロビ 完全版（5〜9車対応）", layout="wide")

"""
ヴェロビ（欠車対応・統一版 / 5〜9車立て対応 / 男子・ガールズ分岐 + note出力）
— 前走/前々走の着順入力を廃止し、2連対率・3連対率で“着内実力”を反映 —
— 男子：◎はΔ≤5pt母集団のスコア首位、○▲は同ライン最上位と他ライン最上位を直接比較し、
          「○が同ラインなら▲は他ライン／○が他ラインなら▲は同ライン」を徹底 —
"""

# =========================================================
# 定数
# =========================================================
WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035
}
BASE_SCORE = {'逃': 1.577, '両': 1.628, '追': 1.796}
DAY_DELTA = {1: 1, 2: 2, 3: 3}

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
def _parse_float_flexible(s: str) -> float | None:
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
        return None
    try:
        return float(s)
    except Exception:
        return None

def _parse_percent_flexible(s: str) -> float:
    """ '7', '12.5', '７', '12.5%' 等を受け取り 0.0〜1.0 に正規化 """
    if s is None:
        return 0.0
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = t.replace("％", "%").replace(",", "")
    if t.endswith("%"):
        t = t[:-1].strip()
    if not re.fullmatch(r"\d+(\.\d+)?", t):
        return 0.0
    v = float(t)
    if v < 0: v = 0.0
    if v > 100: v = 100.0
    return v / 100.0

def _zscore_clip(s, clip=2.5):
    s = pd.Series(s).astype(float)
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - m) / sd).clip(-clip, clip)

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
            line_order_map[line[0]] = 0  # 単騎=先頭扱い
        else:
            for pos, car in enumerate(line, start=1):
                line_order_map[car] = pos
    return line_order_map

def wind_straight_combo_adjust(kakushitsu, wind_direction, wind_speed, straight_length, line_order, pos_multi_map):
    if wind_direction == "無風" or wind_speed == 0:
        return 0.0
    wind_adj = WIND_COEFF.get(wind_direction, 0.0)
    pos_multi = pos_multi_map.get(line_order, 0.30)
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

def dynamic_params(n:int):
    if n <= 7:
        line_bonus = {0:0.03, 1:0.05, 2:0.04, 3:0.03, 4:0.02}
        pos_multi_map = {0:0.30, 1:0.32, 2:0.30, 3:0.25, 4:0.20}
        upper_k = 6 if n >= 6 else n
    else:
        line_bonus = {0:0.03, 1:0.05, 2:0.04, 3:0.03, 4:0.02, 5:0.015}
        pos_multi_map = {0:0.30, 1:0.32, 2:0.30, 3:0.25, 4:0.20, 5:0.18}
        upper_k = 8
    return line_bonus, pos_multi_map, upper_k

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
# UI
# =========================================================
st.title("⭐ ヴェロビ 完全版（5〜9車対応 / note記事用）⭐")
mode = st.radio("開催種別を選択", ["男子", "ガールズ"], horizontal=True)
N_MAX = st.slider("出走車数（5〜9）", 5, 9, 7, 1)

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
    st.write(f"✅ 風向：{st.session_state.get('selected_wind','無風')}")
with c6:
    if st.button("右"): st.session_state.selected_wind = "右"
c7,c8,c9 = st.columns(3)
with c7:
    if st.button("左下"): st.session_state.selected_wind = "左下"
with c8:
    if st.button("下"): st.session_state.selected_wind = "下"
with c9:
    if st.button("右下"): st.session_state.selected_wind = "右下"

selected_track = st.selectbox("競輪場（プリセット）", list(KEIRIN_DATA.keys()))
info = KEIRIN_DATA[selected_track]
wind_speed = st.number_input("風速(m/s)", 0.0, 30.0, 3.0, 0.1)
straight_length = st.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle = st.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length = st.number_input("バンク周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)

# 周回・開催日
base_laps = st.number_input("周回数（通常4、高松など5）", 1, 10, 4, 1)
day_label_to_idx = {"初日":1, "2日目":2, "最終日":3}
day_label = st.selectbox("開催日（疲労補正：初日+1 / 2日目+2 / 最終日+3）", list(day_label_to_idx.keys()))
day_idx = day_label_to_idx[day_label]
eff_laps = int(base_laps) + DAY_DELTA.get(day_idx, 1)

# 選手入力
st.header("【選手データ入力】")
st.subheader("▼ 位置（脚質）：逃＝先頭／両＝番手／追＝3番手以降＆単騎（車番を半角数字で入力）")
car_to_kakushitsu = {}
c = st.columns(3)
for i, k in enumerate(['逃','両','追']):
    with c[i]:
        s = st.text_input(f"{k}", key=f"kaku_{k}", max_chars=18)
    for ch in s:
        if ch.isdigit():
            n = int(ch)
            if 1 <= n <= N_MAX:
                car_to_kakushitsu[n] = k

# 競争得点
st.subheader("▼ 競争得点")
rating, invalid_inputs = [], []
for i in range(N_MAX):
    key_txt = f"pt_txt_v2_{i}"
    key_val = f"pt_val_v2_{i}"
    prev_valid = float(st.session_state.get(key_val, 55.0))
    default_str = st.session_state.get(key_txt, f"{prev_valid:.1f}")
    s = st.text_input(f"{i+1}番 得点（例: 55.0）", value=str(default_str), key=key_txt)
    v = _parse_float_flexible(s)
    if v is None:
        invalid_inputs.append(i + 1)
        v = prev_valid
        st.session_state[key_txt] = f"{v:.1f}"
    else:
        st.session_state[key_val] = float(v)
    rating.append(float(v))
abnormal = [(i+1, v) for i, v in enumerate(rating) if v < 20.0 or v > 120.0]
if invalid_inputs:
    st.error("数値として解釈できない得点入力があったため、前回の有効値に戻しました: " + ", ".join(map(str, invalid_inputs)))
if abnormal:
    st.warning("競争得点の想定外の値があります: " + ", ".join([f"{no}:{val:.1f}" for no, val in abnormal]))

# 2連対率 / 3連対率
st.subheader("▼ 2連対率 / 3連対率（％入力可：7 / 12.5 / ７ / 12.5%）")
P2_list, P3_list = [], []
for i in range(N_MAX):
    key_p2_txt = f"p2_txt_{i+1}"
    key_p3_txt = f"p3_txt_{i+1}"
    default_p2 = st.session_state.get(key_p2_txt, "")
    default_p3 = st.session_state.get(key_p3_txt, "")
    s2 = st.text_input(f"{i+1}番 2連対率(%)", value=str(default_p2), key=key_p2_txt)
    s3 = st.text_input(f"{i+1}番 3連対率(%)", value=str(default_p3), key=key_p3_txt)
    P2_list.append(_parse_percent_flexible(s2))
    P3_list.append(_parse_percent_flexible(s3))

# 隊列
st.subheader("▼ 予想隊列（数字、欠は空欄）")
tairetsu = [st.text_input(f"{i+1}番 隊列順位", key=f"tai_{i}") for i in range(N_MAX)]

# S・B
st.subheader("▼ S・B 回数")
for i in range(N_MAX):
    st.number_input(f"{i+1}番 S回数", 0, 99, 0, key=f"s_{i+1}")
    st.number_input(f"{i+1}番 B回数", 0, 99, 0, key=f"b_{i+1}")

# ライン構成
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

# ===============================
# スコア計算（activeのみ）
# ===============================
active_idx = [i for i in range(N_MAX) if str(tairetsu[i]).isdigit()]
n_cars = len(active_idx)
LINE_BONUS, POS_MULTI_MAP, UPPER_K = dynamic_params(n_cars)

ratings_active = [rating[i] for i in active_idx]
corr_active = score_from_tenscore_list_dynamic(ratings_active, upper_k=UPPER_K)
tenscore_score = [0.0] * N_MAX
for j, k in enumerate(active_idx):
    tenscore_score[k] = corr_active[j]

R_place = [0.6 * P2_list[i] + 0.4 * P3_list[i] for i in range(N_MAX)]
Z_R = _zscore_clip([R_place[i] for i in active_idx]) if active_idx else pd.Series(dtype=float)
alpha, cap = 0.30, 0.60
Place_Delta = [0.0] * N_MAX
for j, i in enumerate(active_idx):
    delta = float(Z_R.iloc[j]) if len(Z_R) > j else 0.0
    Place_Delta[i] = round(np.clip(alpha * delta, -cap, cap) / 3.0, 3)

score_parts = []
for i in active_idx:
    num = i + 1
    kaku = car_to_kakushitsu.get(num, "追")
    base = BASE_SCORE.get(kaku, 0.0)
    wind = wind_straight_combo_adjust(kaku, st.session_state.selected_wind, wind_speed, straight_length, line_order[i], POS_MULTI_MAP)
    rating_score = tenscore_score[i]
    rain_corr = lap_adjust(kaku, eff_laps)
    s_bonus = min(0.1 * st.session_state.get(f"s_{num}", 0), 0.5)
    b_bonus = min(0.1 * st.session_state.get(f"b_{num}", 0), 0.5)
    sb_bonus = s_bonus + b_bonus
    line_b = line_member_bonus(line_order[i], LINE_BONUS)
    bank_b = bank_character_bonus(kaku, bank_angle, straight_length)
    length_b = bank_length_adjust(kaku, bank_length)
    place_delta = Place_Delta[i]
    total = base + wind + rating_score + rain_corr + sb_bonus + line_b + bank_b + length_b + place_delta
    score_parts.append([num, kaku, base, wind, rating_score, rain_corr, sb_bonus, line_b, bank_b, length_b, place_delta, total])

labels = ["A","B","C","D","E","F","G"]
line_def = {labels[idx]: line for idx, line in enumerate(lines) if line}
car_to_group = {car: g for g, members in line_def.items() for car in members}

group_bonus_map = compute_group_bonus(score_parts, line_def, n_cars)
final_score_parts = []
for row in score_parts:
    group_corr = get_group_bonus(row[0], line_def, group_bonus_map, a_head_bonus=True)
    final_score_parts.append(row[:-1] + [group_corr, row[-1] + group_corr])

columns = ['車番','脚質','基本','風補正','得点補正','周回補正','SB印補正','ライン補正','バンク補正','周長補正','着内Δ','グループ補正','合計スコア']
df = pd.DataFrame(final_score_parts, columns=columns)

try:
    rating_map = {i + 1: rating[i] for i in range(N_MAX)}
    df['競争得点'] = df['車番'].map(rating_map)
    df['2連対率(%)'] = df['車番'].map({i+1: P2_list[i]*100 for i in range(N_MAX)}).round(1)
    df['3連対率(%)'] = df['車番'].map({i+1: P3_list[i]*100 for i in range(N_MAX)}).round(1)
except Exception:
    pass

# =========================================================
# 印決定
# =========================================================
st.markdown("### 📊 合計スコア順（印・スコア・競争得点・理由）")
if df.empty:
    st.error("データが空です。入力を確認してください。")
else:
    df_rank = df.sort_values(by='合計スコア', ascending=False).reset_index(drop=True)
    velobi_sorted = list(zip(df_rank['車番'].tolist(), df_rank['合計スコア'].round(1).tolist()))

    points_df = pd.DataFrame({"車番": [i + 1 for i in active_idx], "得点": [rating[i] for i in active_idx]})
    if not points_df.empty:
        points_df["順位"] = points_df["得点"].rank(ascending=False, method="min").astype(int)
        comp_points_rank = dict(zip(points_df["車番"], points_df["順位"]))
        max_pt = float(points_df["得点"].max())
        delta_map = {int(r.車番): round(max_pt - float(r.得点), 2) for r in points_df.itertuples()}
    else:
        comp_points_rank, delta_map = {}, {}

    marks_order = ["◎","〇","▲","△","×","α","β"]
    result_marks, reasons = {}, {}

    if mode == "ガールズ":
        a, b = pick_girls_anchor_second(velobi_sorted, comp_points_rank)
        if a:
            result_marks["◎"] = a[0]; reasons[a[0]] = "本命(得点1-4)"
        if b:
            result_marks["〇"] = b[0]; reasons[b[0]] = "対抗(得点1-4)"
        used = set(result_marks.values())
        rest = [no for no, _ in velobi_sorted if no not in used]
        for m, n in zip([m for m in marks_order if m not in result_marks], rest):
            result_marks[m] = n
    else:
        # 男子：Δ≤5母集団
        C = [no for no, _ in velobi_sorted if delta_map.get(no, 99) <= 5.0]
        if len(C) <= 2:
            C = [no for no, _ in velobi_sorted if delta_map.get(no, 99) <= 7.0]
        if not C:
            C = [no for no, _ in velobi_sorted[:3]]
        ordered_C = [no for no, _ in velobi_sorted if no in C]

        TOP_N = min(5, len(ordered_C))   # 極端な穴の上位印混入を防ぐ安全弁
        topC = ordered_C[:TOP_N]

        vmap = dict(velobi_sorted)
        gmap = {car: g for g, members in line_def.items() for car in members}

        # ◎：母集団スコア首位
        anchor_no = topC[0]
        result_marks["◎"] = anchor_no
        reasons[anchor_no] = "本命(Δ≤5母集団・スコア首位)"

        # ○▲：同ライン最上位 vs 他ライン最上位を直接比較し、上位を○・残りを▲
        EPS_SAME = 0.05  # 同ラインに微ボーナス（好みで0.03〜0.10）
        g_anchor = gmap.get(anchor_no, None)

        cand = [no for no in topC if no != anchor_no]
        same_line = [no for no in cand if gmap.get(no) == g_anchor]
        other_line = [no for no in cand if gmap.get(no) != g_anchor]

        def eff_score(no):
            if no is None: return -9e9
            bonus = EPS_SAME if (g_anchor and gmap.get(no) == g_anchor) else 0.0
            return vmap.get(no, -9e9) + bonus

        best_same  = same_line[0] if same_line else None
        best_other = other_line[0] if other_line else None

        if best_same and best_other:
            # 直接比較：高い方を○、もう一方を▲（補完関係を徹底）
            if eff_score(best_same) >= eff_score(best_other):
                result_marks["〇"] = best_same;  reasons[best_same]  = "対抗(同ライン上位)"
                result_marks["▲"] = best_other; reasons[best_other] = "単穴(他ライン上位)"
            else:
                result_marks["〇"] = best_other; reasons[best_other] = "対抗(他ライン上位)"
                result_marks["▲"] = best_same;  reasons[best_same]  = "単穴(同ライン上位)"
        elif best_same and not best_other:
            # 他ライン候補がいない→○は同ライン、▲は残りの中から（できれば他ライン、無ければ次点）
            result_marks["〇"] = best_same; reasons[best_same] = "対抗(同ライン上位)"
            # 他ラインが無いので規則上の“補完”は満たせない→スコア次点で埋める
            ordered_rest = [no for no, _ in velobi_sorted if no not in {anchor_no, result_marks["〇"]}]
            if ordered_rest:
                result_marks["▲"] = ordered_rest[0]; reasons[ordered_rest[0]] = "単穴(スコア次点)"
        elif best_other and not best_same:
            # 同ライン候補がいない→○は他ライン、▲は残りの中から（できれば同ライン、無ければ次点）
            result_marks["〇"] = best_other; reasons[best_other] = "対抗(他ライン上位)"
            ordered_rest = [no for no, _ in velobi_sorted if no not in {anchor_no, result_marks["〇"]}]
            if ordered_rest:
                result_marks["▲"] = ordered_rest[0]; reasons[ordered_rest[0]] = "単穴(スコア次点)"
        else:
            # 候補が空（topCが1頭等）→スコア順で○▲
            ordered_rest = [no for no, _ in velobi_sorted if no != anchor_no]
            if ordered_rest:
                result_marks["〇"] = ordered_rest[0]; reasons[ordered_rest[0]] = "対抗(スコア上位)"
            if len(ordered_rest) >= 2:
                result_marks["▲"] = ordered_rest[1]; reasons[ordered_rest[1]] = "単穴(スコア次点)"

        # 残りはスコア順で埋める
        used = set(result_marks.values())
        tail = [no for no, _ in velobi_sorted if no not in used]
        for m, n in zip(["△","×","α","β"], tail):
            result_marks[m] = n

    # 重複排除＆埋め
    def finalize_marks_unique(result_marks: dict, velobi_sorted: list):
        order = ["◎","〇","▲","△","×","α","β"]
        used = set(); final = {}
        for m in order:
            n = result_marks.get(m)
            if n is not None and n not in used:
                final[m] = n; used.add(n)
        for m in order:
            if m not in final:
                for no, _ in velobi_sorted:
                    if no not in used:
                        final[m] = no; used.add(no); break
        return final

    result_marks = finalize_marks_unique(result_marks, velobi_sorted)

    # 表示
    rows = []
    for r, (no, sc) in enumerate(velobi_sorted, start=1):
        mark = [m for m, v in result_marks.items() if v == no]
        reason = reasons.get(no, "")
        pt = df.loc[df['車番'] == no, '競争得点'].iloc[0] if '競争得点' in df.columns else None
        delta_pt = None
        if '競争得点' in df.columns and len(points_df):
            delta_pt = delta_map.get(no, None)
        rows.append({"順": r, "印": "".join(mark), "車": no, "合計スコア": sc, "競争得点": pt, "Δ得点": delta_pt, "理由": reason})
    view_df = pd.DataFrame(rows)
    st.dataframe(view_df, use_container_width=True)

    st.markdown("### 🧩 補正内訳（合計スコア高い順）")
    cols_show = ['車番','脚質','基本','風補正','得点補正','周回補正','SB印補正','ライン補正','バンク補正','周長補正','着内Δ','グループ補正','合計スコア','競争得点','2連対率(%)','3連対率(%)']
    df_rank = df_rank[[c for c in cols_show if c in df_rank.columns]]
    st.dataframe(df_rank, use_container_width=True)

    tag = f"開催日補正 +{DAY_DELTA.get(day_idx,1)}（有効周回={eff_laps}） / 風向:{st.session_state.get('selected_wind','無風')} / 出走:{n_cars}車（入力:{N_MAX}枠）"
    st.caption(tag)

    # note記事用（3行だけ・手動コピー）
    st.markdown("### 📋 note記事用（コピーは手動で）")
    line_text = "　".join([x for x in line_inputs if str(x).strip()])
    score_order_text = " ".join(str(no) for no, _ in velobi_sorted)
    marks_order = ["◎","〇","▲","△","×","α","β"]
    marks_line = " ".join(f"{m}{result_marks[m]}" for m in marks_order if m in result_marks)
    note_text = f"ライン　{line_text}\nスコア順　{score_order_text}\n{marks_line}"
    st.text_area("ここを選択してコピー", note_text, height=96)
    # st.code(note_text, language="")  # クリック全選択派はこちらに切替可
