# app.py  ヴェロビ（5〜9車対応・買い目/的中率/必要オッズ/EVつき 完全版）
# -------------------------------------------------------------
# ◎：ラインSBを加味した総合(=with_SB)で選ぶ。ただし候補は「競争得点の両端(最大/最小)を除いた平均以上」。
# 〇/▲：SBなし(=without_SB)でランク付けし、◎と同ライン/他ラインのバランスで決定。
# 買い目：3連複A/B/C・2車単(1-23)・2車複(1-23)・ワイド(1-23)の想定的中率/必要オッズ/EVを表示し、コピー欄にも出力。
# -------------------------------------------------------------

import math
import random
import statistics as stats
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# 基本設定
# -------------------------------
st.set_page_config(page_title="ヴェロビ 完全版（買い目つき）", layout="wide")

st.title("⭐ ヴェロビ 完全版（5〜9車・買い目/的中率/EVつき）⭐")

# 競輪場リスト（表記用）
KEIRIN_TRACKS = [
    "函館","青森","いわき平","弥彦","前橋","取手","宇都宮","大宮","西武園","京王閣","立川","松戸","川崎","平塚",
    "小田原","伊東","静岡","名古屋","岐阜","大垣","豊橋","富山","松坂","四日市","福井","奈良","向日町","和歌山",
    "岸和田","玉野","広島","防府","高松","小松島","高知","松山","小倉","久留米","武雄","佐世保","別府","熊本","手入力"
]

# ライン係数：日程/級別
DAY_LINE_COEF  = {"初日":1.00, "2日目":0.60, "最終日":0.85}
GRADE_LINE_COEF = {"Ｓ級":1.00, "Ａ級":0.85, "Ａ級チャレンジ":0.75, "ガールズ":0.00}

# 会場バイアス（-2差し ←→ +2先行）を脚質に掛ける係数
BIAS_PER_STEP = 0.05  # 1.0刻み=+/-0.05

# 2着・3着の重み
W_2ND = 0.50
W_3RD = 0.25

# -------------------------------
# ヘルパ
# -------------------------------
def safe_float(x, default=0.0):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def trimmed_mean_exclude_minmax(arr: list[float]) -> float:
    """最小値・最大値を除いて平均。要素が3未満なら通常平均。"""
    x = [safe_float(v) for v in arr if v is not None]
    if len(x) < 3:
        return sum(x)/len(x) if x else 0.0
    mn, mx = min(x), max(x)
    core = [v for v in x if v != mn and v != mx]
    if not core:  # 全部同じ等で空になったとき保険
        return sum(x)/len(x)
    return sum(core)/len(core)

def softmax_strength(scores: list[float], temp: float = 1.0) -> np.ndarray:
    """スコア→強さ。温度=1でsoftmax。"""
    x = np.array(scores, dtype=float)
    if x.std() == 0:
        return np.ones_like(x) / len(x)
    z = (x - x.mean()) / (x.std() if x.std() > 1e-8 else 1.0)
    e = np.exp(z / max(1e-6, temp))
    return e / e.sum()

def sample_finish_order(strength: np.ndarray, rng: random.Random) -> list[int]:
    """強さに基づき、重み付き無作為抽出で着順決定（車番indexリスト）。"""
    idxs = list(range(len(strength)))
    # 無作為重みドロー（Gumbel trick）
    g = np.array([rng.random() for _ in idxs], dtype=float)
    # ランダム性を持たせつつ強いほど上位に
    score = np.log(strength + 1e-12) - np.log(-np.log(g + 1e-12) + 1e-12)
    order = np.argsort(-score).tolist()
    return order

def odds_needed(p: float | None) -> float | None:
    if p is None or p <= 0: return None
    return 1.0 / p

def fmt(x, digits=2, pct=False):
    if x is None: return "-"
    return f"{x*100:.{digits}f}%" if pct else f"{x:.{digits}f}"

# -------------------------------
# サイドバー：開催情報・入力
# -------------------------------
with st.sidebar:
    st.subheader("開催情報")
    track = st.selectbox("競輪場", KEIRIN_TRACKS, index=KEIRIN_TRACKS.index("西武園") if "西武園" in KEIRIN_TRACKS else 0)
    race_no = st.selectbox("レース番号", [str(i) for i in range(1,13)], index=4)
    session = st.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], index=2)
    grade = st.selectbox("級別", ["Ｓ級","Ａ級","Ａ級チャレンジ","ガールズ"], index=1)
    day_label = st.selectbox("開催日", ["初日","2日目","最終日"], index=0)
    n_cars = st.select_slider("出走数（5〜9）", options=list(range(5,10)), value=7)

    st.markdown("---")
    bias_val = st.slider("会場バイアス補正（-2差し ←→ +2先行）", min_value=-2.0, max_value=2.0, value=0.0, step=0.25)
    st.caption(f"会場スタイル：{bias_val:.2f}（逃に{bias_val*BIAS_PER_STEP:+.3f} / 追に{-(bias_val*BIAS_PER_STEP):+.3f}）")
    st.markdown("---")
    sim_trials = st.slider("買い目シミュレーション試行回数", min_value=2000, max_value=50000, value=10000, step=1000)

# -------------------------------
# 選手入力
# -------------------------------
st.header("【選手データ入力】")

# ライン構成（最大7）
cols = st.columns(4)
with cols[0]:
    st.write("ライン構成（例：31 / 254 / 6 / 7）")
line_inputs = [
    st.text_input("ライン1", value=""),
    st.text_input("ライン2", value=""),
    st.text_input("ライン3", value=""),
    st.text_input("ライン4", value=""),
    st.text_input("ライン5", value=""),
    st.text_input("ライン6", value=""),
    st.text_input("ライン7", value=""),
]
def extract_car_list(s: str) -> list[int]:
    return [int(ch) for ch in str(s) if ch.isdigit()]
lines = [extract_car_list(x) for x in line_inputs if str(x).strip()]

# 車番ごとのライン順（0=単騎/先頭、1=番手、2=3番手以降）
car_to_linepos = {}
for line in lines:
    if not line: 
        continue
    if len(line) == 1:
        car_to_linepos[line[0]] = 0
    else:
        for pos, no in enumerate(line, start=0):
            car_to_linepos[no] = pos

def kaku_from_pos(pos:int)->str:
    if pos == 0: return "逃"
    if pos == 1: return "両"
    return "追"

# 入力テーブル
df_in = pd.DataFrame({
    "車番": list(range(1, n_cars+1)),
})
df_in["ライン順"] = df_in["車番"].map(lambda x: car_to_linepos.get(x, 2))
df_in["脚質"] = df_in["ライン順"].map(kaku_from_pos)

c1, c2, c3 = st.columns(3)
with c1:
    pts = [st.number_input(f"{i}番 競争得点", min_value=0.0, max_value=150.0, value=90.0, step=0.1, key=f"pt_{i}") for i in range(1, n_cars+1)]
with c2:
    r2 = [st.number_input(f"{i}番 2着率(%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, key=f"r2_{i}") for i in range(1, n_cars+1)]
with c3:
    r3 = [st.number_input(f"{i}番 3着率(%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, key=f"r3_{i}") for i in range(1, n_cars+1)]

df_in["得点"] = pts
df_in["2着率"] = r2
df_in["3着率"] = r3

# -------------------------------
# スコア計算
# -------------------------------
# 得点の軽い標準化スコア（過度に効かせない）
if df_in["得点"].std(ddof=0) > 1e-9:
    score_pts = (df_in["得点"] - df_in["得点"].mean()) / df_in["得点"].std(ddof=0)
else:
    score_pts = pd.Series([0.0]*n_cars)

# 2/3着率（上限処理：2着率=そのまま、3着率そのまま）
r2_eff = np.clip(df_in["2着率"].values, 0, 100) / 100.0
r3_eff = np.clip(df_in["3着率"].values, 0, 100) / 100.0
score_place = W_2ND * r2_eff + W_3RD * r3_eff  # 0〜0.75くらいの帯

# 会場バイアス：逃に+、追に-（両は中間）
bias_adj = []
for pos in df_in["ライン順"]:
    if pos == 0:  # 逃
        bias_adj.append(+bias_val * BIAS_PER_STEP)
    elif pos == 1:  # 両
        bias_adj.append(0.0)
    else:  # 追
        bias_adj.append(-bias_val * BIAS_PER_STEP)
bias_adj = np.array(bias_adj)

# ベース（SBなし）合計
base_wo = 1.60 + 0.10*score_pts.values + score_place + bias_adj

# ライン補正（グループボーナス）：日程×級別を効かせる（ガールズは0）
line_coef = DAY_LINE_COEF.get(day_label,1.0) * GRADE_LINE_COEF.get(grade,1.0)

# ライングループ割当
labels = list("ABCDEFG")
line_def = {}
li = 0
for ln in lines:
    if not ln: 
        continue
    name = labels[li] if li < len(labels) else f"L{li+1}"
    line_def[name] = ln
    li += 1
# 単騎を拾う（ライン未記載で出走している車）
in_lines = {x for xs in line_def.values() for x in xs}
for no in range(1, n_cars+1):
    if no not in in_lines:
        name = labels[li] if li < len(labels) else f"L{li+1}"
        line_def[name] = [no]
        li += 1

# ライングループ別の強さ（SBなしベースで集計して配分）
group_sum = {g: sum(base_wo[no-1] for no in mem) for g, mem in line_def.items()}
# 強いラインからボーナスを多く（正規化）
if len(group_sum) > 0:
    ranked_groups = sorted(group_sum.items(), key=lambda x: x[1], reverse=True)
    weights = np.array([0.8**i for i in range(len(ranked_groups))], dtype=float)
    weights = weights / weights.sum()
    # 総ボーナス枠（人数依存）
    total_budget = 0.35 * math.sqrt(n_cars/7.0)
    group_bonus = {g: float(total_budget * w * line_coef) for (g,_), w in zip(ranked_groups, weights)}
else:
    group_bonus = {}

# 車番→グループ
car_to_group = {}
for g, mem in line_def.items():
    for no in mem:
        car_to_group[no] = g

# 各車へのボーナス配賦（ライン人数で割る。先頭は微加点）
with_bonus = base_wo.copy()
for g, mem in line_def.items():
    if g in group_bonus:
        add = group_bonus[g] / max(1, len(mem))
        for idx, no in enumerate(mem):
            head_plus = 0.05 if idx == 0 else 0.0
            with_bonus[no-1] += add + head_plus*line_coef

# SBあり/なしのスコア
df_calc = df_in.copy()
df_calc["SBなしスコア"] = np.round(without_sb := with_bonus - (line_coef*(0.0)) , 3)  # 名前だけ維持
df_calc["SBありスコア"] = np.round(with_bonus, 3)

# -------------------------------
# ランキング・印
# -------------------------------
order_wo = sorted(range(1, n_cars+1), key=lambda no: df_calc.loc[no-1,"SBなしスコア"], reverse=True)
order_with = sorted(range(1, n_cars+1), key=lambda no: df_calc.loc[no-1,"SBありスコア"], reverse=True)

# ◎候補：得点の「両端除外平均」以上
pt_tmean = trimmed_mean_exclude_minmax(df_calc["得点"].tolist())
candidates = [no for no in range(1, n_cars+1) if df_calc.loc[no-1,"得点"] >= pt_tmean]
if not candidates:
    candidates = list(range(1, n_cars+1))

# ◎：候補内でSBありスコア最大
anchor = max(candidates, key=lambda no: df_calc.loc[no-1,"SBありスコア"])

# 〇/▲：SBなしでランク付け。◎同ライン・他ラインのバランス
same_line_cands = [no for no in order_wo if no != anchor and car_to_group.get(no,"") == car_to_group.get(anchor,"")]
other_line_cands = [no for no in order_wo if no != anchor and car_to_group.get(no,"") != car_to_group.get(anchor,"")]

cand_same = same_line_cands[0] if same_line_cands else None
cand_other = other_line_cands[0] if other_line_cands else None

if cand_same and cand_other:
    # どちらが上位かで〇/▲を割り振り
    if order_wo.index(cand_same) < order_wo.index(cand_other):
        circle, triangle = cand_same, cand_other
    else:
        circle, triangle = cand_other, cand_same
elif cand_same:
    circle, triangle = cand_same, (other_line_cands[1] if len(other_line_cands)>1 else (same_line_cands[1] if len(same_line_cands)>1 else None))
elif cand_other:
    circle, triangle = cand_other, (same_line_cands[0] if same_line_cands else (other_line_cands[1] if len(other_line_cands)>1 else None))
else:
    circle, triangle = (order_wo[1] if len(order_wo)>1 else anchor), (order_wo[2] if len(order_wo)>2 else None)

marks_order = ["◎","〇","▲","△","×","α","β"]
result_marks = {"◎":anchor}
if circle:   result_marks["〇"] = circle
if triangle: result_marks["▲"] = triangle
# 残りをSBなし順で埋める
used = set(result_marks.values())
rest = [no for no in order_wo if no not in used]
for m, no in zip([m for m in marks_order if m not in result_marks], rest):
    result_marks[m] = no

# 表示テーブル
rank_rows = []
for rank_pos, no in enumerate(order_wo, start=1):
    mark = "".join([m for m,v in result_marks.items() if v == no])
    rank_rows.append({
        "順(SBなし)": rank_pos,
        "印": mark,
        "車": no,
        "SBなしスコア": round(df_calc.loc[no-1,"SBなしスコア"],3),
        "SBありスコア": round(df_calc.loc[no-1,"SBありスコア"],3),
        "得点": df_calc.loc[no-1,"得点"],
        "1=◎/2=〇/3=▲": (1 if no==anchor else (2 if no==circle else (3 if no==triangle else ""))),
        "ライン": car_to_group.get(no,"-")
    })
st.subheader("ランキング＆印（◎＝SBあり / 〇＝安定枠 / ▲＝穴枠）")
st.dataframe(pd.DataFrame(rank_rows), use_container_width=True)

# -------------------------------
# 的中率シミュレーション（買い目）
# -------------------------------
st.header("買い目シミュレーション（的中率・必要オッズ・EV）")

# 強さは SBなしスコアをsoftmax化
strength = softmax_strength([df_calc.loc[i,"SBなしスコア"] for i in range(n_cars)])

# ◎=1, 〇=2, ▲=3 として集合を用意
one = anchor
two = circle
three = triangle
rest_list = [no for no in order_wo if no not in [one, two, three]]

# 3連複パターン定義
# A: 1-2-34567（◎/〇固定＋残り上位から最大5（▲を優先して含む））
third_set_A = [x for x in [three] + rest_list if x is not None][:min(5, max(1, n_cars-2))]

# B: 1-2345-2345（◎固定＋相手4頭ボックス：▲＋残り上位）
box_B = [x for x in [two, three] + rest_list][:4]
box_B = [x for x in box_B if x is not None]

# C: 1-23-全（◎固定＋{〇,▲}のいずれか＋相手総流し）
set_23 = [x for x in [two, three] if x is not None]
set_all_ex1 = [no for no in range(1, n_cars+1) if no != one]

# イベント判定
def tri_A_hit(top3:set[int])->bool:
    return (one in top3) and (two in top3) and (len(top3.intersection(set(third_set_A)))>=1)

def tri_B_hit(top3:set[int])->bool:
    # ◎がtop3、かつ box_Bから2つ以上top3に含まれる
    return (one in top3) and (len(top3.intersection(set(box_B)))>=2)

def tri_C_hit(top3:set[int])->bool:
    # ◎がtop3、かつ {〇,▲}のいずれかがtop3
    return (one in top3) and (len(top3.intersection(set(set_23)))>=1)

def exacta_hit(order:list[int])->bool:
    # 2車単 1-23
    return len(set_23)>0 and order[0]==one and order[1] in set_23

def quinella_hit(order:list[int])->bool:
    # 2車複 1-23
    return len(set_23)>0 and set(order[:2])==set([one, set_23[0]]) or (len(set_23)>1 and set(order[:2])==set([one, set_23[1]]))

def wide_hit(order:list[int])->bool:
    # ワイド 1-23（◎と2or3がどちらも3着内）
    return (one in order[:3]) and (len(set(order[:3]).intersection(set(set_23)))>=1)

# シミュレーション
rng = random.Random(20250830)
hit_A = hit_B = hit_C = hit_EX = hit_QN = hit_WD = 0
for _ in range(sim_trials):
    ord_idx = sample_finish_order(strength, rng)  # 0-based
    order_no = [i+1 for i in ord_idx]
    top3 = set(order_no[:3])

    if tri_A_hit(top3): hit_A += 1
    if tri_B_hit(top3): hit_B += 1
    if tri_C_hit(top3): hit_C += 1
    if exacta_hit(order_no): hit_EX += 1
    if quinella_hit(order_no): hit_QN += 1
    if wide_hit(order_no): hit_WD += 1

p_A  = hit_A / sim_trials
p_B  = hit_B / sim_trials
p_C  = hit_C / sim_trials
p_EX = hit_EX / sim_trials
p_QN = hit_QN / sim_trials
p_WD = hit_WD / sim_trials

# 必要オッズ（=1/p）
need_A, need_B, need_C = odds_needed(p_A), odds_needed(p_B), odds_needed(p_C)
need_EX, need_QN, need_WD = odds_needed(p_EX), odds_needed(p_QN), odds_needed(p_WD)

# オッズ入力（任意）→ EV計算
st.caption("※ 実オッズを入れるとEV（期待値）を表示します。未入力なら『-』。")
c_oa, c_ob, c_oc = st.columns(3)
with c_oa:
    odds_A = st.number_input("三連複A 1-2-… の実オッズ（合算想定）", min_value=0.0, max_value=9999.9, value=0.0, step=0.1)
with c_ob:
    odds_B = st.number_input("三連複B 1-…-… の実オッズ（合算想定）", min_value=0.0, max_value=9999.9, value=0.0, step=0.1)
with c_oc:
    odds_C = st.number_input("三連複C 1-23-全 の実オッズ（合算想定）", min_value=0.0, max_value=9999.9, value=0.0, step=0.1)

c_ox, c_oq, c_ow = st.columns(3)
with c_ox:
    odds_EX = st.number_input("二車単 1-23 の実オッズ", min_value=0.0, max_value=9999.9, value=0.0, step=0.1)
with c_oq:
    odds_QN = st.number_input("二車複 1-23 の実オッズ", min_value=0.0, max_value=9999.9, value=0.0, step=0.1)
with c_ow:
    odds_WD = st.number_input("ワイド 1-23 の実オッズ", min_value=0.0, max_value=9999.9, value=0.0, step=0.1)

def ev(p, o):
    if p is None or p<=0 or o is None or o<=0: return None
    return p*o - 1.0

ev_A  = ev(p_A , odds_A) if odds_A>0 else None
ev_B  = ev(p_B , odds_B) if odds_B>0 else None
ev_C  = ev(p_C , odds_C) if odds_C>0 else None
ev_EX = ev(p_EX, odds_EX) if odds_EX>0 else None
ev_QN = ev(p_QN, odds_QN) if odds_QN>0 else None
ev_WD = ev(p_WD, odds_WD) if odds_WD>0 else None

# 自信度（簡易）
def confidence_tag(p_main: float) -> str:
    if p_main >= 0.35: return "強"
    if p_main >= 0.22: return "中"
    return "弱"
conf_tag = confidence_tag(p_A)

# 表示
tbl = pd.DataFrame([
    ["三連複A","1-2-{}".format("".join([str(x) for x in third_set_A])), p_A, need_A, ev_A],
    ["三連複B","1-{}-{}".format("".join([str(x) for x in box_B]), "".join([str(x) for x in box_B])), p_B, need_B, ev_B],
    ["三連複C","1-23-全", p_C, need_C, ev_C],
    ["二車単","1-23", p_EX, need_EX, ev_EX],
    ["二車複","1-23", p_QN, need_QN, ev_QN],
    ["ワイド","1-23", p_WD, need_WD, ev_WD],
], columns=["券種","買い目","想定的中率","必要オッズ","EV(入力時)"])
tbl["想定的中率"] = tbl["想定的中率"].map(lambda x: fmt(x,1,True))
tbl["必要オッズ"] = tbl["必要オッズ"].map(lambda x: fmt(x,2,False))
tbl["EV(入力時)"] = tbl["EV(入力時)"].map(lambda x: fmt(x,2,False))
st.dataframe(tbl, use_container_width=True)

# -------------------------------
# note（手動コピー：買い目一覧つき）
# -------------------------------
st.subheader("📋 note記事用（コピー可）")

line_text = "　".join([x for x in line_inputs if str(x).strip()])
score_order_text = " ".join(str(no) for no in order_wo)
marks_line = " ".join(f"{m}{result_marks[m]}" for m in marks_order if m in result_marks)

tri_lines = []
tri_best_key = max({"A":p_A, "B":p_B, "C":p_C}, key=lambda k: {"A":p_A,"B":p_B,"C":p_C}[k])
for key, label, p, need, evv in [
    ("A", f"1-2-{''.join([str(x) for x in third_set_A])}", p_A, need_A, ev_A),
    ("B", f"1-{''.join([str(x) for x in box_B])}-{''.join([str(x) for x in box_B])}", p_B, need_B, ev_B),
    ("C", "1-23-全", p_C, need_C, ev_C),
]:
    tag = " ◎推奨" if key==tri_best_key else ""
    tri_lines.append(f"三連複{key} {label}  p={fmt(p,1,True)} / 必要={fmt(need,2)}倍" + (f" / EV={fmt(evv,2)}" if evv is not None else "") + tag)

pair_lines = [
    f"二車単 1-23  p={fmt(p_EX,1,True)} / 必要={fmt(need_EX,2)}倍" + (f" / EV={fmt(ev_EX,2)}" if ev_EX is not None else ""),
    f"二車複 1-23  p={fmt(p_QN,1,True)} / 必要={fmt(need_QN,2)}倍" + (f" / EV={fmt(ev_QN,2)}" if ev_QN is not None else ""),
    f"ワイド 1-23  p={fmt(p_WD,1,True)} / 必要={fmt(need_WD,2)}倍" + (f" / EV={fmt(ev_WD,2)}" if ev_WD is not None else ""),
]

note_text = (
    f"競輪場　{track}{race_no}R\n"
    f"{session}　{grade}\n"
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}\n"
    f"自信度：{conf_tag}\n"
    "――――――――――\n"
    "【買い目（想定的中率 / 必要オッズ / EV）】\n" +
    "\n".join(tri_lines + pair_lines)
)
st.text_area("ここを選択してコピー", note_text, height=220)

# セッションにも保持（他ページ連携したい場合用）
st.session_state["p_tri_A"]=p_A; st.session_state["p_tri_B"]=p_B; st.session_state["p_tri_C"]=p_C
st.session_state["p_exacta"]=p_EX; st.session_state["p_quin"]=p_QN; st.session_state["p_wide"]=p_WD
st.session_state["ev_tri_A"]=ev_A; st.session_state["ev_tri_B"]=ev_B; st.session_state["ev_tri_C"]=ev_C
st.session_state["ev_exacta"]=ev_EX; st.session_state["ev_quin"]=ev_QN; st.session_state["ev_wide"]=ev_WD
st.session_state["selected_track"]=track
st.session_state["race_no"]=race_no
st.session_state["race_time_label"]=session
st.session_state["race_class_label"]=grade
st.session_state["confidence_tag"]=conf_tag
st.session_state["line_inputs"]=line_inputs
