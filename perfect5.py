# app.py
# ヴェロビ（SB分離・完全版 / 両端除外平均 + s★(SBあり首位)強制包含）
# 5〜9車対応 / 欠車対応 / 男女統一 / note出力（手動コピー）
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import re, unicodedata

st.set_page_config(page_title="ヴェロビ（SB分離・完全版）", layout="wide")

# =========================
# 定数
# =========================
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

# =========================
# 補助
# =========================
def _parse_float_flexible(s: str) -> float | None:
    if s is None: return None
    s = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s): return None
    try: return float(s)
    except Exception: return None

def _parse_percent(s: str) -> float:
    if s is None: return 0.0
    t = unicodedata.normalize("NFKC", str(s)).strip().replace("％","%").replace(",","")
    if t.endswith("%"): t = t[:-1].strip()
    if not re.fullmatch(r"\d+(\.\d+)?", t): return 0.0
    v = float(t)
    return max(0.0, min(v, 100.0)) / 100.0

def _zscore_clip(vals, clip=2.5):
    s = pd.Series(vals).astype(float)
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series(0.0, index=s.index)
    return ((s - m) / sd).clip(-clip, clip)

def extract_car_list(x):
    if isinstance(x, str):  return [int(c) for c in x if c.isdigit()]
    if isinstance(x, list): return [int(c) for c in x if isinstance(c, (str,int)) and str(c).isdigit()]
    return []

def build_line_position_map(lines):
    mp = {}
    for line in lines:
        if not line: continue
        if len(line) == 1: mp[line[0]] = 0
        else:
            for pos,car in enumerate(line, start=1): mp[car] = pos
    return mp

def wind_straight_combo_adjust(kaku, wind_dir, wind_spd, straight_len, line_order, pos_multi):
    if wind_dir == "無風" or wind_spd == 0: return 0.0
    wind_adj = WIND_COEFF.get(wind_dir, 0.0)
    pos_m = pos_multi.get(line_order, 0.30)
    coeff = {'逃':1.0, '両':0.7, '追':0.4}.get(kaku, 0.5)
    total = wind_spd * wind_adj * coeff * pos_m
    return round(max(min(total, 0.05), -0.05), 3)

def lap_adjust(kaku, laps):
    d = max(int(laps) - 2, 0)
    return {'逃':round(-0.1*d,1), '追':round(+0.05*d,1), '両':0.0}.get(kaku, 0.0)

def line_member_bonus(line_order, bonus_map): return bonus_map.get(line_order, 0.0)

def bank_character_bonus(kaku, bank_angle, straight_len):
    s = (float(straight_len)-40.0)/10.0
    a = (float(bank_angle)-25.0)/5.0
    tf = max(min(-0.1*s + 0.1*a, 0.05), -0.05)
    return round({'逃':+tf,'追':-tf,'両':+0.25*tf}.get(kaku,0.0),2)

def bank_length_adjust(kaku, bank_len):
    d = (float(bank_len)-411.0)/100.0
    d = max(min(d,0.05),-0.05)
    return round({'逃':1.0*d,'両':2.0*d,'追':3.0*d}.get(kaku,0.0),2)

def score_from_tenscore_list_dynamic(tens, upper_k=8):
    n = len(tens)
    if n <= 2: return [0.0]*n
    df = pd.DataFrame({"得点":tens})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n, int(upper_k))
    baseline = df[df["順位"].between(2, hi)]["得点"].mean()
    def corr(row): return round(abs(baseline-row["得点"])*0.03,3) if row["順位"] in [2,3,4] else 0.0
    return (df.apply(corr, axis=1)).tolist()

def dynamic_params(n):
    if n <= 7:
        return ({0:0.03,1:0.05,2:0.04,3:0.03,4:0.02},
                {0:0.30,1:0.32,2:0.30,3:0.25,4:0.20},
                (6 if n>=6 else n))
    return ({0:0.03,1:0.05,2:0.04,3:0.03,4:0.02,5:0.015},
            {0:0.30,1:0.32,2:0.30,3:0.25,4:0.20,5:0.18},
            8)

def compute_group_bonus_from_totals(line_def:dict, car_total:dict[int,float], n_cars:int)->dict[str,float]:
    if not line_def: return {}
    alpha = 0.0 if n_cars<=7 else (0.25 if n_cars==8 else 0.5)
    total_budget = 0.42*((max(n_cars,1)/7.0)**0.5)
    sums = {g:0.0 for g in line_def}
    sizes= {g:max(len(mem),1) for g,mem in line_def.items()}
    for g,mem in line_def.items():
        for no in mem: sums[g]+=float(car_total.get(no,0.0))
    adj = {g:(sums[g]/(sizes[g]**alpha)) for g in line_def}
    ordered = sorted(adj.items(), key=lambda x:x[1], reverse=True)
    r=0.80; ws=[r**i for i in range(len(ordered))]; sw=sum(ws) or 1.0
    bonuses=[(w/sw)*total_budget for w in ws]
    return {g:bonuses[i] for i,(g,_) in enumerate(ordered)}

def get_group_bonus(car_no, line_def, bonus_map, a_head_bonus=True):
    for g,mem in line_def.items():
        if car_no in mem: return bonus_map.get(g,0.0)+(0.15 if (a_head_bonus and g=='A') else 0.0)
    return 0.0

# =========================
# UI
# =========================
st.title("⭐ ヴェロビ（SB分離・完全版 / 5〜9車・note用）⭐")
N_MAX = st.slider("出走車数（5〜9）", 5, 9, 7, 1)

# 風・バンク
if "selected_wind" not in st.session_state: st.session_state.selected_wind = "無風"

st.header("【バンク・風条件】")
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("左上"): st.session_state.selected_wind="左上"
with c2:
    if st.button("上"): st.session_state.selected_wind="上"
with c3:
    if st.button("右上"): st.session_state.selected_wind="右上"
c4,c5,c6 = st.columns(3)
with c4:
    if st.button("左"): st.session_state.selected_wind="左"
with c5:
    st.write(f"✅ 風向：{st.session_state.get('selected_wind','無風')}")
with c6:
    if st.button("右"): st.session_state.selected_wind="右"
c7,c8,c9 = st.columns(3)
with c7:
    if st.button("左下"): st.session_state.selected_wind="左下"
with c8:
    if st.button("下"): st.session_state.selected_wind="下"
with c9:
    if st.button("右下"): st.session_state.selected_wind="右下"

selected_track = st.selectbox("競輪場（プリセット）", list(KEIRIN_DATA.keys()))
info = KEIRIN_DATA[selected_track]
wind_speed      = st.number_input("風速(m/s)", 0.0, 30.0, 3.0, 0.1)
straight_length = st.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle      = st.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length     = st.number_input("バンク周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)

# 周回・開催日
base_laps = st.number_input("周回数（通常4、高松など5）", 1, 10, 4, 1)
day_label_to_idx = {"初日":1, "2日目":2, "最終日":3}
day_label = st.selectbox("開催日（疲労補正：初日+1 / 2日目+2 / 最終日+3）", list(day_label_to_idx.keys()))
day_idx = day_label_to_idx[day_label]
eff_laps = int(base_laps) + DAY_DELTA.get(day_idx, 1)

# 入力
st.header("【選手データ入力】")
st.subheader("▼ 位置（脚質）：逃=先頭／両=番手／追=3番手以降＆単騎（車番を半角数字）")
car_to_kakushitsu = {}
cols = st.columns(3)
for i,k in enumerate(['逃','両','追']):
    with cols[i]:
        s = st.text_input(f"{k}", key=f"kaku_{k}", max_chars=18)
    for ch in s:
        if ch.isdigit():
            n = int(ch)
            if 1 <= n <= N_MAX: car_to_kakushitsu[n]=k

# 競争得点
st.subheader("▼ 競争得点")
rating, invalid = [], []
for i in range(N_MAX):
    kt=f"pt_txt_{i}"; kv=f"pt_val_{i}"
    prev=float(st.session_state.get(kv, 55.0))
    default=st.session_state.get(kt, f"{prev:.1f}")
    s=st.text_input(f"{i+1}番 得点（例: 55.0）", value=str(default), key=kt)
    v=_parse_float_flexible(s)
    if v is None:
        invalid.append(i+1); v=prev; st.session_state[kt]=f"{v:.1f}"
    else:
        st.session_state[kv]=float(v)
    rating.append(float(v))
if invalid: st.error("数値として解釈できない得点入力: "+", ".join(map(str, invalid)))

# 2連対率 / 3連対率
st.subheader("▼ 2連対率 / 3連対率（％入力OK：7 / 12.5 / ７ / 12.5%）")
P2,P3=[],[]
for i in range(N_MAX):
    s2=st.text_input(f"{i+1}番 2連対率(%)", key=f"p2_{i}")
    s3=st.text_input(f"{i+1}番 3連対率(%)", key=f"p3_{i}")
    P2.append(_parse_percent(s2)); P3.append(_parse_percent(s3))

# 隊列
st.subheader("▼ 予想隊列（数字、欠は空欄）")
tairetsu=[st.text_input(f"{i+1}番 隊列順位", key=f"tai_{i}") for i in range(N_MAX)]

# SB
st.subheader("▼ S・B 回数")
for i in range(N_MAX):
    st.number_input(f"{i+1}番 S回数", 0, 99, 0, key=f"s_{i+1}")
    st.number_input(f"{i+1}番 B回数", 0, 99, 0, key=f"b_{i+1}")

# ライン
st.subheader("▼ ライン構成（最大7：単騎も1ライン）")
line_inputs=[
    st.text_input("ライン1（例：4）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：12）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：35）", key="line_3", max_chars=9),
    st.text_input("ライン4（例：7）", key="line_4", max_chars=9),
    st.text_input("ライン5（任意）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
]
lines=[extract_car_list(x) for x in line_inputs if str(x).strip()]
line_order_map = build_line_position_map(lines)
line_order=[line_order_map.get(i+1,0) for i in range(N_MAX)]

# =========================
# スコア計算
# =========================
active_idx=[i for i in range(N_MAX) if str(tairetsu[i]).isdigit()]
n_cars=len(active_idx)
LINE_BONUS, POS_MULTI, UPPER_K = dynamic_params(n_cars)

ratings_active=[rating[i] for i in active_idx]
corr_active=score_from_tenscore_list_dynamic(ratings_active, upper_k=UPPER_K)
tenscore_score=[0.0]*N_MAX
for j,k in enumerate(active_idx): tenscore_score[k]=corr_active[j]

# 2着・3着（0.5:0.25）
R_place=[0.5*P2[i] + 0.25*P3[i] for i in range(N_MAX)]
Z_R=_zscore_clip([R_place[i] for i in active_idx]) if active_idx else pd.Series(dtype=float)
alpha,cap=0.30,0.60
Place_Delta=[0.0]*N_MAX
for j,i in enumerate(active_idx):
    delta=float(Z_R.iloc[j]) if len(Z_R)>j else 0.0
    Place_Delta[i]=round(np.clip(alpha*delta,-cap,cap)/3.0,3)

# 素点（SBあり／なし）
percar=[]
for i in active_idx:
    num=i+1
    kaku=car_to_kakushitsu.get(num,"追")
    base=BASE_SCORE.get(kaku,0.0)
    wind=wind_straight_combo_adjust(kaku, st.session_state.selected_wind, wind_speed, straight_length, line_order[i], POS_MULTI)
    rating_score=tenscore_score[i]
    lap=lap_adjust(kaku, eff_laps)
    s_bonus=min(0.1*st.session_state.get(f"s_{num}",0),0.5)
    b_bonus=min(0.1*st.session_state.get(f"b_{num}",0),0.5)
    sb_bonus=s_bonus+b_bonus
    line_b=line_member_bonus(line_order[i], LINE_BONUS)
    bank_b=bank_character_bonus(kaku, bank_angle, straight_length)
    len_b=bank_length_adjust(kaku, bank_length)
    place=Place_Delta[i]

    total_with_raw    = base+wind+rating_score+lap+sb_bonus+line_b+bank_b+len_b+place
    total_without_raw = base+wind+rating_score+lap          +line_b+bank_b+len_b+place
    percar.append([num,kaku,base,wind,rating_score,lap,sb_bonus,line_b,bank_b,len_b,place,total_with_raw,total_without_raw])

labels=["A","B","C","D","E","F","G"]
line_def={labels[idx]:line for idx,line in enumerate(lines) if line}
car_to_group={car:g for g,mem in line_def.items() for car in mem}

# グループ補正（SBあり／なしを完全分離）
car_total_with_map    ={row[0]:row[-2] for row in percar}
car_total_without_map ={row[0]:row[-1] for row in percar}
group_with    = compute_group_bonus_from_totals(line_def, car_total_with_map,    n_cars)
group_without = compute_group_bonus_from_totals(line_def, car_total_without_map, n_cars)

# 最終スコア
final_rows=[]
for row in percar:
    no=row[0]
    g_with=get_group_bonus(no, line_def, group_with, a_head_bonus=True)
    g_wo  =get_group_bonus(no, line_def, group_without, a_head_bonus=True)
    total_with=row[-2]+g_with
    total_wo  =row[-1]+g_wo
    final_rows.append(row[:-2]+[g_with,g_wo,total_with,total_wo])

columns=['車番','脚質','基本','風補正','得点補正','周回補正','SB印補正','ライン補正','バンク補正','周長補正','着内Δ',
         'グループ補正_with','グループ補正_wo','合計_SBあり','合計_SBなし']
df=pd.DataFrame(final_rows, columns=columns)

# 付加表示
try:
    df['競争得点']=df['車番'].map({i+1:rating[i] for i in range(N_MAX)})
    df['2連対率(%)']=df['車番'].map({i+1:P2[i]*100 for i in range(N_MAX)}).round(1)
    df['3連対率(%)']=df['車番'].map({i+1:P3[i]*100 for i in range(N_MAX)}).round(1)
except Exception: pass

# =========================
# 印（◎は C = B ∪ {s★} から）
# =========================
st.markdown("### 📊 ランキング & 印（◎=SBあり / 紐=SBなし）")
if df.empty:
    st.error("データが空です。入力を確認してください。")
else:
    # 得点降順
    points_pairs=sorted([(i+1,float(rating[i])) for i in active_idx], key=lambda x:x[1], reverse=True)

    # 両端除外平均 μ_trim
    if len(points_pairs)>=3:
        core=[pts for _,pts in points_pairs][1:-1]
        mu_trim=(sum(core)/len(core)) if core else (sum(pts for _,pts in points_pairs)/len(points_pairs))
    else:
        mu_trim=(sum(pts for _,pts in points_pairs)/len(points_pairs)) if points_pairs else 0.0

    # 連続ブロックB（得点≥μ_trim）
    B=[]
    for no,pts in points_pairs:
        if pts+1e-9 >= mu_trim: B.append(no)
        else: break
    if not B and points_pairs: B=[no for no,_ in points_pairs[:3]]

    # s★ = SBあり総合首位
    df_with = df.sort_values(by='合計_SBあり', ascending=False).reset_index(drop=True)
    velobi_with = list(zip(df_with['車番'], df_with['合計_SBあり'].round(3)))
    s_star = velobi_with[0][0] if velobi_with else None

    # C = B ∪ {s★}
    C = set(B)
    if s_star is not None: C.add(s_star)

    # ◎：C内で SBあり総合最大（同点: SBなし→得点→低車番）
    cand_with = [(no, score) for no,score in velobi_with if no in C]
    if not cand_with: cand_with = velobi_with[:]  # 保険
    best = cand_with[0]
    # 同点処理
    top_score = best[1]
    ties = [no for no,sc in cand_with if abs(sc - top_score) < 1e-9]
    if len(ties) > 1:
        v_wo = dict(df.sort_values(by='合計_SBなし', ascending=False)[['車番','合計_SBなし']].values)
        ties = sorted(ties, key=lambda x:(-v_wo.get(x,-1e9), -dict(points_pairs).get(x,-1e9), x))
        anchor_no = ties[0]
    else:
        anchor_no = best[0]

    result_marks, reasons = {}, {}
    result_marks["◎"]=anchor_no
    reasons[anchor_no]=f"本命(SBあり首位 / C=B∪{{s★}}, μ_trim={mu_trim:.2f})"

    # 〇▲：SBなしで ◎同ライン首位 vs 他ライン首位
    df_wo = df.sort_values(by='合計_SBなし', ascending=False).reset_index(drop=True)
    velobi_wo = list(zip(df_wo['車番'], df_wo['合計_SBなし'].round(3)))
    v_wo = dict(velobi_wo)
    gmap={car:g for g,mem in line_def.items() for car in mem}
    g_anchor=gmap.get(anchor_no, None)

    cand_wo=[no for no,_ in velobi_wo if no!=anchor_no]
    same_line=[no for no in cand_wo if gmap.get(no)==g_anchor]
    other_line=[no for no in cand_wo if gmap.get(no)!=g_anchor]

    EPS_SAME=0.05
    def eff(no):
        if no is None: return -9e9
        bonus=EPS_SAME if (g_anchor and gmap.get(no)==g_anchor) else 0.0
        return v_wo.get(no,-9e9)+bonus

    best_same  = same_line[0]  if same_line  else None
    best_other = other_line[0] if other_line else None

    if best_same and best_other:
        if eff(best_same) >= eff(best_other):
            result_marks["〇"]=best_same;  reasons[best_same]="対抗(同ライン首位/SBなし)"
            result_marks["▲"]=best_other; reasons[best_other]="単穴(他ライン首位/SBなし)"
        else:
            result_marks["〇"]=best_other; reasons[best_other]="対抗(他ライン首位/SBなし)"
            result_marks["▲"]=best_same;  reasons[best_same]="単穴(同ライン首位/SBなし)"
    elif best_same:
        result_marks["〇"]=best_same; reasons[best_same]="対抗(同ライン首位/SBなし)"
        rest=[no for no in cand_wo if no!=best_same]
        if rest: result_marks["▲"]=rest[0]; reasons[rest[0]]="単穴(次点/SBなし)"
    elif best_other:
        result_marks["〇"]=best_other; reasons[best_other]="対抗(他ライン首位/SBなし)"
        rest=[no for no in cand_wo if no!=best_other]
        if rest: result_marks["▲"]=rest[0]; reasons[rest[0]]="単穴(次点/SBなし)"
    else:
        rest=[no for no,_ in velobi_wo if no!=anchor_no]
        if rest:
            result_marks["〇"]=rest[0]; reasons[rest[0]]="対抗(上位/SBなし)"
        if len(rest)>=2:
            result_marks["▲"]=rest[1]; reasons[rest[1]]="単穴(次点/SBなし)"

    # 残り印：SBなし順
    used=set(result_marks.values())
    tail=[no for no,_ in velobi_wo if no not in used]
    for m,n in zip(["△","×","α","β"], tail): result_marks[m]=n

    # 表示
    rows=[]
    for r,(no,sc_wo) in enumerate(velobi_wo, start=1):
        mark=[m for m,v in result_marks.items() if v==no]
        reason=reasons.get(no,"")
        pt=df.loc[df['車番']==no,'競争得点'].iloc[0] if '競争得点' in df.columns else None
        sc_w=dict(velobi_with).get(no,None)
        rows.append({"順(SBなし)":r,"印":"".join(mark),"車":no,
                     "SBなしスコア":sc_wo,"SBありスコア":sc_w,"競争得点":pt,"理由":reason})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### 🧩 補正内訳（SBあり・なし比較）")
    show_cols=['車番','脚質','基本','風補正','得点補正','周回補正','SB印補正','ライン補正','バンク補正','周長補正','着内Δ',
               'グループ補正_with','グループ補正_wo','合計_SBあり','合計_SBなし','競争得点','2連対率(%)','3連対率(%)']
    st.dataframe(df[show_cols].sort_values(by='合計_SBなし', ascending=False),
                 use_container_width=True)

    tag=f"開催日補正 +{DAY_DELTA.get(day_idx,1)}（有効周回={eff_laps}） / 風向:{st.session_state.get('selected_wind','無風')} / 出走:{n_cars}車 / μ_trim={mu_trim:.2f}"
    st.caption(tag)

# =====================================================
# note記事用（完成形テキストをそのままコピー）
# =====================================================
st.markdown("### 📋 note記事用（コピー可）")

# 追加UI：レース番号 / 開催区分 / 級別
race_no = st.selectbox("レース番号", list(range(1, 13)), index=0)  # 1〜12
race_time = st.selectbox("開催区分", ["モーニング", "デイ", "ナイター", "ミッドナイト"])
race_class = st.selectbox("級別", ["Ａ級チャレンジ", "Ａ級", "Ｓ級", "ガールズ"])

# テキスト生成（スコア順は“紐基準=SBなし”の並びを使用）
line_text = "　".join([x for x in line_inputs if str(x).strip()])
score_order_text = " ".join(str(no) for no, _ in velobi_wo)
marks_line = " ".join(
    f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks
)

note_text = (
    f"競輪場　{selected_track}{race_no}R\n"
    f"{race_time}　{race_class}\n"
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}"
)

st.text_area("ここを選択してコピー", note_text, height=160)

