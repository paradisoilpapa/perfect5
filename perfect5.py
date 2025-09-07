# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re
import math, random, json, itertools

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き）", layout="wide")

# ==============================
# 定数
# ==============================
WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035,
    "無風": 0.0
}
BASE_BY_KAKU = {"逃":1.58, "捲":1.65, "差":1.79, "マ":1.45}

KEIRIN_DATA = {
    "函館":{"bank_angle":30.6,"straight_length":51.3,"bank_length":400},
    "青森":{"bank_angle":32.3,"straight_length":58.9,"bank_length":400},
    "いわき平":{"bank_angle":32.9,"straight_length":62.7,"bank_length":400},
    "弥彦":{"bank_angle":32.4,"straight_length":63.1,"bank_length":400},
    "前橋":{"bank_angle":36.0,"straight_length":46.7,"bank_length":335},
    "取手":{"bank_angle":31.5,"straight_length":54.8,"bank_length":400},
    "宇都宮":{"bank_angle":25.8,"straight_length":63.3,"bank_length":500},
    "大宮":{"bank_angle":26.3,"straight_length":66.7,"bank_length":500},
    "西武園":{"bank_angle":29.4,"straight_length":47.6,"bank_length":400},
    "京王閣":{"bank_angle":32.2,"straight_length":51.5,"bank_length":400},
    "立川":{"bank_angle":31.2,"straight_length":58.0,"bank_length":400},
    "松戸":{"bank_angle":29.8,"straight_length":38.2,"bank_length":333},
    "川崎":{"bank_angle":32.2,"straight_length":58.0,"bank_length":400},
    "平塚":{"bank_angle":31.5,"straight_length":54.2,"bank_length":400},
    "小田原":{"bank_angle":35.6,"straight_length":36.1,"bank_length":333},
    "伊東":{"bank_angle":34.7,"straight_length":46.6,"bank_length":333},
    "静岡":{"bank_angle":30.7,"straight_length":56.4,"bank_length":400},
    "名古屋":{"bank_angle":34.0,"straight_length":58.8,"bank_length":400},
    "岐阜":{"bank_angle":32.3,"straight_length":59.3,"bank_length":400},
    "大垣":{"bank_angle":30.6,"straight_length":56.0,"bank_length":400},
    "豊橋":{"bank_angle":33.8,"straight_length":60.3,"bank_length":400},
    "富山":{"bank_angle":33.7,"straight_length":43.0,"bank_length":333},
    "松坂":{"bank_angle":34.4,"straight_length":61.5,"bank_length":400},
    "四日市":{"bank_angle":32.3,"straight_length":62.4,"bank_length":400},
    "福井":{"bank_angle":31.5,"straight_length":52.8,"bank_length":400},
    "奈良":{"bank_angle":33.4,"straight_length":38.0,"bank_length":333},
    "向日町":{"bank_angle":30.5,"straight_length":47.3,"bank_length":400},
    "和歌山":{"bank_angle":32.3,"straight_length":59.9,"bank_length":400},
    "岸和田":{"bank_angle":30.9,"straight_length":56.7,"bank_length":400},
    "玉野":{"bank_angle":30.6,"straight_length":47.9,"bank_length":400},
    "広島":{"bank_angle":30.8,"straight_length":57.9,"bank_length":400},
    "防府":{"bank_angle":34.7,"straight_length":42.5,"bank_length":333},
    "高松":{"bank_angle":33.3,"straight_length":54.8,"bank_length":400},
    "小松島":{"bank_angle":29.8,"straight_length":55.5,"bank_length":400},
    "高知":{"bank_angle":24.5,"straight_length":52.0,"bank_length":500},
    "松山":{"bank_angle":34.0,"straight_length":58.6,"bank_length":400},
    "小倉":{"bank_angle":34.0,"straight_length":56.9,"bank_length":400},
    "久留米":{"bank_angle":31.5,"straight_length":50.7,"bank_length":400},
    "武雄":{"bank_angle":32.0,"straight_length":64.4,"bank_length":400},
    "佐世保":{"bank_angle":31.5,"straight_length":40.2,"bank_length":400},
    "別府":{"bank_angle":33.7,"straight_length":59.9,"bank_length":400},
    "熊本":{"bank_angle":34.3,"straight_length":60.3,"bank_length":400},
    "手入力":{"bank_angle":30.0,"straight_length":52.0,"bank_length":400},
}

# 直近集計：印別の実測率（%→小数）
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},
}
RANK_FALLBACK_MARK = "α"

# 期待値ルール（固定）
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60

# --- KO(勝ち上がり) 係数（男子のみ有効／ガールズは無効化） ---
KO_GIRLS_SCALE = 0.0               # ガールズは0.0=無効
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.010               # 同ライン連結の“隙間”閾値
KO_STEP_SIGMA = 0.4                # KOランクをスコアに写すときの段差幅(σ倍率)

# === ◎ライン格上げ（A方式：スコア加点） ==============================
LINE_BONUS_ON_TENKAI = {"優位"}   # 展開がこの集合のときだけ発火
LINE_BONUS = {"second": 0.08, "thirdplus": 0.04}  # 役割別ボーナス（番手/三番手）
LINE_BONUS_CAP = 0.10

# === （任意）確率乗数（B方式：デフォルト無効=0.0） ===================
PROB_U = {"second": 0.00, "thirdplus": 0.00}

# ==============================
# ユーティリティ
# ==============================
def clamp(x,a,b): 
    return max(a, min(b, x))

def zscore_list(arr):
    arr = np.array(arr, dtype=float)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return np.zeros_like(arr) if s==0 else (arr-m)/s

def zscore_val(x, xs):
    xs = np.array(xs, dtype=float)
    m, s = float(np.mean(xs)), float(np.std(xs))
    return 0.0 if s==0 else (float(x)-m)/s

def extract_car_list(s, nmax):
    s = str(s or "").strip()
    return [int(c) for c in s if c.isdigit() and 1 <= int(c) <= nmax]

def build_line_maps(lines):
    labels = "ABCDEFG"
    line_def = {labels[i]: lst for i,lst in enumerate(lines) if lst}
    car_to_group = {c:g for g,mem in line_def.items() for c in mem}
    return line_def, car_to_group

def role_in_line(car, line_def):
    for g, mem in line_def.items():
        if car in mem:
            if len(mem)==1: 
                return 'single'
            idx = mem.index(car)
            return ['head','second','thirdplus'][idx] if idx<3 else 'thirdplus'
    return 'single'

def pos_coeff(role, line_factor):
    base = {'head':1.0,'second':0.7,'thirdplus':0.5,'single':0.9}.get(role,0.9)
    return base * line_factor

def tenscore_correction(tenscores):
    n = len(tenscores)
    if n<=2: return [0.0]*n
    df = pd.DataFrame({"得点":tenscores})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n,8)
    baseline = df[df["順位"].between(2,hi)]["得点"].mean()
    def corr(row): 
        return round(abs(baseline-row["得点"])*0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return df.apply(corr, axis=1).tolist()

def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    if wind_dir=="無風" or wind_speed==0: 
        return 0.0
    wd = WIND_COEFF.get(wind_dir,0.0)
    pos_multi = {'head':0.32,'second':0.30,'thirdplus':0.25,'single':0.30}.get(role,0.30)
    coeff = 0.4 + 0.6*prof_escape
    val = wind_speed * wd * pos_multi * coeff
    return round(clamp(val, -0.05, 0.05), 3)

def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi):
    straight_factor = (float(straight_length)-40.0)/10.0
    angle_factor = (float(bank_angle)-25.0)/5.0
    total = clamp(-0.1*straight_factor + 0.1*angle_factor, -0.05, 0.05)
    return round(total*prof_escape - 0.5*total*prof_sashi, 3)

def bank_length_adjust(bank_length, prof_oikomi):
    delta = clamp((float(bank_length)-411.0)/100.0, -0.05, 0.05)
    return round(delta*prof_oikomi, 3)

def compute_lineSB_bonus(line_def, S, B, line_factor=1.0, exclude=None, cap=0.06, enable=True):
    if not enable or not line_def:
        return {g:0.0 for g in line_def.keys()} if line_def else {}, {}
    w_pos_base = {'head':1.0,'second':0.4,'thirdplus':0.2,'single':0.7}
    Sg, Bg = {}, {}
    for g, mem in line_def.items():
        s=b=0.0
        for car in mem:
            if exclude is not None and car==exclude: 
                continue
            w = w_pos_base[role_in_line(car, line_def)] * line_factor
            s += w*float(S.get(car,0)); b += w*float(B.get(car,0))
        Sg[g]=s; Bg[g]=b
    raw={}
    for g in line_def.keys():
        s, b = Sg[g], Bg[g]
        ratioS = s/(s+b+1e-6)
        raw[g] = (0.6*b + 0.4*s) * (0.6 + 0.4*ratioS)
    zz = zscore_list(list(raw.values())) if raw else []
    bonus={g: clamp(0.02*float(zz[i]), -cap, cap) for i,g in enumerate(raw.keys())}
    return bonus, raw

def input_float_text(label: str, key: str, placeholder: str = "") -> float | None:
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "": 
        return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)

# --- KOユーティリティ（ライン対ラインの勝ち上がりシード） ---
def _role_of(car, mem):
    if len(mem)==1: return 'single'
    i = mem.index(car)
    return ['head','second','thirdplus'][i] if i<3 else 'thirdplus'

def _line_strength_raw(line_def, S, B, line_factor=1.0):
    if not line_def: return {}
    w_pos = {'head':1.0,'second':0.4,'thirdplus':0.2,'single':0.7}
    raw={}
    for g, mem in line_def.items():
        s=b=0.0
        for c in mem:
            w = w_pos[_role_of(c, mem)] * line_factor
            s += w*float(S.get(c,0)); b += w*float(B.get(c,0))
        ratioS = s/(s+b+1e-6)
        raw[g] = (0.6*b + 0.4*s) * (0.6 + 0.4*ratioS)
    return raw

def _top2_lines(line_def, S, B, line_factor=1.0):
    raw = _line_strength_raw(line_def, S, B, line_factor)
    order = sorted(raw.keys(), key=lambda g: raw[g], reverse=True)
    return (order[0], order[1]) if len(order)>=2 else (order[0], None) if order else (None, None)

def _extract_role_car(line_def, gid, role_name):
    if gid is None or gid not in line_def: return None
    mem = line_def[gid]
    if role_name=='head':    return mem[0] if len(mem)>=1 else None
    if role_name=='second':  return mem[1] if len(mem)>=2 else None
    return None  # third+ は KO の others プールへ

def _ko_order(v_base_map, line_def, S, B, line_factor=1.0, gap_delta=0.010):
    cars = list(v_base_map.keys())
    if not line_def or len(line_def)<1:
        return [c for c,_ in sorted(v_base_map.items(), key=lambda x:x[1], reverse=True)]

    g1, g2 = _top2_lines(line_def, S, B, line_factor)
    head1 = _extract_role_car(line_def, g1, 'head')
    head2 = _extract_role_car(line_def, g2, 'head')
    sec1  = _extract_role_car(line_def, g1, 'second')
    sec2  = _extract_role_car(line_def, g2, 'second')

    others=[]
    if g1:
        mem = line_def[g1]
        if len(mem)>=3: others += mem[2:]
    if g2:
        mem = line_def[g2]
        if len(mem)>=3: others += mem[2:]
    for g, mem in line_def.items():
        if g not in {g1,g2}:
            others += mem

    order = []
    head_pair = [x for x in [head1, head2] if x is not None]
    order += sorted(head_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    sec_pair = [x for x in [sec1, sec2] if x is not None]
    order += sorted(sec_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    others = list(dict.fromkeys([c for c in others if c is not None]))
    others_sorted = sorted(others, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    order += [c for c in others_sorted if c not in order]

    for c in cars:
        if c not in order:
            order.append(c)

    def _same_group(a,b):
        if a is None or b is None: return False
        ga = next((g for g,mem in line_def.items() if a in mem), None)
        gb = next((g for g,mem in line_def.items() if b in mem), None)
        return ga is not None and ga==gb

    i=0
    while i < len(order)-2:
        a, b, c = order[i], order[i+1], order[i+2]
        if _same_group(a, b):
            vx = v_base_map.get(b,0.0) - v_base_map.get(c,0.0)
            if vx >= -gap_delta:
                order.pop(i+2)
                order.insert(i+1, b)
        i += 1

    return order

# --- ヘルパー：オッズ帯 ---
def _zone_from_p(p: float):
    needed = 1.0 / max(p, 1e-12)
    return needed, needed*(1.0+E_MIN), needed*(1.0+E_MAX)

def _format_line_zone(name: str, bet_type: str, p: float) -> str | None:
    floor = P_FLOOR[bet_type]
    if p < floor:
        return None
    _, low, high = _zone_from_p(p)
    return f"{name}：{low:.1f}〜{high:.1f}倍なら買い"

# --- 並べ替えキー（統一版） ---
def _sort_key_by_numbers(name: str) -> list[int]:
    return list(map(int, re.findall(r"\d+", str(name))))

# === ◎ライン格上げの中核 ===
def apply_anchor_line_bonus(score_raw: dict[int,float],
                            line_of: dict[int,int],
                            role_map: dict[int,str],
                            anchor: int,
                            tenkai: str) -> dict[int,float]:
    a_line = line_of.get(anchor, None)
    is_on = (tenkai in LINE_BONUS_ON_TENKAI) and (a_line is not None)
    score_adj: dict[int,float] = {}
    for i, s in score_raw.items():
        bonus = 0.0
        if is_on and line_of.get(i) == a_line and i != anchor:
            role = role_map.get(i, "single")
            bonus = min(max(0.0, LINE_BONUS.get(role, 0.0)), LINE_BONUS_CAP)
        score_adj[i] = s + bonus
    return score_adj

def format_rank_all(score_map: dict[int,float], P_floor_val: float | None = None) -> str:
    order = sorted(score_map.keys(), key=lambda k: (-score_map[k], k))
    rows = []
    for i in order:
        if P_floor_val is None:
            rows.append(f"{i}")
        else:
            rows.append(f"{i}" if score_map[i] >= P_floor_val else f"{i}(P未満)")
    return " ".join(rows)

# ==============================
# サイドバー：開催情報 / バンク・風・頭数
# ==============================
st.sidebar.header("開催情報 / バンク・風・頭数")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)

track_names = list(KEIRIN_DATA.keys())
track = st.sidebar.selectbox("競輪場（プリセット）", track_names, index=track_names.index("川崎") if "川崎" in track_names else 0)
info = KEIRIN_DATA[track]

wind_dir = st.sidebar.selectbox("風向", ["無風","左上","上","右上","左","右","左下","下","右下"], 0)
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 30.0, 3.0, 0.1)
straight_length = st.sidebar.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle = st.sidebar.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length = st.sidebar.number_input("周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)

base_laps = st.sidebar.number_input("周回（通常4）", 1, 10, 4, 1)
day_label = st.sidebar.selectbox("開催日", ["初日","2日目","最終日"], 0)
eff_laps = int(base_laps) + {"初日":1,"2日目":2,"最終日":3}[day_label]

race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_class = st.sidebar.selectbox("級別", ["Ｓ級","Ａ級","Ａ級チャレンジ","ガールズ"], 0)

angles = [KEIRIN_DATA[k]["bank_angle"] for k in KEIRIN_DATA]
straights = [KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA]
lengths = [KEIRIN_DATA[k]["bank_length"] for k in KEIRIN_DATA]
angle_z = zscore_val(bank_angle, angles)
straight_z = zscore_val(straight_length, straights)
length_z = zscore_val(bank_length, lengths)
style_raw = clamp(0.50*angle_z - 0.35*straight_z - 0.30*length_z, -1.0, +1.0)

override = st.sidebar.slider("会場バイアス補正（−2差し ←→ +2先行）", -2.0, 2.0, 0.0, 0.1)
style = clamp(style_raw + 0.25*override, -1.0, +1.0)

CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

DAY_FACTOR = {"初日":1.00, "2日目":0.60, "最終日":0.85}
day_factor = DAY_FACTOR[day_label]

cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

line_sb_enable = (race_class != "ガールズ")

st.sidebar.caption(
    f"会場スタイル: {style:+.2f}（raw {style_raw:+.2f}） / "
    f"級別: spread={cf['spread']:.2f}, line={cf['line']:.2f} / "
    f"日程係数(line)={day_factor:.2f} → line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}"
)

# ==============================
# メイン
# ==============================
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き）⭐")

# レース番号
st.subheader("レース番号（直前にサクッと変更）")
if "race_no_main" not in st.session_state:
    st.session_state["race_no_main"] = 1
c1, c2, c3 = st.columns([6,2,2])
with c1:
    race_no_input = st.number_input("R", min_value=1, max_value=12, step=1,
                                    value=int(st.session_state["race_no_main"]),
                                    key="race_no_input")
with c2:
    prev_clicked = st.button("◀ 前のR", use_container_width=True)
with c3:
    next_clicked = st.button("次のR ▶", use_container_width=True)
if prev_clicked:
    st.session_state["race_no_main"] = max(1, int(race_no_input) - 1); st.rerun()
elif next_clicked:
    st.session_state["race_no_main"] = min(12, int(race_no_input) + 1); st.rerun()
else:
    st.session_state["race_no_main"] = int(race_no_input)
race_no = int(st.session_state["race_no_main"])

# ライン入力
st.subheader("ライン構成（最大7：単騎も1ライン）")
line_inputs = [
    st.text_input("ライン1（例：317）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：6）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：425）", key="line_3", max_chars=9),
    st.text_input("ライン4（任意）", key="line_4", max_chars=9),
    st.text_input("ライン5（任意）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
]
lines = [extract_car_list(x, n_cars) for x in line_inputs if str(x).strip()]
line_def, car_to_group = build_line_maps(lines)
active_cars = sorted({c for lst in lines for c in lst}) if lines else list(range(1, n_cars+1))

# 個人データ
st.subheader("個人データ（直近4か月：回数）")
cols = st.columns(n_cars)
ratings, S, B = {}, {}, {}
k_esc, k_mak, k_sashi, k_mark = {}, {}, {}, {}
x1, x2, x3, x_out = {}, {}, {}, {}

for i, no in enumerate(active_cars):
    with cols[i]:
        st.markdown(f"**{no}番**")
        ratings[no] = input_float_text("得点（空欄可）", key=f"pt_{no}", placeholder="例: 55.0")
        S[no] = st.number_input("S", 0, 99, 0, key=f"s_{no}")
        B[no] = st.number_input("B", 0, 99, 0, key=f"b_{no}")
        k_esc[no]   = st.number_input("逃", 0, 99, 0, key=f"ke_{no}")
        k_mak[no]   = st.number_input("捲", 0, 99, 0, key=f"km_{no}")
        k_sashi[no] = st.number_input("差", 0, 99, 0, key=f"ks_{no}")
        k_mark[no]  = st.number_input("マ", 0, 99, 0, key=f"kk_{no}")
        x1[no]  = st.number_input("1着", 0, 99, 0, key=f"x1_{no}")
        x2[no]  = st.number_input("2着", 0, 99, 0, key=f"x2_{no}")
        x3[no]  = st.number_input("3着", 0, 99, 0, key=f"x3_{no}")
        x_out[no]= st.number_input("着外", 0, 99, 0, key=f"xo_{no}")

ratings_val = {no: (ratings[no] if ratings[no] is not None else 55.0) for no in active_cars}

# 1着・2着の縮約（級別×会場の事前分布を混ぜる）
def prior_by_class(cls, style_adj):
    if "ガール" in cls: p1,p2 = 0.18,0.24
    elif "Ｓ級" in cls: p1,p2 = 0.22,0.26
    elif "チャレンジ" in cls: p1,p2 = 0.18,0.22
    else: p1,p2 = 0.20,0.25
    p1 += 0.010*style_adj; p2 -= 0.005*style_adj
    return clamp(p1,0.05,0.60), clamp(p2,0.05,0.60)

def n0_by_n(n):
    if n<=6: return 12
    if n<=14: return 8
    if n<=29: return 5
    return 3

p1_eff, p2_eff = {}, {}
for no in active_cars:
    n = x1[no]+x2[no]+x3[no]+x_out[no]
    p1_prior, p2_prior = prior_by_class(race_class, style)
    n0 = n0_by_n(n)
    if n==0:
        p1_eff[no], p2_eff[no] = p1_prior, p2_prior
    else:
        p1_eff[no] = clamp((x1[no] + n0*p1_prior)/(n+n0), 0.0, 0.40)
        p2_eff[no] = clamp((x2[no] + n0*p2_prior)/(n+n0), 0.0, 0.50)

Form = {no: 0.7*p1_eff[no] + 0.3*p2_eff[no] for no in active_cars}

# 脚質プロフィール（会場適性）
prof_base, prof_escape, prof_sashi, prof_oikomi = {}, {}, {}, {}
for no in active_cars:
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark = 0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    prof_escape[no]=esc; prof_sashi[no]=sashi; prof_oikomi[no]=mark
    base = esc*BASE_BY_KAKU["逃"] + mak*BASE_BY_KAKU["捲"] + sashi*BASE_BY_KAKU["差"] + mark*BASE_BY_KAKU["マ"]
    k = 0.06
    venue_bonus = k * style * ( +1.00*esc +0.40*mak -0.60*sashi -0.25*mark )
    prof_base[no] = base + clamp(venue_bonus, -0.06, +0.06)

# ======== 個人補正（得点/脚質上位/着順分布） ========
ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank = {no: i+1 for i,no in enumerate(ratings_sorted)}

def tenscore_bonus(no):
    r = ratings_rank[no]
    top_n = min(3, len(active_cars))
    bottom_n = min(3, len(active_cars))
    if r <= top_n: return +0.03
    if r >= len(active_cars)-bottom_n+1: return -0.02
    return 0.0

def topk_bonus(k_dict, topn=3, val=0.02):
    order = sorted(k_dict.items(), key=lambda x:(x[1], -x[0]), reverse=True)
    grant = set([no for i,(no,v) in enumerate(order) if i<topn])
    return {no:(val if no in grant else 0.0) for no in k_dict}

esc_bonus   = topk_bonus(k_esc,   topn=3, val=0.02)
mak_bonus   = topk_bonus(k_mak,   topn=3, val=0.02)
sashi_bonus = topk_bonus(k_sashi, topn=3, val=0.015)
mark_bonus  = topk_bonus(k_mark,  topn=3, val=0.01)

def finish_bonus(no):
    tot = x1[no]+x2[no]+x3[no]+x_out[no]
    if tot == 0: return 0.0
    in3 = (x1[no]+x2[no]+x3[no]) / tot
    out = x_out[no] / tot
    bonus = 0.0
    if in3 > 0.50: bonus += 0.03
    if out > 0.70: bonus -= 0.03
    if out < 0.40: bonus += 0.02
    return bonus

extra_bonus = {}
for no in active_cars:
    total = (tenscore_bonus(no) +
             esc_bonus.get(no,0.0) + mak_bonus.get(no,0.0) +
             sashi_bonus.get(no,0.0) + mark_bonus.get(no,0.0) +
             finish_bonus(no))
    extra_bonus[no] = clamp(total, -0.10, +0.10)

# SBなし合計（環境補正 + 得点微補正 + 個人補正）
tens_list = [ratings_val[no] for no in active_cars]
t_corr = tenscore_correction(tens_list) if active_cars else []
tens_corr = {no:t_corr[i] for i,no in enumerate(active_cars)} if active_cars else {}

rows=[]
for no in active_cars:
    role = role_in_line(no, line_def)
    wind = wind_adjust(wind_dir, wind_speed, role, prof_escape[no])
    extra = max(eff_laps-2, 0)
    fatigue_scale = 1.0 if race_class=="Ｓ級" else (1.1 if race_class=="Ａ級" else (1.2 if race_class=="Ａ級チャレンジ" else 1.05))
    laps_adj = (-0.10*extra*(1.0 if prof_escape[no]>0.5 else 0.0) + 0.05*extra*(1.0 if prof_oikomi[no]>0.4 else 0.0)) * fatigue_scale
    bank_b = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no])
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv = extra_bonus.get(no, 0.0)

    total_raw = (prof_base[no] + wind + cf["spread"]*tens_corr.get(no,0.0) + bank_b + length_b + laps_adj + indiv)
    rows.append([no, role, round(prof_base[no],3), wind, round(cf["spread"]*tens_corr.get(no,0.0),3),
                 round(bank_b,3), round(length_b,3), round(laps_adj,3), round(indiv,3), total_raw])

df = pd.DataFrame(rows, columns=["車番","役割","脚質基準(会場)","風補正","得点補正","バンク補正","周長補正","周回補正","個人補正","合計_SBなし_raw"])
mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0*(df["合計_SBなし_raw"] - mu)

# ===== KO方式：最終並びの反映（男子のみ／ガールズは無効） =====
# キー型不一致対策：必ず int キー / float 値に統一
v_wo = {
    int(k): float(v)
    for k, v in zip(df["車番"].astype(int), df["合計_SBなし"].astype(float))
}

_is_girls = (race_class == "ガールズ")
head_scale = KO_HEADCOUNT_SCALE.get(int(n_cars), 1.0)
ko_scale = (KO_GIRLS_SCALE if _is_girls else 1.0) * head_scale  # ガールズは0.0で無効

if ko_scale > 0.0 and line_def and len(line_def) >= 1:
    ko_order = _ko_order(v_wo, line_def, S, B, line_factor=line_factor_eff, gap_delta=KO_GAP_DELTA)
    vals = [v_wo[c] for c in v_wo.keys()]
    mu0  = float(np.mean(vals)); sd0 = float(np.std(vals) + 1e-12)
    step = KO_STEP_SIGMA * sd0
    new_scores = {}
    for rank, car in enumerate(ko_order, start=1):
        rank_adjust = step * (len(ko_order) - rank)
        blended = (1.0 - ko_scale) * v_wo[car] + ko_scale * (mu0 + rank_adjust - (len(ko_order)/2.0 - 0.5)*step)
        new_scores[car] = blended
    # ここでも必ず int/float に統一
    v_final = {int(k): float(v) for k, v in new_scores.items()}
else:
    v_final = {int(k): float(v) for k, v in v_wo.items()}

# --- 純SBなしランキング（KOまで／格上げ前）---
df_sorted_pure = pd.DataFrame({
    "車番": list(v_final.keys()),
    "合計_SBなし": [round(float(v_final[c]), 6) for c in v_final.keys()]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

# --- 一旦のランキング（◎選出の内部参照用・表示は後で格上げ版で上書き） ---
df_sorted_wo_tmp = pd.DataFrame({
    "車番": active_cars,
    "合計_SBなし": [round(float(v_final.get(int(c), float("-inf"))), 6) for c in active_cars]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)


# ===== ここから（印選定→◎確定） =====
# 候補C（得点×2着率ブレンド 上位3）
blend = {no: (ratings_val[no] + min(50.0, p2_eff[no]*100.0))/2.0 for no in active_cars}
C = [kv[0] for kv in sorted(blend.items(), key=lambda x:x[1], reverse=True)[:min(3,len(blend))]]

# ラインSB（◎選出用）
bonus_init,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)

def anchor_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    sb = bonus_init.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
    zt_map = {n:float(zt[i]) for i,n in enumerate(active_cars)} if active_cars else {}
    return v_final.get(no, -1e9) + sb + 0.01*zt_map.get(no, 0.0)

anchor_no_pre = max(C, key=lambda x: anchor_score(x)) if C else int(df_sorted_wo_tmp.loc[0,"車番"])

ratings_sorted2 = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank2 = {no: i+1 for i, no in enumerate(ratings_sorted2)}
ALLOWED_MAX_RANK = 4
C_hard = [no for no in C if ratings_rank2.get(no, 999) <= ALLOWED_MAX_RANK]
C_use = C_hard if C_hard else ratings_sorted2[:ALLOWED_MAX_RANK]
anchor_no = max(C_use, key=lambda x: anchor_score(x))

if anchor_no != anchor_no_pre:
    st.caption(f"※ ◎は『競走得点 上位{ALLOWED_MAX_RANK}位以内』縛りにより {anchor_no_pre}→{anchor_no} に調整しています。")

# --- ◎ライン格上げ（A方式）適用：表示用スコアを上書き ---
role_map = {no: role_in_line(no, line_def) for no in active_cars}
confidence = None  # 下で計算

# 仮の信頼度を先に算出（従来ロジック）
cand_scores = [anchor_score(no) for no in C] if len(C)>=2 else [0,0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf_gap = cand_scores_sorted[0]-cand_scores_sorted[1] if len(cand_scores_sorted)>=2 else 0.0
spread = float(np.std(list(v_final.values()))) if len(v_final)>=2 else 0.0
norm = conf_gap / (spread if spread>1e-6 else 1.0)
confidence = "優位" if norm>=1.0 else ("互角" if norm>=0.5 else "混戦")

score_adj_map = apply_anchor_line_bonus(
    score_raw=v_final,
    line_of=car_to_group,
    role_map=role_map,
    anchor=anchor_no,
    tenkai=confidence
)

# 表示・note・買い目の“SBなしランキング”は格上げ後で統一（フォールバックに -1e9 は使わない）
df_sorted_wo = pd.DataFrame({
    "車番": active_cars,
    "合計_SBなし": [round(float(score_adj_map.get(int(c), v_final.get(int(c), float("-inf")))), 6) for c in active_cars]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

velobi_wo = list(zip(df_sorted_wo["車番"].astype(int).tolist(),
                     df_sorted_wo["合計_SBなし"].round(3).tolist()))

# 印集約（◎ライン優先：同ラインを上から順に採用）
rank_wo = {int(df_sorted_wo.loc[i, "車番"]): i+1 for i in range(len(df_sorted_wo))}
result_marks, reasons = {}, {}
result_marks["◎"] = anchor_no
reasons[anchor_no] = "本命(C上位3→得点4位以内ゲート→ラインSB重視＋KO並び)"

# スコア辞書（格上げ後）
score_map = {int(df_sorted_wo.loc[i, "車番"]): float(df_sorted_wo.loc[i, "合計_SBなし"])
             for i in range(len(df_sorted_wo))}

# 全体並び（◎除外）
overall_rest = [int(df_sorted_wo.loc[i, "車番"])
                for i in range(len(df_sorted_wo))
                if int(df_sorted_wo.loc[i, "車番"]) != anchor_no]

# ◎のラインメンバー（◎を除外）をスコア降順に
a_gid = car_to_group.get(anchor_no, None)
mates_sorted = []
if a_gid is not None and a_gid in line_def:
    mates_sorted = sorted(
        [c for c in line_def[a_gid] if c != anchor_no],
        key=lambda x: (-score_map.get(x, -1e9), x)
    )

# 〇：全体トップ（◎除外）
if overall_rest:
    result_marks["〇"] = overall_rest[0]
    reasons[overall_rest[0]] = "対抗（格上げ後SBなしスコア順）"

used = set(result_marks.values())

# ▲：◎ラインから最上位を“強制”採用（〇が同ラインなら次点）
mate_candidates = [c for c in mates_sorted if c not in used]
if mate_candidates:
    pick = mate_candidates[0]
    result_marks["▲"] = pick
    reasons[pick] = "単穴（◎ライン優先：同ライン最上位を採用）"
else:
    # 同ラインに候補が無い（単騎など）のときは全体次点
    rest_global = [c for c in overall_rest if c not in used]
    if rest_global:
        pick = rest_global[0]
        result_marks["▲"] = pick
        reasons[pick] = "単穴（格上げ後SBなしスコア順）"

used = set(result_marks.values())

# 残り印（△ → × → α → β）は“◎ライン残メンバーを先に消化”、その後に全体残り
tail_priority = [c for c in mates_sorted if c not in used]
tail_priority += [c for c in overall_rest if c not in used and c not in tail_priority]

for mk in ["△","×","α","β"]:
    if mk in result_marks:
        continue
    if not tail_priority:
        break
    no = tail_priority.pop(0)
    result_marks[mk] = no
    reasons[no] = f"{mk}（◎ライン優先→残りスコア順）"

# 出力（SBなしランキング）
st.markdown("### ランキング＆印（◎ライン格上げ反映済み）")
rows_out=[]
for r,(no,sc) in enumerate(velobi_wo, start=1):
    mark = "".join([m for m,v in result_marks.items() if v==no])
    n_tot = x1.get(no,0)+x2.get(no,0)+x3.get(no,0)+x_out.get(no,0)
    p1 = (x1.get(no,0)/(n_tot+1e-9))*100
    p2 = (x2.get(no,0)/(n_tot+1e-9))*100
    rows_out.append({
        "順(SBなし)": r, "印": mark, "車": no,
        "SBなしスコア": sc,
        "得点": ratings_val.get(no, None),
        "1着回": x1.get(no,0), "2着回": x2.get(no,0), "3着回": x3.get(no,0), "着外": x_out.get(no,0),
        "1着%": round(p1,1), "2着%": round(p2,1),
        "ライン": car_to_group.get(no,"-")
    })
st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

st.markdown("#### 補正内訳（SBなし）")
show=[]
for no,_ in velobi_wo:
    rec = df[df["車番"]==no].iloc[0]
    show.append({
        "車":int(no),"ライン":car_to_group.get(int(no),"-"),
        "脚質基準(会場)":round(rec["脚質基準(会場)"],3),
        "風補正":rec["風補正"],"得点補正":rec["得点補正"],
        "バンク補正":rec["バンク補正"],"周長補正":rec["周長補正"],
        "周回補正":rec["周回補正"],"個人補正":rec["個人補正"],
        "合計_SBなし_raw":round(rec["合計_SBなし_raw"],3),
        "合計_SBなし":round(rec["合計_SBなし"],3)
    })
st.dataframe(pd.DataFrame(show), use_container_width=True)

# 「スコア順（SBなし）」のnote表記用（格上げ前＝KOまで）
score_order_text = format_rank_all(
    {int(r["車番"]): float(r["合計_SBなし"]) for _, r in df_sorted_pure.iterrows()},
    P_floor_val=None
)


st.caption(
    f"競輪場　{track}{race_no}R / {race_time}　{race_class} / "
    f"開催日：{day_label}（line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}） / "
    f"会場スタイル:{style:+.2f} / 風:{wind_dir} / 有効周回={eff_laps} / 展開評価：**{confidence}**（Norm={norm:.2f})"
)

# ==============================
# 買い目（想定的中率 → 必要オッズ=1/p）
# ==============================
st.markdown("### 🎯 買い目（想定的中率 → 必要オッズ=1/p）")

one = result_marks.get("◎", None)
two = result_marks.get("〇", None)
three = result_marks.get("▲", None)

if one is None:
    st.warning("◎未決定のため買い目はスキップ")
    trioC_df = wide_df = qn_df = ex_df = santan_df = None
else:
    # base：格上げ後スコア → softmax
    strength_map = dict(velobi_wo)
    xs = np.array([strength_map.get(i, 0.0) for i in range(1, n_cars+1)], dtype=float)
    if xs.std() < 1e-12:
        base = np.ones_like(xs)/len(xs)
    else:
        z = (xs - xs.mean())/(xs.std()+1e-12)
        base = np.exp(z); base = base/base.sum()

    mark_by_car = {car: None for car in range(1, n_cars+1)}
    for mk, car in result_marks.items():
        if car is not None and 1 <= car <= n_cars:
            mark_by_car[car] = mk

    # ★キャリブレーション弱体化（固定化防止＋“1.0倍地獄”回避）
    def calibrate_probs(base_vec: np.ndarray, stat_key: str) -> np.ndarray:
        base_norm = base_vec / max(base_vec.sum(), 1e-12)
        m = np.ones(n_cars, dtype=float)
        expo_map = {"優位": 0.60, "互角": 0.80, "混戦": 1.00}
        expo_eff = expo_map.get(confidence, 0.80)
        for idx, car in enumerate(range(1, n_cars+1)):
            mk = mark_by_car.get(car)
            if mk not in RANK_STATS:
                mk = RANK_FALLBACK_MARK
            tgt = float(RANK_STATS[mk][stat_key])
            ratio = tgt / max(float(base_norm[idx]), 1e-9)
            m[idx] = float(np.clip(ratio ** expo_eff, 0.70, 1.50))
        probs = base_norm * m
        probs = probs / max(probs.sum(), 1e-12)
        # さらにフラット化（固定メタ防止）
        alpha = 0.15
        probs = (1.0 - alpha) * probs + alpha * (np.ones_like(probs) / len(probs))
        return probs

    probs_p3 = calibrate_probs(base, "pTop3")
    probs_p2 = calibrate_probs(base, "pTop2")
    probs_p1 = calibrate_probs(base, "p1")

    rng = np.random.default_rng(20250830)
    trials = st.slider("シミュレーション試行回数", 1000, 20000, 8000, 1000)

    # ★サンプリングの温度付け（序列の硬直回避）
    def sample_order_from_probs(pvec: np.ndarray, tau: float = 1.6) -> list[int]:
        logits = np.log(np.clip(pvec, 1e-12, 1.0)) / tau
        g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
        score = logits + g
        return (np.argsort(-score)+1).tolist()

    mates = [x for x in [two, three] if x is not None]
    all_others = [i for i in range(1, n_cars+1) if i != one]

    trioC_counts = {}
    wide_counts = {k:0 for k in all_others}
    qn_counts   = {k:0 for k in all_others}
    ex_counts   = {k:0 for k in all_others}
    st3_counts  = {}

    trioC_list = []
    if len(mates) > 0:
        for a in all_others:
            for b in all_others:
                if a >= b: continue
                if (a in mates) or (b in mates):
                    t = tuple(sorted([a, b, one]))
                    trioC_list.append(t)
        trioC_list = sorted(set(trioC_list))

    for _ in range(trials):
        order_p3 = sample_order_from_probs(probs_p3)
        top3_p3 = set(order_p3[:3])

        if one in top3_p3:
            for k in wide_counts.keys():
                if k in top3_p3:
                    wide_counts[k] += 1
            if len(trioC_list) > 0:
                others = list(top3_p3 - {one})
                if len(others) == 2:
                    a, b = sorted(others)
                    if (a in mates) or (b in mates):
                        t = tuple(sorted([a, b, one]))
                        if t in trioC_list:
                            trioC_counts[t] = trioC_counts.get(t, 0) + 1

        order_p2 = sample_order_from_probs(probs_p2)
        top2_p2 = set(order_p2[:2])
        if one in top2_p2:
            for k in qn_counts.keys():
                if k in top2_p2:
                    qn_counts[k] += 1

        order_p1 = sample_order_from_probs(probs_p1)
        if order_p1[0] == one:
            k2 = order_p1[1]
            if k2 in ex_counts:
                ex_counts[k2] += 1
            if len(mates) > 0 and len(order_p1) >= 3:
                k3 = order_p1[2]
                if (k2 in mates) and (k3 not in (one, k2)):
                    st3_counts[(k2, k3)] = st3_counts.get((k2, k3), 0) + 1

    # （任意）B方式の微小ブースト
    if any(v > 0 for v in PROB_U.values()):
        a_line = car_to_group.get(one, None)
        def role_of(i): return role_in_line(i, line_def)
        # ワイド
        for k in list(wide_counts.keys()):
            if a_line is not None and car_to_group.get(k) == a_line and k != one:
                u = PROB_U.get(role_of(k), 0.0)
                if u > 0.0:
                    wide_counts[k] = int(round(wide_counts[k] * (1.0 + u)))
        # 三連複C
        new_trioC_counts = {}
        for t, cnt in trioC_counts.items():
            factor = 1.0
            for x in t:
                if x == one: 
                    continue
                if a_line is not None and car_to_group.get(x) == a_line:
                    u = PROB_U.get(role_of(x), 0.0)
                    factor *= (1.0 + u)
            new_trioC_counts[t] = int(round(cnt * factor))
        trioC_counts = new_trioC_counts

    # PフロアとEV帯（開催の混線度で微調整）
    P_FLOOR = globals().get("P_FLOOR", {"wide": 0.060, "sanpuku": 0.040, "nifuku": 0.050, "nitan": 0.040, "santan": 0.030})
    scale = 1.00
    if confidence == "優位":   scale = 0.90
    elif confidence == "混戦": scale = 1.10
    for k in ("wide","sanpuku","nifuku"):
        P_FLOOR[k] *= scale

    E_MIN = globals().get("E_MIN", 0.00)
    E_MAX = globals().get("E_MAX", 0.50)

    # ===== 三連複C =====
    if len(trioC_list) > 0:
        rows = []
        for t in trioC_list:
            cnt = int(trioC_counts.get(t, 0) or 0)
            p = cnt / float(trials)
            rows.append({
                "買い目": f"{t[0]}-{t[1]}-{t[2]}",
                "p(想定的中率)": round(p, 4),
                "必要オッズ(=1/p)": "-" if cnt==0 else round(1.0/max(p,1e-12), 2)
            })
        trioC_df = pd.DataFrame(rows)
        st.markdown("#### 三連複C（◎-[相手]-全）※車番順")
        def _key_nums_tri(s): return list(map(int, re.findall(r"\d+", s)))
        trioC_df = trioC_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_tri)).reset_index(drop=True)
        st.dataframe(trioC_df, use_container_width=True)
    else:
        trioC_df = None
        st.info("三連複C：相手（〇/▲）が未設定のため表示なし")

    # 三連複バスケット合成オッズと相手集合S
    Sset = set()
    O_combo = None
    if trioC_df is not None and len(trioC_df) > 0:
        need_list = []
        for _, r in trioC_df.iterrows():
            name = str(r["買い目"])
            nums = list(map(int, re.findall(r"\d+", name)))
            others = [x for x in nums if x != one]
            Sset.update(others)
            need_val = r.get("必要オッズ(=1/p)")
            if isinstance(need_val, (int, float)) and float(need_val) > 0:
                need_list.append(float(need_val))
        if need_list:
            denom = sum(1.0/x for x in need_list if x > 0)
            if denom > 0:
                O_combo = float(f"{(1.0 / denom):.2f}")

    # ===== ワイド（◎-全） =====
    rows = []
    for k in sorted([i for i in range(1, n_cars+1) if i != one]):
        cnt = int(wide_counts.get(k, 0) or 0)
        p = cnt / float(trials)
        if p < float(P_FLOOR.get("wide", 0.06)):  # Pフロア（命）
            continue
        if cnt <= 0:
            continue
        need = 1.0 / max(p, 1e-12)  # EV下限
        # 三連複の相手集合Sに該当する場合は「合成オッズ」も下限に加える
        if (O_combo is not None) and (k in Sset):
            need = max(need, float(O_combo))
            rule_note = f"三複被り→合成{float(O_combo):.2f}倍以上"
        else:
            rule_note = "必要オッズ以上"
        if not np.isfinite(need) or need <= 0:
            continue
        rows.append({
            "買い目": f"{one}-{k}",
            "p(想定的中率)": round(p, 4),
            "必要オッズ(=1/p)": round(need, 2),
            "ルール": rule_note
        })
    wide_df = pd.DataFrame(rows)
    st.markdown("#### ワイド（◎-全）※車番順")
    if len(wide_df) > 0:
        wide_df = wide_df.sort_values(by="買い目", key=lambda s: s.map(_sort_key_by_numbers)).reset_index(drop=True)
        st.dataframe(wide_df, use_container_width=True)
        if O_combo is not None:
            st.caption("※三連複で使用した相手（S側）は **max(必要オッズ, 合成オッズ)** 以上で採用。S外は **必要オッズ以上**で採用。")
        else:
            st.caption("※ワイドは **必要オッズ(=1/p)以上**で採用（上限撤廃）。")
    else:
        st.info("ワイド：対象外（Pフロア未満、または合成オッズ基準で除外）")

    # ===== 二車複 =====
    rows = []
    for k in sorted([i for i in range(1, n_cars+1) if i != one]):
        cnt = int(qn_counts.get(k, 0) or 0)
        p = cnt / float(trials)
        if p < float(P_FLOOR.get("nifuku", 0.05)):  # Pフロア（命）
            continue
        if cnt <= 0:
            continue
        need = 1.0 / max(p, 1e-12)  # EV下限
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({
            "買い目": f"{one}-{k}",
            "p(想定的中率)": round(p, 4),
            "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"
        })
    qn_df = pd.DataFrame(rows)
    st.markdown("#### 二車複（◎-全）※車番順")
    if len(qn_df) > 0:
        def _key_nums_qn(s): return list(map(int, re.findall(r"\d+", s)))
        qn_df = qn_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_qn)).reset_index(drop=True)
        st.dataframe(qn_df, use_container_width=True)
    else:
        st.info("二車複：対象外")

    # ===== 二車単 =====
    rows = []
    for k in sorted([i for i in range(1, n_cars+1) if i != one]):
        cnt = int(ex_counts.get(k, 0) or 0)
        p = cnt / float(trials)
        if p < float(P_FLOOR.get("nitan", 0.04)):  # Pフロア（命）
            continue
        if cnt <= 0:
            continue
        need = 1.0 / max(p, 1e-12)  # EV下限
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({
            "買い目": f"{one}->{k}",
            "p(想定的中率)": round(p, 4),
            "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"
        })
    ex_df = pd.DataFrame(rows)
    st.markdown("#### 二車単（◎→全）※車番順")
    if len(ex_df) > 0:
        def _key_nums_ex(s): return list(map(int, re.findall(r"\d+", s)))
        ex_df = ex_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_ex)).reset_index(drop=True)
        st.dataframe(ex_df, use_container_width=True)
    else:
        st.info("二車単：対象外")

    # ===== 三連単（◎→[相手]→全） =====
    rows = []
    p_floor_santan = float(P_FLOOR.get("santan", 0.03))
    for (sec, thr), cnt in st3_counts.items():
        cnt = int(cnt or 0)
        if cnt <= 0:
            continue
        p = cnt / float(trials)
        if p < p_floor_santan:  # Pフロア（命）
            continue
        need = 1.0 / max(p, 1e-12)  # EV下限
        low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
        rows.append({
            "買い目": f"{one}->{sec}->{thr}",
            "p(想定的中率)": round(p, 5),
            "買える帯": f"{low:.1f}〜{high:.1f}倍なら買い"
        })
    if rows:
        santan_df = pd.DataFrame(rows)
        def _key_nums_st(s): return list(map(int, re.findall(r"\d+", s)))
        santan_df = santan_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_st)).reset_index(drop=True)
        st.markdown("#### 三連単（◎→[相手]→全）※車番順")
        st.dataframe(santan_df, use_container_width=True)
    else:
        santan_df = None
        st.info("三連単：対象外（Pフロア未満・相手未設定・該当なし）")

# ==============================
# note用：ヘッダー〜展開評価＋“買えるオッズ帯”
# ==============================
st.markdown("### 📋 note用（ヘッダー〜展開評価＋“買えるオッズ帯”）")

def _zone_lines_from_df(df: pd.DataFrame | None, bet_type_key: str) -> list[str]:
    """
    DataFrame から note 出力用の「買える帯」行を安全に作る。
    - '買える帯' があればそれを優先
    - 無ければ '必要オッズ(=1/p)' から帯を作る（wide は '以上で買い'、その他は EV 帯）
    - いずれも無ければスキップ
    返り値は '買い目：テキスト' の完全な行の配列
    """
    if df is None or len(df) == 0 or ("買い目" not in df.columns):
        return []

    out_rows: list[tuple[str, str]] = []  # (name, line_text)
    for _, r in df.iterrows():
        name = str(r.get("買い目", "")).strip()
        if not name:
            continue

        # 1) 既に「買える帯」があるならそれを使う
        line_txt = None
        if "買える帯" in r and pd.notna(r["買える帯"]):
            s = str(r["買える帯"]).strip()
            if s:
                line_txt = f"{name}：{s}"

        # 2) 無ければ「必要オッズ(=1/p)」から作る
        if line_txt is None:
            need_val = r.get("必要オッズ(=1/p)")
            if need_val is not None and need_val != "-" and str(need_val).strip() != "":
                try:
                    need = float(need_val)
                    if np.isfinite(need) and need > 0:
                        if bet_type_key == "wide":
                            line_txt = f"{name}：{need:.1f}倍以上で買い"
                        else:
                            low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
                            line_txt = f"{name}：{low:.1f}〜{high:.1f}倍なら買い"
                except Exception:
                    pass  # 変換失敗は無視

        if line_txt:
            out_rows.append((name, line_txt))

    # 買い目の数字順に並べ替え
    out_rows_sorted = sorted(out_rows, key=lambda x: _sort_key_by_numbers(x[0]))
    # ここで完成テキストだけ返す（splitは使わない）
    return [t for _, t in out_rows_sorted]


def _section_text(title: str, lines: list[str]) -> str:
    if not lines: return f"{title}\n対象外"
    return f"{title}\n" + "\n".join(lines)

line_text = "　".join([x for x in line_inputs if str(x).strip()])
# スコア順（SBなし）は格上げ適用後の df_sorted_wo から必ず作る
score_map_for_note = {int(r["車番"]): float(r["合計_SBなし"]) for _, r in df_sorted_wo.iterrows()}
score_order_text = format_rank_all(score_map_for_note, P_floor_val=None)
marks_line = " ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)

txt_trioC = _section_text("三連複C（◎-[相手]-全）",
                          _zone_lines_from_df(trioC_df, "sanpuku") if one is not None else [])
txt_st    = _section_text("三連単（◎→[相手]→全）",
                          _zone_lines_from_df(santan_df, "santan") if one is not None else [])
txt_wide  = _section_text("ワイド（◎-全）",
                          _zone_lines_from_df(wide_df, "wide") if one is not None else [])
txt_qn    = _section_text("二車複（◎-全）",
                          _zone_lines_from_df(qn_df, "nifuku") if one is not None else [])
txt_ex    = _section_text("二車単（◎→全）",
                          _zone_lines_from_df(ex_df, "nitan") if one is not None else [])

wide_rule_note = "（ワイドは上限撤廃：三連複で使用した相手は合成オッズ以上／三連複から漏れた相手は必要オッズ以上で買い）"

note_text = (
    f"競輪場　{track}{race_no}R\n"
    f"展開評価：{confidence}\n"
    f"{race_time}　{race_class}\n"
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}\n"
    f"\n"
    f"{txt_trioC}\n\n"
    f"{txt_st}\n\n"
    f"{txt_wide}\n\n"
    f"{txt_qn}\n\n"
    f"{txt_ex}\n"
    f"\n（※“対象外”＝Pフロア未満。どんなオッズでも買わない）\n"
    f"{wide_rule_note}"
)

st.text_area("ここを選択してコピー", note_text, height=380)
