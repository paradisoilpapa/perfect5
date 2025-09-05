# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re, math, random, json
import itertools

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き / 改修統合版）", layout="wide")

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

# --- 直近集計：印別の実測率（%→小数） ---
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},  # N=98
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},  # N=93
}

# 印が付かない車（8〜9車時の余り）へのフォールバック
RANK_FALLBACK_MARK = "α"

# ===== 期待値ルール（固定） =====
P_FLOOR = {
    "sanpuku": 0.06,  # 三連複
    "nifuku" : 0.12,  # 二車複（7車）
    "wide"   : 0.25,  # ワイド
    "nitan"  : 0.07,  # 二車単
}
E_MIN, E_MAX = 0.10, 0.60  # EV +10% ～ +60%（買える帯）

# ==============================
# ユーティリティ
# ==============================
def clamp(x,a,b): return max(a, min(b, x))

def zscore_list(arr):
    arr = np.array(arr, dtype=float)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return np.zeros_like(arr) if s==0 else (arr-m)/s

def zscore_val(x, xs):
    xs = np.array(xs, dtype=float); m, s = float(np.mean(xs)), float(np.std(xs))
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
            if len(mem)==1: return 'single'
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
    hi = min(n,8); baseline = df[df["順位"].between(2,hi)]["得点"].mean()
    def corr(row): return round(abs(baseline-row["得点"])*0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return df.apply(corr, axis=1).tolist()

def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    if wind_dir=="無風" or wind_speed==0: return 0.0
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
            if exclude is not None and car==exclude: continue
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
    return bonus, raw  # ← rawを返す（後段のライン評価で使用）

def input_float_text(label: str, key: str, placeholder: str = "") -> float | None:
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "": return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)

# ==== ゾーン出力ヘルパー（文章形式・車番順） ====
def _zone_from_p(p: float) -> tuple[float,float,float]:
    needed = 1.0 / max(p, 1e-12)
    return needed, needed*(1.0+E_MIN), needed*(1.0+E_MAX)

def _format_line_zone(name: str, bet_type: str, p: float) -> str | None:
    floor = P_FLOOR[bet_type]
    if p < floor:
        return None
    _, low, high = _zone_from_p(p)
    return f"{name}：{low:.1f}〜{high:.1f}倍なら買い"

def _sort_key_by_numbers(name: str) -> list[int]:
    return list(map(int, re.findall(r"\d+", str(name))))

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

# === 改修: 天候トグル（簡易・経験則） ===
weather = st.sidebar.selectbox("天候", ["晴れ","雨"], index=0)

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
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き・改修統合）⭐")

# ▼▼ レース番号 ▼▼
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
# ▲▲

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

# 得点の実数（未入力は55.0）
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

# SBなし合計（環境補正 + 得点微補正）
tens_list = [ratings_val[no] for no in active_cars]
t_corr = tenscore_correction(tens_list) if active_cars else []
tens_corr = {no:t_corr[i] for i,no in enumerate(active_cars)} if active_cars else {}

rows=[]
for no in active_cars:
    role = role_in_line(no, line_def)
    wind = wind_adjust(wind_dir, wind_speed, role, prof_escape[no])
    # === 改修: 雨補正（簡易） ===
    rain_adj = 0.0
    if weather == "雨":
        rain_adj = (-0.03*prof_escape[no] + 0.02*prof_sashi[no] + 0.01*prof_oikomi[no])
    extra = max(eff_laps-2, 0)
    fatigue_scale = 1.0 if race_class=="Ｓ級" else (1.1 if race_class=="Ａ級" else (1.2 if race_class=="Ａ級チャレンジ" else 1.05))
    laps_adj = (-0.10*extra*(1.0 if prof_escape[no]>0.5 else 0.0) + 0.05*extra*(1.0 if prof_oikomi[no]>0.4 else 0.0)) * fatigue_scale
    bank_b = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no])
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    total_raw = (prof_base[no] + wind + cf["spread"]*tens_corr.get(no,0.0) + bank_b + length_b + laps_adj + rain_adj)
    rows.append([no, role, round(prof_base[no],3), wind, round(cf["spread"]*tens_corr.get(no,0.0),3),
                 round(bank_b,3), round(length_b,3), round(laps_adj,3), round(rain_adj,3), total_raw])

df = pd.DataFrame(rows, columns=["車番","役割","脚質基準(会場)","風補正","得点補正","バンク補正","周長補正","周回補正","雨補正","合計_SBなし_raw"])
mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0*(df["合計_SBなし_raw"] - mu)
df_sorted_wo = df.sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

# 候補C（得点×2着率ブレンド 上位3）
blend = {no: (ratings_val[no] + min(50.0, p2_eff[no]*100.0))/2.0 for no in active_cars}
C = [kv[0] for kv in sorted(blend.items(), key=lambda x:x[1], reverse=True)[:min(3,len(blend))]]

# ラインSB
bonus_init, raw_line = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)
v_wo = dict(zip(df["車番"], df["合計_SBなし"]))

# === 改修: ◎（本命）ソフト制約＋底抜け防止 ===
def anchor_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    sb = bonus_init.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
    zt_map = {n:float(zt[i]) for i,n in enumerate(active_cars)} if active_cars else {}
    base = v_wo.get(no, -1e9) + sb + 0.01*zt_map.get(no, 0.0)
    # 得点順位ペナルティ（上位4外に滑らかに）
    ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
    ratings_rank = {n:i+1 for i,n in enumerate(ratings_sorted)}
    rank = ratings_rank.get(no, 999)
    penalty = 0.0
    if rank > 4:
        penalty = -0.02*(rank-4)  # 緩やか
    return base + penalty

anchor_no_pre = max(C, key=lambda x: anchor_score(x)) if C else int(df_sorted_wo.loc[0,"車番"])

ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank = {no: i+1 for i, no in enumerate(ratings_sorted)}
ALLOWED_MAX_RANK = 4

# “昇格チケット”：上位外でもOK条件
def has_promotion_ticket(no):
    # ラインSB優位＋会場適合＋最低フォーム
    g = car_to_group.get(no, None)
    sb_ok = bonus_init.get(g, 0.0) > 0.0
    vm_ok = (style * (1.0*prof_escape[no] + 0.4*(k_mak[no]/max(1,k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no])) - 0.6*prof_sashi[no] - 0.25*prof_oikomi[no])) >= -0.05
    form_ok = p2_eff[no] >= (0.22 if race_class=="Ｓ級" else 0.20)
    return sb_ok and vm_ok and form_ok

C_hard = [no for no in C if ratings_rank.get(no, 999) <= ALLOWED_MAX_RANK]
C_use = C_hard if C_hard else ratings_sorted[:ALLOWED_MAX_RANK]
anchor_no = max(C_use+[anchor_no_pre], key=lambda x: anchor_score(x) if (ratings_rank.get(x,999)<=4 or has_promotion_ticket(x)) else -1e9)
if anchor_no != anchor_no_pre:
    st.caption(f"※ ◎は『得点上位4ソフト制約』により {anchor_no_pre}→{anchor_no} に調整（昇格条件を満たす場合のみ）。")

cand_scores = [anchor_score(no) for no in C] if len(C)>=2 else [0,0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf = cand_scores_sorted[0]-cand_scores_sorted[1] if len(cand_scores_sorted)>=2 else 0.0
spread = float(np.std(list(v_wo.values()))) if len(v_wo)>=2 else 0.0
norm = conf / (spread if spread>1e-6 else 1.0)
confidence = "優位" if norm>=1.0 else ("互角" if norm>=0.5 else "混線")

bonus_re,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=anchor_no, cap=cap_SB_eff, enable=line_sb_enable)
def himo_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    sb = bonus_re.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    return v_wo.get(no, -1e9) + sb

restC = [no for no in C if no!=anchor_no]
o_no = max(restC, key=lambda x: himo_score(x)) if restC else None

def venue_match(no):
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark=0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    return style * (1.00*esc + 0.40*mak - 0.60*sashi - 0.25*mark)

rank_wo = {int(df_sorted_wo.loc[i,"車番"]): i+1 for i in range(len(df_sorted_wo))}

# === 改修: ▲の男女別ロジック ===
def pick_sanha():
    # ガールズ：下位40〜100%帯から（穴寄せ）
    if race_class == "ガールズ":
        thr = max(1, int(math.ceil(len(df_sorted_wo)*0.6)))
        pool = [no for no in active_cars if rank_wo.get(no,99) >= thr and no not in {anchor_no, o_no}]
        pool = [no for no in pool if p2_eff[no] >= 0.20 and venue_match(no) >= 0.0]
        return max(pool, key=lambda x: 0.6*venue_match(x)+0.4*p2_eff[x]) if pool else None
    # 男子：中位帯（極端下位は避ける）
    else:
        # 目安：9車→3〜7位、7車→3〜6位
        lo = 3; hi = 7 if n_cars==9 else 6 if n_cars==7 else max(3, int(round(len(df_sorted_wo)*0.75)))
        # ランク逆引き：順位→車
        ordered = [int(df_sorted_wo.loc[i,"車番"]) for i in range(len(df_sorted_wo))]
        cand = [ordered[i-1] for i in range(lo, min(hi,len(ordered))+1)]
        pool = [no for no in cand if no not in {anchor_no, o_no}]
        pool = [no for no in pool if p2_eff[no] >= (0.22 if race_class=="Ｓ級" else 0.20)]
        return max(pool, key=lambda x: 0.5*venue_match(x)+0.3*p2_eff[x]+0.2*himo_score(x)) if pool else None

a_no = pick_sanha()

# 印集約
result_marks, reasons = {}, {}
result_marks["◎"] = anchor_no; reasons[anchor_no] = "本命（ソフト制約＋SB）"
if o_no is not None:
    result_marks["〇"] = o_no; reasons[o_no] = "対抗（◎除外SB再計算）"
if a_no is not None:
    result_marks["▲"] = a_no; reasons[a_no] = "単穴（男女別ロジック）"

used = set(result_marks.values())
for m,no in zip([m for m in ["△","×","α","β"] if m not in result_marks],
                [int(df_sorted_wo.loc[i,"車番"]) for i in range(len(df_sorted_wo)) if int(df_sorted_wo.loc[i,"車番"]) not in used]):
    result_marks[m]=no

# 出力（SBなしランキング）
st.markdown("### ランキング＆印（◎=ソフト制約 / 〇=安定 / ▲=男女別穴）")
velobi_wo = list(zip(df_sorted_wo["車番"].astype(int).tolist(), df_sorted_wo["合計_SBなし"].round(3).tolist()))

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
        "周回補正":rec["周回補正"],"雨補正":rec["雨補正"],
        "合計_SBなし_raw":round(rec["合計_SBなし_raw"],3),
        "合計_SBなし":round(rec["合計_SBなし"],3)
    })
st.dataframe(pd.DataFrame(show), use_container_width=True)

st.caption(
    f"競輪場　{track}{race_no}R / {race_time}　{race_class} / 天候:{weather} / "
    f"開催日：{day_label}（line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}） / "
    f"会場スタイル:{style:+.2f} / 風:{wind_dir} / 有効周回={eff_laps} / 展開評価：**{confidence}**（Norm={norm:.2f}）"
)

# === 改修: 3軸評価（個人 / ライン・単騎 / 総合） =========================
st.markdown("### 🧭 3軸ランキング（個人 / ライン・単騎 / 総合）")

# ライン“生強度”をraw_lineから作る（/√人数 で正規化）
line_strength_g = {}
if line_def:
    for g, mem in line_def.items():
        base = float(raw_line.get(g, 0.0))
        n = max(1, len(mem))
        line_strength_g[g] = base / (n ** 0.5)

role_w = {'head': 1.00, 'second': 0.75, 'thirdplus': 0.55, 'single': 0.60}
line_eval = {no: 0.0 for no in active_cars}
for no in active_cars:
    g = car_to_group.get(no, None)
    role = role_in_line(no, line_def)
    base = line_strength_g.get(g, 0.0) if g in line_strength_g else 0.0
    line_eval[no] = role_w.get(role, 0.60) * base

def _znorm(d):
    vals = np.array(list(d.values()), dtype=float)
    if vals.size == 0:
        return {k: 0.0 for k in d}
    m, s = float(vals.mean()), float(vals.std())
    if s < 1e-12:
        return {k: 0.0 for k in d}
    z = (vals - m) / s
    return {k: float(z[i]) for i, k in enumerate(d.keys())}

line_eval_z = _znorm(line_eval)

# 単騎ボーナス（堅い展開ほど縮小）
solo_eval = {}
expo_scale = 0.7 if (confidence == "優位") else (1.0 if confidence == "互角" else 1.2)
for no in active_cars:
    role = role_in_line(no, line_def)
    solo_eval[no] = (0.25 if role=='single' else 0.0) / expo_scale

ind_eval = {int(no): float(sc) for no, sc in velobi_wo}
ind_eval_z = _znorm(ind_eval)

if race_class == "ガールズ":
    w_ind, w_line, w_solo = 0.70, 0.15, 0.15
elif race_class == "Ｓ級":
    w_ind, w_line, w_solo = 0.55, 0.35, 0.10
else:
    w_ind, w_line, w_solo = 0.60, 0.30, 0.10

if n_cars == 9:
    w_line *= 1.10
elif n_cars == 7:
    w_line *= 0.90

combined_raw = {}
for no in active_cars:
    combined_raw[no] = (w_ind*ind_eval_z.get(no,0.0)
                        + w_line*line_eval_z.get(no,0.0)
                        + w_solo*solo_eval.get(no,0.0))
combined_z = _znorm(combined_raw)

df_ind = pd.DataFrame([
    {"順位": i+1, "車": no, "個人(SBなし)z": round(ind_eval_z.get(no, 0.0), 3)}
    for i, (no, _) in enumerate(velobi_wo)
])
order_line = sorted(active_cars, key=lambda n: line_eval_z.get(n, 0.0), reverse=True)
df_line = pd.DataFrame([
    {"順位": i+1, "車": no, "ライン評価z": round(line_eval_z.get(no, 0.0), 3), "ライン": car_to_group.get(no, "-")}
    for i, no in enumerate(order_line)
])
order_comb = sorted(active_cars, key=lambda n: combined_z.get(n, 0.0), reverse=True)
df_comb = pd.DataFrame([
    {"順位": i+1, "車": no, "総合評価z": round(combined_z.get(no, 0.0), 3)}
    for i, no in enumerate(order_comb)
])

cA, cB, cC = st.columns(3)
with cA:
    st.caption("個人評価（SBなしベース）")
    st.dataframe(df_ind, use_container_width=True)
with cB:
    st.caption("ライン・単騎評価（同線配分 / 単騎ボーナス）")
    st.dataframe(df_line, use_container_width=True)
with cC:
    st.caption("総合評価（加重合成）")
    st.dataframe(df_comb, use_container_width=True)

# ==============================
# 買い目（想定的中率 → 必要オッズ=1/p）
# ==============================
st.markdown("### 🎯 買い目（想定的中率 → 必要オッズ=1/p）")

one = result_marks.get("◎", None)
two = result_marks.get("〇", None)
three = result_marks.get("▲", None)

# === 改修: 同ライン判定 & Pフロア緩和（男子のみ / 雨・7車で弱め） ===
def is_same_line(a,b):
    ga, gb = car_to_group.get(a, None), car_to_group.get(b, None)
    return (ga is not None) and (ga == gb)

def floor_adj(base_floor, a, b):
    if race_class == "ガールズ":
        return base_floor  # 個人戦は現状維持
    adj = 1.0
    if is_same_line(a,b):
        adj *= (0.85 if weather=="晴れ" else 0.90)   # 雨は緩和弱め
        if confidence == "優位":
            adj *= 0.90
        if n_cars == 7:
            adj *= 1.05  # 7車は緩和抑制
    return max(0.01, base_floor * adj)

if one is None:
    st.warning("◎未決定のため買い目はスキップ")
    trioC_df = wide_df = qn_df = ex_df = santan_df = None
else:
    # === 改修: 確率母ベクトルに“ラインSB”を反映（同ラインの確率が上がる） ===
    sb_per_car = {}
    for no in active_cars:
        g = car_to_group.get(no, None)
        role = role_in_line(no, line_def)
        sb_per_car[no] = (bonus_init.get(g, 0.0) if g in bonus_init else 0.0) * pos_coeff(role, 1.0)
    strength_map = {no: v_wo.get(no, 0.0) + sb_per_car.get(no, 0.0) for no in active_cars}

    xs = np.array([strength_map.get(i,0.0) for i in range(1, n_cars+1)], dtype=float)
    if xs.std() < 1e-12:
        base = np.ones_like(xs)/len(xs)
    else:
        z = (xs - xs.mean())/(xs.std()+1e-12)
        base = np.exp(z); base = base/base.sum()

    # --- 券種別キャリブレーション用：印→車番 ---
    mark_by_car = {car: None for car in range(1, n_cars+1)}
    for mk, car in result_marks.items():
        if car is not None and 1 <= car <= n_cars:
            mark_by_car[car] = mk

    # 展開評価でスケール
    expo = 0.7 if confidence == "優位" else (1.0 if confidence == "互角" else 1.3)

    def calibrate_probs(base_vec: np.ndarray, stat_key: str) -> np.ndarray:
        m = np.ones(n_cars, dtype=float)
        for idx, car in enumerate(range(1, n_cars+1)):
            mk = mark_by_car.get(car)
            if mk not in RANK_STATS:
                mk = RANK_FALLBACK_MARK
            tgt = float(RANK_STATS[mk][stat_key])
            ratio = tgt / max(float(base_vec[idx]), 1e-9)
            m[idx] = float(np.clip(ratio**(0.5*expo), 0.25, 2.5))
        probs = base_vec * m
        probs = probs / probs.sum()
        return probs

    # 券種ごとの分布：ワイド/三複=Top3、二複=Top2、二単/三単=1着
    probs_p3 = calibrate_probs(base, "pTop3")  # ワイド・三連複
    probs_p2 = calibrate_probs(base, "pTop2")  # 二車複
    probs_p1 = calibrate_probs(base, "p1")     # 二車単・三連単

    rng = np.random.default_rng(20250830)
    trials = st.slider("シミュレーション試行回数", 1000, 20000, 8000, 1000)

    def sample_order_from_probs(pvec: np.ndarray) -> list[int]:
        # Plackett–Luce風のGumbelノイズ順位決定（券種間一貫）
        g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
        score = np.log(pvec+1e-12) + g
        return (np.argsort(-score)+1).tolist()

    mates = [x for x in [two, three] if x is not None]

    # === 改修: 三連複Cに“◎の同ライン相棒”を1頭だけ追加（男子のみ） ===
    if race_class != "ガールズ":
        g_one = car_to_group.get(one, None)
        if g_one in line_def and len(line_def[g_one]) >= 2:
            cand_mates = [c for c in line_def[g_one] if c != one]
            if cand_mates:
                best_line_mate = max(cand_mates, key=lambda x: himo_score(x))
                if best_line_mate not in mates:
                    mates.append(best_line_mate)

    all_others = [i for i in range(1, n_cars+1) if i != one]

    # === カウント器 ===
    trioC_counts = {}
    wide_counts = {k:0 for k in all_others}
    qn_counts   = {k:0 for k in all_others}
    ex_counts   = {k:0 for k in all_others}
    st3_counts  = {}  # 三連単（◎→[相手]→全）： key=(second,third) -> 回数

    # 三連複Cの組み合わせ（◎-[相手]-全）
    trioC_list = []
    if len(mates) > 0:
        for a in all_others:
            for b in all_others:
                if a >= b: continue
                if (a in mates) or (b in mates):
                    t = tuple(sorted([a, b, one]))
                    trioC_list.append(t)
        trioC_list = sorted(set(trioC_list))

    # === シミュレーション ===
    for _ in range(trials):
        # ワイド/三連複：Top3率ベース
        order_p3 = sample_order_from_probs(probs_p3)
        top3_p3 = set(order_p3[:3]); top2_p3 = set(order_p3[:2])

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

        # 二車複：連対率ベース
        order_p2 = sample_order_from_probs(probs_p2)
        top2_p2 = set(order_p2[:2])
        if one in top2_p2:
            for k in qn_counts.keys():
                if k in top2_p2:
                    qn_counts[k] += 1

        # 二車単/三連単：1着率ベース
        order_p1 = sample_order_from_probs(probs_p1)
        if order_p1[0] == one:
            k2 = order_p1[1]
            if k2 in ex_counts:
                ex_counts[k2] += 1
            # 三連単（◎→[相手]→全）：2着は {〇,▲, 同ライン追加} 限定、3着は全（ただし重複不可）
            if len(mates) > 0:
                k3 = order_p1[2]
                if (k2 in mates) and (k3 not in (one, k2)):
                    st3_counts[(k2, k3)] = st3_counts.get((k2, k3), 0) + 1

    # ====== Pフロア（最低想定p）とEV帯 ======
    P_FLOOR = globals().get("P_FLOOR", {
        "wide": 0.060, "sanpuku": 0.040, "nifuku": 0.050, "nitan": 0.040, "santan": 0.030
    })
    # 展開で複系だけ微調整（±10%）
    scale = 1.00
    if confidence == "優位":   scale = 0.90
    elif confidence == "混線": scale = 1.10
    for k in ("wide","sanpuku","nifuku"):
        P_FLOOR[k] *= scale

    E_MIN = globals().get("E_MIN", 0.00)
    E_MAX = globals().get("E_MAX", 0.50)

    def need_from_count(cnt: int) -> float | None:
        if cnt <= 0: return None
        p = cnt / trials
        return round(1.0 / p, 2)

    # === 三連複C ===
    if len(trioC_list) > 0:
        rows = []
        for t in trioC_list:
            cnt = trioC_counts.get(t, 0)
            p = cnt / trials
            # 三連複は基本：pフロアのみ（上限なしの帯提示はnote側で）
            if p < P_FLOOR.get("sanpuku", 0.06):
                continue
            rows.append({
                "買い目": f"{t[0]}-{t[1]}-{t[2]}",
                "p(想定的中率)": round(p, 4),
                "必要オッズ(=1/p)": "-" if cnt==0 else need_from_count(cnt)
            })
        trioC_df = pd.DataFrame(rows)
        st.markdown("#### 三連複C（◎-[相手]-全）※車番順")
        def _key_nums_tri(s): return list(map(int, re.findall(r"\d+", s)))
        if len(trioC_df)>0:
            trioC_df = trioC_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_tri)).reset_index(drop=True)
        st.dataframe(trioC_df, use_container_width=True)
    else:
        trioC_df = None
        st.info("三連複C：相手（〇/▲）が未設定のため表示なし")

    # === 三連複バスケット合成オッズと「相手集合S」 ===
    S_set = set()
    O_combo = None
    if trioC_df is not None and len(trioC_df) > 0:
        need_list = []
        for _, r in trioC_df.iterrows():
            name = str(r["買い目"])
            nums = list(map(int, re.findall(r"\d+", name)))
            others = [x for x in nums if x != one]
            S_set.update(others)
            need_val = r.get("必要オッズ(=1/p)")
            if isinstance(need_val, (int, float)):
                if float(need_val) > 0:
                    need_list.append(float(need_val))
        if need_list:
            denom = sum(1.0/x for x in need_list if x > 0)
            if denom > 0:
                O_combo = float(f"{(1.0/denom):.2f}")

    if O_combo is not None and len(S_set) > 0:
        st.caption(f"三連複バスケット合成オッズ（下限基準）：**{O_combo:.2f}倍** / 相手集合S：{sorted(S_set)}")
    elif trioC_df is not None and len(trioC_df) > 0:
        st.caption("三連複バスケット合成オッズ：算出不可（必要オッズが'-'のみ）")

    # === ワイド（◎-全）— 三連複と被る側は合成オッズで足切り／漏れ側は必要オッズ ===
    rows = []
    for k in sorted(wide_counts.keys()):
        cnt = wide_counts[k]
        p = cnt / trials
        floor = floor_adj(P_FLOOR.get("wide", 0.25), one, k)  # ← 同ライン緩和
        if p < floor:
            continue
        need = None if cnt == 0 else (1.0 / p)
        if need is None or need <= 0:
            continue
        eligible = True
        rule_note = "必要オッズ以上"
        if (O_combo is not None) and (k in S_set):
            if need >= O_combo:
                eligible = True
                rule_note = f"三複被り→合成{O_combo:.2f}倍以上"
            else:
                eligible = False
        if not eligible:
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
        def _key_nums_w(s): return list(map(int, re.findall(r"\d+", s)))
        wide_df = wide_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_w)).reset_index(drop=True)
        st.dataframe(wide_df, use_container_width=True)
        if O_combo is not None:
            st.caption("※ワイドは上限撤廃：三連複で使用した相手は **合成オッズ以上**、それ以外は **必要オッズ以上**。")
        else:
            st.caption("※ワイドは上限撤廃：**必要オッズ以上**で採用。")
    else:
        st.info("ワイド：対象外（Pフロア未満、または合成オッズ基準で除外）")

    # === 二車複 ===
    rows = []
    for k in sorted(qn_counts.keys()):
        cnt = qn_counts[k]; p = cnt / trials
        floor = floor_adj(P_FLOOR.get("nifuku", 0.12), one, k)  # ← 同ライン緩和
        if p < floor:
            continue
        need = None if cnt==0 else (1.0/p)
        if need is None: continue
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

    # === 二車単 ===
    rows = []
    for k in sorted(ex_counts.keys()):
        cnt = ex_counts[k]; p = cnt / trials
        floor = floor_adj(P_FLOOR.get("nitan", 0.07), one, k)  # ← 同ライン緩和
        if p < floor:
            continue
        need = None if cnt==0 else (1.0/p)
        if need is None: continue
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

    # === 三連単（◎→[相手]→全） ===
    rows = []
    p_floor_santan = P_FLOOR.get("santan", 0.03)
    for (sec, thr), cnt in st3_counts.items():
        p = cnt / trials
        if p < p_floor_santan or p <= 0:
            continue
        need = 1.0 / p
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
# note用：ヘッダー〜展開評価 ＋ 「買えるオッズ帯」
# ==============================
st.markdown("### 📋 note用（ヘッダー〜展開評価＋“買えるオッズ帯”）")

if '_sort_key_by_numbers' not in globals():
    def _sort_key_by_numbers(s: str) -> tuple:
        return tuple(map(int, re.findall(r"\d+", s)))

def _format_line_zone_note(name: str, bet_type: str, p: float) -> str | None:
    floor = P_FLOOR.get(bet_type, 0.03 if bet_type=="santan" else 0.0)
    if p < floor: return None
    need = 1.0 / max(p, 1e-12)
    if bet_type == "wide":
        return f"{name}：{need:.1f}倍以上で買い"  # 上限撤廃
    low, high = need*(1.0+E_MIN), need*(1.0+E_MAX)
    return f"{name}：{low:.1f}〜{high:.1f}倍なら買い"

def _zone_lines_from_df(df: pd.DataFrame | None, bet_type_key: str) -> list[str]:
    if df is None or len(df) == 0 or "買い目" not in df.columns:
        return []
    rows = []
    for _, r in df.iterrows():
        name = str(r["買い目"])
        if "買える帯" in r and r["買える帯"]:
            rows.append((name, f"{name}：{r['買える帯']}"))
        else:
            p = float(r.get("p(想定的中率)", 0.0) or 0.0)
            line_txt = _format_line_zone_note(name, bet_type_key, p)
            if line_txt:
                rows.append((name, line_txt))
    rows_sorted = sorted(rows, key=lambda x: _sort_key_by_numbers(x[0]))
    return [ln for _, ln in rows_sorted]

def _section_text(title: str, lines: list[str]) -> str:
    if not lines: return f"{title}\n対象外"
    return f"{title}\n" + "\n".join(lines)

line_text = "　".join([x for x in line_inputs if str(x).strip()])
score_order_text = " ".join(str(no) for no,_ in velobi_wo)
marks_line = " ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)

txt_trioC = _section_text("三連複C（◎-[相手]-全）",
                          _zone_lines_from_df(globals().get("trioC_df", None), "sanpuku"))
txt_st    = _section_text("三連単（◎→[相手]→全）",
                          _zone_lines_from_df(globals().get("santan_df", None), "santan"))
txt_wide  = _section_text("ワイド（◎-全）",
                          _zone_lines_from_df(globals().get("wide_df", None), "wide"))
txt_qn    = _section_text("二車複（◎-全）",
                          _zone_lines_from_df(globals().get("qn_df", None), "nifuku"))
txt_ex    = _section_text("二車単（◎→全）",
                          _zone_lines_from_df(globals().get("ex_df", None), "nitan"))

wide_rule_note = "（ワイドは上限撤廃：三連複で使用した相手は合成オッズ以上／三連複から漏れた相手は必要オッズ以上で買い）"

note_text = (
    f"{track}競輪 {race_no}R\n"
    f"展開評価：{confidence}\n"
    f"{race_time}　{race_class}　天候:{weather}\n"
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


