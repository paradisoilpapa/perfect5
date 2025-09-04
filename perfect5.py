# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re, math, random, json
import itertools
from typing import Optional

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き）", layout="wide")

# ==============================
# 定数（あなたの実測・既存値は変更しません）
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

# --- 直近集計：印別の実測率（%→小数） あなたの毎日更新をそのまま使う ---
RANK_STATS = {
    "◎": {"p1": 0.200, "pTop2": 0.480, "pTop3": 0.610},
    "〇": {"p1": 0.200, "pTop2": 0.390, "pTop3": 0.470},
    "▲": {"p1": 0.100, "pTop2": 0.260, "pTop3": 0.430},
    "△": {"p1": 0.130, "pTop2": 0.240, "pTop3": 0.400},
    "×": {"p1": 0.190, "pTop2": 0.240, "pTop3": 0.410},
    "α": {"p1": 0.133, "pTop2": 0.184, "pTop3": 0.347},  # N=98
    "β": {"p1": 0.108, "pTop2": 0.269, "pTop3": 0.409},  # N=93
}
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
# ◎ソフト制約＋底抜け防止（NEW）
# ==============================
K_SOFT = 4      # 得点ランク4位以内は素通り
K_HARD = 7      # 得点ランク>7 は◎にしない（絶対NG）
THETA_SB = {"優位": 0.025, "互角": 0.020, "混線": 0.015}  # 昇格チケット：SB差
THETA_V  = {"優位": 0.020, "互角": 0.015, "混線": 0.010}  # 昇格チケット：会場適合
P1_MIN_BY_CLASS = {"Ｓ級": 0.15, "Ａ級": 0.12, "Ａ級チャレンジ": 0.10, "ガールズ": 0.14}
LAMBDA_BY_CONF = {"優位": 0.020, "互角": 0.015, "混線": 0.010}

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

# =====（NEW）得点2–4位恒常上振れ補正はやめる：常時0 =====
def tenscore_correction(tenscores):
    return [0.0]*len(tenscores)

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

# ===== ラインSB（NEW：/√N 正規化） =====
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
        r = (0.6*b + 0.4*s) * (0.6 + 0.4*ratioS)
        n = max(1, len(line_def[g]))       # NEW: 人数で安定化
        raw[g] = r / (n**0.5)
    zz = zscore_list(list(raw.values())) if raw else []
    bonus={g: clamp(0.02*float(zz[i]), -cap, cap) for i,g in enumerate(raw.keys())}
    return bonus, raw

def input_float_text(label: str, key: str, placeholder: str = "") -> Optional[float]:
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
# サイドバー
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

# ▼ レース番号
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

# SBなし合計
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
    total_raw = (prof_base[no] + wind + cf["spread"]*tens_corr.get(no,0.0) + bank_b + length_b + laps_adj)
    rows.append([no, role, round(prof_base[no],3), wind, round(cf["spread"]*tens_corr.get(no,0.0),3),
                 round(bank_b,3), round(length_b,3), round(laps_adj,3), total_raw])

df = pd.DataFrame(rows, columns=["車番","役割","脚質基準(会場)","風補正","得点補正","バンク補正","周長補正","周回補正","合計_SBなし_raw"])
mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0*(df["合計_SBなし_raw"] - mu)
df_sorted_wo = df.sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

# 候補C（得点×2着率ブレンド 上位3）
blend = {no: (ratings_val[no] + min(50.0, p2_eff[no]*100.0))/2.0 for no in active_cars}
C = [kv[0] for kv in sorted(blend.items(), key=lambda x:x[1], reverse=True)[:min(3,len(blend))]]

# ラインSB
bonus_init,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)
v_wo = dict(zip(df["車番"], df["合計_SBなし"]))

ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank = {no: i+1 for i, no in enumerate(ratings_sorted)}

# ===== 展開評価（あとで◎ペナルティ強度に使う）=====
cand_scores = [v_wo.get(no, -1e9) for no in C] if len(C)>=2 else [0,0]
conf = (sorted(cand_scores, reverse=True)[0] - sorted(cand_scores, reverse=True)[1]) if len(cand_scores)>=2 else 0.0
spread = float(np.std(list(v_wo.values()))) if len(v_wo)>=2 else 0.0
norm = conf / (spread if spread>1e-6 else 1.0)
confidence = "優位" if norm>=1.0 else ("互角" if norm>=0.5 else "混線")

# =====（NEW）印連続係数：RANK_STATS→MARK_MULを毎回導出 =====
def derive_mark_mul(rank_stats: dict, alpha: float = 0.5, lo: float = 0.90, hi: float = 1.20, eps: float = 1e-9) -> dict:
    keys = ["p1", "pTop2", "pTop3"]
    weights = {m: float(rs.get("N", 1.0)) for m, rs in rank_stats.items()}
    wsum = sum(weights.values()) if weights else 1.0
    base = {}
    for k in keys:
        num = 0.0
        for m, rs in rank_stats.items():
            num += weights[m] * float(rs.get(k, 0.0))
        base[k] = num / max(wsum, eps)
    mul = {}
    for m, rs in rank_stats.items():
        mul[m] = {}
        for k in keys:
            obs = float(rs.get(k, 0.0))
            ref = base[k]
            ratio = obs / max(ref, eps)
            mul[m][k] = float(np.clip(ratio ** alpha, lo, hi))
    return mul
MARK_MUL = derive_mark_mul(RANK_STATS, alpha=0.5, lo=0.90, hi=1.20)
MARK_FALLBACK = RANK_FALLBACK_MARK

# =====（NEW）◎決定：ソフト制約＋底抜け防止＋昇格チケット =====
def venue_match(no):
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark=0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    return style * (1.00*esc + 0.40*mak - 0.60*sashi - 0.25*mark)

bonus_re,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)

def eligible_promote(no, confidence):
    rnk = ratings_rank.get(no, 999)
    if rnk <= K_SOFT:
        return True
    if rnk > K_HARD:
        return False
    g = car_to_group.get(no, None)
    role = role_in_line(no, line_def)
    sb_with = bonus_init.get(g, 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    sb_without = bonus_re.get(g, 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    sb_adv = float(sb_with - sb_without)
    vmatch = float(venue_match(no))
    p1_min = P1_MIN_BY_CLASS.get(race_class, 0.12)
    ok_form = (p1_eff.get(no, 0.0) >= p1_min)
    return (sb_adv >= THETA_SB[confidence]) and (vmatch >= THETA_V[confidence]) and ok_form

def anchor_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    sb = bonus_init.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
    zt_map = {n:float(zt[i]) for i,n in enumerate(active_cars)} if active_cars else {}
    base = v_wo.get(no, -1e9) + sb + 0.01*zt_map.get(no, 0.0)

    rnk = ratings_rank.get(no, 999)
    lam = LAMBDA_BY_CONF[confidence]
    if rnk > K_HARD:
        return base - 1.0
    if rnk > K_SOFT and not eligible_promote(no, confidence):
        return base - 10*lam
    penalty = -lam * max(0, rnk - K_SOFT)
    return base + penalty

anchor_no = max(C, key=lambda x: anchor_score(x)) if C else (int(df_sorted_wo.loc[0,"車番"]) if not df_sorted_wo.empty else 1)

# 対抗・単穴
bonus_re_ex,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=anchor_no, cap=cap_SB_eff, enable=line_sb_enable)
def himo_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    sb = bonus_re_ex.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    return v_wo.get(no, -1e9) + sb
restC = [no for no in C if no!=anchor_no]
o_no = max(restC, key=lambda x: himo_score(x)) if restC else None

rank_wo = {int(df_sorted_wo.loc[i,"車番"]): i+1 for i in range(len(df_sorted_wo))}
lower_rank_threshold = max(5, int(np.ceil(len(df_sorted_wo)*0.6)))
lower_pool = [no for no in active_cars if rank_wo.get(no,99) >= lower_rank_threshold]
p2_C_mean = np.mean([p2_eff[no] for no in C]) if C else 0.0
min_p2 = 0.22 if race_class=="Ｓ級" else 0.20
pool_filtered = [no for no in lower_pool
                 if no not in {anchor_no, o_no}
                 and ( p2_eff[no] >= min_p2 )
                 and ( p2_eff[no] <= p2_C_mean + 1e-9 )]
a_no = max(pool_filtered, key=lambda x: venue_match(x)) if pool_filtered else None
if a_no is None:
    fb = [no for no in lower_pool if no not in {anchor_no, o_no}]
    if fb: a_no = max(fb, key=lambda x: venue_match(x))

# 印集約
st.markdown("### ランキング＆印（◎=ソフト制約／昇格チケットあり）")
velobi_wo = list(zip(df_sorted_wo["車番"].astype(int).tolist(), df_sorted_wo["合計_SBなし"].round(3).tolist()))

result_marks, reasons = {}, {}
result_marks["◎"] = anchor_no; reasons[anchor_no] = "本命(ソフト制約+昇格チケット)"
if o_no is not None:
    result_marks["〇"] = o_no; reasons[o_no] = "対抗(◎除外SB再計算)"
if a_no is not None:
    result_marks["▲"] = a_no; reasons[a_no] = "単穴(SBなし下位×会場適合×2着%)"

used = set(result_marks.values())
for m,no in zip([m for m in ["△","×","α","β"] if m not in result_marks],
                [int(df_sorted_wo.loc[i,"車番"]) for i in range(len(df_sorted_wo)) if int(df_sorted_wo.loc[i,"車番"]) not in used]):
    result_marks[m]=no

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
        "周回補正":rec["周回補正"],"合計_SBなし_raw":round(rec["合計_SBなし_raw"],3),
        "合計_SBなし":round(rec["合計_SBなし"],3)
    })
st.dataframe(pd.DataFrame(show), use_container_width=True)

st.caption(
    f"競輪場　{track}{race_no}R / {race_time}　{race_class} / "
    f"開催日：{day_label}（line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}） / "
    f"会場スタイル:{style:+.2f} / 風:{wind_dir} / 有効周回={eff_laps} / 展開評価：**{confidence}**（Norm={norm:.2f}）"
)

# ==============================
# 買い目（NEW：単一順位生成→全券種）
# ==============================
st.markdown("### 🎯 買い目（単一の順位サンプルから全券種を一貫評価）")

one = result_marks.get("◎", None)
two = result_marks.get("〇", None)
three = result_marks.get("▲", None)

if one is None:
    st.warning("◎未決定のため買い目はスキップ")
    trioC_df = wide_df = qn_df = ex_df = santan_df = None
else:
    # 基底確率（softmax）
    strength_map = dict(velobi_wo)
    xs = np.array([strength_map.get(i,0.0) for i in range(1, n_cars+1)], dtype=float)
    if xs.std() < 1e-12: base = np.ones_like(xs)/len(xs)
    else:
        z = (xs - xs.mean())/(xs.std()+1e-12)
        base = np.exp(z); base = base/base.sum()

    # 印係数で“軽く傾ける”（実測→毎回導出した MARK_MUL を使用）
    mark_by_car = {car: None for car in range(1, n_cars+1)}
    for mk, car in result_marks.items():
        if car is not None and 1 <= car <= n_cars:
            mark_by_car[car] = mk

    def apply_mark_mul(base_vec: np.ndarray, key: str) -> np.ndarray:
        m = np.ones(n_cars, dtype=float)
        expo = 0.7 if confidence == "優位" else (1.0 if confidence == "互角" else 1.3)
        for idx, car in enumerate(range(1, n_cars+1)):
            mk = mark_by_car.get(car) or MARK_FALLBACK
            mul = MARK_MUL.get(mk, MARK_MUL[MARK_FALLBACK]).get(
                "p1" if key=="p1" else ("pTop2" if key=="p2" else "pTop3"), 1.0
            )
            m[idx] = float(np.clip(mul**(0.5*expo), 0.85, 1.20))
        vec = base_vec * m
        return vec / vec.sum()

    # 単一の順位生成用確率（1着ベース）
    probs_rank = apply_mark_mul(base, "p1")

    # 条件依存シード（同条件=再現）
    seed = abs(hash((track, race_no, n_cars, confidence))) % (2**32)
    rng = np.random.default_rng(seed)
    trials = st.slider("シミュレーション試行回数", 1000, 20000, 8000, 1000)

    def sample_order_from_probs(pvec: np.ndarray) -> list[int]:
        g = -np.log(-np.log(np.clip(rng.random(len(pvec)), 1e-12, 1-1e-12)))
        score = np.log(pvec+1e-12) + g
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
        order = sample_order_from_probs(probs_rank)
        top3 = set(order[:3]); top2 = set(order[:2])

        if one in top3:
            for k in wide_counts.keys():
                if k in top3:
                    wide_counts[k] += 1
            if len(trioC_list) > 0:
                others = list(top3 - {one})
                if len(others) == 2:
                    a, b = sorted(others)
                    if (a in mates) or (b in mates):
                        t = tuple(sorted([a, b, one]))
                        if t in trioC_list:
                            trioC_counts[t] = trioC_counts.get(t, 0) + 1

        if one in top2:
            for k in qn_counts.keys():
                if k in top2:
                    qn_counts[k] += 1

        if order[0] == one:
            k2 = order[1]
            if k2 in ex_counts:
                ex_counts[k2] += 1
            if len(mates) > 0:
                k3 = order[2]
                if (k2 in mates) and (k3 not in (one, k2)):
                    st3_counts[(k2, k3)] = st3_counts.get((k2, k3), 0) + 1

    def need_from_count(cnt: int) -> Optional[float]:
        if cnt <= 0: return None
        p = cnt / trials
        return round(1.0 / p, 2)

    # 三連複C
    if len(trioC_list) > 0:
        rows = []
        for t in trioC_list:
            cnt = trioC_counts.get(t, 0)
            p = cnt / trials
            rows.append({
                "買い目": f"{t[0]}-{t[1]}-{t[2]}",
                "p(想定的中率)": round(p, 4),
                "必要オッズ(=1/p)": "-" if cnt==0 else need_from_count(cnt)
            })
        trioC_df = pd.DataFrame(rows)
        st.markdown("#### 三連複C（◎-[相手]-全）※車番順")
        def _key_nums_tri(s): return list(map(int, re.findall(r"\d+", s)))
        trioC_df = trioC_df.sort_values(by="買い目", key=lambda s: s.map(_key_nums_tri)).reset_index(drop=True)
        st.dataframe(trioC_df, use_container_width=True)
    else:
        trioC_df = None
        st.info("三連複C：相手（〇/▲）が未設定のため表示なし")

    # 合成オッズと相手集合S
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
            if isinstance(need_val, (int, float)) and float(need_val) > 0:
                need_list.append(float(need_val))
        if need_list:
            denom = sum(1.0/x for x in need_list if x > 0)
            if denom > 0:
                O_combo = float(f"{(1.0 / denom):.2f}")

    if O_combo is not None and len(S_set) > 0:
        st.caption(f"三連複バスケット合成オッズ（下限基準）：**{O_combo:.2f}倍** / 相手集合S：{sorted(list(S_set))}")
    elif trioC_df is not None and len(trioC_df) > 0:
        st.caption("三連複バスケット合成オッズ：算出不可（必要オッズが'-'のみ）")

    # ワイド
    rows = []
    for k in sorted(wide_counts.keys()):
        cnt = wide_counts[k]; p = cnt / trials
        if p < P_FLOOR.get("wide", 0.06):
            continue
        need = None if cnt == 0 else (1.0 / p)
        if need is None or need <= 0: continue
        eligible = True
        rule_note = "必要オッズ以上"
        if (O_combo is not None) and (k in S_set):
            if need >= O_combo:
                rule_note = f"三複被り→合成{O_combo:.2f}倍以上"
            else:
                eligible = False
        if not eligible: continue
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
            st.caption("※三連複で使用した相手（S側）は **合成オッズ以上**のワイドのみ採用。S外は **必要オッズ以上**で採用。ワイドは上限撤廃＝『◯倍以上で買い』。")
        else:
            st.caption("※三連複が無い／合成不可の場合、ワイドは **必要オッズ以上**で採用（上限撤廃）。")
    else:
        st.info("ワイド：対象外（Pフロア未満、または合成オッズ基準で除外）")

    # 二車複
    rows = []
    for k in sorted(qn_counts.keys()):
        cnt = qn_counts[k]; p = cnt / trials
        if p < P_FLOOR.get("nifuku", 0.05):
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

    # 二車単
    rows = []
    for k in sorted(ex_counts.keys()):
        cnt = ex_counts[k]; p = cnt / trials
        if p < P_FLOOR.get("nitan", 0.04):
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

    # 三連単
    rows = []
    p_floor_santan = 0.03
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
# note用
# ==============================
st.markdown("### 📋 note用（ヘッダー〜展開評価＋“買えるオッズ帯”）")
def _format_line_zone_note(name: str, bet_type: str, p: float) -> Optional[str]:
    floor = P_FLOOR.get(bet_type, 0.03 if bet_type=="santan" else 0.0)
    if p < floor: return None
    need = 1.0 / max(p, 1e-12)
    if bet_type == "wide":
        return f"{name}：{need:.1f}倍以上で買い"
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

line_text = "　".join([x for x in line_inputs if str(x).strip()])
score_order_text = " ".join(str(no) for no,_ in velobi_wo)
marks_line = " ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)

txt_trioC = "対象外"; txt_st="対象外"; txt_wide="対象外"; txt_qn="対象外"; txt_ex="対象外"
if one is not None:
    def _sec(title, df, key):
        arr = _zone_lines_from_df(df, key)
        return f"{title}\n" + ("\n".join(arr) if arr else "対象外")
    txt_trioC = _sec("三連複C（◎-[相手]-全）", locals().get("trioC_df"), "sanpuku")
    txt_st    = _sec("三連単（◎→[相手]→全）", locals().get("santan_df"), "santan")
    txt_wide  = _sec("ワイド（◎-全）",         locals().get("wide_df"),   "wide")
    txt_qn    = _sec("二車複（◎-全）",         locals().get("qn_df"),     "nifuku")
    txt_ex    = _sec("二車単（◎→全）",         locals().get("ex_df"),     "nitan")

note_text = (
    f"競輪場　{track}{race_no}R\n"
    f"展開評価：{confidence}\n"
    f"{race_time}　{race_class}\n"
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}\n"
    f"\n{txt_trioC}\n\n{txt_st}\n\n{txt_wide}\n\n{txt_qn}\n\n{txt_ex}\n"
    f"\n（※“対象外”＝Pフロア未満。どんなオッズでも買わない）\n"
    f"（ワイドは上限撤廃：三連複で使用した相手は合成オッズ以上／三連複から漏れた相手は必要オッズ以上で買い）"
)
st.text_area("ここを選択してコピー", note_text, height=380)

