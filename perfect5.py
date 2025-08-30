# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re, math, random, json

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
    return bonus, raw

def input_float_text(label: str, key: str, placeholder: str = "") -> float | None:
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "": return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)

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

race_no = st.sidebar.selectbox("レース番号", list(range(1,13)), 0)
race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_class = st.sidebar.selectbox("級別", ["Ｓ級","Ａ級","Ａ級チャレンジ","ガールズ"], 0)

# 会場style（先行⇄差し）
angles = [KEIRIN_DATA[k]["bank_angle"] for k in KEIRIN_DATA]
straights = [KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA]
lengths = [KEIRIN_DATA[k]["bank_length"] for k in KEIRIN_DATA]
angle_z = zscore_val(bank_angle, angles)
straight_z = zscore_val(straight_length, straights)
length_z = zscore_val(bank_length, lengths)
style_raw = clamp(0.50*angle_z - 0.35*straight_z - 0.30*length_z, -1.0, +1.0)

override = st.sidebar.slider("会場バイアス補正（−2差し ←→ +2先行）", -2.0, 2.0, 0.0, 0.1)
style = clamp(style_raw + 0.25*override, -1.0, +1.0)

# 級別係数（スコア広がり/ライン結束）
CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

# 日程係数（ライン力）
DAY_FACTOR = {"初日":1.00, "2日目":0.60, "最終日":0.85}
day_factor = DAY_FACTOR[day_label]

# ライン係数・SBcap（会場→級別→日程）
cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

# ガールズはラインSB無効
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

# フォーム指標（0-1）
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

# SBなし合計（環境補正 + 得点微補正） → 級別spreadで広がり圧縮
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

# ラインSB（級別×日程×会場cap、ガールズは無効）
bonus_init,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)

v_wo = dict(zip(df["車番"], df["合計_SBなし"]))

# ◎：C内で「SB加点 + SBなし合計」が最大
def anchor_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    # ★修正①：line_factorの二重掛けを解消（pos_coeffの第二引数を 1.0 固定）
    sb = bonus_init.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
    zt_map = {n:float(zt[i]) for i,n in enumerate(active_cars)} if active_cars else {}
    return v_wo.get(no, -1e9) + sb + 0.01*zt_map.get(no, 0.0)

anchor_no = max(C, key=lambda x: anchor_score(x)) if C else int(df_sorted_wo.loc[0,"車番"])

# 自信度（候補C内の差 / 広がり）
cand_scores = [anchor_score(no) for no in C] if len(C)>=2 else [0,0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf = cand_scores_sorted[0]-cand_scores_sorted[1] if len(cand_scores_sorted)>=2 else 0.0
spread = float(np.std(list(v_wo.values()))) if len(v_wo)>=2 else 0.0
norm = conf / (spread if spread>1e-6 else 1.0)
confidence = "強" if norm>=1.0 else ("中" if norm>=0.5 else "弱")

# 〇：C残り → ◎除外でSB再計算 → 最大
bonus_re,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=anchor_no, cap=cap_SB_eff, enable=line_sb_enable)
def himo_score(no):
    g = car_to_group.get(no, None); role = role_in_line(no, line_def)
    # ★修正①：二重掛け解消
    sb = bonus_re.get(g,0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    return v_wo.get(no, -1e9) + sb

restC = [no for no in C if no!=anchor_no]
o_no = max(restC, key=lambda x: himo_score(x)) if restC else None

# ▲：SBなし下位域から、会場適合×“平滑2着%” 条件で1頭
def venue_match(no):
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark=0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    return style * (1.00*esc + 0.40*mak - 0.60*sashi - 0.25*mark)

rank_wo = {int(df_sorted_wo.loc[i,"車番"]): i+1 for i in range(len(df_sorted_wo))}
# ★修正③：下位40%しきい
lower_rank_threshold = max(5, int(np.ceil(len(df_sorted_wo)*0.6)))
lower_pool = [no for no in active_cars if rank_wo.get(no,99) >= lower_rank_threshold]

p2_C_mean = np.mean([p2_eff[no] for no in C]) if C else 0.0
min_p2 = 0.22 if race_class=="Ｓ級" else 0.20

# ★修正②：生比率ではなく平滑 p2_eff を使用
pool_filtered = [no for no in lower_pool
                 if no not in {anchor_no, o_no}
                 and ( p2_eff[no] >= min_p2 )
                 and ( p2_eff[no] <= p2_C_mean + 1e-9 )]

a_no = max(pool_filtered, key=lambda x: venue_match(x)) if pool_filtered else None
if a_no is None:
    fb = [no for no in lower_pool if no not in {anchor_no, o_no}]
    if fb: a_no = max(fb, key=lambda x: venue_match(x))

# 印集約
result_marks, reasons = {}, {}
result_marks["◎"] = anchor_no; reasons[anchor_no] = "本命(C上位3→ラインSB重視)※ガールズは個人要素のみ"
if o_no is not None:
    result_marks["〇"] = o_no; reasons[o_no] = "対抗(C残り→◎除外SB再計算)"
if a_no is not None:
    result_marks["▲"] = a_no; reasons[a_no] = "単穴(SBなし下位×会場適合×2着%)"

used = set(result_marks.values())
for m,no in zip([m for m in ["△","×","α","β"] if m not in result_marks],
                [int(df_sorted_wo.loc[i,"車番"]) for i in range(len(df_sorted_wo)) if int(df_sorted_wo.loc[i,"車番"]) not in used]):
    result_marks[m]=no

# 出力（SBなしランキング）
st.markdown("### ランキング＆印（◎=ラインSB考慮 / 〇=安定 / ▲=逆襲）")
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
        "周回補正":rec["周回補正"],"合計_SBなし_raw":round(rec["合計_SBなし_raw"],3),
        "合計_SBなし":round(rec["合計_SBなし"],3)
    })
st.dataframe(pd.DataFrame(show), use_container_width=True)

st.caption(
    f"競輪場　{track}{race_no}R / {race_time}　{race_class} / "
    f"開催日：{day_label}（line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}） / "
    f"会場スタイル:{style:+.2f} / 風:{wind_dir} / 有効周回={eff_laps} / 自信度：**{confidence}**（Norm={norm:.2f}）"
)

# ==============================
# 買い目（想定的中率 / 必要オッズ / EV）
# ==============================
st.markdown("### 🎯 買い目（想定的中率 / 必要オッズ / EV）")

one = result_marks.get("◎", None)
two = result_marks.get("〇", None)
three = result_marks.get("▲", None)
if one is None:
    st.warning("◎未決定のため買い目はスキップ")
    buy_table = None
else:
    # 強さベクトル：SBなしスコア → 標準化softmax
    strength_map = dict(velobi_wo)
    xs = np.array([strength_map.get(i,0.0) for i in range(1, n_cars+1)], dtype=float)
    if xs.std()<1e-12:
        probs = np.ones_like(xs)/len(xs)
    else:
        z = (xs - xs.mean())/(xs.std()+1e-12); e = np.exp(z); probs = e/e.sum()

    rng = np.random.default_rng(20250830)
    trials = st.slider("シミュレーション試行回数", 1000, 20000, 8000, 1000)

    def sample_order():
        # Gumbel-Max
        g = -np.log(-np.log(np.clip(rng.random(len(probs)),1e-12,1-1e-12)))
        score = np.log(probs+1e-12) + g
        order = np.argsort(-score)+1
        return order.tolist()

    # 2列目候補セット
    second_set = [x for x in [two, three] if x is not None]
    rest = [i for i in range(1, n_cars+1) if i not in {one} | set(second_set)]
    # 3連複A: ◎-（〇/▲）-{残り上位から最大5頭}
    third_A = (second_set + rest)[:min(5, max(1, n_cars-2))]
    # 3連複B: ◎-{二列候補4頭}-同枠
    box_B = (second_set + rest)[:4]
    # 3連複C: ◎-（〇/▲）-全
    set_23 = second_set[:]

    hit_A=hit_B=hit_C=hit_EX=hit_QN=hit_WD=0
    for _ in range(trials):
        order = sample_order()
        top2 = set(order[:2]); top3 = set(order[:3])

        if (one in top3) and (len(top3 & set([two, three]))>=1) and (len(top3 & set(third_A))>=2):
            hit_A += 1
        if (one in top3) and (len(top3 & set(box_B))>=2):
            hit_B += 1
        if (one in top3) and (len(top3 & set(set_23))>=1):
            hit_C += 1

        if len(set_23)>0 and order[0]==one and order[1] in set_23:
            hit_EX += 1
        if len(set_23)>0 and set(order[:2]) & set(set_23) and one in set(order[:2]):
            hit_QN += 1
        if (one in order[:3]) and (len(set(order[:3]) & set(set_23))>=1):
            hit_WD += 1

    def pct(x): return f"{(x/trials)*100:.1f}%"
    def need(x): return "-" if x==0 else f"{trials/x:.2f}"

    buy_rows = [
        ["三連複A", f"{one}-{set_23 if set_23 else '—'}-{third_A}", pct(hit_A), need(hit_A)],
        ["三連複B", f"{one}-{box_B}-{box_B}", pct(hit_B), need(hit_B)],
        ["三連複C", f"{one}-{set_23 if set_23 else '—'}-全", pct(hit_C), need(hit_C)],
        ["二車単",   f"{one}-{set_23}" , pct(hit_EX), need(hit_EX)],
        ["二車複",   f"{one}-{set_23}" , pct(hit_QN), need(hit_QN)],
        ["ワイド",   f"{one}-{set_23}" , pct(hit_WD), need(hit_WD)],
    ]
    buy_table = pd.DataFrame(buy_rows, columns=["券種","買い目（1=◎,2=〇/▲）","想定的中率","必要オッズ"])

st.dataframe(buy_table, use_container_width=True) if 'buy_table' in locals() and buy_table is not None else None

# note（手動コピー：買い目付き）
st.markdown("### 📋 note記事用（手動コピー）")
line_text="　".join([x for x in line_inputs if str(x).strip()])
score_order_text=" ".join(str(no) for no,_ in velobi_wo)  # SBなし順
marks_line=" ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)

buy_lines = ""
if 'buy_table' in locals() and buy_table is not None:
    bl = []
    for _,r in buy_table.iterrows():
        bl.append(f"{r['券種']}: {r['買い目（1=◎,2=〇/▲）']}  p={r['想定的中率']} 必要={r['必要オッズ']}倍")
    buy_lines = "【買い目（想定的中率 / 必要オッズ）】\n" + "\n".join(bl)

note_text=(f"競輪場　{track}{race_no}R\n"
           f"{race_time}　{race_class}\n"
           f"ライン　{line_text}\n"
           f"スコア順（SBなし）　{score_order_text}\n"
           f"{marks_line}\n"
           f"自信度：{confidence}\n"
           + (("\n"+buy_lines) if buy_lines else ""))

st.text_area("ここを選択してコピー", note_text, height=260)
