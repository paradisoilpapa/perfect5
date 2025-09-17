# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re
import math, json, requests
from datetime import datetime, date, time, timedelta, timezone

# ===========================F===
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き / 統合版）", layout="wide")

# ==============================
# ★ 新規パラメータ（偏差値＆推奨ロジック）
# ==============================
HEN_W_SB   = 0.20   # SB重み
HEN_W_PROF = 0.30   # 脚質重み
HEN_W_IN   = 0.50   # 入着重み（縮約3着内率）
HEN_DEC_PLACES = 1  # 偏差値 小数一桁

HEN_THRESHOLD = 55.0     # 偏差値クリア閾値
HEN_STRONG_ONE = 60.0    # 単独強者の目安

MAX_TICKETS = 6          # 買い目最大点数

# 推奨ラベル判定用（クリア台数→方針）
# k>=5:「2車複・ワイド」中心（広く） / k=3,4:「3連複」 / k=1,2:「状況次第（軸流し寄り）」 / k=0:ケン
LABEL_MAP = {
    "wide_qn": lambda k: k >= 5,
    "trio":    lambda k: 3 <= k <= 4,
    "axis":    lambda k: k in (1,2),
    "ken":     lambda k: k == 0,
}

# 期待値レンジ（内部基準で使用可。画面非表示）
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60

# ==============================
# 既存：風・会場・マスタ
# ==============================
WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035,
    "無風": 0.0
}
WIND_MODE = "speed_only"
WIND_SIGN = -1
WIND_GAIN = 3.0
WIND_CAP  = 0.10
WIND_ZERO = 1.5
SPECIAL_DIRECTIONAL_VELODROMES = {"弥彦", "前橋"}

SESSION_HOUR = {"モーニング": 8, "デイ": 11, "ナイター": 18, "ミッドナイト": 22}
JST = timezone(timedelta(hours=9))

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
VELODROME_MASTER = {
    "函館":{"lat":41.77694,"lon":140.76283,"home_azimuth":None},
    "青森":{"lat":40.79717,"lon":140.66469,"home_azimuth":None},
    "いわき平":{"lat":37.04533,"lon":140.89150,"home_azimuth":None},
    "弥彦":{"lat":37.70778,"lon":138.82886,"home_azimuth":None},
    "前橋":{"lat":36.39728,"lon":139.05778,"home_azimuth":None},
    "取手":{"lat":35.90175,"lon":140.05631,"home_azimuth":None},
    "宇都宮":{"lat":36.57197,"lon":139.88281,"home_azimuth":None},
    "大宮":{"lat":35.91962,"lon":139.63417,"home_azimuth":None},
    "西武園":{"lat":35.76983,"lon":139.44686,"home_azimuth":None},
    "京王閣":{"lat":35.64294,"lon":139.53372,"home_azimuth":None},
    "立川":{"lat":35.70214,"lon":139.42300,"home_azimuth":None},
    "松戸":{"lat":35.80417,"lon":139.91119,"home_azimuth":None},
    "川崎":{"lat":35.52844,"lon":139.70944,"home_azimuth":None},
    "平塚":{"lat":35.32547,"lon":139.36342,"home_azimuth":None},
    "小田原":{"lat":35.25089,"lon":139.14947,"home_azimuth":None},
    "伊東":{"lat":34.954667,"lon":139.092639,"home_azimuth":None},
    "静岡":{"lat":34.973722,"lon":138.419417,"home_azimuth":None},
    "名古屋":{"lat":35.175560,"lon":136.854028,"home_azimuth":None},
    "岐阜":{"lat":35.414194,"lon":136.783917,"home_azimuth":None},
    "大垣":{"lat":35.361389,"lon":136.628444,"home_azimuth":None},
    "豊橋":{"lat":34.770167,"lon":137.417250,"home_azimuth":None},
    "富山":{"lat":36.757250,"lon":137.234833,"home_azimuth":None},
    "松坂":{"lat":34.564611,"lon":136.533833,"home_azimuth":None},
    "四日市":{"lat":34.965389,"lon":136.634500,"home_azimuth":None},
    "福井":{"lat":36.066889,"lon":136.253722,"home_azimuth":None},
    "奈良":{"lat":34.681111,"lon":135.823083,"home_azimuth":None},
    "向日町":{"lat":34.949222,"lon":135.708389,"home_azimuth":None},
    "和歌山":{"lat":34.228694,"lon":135.171833,"home_azimuth":None},
    "岸和田":{"lat":34.477500,"lon":135.369389,"home_azimuth":None},
    "玉野":{"lat":34.497333,"lon":133.961389,"home_azimuth":None},
    "広島":{"lat":34.359778,"lon":132.502889,"home_azimuth":None},
    "防府":{"lat":34.048778,"lon":131.568611,"home_azimuth":None},
    "高松":{"lat":34.345936,"lon":134.061994,"home_azimuth":None},
    "小松島":{"lat":34.005667,"lon":134.594556,"home_azimuth":None},
    "高知":{"lat":33.566694,"lon":133.526083,"home_azimuth":None},
    "松山":{"lat":33.808889,"lon":132.742333,"home_azimuth":None},
    "小倉":{"lat":33.885722,"lon":130.883167,"home_azimuth":None},
    "久留米":{"lat":33.316667,"lon":130.547778,"home_azimuth":None},
    "武雄":{"lat":33.194083,"lon":130.023083,"home_azimuth":None},
    "佐世保":{"lat":33.161667,"lon":129.712833,"home_azimuth":None},
    "別府":{"lat":33.282806,"lon":131.460472,"home_azimuth":None},
    "熊本":{"lat":32.789167,"lon":130.754722,"home_azimuth":None},
    "手入力":{"lat":None,"lon":None,"home_azimuth":None},
}

# --- 最新の印別実測率（表示はしないが内部保持）
RANK_STATS = {
    "◎": {"p1": 0.216, "pTop2": 0.456, "pTop3": 0.624},
    "〇": {"p1": 0.193, "pTop2": 0.360, "pTop3": 0.512},
    "▲": {"p1": 0.208, "pTop2": 0.384, "pTop3": 0.552},
    "△": {"p1": 0.152, "pTop2": 0.248, "pTop3": 0.384},
    "×": {"p1": 0.128, "pTop2": 0.256, "pTop3": 0.384},
    "α": {"p1": 0.088, "pTop2": 0.152, "pTop3": 0.312},
    "β": {"p1": 0.076, "pTop2": 0.151, "pTop3": 0.244},
}
RANK_FALLBACK_MARK = "△"
if RANK_FALLBACK_MARK not in RANK_STATS:
    RANK_FALLBACK_MARK = next(iter(RANK_STATS.keys()))
FALLBACK_DIST = RANK_STATS.get(RANK_FALLBACK_MARK, {"p1": 0.15, "pTop2": 0.30, "pTop3": 0.45})

# KO(勝ち上がり)関連
KO_GIRLS_SCALE = 0.0
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.010
KO_STEP_SIGMA = 0.4

# ◎ライン格上げ
LINE_BONUS_ON_TENKAI = {"優位"}
LINE_BONUS = {"second": 0.08, "thirdplus": 0.04}
LINE_BONUS_CAP = 0.10
PROB_U = {"second": 0.00, "thirdplus": 0.00}

# --- 安定度（着順分布）をT本体に入れるための重み ---
STAB_W_IN3  = 0.10   # 3着内率の重み
STAB_W_OUT  = 0.12   # 着外率の重み（マイナス補正）
STAB_W_LOWN = 0.05   # サンプル不足補正
STAB_PRIOR_IN3 = 0.33
STAB_PRIOR_OUT = 0.45
def _stab_n0(n: int) -> int:
    """サンプル不足時の事前分布の強さ（nが小さいほど強く効かせる）"""
    if n <= 6: return 12
    if n <= 14: return 8
    if n <= 29: return 5
    return 3
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
    hi = min(n,8)
    baseline = df[df["順位"].between(2,hi)]["得点"].mean()
    def corr(row):
        return round(abs(baseline-row["得点"])*0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return df.apply(corr, axis=1).tolist()

def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    s = max(0.0, float(wind_speed))
    if s <= WIND_ZERO:
        base = 0.0
    elif s <= 5.0:
        base = 0.006 * (s - WIND_ZERO)
    elif s <= 8.0:
        base = 0.021 + 0.008 * (s - 5.0)
    else:
        base = 0.045 + 0.010 * min(s - 8.0, 4.0)
    pos = {'head':1.00,'second':0.85,'single':0.75,'thirdplus':0.65}.get(role, 0.75)
    prof = 0.35 + 0.65*float(prof_escape)
    val = base * pos * prof
    if (WIND_MODE == "directional") or (s >= 7.0 and st.session_state.get("track", "") in SPECIAL_DIRECTIONAL_VELODROMES):
        wd = WIND_COEFF.get(wind_dir, 0.0)
        dir_term = clamp(s * wd * (0.30 + 0.70*float(prof_escape)) * 0.6, -0.03, 0.03)
        val += dir_term
    val = (val * float(WIND_SIGN)) * float(WIND_GAIN)
    return round(clamp(val, -float(WIND_CAP), float(WIND_CAP)), 3)

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

# KO Utilities
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
    return None
def _ko_order(v_base_map, line_def, S, B, line_factor=1.0, gap_delta=0.010):
    cars = list(v_base_map.keys())
    if not line_def or len(line_def)<1:
        return [c for c,_ in sorted(v_base_map.items(), key=lambda x:x[1], reverse=True)]
    g1, g2 = _top2_lines(line_def, S, B, line_factor)
    head1 = _extract_role_car(line_def, g1, 'head');  head2 = _extract_role_car(line_def, g2, 'head')
    sec1  = _extract_role_car(line_def, g1, 'second');sec2  = _extract_role_car(line_def, g2, 'second')
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

def _zone_from_p(p: float):
    needed = 1.0 / max(p, 1e-12)
    return needed, needed*(1.0+E_MIN), needed*(1.0+E_MAX)

def apply_anchor_line_bonus(score_raw: dict[int,float], line_of: dict[int,int], role_map: dict[int,str], anchor: int, tenkai: str) -> dict[int,float]:
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
# 風の自動取得（Open-Meteo / 時刻固定）
# ==============================
def make_target_dt_naive(jst_date, race_slot: str):
    h = SESSION_HOUR.get(race_slot, 11)
    if isinstance(jst_date, datetime):
        jst_date = jst_date.date()
    try:
        y, m, d = jst_date.year, jst_date.month, jst_date.day
    except Exception:
        dt = pd.to_datetime(str(jst_date))
        y, m, d = dt.year, dt.month, dt.day
    return datetime(y, m, d, h, 0, 0)

def fetch_openmeteo_hour(lat, lon, target_dt_naive):
    import numpy as np
    d = target_dt_naive.strftime("%Y-%m-%d")
    base = "https://api.open-meteo.com/v1/forecast"
    urls = [
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,wind_direction_10m"
         "&timezone=Asia%2FTokyo"
         f"&start_date={d}&end_date={d}", True),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m"
         "&timezone=Asia%2FTokyo"
         f"&start_date={d}&end_date={d}", False),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,wind_direction_10m"
         "&timezone=Asia%2FTokyo&past_days=2&forecast_days=2", True),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m"
         "&timezone=Asia%2FTokyo&past_days=2&forecast_days=2", False),
    ]
    last_err = None
    for url, with_dir in urls:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            j = r.json().get("hourly", {})
            times = [datetime.fromisoformat(t) for t in j.get("time", [])]
            if not times: raise RuntimeError("empty hourly times")
            diffs = [abs((t - target_dt_naive).total_seconds()) for t in times]
            k = int(np.argmin(diffs))
            sp = j.get("wind_speed_10m", [])
            di = j.get("wind_direction_10m", []) if with_dir else []
            speed = float(sp[k]) if k < len(sp) else float("nan")
            deg   = (float(di[k]) if with_dir and k < len(di) else None)
            return {"time": times[k], "speed_ms": speed, "deg": deg, "diff_min": diffs[k]/60.0}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Open-Meteo取得失敗（最後のエラー: {last_err}）")

# ==============================
# サイドバー：開催情報 / バンク・風・頭数
# ==============================

# --- 会場差分（得意会場平均を標準）ヘルパー（このブロック内に自己完結）
FAVORABLE_VENUES = ["名古屋","いわき平","前橋","立川","宇都宮","岸和田","高知"]

def _std_from_venues(names):
    Ls = [KEIRIN_DATA[v]["straight_length"] for v in names if v in KEIRIN_DATA]
    Th = [KEIRIN_DATA[v]["bank_angle"]      for v in names if v in KEIRIN_DATA]
    Cs = [KEIRIN_DATA[v]["bank_length"]     for v in names if v in KEIRIN_DATA]
    return (float(np.mean(Th)), float(np.mean(Ls)), float(np.mean(Cs)))

TH_STD, L_STD, C_STD = _std_from_venues(FAVORABLE_VENUES)

_ALL_L = np.array([KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA], float)
_ALL_TH = np.array([KEIRIN_DATA[k]["bank_angle"]      for k in KEIRIN_DATA], float)
SIG_L  = float(np.std(_ALL_L)) if np.std(_ALL_L)>1e-9 else 1.0
SIG_TH = float(np.std(_ALL_TH)) if np.std(_ALL_TH)>1e-9 else 1.0

def venue_z_terms(straight_length: float, bank_angle: float, bank_length: float):
    zL  = (float(straight_length) - L_STD)  / SIG_L
    zTH = (float(bank_angle)      - TH_STD) / SIG_TH
    if bank_length >= 480: dC = +0.4
    elif bank_length >= 380: dC = 0.0
    else: dC = -0.4
    return zL, zTH, dC

def venue_mix(zL, zTH, dC):
    # 直線長↑＝差し/捲り寄り(−)、カント↑＝先行/スピード勝負(+)、333短周長＝ライン寄り(−)
    return float(clamp(0.50*zTH - 0.35*zL - 0.30*dC, -1.0, +1.0))


st.sidebar.header("開催情報 / バンク・風・頭数")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)
track_names = list(KEIRIN_DATA.keys())
track = st.sidebar.selectbox("競輪場（プリセット）", track_names, index=track_names.index("川崎") if "川崎" in track_names else 0)
info = KEIRIN_DATA[track]
st.session_state["track"] = track

race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_day = st.sidebar.date_input("開催日（風の取得基準日）", value=date.today())

wind_dir = st.sidebar.selectbox("風向", ["無風","左上","上","右上","左","右","左下","下","右下"], index=0, key="wind_dir_input")
wind_speed_default = st.session_state.get("wind_speed", 3.0)
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 30.0, float(wind_speed_default), 0.1)

with st.sidebar.expander("🌀 風をAPIで自動取得（Open-Meteo）", expanded=False):
    api_date = st.date_input("開催日（風の取得基準日）", value=pd.to_datetime("today").date(), key="api_date")
    st.caption("基準時刻：モ=8時 / デ=11時 / ナ=18時 / ミ=22時（JST・tzなしで取得）")
    if st.button("APIで取得→風速に反映", use_container_width=True):
        info_xy = VELODROME_MASTER.get(track)
        if not info_xy or info_xy.get("lat") is None or info_xy.get("lon") is None:
            st.error(f"{track} の座標が未登録です（VELODROME_MASTER に lat/lon を入れてください）")
        else:
            try:
                target = make_target_dt_naive(api_date, race_time)
                data = fetch_openmeteo_hour(info_xy["lat"], info_xy["lon"], target)
                st.session_state["wind_speed"] = round(float(data["speed_ms"]), 2)
                st.success(f"{track} {target:%Y-%m-%d %H:%M} 風速 {st.session_state['wind_speed']:.1f} m/s （API側と{data['diff_min']:.0f}分ズレ）")
                st.rerun()
            except Exception as e:
                st.error(f"取得に失敗：{e}")

straight_length = st.sidebar.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle = st.sidebar.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length = st.sidebar.number_input("周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)

base_laps = st.sidebar.number_input("周回（通常4）", 1, 10, 4, 1)
day_label = st.sidebar.selectbox("開催日", ["初日","2日目","最終日"], 0)
eff_laps = int(base_laps) + {"初日":1,"2日目":2,"最終日":3}[day_label]

race_class = st.sidebar.selectbox("級別", ["Ｓ級","Ａ級","Ａ級チャレンジ","ガールズ"], 0)

# === 会場styleを「得意会場平均」を基準に再定義
zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bank_length)
style_raw = venue_mix(zL, zTH, dC)
override = st.sidebar.slider("会場バイアス補正（−2差し ←→ +2先行）", -2.0, 2.0, 0.0, 0.1)
style = clamp(style_raw + 0.25*override, -1.0, +1.0)

CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

# 旧：
# DAY_FACTOR = {"初日":1.00, "2日目":0.60, "最終日":0.85}

# 新（まずは完全フラット）：
DAY_FACTOR = {"初日":1.00, "2日目":1.00, "最終日":1.00}
day_factor = DAY_FACTOR[day_label]

cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

# ===== 日程・級別・頭数で“周回疲労の効き”を薄くシフト（出力には出さない） =====
DAY_SHIFT = {"初日": -0.5, "2日目": 0.0, "最終日": +0.5}
CLASS_SHIFT = {"Ｓ級": 0.0, "Ａ級": +0.10, "Ａ級チャレンジ": +0.20, "ガールズ": -0.10}
HEADCOUNT_SHIFT = {5: -0.20, 6: -0.10, 7: -0.05, 8: 0.0, 9: +0.10}

def fatigue_extra(eff_laps: int, day_label: str, n_cars: int, race_class: str) -> float:
    """
    既存の extra = max(eff_laps - 2, 0) をベースに、
    ・日程シフト：初日 -0.5／2日目 0／最終日 +0.5
    ・級別シフト：A級/チャレンジをやや重め、ガールズはやや軽め
    ・頭数シフト：9車は少し重く、5〜7車は少し軽く
    """
    d = float(DAY_SHIFT.get(day_label, 0.0))
    c = float(CLASS_SHIFT.get(race_class, 0.0))
    h = float(HEADCOUNT_SHIFT.get(int(n_cars), 0.0))
    x = (float(eff_laps) - 2.0) + d + c + h
    return max(0.0, x)


line_sb_enable = (race_class != "ガールズ")

st.sidebar.caption(
    f"会場スタイル: {style:+.2f}（raw {style_raw:+.2f}） / "
    f"級別: spread={cf['spread']:.2f}, line={cf['line']:.2f} / "
    f"日程係数(line)={day_factor:.2f} → line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}"
)

# ==============================
# メイン：入力
# ==============================
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き：統合版）⭐")
st.caption(f"風補正モード: {WIND_MODE}（'speed_only'=風速のみ / 'directional'=向きも薄く考慮）")

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
n_cars = int(n_cars)
lines = [extract_car_list(x, n_cars) for x in line_inputs if str(x).strip()]
line_def, car_to_group = build_line_maps(lines)
active_cars = sorted({c for lst in lines for c in lst}) if lines else list(range(1, n_cars+1))

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

# --- 脚質プロフィール（会場適性：得意会場平均基準のstyleを掛ける）
prof_base, prof_escape, prof_sashi, prof_oikomi = {}, {}, {}, {}
for no in active_cars:
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark = 0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    prof_escape[no]=esc; prof_sashi[no]=sashi; prof_oikomi[no]=mark
    base = esc*BASE_BY_KAKU["逃"] + mak*BASE_BY_KAKU["捲"] + sashi*BASE_BY_KAKU["差"] + mark*BASE_BY_KAKU["マ"]
    vmix = style
    venue_bonus = 0.06 * vmix * ( +1.00*esc + 0.40*mak - 0.60*sashi - 0.25*mark )
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

# ===== 会場個性を“個人スコア”に浸透：bank系補正を差し替え =====
def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi):
    zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bank_length)
    base = clamp(0.06*zTH - 0.05*zL - 0.03*dC, -0.08, +0.08)
    return round(base*float(prof_escape) - 0.5*base*float(prof_sashi), 3)

def bank_length_adjust(bank_length, prof_oikomi):
    dC = (+0.4 if bank_length>=480 else 0.0 if bank_length>=380 else -0.4)
    return round(0.03*(-dC)*float(prof_oikomi), 3)

# --- 安定度（着順分布）をT本体に入れるための重み（強化版） ---
STAB_W_IN3  = 0.18   # 3着内の寄与
STAB_W_OUT  = 0.22   # 着外のペナルティ
STAB_W_LOWN = 0.06   # サンプル不足ペナルティ
STAB_PRIOR_IN3 = 0.33
STAB_PRIOR_OUT = 0.45

def stability_score(no: int) -> float:
    n1 = x1.get(no, 0); n2 = x2.get(no, 0); n3 = x3.get(no, 0); nOut = x_out.get(no, 0)
    n  = n1 + n2 + n3 + nOut
    if n <= 0:
        return 0.0
    # 少サンプル縮約（この関数内で完結）
    if n <= 6:    n0 = 12
    elif n <= 14: n0 = 8
    elif n <= 29: n0 = 5
    else:         n0 = 3

    in3  = (n1 + n2 + n3 + n0*STAB_PRIOR_IN3) / (n + n0)
    out_ = (nOut          + n0*STAB_PRIOR_OUT) / (n + n0)

    bonus = 0.0
    bonus += STAB_W_IN3 * (in3 - STAB_PRIOR_IN3) * 2.0
    bonus -= STAB_W_OUT * (out_ - STAB_PRIOR_OUT) * 2.0

    if n < 10:
        bonus -= STAB_W_LOWN * (10 - n) / 10.0

    # キャップ：nに応じて段階的に広げる（±0.35〜±0.45）
    cap = 0.35
    if n >= 15: cap = 0.45
    elif n >= 10: cap = 0.40

    return clamp(bonus, -cap, +cap)

# ===== SBなし合計（環境補正 + 得点微補正 + 個人補正 + 周回疲労 + 安定度） =====
tens_list = [ratings_val[no] for no in active_cars]
t_corr = tenscore_correction(tens_list) if active_cars else []
tens_corr = {no:t_corr[i] for i,no in enumerate(active_cars)} if active_cars else {}

rows = []
_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir",   wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

for no in active_cars:
    role = role_in_line(no, line_def)

    # 周回疲労（DAY×頭数×級別を反映）
    extra = fatigue_extra(eff_laps, day_label, n_cars, race_class)
    fatigue_scale = (1.0 if race_class == "Ｓ級" else
                     1.1 if race_class == "Ａ級" else
                     1.2 if race_class == "Ａ級チャレンジ" else
                     1.05)
    laps_adj = (
        -0.10 * extra * (1.0 if prof_escape[no] > 0.5 else 0.0)
        + 0.05 * extra * (1.0 if prof_oikomi[no] > 0.4 else 0.0)
    ) * fatigue_scale

    wind = _wind_func(eff_wind_dir, float(eff_wind_speed or 0.0), role, float(prof_escape[no]))
    bank_b   = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no])
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv = extra_bonus.get(no, 0.0)
    stab  = stability_score(no)  # 安定度

    total_raw = (prof_base[no] + wind + cf["spread"] * tens_corr.get(no, 0.0)
                 + bank_b + length_b + laps_adj + indiv + stab)

    rows.append([int(no), role, round(prof_base[no],3), round(wind,3),
                 round(cf["spread"] * tens_corr.get(no, 0.0),3),
                 round(bank_b,3), round(length_b,3), round(laps_adj,3),
                 round(indiv,3), round(stab,3), total_raw])


df = pd.DataFrame(rows, columns=[
    "車番","役割","脚質基準(会場)","風補正","得点補正","バンク補正",
    "周長補正","周回補正","個人補正","安定度","合計_SBなし_raw",
])
mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0 * (df["合計_SBなし_raw"] - mu)

# ===== KO方式（印に混ぜず：展開・ケンで利用） =====
v_wo = {int(k): float(v) for k, v in zip(df["車番"].astype(int), df["合計_SBなし"].astype(float))}
_is_girls = (race_class == "ガールズ")
head_scale = KO_HEADCOUNT_SCALE.get(int(n_cars), 1.0)
ko_scale_raw = (KO_GIRLS_SCALE if _is_girls else 1.0) * head_scale
KO_SCALE_MAX = 0.45
ko_scale = min(ko_scale_raw, KO_SCALE_MAX)

if ko_scale > 0.0 and line_def and len(line_def) >= 1:
    ko_order = _ko_order(v_wo, line_def, S, B,
                         line_factor=line_factor_eff,
                         gap_delta=KO_GAP_DELTA)
    vals = [v_wo[c] for c in v_wo.keys()]
    mu0  = float(np.mean(vals)); sd0 = float(np.std(vals) + 1e-12)
    KO_STEP_SIGMA_LOCAL = max(0.25, KO_STEP_SIGMA * 0.7)
    step = KO_STEP_SIGMA_LOCAL * sd0

    new_scores = {}
    for rank, car in enumerate(ko_order, start=1):
        rank_adjust = step * (len(ko_order) - rank)
        blended = (1.0 - ko_scale) * v_wo[car] + ko_scale * (
            mu0 + rank_adjust - (len(ko_order)/2.0 - 0.5)*step
        )
        new_scores[car] = blended
    v_final = {int(k): float(v) for k, v in new_scores.items()}
else:
    if v_wo:
        ko_order = sorted(v_wo.keys(), key=lambda c: v_wo[c], reverse=True)
        v_final = {int(c): float(v_wo[c]) for c in ko_order}
    else:
        ko_order = []
        v_final = {}

# --- 純SBなしランキング（KOまで／格上げ前）
df_sorted_pure = pd.DataFrame({
    "車番": list(v_final.keys()),
    "合計_SBなし": [round(float(v_final[c]), 6) for c in v_final.keys()]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

# ===== 印用（既存の安全弁を維持）
FINISH_WEIGHT   = globals().get("FINISH_WEIGHT", 6.0)
FINISH_WEIGHT_G = globals().get("FINISH_WEIGHT_G", 3.0)
POS_BONUS  = globals().get("POS_BONUS", {0: 0.0, 1: -0.6, 2: -0.9, 3: -1.2, 4: -1.4})
POS_WEIGHT = globals().get("POS_WEIGHT", 1.0)
SMALL_Z_RATING = globals().get("SMALL_Z_RATING", 0.01)
FINISH_CLIP = globals().get("FINISH_CLIP", 4.0)
TIE_EPSILON  = globals().get("TIE_EPSILON", 0.8)

p2_list = [float(p2_eff.get(n, 0.0)) for n in active_cars]
if len(p2_list) >= 1:
    mu_p2  = float(np.mean(p2_list))
    sd_p2  = float(np.std(p2_list) + 1e-12)
else:
    mu_p2, sd_p2 = 0.0, 1.0
p2z_map = {n: (float(p2_eff.get(n, 0.0)) - mu_p2) / sd_p2 for n in active_cars}
p1_eff_safe = {n: float(p1_eff.get(n, 0.0)) if 'p1_eff' in globals() and p1_eff is not None else 0.0 for n in active_cars}
p2only_map = {n: max(0.0, float(p2_eff.get(n, 0.0)) - float(p1_eff_safe.get(n, 0.0))) for n in active_cars}
zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
zt_map = {n: float(zt[i]) for i, n in enumerate(active_cars)} if active_cars else {}

def _pos_idx(no:int) -> int:
    g = car_to_group.get(no, None)
    if g is None or g not in line_def:
        return 0
    grp = line_def[g]
    try:
        return max(0, int(grp.index(no)))
    except Exception:
        return 0

bonus_init,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)

def anchor_score(no:int) -> float:
    base = float(v_final.get(no, -1e9))
    role = role_in_line(no, line_def)
    sb = float(bonus_init.get(car_to_group.get(no, None), 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0))
    pos_term = POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)
    if _is_girls:
        raw_finish = FINISH_WEIGHT_G * float(p2z_map.get(no, 0.0))
    else:
        raw_finish = FINISH_WEIGHT * float(p2z_map.get(no, 0.0))
    finish_term = max(-FINISH_CLIP, min(FINISH_CLIP, raw_finish))
    return base + sb + pos_term + finish_term + SMALL_Z_RATING * zt_map.get(no, 0.0)

# ===== ◎候補抽出（既存ロジック維持）
cand_sorted = sorted(active_cars, key=lambda n: anchor_score(n), reverse=True)
C = cand_sorted[:min(3, len(cand_sorted))]
ratings_sorted2 = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank2 = {n: i+1 for i,n in enumerate(ratings_sorted2)}
ALLOWED_MAX_RANK = globals().get("ALLOWED_MAX_RANK", 5)

guarantee_top_rating = True
if guarantee_top_rating and (race_class == "ガールズ") and len(ratings_sorted2) >= 1:
    top_rating_car = ratings_sorted2[0]
    if top_rating_car not in C:
        C = [top_rating_car] + [c for c in C if c != top_rating_car]
        C = C[:min(3, len(cand_sorted))]

ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)
rank_pure = {int(df_sorted_pure.loc[i, "車番"]): i+1 for i in range(len(df_sorted_pure))}
cand_pool = [c for c in C if rank_pure.get(c, 999) <= ANCHOR_CAND_SB_TOPK]
if not cand_pool:
    cand_pool = [int(df_sorted_pure.loc[i, "車番"]) for i in range(min(ANCHOR_CAND_SB_TOPK, len(df_sorted_pure)))]
anchor_no_pre = max(cand_pool, key=lambda x: anchor_score(x)) if cand_pool else int(df_sorted_pure.loc[0, "車番"])
anchor_no = anchor_no_pre
top2 = sorted(cand_pool, key=lambda x: anchor_score(x), reverse=True)[:2]
if len(top2) >= 2:
    s1 = anchor_score(top2[0]); s2 = anchor_score(top2[1])
    if (s1 - s2) < TIE_EPSILON:
        better_by_rating = min(top2, key=lambda x: ratings_rank2.get(x, 999))
        anchor_no = better_by_rating
if rank_pure.get(anchor_no, 999) > ANCHOR_REQUIRE_TOP_SB:
    pool = [c for c in cand_pool if rank_pure.get(c, 999) <= ANCHOR_REQUIRE_TOP_SB]
    if pool:
        anchor_no = max(pool, key=lambda x: anchor_score(x))
    else:
        anchor_no = int(df_sorted_pure.loc[0, "車番"])
    st.caption(f"※ ◎は『SBなし 上位{ANCHOR_REQUIRE_TOP_SB}位以内』縛りで {anchor_no_pre}→{anchor_no} に調整。")

role_map = {no: role_in_line(no, line_def) for no in active_cars}
cand_scores = [anchor_score(no) for no in C] if len(C) >= 2 else [0, 0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf_gap = cand_scores_sorted[0] - cand_scores_sorted[1] if len(cand_scores_sorted) >= 2 else 0.0
spread = float(np.std(list(v_final.values()))) if len(v_final) >= 2 else 0.0
norm = conf_gap / (spread if spread > 1e-6 else 1.0)
confidence = "優位" if norm >= 1.0 else ("互角" if norm >= 0.5 else "混戦")

score_adj_map = apply_anchor_line_bonus(v_final, car_to_group, role_map, anchor_no, confidence)

df_sorted_wo = pd.DataFrame({
    "車番": active_cars,
    "合計_SBなし": [round(float(score_adj_map.get(int(c), v_final.get(int(c), float("-inf")))), 6) for c in active_cars]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

velobi_wo = list(zip(df_sorted_wo["車番"].astype(int).tolist(),
                     df_sorted_wo["合計_SBなし"].round(3).tolist()))

# ==============================
# ★ レース内T偏差値 → 印 → 買い目 → note出力（2車系対応＋会場個性浸透版）
# ==============================
import math
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations

# ===== しきい値（S＝偏差値Tの合算） =====
S_TRIO_MIN_WIDE  = 158.0   # 三連複：手広く
S_TRIO_MIN_CORE  = 163.0   # 三連複：基準クリア（これが“本線”）
S_QN_MIN         = 122.0
S_WIDE_MIN       = 116.0

# 三連単は“基準クリア”側に合わせて運用（相談どおり164）
S_TRIFECTA_MIN   = 164.0

# 目標回収率（据え置き）
TARGET_ROI = {"trio":1.20, "qn":1.10, "wide":1.05}
ODDS_FLOOR_QN   = 8.0
ODDS_FLOOR_WIDE = 4.0
HEN_DEC_PLACES = 1
EPS = 1e-12


# ====== ユーティリティ ======
def coerce_score_map(d, n_cars: int) -> dict[int, float]:
    out: dict[int, float] = {}
    t = str(type(d)).lower()
    if "pandas.core.frame" in t:
        df_ = d
        car_col = "車番" if "車番" in df_.columns else None
        if car_col is None:
            for c in df_.columns:
                if np.issubdtype(df_[c].dtype, np.integer):
                    car_col = c; break
        score_col = None
        for cand in ["合計_SBなし","SBなし","スコア","score","SB_wo","SB"]:
            if cand in df_.columns:
                score_col = cand; break
        if score_col is None:
            for c in df_.columns:
                if c == car_col: continue
                if np.issubdtype(df_[c].dtype, np.number):
                    score_col = c; break
        if car_col is not None and score_col is not None:
            for _, r in df_.iterrows():
                try:
                    i = int(r[car_col]); x = float(r[score_col])
                except Exception:
                    continue
                out[i] = x
    elif "pandas.core.series" in t:
        for k, v in d.to_dict().items():
            try:
                i = int(k); x = float(v)
            except Exception:
                continue
            out[i] = x
    elif hasattr(d, "items"):
        for k, v in d.items():
            try:
                i = int(k); x = float(v)
            except Exception:
                continue
            out[i] = x
    elif isinstance(d, (list, tuple, np.ndarray)):
        arr = list(d)
        if len(arr) == n_cars and all(not isinstance(x,(list,tuple,dict)) for x in arr):
            for idx, v in enumerate(arr, start=1):
                try: out[idx] = float(v)
                except Exception: out[idx] = np.nan
        else:
            for it in arr:
                if isinstance(it,(list,tuple)) and len(it) >= 2:
                    try:
                        i = int(it[0]); x = float(it[1])
                        out[i] = x
                    except Exception:
                        continue
    for i in range(1, int(n_cars)+1):
        out.setdefault(i, np.nan)
    return out

def t_score_from_finite(values: np.ndarray, eps: float = 1e-9):
    v = values.astype(float, copy=True)
    finite = np.isfinite(v)
    k = int(finite.sum())
    if k < 2:
        return np.full_like(v, 50.0), (float("nan") if k==0 else float(v[finite][0])), 0.0, k
    mu = float(np.mean(v[finite])); sd = float(np.std(v[finite], ddof=0))
    if (not np.isfinite(sd)) or sd < eps:
        return np.full_like(v, 50.0), mu, 0.0, k
    T = 50.0 + 10.0 * ((v - mu) / sd)
    T[~finite] = 50.0
    return T, mu, sd, k

def _format_rank_from_array(ids, arr):
    pairs = [(i, float(arr[idx])) for idx, i in enumerate(ids)]
    pairs.sort(key=lambda kv: ((1,0) if not np.isfinite(kv[1]) else (0,-kv[1]), kv[0]))
    return " ".join(str(i) for i,_ in pairs)

# ====== ここから処理本体 ======

# 1) 母集団車番
try:
    USED_IDS = sorted(int(i) for i in (active_cars if active_cars else range(1, n_cars+1)))
except Exception:
    USED_IDS = list(range(1, int(n_cars)+1))
M = len(USED_IDS)

# 2) SBなしのソース（df優先→velobi_wo）
score_map_from_df = coerce_score_map(globals().get("df_sorted_wo", None), n_cars)
score_map_vwo     = coerce_score_map(globals().get("velobi_wo", None),   n_cars)
SB_BASE_MAP = score_map_from_df if any(np.isfinite(list(score_map_from_df.values()))) else score_map_vwo

# 3) スコア配列（スコア順表示と偏差値母集団を共用）
xs_base_raw = np.array([SB_BASE_MAP.get(i, np.nan) for i in USED_IDS], dtype=float)

# 4) 偏差値T（レース内：平均50・SD10、NaN→50）
xs_race_t, mu_sb, sd_sb, k_finite = t_score_from_finite(xs_base_raw)

missing = ~np.isfinite(xs_base_raw)
if missing.any():
    sb_for_sort = {i: SB_BASE_MAP.get(i, -1e18) for i in USED_IDS}
    idxs = np.where(missing)[0].tolist()
    idxs.sort(key=lambda ii: (-float(sb_for_sort.get(USED_IDS[ii], -1e18)), USED_IDS[ii]))
    k = len(idxs); delta = 0.12; center = (k - 1)/2.0 if k > 1 else 0.0
    for r, ii in enumerate(idxs):
        xs_race_t[ii] = 50.0 + delta * (center - r)

# 5) dict化・表示用
race_t = {USED_IDS[idx]: float(round(xs_race_t[idx], HEN_DEC_PLACES)) for idx in range(M)}

race_z = (xs_race_t - 50.0) / 10.0

hen_df = pd.DataFrame({
    "車": USED_IDS,
    "SBなし(母集団)": [None if not np.isfinite(x) else float(x) for x in xs_base_raw],
    "偏差値T(レース内)": [race_t[i] for i in USED_IDS],
}).sort_values(["偏差値T(レース内)","車"], ascending=[False, True]).reset_index(drop=True)

st.markdown("### 偏差値（レース内T＝平均50・SD10｜SBなしと同一母集団）")
st.caption(f"μ={mu_sb if np.isfinite(mu_sb) else 'nan'} / σ={sd_sb:.6f} / 有効件数k={k_finite}")
st.dataframe(hen_df, use_container_width=True)

# 6) PL用重み（購入計算に使用：既存近似）
tau = 1.0
w   = np.exp(race_z * tau)
S_w = float(np.sum(w))
w_idx = {USED_IDS[idx]: float(w[idx]) for idx in range(M)}

def prob_top2_pair_pl(i: int, j: int) -> float:
    wi, wj = w_idx[i], w_idx[j]
    d_i = max(S_w - wi, EPS); d_j = max(S_w - wj, EPS)
    return (wi / S_w) * (wj / d_i) + (wj / S_w) * (wi / d_j)

def prob_top3_triple_pl(i: int, j: int, k: int) -> float:
    a, b, c = w_idx[i], w_idx[j], w_idx[k]
    total = 0.0
    for x, y, z in ((a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)):
        d1 = max(S_w - x, EPS)
        d2 = max(S_w - x - y, EPS)
        total += (x / S_w) * (y / d1) * (z / d2)
    return total

def prob_wide_pair_pl(i: int, j: int) -> float:
    total = 0.0
    for k in USED_IDS:
        if k == i or k == j: continue
        total += prob_top3_triple_pl(i, j, k)
    return total

# 7) 印（◎〇▲）＝ T↓ → SBなし↓ → 車番↑（βは除外）
if "select_beta" not in globals():
    def select_beta(cars): return None
if "enforce_alpha_eligibility" not in globals():
    def enforce_alpha_eligibility(m): return m

# ===== βラベル付与（単なる順位ラベル） =====
def assign_beta_label(result_marks: dict[str,int], used_ids: list[int], df_sorted) -> dict[str,int]:
    marks = dict(result_marks)
    # 6車以下は出さない（集計仕様）
    if len(used_ids) <= 6:
        return marks
    # 既にβがあれば何もしない
    if "β" in marks:
        return marks
    try:
        last_car = int(df_sorted.loc[len(df_sorted)-1, "車番"])
        if last_car not in marks.values():
            marks["β"] = last_car
    except Exception:
        pass
    return marks


try:
    beta_id = beta_id if ('beta_id' in globals() and beta_id is not None) else select_beta(list(USED_IDS))
except Exception:
    beta_id = None

result_marks = {}
reasons = {}

if beta_id is not None:
    result_marks["β"] = int(beta_id)
    reasons[beta_id] = reasons.get(beta_id, "β（来ない枠：選別ロジック）")

sb_base = {USED_IDS[idx]: float(xs_base_raw[idx]) if np.isfinite(xs_base_raw[idx]) else float("-inf") for idx in range(M)}

def _race_t_val(i: int) -> float:
    try: return float(race_t.get(int(i), 50.0))
    except Exception: return 50.0

seed_pool = [i for i in USED_IDS if i != result_marks.get("β")]
order_by_T = sorted(seed_pool, key=lambda i: (-_race_t_val(i), -sb_base.get(i, float("-inf")), i))
for mk, car in zip(["◎","〇","▲"], order_by_T):
    result_marks[mk] = car

line_def     = globals().get("line_def", {}) or {}
car_to_group = globals().get("car_to_group", {}) or {}

anchor_no = result_marks.get("◎", None)
mates_sorted = []
if anchor_no is not None:
    a_gid = car_to_group.get(anchor_no, None)
    if a_gid is not None and a_gid in line_def:
        used_now = set(result_marks.values())
        mates_sorted = sorted(
            [c for c in line_def[a_gid] if c not in used_now and c != result_marks.get("β")],
            key=lambda x: (-sb_base.get(x, float("-inf")), x)
        )

used = set(result_marks.values())
overall_rest = [c for c in USED_IDS if c not in used]
overall_rest = sorted(overall_rest, key=lambda x: (-sb_base.get(x, float("-inf")), x))
tail_priority = mates_sorted + [c for c in overall_rest if c not in mates_sorted]

for mk in ["△","×","α"]:
    if mk in result_marks: continue
    if not tail_priority: break
    no = tail_priority.pop(0)
    result_marks[mk] = no
    reasons[no] = f"{mk}（◎ライン優先→残りスコア順）"

result_marks = enforce_alpha_eligibility(result_marks)

result_marks = assign_beta_label(result_marks, USED_IDS, df_sorted_wo)


if "α" not in result_marks:
    used_now = set(result_marks.values())
    pool = [i for i in USED_IDS if (i not in used_now and i != beta_id)]
    if pool:
        alpha_pick = pool[-1]
        result_marks["α"] = alpha_pick
        reasons[alpha_pick] = reasons.get(alpha_pick, "α（フォールバック：禁止条件全滅→最弱を採用）")

# 偏差値ベースの合算
S_BASE_MAP = {i: float(race_t[i]) for i in USED_IDS}

def _pair_score(a,b):   return S_BASE_MAP.get(a,0.0) + S_BASE_MAP.get(b,0.0)
def _trio_score(a,b,c): return S_BASE_MAP.get(a,0.0) + S_BASE_MAP.get(b,0.0) + S_BASE_MAP.get(c,0.0)

pairs = [(a,b,_pair_score(a,b)) for (a,b) in combinations(USED_IDS, 2)]
trios = [(a,b,c,_trio_score(a,b,c)) for (a,b,c) in combinations(USED_IDS, 3)]

# 三連複を「基準クリア(>=164)」「手広く(158〜163.9)」に分割
def _trio_rows_with_star(rows, anchor):
    out = []
    for (a,b,c,s) in rows:
        buy = "-".join(map(str, sorted([a,b,c])))
        star = "☆" if (anchor in (a,b,c)) else ""
        out.append((buy, round(s,1), star))
    # 表示は S↓ → 買い目文字列↑
    out.sort(key=lambda x: (-x[1], x[0]))
    return out

trios_core_raw = [(a,b,c,s) for (a,b,c,s) in trios if s >= S_TRIO_MIN_CORE]
trios_wide_raw = [(a,b,c,s) for (a,b,c,s) in trios if (S_TRIO_MIN_WIDE <= s < S_TRIO_MIN_CORE)]

trios_core = _trio_rows_with_star(trios_core_raw, anchor_no)
trios_wide = _trio_rows_with_star(trios_wide_raw, anchor_no)

# --- 三連複を 164 以上(core) / 158〜163.9(wide) に分割した直後に置く ---
all_trios_for_odds = trios_core_raw + trios_wide_raw

def _min_required_from_trios(rows, p_func, roi: float) -> float|None:
    if not rows: 
        return None
    reqs = []
    for a,b,c,s in rows:
        p = p_func(a,b,c)
        if p > EPS:
            reqs.append(roi / p)
    if not reqs:
        return None
    m = min(reqs)
    return math.floor(m*2 + 0.5) / 2.0



# ワイド/二車複は従来どおり
pairs_w = sorted([(a,b,s) for (a,b,s) in pairs if s >= S_WIDE_MIN], key=lambda x:(-x[2], x[0], x[1]))


min_odds_wide = _min_required_from_pairs(pairs_w,   prob_wide_pair_pl,   TARGET_ROI["wide"])

if min_odds_wide is not None: min_odds_wide = max(min_odds_wide, ODDS_FLOOR_WIDE)

anchor_no = result_marks.get("◎", None)

def _df_trio(rows):
    out = []
    for (a,b,c,s) in rows:
        combo = sorted([a,b,c])
        label = "-".join(map(str, combo))
        if anchor_no in combo:
            label += "☆"
        out.append({"買い目": label, "偏差値S": round(s,1)})
    return pd.DataFrame(out)

def _df_pair(rows):
    return pd.DataFrame([{"買い目":f"{a}-{b}", "偏差値S":round(s,1)} for (a,b,s) in rows])

# ====== 二車系 専用偏差値（新設） ======
def _to_hensachi(arr):
    a = np.asarray(arr, float)
    m, s = float(np.mean(a)), float(np.std(a))
    if not np.isfinite(s) or s < 1e-12:
        return [50.0 for _ in a]
    return list(50.0 + 10.0*(a - m)/s)

# 連対率（2着内）と勝率
pTop2_map = {i: float(p1_eff.get(i,0.0)+p2_eff.get(i,0.0)) for i in USED_IDS}
p1_map    = {i: float(p1_eff.get(i,0.0))                   for i in USED_IDS}

hensachi_top2 = {i: float(round(x,1)) for i,x in zip(USED_IDS, _to_hensachi([pTop2_map[i] for i in USED_IDS]))}
hensachi_win  = {i: float(round(x,1)) for i,x in zip(USED_IDS, _to_hensachi([p1_map[i]    for i in USED_IDS]))}

def _pair_score_top2(a,b): return hensachi_top2.get(a,50.0) + hensachi_top2.get(b,50.0)

# 二車複（新：連対率偏差値）
S_QN2_MIN = 122.0
pairs_qn2 = []
for (a,b) in combinations(USED_IDS, 2):
    s = _pair_score_top2(a,b)
    if s >= S_QN2_MIN:
        pairs_qn2.append((a,b,s))
pairs_qn2.sort(key=lambda x:(-x[2], x[0], x[1]))

# 二車単（新：勝率偏差値）
S_NITAN_MIN = 124.0
rows_nitan = []
for (a,b) in combinations(USED_IDS, 2):
    s_ab = hensachi_win.get(a,50.0) + hensachi_top2.get(b,50.0)
    s_ba = hensachi_win.get(b,50.0) + hensachi_top2.get(a,50.0)
    if s_ab >= S_NITAN_MIN: rows_nitan.append((f"{a}-{b}", s_ab))
    if s_ba >= S_NITAN_MIN: rows_nitan.append((f"{b}-{a}", s_ba))
rows_nitan.sort(key=lambda x:(-x[1], x[0]))

# ===== 三連単生成（新：二車単＋三連複対応） =====
def _gen_trifecta_from_nitan(trios_rows, nitan_pairs):
    """
    trios_rows: [(a,b,c,s)]  # 三連複候補（Sは元の三複S）
    nitan_pairs: [(h,t)]     # 二車単合格ペア（順序あり：h→t）
    return: [{"買い目":"h-t-x","元三連複S":s}, ...]
    """
    out = []
    for (a,b,c,s) in trios_rows:
        comb = {a,b,c}
        for (h,t) in nitan_pairs:
            if h in comb and t in comb:
                third = a if a not in (h,t) else (b if b not in (h,t) else c)
                out.append((f"{h}-{t}-{third}", s))
    # 重複はSが高い方を採用
    dedup = {}
    for k, s in out:
        if s > dedup.get(k, -1e18):
            dedup[k] = s
    return [{"買い目": k, "元三連複S": round(v,1)} for k,v in
            sorted(dedup.items(), key=lambda kv: (-kv[1], kv[0]))]

# 二車単の合格ペア（順序あり）を抽出
_nitan_pairs = []
for k, _s in rows_nitan:
    try:
        h, t = map(int, k.split("-"))
        _nitan_pairs.append((h, t))
    except Exception:
        pass

# rows_nitan は既存（勝率偏差値S1>=124）を使用
# 三連単候補：二車単(先着a-b)が含まれる三連複の3番目を全列挙（Sが高い順でソート）
def build_trifecta_from_nitan_and_trio(nitan_rows, trio_rows, s_min=S_TRIFECTA_MIN):
    # trio_rows は (buy_str, S, star) 形式 → 逆変換して参照テーブル化
    trio_sets = []  # (frozenset{a,b,c}, S)
    for buy, s, _ in trio_rows:
        a,b,c = map(int, buy.split("-"))
        trio_sets.append((frozenset([a,b,c]), s))
    out = []
    for k, _s1 in nitan_rows:
        a,b = map(int, k.split("-"))
        for st, s in trio_sets:
            if s < s_min:  # 三連単もしきい値164以上で絞る
                continue
            if a in st and b in st:
                third = list(st - {a,b})[0]
                out.append((f"{a}-{b}-{third}", round(s,1)))
    # ダブり排除 & S↓ → 目安として買い目文字列↑
    uniq = {}
    for k,s in out:
        if k not in uniq or s > uniq[k]:
            uniq[k] = s
    rows = [(k, uniq[k]) for k in uniq]
    rows.sort(key=lambda x:(-x[1], x[0]))
    return rows

rows_trifecta = build_trifecta_from_nitan_and_trio(rows_nitan, trios_core + trios_wide, s_min=S_TRIFECTA_MIN)



# === 三連複（基準クリア） ===
if trios_core:
    hdr = "#### 三連複（基準クリア 164以上）"
    if min_odds_trio is not None:
        hdr += f"　/　**最低限オッズ {min_odds_trio:.1f}倍以上**"
    st.markdown(hdr)
    st.dataframe(pd.DataFrame([{"買い目":b + ("☆" if star else ""), "偏差値S":s} for (b,s,star) in trios_core]),
                 use_container_width=True)
else:
    st.markdown("#### 三連複（基準クリア 164以上：該当なし）")

# === 三連複（手広く 158〜163.9） ===
if trios_wide:
    st.markdown("#### 三連複（手広く 158〜163.9）")
    st.dataframe(pd.DataFrame([{"買い目":b + ("☆" if star else ""), "偏差値S":s} for (b,s,star) in trios_wide]),
                 use_container_width=True)
else:
    st.markdown("#### 三連複（手広く 158〜163.9：該当なし）")

# 二車複・ワイド・二車単（従来の表示はそのまま）
# ...

# === 三連単（連動セクション） ===
if rows_trifecta:
    st.markdown("#### 三連単（**二車単＋三連複** 連動・S≥164）")
    st.dataframe(pd.DataFrame([{"買い目":k, "参考S(三連複S)":v} for (k,v) in rows_trifecta]),
                 use_container_width=True)
else:
    st.markdown("#### 三連単（連動：該当なし）")


# 9) note用出力（ヘッダー順：{track}{race_no}R → 展開評価 → 推奨）
def _join_trio_note(rows):
    # rows: (buy, s, star)
    return "\n".join([f"{b}{('☆' if star else '')}（S={s:.1f}）" for (b,s,star) in rows]) if rows else "対象外"

def _fmt_hen_lines(ts_map: dict, ids: list[int]) -> str:
    out = []
    for n in ids:
        v = ts_map.get(n, "—")
        out.append(f"{n}: {float(v):.{HEN_DEC_PLACES}f}" if isinstance(v,(int,float)) else f"{n}: —")
    return "\n".join(out)

line_text  = "　".join([x for x in globals().get("line_inputs", []) if str(x).strip()])
marks_line = " ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)
score_order_text = _format_rank_from_array(USED_IDS, xs_base_raw)

header_lines = [
    f"{track}{race_no}R",
    f"展開評価：{confidence}",
    ("推奨 3連複" if (trios_core or trios_wide) else
     ("推奨 2車複・ワイド" if (len(pairs_qn2)+len(pairs_w))>=1 else "推奨 ケン")),
]
header_text = "\n".join(header_lines)

note_text = (
    header_text + "\n\n"
    f"{race_time}　{race_class}\n"
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}\n\n"
    "偏差値（風・ライン込み）\n"
    "— レース内基準（平均50・SD10） —\n"
    f"{_fmt_hen_lines(race_t, USED_IDS)}\n\n"
    # 三連複：二段構え＋☆注記
    + ("三連複（**基準クリア 164以上**／最低限オッズ "
       + (f"{min_odds_trio:.1f}" if min_odds_trio is not None else "—")
       + "倍以上）\n"
       + _join_trio_note(trios_core) + "\n\n")
    + ("三連複（**手広く 158〜163.9**）\n"
       + _join_trio_note(trios_wide)
       + "\n\n※「☆」は◎が絡む買い目 → ◎の３着率を考えると、絞るならここだけでも。\n\n")
    # 二車複
    + (("二車複（連対率偏差値 合計S2≥" + str(int(S_QN2_MIN)) + "）\n" +
        ("\n".join([f"{a}-{b}（S2={s:.1f}）" for (a,b,s) in pairs_qn2]) if pairs_qn2 else "対象外") + "\n\n"))
    # ワイド
    + (("ワイド（基準" + str(int(S_WIDE_MIN)) + "以上）\n" +
        ("\n".join([f"{a}-{b}（S={round(s,1)}）" for (a,b,s) in pairs_w]) if pairs_w else "対象外") + "\n\n"))
    # 二車単
    + (("二車単（勝率偏差値 合計S1≥" + str(int(S_NITAN_MIN)) + "）\n" +
        ("\n".join([f"{k}（S1={v:.1f}）" for (k,v) in rows_nitan]) if rows_nitan else "対象外") + "\n\n"))
    # 三連単 連動
    + ("三連単（**二車単＋三連複** 連動・S≥164）\n"
       + ("\n".join([f"{k}（参考S={v:.1f}）" for (k,v) in rows_trifecta]) if rows_trifecta else "対象外"))
)

st.markdown("### 📋 note用（コピーエリア）")
st.text_area("ここを選択してコピー", note_text, height=560)


st.markdown("### 📋 note用（コピーエリア）")
st.text_area("ここを選択してコピー", note_text, height=520)

