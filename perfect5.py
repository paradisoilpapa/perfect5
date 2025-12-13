    # -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np   # ← ここ！NumPy を np にする
import unicodedata, re
import math, json, requests
from statistics import mean, pstdev
from itertools import combinations
from datetime import datetime, date, time, timedelta, timezone

# ==============================
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

# --- 最新の印別実測率（2025/09/25版：画像反映済） -----------------
# === ランク別統計データ 最新版 (2025/9/28) ===

# --- 全体 ---
RANK_STATS_TOTAL = {
    "◎": {"p1": 0.261, "pTop2": 0.459, "pTop3": 0.617},
    "〇": {"p1": 0.235, "pTop2": 0.403, "pTop3": 0.533},
    "▲": {"p1": 0.175, "pTop2": 0.331, "pTop3": 0.484},
    "△": {"p1": 0.133, "pTop2": 0.282, "pTop3": 0.434},
    "×": {"p1": 0.109, "pTop2": 0.242, "pTop3": 0.39},
    "α": {"p1": 0.059, "pTop2": 0.167, "pTop3": 0.295},
    "無": {"p1": 0.003, "pTop2": 0.118, "pTop3": 0.256},
}





# KO(勝ち上がり)関連
KO_GIRLS_SCALE = 0.0
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.007   # 0.010 → 0.007
KO_STEP_SIGMA = 0.35   # 0.4 → 0.35


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

def t_score_from_finite(values: np.ndarray, eps: float = 1e-9):
    """NaNを除いた母集団でT=50+10*(x-μ)/σを作り、NaNは50に置換して返す"""
    v = values.astype(float, copy=True)
    finite = np.isfinite(v)
    k = int(finite.sum())
    if k < 2:
        return np.full_like(v, 50.0), (float("nan") if k==0 else float(v[finite][0])), 0.0, k
    mu = float(np.mean(v[finite]))
    sd = float(np.std(v[finite], ddof=0))
    if (not np.isfinite(sd)) or sd < eps:
        return np.full_like(v, 50.0), mu, 0.0, k
    T = 50.0 + 10.0 * ((v - mu) / sd)
    T[~finite] = 50.0
    return T, mu, sd, k

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

# 単騎を全体的に抑える共通係数（あとでいじれるようにする）
SINGLE_NERF = float(globals().get("SINGLE_NERF", 0.85))  # 0.80〜0.88くらいで調整

def pos_coeff(role, line_factor):
    base_map = {
        'head':      1.00,
        'second':    0.72,   # 0.70→0.72に少し上げてライン2番手をちゃんと評価
        'thirdplus': 0.55,
        'single':    0.52,   # 0.90 → 0.52 にドンと落とす
    }
    base = base_map.get(role, 0.52)
    if role == 'single':
        base *= SINGLE_NERF      # ここでさらに細かく落とせる
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
    WIND_ZERO   = float(globals().get("WIND_ZERO", 0.0))
    WIND_SIGN   = float(globals().get("WIND_SIGN", 1.0))
    WIND_GAIN   = float(globals().get("WIND_GAIN", 1.0))  # 33では別処理で0.5倍にしておく想定
    WIND_CAP    = float(globals().get("WIND_CAP", 0.06))
    WIND_MODE   = globals().get("WIND_MODE", "scalar")
    WIND_COEFF  = globals().get("WIND_COEFF", {})
    SPECIAL_DIRECTIONAL_VELODROMES = globals().get("SPECIAL_DIRECTIONAL_VELODROMES", set())
    s_state_track = None
    try:
        s_state_track = st.session_state.get("track", "")
    except Exception:
        pass

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

    if (WIND_MODE == "directional") or (s >= 7.0 and s_state_track in SPECIAL_DIRECTIONAL_VELODROMES):
        wd = WIND_COEFF.get(wind_dir, 0.0)
        dir_term = clamp(s * wd * (0.30 + 0.70*float(prof_escape)) * 0.6, -0.03, 0.03)
        val += dir_term

    val = (val * float(WIND_SIGN)) * float(WIND_GAIN)
    return round(clamp(val, -float(WIND_CAP), float(WIND_CAP)), 3)


# === 直線ラスト200m（残脚）補正｜33バンク対応版 ==============================
# 33（<=340m）は「先行ペナ弱め／差し・追込ボーナス控えめ」へ最適化
L200_ESC_PENALTY = float(globals().get("L200_ESC_PENALTY", -0.06))  # 先行は垂れやすい（基本）
L200_SASHI_BONUS = float(globals().get("L200_SASHI_BONUS", +0.03))  # 差しは伸びやすい
L200_MARK_BONUS  = float(globals().get("L200_MARK_BONUS",  +0.02))  # 追込は少し上げ

L200_GRADE_GAIN  = globals().get("L200_GRADE_GAIN", {
    "F2": 1.18, "F1": 1.10, "G": 1.05, "GIRLS": 0.95, "TOTAL": 1.00
})

# 短走路増幅：旧1.15 → 33はむしろ緩和（0.85）
L200_SHORT_GAIN_33   = float(globals().get("L200_SHORT_GAIN_33", 0.85))
L200_SHORT_GAIN_OTH  = float(globals().get("L200_SHORT_GAIN_OTH", 1.00))
L200_LONG_RELAX      = float(globals().get("L200_LONG_RELAX", 0.90))
L200_CAP             = float(globals().get("L200_CAP", 0.08))
L200_WET_GAIN        = float(globals().get("L200_WET_GAIN", 1.15))

# 33専用 成分別スケーリング
L200_33_ESC_MULT   = float(globals().get("L200_33_ESC_MULT", 0.80))  # 逃ペナ 20%縮小
L200_33_SASHI_MULT = float(globals().get("L200_33_SASHI_MULT", 0.85))# 差し  15%縮小
L200_33_MARK_MULT  = float(globals().get("L200_33_MARK_MULT", 0.90)) # 追込  10%縮小

def _grade_key_from_class(race_class: str) -> str:
    if "ガール" in race_class: return "GIRLS"
    if "Ｓ級" in race_class or "S級" in race_class: return "G"
    if "チャレンジ" in race_class: return "F2"
    if "Ａ級" in race_class or "A級" in race_class: return "F1"
    return "TOTAL"

def l200_adjust(role: str,
                straight_length: float,
                bank_length: float,
                race_class: str,
                prof_escape: float,    # 逃
                prof_sashi: float,     # 差
                prof_oikomi: float,    # マ
                is_wet: bool = False) -> float:
    """
    ラスト200mの“残脚”を脚質×バンク×グレードで調整した無次元値（±）を返す。
    ※ ENV合計（total_raw）には足さず、独立柱として z 化→anchor_score へ。
    """
    esc_term   = L200_ESC_PENALTY * float(prof_escape)
    sashi_term = L200_SASHI_BONUS * float(prof_sashi)
    mark_term  = L200_MARK_BONUS  * float(prof_oikomi)

    is_33 = float(bank_length) <= 340.0
    if is_33:
        esc_term   *= L200_33_ESC_MULT
        sashi_term *= L200_33_SASHI_MULT
        mark_term  *= L200_33_MARK_MULT

    base = esc_term + sashi_term + mark_term

    if is_33:
        base *= L200_SHORT_GAIN_33
    else:
        base *= L200_SHORT_GAIN_OTH

    if float(straight_length) >= 60.0:
        base *= L200_LONG_RELAX

    base *= float(L200_GRADE_GAIN.get(_grade_key_from_class(race_class), 1.0))

    if is_wet:
        base *= L200_WET_GAIN

    pos_factor = {'head':1.00,'second':0.85,'thirdplus':0.70,'single':0.80}.get(role, 0.80)
    base *= pos_factor

    return round(clamp(base, -float(L200_CAP), float(L200_CAP)), 3)


def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi):
    straight_factor = (float(straight_length)-40.0)/10.0
    angle_factor = (float(bank_angle)-25.0)/5.0
    total = clamp(-0.1*straight_factor + 0.1*angle_factor, -0.05, 0.05)
    return round(total*prof_escape - 0.5*total*prof_sashi, 3)

def bank_length_adjust(bank_length, prof_oikomi):
    delta = clamp((float(bank_length)-411.0)/100.0, -0.05, 0.05)
    return round(delta*prof_oikomi, 3)

# --- ラインSBボーナス（33mは自動で半減） --------------------
def compute_lineSB_bonus(line_def, S, B, line_factor=1.0, exclude=None, cap=0.06, enable=True):
    """
    33m系（<=340）では自動で効きを半減:
      - LINE_SB_33_MULT（既定0.5）を line_factor に乗算
      - LINE_SB_CAP_33_MULT（既定0.5）を cap に乗算
    """
    if not enable or not line_def:
        return ({g: 0.0 for g in line_def.keys()} if line_def else {}), {}

    # 33かどうかの自動推定
    try:
        bank_len = st.session_state.get("bank_length", st.session_state.get("track_length", None))
    except Exception:
        bank_len = globals().get("BANK_LENGTH", None)

    eff_line_factor = float(line_factor)
    eff_cap = float(cap)

    if bank_len is not None:
        try:
            if float(bank_len) <= 340.0:
                mult = float(globals().get("LINE_SB_33_MULT", 0.50))
                capm = float(globals().get("LINE_SB_CAP_33_MULT", 0.50))
                eff_line_factor *= mult
                eff_cap *= capm
        except Exception:
            pass

    # ライン内の位置重み（単騎を下げる）
    w_pos_base = {
        "head":      1.00,
        "second":    0.55,
        "thirdplus": 0.38,
        "single":    0.34,
    }

    # ラインごとのS/B集計
    Sg = {}
    Bg = {}
    for g, mem in line_def.items():
        s = 0.0
        b = 0.0
        for car in mem:
            if exclude is not None and car == exclude:
                continue
            role = role_in_line(car, line_def)
            w = w_pos_base[role] * eff_line_factor
            s += w * float(S.get(car, 0))
            b += w * float(B.get(car, 0))
        Sg[g] = s
        Bg[g] = b

    # ラインごとの“強さ”スコア
    raw = {}
    for g in line_def.keys():
        s = Sg[g]
        b = Bg[g]
        ratioS = s / (s + b + 1e-6)
        raw[g] = (0.6 * b + 0.4 * s) * (0.6 + 0.4 * ratioS)

    # z化してボーナス化
    zz = zscore_list(list(raw.values())) if raw else []
    bonus = {}
    for i, g in enumerate(raw.keys()):
        bonus[g] = clamp(0.02 * float(zz[i]), -eff_cap, eff_cap)

    return bonus, raw


# ==============================
# KO Utilities（ここから下を1かたまりで）
# ==============================

def _role_of(car, mem):
    """ラインの中での役割を返す（head / second / thirdplus / single）"""
    if len(mem) == 1:
        return "single"
    idx = mem.index(car)
    return ["head", "second", "thirdplus"][idx] if idx < 3 else "thirdplus"


# KOでも、ライン強度でも、同じ位置重みを使う
LINE_W_POS = {
    "head":      1.00,
    "second":    0.55,
    "thirdplus": 0.38,
    "single":    0.34,
}


def _line_strength_raw(line_def, S, B, line_factor: float = 1.0) -> dict:
    """
    KOやトップ2ライン抽出で使う“生のライン強度”
    compute_lineSB_bonus と式をそろえてある
    """
    if not line_def:
        return {}

    w_pos = {k: v * float(line_factor) for k, v in LINE_W_POS.items()}

    raw: dict[str, float] = {}
    for g, mem in line_def.items():
        s = 0.0
        b = 0.0
        for c in mem:
            role = _role_of(c, mem)
            w = w_pos.get(role, 0.34)
            s += w * float(S.get(c, 0))
            b += w * float(B.get(c, 0))
        ratioS = s / (s + b + 1e-6)
        raw[g] = (0.6 * b + 0.4 * s) * (0.6 + 0.4 * ratioS)
    return raw


def _top2_lines(line_def, S, B, line_factor=1.0):
    """ラインの中から強い2本を取る"""
    raw = _line_strength_raw(line_def, S, B, line_factor)
    order = sorted(raw.keys(), key=lambda g: raw[g], reverse=True)
    return (order[0], order[1]) if len(order) >= 2 else (order[0], None) if order else (None, None)


def _extract_role_car(line_def, gid, role_name):
    """指定ラインのhead/secondを抜く"""
    if gid is None or gid not in line_def:
        return None
    mem = line_def[gid]
    if role_name == "head":
        return mem[0] if len(mem) >= 1 else None
    if role_name == "second":
        return mem[1] if len(mem) >= 2 else None
    return None


def _ko_order(v_base_map,
              line_def,
              S,
              B,
              line_factor: float = 1.0,
              gap_delta: float = 0.007):
    """
    KO用の並びを作る
    1) 上2ラインのhead
    2) 上2ラインのsecond
    3) 残りのラインの残りをスコア順
    4) その他の車番
    同じライン内でスコア差が gap_delta 以内なら寄せる
    """
    cars = list(v_base_map.keys())

    # ラインが無いときはふつうにスコア順
    if not line_def or len(line_def) < 1:
        return [c for c, _ in sorted(v_base_map.items(), key=lambda x: x[1], reverse=True)]

    g1, g2 = _top2_lines(line_def, S, B, line_factor)

    head1 = _extract_role_car(line_def, g1, "head")
    head2 = _extract_role_car(line_def, g2, "head")
    sec1  = _extract_role_car(line_def, g1, "second")
    sec2  = _extract_role_car(line_def, g2, "second")

    others: list[int] = []
    if g1:
        mem = line_def[g1]
        if len(mem) >= 3:
            others += mem[2:]
    if g2:
        mem = line_def[g2]
        if len(mem) >= 3:
            others += mem[2:]
    for g, mem in line_def.items():
        if g not in {g1, g2}:
            others += mem

    order: list[int] = []

    # 1) headをスコア順で
    head_pair = [x for x in [head1, head2] if x is not None]
    order += sorted(head_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    # 2) secondをスコア順で
    sec_pair = [x for x in [sec1, sec2] if x is not None]
    order += sorted(sec_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    # 3) 残りラインの残り（重複を落とす）
    others = list(dict.fromkeys([c for c in others if c is not None]))
    others_sorted = sorted(others, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    order += [c for c in others_sorted if c not in order]

    # 4) まだ出てない車を最後に
    for c in cars:
        if c not in order:
            order.append(c)

    # ライン内の小差詰め
    def _same_group(a, b):
        if a is None or b is None:
            return False
        ga = next((g for g, mem in line_def.items() if a in mem), None)
        gb = next((g for g, mem in line_def.items() if b in mem), None)
        return ga is not None and ga == gb

    i = 0
    while i < len(order) - 2:
        a, b, c = order[i], order[i + 1], order[i + 2]
        if _same_group(a, b):
            vx = v_base_map.get(b, 0.0) - v_base_map.get(c, 0.0)
            if vx >= -gap_delta:
                order.pop(i + 2)
                order.insert(i + 1, b)
        i += 1

    return order


def _zone_from_p(p: float):
    needed = 1.0 / max(p, 1e-12)
    return needed, needed * (1.0 + E_MIN), needed * (1.0 + E_MAX)


def apply_anchor_line_bonus(score_raw: dict[int, float],
                            line_of: dict[int, int],
                            role_map: dict[int, str],
                            anchor: int,
                            tenkai: str) -> dict[int, float]:
    a_line = line_of.get(anchor, None)
    is_on = (tenkai in LINE_BONUS_ON_TENKAI) and (a_line is not None)
    score_adj: dict[int, float] = {}
    for i, s in score_raw.items():
        bonus = 0.0
        if is_on and line_of.get(i) == a_line and i != anchor:
            role = role_map.get(i, "single")
            bonus = min(max(0.0, LINE_BONUS.get(role, 0.0)), LINE_BONUS_CAP)
        score_adj[i] = s + bonus
    return score_adj


def format_rank_all(score_map: dict[int, float], P_floor_val: float | None = None) -> str:
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
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 60.0, float(wind_speed_default), 0.1)

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

# === PATCH-L200: 直線ラスト200mの残脚補正 =========================
# 目的: 逃げ先行が直線で苦しくなる場面を少しだけ減点、差し・マークは微加点。
# 強さはミッドナイト/短走路で少しだけ強めに。

L200_ESC_PENALTY   = -0.06   # 逃げ(先行)の基礎マイナス
L200_SASHI_BONUS   = +0.03   # 差しの基礎プラス
L200_MARK_BONUS    = +0.02   # マーク(追込)の基礎プラス
L200_MNIGHT_GAIN   = 1.20    # ミッドナイトの倍率
L200_SHORT_GAIN    = 1.15    # 333mなど短走路の倍率
L200_LONG_RELAX    = 0.90    # 直線長めはやや緩和
L200_CAP           = 0.08    # 絶対値キャップ（安全弁）

def last200_bonus(no: int, role: str) -> float:
    """脚質×バンク条件からラスト200mの微調整を返す（±0.08程度）。"""
    esc   = float(prof_escape.get(no, 0.0))
    sashi = float(prof_sashi.get(no, 0.0))
    mark  = float(prof_oikomi.get(no, 0.0))

    # 基礎：脚質ミックス
    base = (L200_ESC_PENALTY * esc) + (L200_SASHI_BONUS * sashi) + (L200_MARK_BONUS * mark)

    # トラック条件
    gain = 1.0
    if race_time == "ミッドナイト":
        gain *= L200_MNIGHT_GAIN
    if float(bank_length) <= 360.0:
        gain *= L200_SHORT_GAIN
    if float(straight_length) >= 58.0:
        gain *= L200_LONG_RELAX

    # 位置（先頭＝重め、後ろ薄め）
    pos_w = {'head': 1.00, 'second': 0.70, 'thirdplus': 0.55, 'single': 0.80}.get(role, 0.80)

    val = base * gain * pos_w
    # 会場バイアス（style>0=先行寄り→減点を少し緩める）
    val *= (0.95 if style > 0 else 1.05)

    return round(max(-L200_CAP, min(L200_CAP, val)), 3)
# === PATCH-L200: ここまで ==========================================

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

# ライン構成（最大7：単騎も1ライン）
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

# ←←← ここに入れる
import re, unicodedata
def input_float_text(label: str, key: str, placeholder: str = "") -> float | None:
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "":
        return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)
# →→→ ここまで

st.subheader("個人データ（直近4か月：回数）")
cols = st.columns(n_cars)
ratings, S, B = {}, {}, {}
...

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

# ここは従来通りでOK
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

# ←ここはFormだけ作る（偏差値化はまだしない）
Form = {no: 0.7*p1_eff[no] + 0.3*p2_eff[no] for no in active_cars}

# === Form 偏差値化（平均50, SD10）
form_list = [Form[n] for n in active_cars]
form_T, mu_form, sd_form, _ = t_score_from_finite(np.array(form_list))
form_T_map = {n: float(form_T[i]) for i, n in enumerate(active_cars)}



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
L200_RAW = {}  # ← 新規

for no in active_cars:
    role = role_in_line(no, line_def)

    # --- L200（残脚）生値を計算：ENV合計には“入れない”観測用 ---
    l200 = l200_adjust(
        role=role,
        straight_length=straight_length,
        bank_length=bank_length,
        race_class=race_class,
        prof_escape=float(prof_escape[no]),
        prof_sashi=float(prof_sashi[no]),
        prof_oikomi=float(prof_oikomi[no]),
        is_wet=st.session_state.get("is_wet", False)  # 雨トグル未実装なら False のまま
    )
    L200_RAW[int(no)] = float(l200)

    # --- 周回疲労（既存） ---
    extra = fatigue_extra(eff_laps, day_label, n_cars, race_class)
    fatigue_scale = (1.0 if race_class == "Ｓ級" else
                     1.1 if race_class == "Ａ級" else
                     1.2 if race_class == "Ａ級チャレンジ" else
                     1.05)
    laps_adj = (
        -0.10 * extra * (1.0 if prof_escape[no] > 0.5 else 0.0)
        + 0.05 * extra * (1.0 if prof_oikomi[no] > 0.4 else 0.0)
    ) * fatigue_scale

rows = []
_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir", wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

for no in active_cars:
    role = role_in_line(no, line_def)
    # ここに各種計算と rows.append(...) が続く


    # 周回疲労（DAY×頭数×級別を反映）
    extra = fatigue_extra(eff_laps, day_label, n_cars, race_class)
    fatigue_scale = (
        1.0  if race_class == "Ｓ級" else
        1.1  if race_class == "Ａ級" else
        1.2  if race_class == "Ａ級チャレンジ" else
        1.05
    )
    laps_adj = (
        -0.10 * extra * (1.0 if prof_escape[no] > 0.5 else 0.0)
        + 0.05 * extra * (1.0 if prof_oikomi[no] > 0.4 else 0.0)
    ) * fatigue_scale

    # 環境・個人補正（既存）
    wind     = _wind_func(eff_wind_dir, float(eff_wind_speed or 0.0), role, float(prof_escape[no]))
    bank_b   = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no])
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv    = extra_bonus.get(no, 0.0)
    stab     = stability_score(no)  # 安定度

    # ★ ラスト200（必要なら last200_bonus を l200_adjust に変更）
    l200 = l200_adjust(role, straight_length, bank_length, race_class,
                   float(prof_escape[no]), float(prof_sashi[no]), float(prof_oikomi[no]),
                   is_wet=st.session_state.get("is_wet", False))


    # ★ 合計（SBなし）…ここでは l200 も加算する版
    total_raw = (
        prof_base[no] +
        wind +
        cf["spread"] * tens_corr.get(no, 0.0) +
        bank_b + length_b +
        laps_adj + indiv + stab +
        l200
    )

    rows.append([
        int(no), role,
        round(prof_base[no], 3),
        round(wind, 3),
        round(cf["spread"] * tens_corr.get(no, 0.0), 3),
        round(bank_b, 3),
        round(length_b, 3),
        round(laps_adj, 3),
        round(indiv, 3),
        round(stab, 3),
        round(l200, 3),
        total_raw
    ])

df = pd.DataFrame(rows, columns=[
    "車番","役割","脚質基準(会場)","風補正","得点補正","バンク補正",
    "周長補正","周回補正","個人補正","安定度","ラスト200","合計_SBなし_raw",
])

# === ここは df = pd.DataFrame(...) の直後に貼るだけ ===

# --- fallback: note_sections が無い環境でも落ちないように ---
ns = globals().get("note_sections", None)
if not isinstance(ns, list):
    ns = []
    globals()["note_sections"] = ns
note_sections = ns


# ❶ バンク分類を“みなし直線/周長”から決定（33 / 400 / 500）
def _bank_str_from_lengths(bank_length: float) -> str:
    try:
        bl = float(bank_length)
    except:
        bl = 400.0
    if bl <= 340.0:   # 333系
        return "33"
    elif bl >= 480.0: # 500系
        return "500"
    return "400"

# ❷ 会場の“有利脚質”セット
def _favorable_styles(bank_str: str) -> set[str]:
    if bank_str == "33":   # 33＝先行系・ライン寄り
        return {"逃げ", "マーク"}
    if bank_str == "500":  # 500＝差し・マーク寄り
        return {"差し", "マーク"}
    return {"まくり", "差し"}  # 既定=400

# ❸ 役割の日本語化（lineの並びから）
def _role_jp(no: int, line_def: dict) -> str:
    r = role_in_line(no, line_def)  # 'head'/'second'/'thirdplus'/'single'
    return {"head":"先頭","second":"番手","thirdplus":"三番手","single":"先頭"}.get(r, "先頭")

# ❹ 入力の“逃/捲/差/マ”から、その選手の実脚質を決定（同点時はライン位置でブレない決め方）
def _dominant_style(no: int) -> str:
    vec = [("逃げ", k_esc.get(no,0)), ("まくり", k_mak.get(no,0)),
           ("差し", k_sashi.get(no,0)), ("マーク", k_mark.get(no,0))]
    m = max(v for _,v in vec)
    cand = [s for s,v in vec if v == m and m > 0]
    if cand:
        # タイブレーク：先頭>番手>三番手>単騎 を優先（先行気味→差し→マークの順）
        pr = {"先頭":3,"番手":2,"三番手":1,"単騎":0}
        role = role_in_line(no, line_def)
        role_pr = {"head":"先頭","second":"番手","thirdplus":"三番手","single":"単騎"}.get(role,"単騎")
        if "逃げ" in cand: return "逃げ"
        # 残りはライン位置で“差し”優先、その次に“マーク”
        if "差し" in cand and pr.get(role_pr,0) >= 2: return "差し"
        if "マーク" in cand: return "マーク"
        return cand[0]
    # 出走履歴ゼロなら位置で決める
    role = role_in_line(no, line_def)
    return {"head":"逃げ","second":"差し","thirdplus":"マーク","single":"まくり"}.get(role,"まくり")

# ❺ Rider 構造体（このファイル上部で既に宣言済みなら再定義不要）
from dataclasses import dataclass
@dataclass
class Rider:
    num: int; hensa: float; line_id: int; role: str; style: str

# ❻ 偏差値（Tスコア）を “合計_SBなし_raw” から作る（なければ Form で代用）
# ❻ 安定版：偏差値（Tスコア）を安全に作る
def _hensa_map_from_df(df: pd.DataFrame) -> dict[int,float]:
    col = "合計_SBなし_raw" if "合計_SBなし_raw" in df.columns else None

    # 生値ベクトルを取る（欠損があればフォールバックして補完）
    base = []
    for no in active_cars:
        try:
            v = float(df.loc[df["車番"]==no, col].values[0]) if col else float(form_T_map[no])
        except:
            v = float(form_T_map[no])  # fallback（=従来 Form 偏差値）
        base.append(v)

    base = np.array(base, dtype=float)

    # === 分散チェック：標準偏差が小さすぎる場合の暴走回避 ===
    sd = np.std(base)
    if sd < 1e-6:   # ← 安定化の本丸
        # 全員ほぼ同じ → 差が「無い」ので偏差値の差も付けない
        return {no: 50.0 for no in active_cars}

    # 通常の偏差値化
    T = 50 + 10 * (base - np.mean(base)) / sd

    # 浮動誤差対策で丸め
    T = np.clip(T, 20, 80)

    return {no: float(T[i]) for i,no in enumerate(active_cars)}


# ❼ RIDERS を“実データ”で構築（脚質は ❹、偏差値は ❻）
bank_str = _bank_str_from_lengths(bank_length)
hensa_map = _hensa_map_from_df(df)
RIDERS = []
for no in active_cars:
    # ラインIDは“そのラインの先頭車番”を代表IDに
    gid = None
    for g, mem in line_def.items():
        if no in mem:
            gid = mem[0]; break
    if gid is None: gid = no
    RIDERS.append(
        Rider(
            num=int(no),
            hensa=float(hensa_map[no]),
            line_id=int(gid),
            role=_role_jp(no, line_def),
            style=_dominant_style(no),
        )
    )

# ❽ フォーメーション（本命−2−全）：1列目=有利脚質内の偏差値最大
def _pick_axis(riders: list[Rider], bank_str: str) -> Rider:
    fav = _favorable_styles(bank_str)
    cand = [r for r in riders if r.style in fav]
    if not cand:
        raise ValueError(f"有利脚質{sorted(fav)}に該当0（bank={bank_str} / style誤りの可能性）")
    return max(cand, key=lambda r: r.hensa)

def _role_priority(bank_str: str) -> dict[str,int]:
    return ({"マーク":3,"番手":2,"三番手":1,"先頭":0} if bank_str=="33"
            else {"番手":3,"マーク":2,"三番手":1,"先頭":0})

def _pick_support(riders: list[Rider], first: Rider, bank_str: str) -> Rider|None:
    pr = _role_priority(bank_str)
    same = [r for r in riders if r.line_id==first.line_id and r.num!=first.num]
    if not same: return None
    same.sort(key=lambda r: (pr.get(r.role,0), r.hensa), reverse=True)
    return same[0]

# 印（◎→▲→偏差値補完）
def _read_marks_idmap() -> dict[int,str]:
    rm = globals().get("result_marks") or globals().get("marks") or {}
    out={}
    if isinstance(rm, dict):
        if any(isinstance(k,int) or (isinstance(k,str) and k.isdigit()) for k in rm.keys()):
            for k,v in rm.items():
                try: out[int(k)] = ("○" if str(v) in ("○","〇") else str(v))
                except: pass
        else:
            for sym,vid in rm.items():
                try: out[int(vid)] = ("○" if str(sym) in ("○","〇") else str(sym))
                except: pass
    return out

def _pick_partner(riders: list[Rider], used: set[int]) -> int|None:
    id2sym = _read_marks_idmap()
    for want in ("◎","▲"):
        t = next((i for i,s in id2sym.items() if i not in used and s==want), None)
        if t is not None: return t
    # 補完：偏差値上位
    rest = sorted([r for r in riders if r.num not in used], key=lambda r: r.hensa, reverse=True)
    return rest[0].num if rest else None

def make_trio_formation_final(riders: list[Rider], bank_str: str) -> str:
    first = _pick_axis(riders, bank_str)
    support = _pick_support(riders, first, bank_str)
    used = {first.num} | ({support.num} if support else set())
    partner = _pick_partner(riders, used)
    second = []
    if support: second.append(support.num)
    if partner is not None: second.append(partner)
    if len(second) < 2:
        # 2車に満たなければ偏差値補完
        rest = sorted([r.num for r in riders if r.num not in ({first.num}|set(second))],
                      key=lambda n: next(rr.hensa for rr in riders if rr.num==n),
                      reverse=True)
        if rest: second.append(rest[0])
    second = sorted(set(second))[:2]
    return f"三連複フォーメーション：{first.num}－{','.join(map(str, second))}－全"

# ❾ 出力（note_sections があればそこへ）
try:
    out = make_trio_formation_final(RIDERS, bank_str)
    (note_sections.append if isinstance(note_sections, list) else print)(f"【狙いたいレースフォーメーション】 {out}")
except Exception as e:
    (note_sections.append if isinstance(note_sections, list) else print)(f"【狙いたいレースフォーメーション】 エラー: {e}")


mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0 * (df["合計_SBなし_raw"] - mu)

# === [PATCH-A] 安定度をENVから分離し、各柱をレース内z化（SD固定） ===
SD_FORM = 0.28
SD_ENV  = 0.20
SD_STAB = 0.12
SD_L200 = float(globals().get("SD_L200", 0.22))  # ← 追加。まず0.22〜0.30で様子見


# 安定度（raw）と、ENVのベース（= 合計_SBなし_raw から安定度だけ除いたもの）
STAB_RAW = {int(df.loc[i, "車番"]): float(df.loc[i, "安定度"]) for i in df.index}
ENV_BASE = {
    int(df.loc[i, "車番"]): float(df.loc[i, "合計_SBなし_raw"]) - float(df.loc[i, "安定度"])
    for i in df.index
}

# ENV → z
_env_arr = np.array([float(ENV_BASE.get(n, np.nan)) for n in active_cars], dtype=float)
_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask])); sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0
_den_env = (sd_env if sd_env > 1e-12 else 1.0)
ENV_Z = {int(n): (float(ENV_BASE.get(n, mu_env)) - mu_env) / _den_env for n in active_cars}

# FORM（すでに form_T_map は作ってある前提） → z
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}

# STAB（安定度 raw） → z
_stab_arr = np.array([float(STAB_RAW.get(n, np.nan)) for n in active_cars], dtype=float)
_m2 = np.isfinite(_stab_arr)
if int(_m2.sum()) >= 2:
    mu_st = float(np.mean(_stab_arr[_m2])); sd_st = float(np.std(_stab_arr[_m2]))
else:
    mu_st, sd_st = 0.0, 1.0
_den_st = (sd_st if sd_st > 1e-12 else 1.0)
STAB_Z = {int(n): (float(STAB_RAW.get(n, mu_st)) - mu_st) / _den_st for n in active_cars}

# L200（残脚）→ z
_l200_arr = np.array([float(L200_RAW.get(n, np.nan)) for n in active_cars], dtype=float)
_m3 = np.isfinite(_l200_arr)
if int(_m3.sum()) >= 2:
    mu_l2 = float(np.mean(_l200_arr[_m3])); sd_l2 = float(np.std(_l200_arr[_m3]))
else:
    mu_l2, sd_l2 = 0.0, 1.0
_den_l2 = (sd_l2 if sd_l2 > 1e-12 else 1.0)
L200_Z = {int(n): (float(L200_RAW.get(n, mu_l2)) - mu_l2) / _den_l2 for n in active_cars}


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

# ===== 印用（既存の安全弁を維持） =====
FINISH_WEIGHT   = globals().get("FINISH_WEIGHT", 6.0)
FINISH_WEIGHT_G = globals().get("FINISH_WEIGHT_G", 3.0)
POS_BONUS  = globals().get("POS_BONUS", {0: 0.0, 1: -0.6, 2: -0.9, 3: -1.2, 4: -1.4})
POS_WEIGHT = globals().get("POS_WEIGHT", 1.0)
SMALL_Z_RATING = globals().get("SMALL_Z_RATING", 0.01)
FINISH_CLIP = globals().get("FINISH_CLIP", 4.0)
TIE_EPSILON  = globals().get("TIE_EPSILON", 0.8)

# --- p2のZ化など（従来どおり） ---
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

# === ★Form 偏差値化（anchor_scoreより前に必ず置く！） ===
# すでに上で Form = 0.7*p1_eff + 0.3*p2_eff を作ってある前提
# t_score_from_finite はこのファイル内に定義済みである前提
form_list = [Form[n] for n in active_cars]
form_T, mu_form, sd_form, _ = t_score_from_finite(np.array(form_list))
form_T_map = {n: float(form_T[i]) for i, n in enumerate(active_cars)}

# === [PATCH-1] ENV/FORM をレース内で z 化し、目標SDを掛ける（anchor_score の前に置く） ===
SD_FORM = 0.28   # Balanced 既定
SD_ENV  = 0.20

# ENV = v_final（風・会場・周回疲労・個人補正・安定度 等を含む“Form以外”）
_env_arr = np.array([float(v_final.get(n, np.nan)) for n in active_cars], dtype=float)
_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask])); sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0
_den = (sd_env if sd_env > 1e-12 else 1.0)
ENV_Z = {int(n): (float(v_final.get(n, mu_env)) - mu_env) / _den for n in active_cars}

# FORM = form_T_map（T=50, SD=10）→ z 化
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}


def _pos_idx(no:int) -> int:
    g = car_to_group.get(no, None)
    if g is None or g not in line_def:
        return 0
    grp = line_def[g]
    try:
        return max(0, int(grp.index(no)))
    except Exception:
        return 0

bonus_init,_ = compute_lineSB_bonus(
    line_def, S, B,
    line_factor=line_factor_eff,
    exclude=None, cap=cap_SB_eff,
    enable=line_sb_enable
)



def anchor_score(no: int) -> float:
    role = role_in_line(no, line_def)
    sb = float(bonus_init.get(car_to_group.get(no, None), 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0))
    pos_term  = POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)
    env_term  = SD_ENV  * float(ENV_Z.get(int(no), 0.0))
    form_term = SD_FORM * float(FORM_Z.get(int(no), 0.0))
    stab_term = SD_STAB * float(STAB_Z.get(int(no), 0.0))
    l200_term = SD_L200 * float(L200_Z.get(int(no), 0.0))   # ← 追加
    tiny      = SMALL_Z_RATING * float(zt_map.get(int(no), 0.0))
    return env_term + form_term + stab_term + l200_term + sb + pos_term + tiny



# === デバッグ表示（必要なときだけ / anchor_score定義の後, 印出力の前） ===
# for no in active_cars:
#     role = role_in_line(no, line_def)
#     sb_dbg  = bonus_init.get(car_to_group.get(no, None), 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
#     pos_dbg = POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)
#     form_dbg = SD_FORM * FORM_Z.get(no, 0.0)
#     env_dbg  = SD_ENV  * ENV_Z.get(no, 0.0)
#     stab_dbg = (SD_STAB * STAB_Z.get(no, 0.0)) if 'STAB_Z' in globals() else 0.0
#     tiny_dbg = SMALL_Z_RATING * zt_map.get(no, 0.0)

#     total = form_dbg + env_dbg + stab_dbg + sb_dbg + pos_dbg + tiny_dbg
#     st.write(no, {
#         "form": round(form_dbg, 4),
#         "env":  round(env_dbg, 4),
#         "stab": round(stab_dbg, 4),
#         "sb":   round(sb_dbg, 4),
#         "pos":  round(pos_dbg, 4),
#         "tiny": round(tiny_dbg, 4),
#         "TOTAL(anchor_score期待値)": round(total, 4),
#     })



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
    mu = float(np.mean(v[finite]))
    sd = float(np.std(v[finite], ddof=0))
    if (not np.isfinite(sd)) or sd < eps:
        return np.full_like(v, 50.0), mu, 0.0, k
    T = 50.0 + 10.0 * ((v - mu) / sd)
    T[~finite] = 50.0
    return T, mu, sd, k


# ★Form の偏差値化（t_score_from_finite 定義の直後）
form_list = [Form[n] for n in active_cars]
form_T, mu_form, sd_form, _ = t_score_from_finite(np.array(form_list))
form_T_map = {n: float(form_T[i]) for i, n in enumerate(active_cars)}






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

# ★強制：偏差値の母集団を anchor_score に統一（ここが命）
SB_BASE_MAP = {int(i): float(anchor_score(int(i))) for i in USED_IDS}


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

# === 5.5) クラス別ライン偏差値ボーナス（ライン間→ライン内：低T優先 3:2:1） ===
# クラス別の総ポイント（Girlsは無効）
CLASS_LINE_POOL = {
    "Ｓ級":           21.0,
    "Ａ級":           15.0,
    "Ａ級チャレンジ":  9.0,
    "ガールズ":        0.0,
}
pool_total = float(CLASS_LINE_POOL.get(race_class, 0.0))

def _line_rank_weights(n_lines: int) -> list[float]:
    # 2本: 3:2 / 3本: 5:4:3 / 4本以上: 6,5,4,3,2,1...
    if n_lines <= 1: return [1.0]
    if n_lines == 2: return [3.0, 2.0]
    if n_lines == 3: return [5.0, 4.0, 3.0]
    base = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    if n_lines <= len(base): return base[:n_lines]
    ext = base[:]
    while len(ext) < n_lines:
        ext.append(max(1.0, ext[-1]-1.0))
    return ext[:n_lines]

def _in_line_weights(members_sorted_lowT_first: list[int]) -> dict[int, float]:
    # ライン内は「低T優先で 3:2:1、4人目以降0」→合計1に正規化
    raw = [3.0, 2.0, 1.0]
    w = {}
    for i, car in enumerate(members_sorted_lowT_first):
        w[int(car)] = (raw[i] if i < len(raw) else 0.0)
    s = sum(w.values())
    return {k: (v/s if s > 0 else 0.0) for k, v in w.items()}

_lines = list((globals().get("line_def") or {}).values())
if pool_total > 0.0 and _lines:
    # ライン強度＝そのラインの race_t 平均
    line_scores = []
    for mem in _lines:
        if not mem: 
            continue
        avg_t = float(np.mean([race_t.get(int(c), 50.0) for c in mem]))
        line_scores.append((tuple(mem), avg_t))
    # 強い順に並べてライン間ポイント配分
    line_scores.sort(key=lambda x: (-x[1], x[0]))
    rank_w = _line_rank_weights(len(line_scores))
    sum_rank_w = float(sum(rank_w)) if rank_w else 1.0
    line_share = {}
    for (mem, _avg), wr in zip(line_scores, rank_w):
        line_share[mem] = pool_total * (float(wr) / sum_rank_w)

    # 各ラインの配分を「低T→高T」の順に 3:2:1 で割り振り
    bonus_map = {int(i): 0.0 for i in USED_IDS}
    for mem, share in line_share.items():
        mem = list(mem)
        mem_sorted_lowT = sorted(mem, key=lambda c: (race_t.get(int(c), 50.0), int(c)))
        w_in = _in_line_weights(mem_sorted_lowT)  # 合計1
        for car in mem_sorted_lowT:
            bonus_map[int(car)] += share * w_in[int(car)]

    # 偏差値に加算（xs_race_tが計算本体。race_tは表示用に丸め直す）
    for idx, car in enumerate(USED_IDS):
        add = float(bonus_map.get(int(car), 0.0))
        xs_race_t[idx] = float(xs_race_t[idx]) + add
        race_t[int(car)] = float(round(xs_race_t[idx], HEN_DEC_PLACES))
# ← この後に既存の race_z 計算が続く



race_z = (xs_race_t - 50.0) / 10.0

hen_df = pd.DataFrame({
    "車": USED_IDS,
    "SBなし(母集団)": [None if not np.isfinite(x) else float(x) for x in xs_base_raw],
    "偏差値T(レース内)": [race_t[i] for i in USED_IDS],
}).sort_values(["偏差値T(レース内)","車"], ascending=[False, True]).reset_index(drop=True)

st.markdown("### 偏差値（レース内T＝平均50・SD10｜SBなしと同一母集団）")
st.caption(f"μ={mu_sb if np.isfinite(mu_sb) else 'nan'} / σ={sd_sb:.6f} / 有効件数k={k_finite}")
st.dataframe(hen_df, use_container_width=True)


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


# ===== 印の採番（β廃止→無印で保持）========================================
# 依存: USED_IDS, race_t, xs_base_raw, line_def, car_to_group が上で定義済み

# スコアの補助（安定のため race_t 優先→同点は sb_base でタイブレーク）
sb_base = {
    int(USED_IDS[idx]): float(xs_base_raw[idx]) if np.isfinite(xs_base_raw[idx]) else float("-inf")
    for idx in range(len(USED_IDS))
}

def _race_t_val(i: int) -> float:
    try:
        return float(race_t.get(int(i), 50.0))
    except Exception:
        return 50.0

# === βは作らない。全員を候補にして上位から印を振る
seed_pool = list(map(int, USED_IDS))
order_by_T = sorted(
    seed_pool,
    key=lambda i: (-_race_t_val(i), -sb_base.get(i, float("-inf")), i)
)

result_marks: dict[str,int] = {}
reasons: dict[int,str] = {}

# ◎〇▲ を上位から
for mk, car in zip(["◎","〇","▲"], order_by_T):
    result_marks[mk] = int(car)

# ◎の同ラインを優先して残り印（△, ×, α）を埋める
line_def     = globals().get("line_def", {}) or {}
car_to_group = globals().get("car_to_group", {}) or {}
anchor_no    = result_marks.get("◎", None)

mates_sorted: list[int] = []
if anchor_no is not None:
    a_gid = car_to_group.get(anchor_no, None)
    if a_gid is not None and a_gid in line_def:
        used_now = set(result_marks.values())
        mates_sorted = sorted(
            [int(c) for c in line_def[a_gid] if int(c) not in used_now],
            key=lambda x: (-sb_base.get(int(x), float("-inf")), int(x))
        )

used = set(result_marks.values())
overall_rest = [int(c) for c in USED_IDS if int(c) not in used]
overall_rest = sorted(
    overall_rest,
    key=lambda x: (-sb_base.get(int(x), float("-inf")), int(x))
)

# 同ライン優先 → 残りスコア順
tail_priority = mates_sorted + [c for c in overall_rest if c not in mates_sorted]

for mk in ["△","×","α"]:
    if mk in result_marks:
        continue
    if not tail_priority:
        break
    no = int(tail_priority.pop(0))
    result_marks[mk] = no
    reasons[no] = f"{mk}（◎ライン優先→残りスコア順）"

# === 無印の集合（＝上の印が付かなかった残り全員）
marked_ids = set(result_marks.values())
no_mark_ids = [int(c) for c in USED_IDS if int(c) not in marked_ids]
# 表示はT優先・同点はsb_base
no_mark_ids = sorted(
    no_mark_ids,
    key=lambda x: (-_race_t_val(int(x)), -sb_base.get(int(x), float("-inf")), int(x))
)

# ===== 以降のUI出力での使い方 ==============================================
# ・印の一行（note用）: 既存の join を差し替え
#   例）(' '.join(f'{m}{result_marks[m]}' for m in ['◎','〇','▲','△','×','α'] if m in result_marks))
#   の直後などに「無」を追加
#   例）
#   ('無　' + (' '.join(map(str, no_mark_ids)) if no_mark_ids else '—'))
#
# ・以降のロジックでは「β」への参照を残さないこと（Noneチェック含め全削除OK）
#   もし `if i != result_marks.get("β")` のような行が残っていたら、単に削除してください。


if "α" not in result_marks:
    used_now = set(result_marks.values())
    pool = [i for i in USED_IDS if i not in used_now]
    if pool:
        alpha_pick = pool[-1]
        result_marks["α"] = alpha_pick
        reasons[alpha_pick] = reasons.get(alpha_pick, "α（フォールバック：禁止条件全滅→最弱を採用）")


# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import math
from statistics import mean, pstdev


# =========================
#  Tesla369｜出力統合・最終ブロック（安定版・重複なし / 3車ライン厚め対応）
# =========================
import re, json, hashlib, math
from typing import List, Dict, Any, Optional

# ---------- 基本ヘルパ ----------
def _t369_norm(s) -> str:
    return (str(s) if s is not None else "").replace("　", " ").strip()

def _t369_safe_mean(xs, default: float = 0.0) -> float:
    try:
        return sum(xs) / len(xs) if xs else default
    except Exception:
        return default

def _t369_sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-2.0 * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ---------- 文脈→ライン/印/スコア復元 ----------
def _t369_parse_lines_from_context() -> List[List[int]]:
    # _groups 優先
    try:
        _gs = globals().get("_groups") or []
        if _gs:
            out: List[List[int]] = []
            for g in _gs:
                ln = [int(x) for x in g if str(x).strip()]
                if ln: out.append(ln)
            if out: return out
    except Exception:
        pass
    # line_inputs（例："16","524","37"...）
    try:
        arr = [_t369_norm(x) for x in (globals().get("line_inputs") or []) if _t369_norm(x)]
        out: List[List[int]] = []
        for s in arr:
            nums = [int(ch) for ch in s if ch.isdigit()]
            if nums: out.append(nums)
        return out
    except Exception:
        return []

def _t369_lines_str(lines: List[List[int]]) -> str:
    return " ".join("".join(str(n) for n in ln) for ln in lines)

def _t369_buckets(lines: List[List[int]]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    lid = 0
    for ln in lines:
        if len(ln) == 1:
            m[ln[0]] = f"S{ln[0]}"
        else:
            lid += 1
            for n in ln: m[n] = f"L{lid}"
    return m

# ライン
_lines_list: List[List[int]] = _t369_parse_lines_from_context()
lines_str: str = globals().get("lines_str") or _t369_lines_str(_lines_list)

# 印（result_marks → {"◎":3,...}）
_result_marks_raw = (globals().get("result_marks", {}) or {})
marks: Dict[str, int] = {}
for k, v in _result_marks_raw.items():
    m = re.search(r"\d+", str(v))
    if m:
        try: marks[str(k)] = int(m.group(0))
        except Exception: pass

# スコア（race_t / USED_IDS）
race_t   = dict(globals().get("race_t", {}) or {})
USED_IDS = list(globals().get("USED_IDS", []) or [])

def _t369_num(v) -> float:
    try: return float(v)
    except Exception:
        try: return float(str(v).replace("%","").strip())
        except Exception: return 0.0

def _t369_get_score_from_entry(e: Any) -> float:
    if isinstance(e, (int, float)): return float(e)
    if isinstance(e, dict):
        for k in ("偏差値","hensachi","dev","score","sc","S","s","val","value"):
            if k in e: return _t369_num(e[k])
    return 0.0

scores: Dict[int, float] = {}
ids_source = USED_IDS[:] or [n for ln in _lines_list for n in ln]
for n in ids_source:
    e = race_t.get(n, race_t.get(int(n), race_t.get(str(n), {})))
    scores[int(n)] = _t369_get_score_from_entry(e)
for n in [x for ln in _lines_list for x in ln]:
    scores.setdefault(int(n), 0.0)

# ---------- 流れ指標（簡潔・安定版） ----------
def compute_flow_indicators(lines_str, marks, scores):
    parts = [_t369_norm(p) for p in str(lines_str).split() if _t369_norm(p)]
    lines = [[int(ch) for ch in p if ch.isdigit()] for p in parts if any(ch.isdigit() for ch in p)]
    if not lines:
        return {
            "VTX": 0.0, "FR": 0.0, "U": 0.0,
            "note": "【流れ未循環】ラインなし → ケン",
            "waves": {}, "vtx_bid": "", "lines": [], "dbg": {}
        }

    buckets = _t369_buckets(lines)
    bucket_to_members = {buckets[ln[0]]: ln for ln in lines}

    def mean(xs, d=0.0):
        try:
            return sum(xs)/len(xs) if xs else d
        except Exception:
            return d

    def avg_score(mem):
        return mean([scores.get(n, 50.0) for n in mem], 50.0)

    muA = mean([avg_score(ln) for ln in lines], 50.0)/100.0
    star_id = marks.get("◎", -999)
    none_id = marks.get("無", -999)

    def est(mem):
        A = max(10.0, min(avg_score(mem), 90.0))/100.0
        if star_id in mem:
            phi0, d = -0.8, +1
        elif none_id in mem:
            phi0, d = +0.8, -1
        else:
            phi0, d = +0.2, +1
        phi = phi0 + 1.2*(A - muA)
        return A, phi, d

    def S_end(A, phi, t=0.9, f=0.9, gamma=0.12):
        return A*math.exp(-gamma*t)*(2*math.pi*f*math.cos(2*math.pi*f*t+phi) - gamma*math.sin(2*math.pi*f*t+phi))

    waves = {}
    for bid, mem in bucket_to_members.items():
        A, phi, d = est(mem)
        waves[bid] = {"A": A, "phi": phi, "d": d, "S": S_end(A, phi, t=0.9)}

    def bucket_of(x):
        try:
            return buckets.get(int(x), "")
        except Exception:
            return ""

    def I(bi, bj):
        if not bi or not bj or bi not in waves or bj not in waves:
            return 0.0
        return math.cos(waves[bi]["phi"] - waves[bj]["phi"])

    # --- ◎（順流）と 無（逆流）の決定 ---
    b_star = bucket_of(star_id)
    if not b_star:
        try:
            b_star = max(
                bucket_to_members.keys(),
                key=lambda bid: _t369_safe_mean(
                    [scores.get(n, 50.0) for n in bucket_to_members[bid]], 50.0
                )
            )
        except Exception:
            b_star = ""

    all_buckets = list(bucket_to_members.keys())
    cand_buckets = [bid for bid in all_buckets if bid != b_star]

    b_none = bucket_of(none_id)
    if (not b_none) or (b_none == b_star):
        b_none = None

    if b_none is None:
        posS = [
            (waves.get(bid, {}).get("S", -1e9), bid)
            for bid in cand_buckets
            if waves.get(bid, {}).get("S", -1e9) > 0
        ]
        if posS:
            b_none = max(posS)[1]
    if b_none is None:
        low_mu = sorted(
            cand_buckets,
            key=lambda bid: _t369_safe_mean(
                [scores.get(n, 50.0) for n in bucket_to_members[bid]], 50.0
            )
        )
        if low_mu:
            b_none = low_mu[0]
    if b_none is None:
        anyS = [(waves.get(bid, {}).get("S", -1e9), bid) for bid in cand_buckets]
        if anyS:
            b_none = max(anyS)[1]
    if (not b_none) or (b_none == b_star):
        b_none = cand_buckets[0] if cand_buckets else ""

    # --- VTX（位相差×振幅） ---
    vtx_list = []
    for bid, mem in bucket_to_members.items():
        if bid in (b_star, b_none):
            continue
        if waves.get(bid, {}).get("S", -1e9) < -0.02:
            continue
        wA = 0.5 + 0.5*waves[bid]["A"]
        v = (0.6*abs(I(bid, b_star)) + 0.4*abs(I(bid, b_none))) * wA
        vtx_list.append((v, bid))
    vtx_list.sort(reverse=True, key=lambda x: x[0])
    VTX     = vtx_list[0][0] if vtx_list else 0.0
    VTX_bid = vtx_list[0][1] if vtx_list else ""

    # --- FR（◎下向き×無上向き） ---
    ws, wn = waves.get(b_star, {}), waves.get(b_none, {})
    def S_point(w, t=0.95, f=0.9, gamma=0.12):
        if not w:
            return 0.0
        A, phi = w.get("A", 0.0), w.get("phi", 0.0)
        return A * math.exp(-gamma * t) * (
            2*math.pi*f*math.cos(2*math.pi*f*t + phi) - gamma*math.sin(2*math.pi*f*t + phi)
        )
    blend_star = 0.6 * S_point(ws) + 0.4 * ws.get("S", 0.0)
    blend_none = 0.6 * S_point(wn) + 0.4 * wn.get("S", 0.0)
    def sig(x, k=3.0):
        try:
            return 1.0/(1.0+math.exp(-k*x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    sd_raw = (sig(-blend_star, 3.0) - 0.5) * 2.0
    nu_raw = (sig( blend_none, 3.0) - 0.5) * 2.0
    sd = max(0.0, sd_raw)
    nu = max(0.05, nu_raw)
    FR = sd * nu

    # --- U（逆流圧） ---
    vtx_vals = [v for v, _ in vtx_list] or [0.0]
    vtx_mu = _t369_safe_mean(vtx_vals, 0.0)
    vtx_sd = (_t369_safe_mean([(x - vtx_mu)**2 for x in vtx_vals], 0.0))**0.5
    vtx_hi = max(0.60, vtx_mu + 0.35*vtx_sd)
    VTX_high = 1.0 if VTX >= vtx_hi else 0.0
    FR_high  = 1.0 if FR  >= 0.12 else 0.0
    S_max = max(1e-6, max(abs(w["S"]) for w in waves.values()))
    S_noneN = max(0.0, wn.get("S", 0.0)) / S_max
    U_raw = sig(I(b_none, b_star), k=2.0)
    U = max(0.05, (0.6*U_raw + 0.4*S_noneN) * (1.0 if VTX_high > 0 else 0.8))

    def label(bid):
        mem = bucket_to_members.get(bid, [])
        return "".join(map(str, mem)) if mem else "—"

    tag = "点灯" if (VTX_high > 0 and FR_high > 0) else "判定基準内"
    note = "\n".join([
        f"【順流】◎ライン {label(b_star)}：失速危険 {'高' if FR>=0.15 else ('中' if FR>=0.05 else '低')}",
        f"【渦】候補ライン：{label(VTX_bid)}（VTX={VTX:.2f}）",
        f"【逆流】無ライン {label(b_none)}：U={U:.2f}（※判定基準内）",
    ])

    dbg = {"blend_star": blend_star, "blend_none": blend_none, "sd": sd, "nu": nu, "vtx_hi": vtx_hi}
    return {"VTX": VTX, "FR": FR, "U": U, "note": note, "waves": waves,
            "vtx_bid": VTX_bid, "lines": lines, "dbg": dbg}


# === v2.2: 相手4枠ロジック（3車厚め“強制保証”＆U高域でも最大2枚まで許容）===
def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: Dict[int, float],              # 偏差値/スコアのマップ
    vtx: float,                          # 渦の強さ（0〜1）
    u: float,                            # 逆流の強さ（0〜1）
    marks: Dict[str, int],               # 印（{'◎':5, ...}）
    shissoku_label: str = "中",         # ◎ラインの「失速危険」ラベル：'低'/'中'/'高'
    vtx_line_str: Optional[str] = None,  # 渦候補ライン文字列（例 '375'）
    u_line_str: Optional[str] = None,    # 逆流ライン文字列（例 '63'）
    n_opps: int = 4
) -> List[int]:
    U_HIGH = 0.90  # ← 0.85→0.90に引き上げ（代表1枚化の発動を絞る）

    groups     = _t369p_parse_groups(lines_str)
    axis_line  = _t369p_find_line_of(int(axis), groups)
    others_all = [x for g in groups for x in g if x != axis]

    vtx_group = _t369p_parse_groups(vtx_line_str)[0] if vtx_line_str else []
    u_group   = _t369p_parse_groups(u_line_str)[0]   if u_line_str   else []

    # FRライン（◎のライン。なければ平均最大ライン）
    g_star  = marks.get("◎")
    FR_line = _t369p_find_line_of(int(g_star), groups) if isinstance(g_star, int) else []
    if not FR_line and groups:
        FR_line = max(groups, key=lambda g: _t369p_line_avg(g, hens))

    thick_groups = [g for g in groups if len(g) >= 3]  # 3車(以上)ライン
    # 軸ライン以外の“最厚”を特定（平均偏差で最大）
    thick_others = [g for g in thick_groups if g != (axis_line or [])]
    best_thick_other = max(thick_others, key=lambda g: _t369p_line_avg(g, hens), default=None)

    # 必須候補
    picks_must: List[int] = []

    # ① 軸相方（番手）を強採用
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if axis_partner is not None:
        picks_must.append(axis_partner)

    # ② 対抗ライン代表（平均偏差最大ラインの代表）
    other_lines = [g for g in groups if g != axis_line]
    best_other_line = max(other_lines, key=lambda g: _t369p_line_avg(g, hens), default=None)
    opp_rep = _t369p_best_in_group(best_other_line, hens, exclude=None) if best_other_line else None
    if opp_rep is not None:
        picks_must.append(opp_rep)

    # ③ 逆流代表（U高域のみ“代表”）。※3車u_groupは最大2枚まで許容
    u_rep = None
    if u >= U_HIGH:
        if u_group:
            u_rep = _t369p_best_in_group(u_group, hens, exclude=None)
        else:
            pool = [x for x in others_all if x not in (axis_line or [])]
            u_rep = max(pool, key=lambda x: hens.get(x, 0.0), default=None) if pool else None
        if u_rep is not None:
            picks_must.append(u_rep)

    # ④ スコアリング
    scores_local: Dict[int, float] = {x: 0.0 for x in others_all}
    for x in scores_local:
        scores_local[x] += hens.get(x, 0.0) / 100.0  # 土台

    # 軸ライン：相方強化＋同ライン控えめ
    if axis_partner is not None and axis_partner in scores_local:
        scores_local[axis_partner] += 1.50
    for x in (axis_line or []):
        if x not in (axis, axis_partner) and x in scores_local:
            scores_local[x] += 0.20

    # 対抗代表を加点
    if opp_rep is not None and opp_rep in scores_local:
        scores_local[opp_rep] += 1.20

    # U高域：代表強化＋“2枚目抑制（3車なら許容2まで）”
    if u >= U_HIGH and u_rep is not None and u_rep in scores_local:
        scores_local[u_rep] += 1.00
        if u_group:
            # 3車以上ならペナルティ緩和（-0.15）、それ以外は従来（-0.40）
            penalty = 0.15 if len(u_group) >= 3 else 0.40
            for x in u_group:
                if x != u_rep and x in scores_local:
                    scores_local[x] -= penalty

    # VTX境界の調律
    if vtx <= 0.55:
        if opp_rep is not None and opp_rep in scores_local:
            scores_local[opp_rep] += 0.40
        for x in (vtx_group or []):
            if x in scores_local:
                scores_local[x] -= 0.20
    elif vtx >= 0.60:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None) if vtx_group else None
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.50

    # ◎「失速=高」→ ◎本人を減点・番手を加点
    if isinstance(g_star, int) and shissoku_label == "高":
        g_line = _t369p_find_line_of(g_star, groups)
        g_ban  = _t369p_best_in_group(g_line, hens, exclude=g_star) if g_line else None
        if g_star in scores_local: scores_local[g_star] -= 0.60
        if g_ban is not None and g_ban in scores_local:
            scores_local[g_ban] += 0.70

    # ★ 3車(以上)ラインは厚め（基礎加点）
    for g3 in thick_groups:
        for x in g3:
            if x != axis and x in scores_local:
                scores_local[x] += 0.25
    #  軸が3車(以上)なら“同ライン2枚体制”を最低保証（後段で強制補正も入れる）
    if axis_line and len(axis_line) >= 3:
        for x in axis_line:
            if x not in (axis, axis_partner) and x in scores_local:
                scores_local[x] += 0.35
    #  渦/FRが3車(以上)なら中核を少し厚め
    if vtx_group and len(vtx_group) >= 3:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None)
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.30
    if FR_line and len(FR_line) >= 3:
        add_fr = 0.30 if shissoku_label != "高" else 0.15
        for x in FR_line:
            if x != axis and x in scores_local:
                scores_local[x] += add_fr

    # まずは必須枠を採用（順序維持）
    def _unique_keep_order(xs: List[int]) -> List[int]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out
    picks = [x for x in _unique_keep_order(picks_must) if x in scores_local and x != axis]

    # 補充：スコア高い順。ただしU高域では u_group の人数上限（1 or 2）を守る
    def _same_group(a: int, b: int, group: List[int]) -> bool:
        return bool(group and a in group and b in group)

    for x, _sc in sorted(scores_local.items(), key=lambda kv: kv[1], reverse=True):
        if x in picks or x == axis:
            continue
        if u >= U_HIGH and u_group:
            limit = 2 if len(u_group) >= 3 else 1
            cnt_u = sum(1 for y in picks if y in u_group)
            if cnt_u >= limit and any(_same_group(x, y, u_group) for y in picks):
                continue
        picks.append(x)
        if len(picks) >= n_opps:
            break

    # ★ 強制保証１：軸が3車(以上)なら、相手4枠に同ライン2枚（相方＋もう1枚）を必ず確保
    if axis_line and len(axis_line) >= 3:
        axis_members = [x for x in axis_line if x != axis]
        present = [x for x in picks if x in axis_members]
        if len(present) < 2 and len(axis_members) >= 2:
            cand = max([x for x in axis_members if x not in picks], key=lambda x: hens.get(x, 0.0), default=None)
            if cand is not None:
                drop_cands = [x for x in picks if x not in axis_members]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [cand]

    # ★ 強制保証２：軸ライン以外で“最厚”の3車(以上)ラインは、相手4枠に最低2枚を確保
    if best_thick_other:
        have = [x for x in picks if x in best_thick_other]
        need = min(2, len(best_thick_other))  # 2枚（グループ人数が2ならその人数）
        while len(have) < need and len(picks) > 0:
            cand = max([x for x in best_thick_other if x not in picks and x != axis],
                       key=lambda x: hens.get(x, 0.0), default=None)
            if cand is None:
                break
            # 落とし：その厚めグループ外で最もスコアの低い1名
            drop_cands = [x for x in picks if x not in best_thick_other]
            if not drop_cands:
                break
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            if worst == cand:
                break
            picks = [x for x in picks if x != worst] + [cand]
            have = [x for x in picks if x in best_thick_other]

    # 最終保険：不足分があれば偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        rest_sorted = sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True)
        for x in rest_sorted:
            picks.append(x)
            if len(picks) >= n_opps:
                break

   # 変更後シグネチャ（fr_vを追加）
def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: Dict[int, float],
    vtx: float,
    u: float,
    marks: Dict[str, int],
    shissoku_label: str = "中",
    vtx_line_str: Optional[str] = None,
    u_line_str: Optional[str] = None,
    n_opps: int = 4,
    fr_v: float | None = None,   # ← 追加
) -> List[int]:
    ...
    # （中略：既存ロジックそのまま）
    ...

    # === ここから「最終保険〜return」までを置換 ===

    # 最終保険：不足分があれば偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        rest_sorted = sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True)
        for x in rest_sorted:
            picks.append(x)
            if len(picks) >= n_opps:
                break

       # ==== 3車ラインの「3番手」保証（FR帯 0.25〜0.65 限定） ====
    BAND_LO, BAND_HI = 0.25, 0.65
    THIRD_MIN = 40.0  # しきい値はあなたの設定どおり
    _FRv = float(fr_v or 0.0)

    # --- 二車軸ロック（相方は絶対保持） ---
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if (axis_partner is not None) and (axis_partner not in picks):
        # 相方を必ず入れる（相方以外を1名落とす）
        drop_cands = [x for x in picks if x not in (axis_line or []) or x == _t369p_best_in_group(axis_line, hens, exclude=axis_partner)]
        if drop_cands:
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            picks = [x for x in picks if x != worst] + [axis_partner]
        else:
            picks.append(axis_partner)

    # --- 3番手保証（相方は落とさない） ---
    if BAND_LO <= _FRv <= BAND_HI:
        target = axis_line if (axis_line and len(axis_line) >= 3) else (
            best_thick_other if (best_thick_other and len(best_thick_other) >= 3) else None
        )
        if target:
            g_sorted = sorted(target, key=lambda x: hens.get(x, 0.0), reverse=True)
            if len(g_sorted) >= 3:
                third = g_sorted[2]
                if (third not in picks) and (hens.get(third, 0.0) >= THIRD_MIN):
                    # 相方を落とさない・target外から落とす
                    drop_cands = [x for x in picks if (x not in target) and (x != axis_partner)]
                    if drop_cands:
                        worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                        if worst != third:
                            picks = [x for x in picks if x != worst] + [third]
                    # どうしても対象が無ければ“target内の相方以外の末位”と置換
                    elif len(picks) >= n_opps:
                        target_inside = [x for x in picks if (x in target) and (x not in (axis, axis_partner))]
                        if target_inside:
                            worst = min(target_inside, key=lambda x: scores_local.get(x, -1e9))
                            if worst != third:
                                picks = [x for x in picks if x != worst] + [third]
                        else:
                            # 入りきらない場合でも相方は守る
                            if len(picks) < n_opps:
                                picks.append(third)

    # --- 二車軸の最終確認（相方が必ず残るよう再チェック） ---
    if (axis_partner is not None) and (axis_partner not in picks):
        # 相方復帰（相方以外の最低スコアを落とす）
        drop_cands = [x for x in picks if x != axis_partner]
        if drop_cands:
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            picks = [x for x in picks if x != worst] + [axis_partner]
        else:
            picks.append(axis_partner)

    # --- ユニーク＆サイズ調整 ---
    seen = set()
    picks = [x for x in picks if not (x in seen or seen.add(x))]
    if len(picks) > n_opps:
        # 相方を保護しつつ超過を落とす
        to_drop = len(picks) - n_opps
        cand = [x for x in picks if x != axis_partner]
        cand_sorted = sorted(cand, key=lambda x: scores_local.get(x, -1e9))
        for i in range(min(to_drop, len(cand_sorted))):
            picks.remove(cand_sorted[i])

    return picks



# === /v2.2 ===



def format_tri_1x4(axis: int, opps: List[int]) -> str:
    opps_sorted = ''.join(str(x) for x in sorted(opps))
    return f"{axis}-{opps_sorted}-{opps_sorted}"

# === PATCH（generate_tesla_bets の直前に挿入）==============================
# ※ re は上で import 済みの想定。未インポートなら `import re` を先頭に追加。

# 軸選定用（generate_tesla_bets から呼ばれる）
def _topk(line, k, scores):
    line = list(line or [])
    return sorted(line, key=lambda x: (scores.get(x, -1.0), -int(x)), reverse=True)[:k]

# ---- 相手4枠ロジック v2.3（3車厚め“強制保証”＋3列目ブースト＋U高域でも最大2枚許容）----
from typing import List, Dict, Optional

def _t369p_parse_groups(lines_str: str) -> List[List[int]]:
    parts = re.findall(r'[0-9]+', str(lines_str or ""))
    groups: List[List[int]] = []
    for p in parts:
        g = [int(ch) for ch in p]
        if g: groups.append(g)
    return groups

def _t369p_find_line_of(num: int, groups: List[List[int]]) -> List[int]:
    for g in groups:
        if num in g:
            return g
    return []

def _t369p_line_avg(g: List[int], hens: Dict[int, float]) -> float:
    if not g: return -1e9
    return sum(hens.get(x, 0.0) for x in g) / len(g)

def _t369p_best_in_group(g: List[int], hens: Dict[int, float], exclude: Optional[int] = None) -> Optional[int]:
    cand = [x for x in (g or []) if x != exclude]
    if not cand: return None
    return max(cand, key=lambda x: hens.get(x, 0.0), default=None)

def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: Dict[int, float],              # 偏差値/スコアのマップ
    vtx: float,                          # 渦の強さ（0〜1）
    u: float,                            # 逆流の強さ（0〜1）
    marks: Dict[str, int],               # 印（{'◎':5, ...}）※ {印:車番}でもOK（外で正規化済を推奨）
    shissoku_label: str = "中",          # ◎ラインの「失速危険」：'低'/'中'/'高'
    vtx_line_str: Optional[str] = None,  # 渦候補ライン（例 '375'）
    u_line_str: Optional[str] = None,    # 逆流ライン（例 '63'）
    n_opps: int = 4,
    fr_v: float | None = None,           # ← レースFRを必ず渡す（帯判定用）
) -> List[int]:
    # しきい値/ブースト
    U_HIGH       = 0.90
    THIRD_BOOST  = 0.18
    THICK_BASE   = 0.25
    AXIS_LINE_2P = 0.35

    groups     = _t369p_parse_groups(lines_str)
    axis_line  = _t369p_find_line_of(int(axis), groups)
    others_all = [x for g in groups for x in g if x != axis]

    vtx_group = _t369p_parse_groups(vtx_line_str)[0] if vtx_line_str else []
    u_group   = _t369p_parse_groups(u_line_str)[0]   if u_line_str   else []

    # FRライン（◎のライン。なければ平均最大）
    # marksは {車番:印} でも {印:車番} でも来ることがあるので両対応
    g_star = None
    if marks:
        # {印:車番}
        if all(isinstance(v, int) for v in marks.values()):
            g_star = marks.get("◎", None)
        else:
            # {車番:印}
            for cid, sym in marks.items():
                try:
                    if sym == "◎":
                        g_star = int(cid)
                        break
                except Exception:
                    pass

    FR_line = _t369p_find_line_of(int(g_star), groups) if isinstance(g_star, int) else []
    if not FR_line and groups:
        FR_line = max(groups, key=lambda g: _t369p_line_avg(g, hens))

    # 3車(以上)ライン群
    thick_groups     = [g for g in groups if len(g) >= 3]
    thick_others     = [g for g in thick_groups if g != (axis_line or [])]
    best_thick_other = max(thick_others, key=lambda g: _t369p_line_avg(g, hens), default=None)

    # 必須枠
    picks_must: List[int] = []

    # ① 軸相方（番手）
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if axis_partner is not None:
        picks_must.append(axis_partner)

    # ② 対抗ライン代表（平均偏差最大ライン）
    other_lines = [g for g in groups if g != axis_line]
    best_other_line = max(other_lines, key=lambda g: _t369p_line_avg(g, hens), default=None)
    opp_rep = _t369p_best_in_group(best_other_line, hens, exclude=None) if best_other_line else None
    if opp_rep is not None:
        picks_must.append(opp_rep)

    # ③ 逆流代表（U高域のみ）。※3車u_groupは最大2枚まで許容
    u_rep = None
    if u >= U_HIGH:
        if u_group:
            u_rep = _t369p_best_in_group(u_group, hens, exclude=None)
        else:
            pool = [x for x in others_all if x not in (axis_line or [])]
            u_rep = max(pool, key=lambda x: hens.get(x, 0.0), default=None) if pool else None
        if u_rep is not None:
            picks_must.append(u_rep)

    # ④ スコアリング
    scores_local: Dict[int, float] = {x: 0.0 for x in others_all}
    for x in scores_local:
        scores_local[x] += hens.get(x, 0.0) / 100.0  # 土台

    # 軸ライン：相方を強化、同ライン他は控えめ
    if axis_partner is not None and axis_partner in scores_local:
        scores_local[axis_partner] += 1.50
    for x in (axis_line or []):
        if x not in (axis, axis_partner) and x in scores_local:
            scores_local[x] += 0.20

    # 対抗代表の底上げ
    if opp_rep is not None and opp_rep in scores_local:
        scores_local[opp_rep] += 1.20

    # U高域：代表強化＋“2枚目抑制（3車はペナルティ緩和）”
    if u >= U_HIGH and u_rep is not None and u_rep in scores_local:
        scores_local[u_rep] += 1.00
        if u_group:
            penalty = 0.15 if len(u_group) >= 3 else 0.40
            for x in u_group:
                if x != u_rep and x in scores_local:
                    scores_local[x] -= penalty

    # VTX境界の調律
    if vtx <= 0.55:
        if opp_rep is not None and opp_rep in scores_local:
            scores_local[opp_rep] += 0.40
        for x in (vtx_group or []):
            if x in scores_local:
                scores_local[x] -= 0.20
    elif vtx >= 0.60:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None) if vtx_group else None
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.50

    # ◎「失速=高」→ ◎本人を減点・番手を加点
    if isinstance(g_star, int) and shissoku_label == "高":
        g_line = _t369p_find_line_of(g_star, groups)
        g_ban  = _t369p_best_in_group(g_line, hens, exclude=g_star) if g_line else None
        if g_star in scores_local: scores_local[g_star] -= 0.60
        if g_ban is not None and g_ban in scores_local:
            scores_local[g_ban] += 0.70

    # ★ 3車(以上)ライン厚め：基礎加点＋“3列目”ブースト
    for g3 in thick_groups:
        for x in g3:
            if x != axis and x in scores_local:
                scores_local[x] += THICK_BASE
        g_sorted = sorted(g3, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            third = g_sorted[2]
            if third != axis and third in scores_local:
                scores_local[third] += THIRD_BOOST

    # 軸が3車(以上)：同ライン2枚体制を強化
    if axis_line and len(axis_line) >= 3:
        for x in axis_line:
            if x not in (axis, axis_partner) and x in scores_local:
                scores_local[x] += AXIS_LINE_2P

    # 渦/FRが3車(以上)：中核を少し厚め
    if vtx_group and len(vtx_group) >= 3:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None)
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.30
    if FR_line and len(FR_line) >= 3:
        add_fr = 0.30 if shissoku_label != "高" else 0.15
        for x in FR_line:
            if x != axis and x in scores_local:
                scores_local[x] += add_fr

    # 必須（順序維持）
    def _unique_keep_order(xs: List[int]) -> List[int]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out
    picks = [x for x in _unique_keep_order(picks_must) if x in scores_local and x != axis]

    # 補充：スコア順。U高域では u_group の人数上限（1 or 2）を守る
    def _same_group(a: int, b: int, group: List[int]) -> bool:
        return bool(group and a in group and b in group)

    for x, _sc in sorted(scores_local.items(), key=lambda kv: kv[1], reverse=True):
        if x in picks or x == axis:
            continue
        if u >= U_HIGH and u_group:
            limit = 2 if len(u_group) >= 3 else 1
            cnt_u = sum(1 for y in picks if y in u_group)
            if cnt_u >= limit and any(_same_group(x, y, u_group) for y in picks):
                continue
        picks.append(x)
        if len(picks) >= n_opps:
            break

    # ★ 強制保証１：軸が3車(以上)→相手4枠に同ライン2枚（相方＋もう1枚）を確保
    if axis_line and len(axis_line) >= 3:
        axis_members = [x for x in axis_line if x != axis]
        present = [x for x in picks if x in axis_members]
        if len(present) < 2 and len(axis_members) >= 2:
            cand = max([x for x in axis_members if x not in picks], key=lambda x: hens.get(x, 0.0), default=None)
            if cand is not None:
                drop_cands = [x for x in picks if x not in axis_members]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [cand]

    # ★ 強制保証２：軸以外で“最厚”の3車(以上)ライン→相手4枠に最低2枚を確保
    if best_thick_other:
        have = [x for x in picks if x in best_thick_other]
        need = min(2, len(best_thick_other))
        while len(have) < need and len(picks) > 0:
            cand = max([x for x in best_thick_other if x not in picks and x != axis],
                       key=lambda x: hens.get(x, 0.0), default=None)
            if cand is None:
                break
            drop_cands = [x for x in picks if x not in best_thick_other]
            if not drop_cands:
                break
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            if worst == cand:
                break
            picks = [x for x in picks if x != worst] + [cand]
            have = [x for x in picks if x in best_thick_other]

    # 最終保険：不足分を偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        rest_sorted = sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True)
        for x in rest_sorted:
            picks.append(x)
            if len(picks) >= n_opps:
                break

       # ===== 3車ラインの「3番手」保証（FR帯 0.25〜0.65）=====
    BAND_LO, BAND_HI = 0.25, 0.65
    THIRD_MIN = 40.0  # 3番手の最低偏差値しきい値（あなたの指定）

    # 軸ライン(3車以上)の3番手を抽出
    axis_line = _t369p_find_line_of(int(axis), groups)
    axis_third = None
    if axis_line and len(axis_line) >= 3:
        g_sorted = sorted(axis_line, key=lambda x: hens.get(x, 0.0), reverse=True)
        # g_sorted[0] が軸 or 相方になりやすいので、3番手はインデックス2
        if len(g_sorted) >= 3:
            axis_third = g_sorted[2]

    # FR帯が 0.25〜0.65 のときだけ発動
    if (fr_v is not None) and (BAND_LO <= float(fr_v) <= BAND_HI) and axis_third is not None:
        if hens.get(axis_third, 0.0) >= THIRD_MIN and axis_third not in picks:
            # target外（=軸ライン外）から最もスコアの低い1名を落として3番手を入れる
            drop_cands = [x for x in picks if x not in axis_line]
            if drop_cands:
                worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                if worst != axis_third:
                    picks = [x for x in picks if x != worst] + [axis_third]

    # ユニーク＆サイズ調整
    seen = set()
    picks = [x for x in picks if not (x in seen or seen.add(x))][:n_opps]
    return picks

def _format_tri_axis_partner_rest(axis: int, opps: list, axis_line: list,
                                  hens: dict, lines: list) -> str:
    """
    出力形式： 軸・相方 － 残り3枠 － 残り3枠
    並び規則：対抗ラインの2名（番号昇順）→ 軸ラインの3番手（存在時）→ 残りをスコア順で充填
    ※ 常に 3 枠埋め切る
    """
    if not isinstance(axis, int) or axis <= 0 or not isinstance(opps, list):
        return "—"

    hens = {int(k): float(v) for k, v in (hens or {}).items() if str(k).isdigit()}
    axis_line = list(axis_line or [])

    # 相方（軸ライン内の最上位・軸以外）
    partner = None
    if axis in axis_line:
        cands = [x for x in axis_line if x != axis]
        if cands:
            partner = max(cands, key=lambda x: (hens.get(x, 0.0), -int(x)))

    # フォールバック：相方不在なら通常 1-XXXX-XXXX
    if partner is None:
        rest = ''.join(str(x) for x in sorted(opps))
        return f"{axis}-{rest}-{rest}"

    # 軸3番手
    axis_third = None
    if len(axis_line) >= 3:
        g_sorted = sorted(axis_line, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            axis_third = g_sorted[2]

    # 対抗ライン（＝軸ライン以外で平均偏差最大）
    def _line_avg(g):
        return sum(hens.get(x, 0.0) for x in g)/len(g) if g else -1e9
    other_lines = [g for g in (lines or []) if g != axis_line]
    opp_line = max(other_lines, key=_line_avg) if other_lines else []

    # 残り3枠（相方を除く）
    pool = [x for x in opps if x != partner]

    # まず対抗ラインの2名（昇順で最大2名まで）
    opp_two = sorted([x for x in pool if x in (opp_line or [])])[:2]

    rest_three = []
    rest_three.extend(opp_two)

    # 軸3番手を優先的に追加（まだ入っておらず、プールに居るなら）
    if axis_third is not None and axis_third in pool and axis_third not in rest_three:
        rest_three.append(axis_third)

    # ★不足充填：3枠になるまでスコア（偏差）降順→番号昇順で埋める
    if len(rest_three) < 3:
        remain = [x for x in pool if x not in rest_three]
        remain_sorted = sorted(remain, key=lambda x: (hens.get(x, 0.0), -int(x)), reverse=True)
        take = 3 - len(rest_three)
        rest_three.extend(remain_sorted[:take])

    # 最終整形（ちょうど3つ）
    rest_three = rest_three[:3]
    # 表示は「対抗2名（昇順） → 軸3番手（ある場合）」の並びを保つ
    def _fmt(rest):
        # 対抗に入っているものは昇順、残りはそのままの順を尊重
        in_opp = [x for x in rest if x in (opp_line or [])]
        not_opp = [x for x in rest if x not in (opp_line or [])]
        return ''.join(str(x) for x in (sorted(in_opp) + not_opp))
    rest_str = _fmt(rest_three)

    return f"{axis}・{partner} － {rest_str} － {rest_str}"




# === /PATCH ==============================================================

# ======================= T369｜FREE-ONLY 完全置換ブロック（精簡版） =======================

# ---- 小ヘルパ（ローカル名で衝突回避） -----------------------------------------
def _free_fmt_nums(arr):
    if isinstance(arr, list):
        return "".join(str(x) for x in arr) if arr else "—"
    return "—"

def _free_fmt_hens(ts_map: dict, ids) -> str:
    ids = list(ids or [])
    ts_map = ts_map or {}
    lines = []
    for n in ids:
        v = ts_map.get(n, ts_map.get(str(n), "—"))
        lines.append(f"{n}: {float(v):.1f}" if isinstance(v, (int, float)) else f"{n}: —")
    return "\n".join(lines)

def _free_norm_marks(marks_any):
    marks_any = dict(marks_any or {})
    if not marks_any:
        return {}
    # 値が全部 int → {印:車番} と判断し反転
    if all(isinstance(v, int) for v in marks_any.values()):
        out = {}
        for k, v in marks_any.items():
            try:
                out[int(v)] = str(k)
            except Exception:
                pass
        return out
    # それ以外は {車番:印}
    out = {}
    for k, v in marks_any.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out

def _free_fmt_marks_line(raw_marks: dict, used_ids: list) -> tuple[str, str]:
    """
    raw_marks: {車番:int -> '◎'} または { '◎' -> 車番:int } の両方に対応
    used_ids:  表示対象の車番リスト（スコア順など）
    戻り値: ("◎5 〇3 ▲1 △2 ×6 α7", "を除く未指名：...") のタプル
    """
    used_ids = [int(x) for x in (used_ids or [])]
    marks = _free_norm_marks(raw_marks)
    prio = ["◎", "〇", "▲", "△", "×", "α"]
    parts = []
    for s in prio:
        ids = [cid for cid, sym in marks.items() if sym == s]
        ids_sorted = sorted(ids, key=lambda c: (used_ids.index(c) if c in used_ids else 10**9, c))
        parts.extend([f"{s}{cid}" for cid in ids_sorted])
    marks_str = " ".join(parts)
    un = [cid for cid in used_ids if cid not in marks]
    no_str = ("を除く未指名：" + " ".join(map(str, un))) if un else ""
    return marks_str, no_str

# --- 3区分バンド（短評で使うなら残す） ---
def _band3_fr(fr: float) -> str:
    if fr >= 0.61: return "不利域"
    if fr >= 0.46: return "標準域"
    return "有利域"

def _band3_vtx(v: float) -> str:
    if v > 0.60:  return "不利域"
    if v >= 0.52: return "標準域"
    return "有利域"

def _band3_u(u: float) -> str:
    if u > 0.65:  return "不利域"
    if u >= 0.55: return "標準域"
    return "有利域"

# --- 優位/互角/混戦 判定（必要なら残す） ---
def infer_eval_with_share(fr_v: float, vtx_v: float, u_v: float, share_pct: float | None) -> str:
    fr_low, fr_high = 0.40, 0.60
    vtx_strong, u_strong = 0.60, 0.65
    share_lo, share_hi = 25.0, 33.0  # %
    if (fr_v > fr_high) and (vtx_v <= vtx_strong) and (u_v <= u_strong) and (share_pct is not None and share_pct >= share_hi):
        return "優位"
    if (fr_v < fr_low) or ((vtx_v > vtx_strong) and (u_v > u_strong)) or (share_pct is not None and share_pct <= share_lo):
        return "混戦"
    return "互角"

# --- carFR順位が未定義でも動かすための安全ガード ---
if "compute_carFR_ranking" not in globals():
    def compute_carFR_ranking(lines, hensa_map, line_fr_map):
        try:
            lines = list(lines or [])
            hensa_map = {int(k): float(v) for k, v in (hensa_map or {}).items() if str(k).isdigit()}
            car_ids = sorted({int(c) for ln in lines for c in (ln or [])}) or sorted(hensa_map.keys())
            car_fr = {cid: 0.0 for cid in car_ids}
            for ln in lines:
                key = "".join(map(str, ln or []))
                lfr = float((line_fr_map or {}).get(key, 0.0) or 0.0)
                if not ln: continue
                hs = [float(hensa_map.get(int(c), 0.0)) for c in ln]
                s = sum(hs)
                w = ([1.0/len(ln)]*len(ln)) if s <= 0.0 else [h/s for h in hs]
                for c, wj in zip(ln, w):
                    car_fr[int(c)] = car_fr.get(int(c), 0.0) + lfr * wj
            def _hs(c): return float(hensa_map.get(int(c), 0.0))
            ordered_pairs = sorted(car_fr.items(), key=lambda kv: (kv[1], _hs(kv[0]), -int(kv[0])), reverse=True)
            text = "\n".join(f"{i}位：{cid} ({v:.4f})" for i,(cid,v) in enumerate(ordered_pairs,1)) if ordered_pairs else "—"
            return text, ordered_pairs, car_fr
        except Exception:
            return "—", [], {}

# ---------- 1) FRで車番を並べる（carFR順位で買い目を固定） ----------
def trio_free_completion(scores, marks_any, flow_ctx=None):
    """
    買い目：carFR順位の1位を軸、2〜5位を相手（順位順のまま）
      → 三連複：1位-2345位-2345位
    戻り値: (trio_text, axis_id, axis_car_fr)
    """
    hens = {int(k): float(v) for k, v in (scores or {}).items() if str(k).isdigit()}
    if not hens:
        return ("—", None, None)

    flow_ctx = dict(flow_ctx or {})
    FRv   = float(flow_ctx.get("FR", 0.0) or 0.0)
    lines = [list(map(int, ln)) for ln in (flow_ctx.get("lines") or [])]

    # 表示と整合を取るためのラインFR推定
    line_fr_map = {}
    if lines:
        line_sums = [(ln, sum(hens.get(x, 0.0) for x in ln)) for ln in lines]
        tot = sum(s for _, s in line_sums) or 1.0
        for ln, s in line_sums:
            line_fr_map["".join(map(str, ln))] = FRv * (s / tot) if FRv > 0.0 else 0.0

    # carFR順位（存在する compute_carFR_ranking を優先）
    _carfr_txt, _carfr_rank, _carfr_map = compute_carFR_ranking(lines, hens, line_fr_map)
    if not _carfr_rank or len(_carfr_rank) < 3:
        return ("—", None, None)

    ordered_ids = [int(cid) for (cid, _) in _carfr_rank]
    axis = ordered_ids[0]
    opps = [c for c in ordered_ids[1:] if c != axis][:4]
    if len(opps) < 2:
        return ("—", None, None)

    mid = "".join(map(str, opps))          # 例: 2345（順位順のまま）
    trio_text = f"{axis}-{mid}-{mid}"
    axis_car_fr = (_carfr_map or {}).get(axis, None)
    return (trio_text, axis, axis_car_fr)

# === 想定FRをラインごとに作り、買目テキストを確定（他の出力は維持） ===
def generate_tesla_bets(flow, lines_str, marks_any, scores):
    flow   = dict(flow or {})
    scores = {int(k): float(v) for k, v in (scores or {}).items() if str(k).isdigit()}

    # 印正規化（表示用）
    marks = _free_norm_marks(marks_any)

    FRv  = float(flow.get("FR", 0.0) or 0.0)
    VTXv = float(flow.get("VTX", 0.0) or 0.0)
    Uv   = float(flow.get("U", 0.0) or 0.0)
    lines = [list(map(int, ln)) for ln in (flow.get("lines") or [])]

    # 表示用ラインFR（従来どおり）
    line_fr_map = {}
    if FRv > 0.0 and lines:
        line_sums = [(ln, sum(scores.get(x, 0.0) for x in ln)) for ln in lines]
        total = sum(s for _, s in line_sums) or 1.0
        for ln, s in line_sums:
            line_fr_map["".join(map(str, ln))] = FRv * (s / total)

    FR_line  = flow.get("FR_line")
    VTX_line = flow.get("VTX_line")
    U_line   = flow.get("U_line")

    if (FR_line is None or FR_line == []) and lines:
        star_id = next((cid for cid, m in marks.items() if m == "◎"), None)
        FR_line = next((ln for ln in lines if isinstance(star_id, int) and star_id in ln), lines[0])

    if (VTX_line is None or VTX_line == []) and lines:
        def _key_of(ln): return line_fr_map.get("".join(map(str, ln)), 0.0)
        others = [ln for ln in lines if ln != FR_line]
        VTX_line = max(others, key=_key_of) if others else (FR_line or [])

    if (U_line is None or U_line == []) and lines:
        def _key_of(ln): return line_fr_map.get("".join(map(str, ln)), 0.0)
        others = [ln for ln in lines if ln not in (FR_line, VTX_line)]
        U_line = min(others, key=_key_of) if others else (VTX_line or FR_line or [])

    # ★買い目は carFR順位フォーマットをそのまま採用
    trio_text, axis_id, axis_fr = trio_free_completion(scores, marks, flow_ctx=flow)
    note_lines = ["【買い目】", f"三連複：{trio_text}" if trio_text not in (None, "—") else "三連複：—"]

    return {
        "FR_line": FR_line,
        "VTX_line": VTX_line,
        "U_line": U_line,
        "FRv": FRv,
        "VTXv": VTXv,
        "Uv": Uv,
        "axis_id": axis_id,
        "axis_fr": axis_fr,
        "line_fr_map": line_fr_map,
        "note": "\n".join(note_lines),
    }

# ---------- 3) 安全ラッパ ----------
def _safe_flow(lines_str, marks, scores):
    try:
        fr = compute_flow_indicators(lines_str, marks, scores)
        return fr if isinstance(fr, dict) else {}
    except Exception:
        return {}

def _safe_generate(flow, lines_str, marks, scores):
    try:
        res = generate_tesla_bets(flow, lines_str, marks, scores)
        return res if isinstance(res, dict) else {"note": "【買い目】出力なし"}
    except Exception as e:
        return {"note": f"⚠ generate_tesla_betsエラー: {type(e).__name__}: {e}"}

# ===================== /T369｜FREE-ONLY 出力一括ブロック（レイアウト改） =====================

# ---------- 4) 出力本体 ----------
_flow = _safe_flow(globals().get("lines_str", ""), globals().get("marks", {}), globals().get("scores", {}))
_bets = _safe_generate(_flow, globals().get("lines_str", ""), globals().get("marks", {}), globals().get("scores", {}))

if "note_sections" not in globals() or not isinstance(note_sections, list):
    note_sections = []

def _free_fmt_nums(arr):
    if isinstance(arr, list):
        return "".join(str(x) for x in arr) if arr else "—"
    return "—"

def _free_fmt_hens(ts_map: dict, ids) -> str:
    ids = list(ids or [])
    ts_map = ts_map or {}
    lines = []
    for n in ids:
        v = ts_map.get(n, ts_map.get(str(n), "—"))
        lines.append(f"{n}: {float(v):.1f}" if isinstance(v, (int, float)) else f"{n}: —")
    return "\n".join(lines)

def _free_fmt_marks_line(raw_marks: dict, used_ids: list) -> tuple[str, str]:
    used_ids = [int(x) for x in (used_ids or [])]
    def _free_norm_marks(marks_any):
        marks_any = dict(marks_any or {})
        if not marks_any:
            return {}
        if all(isinstance(v, int) for v in marks_any.values()):
            out = {}
            for k, v in marks_any.items():
                try:
                    out[int(v)] = str(k)
                except Exception:
                    pass
            return out
        out = {}
        for k, v in marks_any.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                pass
        return out
    marks = _free_norm_marks(raw_marks)
    prio = ["◎", "〇", "▲", "△", "×", "α"]
    parts = []
    for s in prio:
        ids = [cid for cid, sym in marks.items() if sym == s]
        ids_sorted = sorted(ids, key=lambda c: (used_ids.index(c) if c in used_ids else 10**9, c))
        parts.extend([f"{s}{cid}" for cid in ids_sorted])
    marks_str = " ".join(parts)
    un = [cid for cid in used_ids if cid not in marks]
    no_str = ("を除く未指名：" + " ".join(map(str, un))) if un else ""
    return marks_str, no_str

# 旧ゴミ掃除
def _free_kill_old(s: str) -> bool:
    if not isinstance(s, str): return False
    t = s.strip()
    return (
        t.startswith("DBG:") or
        t.startswith("【買い目】") or
        t.startswith("三連複：") or
        "三連複フォーメーション" in t or
        "フォーメーション（固定" in t
    )
note_sections = [s for s in note_sections if not _free_kill_old(s)]

# 事前に各数値を揃える（見出し直後で使用）
FRv         = float(_bets.get("FRv", 0.0) or 0.0)
VTXv        = float(_bets.get("VTXv", 0.0) or 0.0)
Uv          = float(_bets.get("Uv", 0.0) or 0.0)
axis_id     = _bets.get("axis_id")
line_fr_map = _bets.get("line_fr_map", {}) or {}
all_lines   = list(_flow.get("lines") or [])

def _line_key(ln):
    return "" if not ln else "".join(str(x) for x in ln)

axis_line = next((ln for ln in all_lines if isinstance(axis_id, int) and axis_id in ln), [])
axis_line_fr = float(line_fr_map.get(_line_key(axis_line), 0.0) or 0.0)
share_pct = (axis_line_fr / FRv * 100.0) if (FRv > 0 and axis_line) else None

# === 見出し（レース名） ===
venue   = str(globals().get("track") or globals().get("place") or "").strip()
race_no = str(globals().get("race_no") or "").strip()
if venue or race_no:
    _rn = race_no if (race_no.endswith("R") or race_no == "") else f"{race_no}R"
    note_sections.append(f"{venue}{_rn}")

# === 展開評価（判定＋軸ラインFR） ===
def infer_eval_with_share(fr_v: float, vtx_v: float, u_v: float, share_pct: float | None) -> str:
    fr_low, fr_high = 0.40, 0.60
    vtx_strong, u_strong = 0.60, 0.65
    share_lo, share_hi = 25.0, 33.0  # %
    if (fr_v > fr_high) and (vtx_v <= vtx_strong) and (u_v <= u_strong) and (share_pct is not None and share_pct >= share_hi):
        return "優位"
    if (fr_v < fr_low) or ((vtx_v > vtx_strong) and (u_v > u_strong)) or (share_pct is not None and share_pct <= share_lo):
        return "混戦"
    return "互角"

note_sections.append(f"展開評価：{infer_eval_with_share(FRv, VTXv, Uv, share_pct)}")

note_sections.append("")  # 空行

# === 時刻・クラス ===
race_time  = str(globals().get("race_time", "") or "")
race_class = str(globals().get("race_class", "") or "")
hdr = f"{race_time}　{race_class}".strip()
if hdr:
    note_sections.append(hdr)

# === ライン ===
line_inputs = globals().get("line_inputs", [])
if isinstance(line_inputs, list) and any(str(x).strip() for x in line_inputs):
    note_sections.append("ライン　" + "　".join([x for x in line_inputs if str(x).strip()]))


note_sections.append("")  # 空行

# === ライン想定FR（順流/渦/逆流 + その他） ===
_FR_line  = _bets.get("FR_line", _flow.get("FR_line"))
_VTX_line = _bets.get("VTX_line", _flow.get("VTX_line"))
_U_line   = _bets.get("U_line",  _flow.get("U_line"))

def _line_fr_val(ln):
    return float(line_fr_map.get(_line_key(ln), 0.0) or 0.0)

note_sections.append(f"【順流】◎ライン {_free_fmt_nums(_FR_line)}：想定FR={_line_fr_val(_FR_line):.3f}")
note_sections.append(f"【渦】候補ライン：{_free_fmt_nums(_VTX_line)}：想定FR={_line_fr_val(_VTX_line):.3f}")
note_sections.append(f"【逆流】無ライン {_free_fmt_nums(_U_line)}：想定FR={_line_fr_val(_U_line):.3f}")
for ln in all_lines:
    if ln == _FR_line or ln == _VTX_line or ln == _U_line:
        continue
    note_sections.append(f"　　　その他ライン {_free_fmt_nums(ln)}：想定FR={_line_fr_val(ln):.3f}")

# === carFR順位（表示） ===
try:
    import re, statistics  # 追加
    _scores_for_rank = {int(k): float(v) for k, v in (globals().get("scores", {}) or {}).items() if str(k).isdigit()}
    _carfr_txt, _carfr_rank, _carfr_map = compute_carFR_ranking(all_lines, _scores_for_rank, line_fr_map)
    note_sections.append("\n【carFR順位】")
    note_sections.append(_carfr_txt)

    # ↓↓↓ ここから追加（平均値を出して追記）
    _vals = [float(x) for x in re.findall(r'\((\d+\.\d+)\)', _carfr_txt)]
    _avg = statistics.mean(_vals) if _vals else 0.0
    note_sections.append(f"\n平均値 {_avg:.5f}")
    # ↑↑↑ ここまで

except Exception:
    pass


note_sections.append("")  # 空行


# === ＜短評＞（コンパクト） ===
try:
    lines_out = ["\n＜短評＞"]
    lines_out.append(f"・レースFR={FRv:.3f}［{_band3_fr(FRv)}］")
    if axis_line:
        lines_out.append(
            f"・軸ラインFR={axis_line_fr:.3f}（取り分≈{(share_pct or 0.0):.1f}%：軸={axis_id}／ライン={_free_fmt_nums(axis_line)}）"
        )
    lines_out.append(f"・VTX={VTXv:.3f}［{_band3_vtx(VTXv)}］")
    lines_out.append(f"・U={Uv:.3f}［{_band3_u(Uv)}］")

    dbg = _flow.get("dbg", {})
    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star",0.0) or 0.0)
        bn = float(dbg.get("blend_none",0.0) or 0.0)
        sd = float(dbg.get("sd",0.0) or 0.0)
        nu = float(dbg.get("nu",0.0) or 0.0)
        star_txt = "先頭負担:強" if bs <= -0.60 else ("先頭負担:中" if bs <= -0.30 else "先頭負担:小")
        none_txt = "無印押上げ:強" if bn >= 1.20 else ("無印押上げ:中" if bn >= 0.60 else "無印押上げ:小")
        sd_txt   = "ライン偏差:大" if sd >= 0.60 else ("ライン偏差:中" if sd >= 0.30 else "ライン偏差:小")
        nu_txt   = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"
        lines_out.append(f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}")

    note_sections += lines_out

    # 末尾に最終判定
    note_sections.append(f"\n判定：{tier}")
except Exception:
    pass

# ===================== /T369｜FREE-ONLY 出力一括ブロック（レイアウト改） =====================


# =========================
note_text = "\n".join(note_sections)
st.markdown("### 📋 note用（コピーエリア）")
st.text_area("ここを選択してコピー", note_text, height=560)
# =========================


# =========================
#  一括置換ブロック ここまで
# =========================
