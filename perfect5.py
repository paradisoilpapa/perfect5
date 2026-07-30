# -*- coding: utf-8 -*-
# v282（ライン2車換算・単騎2倍修正版）:
# ・ライン勢力は、ライン内KO使用スコア上位2車の合計で算出する。
# ・単騎勢力は、本人のKO使用スコアを2倍し、ラインと同じ2車換算で比較する。
# ・ライン評価グループと流れ想定比率は、この2車換算勢力から再構築する。
# ・採用流れは、各流れ1着候補の個人KO比較ではなく、その候補が属するライン／単騎の2車換算勢力で決める。
# ・採用流れ1・2位のAI低評価側を最終軸にする処理と、最終軸の同ライン全車必須保護は維持する。
# ・券種は三連複1車軸－4車－4車の6点固定を維持する。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想、短評は削除・折り畳み・置換しない。
# v281（同ライン保護・AI妙味軸修正版）:
# ・券種は三連複1車軸－4車－4車の6点固定を維持する。
# ・順流／渦／逆流それぞれの着順予想1位を流れ選定候補として抽出する。
# ・各流れ1位候補のうちKO使用スコア最上位の車が属する流れを採用流れにする。
# ・採用流れの1位・2位を評価軸候補とし、AI評価が低い方を最終軸にする。
# ・AI評価順は ◎＞〇＞△＞×＞無印 とし、AI印は流れ選定には使用しない。
# ・最終軸の同ライン車は軸以外をすべて先にヒモへ確保し、ライン3番手以降も必ず保護する。
# ・残りのヒモ枠は採用流れの着順予想上位から補充し、合計4車にする。
# ・同一車が複数流れで1着候補の場合、採用流れは流れ想定比率が高い方、同率時は順流→渦→逆流とする。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想、短評は削除・折り畳み・置換しない。
# v280（固定流れ軸・三連複6点修正版）:
# ・A～Eの自動振り分け、券種変更、AI印による軸・ヒモ選定を廃止。
# ・順流／渦／逆流それぞれの着順予想1位を軸候補として抽出する。
# ・軸候補の中からKO使用スコア最上位の1車を最終軸にする。
# ・最終軸が1着予想となった流れを採用流れとし、その着順予想から軸を除く上位4車をヒモにする。
# ・買い目は常に三連複1車軸－4車－4車の6点。券種・点数をレースごとに変更しない。
# ・同一車が複数流れで1着候補の場合、ヒモ抽出元は流れ想定比率が高い流れを採用し、同率時は順流→渦→逆流の順とする。
# ・KO使用スコアが完全同点の場合のみ、候補初出順（順流→渦→逆流）と車番順で決定する。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想、短評は削除・折り畳み・置換しない。
# v279（軸・ヒモ・無印ピラミッド判定修正版）:
# ・買い目の形からA～Eを逆算する処理を廃止し、先に「軸層／ヒモ層／評価下位無印層」のピラミッドを確定する。
# ・軸層は従来どおり、着内支持率→加重平均着順→1着支持率→的中評価→KOのヴェロビ軸順位1位。
# ・有効流れの過半数で軸層が残る着順により、A=軸不成立、B=着内軸、C=連対軸、D=1着軸へ段階判定する。
# ・EはAI◎の軸層が1着軸として成立し、ヒモ層2車との上位3車構造も有効流れの過半数で共通する場合だけ。
# ・相手3車はヒモ層を優先し、不足分だけ評価下位無印層から補う。Aも軸層を除いたヒモ／無印層の3車BOXとする。
# ・A=2車複3車BOX、B=3連複1車軸-3車-3車、C=2車複1車軸-3車、D=2車単1着軸→3車、E=3連単12-123-123（4点）。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想、短評は削除・折り畳み・置換しない。
# v278（A候補BOX評価・A表示整合修正版）:
# ・A候補3車は、加重2車複評価表の3組総合点合計を最優先し、同点時は3組の最低総合点、流れ別連対組の包含率の順で決める。
# ・A候補順位はBの相手3車、C・Dの相手候補にも共通使用する。
# ・パターンAでは買い目上の軸がないため「軸層」と表示せず、「ヴェロビ候補1位」と「A候補3車」を表示する。
# ・Aの判定理由を、加重2車複評価が最も安定する3車への集約と明記する。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想は削除・折り畳み・置換しない。
# v277（3点体系・共通相手3車修正版）:
# ・A=2車複3車BOX、B=3連複1車軸-3車-3車、C=2車複1車軸-3車、D=2車単1着軸→3車、E=3連単12-123-123（4点）。
# ・A用の候補順位を共通化し、C・Dの相手も軸を除いたA候補上位3車を使用する。
# ・BはA候補3車の外から「流れ加重的中単騎評価」最上位を軸候補とし、着内支持70%以上なら採用する。
# ・Dは1着支持50%以上、Cは連対支持50%以上で判定し、AI◎一致だけではDへ格上げしない。
# ・A～Dは3点、Eのみ4点。ワイドは使用しない。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想は削除・折り畳み・置換しない。
# v276（5パターン正式化修正版）:
# ・note表示の「全体妙味」「全体分類」を廃止し、最終買い目をパターンA～Eで表示する。
# ・A=2車複4車BOX、B=3連複◎-4車-4車、C=2車複◎-全、D=2車単◎→全、E=3連単12-123-123＋3連複1-2-34。
# ・ヴェロビ軸順位1位とAI◎が一致すれば堅軸。三流れ共通の着内・連対状態と上位2車の核からパターンを決める。
# ・全パターン最大6点。ワイドは使用しない。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想は削除・折り畳み・置換しない。
# v275（旧買い目表示削除修正版）:
# ・note上部の推奨表示は、三層分類・A～E分類・最大6点の新方式へ一本化。
# ・旧「買い目構成／構成詳細／推奨券種／AI信頼判定／旧判定理由／買い目サマリー」は表示しない。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想、短評は一切削除・折り畳み・置換しない。
# ・全体分類は各流れのA～E構成比の加重中心で決め、最も波乱側の分類も併記する。
# v274（表示整理修正版）:
# ・v273の三層分類と最大6点の券種候補は維持し、note上部の新方式表示だけを簡潔化。
# ・表示は「軸層／ヒモ層／評価下位無印層／全体分類／券種候補／判定理由／買い目」に限定。
# ・既存の総合加重単騎評価、加重2車複・3連複評価表、ライン評価、KO、三流れ着順予想、短評は一切削除・折り畳み・置換しない。
# ・全体分類は各流れのA～E構成比の加重中心で決め、最も波乱側の分類も併記する。
# v273（三層分類・券種試作版）:
# ・現行v270-R3の買い目生成は変更せず、比較検証用の新方式ブロックを追加。
# ・三流れの着内支持率→加重平均着順→1着支持率→的中評価→KOの順で「ヴェロビ軸順位」を作る。
# ・ヴェロビ軸順位1位を軸とし、AI◎なら堅軸、AI〇なら支持軸、AI△/×なら妙味軸、AI無印なら穴軸。
# ・軸以外を「AI印ありのヒモ」「AI無印の評価下位」に三層分類し、各流れ上位3車をA～Eへ分類。
# ・最大6点の試作券種候補を表示するが、現行の推奨券種・買い目は上書きしない。複数レース検証後に正式移行する。
# v270-R3（大掃除完成版）:
# ・本来のv270を基準に、男子3連単非該当・ガールズは元5車を一車も切らず三連複12-123-12345の7点。
# ・3連単は、ライン主体の説明可能な展開、3単参考1・2着共通、直後同ライン3着候補の総合点単独1位、3車以上ラインをすべて満たす場合だけ。
# ・AI印、ライン／非ライン加重比較は補助情報に限定し、券種・買い目骨格を上書きしない。
# ・順流／渦／逆流はLINE_ZONE_MAPを唯一の基準とし、旧分類は取得不能時のみフォールバック。
# ・廃止済みの流れ1-2下限計算、34-12切替、旧期待値推奨、三展開合成フォメ、旧VeloBi列フォメ生成をコードから完全撤去。重複関数・重複import・自己grepデバッグも整理。
# ・現行で使用する流れ加重的中／妙味評価、2車複21通り、3連複35通り、KO、ライン強度、全体妙味表示は維持。
# v270-R2: 流れラベルの対応と券種判定理由の矛盾を修正。
# v270-R : 誤ったv270・v271・v272を撤回し、本来の元5車三連複7点を復元。
# v267: 男子の元5車は比率1～3位の三流れ・三ライン代表A/B/Cと、代表ライン後位等D/Eから選ぶ。
# v264: 5車をライン分散し、3連単は3車以上ラインに限定。
# v252: 3単参考の1・2着が割れる場合は三連複を維持。
# v259: ライン強度を位置係数の正規化加重平均へ統一。

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ==============================
# 偏差値T（車番→T）自動検出ユーティリティ
# ==============================
def _extract_car_t_map_from_obj(obj):
    """
    obj から「車番→偏差値T(dict)」を取り出す。
    - dict: {1: 52.3, "4": 47.1, ...}
    - Series: indexが車番
    - 1列DataFrame: indexが車番
    """
    if obj is None:
        return None

    # dict
    if isinstance(obj, dict) and obj:
        out = {}
        for k, v in obj.items():
            ks = "".join(ch for ch in str(k) if ch.isdigit())
            if not ks:
                continue
            try:
                out[ks] = 50.0 if v is None else float(v)
            except Exception:
                continue
        return out if out else None

    # pandas Series
    if isinstance(obj, pd.Series) and not obj.empty:
        out = {}
        for k, v in obj.to_dict().items():
            ks = "".join(ch for ch in str(k) if ch.isdigit())
            if not ks:
                continue
            try:
                out[ks] = 50.0 if v is None else float(v)
            except Exception:
                continue
        return out if out else None

    # pandas DataFrame（1列だけ偏差値が入ってる想定）
    if isinstance(obj, pd.DataFrame) and (not obj.empty):
        if obj.shape[1] >= 1:
            s = obj.iloc[:, 0]
            return _extract_car_t_map_from_obj(s)

    return None


# =========================================================
# 必須：グローバル共通部品（参照より先に必ず定義）
# =========================================================

def _digits_of_line(ln):
    s = "".join(ch for ch in str(ln) if ch.isdigit())
    return [int(ch) for ch in s] if s else []

# _PATTERNS をどこかで for で回しているなら、最低限ここで存在させる
_PATTERNS = []   # ← まず NameError を止めるための保険（本来は下で登録する）


# =========================================================
# 現行中核
# 1) ライン・順流／渦／逆流・KO評価
# 2) 流れ加重的中／妙味の単騎評価
# 3) 加重2車複21通り・加重3連複35通り
# 4) 3連単成立判定、または元5車三連複7点
# 5) note用の買い目・根拠・評価表
# =========================================================

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き / 統合版）", layout="wide")

# ==============================
# 現行の流れ比率判定
# ==============================
def _select_recommended_style_by_flow_ratio(current_style, ratio_map):
    """
    v250: 流れ想定比率の単独1位を最終推奨流れにする。

    ・単独1位がある場合は、順流／逆流／渦の別を問わずその流れを採用。
    ・同率の場合は比率だけでは決められないため、直前までの推奨流れを維持。
    ・比率を取得できない場合も直前までの推奨流れを維持。
    """
    _styles = ("順流", "逆流", "渦")
    _current = str(current_style or "")
    _clean = {}

    try:
        for _style in _styles:
            _v = float((ratio_map or {}).get(_style, 0.0) or 0.0)
            if math.isfinite(_v) and _v >= 0.0:
                _clean[_style] = _v
    except Exception:
        return _current, tuple(), "流れ想定比率取得不可"

    if len(_clean) != len(_styles) or sum(_clean.values()) <= 0.0:
        return _current, tuple(), "流れ想定比率取得不可"

    _max_value = max(_clean.values())
    _top_styles = tuple(
        _style for _style in _styles
        if abs(float(_clean[_style]) - float(_max_value)) <= 1e-12
    )

    if len(_top_styles) == 1:
        return _top_styles[0], _top_styles, "流れ想定比率単独1位"

    return _current, _top_styles, "流れ想定比率同率のため既存判定維持"



# ==============================
# v178：開催場決まり手成績 → 会場決まり手補正
# ==============================
# 入力値はオッズパーク等の表をそのまま転記する想定。
# 補正量はコード側で作るため、後から係数だけ調整できる。

VENUE_KIMARITE_BASELINE = {
    "win_escape": 25.0,
    "win_sashi": 50.0,
    "win_makuri": 25.0,
    "sec_escape": 20.0,
    "sec_sashi": 30.0,
    "sec_makuri": 15.0,
    "sec_mark": 35.0,
}

def _pct_input_to_float(v, default=0.0):
    """13.9 / 13.9% / 0.139 のどれでも受ける。戻り値は％値。"""
    try:
        if v is None or v == "":
            return float(default)
        x = float(str(v).replace("%", "").strip())
        if 0.0 < x <= 1.0:
            x *= 100.0
        if not math.isfinite(x):
            return float(default)
        return float(clamp(x, 0.0, 100.0))
    except Exception:
        return float(default)

def _venue_kimarite_reliability(sample_count):
    """回数が少ない時は補正を弱める。150回以上は満額。"""
    try:
        n = int(float(sample_count or 0))
    except Exception:
        n = 0
    if n <= 0:
        return 0.0
    return float(clamp((n / 150.0) ** 0.5, 0.35, 1.0))

def _calc_venue_kimarite_role_bonus_map(stats, max_abs=0.35):
    """
    開催場決まり手成績から、ライン役割別の小幅補正を作る。
    head      : 逃げ/先行残り
    second    : 番手差し・2着マーク
    thirdplus : 後位マーク残り
    single    : 捲り/単騎一撃
    """
    if not isinstance(stats, dict) or not stats.get("enabled", False):
        return {"head":0.0, "second":0.0, "thirdplus":0.0, "single":0.0}, 0.0, {}

    base = VENUE_KIMARITE_BASELINE
    rel = _venue_kimarite_reliability(stats.get("sample_count", 0))
    if rel <= 0.0:
        return {"head":0.0, "second":0.0, "thirdplus":0.0, "single":0.0}, 0.0, {}

    d = {
        k: _pct_input_to_float(stats.get(k, base[k]), base[k]) - base[k]
        for k in base.keys()
    }

    # 1%差を何ptに変換するか。
    # v180: 先頭車・番手車の補正だけv179比50%へ弱化。
    # v181: 3列目保護へ干渉させないため、thirdplus / single への会場決まり手補正は使わない。
    raw = {
        "head":      0.010*d["win_escape"] + 0.005*d["sec_escape"],
        "second":    0.010*d["win_sashi"]  + 0.005*d["sec_mark"] + 0.003*d["sec_sashi"],
        "thirdplus": 0.0,
        "single":    0.0,
    }

    role_bonus = {
        k: float(clamp(v * rel, -float(max_abs), float(max_abs)))
        for k, v in raw.items()
    }

    detail = {"diff": d, "raw": raw, "reliability": rel, "max_abs": float(max_abs)}
    return role_bonus, rel, detail

def _fmt_signed_pt(v):
    try:
        return f"{float(v):+.2f}pt"
    except Exception:
        return "+0.00pt"

def _apply_venue_kimarite_to_score_map(score_map, line_def, stats):
    """score_mapへ会場決まり手補正を常時小幅反映する。"""
    role_bonus, rel, detail = _calc_venue_kimarite_role_bonus_map(stats)
    reason_map = {}
    out = dict(score_map or {})

    for k in list(out.keys()):
        try:
            car = int(k)
            role = role_in_line(car, line_def) if isinstance(line_def, dict) else "single"
            if role not in role_bonus:
                role = "single"
            b = float(role_bonus.get(role, 0.0) or 0.0)
            out[k] = float(out.get(k, 0.0) or 0.0) + b
            reason_map[car] = f"{role}:{_fmt_signed_pt(b)}"
        except Exception:
            continue

    return out, role_bonus, rel, detail, reason_map

# ==============================
# 既存：風・会場・マスタ
# ==============================
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

# KO(勝ち上がり)関連
KO_GIRLS_SCALE = 0.0
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.007   # 0.010 → 0.007
KO_STEP_SIGMA = 0.35   # 0.4 → 0.35


# ◎ライン格上げ
LINE_BONUS_ON_TENKAI = {"優位"}
LINE_BONUS = {"second": 0.08, "thirdplus": 0.04}
LINE_BONUS_CAP = 0.10
# ==============================
# ユーティリティ
# ==============================
def clamp(x,a,b): return max(a, min(b, x))

def zscore_list(arr):
    arr = np.array(arr, dtype=float)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return np.zeros_like(arr) if s==0 else (arr-m)/s

# ==============================
# H：最終ホーム地力補正
# ==============================
H_SCORE_SCALE = float(globals().get("H_SCORE_SCALE", 0.045))
H_SCORE_CAP   = float(globals().get("H_SCORE_CAP", 0.075))

def calc_h_score_map(H: dict, active_cars: list[int]) -> dict[int, float]:
    """
    Hをレース内z化して、車番ごとの相対H評価を作る。
    絶対値ではなく、そのレース内でHが高いか低いかを見る。
    """
    vals = np.array(
        [float(H.get(int(n), 0.0)) for n in active_cars],
        dtype=float
    )

    if len(vals) < 2:
        return {int(n): 0.0 for n in active_cars}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))

    if sd < 1e-12:
        return {int(n): 0.0 for n in active_cars}

    return {
        int(n): float((float(H.get(int(n), 0.0)) - mu) / sd)
        for n in active_cars
    }


def h_home_bonus(no: int, role: str, H_Z: dict[int, float]) -> float:
    """
    H補正。
    ライン先頭・単騎を中心に加点。
    番手・三番手は薄く反映。
    """
    role_mult = {
        "head": 1.00,
        "single": 0.70,
        "second": 0.30,
        "thirdplus": 0.15,
    }.get(role, 0.20)

    raw = H_SCORE_SCALE * float(H_Z.get(int(no), 0.0)) * role_mult
    return round(clamp(raw, -H_SCORE_CAP, H_SCORE_CAP), 3)


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

def extract_car_list(s, n_cars=None):
    """
    ライン入力文字列から車番を抽出する。

    v179修正：
    ・単騎の「6」も [6] として必ず有効扱いする。
    ・2桁以上のラインだけを有効にする判定は行わない。
    ・出走数 n_cars では車番を制限しない。
      5車立てでも欠番あり入力を許可するため。
    ・同一ライン内の重複は先頭1回だけ残す。
    """
    cars = []
    seen = set()

    raw = "" if s is None else str(s)
    for ch in raw:
        if not ch.isdigit():
            continue
        v = int(ch)
        if 1 <= v <= 9 and v not in seen:
            cars.append(v)
            seen.add(v)

    return cars


def build_line_maps(lines):
    # 最大9ラインまで対応。単騎も1ライン。
    labels = "ABCDEFGHI"
    line_def = {}
    for i, lst in enumerate(lines):
        if not lst:
            continue
        label = labels[i] if i < len(labels) else f"L{i+1}"
        line_def[label] = list(lst)
    car_to_group = {c: g for g, mem in line_def.items() for c in mem}
    return line_def, car_to_group


def _format_lines_for_check(lines):
    """入力確認用：[[7,1,4],[5,3,2],[6]] → '714 / 532 / 6'"""
    try:
        parts = []
        for lst in lines:
            if not lst:
                continue
            parts.append("".join(str(int(x)) for x in lst))
        return " / ".join(parts) if parts else "—"
    except Exception:
        return "—"


def role_in_line(car, line_def):
    for g, mem in line_def.items():
        if car in mem:
            if len(mem) == 1:
                return 'single'
            idx = mem.index(car)
            return ['head', 'second', 'thirdplus'][idx] if idx < 3 else 'thirdplus'
    return 'single'
# =====================================================
# ラスト半周補正：番手差し・前で動ける上位補正
# =====================================================

LAST_HALF_ENABLE = True

# ラスト半周補正の全体上限
LAST_HALF_CAP = 0.050


def calc_last_half_role_bonus(
    role: str,
    kaku: str,
    tenscore: float,
    leader_tenscore: float,
    race_avg_tenscore: float,
    h_count: float = 0.0,
    b_count: float = 0.0,
    race_score_rank=None,
    ko_score_rank=None,
    tenkai_score_rank=None,
    top_third_limit: int = 3,
    scenario_top_count: int = 0,
    p1_rate=None,
    p2_rate=None,
    p3_rate=None,
):
    """
    ラスト半周〜ゴール前の個人戦補正。

    思想：
    ラスト半周までは団体戦。
    ラスト半周からは個人戦。
    そのため、位置ではなく「実際に着を取れる個人成績」で補正する。

    使用するもの：
    ・1着率
    ・2着内率
    ・3着内率

    使わないもの：
    ・番手位置だけの加点
    ・H/Bだけの加点
    ・自力だから加点
    ・単騎だから加点
    ・H主導3番手以降だから加点
    """

    if not LAST_HALF_ENABLE:
        return 0.0, []

    bonus = 0.0
    reasons = []

    try:
        role = str(role)

        def _rate(v):
            try:
                x = float(v)
                if x > 1.0:
                    x = x / 100.0
                return x
            except Exception:
                return None

        p1 = _rate(p1_rate)
        p2 = _rate(p2_rate)
        p3 = _rate(p3_rate)

        # ---------------------------------------------
        # 個人戦補正
        # ---------------------------------------------
        # 勝ち切れる個人力を強めに評価
        if p1 is not None and p1 >= 0.20:
            bonus += 0.025
            reasons.append(f"1着率{p1 * 100:.0f}%以上")

        # 2着内率は評価するが、1着率より軽くする
        if p2 is not None and p2 >= 0.30:
            bonus += 0.010
            reasons.append(f"2着内率{p2 * 100:.0f}%以上")

        # 3着内率は、2着内率もある場合だけ補正
        # 3着に残るだけの選手をラスト半周個人力として過大評価しない
        if (
            p3 is not None
            and p3 >= 0.40
            and p2 is not None
            and p2 >= 0.30
        ):
            bonus += 0.010
            reasons.append(f"3着内率{p3 * 100:.0f}%以上")

        # ---------------------------------------------
        # 役割別上限
        # 位置で加点はしない。
        # ただし3番手以降だけは暴走防止で上限を低くする。
        # ---------------------------------------------
        if role == "thirdplus":
            role_cap = 0.030
        else:
            role_cap = 0.050

        bonus = clamp(bonus, 0.0, role_cap)
        bonus = clamp(bonus, -LAST_HALF_CAP, LAST_HALF_CAP)

        if not reasons:
            reasons.append("補正なし")

        return round(float(bonus), 3), reasons

    except Exception as e:
        return 0.0, [f"ラスト半周補正エラー:{e}"]

# ==============================

# =====================================================
# 混戦度判定
#   平均得点ではなく、競走得点1位と2位の差で見る
#   High   = 上位差が大きく、順当寄り
#   Middle = 標準
#   Low    = 上位差が小さく、波乱気味
#
#   ※スコア補正には使わない。表示・検証用。
# =====================================================
def calc_race_compactness(ratings_val: dict, active_cars: list):
    vals = []

    for no in active_cars:
        try:
            v = float(ratings_val.get(int(no), 0.0))
            if v > 0:
                vals.append(v)
        except Exception:
            pass

    if len(vals) < 2:
        return {
            "label": "未判定",
            "top1": 0.0,
            "top2": 0.0,
            "top_gap": None,
        }

    vals = sorted(vals, reverse=True)

    top1 = vals[0]
    top2 = vals[1]
    top_gap = top1 - top2

    if top_gap >= 2.00:
        label = "順当寄り"
    elif top_gap >= 1.00:
        label = "標準"
    else:
        label = "波乱気味"

    return {
        "label": label,
        "top1": float(top1),
        "top2": float(top2),
        "top_gap": float(top_gap),
    }

# H：最終ホーム想定ライン
# ==============================
def calc_home_line_scores(line_def: dict, H: dict, B: dict, active_cars: list[int]) -> dict:
    """
    H = 最終ホーム先頭通過回数を使って、
    最終周回ホームで前に出やすいラインを評価する。
    ※本体スコアには混ぜず、展開表示用。
    """
    scores = {}

    for gid, members in line_def.items():
        mem = [int(x) for x in members if int(x) in active_cars]
        if not mem:
            continue

        head = mem[0]
        second = mem[1] if len(mem) >= 2 else None
        third = mem[2] if len(mem) >= 3 else None

        head_h = float(H.get(head, 0))
        second_h = float(H.get(second, 0)) if second is not None else 0.0
        third_h = float(H.get(third, 0)) if third is not None else 0.0

        # 単騎は自分のHをそのまま見る
        if len(mem) == 1:
            score = head_h
        else:
            # ライン先頭のHを主役、番手・三番手は補助
            score = head_h * 0.75 + second_h * 0.15 + third_h * 0.10

        # 同点時の微差用：Bをほんの少しだけ見る
        score += float(B.get(head, 0)) * 0.01

        scores[gid] = round(score, 3)

    return scores


def make_home_line_order(line_def: dict, H: dict, B: dict, active_cars: list[int]) -> list:
    """
    最終ホーム想定ライン順を返す。
    """
    scores = calc_home_line_scores(line_def, H, B, active_cars)

    return sorted(
        scores.keys(),
        key=lambda gid: scores.get(gid, 0.0),
        reverse=True
    )


def format_home_line_order(line_def: dict, order: list) -> str:
    """
    A/B/Cなどのgid順を、実際のライン文字列に変換する。
    例：['B','C','A'] → 26　37　145
    """
    parts = []

    for gid in order:
        members = line_def.get(gid, [])
        if members:
            parts.append("".join(str(int(x)) for x in members))

    return "　".join(parts) if parts else "—"


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

def track_effective_ratio(track_name: str,
                           alpha_goal: float = 0.50,
                           beta_corner: float = 0.25) -> float:
    d = KEIRIN_DATA.get(track_name)
    if not d:
        return 0.50
    lap  = float(d.get("bank_length", 400))
    home = float(d.get("straight_length", 52.0))
    back = 2.0 * home  # ゴール前は半分の仮定
    corner_total = max(lap - home - back, 0.0)
    L_eff = back + alpha_goal * home + beta_corner * corner_total
    ratio = (L_eff / lap) if lap > 0 else 0.50
    return clamp(ratio, 0.20, 0.90)


# =====================================================
# 会場成績手入力補正 × 最終ホームライン流れ補正
#   入力例：
#     的中率 = 12/40     → 30.0%
#     回収率 = 12000/8000 → 150.0%
#   思想：
#     成績が悪い会場ほど、最終H1番手ライン先頭のイン減速を疑い、
#     最終H2番手ライン、とくに番手の外スピード差しを評価する。
# =====================================================

def parse_fraction_rate(text: str, percent: bool = True):
    """
    '12/40' や '12000/8000' を率に変換する。
    percent=True なら 30.0 のように％値で返す。
    空欄・不正値・分母0は None。
    """
    s = str(text or "").strip()
    if not s:
        return None

    try:
        if "/" in s:
            a, b = s.split("/", 1)
            a = float(str(a).replace(",", "").strip())
            b = float(str(b).replace(",", "").strip())
            if b <= 0:
                return None
            rate = a / b
        else:
            v = float(s.replace("%", "").replace(",", "").strip())
            rate = v / 100.0 if v > 1.0 else v

        if not math.isfinite(rate):
            return None

        return rate * 100.0 if percent else rate

    except Exception:
        return None


def judge_venue_profile(hit_rate, return_rate):
    """
    hit_rate / return_rate は％値。
    例：30.0, 120.0
    """
    hr = None if hit_rate is None else float(hit_rate)
    rr = None if return_rate is None else float(return_rate)

    if hr is None and rr is None:
        return "unknown"

    # 回収率が強い。的中率が低ければ一撃型。
    if rr is not None and rr >= 100.0:
        if hr is None or hr >= 35.0:
            return "strong_good"
        return "swing_return"

    # 的中しているのに安い。順位は壊さず必要オッズ側で締める。
    if hr is not None and hr >= 31.0 and rr is not None and rr < 80.0:
        return "cheap_hit"

    # 的中率がかなり低い。
    if hr is not None and hr < 22.0:
        if rr is not None and rr < 50.0:
            return "very_bad"
        return "low_hit_risk"

    # 回収率がかなり悪い。
    if rr is not None and rr < 50.0:
        return "bad"

    # 回収率が低め。
    if rr is not None and rr < 70.0:
        return "normal_watch"

    return "normal"


def _venue_fit_hit_coef(hit_rate):
    """
    v203:
    会場別の的中率を、2車複の「的中期待」へ小幅倍率として反映する。
    hit_rate は％値（例：25.3）。未入力時は 1.00。

    強くしすぎると会場判定に振り回されるため、概ね 0.90〜1.08 に収める。
    """
    try:
        if hit_rate is None:
            return 1.00
        hr = float(hit_rate)
        if not math.isfinite(hr):
            return 1.00
        if hr >= 35.0:
            return 1.08
        if hr >= 30.0:
            return 1.04
        if hr >= 25.0:
            return 1.00
        if hr >= 22.0:
            return 0.96
        if hr >= 18.0:
            return 0.92
        return 0.90
    except Exception:
        return 1.00


def _venue_fit_myoumi_coef(return_rate):
    """
    v203:
    会場別の回収率を、2車複の「妙味期待」へ小幅倍率として反映する。
    return_rate は％値（例：75.5）。未入力時は 1.00。

    回収率が低い開催では、妙味A++頼みの買目が自然に下がる。
    逆に回収率が高い開催では、妙味期待を少し信頼する。
    """
    try:
        if return_rate is None:
            return 1.00
        rr = float(return_rate)
        if not math.isfinite(rr):
            return 1.00
        if rr >= 120.0:
            return 1.10
        if rr >= 100.0:
            return 1.06
        if rr >= 85.0:
            return 1.00
        if rr >= 70.0:
            return 0.94
        if rr >= 50.0:
            return 0.90
        return 0.88
    except Exception:
        return 1.00


VENUE_HOME_FLOW_MULT = {
    "strong_good": 0.50,
    "swing_return": 0.85,
    "normal": 1.00,
    "normal_watch": 1.10,
    "cheap_hit": 0.90,
    "bad": 1.25,
    "low_hit_risk": 1.35,
    "very_bad": 1.50,
    "unknown": 1.00,
}

VENUE_MIN_ODDS_MULT = {
    "strong_good": 0.95,
    "swing_return": 1.05,
    "normal": 1.00,
    "normal_watch": 1.10,
    "cheap_hit": 1.25,
    "bad": 1.20,
    "low_hit_risk": 1.30,
    "very_bad": 1.40,
    "unknown": 1.00,
}

# 係数は「補正点」ではなく倍率前の思想値。
# 実際は HOME_FLOW_BASE_SCALE と会場倍率を掛けて使う。
HOME_FLOW_BASE_SCALE = 0.04
HOME_FLOW_COEF = {
    "top_line": {
        "head":      -0.70,
        "second":    -0.20,
        "third":     -0.10,
        "single":    -0.50,
    },
    "second_line": {
        "head":      +0.50,
        "second":    +0.70,
        "third":     +0.25,
        "single":    +0.20,
    },
    "other_line": {
        "head":       0.00,
        "second":     0.00,
        "third":      0.00,
        "single":     0.00,
    },
}


def calc_venue_shape_index(track_name: str):
    """
    バンク形状から、長いみなし直線リスクを軽く算出する。
    会場成績の補助係数として使い、実績入力を主にする。
    """
    d = KEIRIN_DATA.get(track_name)
    if not d:
        return {"minashi_ratio": 0.0, "bank_support": 0.0, "stretch_risk": 0.0}

    angle = float(d.get("bank_angle", 30.0) or 30.0)
    straight = float(d.get("straight_length", 52.0) or 52.0)
    bank = float(d.get("bank_length", 400.0) or 400.0)

    minashi = 1.75 * straight + 0.25 * bank
    minashi_ratio = minashi / max(bank, 1e-9)
    bank_support = angle / max(minashi_ratio, 1e-9)

    stretch_risk = 0.0
    if minashi_ratio >= 0.520:
        stretch_risk += 1.00
    elif minashi_ratio >= 0.510:
        stretch_risk += 0.60
    elif minashi_ratio >= 0.500:
        stretch_risk += 0.30

    if bank_support < 62.5:
        stretch_risk += 0.60
    elif bank_support < 65.0:
        stretch_risk += 0.30

    if bank <= 340:
        stretch_risk *= 0.75

    return {
        "minashi_ratio": round(float(minashi_ratio), 6),
        "bank_support": round(float(bank_support), 3),
        "stretch_risk": round(float(clamp(stretch_risk, 0.0, 1.50)), 3),
    }


def venue_home_flow_multiplier(track_name: str, venue_profile: str) -> float:
    """
    会場成績による倍率を主、バンク形状リスクを従として合成する。
    strong_good は元評価を壊さないため弱く、very_bad は強くする。
    """
    profile_mult = float(VENUE_HOME_FLOW_MULT.get(str(venue_profile), 1.00))

    try:
        shape = calc_venue_shape_index(track_name)
        shape_risk = float(shape.get("stretch_risk", 0.0) or 0.0)
    except Exception:
        shape_risk = 0.0

    shape_mult = 1.00 + 0.10 * shape_risk
    return round(clamp(profile_mult * shape_mult, 0.40, 1.80), 3)


def home_flow_adjust_by_venue(
    no: int,
    role: str,
    gid,
    home_top_gid,
    home_second_gid,
    track_name: str,
    venue_profile: str,
):
    """
    最終ホーム想定ライン補正。
    - 1番手ライン：イン減速リスクとして減点。特に先頭。
    - 2番手ライン：外スピードラインとして加点。特に番手。
    - その他：据え置き。
    """
    if gid is None:
        return 0.0, "ライン不明"

    if gid == home_top_gid:
        line_pos = "top_line"
        line_label = "H1番手"
    elif home_second_gid is not None and gid == home_second_gid:
        line_pos = "second_line"
        line_label = "H2番手"
    else:
        line_pos = "other_line"
        line_label = "その他"

    r = str(role or "single")
    if r == "thirdplus":
        r = "third"

    mult = venue_home_flow_multiplier(track_name, venue_profile)
    scale = float(HOME_FLOW_BASE_SCALE) * float(mult)
    coef = float(HOME_FLOW_COEF.get(line_pos, {}).get(r, 0.0))
    adj = round(coef * scale, 3)

    reason = f"{line_label}/{r} 係数{coef:+.2f}×倍率{mult:.2f}"
    return adj, reason



def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    s = max(0.0, float(wind_speed))
    WIND_ZERO   = float(globals().get("WIND_ZERO", 0.0))
    WIND_SIGN   = float(globals().get("WIND_SIGN", 1.0))
    WIND_GAIN   = float(globals().get("WIND_GAIN", 1.0))
    WIND_CAP    = float(globals().get("WIND_CAP", 0.06))
    WIND_MODE   = globals().get("WIND_MODE", "scalar")
    WIND_COEFF  = globals().get("WIND_COEFF", {})
    SPECIAL_DIRECTIONAL_VELODROMES = globals().get("SPECIAL_DIRECTIONAL_VELODROMES", set())

    try:
        s_state_track = st.session_state.get("track", "")
    except Exception:
        s_state_track = ""

    # --- 風速→基礎量 ---
    if s <= WIND_ZERO:
        base = 0.0
    elif s <= 5.0:
        base = 0.006 * (s - WIND_ZERO)
    elif s <= 8.0:
        base = 0.021 + 0.008 * (s - 5.0)
    else:
        base = 0.045 + 0.010 * min(s - 8.0, 4.0)

    # --- 位置係数 ---
    pos = {'head':1.00,'second':0.85,'single':0.75,'thirdplus':0.65}.get(role, 0.75)

    # ===== ★ここ①：強風ほど番手・後位を不利にする =====
    wind01 = clamp((s - WIND_ZERO) / (8.0 - WIND_ZERO), 0.0, 1.0)
    track_ratio = track_effective_ratio(s_state_track)
    wind_eff01 = wind01 * track_ratio

    if role in ("second", "thirdplus"):
        pos *= (1.0 - 0.20 * wind_eff01)   # 最大20%だけ削る

    # --- 脚質（自力） ---
    prof = 0.35 + 0.65 * float(prof_escape)
    val = base * pos * prof

    # --- 風向き（既存） ---
    if (WIND_MODE == "directional") or (s >= 7.0 and s_state_track in SPECIAL_DIRECTIONAL_VELODROMES):
        wd = WIND_COEFF.get(wind_dir, 0.0)
        dir_term = clamp(
            s * wd * (0.30 + 0.70 * float(prof_escape)) * 0.6,
            -0.03, 0.03
        )
        val += dir_term

    # ===== ★ここ②：会場ごとに風の効きをスケール =====
    val *= clamp(track_ratio / 0.50, 0.60, 1.40)

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
            # b と c の差が小さいなら入れ替えて “寄せる”
            if vx >= -gap_delta:
                order[i + 1], order[i + 2] = order[i + 2], order[i + 1]
        i += 1

    return order


def apply_anchor_line_bonus(score_raw: dict[int, float],
                            line_of: dict[int, str],   # ★ int→str に直す
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



# ==============================
# 風の自動取得（Open-Meteo / 時刻固定）
# 風向は手入力運用のため、APIでは風速だけ取得する軽量版
# ==============================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openmeteo_hour(lat, lon, target_dt_naive):
    """
    Open-Meteoから風速だけ取得する軽量版。
    風向きはVeloBi側で手入力する前提なので取得しない。
    同じ場・同じ日時は1時間キャッシュして429を避ける。
    """
    import numpy as np

    d = target_dt_naive.strftime("%Y-%m-%d")
    base = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
        "&hourly=wind_speed_10m,precipitation,weather_code"
        "&timezone=Asia%2FTokyo"
        "&windspeed_unit=ms"
        f"&start_date={d}&end_date={d}"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()

        j = r.json().get("hourly", {})
        times = [datetime.fromisoformat(t) for t in j.get("time", [])]

        if not times:
            raise RuntimeError("empty hourly times")

        diffs = [abs((t - target_dt_naive).total_seconds()) for t in times]
        k = int(np.argmin(diffs))

        sp = j.get("wind_speed_10m", [])
        pr = j.get("precipitation", [])
        wc = j.get("weather_code", [])

        speed = float(sp[k]) if k < len(sp) and sp[k] is not None else float("nan")
        precip = float(pr[k]) if k < len(pr) and pr[k] is not None else 0.0
        weather_code = int(wc[k]) if k < len(wc) and wc[k] is not None else None

        return {
            "time": times[k],
            "speed_ms": speed,
            "deg": None,
            "precipitation": precip,
            "weather_code": weather_code,
            "diff_min": diffs[k] / 60.0,
        }

    except requests.exceptions.HTTPError as e:
        if getattr(e.response, "status_code", None) == 429:
            raise RuntimeError(
                "Open-Meteoの取得制限中です。少し時間を空けるか、手入力の風速を使ってください。"
            )
        raise RuntimeError(f"Open-Meteo取得失敗：{e}")

    except Exception as e:
        raise RuntimeError(f"Open-Meteo取得失敗：{e}")

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

_ALL_L  = np.array([KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA], float)
_ALL_TH = np.array([KEIRIN_DATA[k]["bank_angle"]      for k in KEIRIN_DATA], float)
SIG_L  = float(np.std(_ALL_L))  if np.std(_ALL_L)  > 1e-9 else 1.0
SIG_TH = float(np.std(_ALL_TH)) if np.std(_ALL_TH) > 1e-9 else 1.0

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


# ==============================
# ★ 風取得ユーティリティ（名前衝突を解消）
# ==============================

# 1) 取得ターゲット時刻を作る（JST基準・tzなしdatetime）
def build_openmeteo_target_dt(jst_date, race_slot: str):
    h = SESSION_HOUR.get(race_slot, 11)
    if isinstance(jst_date, datetime):
        jst_date = jst_date.date()
    try:
        y, m, d = jst_date.year, jst_date.month, jst_date.day
    except Exception:
        dt = pd.to_datetime(str(jst_date))
        y, m, d = dt.year, dt.month, dt.day
    return datetime(y, m, d, h, 0, 0)


# ==============================
# UI
# ==============================
st.sidebar.header("開催情報 / バンク・風・頭数")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)

track_names = list(KEIRIN_DATA.keys())
track = st.sidebar.selectbox(
    "競輪場（プリセット）",
    track_names,
    index=track_names.index("川崎") if "川崎" in track_names else 0
)
info = KEIRIN_DATA[track]
st.session_state["track"] = track

with st.sidebar.expander("📊 会場別 成績補正", expanded=True):
    venue_hit_input = st.text_input(
        "的中率（的中R/投票R）",
        value="",
        placeholder="例：12/40",
        key="venue_hit_input",
    )
    venue_return_input = st.text_input(
        "回収率（払戻/投資）",
        value="",
        placeholder="例：12000/8000",
        key="venue_return_input",
    )

    venue_hit_rate = parse_fraction_rate(venue_hit_input, percent=True)
    venue_return_rate = parse_fraction_rate(venue_return_input, percent=True)
    venue_profile = judge_venue_profile(venue_hit_rate, venue_return_rate)

    venue_home_flow_mult = venue_home_flow_multiplier(track, venue_profile)
    venue_min_odds_mult = float(VENUE_MIN_ODDS_MULT.get(venue_profile, 1.00))
    venue_hit_expect_coef = _venue_fit_hit_coef(venue_hit_rate)
    venue_myoumi_expect_coef = _venue_fit_myoumi_coef(venue_return_rate)

    venue_shape = calc_venue_shape_index(track)

    hit_txt = "—" if venue_hit_rate is None else f"{venue_hit_rate:.1f}%"
    ret_txt = "—" if venue_return_rate is None else f"{venue_return_rate:.1f}%"

    st.write(f"的中率：**{hit_txt}**")
    st.write(f"回収率：**{ret_txt}**")
    st.write(f"会場判定：**{venue_profile}**")
    st.write(f"開催適合補正：的中期待×**{venue_hit_expect_coef:.2f}** ／ 妙味期待×**{venue_myoumi_expect_coef:.2f}**")
    st.write(f"最終H補正倍率：**{venue_home_flow_mult:.2f}**")
    st.write(f"必要オッズ倍率：**{venue_min_odds_mult:.2f}**")
    st.caption(
        f"みなし直線率 {venue_shape.get('minashi_ratio', 0.0):.3f} / "
        f"カント支え {venue_shape.get('bank_support', 0.0):.1f} / "
        f"形状リスク {venue_shape.get('stretch_risk', 0.0):.2f}"
    )

st.session_state["venue_hit_rate"] = venue_hit_rate
st.session_state["venue_return_rate"] = venue_return_rate
st.session_state["venue_profile"] = venue_profile
st.session_state["venue_home_flow_mult"] = venue_home_flow_mult
st.session_state["venue_min_odds_mult"] = venue_min_odds_mult
st.session_state["venue_hit_expect_coef"] = venue_hit_expect_coef
st.session_state["venue_myoumi_expect_coef"] = venue_myoumi_expect_coef

st.sidebar.markdown("### 🏟️ 開催場決まり手成績")
with st.sidebar.expander("数値入力（オッズパーク等の表をそのまま％入力）", expanded=True):
    venue_kimarite_enabled = st.checkbox(
        "決まり手補正を使う",
        value=bool(st.session_state.get("venue_kimarite_enabled", False)),
        key="venue_kimarite_enabled",
    )
    st.caption("オッズパーク等の表をそのまま％で入力。例：13.9 / 62.4 / 24.2")

    c1, c2, c3 = st.columns(3)
    with c1:
        vk_win_escape = st.number_input("1着 逃げ%", 0.0, 100.0, float(st.session_state.get("vk_win_escape", 0.0) or 0.0), 0.1, key="vk_win_escape")
    with c2:
        vk_win_sashi = st.number_input("1着 差し%", 0.0, 100.0, float(st.session_state.get("vk_win_sashi", 0.0) or 0.0), 0.1, key="vk_win_sashi")
    with c3:
        vk_win_makuri = st.number_input("1着 捲り%", 0.0, 100.0, float(st.session_state.get("vk_win_makuri", 0.0) or 0.0), 0.1, key="vk_win_makuri")

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        vk_sec_escape = st.number_input("2着 逃げ%", 0.0, 100.0, float(st.session_state.get("vk_sec_escape", 0.0) or 0.0), 0.1, key="vk_sec_escape")
    with c5:
        vk_sec_sashi = st.number_input("2着 差し%", 0.0, 100.0, float(st.session_state.get("vk_sec_sashi", 0.0) or 0.0), 0.1, key="vk_sec_sashi")
    with c6:
        vk_sec_makuri = st.number_input("2着 捲り%", 0.0, 100.0, float(st.session_state.get("vk_sec_makuri", 0.0) or 0.0), 0.1, key="vk_sec_makuri")
    with c7:
        vk_sec_mark = st.number_input("2着 マーク%", 0.0, 100.0, float(st.session_state.get("vk_sec_mark", 0.0) or 0.0), 0.1, key="vk_sec_mark")

    vk_sample_count = st.number_input(
        "回数",
        min_value=0,
        max_value=10000,
        value=int(st.session_state.get("vk_sample_count", 0) or 0),
        step=1,
        key="vk_sample_count",
    )

    VENUE_KIMARITE_STATS = {
        "enabled": bool(venue_kimarite_enabled),
        "win_escape": float(vk_win_escape),
        "win_sashi": float(vk_win_sashi),
        "win_makuri": float(vk_win_makuri),
        "sec_escape": float(vk_sec_escape),
        "sec_sashi": float(vk_sec_sashi),
        "sec_makuri": float(vk_sec_makuri),
        "sec_mark": float(vk_sec_mark),
        "sample_count": int(vk_sample_count),
    }

    _vk_role_bonus_preview, _vk_rel_preview, _vk_detail_preview = _calc_venue_kimarite_role_bonus_map(VENUE_KIMARITE_STATS)
    st.caption(
        "補正プレビュー："
        f"先頭 {_fmt_signed_pt(_vk_role_bonus_preview.get('head', 0.0))} / "
        f"番手 {_fmt_signed_pt(_vk_role_bonus_preview.get('second', 0.0))} / "
        f"3番手以降 {_fmt_signed_pt(_vk_role_bonus_preview.get('thirdplus', 0.0))} / "
        f"単騎 {_fmt_signed_pt(_vk_role_bonus_preview.get('single', 0.0))} "
        f"｜信頼係数 {_vk_rel_preview:.2f}"
    )

globals()["VENUE_KIMARITE_STATS"] = VENUE_KIMARITE_STATS
st.session_state["VENUE_KIMARITE_STATS"] = VENUE_KIMARITE_STATS

# v247: 2車複で先行採用する「軸の同ライン相手」の最低妙味点。# v247: 2車複で先行採用する「軸の同ライン相手」の最低妙味点。
# 固定ルールにはせず、検証しながらサイドバーで変更できるようにする。
with st.sidebar.expander("🎯 2車複｜同ライン妙味基準", expanded=True):
    NIFUKU_SAME_LINE_MYOUMI_MIN = st.number_input(
        "同ライン相手の最低妙味点",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state.get("nifuku_same_line_myoumi_min", 7.0)),
        step=0.1,
        format="%.1f",
        key="nifuku_same_line_myoumi_min",
    )
    st.caption(
        "基準以上の同ライン相手を妙味点順で先に採用。"
        "3点未満は他ラインの総合点上位で補完し、基準未満の同ライン相手は復活させません。"
    )

globals()["NIFUKU_SAME_LINE_MYOUMI_MIN"] = float(NIFUKU_SAME_LINE_MYOUMI_MIN)

race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_day = st.sidebar.date_input("日付（風取得用）", value=date.today())

wind_dir = st.sidebar.selectbox(
    "風向", ["無風","左上","上","右上","左","右","左下","下","右下"],
    index=0, key="wind_dir_input"
)

wind_speed_default = st.session_state.get("wind_speed", 3.0)
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 60.0, float(wind_speed_default), 0.1)

with st.sidebar.expander("🌀 風をAPIで自動取得（Open-Meteo）", expanded=False):
    st.sidebar.caption("基準時刻：モ=8時 / デ=11時 / ナ=18時 / ミ=22時（JST・tzなしで取得）")

    if st.sidebar.button("APIで取得→風速に反映", use_container_width=True):
        info_xy = VELODROME_MASTER.get(track)
        if not info_xy or info_xy.get("lat") is None or info_xy.get("lon") is None:
            st.sidebar.error(f"{track} の座標が未登録です（VELODROME_MASTER に lat/lon を入れてください）")
        else:
            try:
                target = build_openmeteo_target_dt(race_day, race_time)
                data = fetch_openmeteo_hour(info_xy["lat"], info_xy["lon"], target)

                st.session_state["wind_speed"] = round(float(data["speed_ms"]), 2)

                precip = float(data.get("precipitation", 0.0) or 0.0)
                weather_code = data.get("weather_code", None)

                st.session_state["precipitation"] = precip
                st.session_state["weather_code"] = weather_code
                st.session_state["is_wet"] = bool(precip >= 0.3)

                st.sidebar.success(
                    f"{track} {target:%Y-%m-%d %H:%M} "
                    f"風速 {st.session_state['wind_speed']:.1f} m/s "
                    f"降水 {precip:.1f}mm/h "
                    f"（API側と{data['diff_min']:.0f}分ズレ）"
                )
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"取得に失敗：{e}")



straight_length = st.sidebar.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle      = st.sidebar.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length     = st.sidebar.number_input("周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)
st.session_state["bank_length"] = float(bank_length)

base_laps = st.sidebar.number_input("周回（通常4）", 1, 10, 4, 1)
day_label = st.sidebar.selectbox(
    "開催日",
    ["初日", "2日目", "3日目", "4日目", "5日目", "最終日"],
    0
)

DAY_LAP_ADD = {
    "初日": 1,
    "2日目": 2,
    "3日目": 3,
    "4日目": 4,
    "5日目": 5,
    "最終日": 6,
}

eff_laps = int(base_laps) + DAY_LAP_ADD[day_label]

race_class = st.sidebar.selectbox(
    "級別",
    ["Ｓ級", "Ａ級", "Ａ級チャレンジ", "ガールズ", "アドバンス"],
    0
)

is_girls_like = race_class in ("ガールズ", "アドバンス")

# === 会場styleを「得意会場平均」を基準に再定義
zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bank_length)
style_raw = venue_mix(zL, zTH, dC)

# 天候による自動バイアス補正
precip = float(st.session_state.get("precipitation", 0.0) or 0.0)

if precip >= 5.0:
    weather_override = 0.6
elif precip >= 2.0:
    weather_override = 0.4
elif precip >= 0.3:
    weather_override = 0.2
else:
    weather_override = 0.0

manual_override = st.sidebar.slider(
    "会場バイアス補正（−2差し ←→ +2先行）",
    -2.0, 2.0, 0.0, 0.1
)

override = clamp(manual_override + weather_override, -2.0, 2.0)

st.sidebar.caption(
    f"天候自動補正：{weather_override:+.1f} / 最終バイアス補正：{override:+.1f}"
)

style = clamp(style_raw + 0.25 * override, -1.0, +1.0)



CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
    "アドバンス":     {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

DAY_FACTOR = {
    "初日": 1.00,
    "2日目": 1.00,
    "3日目": 0.99,
    "4日目": 0.98,
    "5日目": 0.97,
    "最終日": 0.96,
}
day_factor = DAY_FACTOR[day_label]

cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

# ===== 日程・級別・頭数で“周回疲労の効き”を薄くシフト（出力には出さない） =====
DAY_SHIFT = {
    "初日": -0.5,
    "2日目": 0.0,
    "3日目": +0.2,
    "4日目": +0.4,
    "5日目": +0.6,
    "最終日": +0.8,
}
CLASS_SHIFT = {
    "Ｓ級": 0.0,
    "Ａ級": +0.10,
    "Ａ級チャレンジ": +0.20,
    "ガールズ": -0.10,
    "アドバンス": -0.10,
}
HEADCOUNT_SHIFT = {5: -0.20, 6: -0.10, 7: -0.05, 8: 0.0, 9: +0.10}

def fatigue_extra(eff_laps: int, day_label: str, n_cars: int, race_class: str) -> float:
    d = float(DAY_SHIFT.get(day_label, 0.0))
    c = float(CLASS_SHIFT.get(race_class, 0.0))
    h = float(HEADCOUNT_SHIFT.get(int(n_cars), 0.0))
    x = (float(eff_laps) - 2.0) + d + c + h
    return max(0.0, x)

# === PATCH-L200:（以下そのまま） ==========================================
# ...（あなたの last200_bonus 以降は変更なし）

fatigue_value = fatigue_extra(eff_laps, day_label, n_cars, race_class)

globals()["fatigue_value"] = float(fatigue_value)
globals()["fatigue_extra_value"] = float(fatigue_value)

# sidebarの直後あたり（straight_length/style/wind_speedが確定した後）
globals()["straight_length"] = float(straight_length)
globals()["bank_length"]     = float(bank_length)
globals()["bank_angle"]      = float(bank_angle)
globals()["style"]           = float(style)
globals()["wind_speed"]      = float(wind_speed)
globals()["race_class"]      = str(race_class)
globals()["venue_profile"]   = str(st.session_state.get("venue_profile", "unknown"))
globals()["venue_home_flow_mult"] = float(st.session_state.get("venue_home_flow_mult", 1.00))
globals()["venue_min_odds_mult"]  = float(st.session_state.get("venue_min_odds_mult", 1.00))
globals()["venue_hit_expect_coef"] = float(st.session_state.get("venue_hit_expect_coef", 1.00))
globals()["venue_myoumi_expect_coef"] = float(st.session_state.get("venue_myoumi_expect_coef", 1.00))
globals()["n_cars"]          = int(n_cars)
globals()["day_label"] = str(day_label)
globals()["eff_laps"]  = int(eff_laps)
    


# ==============================
# メイン：入力
# ==============================
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き：統合版）⭐")
st.caption(f"風補正モード: {WIND_MODE}（'speed_only'=風速のみ / 'directional'=向きも薄く考慮）")

st.subheader("2026/05/24更新")
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

# ==============================
# メイン入力：通常入力 → 反映ボタンで計算用データを固定
# ※スコア計算ロジックは元コードから変更しない
# ==============================

# ライン構成（最大7：単騎も1ライン）
line_inputs_live = [
    st.text_input("ライン1（例：123）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：456）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：789）", key="line_3", max_chars=9),
    st.text_input("ライン4（任意）", key="line_4", max_chars=9),
    st.text_input("ライン5（任意）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
    st.text_input("ライン8（任意）", key="line_8", max_chars=9),
    st.text_input("ライン9（任意）", key="line_9", max_chars=9),
]
n_cars = int(n_cars)
lines_live = [extract_car_list(x, n_cars) for x in line_inputs_live if str(x).strip()]
line_def_live, car_to_group_live = build_line_maps(lines_live)
active_cars_live = sorted({c for lst in lines_live for c in lst}) if lines_live else list(range(1, n_cars+1))

# v179：単騎ラインも含めた認識確認
if lines_live:
    st.caption(
        f"ライン認識：{_format_lines_for_check(lines_live)} "
        f"｜入力済み車番：{''.join(str(x) for x in active_cars_live)} "
        f"（{len(active_cars_live)}/{int(n_cars)}車）"
    )

# 5〜9車対応：ライン入力漏れチェック（単騎も1車としてカウント）
if len(active_cars_live) != int(n_cars):
    st.warning(
        f"出走数{n_cars}に対して、ライン入力済みは{len(active_cars_live)}車です。"
        " ライン入力漏れを確認してください。"
    )

# -----------------------------------------
# 市場印入力（計算反映前）
# ※全体妙味と加重妙味評価に使うため、反映ボタンより前に置く
# ※出走表を見たまま入力できるように「車番ごとに印を選ぶ」形式にする
# ※内部では従来通り market_honmei_raw / market_taikou_raw / market_tan_raw / market_batsu_raw に変換する
# -----------------------------------------
st.caption("市場印入力（計算反映前）")
st.caption("各車番ごとに外部印を選択してください（未選択は —）。")

_market_mark_options_live = ["—", "◎", "〇", "△", "×"]

# 旧UIで選んでいた値が残っている場合は、初期表示に引き継ぐ
_old_mark_by_car_live = {}
_old_pairs_live = [
    ("◎", st.session_state.get(f"market_honmei_car_r{race_no}", "—")),
    ("〇", st.session_state.get(f"market_taikou_car_r{race_no}", "—")),
    ("△", st.session_state.get(f"market_tan_car_r{race_no}", "—")),
    ("×", st.session_state.get(f"market_batsu_car_r{race_no}", "—")),
]
for _mk, _car in _old_pairs_live:
    if str(_car) != "—":
        _old_mark_by_car_live[str(_car)] = _mk

market_mark_by_car_live = {}

# 1行目：見出し
_header_cols = st.columns([0.9, 1, 1, 1, 1, 1])
_header_cols[0].markdown("**車番**")
for _i, _label in enumerate(_market_mark_options_live, start=1):
    _header_cols[_i].markdown(f"**{_label}**")

# 車番ごとに印を選択
for no in sorted(active_cars_live):
    no_str = str(no)
    default_mark = _old_mark_by_car_live.get(no_str, "—")
    default_idx = _market_mark_options_live.index(default_mark) if default_mark in _market_mark_options_live else 0

    row_cols = st.columns([0.9, 5])
    row_cols[0].markdown(f"**{no}番**")
    with row_cols[1]:
        market_mark_by_car_live[no] = st.radio(
            f"{no}番の市場印",
            _market_mark_options_live,
            index=default_idx,
            horizontal=True,
            key=f"market_mark_by_car_r{race_no}_{no}",
            label_visibility="collapsed",
        )

# 車番→印を、従来形式（印→車番）へ変換
_mark_to_cars_live = {"◎": [], "〇": [], "△": [], "×": []}
for no in sorted(active_cars_live):
    mk = market_mark_by_car_live.get(no, "—")
    if mk in _mark_to_cars_live:
        _mark_to_cars_live[mk].append(str(no))

_duplicate_marks_live = [mk for mk, cars in _mark_to_cars_live.items() if len(cars) >= 2]
if _duplicate_marks_live:
    st.warning(
        "同じ印が複数の車番に入っています。"
        "各印は1車だけにしてください。計算上は車番昇順で先頭の車を採用します。"
    )

market_honmei_raw_live = _mark_to_cars_live["◎"][0] if _mark_to_cars_live["◎"] else "—"
market_taikou_raw_live = _mark_to_cars_live["〇"][0] if _mark_to_cars_live["〇"] else "—"
market_tan_raw_live    = _mark_to_cars_live["△"][0] if _mark_to_cars_live["△"] else "—"
market_batsu_raw_live  = _mark_to_cars_live["×"][0] if _mark_to_cars_live["×"] else "—"

_market_selected_live = [
    ("◎", market_honmei_raw_live),
    ("〇", market_taikou_raw_live),
    ("△", market_tan_raw_live),
    ("×", market_batsu_raw_live),
]
_market_summary_live = "　".join(
    f"{mk}{car}" for mk, car in _market_selected_live if str(car) != "—"
)
st.caption(f"入力印：{_market_summary_live if _market_summary_live else 'なし'}")

# ←←← ここに入れる
def input_float_text(label: str, key: str, placeholder: str = ""):
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
cols = st.columns(len(active_cars_live))
ratings_live, S_live, H_live, B_live = {}, {}, {}, {}

k_esc_live, k_mak_live, k_sashi_live, k_mark_live = {}, {}, {}, {}
x1_live, x2_live, x3_live, x_out_live = {}, {}, {}, {}

for i, no in enumerate(active_cars_live):
    with cols[i]:
        st.markdown(f"**{no}番**")
        ratings_live[no] = input_float_text("得点（空欄可）", key=f"pt_{no}", placeholder="例: 55.0")
        S_live[no] = st.number_input("S", 0, 99, 0, key=f"s_{no}")
        H_live[no] = st.number_input("H", 0, 99, 0, key=f"h_{no}")
        B_live[no] = st.number_input("B", 0, 99, 0, key=f"b_{no}")
        k_esc_live[no]   = st.number_input("逃", 0, 99, 0, key=f"ke_{no}")
        k_mak_live[no]   = st.number_input("捲", 0, 99, 0, key=f"km_{no}")
        k_sashi_live[no] = st.number_input("差", 0, 99, 0, key=f"ks_{no}")
        k_mark_live[no]  = st.number_input("マ", 0, 99, 0, key=f"kk_{no}")
        x1_live[no]  = st.number_input("1着", 0, 99, 0, key=f"x1_{no}")
        x2_live[no]  = st.number_input("2着", 0, 99, 0, key=f"x2_{no}")
        x3_live[no]  = st.number_input("3着", 0, 99, 0, key=f"x3_{no}")
        x_out_live[no]= st.number_input("着外", 0, 99, 0, key=f"xo_{no}")

# =====================================================
# コメントチェック表
#   前検コメントを見て手動チェック
#   自力：自力 / 自力基本 / 自分で / 前で 等
#   自力自在：自力自在 / 何でもやる / 前々自力 等
#   自在：自在 / 前々 / 流れで / 位置取り 等
#   番手：○○君 / ○○へ / 任せる / 近畿勢 等
#   単騎：一人で / 単騎で / 決めず 等（ライン入力上の単騎とは別のコメント補助）
#   競り：競り対象の車番にチェックし、競り相手を選択
#   後位信頼：3番手以降の明確追走/地区まとめ/流動を手動評価
# =====================================================
st.subheader("コメントチェック")

jiryoku_comment_live = {}
jiryoku_jizai_comment_live = {}
jizai_comment_live = {}
target_comment_live = {}
single_comment_live = {}
seri_comment_live = {}
seri_target_live = {}
line_follow_trust_live = {}

comment_cols = st.columns(len(active_cars_live))

for i, no in enumerate(active_cars_live):
    no = int(no)
    with comment_cols[i]:
        st.markdown(f"**{no}番**")

        jiryoku_comment_live[no] = st.checkbox(
            "自力",
            value=False,
            key=f"jiryoku_comment_r{race_no}_{no}"
        )

        jiryoku_jizai_comment_live[no] = st.checkbox(
            "自力自在",
            value=False,
            key=f"jiryoku_jizai_comment_r{race_no}_{no}"
        )

        jizai_comment_live[no] = st.checkbox(
            "自在",
            value=False,
            key=f"jizai_comment_r{race_no}_{no}"
        )

        target_comment_live[no] = st.checkbox(
            "番手",
            value=False,
            key=f"target_comment_r{race_no}_{no}"
        )

        single_comment_live[no] = st.checkbox(
            "単騎",
            value=False,
            key=f"single_comment_r{race_no}_{no}"
        )

        seri_comment_live[no] = st.checkbox(
            "競り",
            value=False,
            key=f"seri_comment_r{race_no}_{no}"
        )

        _seri_target_options = ["—"] + [int(x) for x in active_cars_live if int(x) != int(no)]
        _seri_target_sel = st.selectbox(
            "競り相手",
            options=_seri_target_options,
            index=0,
            key=f"seri_target_r{race_no}_{no}"
        )
        seri_target_live[no] = None if _seri_target_sel == "—" else int(_seri_target_sel)

        # v125: 後位信頼はselectboxではなくチェックボックス式。
        # 単騎コメントは後位信頼ではなく、上の「単騎」チェックで独立管理する。
        # 複数チェック時は、リスクが強い順に 流動 > 地区まとめ > 明確追走 で採用する。
        _old_line_follow_key = f"line_follow_trust_r{race_no}_{no}"
        _old_line_follow_val = str(st.session_state.get(_old_line_follow_key, "通常") or "通常")

        st.caption("後位信頼")
        _lft_clear = st.checkbox(
            "明確",
            value=(_old_line_follow_val == "明確追走"),
            key=f"line_follow_clear_r{race_no}_{no}"
        )
        _lft_district = st.checkbox(
            "地区",
            value=(_old_line_follow_val == "地区まとめ"),
            key=f"line_follow_district_r{race_no}_{no}"
        )
        _lft_flow = st.checkbox(
            "流動",
            value=(_old_line_follow_val == "流動"),
            key=f"line_follow_flow_r{race_no}_{no}"
        )

        _lft_checked_count = sum([
            bool(_lft_clear),
            bool(_lft_district),
            bool(_lft_flow),
        ])
        if _lft_checked_count >= 2:
            st.caption("※複数時は強リスク側を採用")

        if _lft_flow:
            line_follow_trust_live[no] = "流動"
        elif _lft_district:
            line_follow_trust_live[no] = "地区まとめ"
        elif _lft_clear:
            line_follow_trust_live[no] = "明確追走"
        else:
            line_follow_trust_live[no] = "通常"

st.markdown("---")

apply_input = st.button(
    "入力を反映して計算する",
    type="primary",
    use_container_width=True,
    key="apply_input_main"
)

if apply_input:
    st.session_state["race_snapshot"] = {
        "line_inputs": list(line_inputs_live),
        "lines": [list(x) for x in lines_live],
        "line_def": {g: list(mem) for g, mem in line_def_live.items()},
        "car_to_group": dict(car_to_group_live),
        "active_cars": list(active_cars_live),

        "market_honmei_raw": market_honmei_raw_live,
        "market_taikou_raw": market_taikou_raw_live,
        "market_tan_raw": market_tan_raw_live,
        "market_batsu_raw": market_batsu_raw_live,
        # v20: 車番ごとの外部印をそのまま保存する。
        # ここを保存しないと、後段で印→車番の圧縮値から復元するため、
        # 表示上の車番と印がズレる原因になる。
        "market_mark_by_car": {int(k): str(v) for k, v in market_mark_by_car_live.items()},

        "ratings": dict(ratings_live),
        "S": dict(S_live),
        "H": dict(H_live),
        "B": dict(B_live),

        "k_esc": dict(k_esc_live),
        "k_mak": dict(k_mak_live),
        "k_sashi": dict(k_sashi_live),
        "k_mark": dict(k_mark_live),

        "x1": dict(x1_live),
        "x2": dict(x2_live),
        "x3": dict(x3_live),
        "x_out": dict(x_out_live),

        "jiryoku_comment": dict(jiryoku_comment_live),
        "jiryoku_jizai_comment": dict(jiryoku_jizai_comment_live),
        "jizai_comment": dict(jizai_comment_live),
        "target_comment": dict(target_comment_live),
        "single_comment": dict(single_comment_live),
        "seri_comment": dict(seri_comment_live),
        "seri_target": dict(seri_target_live),
        "line_follow_trust": dict(line_follow_trust_live),
    }

snapshot = st.session_state.get("race_snapshot")

if snapshot is None:
    st.info("入力後、『入力を反映して計算する』を押すと本計算します。")
    st.stop()

# ==============================
# ここから下は、反映済みデータだけで計算する
# ==============================

line_inputs = snapshot["line_inputs"]
lines = snapshot["lines"]
line_def = snapshot["line_def"]
car_to_group = snapshot["car_to_group"]
active_cars = snapshot["active_cars"]

ratings = snapshot["ratings"]
S = snapshot["S"]
H = snapshot["H"]
B = snapshot["B"]

k_esc = snapshot["k_esc"]
k_mak = snapshot["k_mak"]
k_sashi = snapshot["k_sashi"]
k_mark = snapshot["k_mark"]

x1 = snapshot["x1"]
x2 = snapshot["x2"]
x3 = snapshot["x3"]
x_out = snapshot["x_out"]

jiryoku_comment = snapshot.get("jiryoku_comment", {})
jiryoku_jizai_comment = snapshot.get("jiryoku_jizai_comment", {})
jizai_comment = snapshot.get("jizai_comment", {})
target_comment = snapshot.get("target_comment", {})
single_comment = snapshot.get("single_comment", {})
seri_comment = snapshot.get("seri_comment", {})
seri_target = snapshot.get("seri_target", {})
line_follow_trust = snapshot.get("line_follow_trust", {})

globals()["jiryoku_comment"] = jiryoku_comment
globals()["jiryoku_jizai_comment"] = jiryoku_jizai_comment
globals()["jizai_comment"] = jizai_comment
globals()["target_comment"] = target_comment
globals()["single_comment"] = single_comment
globals()["seri_comment"] = seri_comment
globals()["seri_target"] = seri_target
globals()["line_follow_trust"] = line_follow_trust

st.caption(
    "反映済みデータで計算中："
    f"車番={active_cars} ／ "
    f"ライン={'　'.join(''.join(map(str, ln)) for ln in lines) if lines else 'なし'}"
)

# 反映済みデータの整合チェック
if len(active_cars) != int(n_cars):
    st.error(
        f"出走数{n_cars}に対して、反映済みラインは{len(active_cars)}車です。"
        f" 反映済み車番: {active_cars}"
    )
    st.stop()

dup_check = []
for lst in lines:
    dup_check.extend(lst)

dups = sorted([x for x in set(dup_check) if dup_check.count(x) >= 2])

if dups:
    st.error(f"同じ車番が複数ラインに入っています: {dups}")
    st.stop()

ratings_val = {no: (float(ratings[no]) if ratings[no] is not None else 55.0) for no in active_cars}

# =====================================================
# 混戦度判定：競走得点1位と2位の差
# ※ active_cars / ratings_val が確定した後で実行する
# =====================================================
race_compact = calc_race_compactness(ratings_val, active_cars)
race_compact_label = race_compact.get("label", "未判定")
race_compact_gap = race_compact.get("top_gap", None)

globals()["race_compact_label"] = race_compact_label
globals()["race_compact_gap"] = race_compact_gap
globals()["race_compact"] = race_compact

# H：最終ホーム想定ライン
home_line_scores = calc_home_line_scores(line_def, H, B, active_cars)

# H：最終ホーム想定ライン
home_line_scores = calc_home_line_scores(line_def, H, B, active_cars)
home_line_order = make_home_line_order(line_def, H, B, active_cars)
home_line_text = format_home_line_order(line_def, home_line_order)

home_top_gid = home_line_order[0] if home_line_order else None
home_second_gid = home_line_order[1] if len(home_line_order) >= 2 else None
globals()["home_top_gid"] = home_top_gid
globals()["home_second_gid"] = home_second_gid

# H主導ライン判定
# Hスコアが低すぎる場合は「主導なし」とする
home_top_score = float(home_line_scores.get(home_top_gid, 0.0)) if home_top_gid is not None else 0.0

if home_top_gid is not None and home_top_score >= 1.0:
    home_top_line = format_home_line_order(line_def, [home_top_gid])
else:
    home_top_line = "主導なし"



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

# === 1〜3着＋着外を “ちゃんと” Form に反映する版（ここだけ置換） ===
p1_eff, p2_eff, p3_eff, pout_eff = {}, {}, {}, {}

for no in active_cars:
    n = x1[no] + x2[no] + x3[no] + x_out[no]

    # 既存：クラス×脚質の prior（あなたの関数をそのまま使う）
    p1_prior, p2_prior = prior_by_class(race_class, style)

    # 追加：3着＆着外の prior（まずは固定で安全運用）
    p3_prior   = 0.10
    pout_prior = 0.55

    n0 = n0_by_n(n)

    if n == 0:
        p1_eff[no], p2_eff[no] = p1_prior, p2_prior
        p3_eff[no]             = p3_prior
        pout_eff[no]           = pout_prior
    else:
        p1_eff[no]  = clamp((x1[no]    + n0*p1_prior ) / (n + n0), 0.0, 0.40)
        p2_eff[no]  = clamp((x2[no]    + n0*p2_prior ) / (n + n0), 0.0, 0.50)
        p3_eff[no]  = clamp((x3[no]    + n0*p3_prior ) / (n + n0), 0.0, 0.55)
        pout_eff[no]= clamp((x_out[no] + n0*pout_prior) / (n + n0), 0.0, 0.95)

    # 合計が暴れない安全弁（1-3着を優先して整える）
    s123 = p1_eff[no] + p2_eff[no] + p3_eff[no]
    if s123 > 0.95:
        scale = 0.95 / s123
        p1_eff[no] *= scale
        p2_eff[no] *= scale
        p3_eff[no] *= scale

    pout_eff[no] = clamp(1.0 - (p1_eff[no] + p2_eff[no] + p3_eff[no]), 0.0, 0.95)

# ★Form：1〜3着を評価、着外は減点（ここが効く）
Form = {
    no: (3.0*p1_eff[no] + 2.0*p2_eff[no] + 1.0*p3_eff[no] - 1.2*pout_eff[no])
    for no in active_cars
}

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

# ==============================
# level_rating_scale 保険定義
# ==============================
if "level_rating_scale" not in globals():
    level_rating_scale = 1.0

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

# ===== 会場個性を“個人スコア”に浸透：bank系補正（差し替え案） =====

def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi, bank_length=None):
    pe = float(prof_escape or 0.0)
    ps = float(prof_sashi  or 0.0)

    # bank_lengthが渡っていない場合の扱いを決める（例：0.0扱い or venue既定値）
    bl = float(bank_length or 0.0)

    zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bl)

    base = clamp(0.06*zTH - 0.05*zL - 0.03*dC, -0.08, +0.08)
    out  = base * pe - 0.5 * base * ps
    return round(out, 3)


def bank_length_adjust(bank_length, prof_oikomi):
    po = float(prof_oikomi or 0.0)
    L  = float(bank_length or 0.0)
    dC = (+0.4 if L >= 480 else 0.0 if L >= 380 else -0.4)

    out = 0.03 * (-dC) * po
    return round(out, 3)



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


# ==============================
# L200_RAW（観測用）を先に作る：ここでは laps_adj 等は一切計算しない
# ==============================
_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir",   wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

L200_RAW = {}
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


# ==============================
# rows（本体計算）ここで laps_adj を計算して使う（2重計算しない）
# ==============================
rows = []

# H：最終ホーム地力補正マップ
H_Z = calc_h_score_map(H, active_cars)

_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir", wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

# =====================================================
# コメント補正用：競り相手・後位信頼の前処理
# =====================================================
jiryoku_comment_map = globals().get("jiryoku_comment", {}) or {}
jiryoku_jizai_comment_map = globals().get("jiryoku_jizai_comment", {}) or {}
jizai_comment_map   = globals().get("jizai_comment", {}) or {}
target_comment_map  = globals().get("target_comment", {}) or {}
single_comment_map  = globals().get("single_comment", {}) or {}
seri_comment_map    = globals().get("seri_comment", {}) or {}
seri_target_map     = globals().get("seri_target", {}) or {}
line_follow_trust_map = globals().get("line_follow_trust", {}) or {}

seri_incoming_map = {}
try:
    for _src, _dst in (seri_target_map or {}).items():
        try:
            _s = int(_src)
            if _dst is None or str(_dst).strip() in ("", "None", "—"):
                continue
            _d = int(_dst)
            if _s == _d:
                continue
            seri_incoming_map.setdefault(_d, []).append(_s)
        except Exception:
            continue
except Exception:
    seri_incoming_map = {}

def _line_follow_trust_bonus_for_car(_no, _role, _is_girls_like=False):
    """
    3番手以降の追走信頼補正。
    ・「〇〇君へ」等の明確追走は3着内・ライン決着を少し救う。
    ・「関東勢へ」等の地区まとめや「流動」は、裏切り/切替リスクとして減点。
    ・3番手以降だけに効かせ、番手評価を歪ませない。
    """
    try:
        if str(_role) != "thirdplus":
            return 0.0
        label = str(line_follow_trust_map.get(int(_no), "通常") or "通常")
        mp = {
            "明確追走": 0.050,
            "通常": 0.000,
            "地区まとめ": -0.025,
            "流動": -0.080,
            "単騎寄り": -0.120,
        }
        v = float(mp.get(label, 0.0))
        if _is_girls_like:
            v *= 0.50
        return round(clamp(v, -0.120, 0.050), 3)
    except Exception:
        return 0.0

for no in active_cars:
    no = int(no)
    role = role_in_line(no, line_def)

    # =====================================================
    # 周回疲労（DAY×頭数×級別を反映）
    # =====================================================
    extra = fatigue_extra(eff_laps, day_label, n_cars, race_class)
    extra = min(extra, 3.0)   # 応急上限（暴走止め）

    fatigue_scale = (
        1.0  if race_class == "Ｓ級" else
        1.1  if race_class == "Ａ級" else
        1.2  if race_class == "Ａ級チャレンジ" else
        1.05
    )

    # =====================================================
    # 周回疲労補正
    # =====================================================
    laps_adj = (
        -0.10 * extra * (1.0 if float(prof_escape[no]) > 0.5 else 0.0)
        + 0.05 * extra * (1.0 if float(prof_oikomi[no]) > 0.4 else 0.0)
    ) * fatigue_scale

    # ガールズは周回疲労を弱める
    if is_girls_like:
        laps_adj *= 0.3

    # 周回疲労の暴走防止
    laps_adj = clamp(laps_adj, -0.22, 0.18)

    # =====================================================
    # コメント補正
    #   自力：本人をプラス補正
    #   番手：本人ではなく、前の自力先頭をライン連動で格上げ
    #   競り：競り対象者を減点
    # =====================================================
    is_jiryoku_comment = bool(jiryoku_comment_map.get(int(no), False))
    is_jiryoku_jizai_comment = bool(jiryoku_jizai_comment_map.get(int(no), False))
    is_jizai_comment   = bool(jizai_comment_map.get(int(no), False))
    is_single_comment  = bool(single_comment_map.get(int(no), False))
    is_seri_comment    = bool(seri_comment_map.get(int(no), False))
    seri_opponents = []
    try:
        _sel_target = seri_target_map.get(int(no), None)
        if _sel_target is not None and str(_sel_target).strip() not in ("", "None", "—"):
            seri_opponents.append(int(_sel_target))
    except Exception:
        pass
    try:
        seri_opponents.extend([int(x) for x in seri_incoming_map.get(int(no), [])])
    except Exception:
        pass
    seri_opponents = [int(x) for x in dict.fromkeys(seri_opponents) if int(x) != int(no)]

        # -----------------------------------------------------
    # 自力・自力自在・自在コメント補正
    #   3つは原則どれか1つ。
    #   自力自在チェック、または自力＋自在の同時チェックは内部的に「自力自在」として扱う。
    #   大きく順位を作り替えず、軸判定・ステップ判定の補助に留める。
    # -----------------------------------------------------
    if is_jiryoku_jizai_comment or (is_jiryoku_comment and is_jizai_comment):
        move_style = "自力自在"
    elif is_jiryoku_comment:
        move_style = "自力"
    elif is_jizai_comment:
        move_style = "自在"
    else:
        move_style = ""

    jiryoku_comment_bonus = 0.0
    jizai_comment_bonus = 0.0

    if move_style == "自力":
        # 主導力寄り。旧自力補正より少し抑え、コメントだけで順位が動きすぎないようにする。
        jiryoku_comment_bonus = 0.105
        if role == "head":
            jiryoku_comment_bonus += 0.015
        try:
            h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
            if h_line and int(h_line[0]) == int(no):
                jiryoku_comment_bonus += 0.025
        except Exception:
            pass
        if is_girls_like:
            jiryoku_comment_bonus *= 0.60

    elif move_style == "自力自在":
        # 主導力と対応力を分割。自力単独より安定寄り、自在単独より主導力あり。
        jiryoku_comment_bonus = 0.065
        jizai_comment_bonus = 0.035
        if role == "head":
            jiryoku_comment_bonus += 0.010
        if role in ("head", "single"):
            jizai_comment_bonus += 0.005
        try:
            h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
            if h_line and int(h_line[0]) == int(no):
                jiryoku_comment_bonus += 0.015
        except Exception:
            pass
        if is_girls_like:
            jiryoku_comment_bonus *= 0.60
            jizai_comment_bonus *= 0.60

    elif move_style == "自在":
        # 自在は1着固定力ではなく、崩れにくさ・位置取りの安定として軽く加点する。
        jizai_comment_bonus = 0.065
        if role in ("head", "single"):
            jizai_comment_bonus += 0.010
        if is_girls_like:
            jizai_comment_bonus *= 0.60

    jiryoku_comment_bonus = clamp(jiryoku_comment_bonus, 0.0, 0.145)
    jizai_comment_bonus = clamp(jizai_comment_bonus, 0.0, 0.080)

    # -----------------------------------------------------
    # 単騎コメント補正
    #   ライン入力上の単騎とは別に、「一人で」「単騎で」「決めず」を明示する補助。
    #   強く減点せず、ライン保護・軸信頼の過信を少し抑える。
    # -----------------------------------------------------
    single_comment_bonus = 0.0
    if is_single_comment:
        single_comment_bonus = -0.010
        if role != "single":
            single_comment_bonus -= 0.010
        if move_style in ("自力", "自力自在", "自在"):
            single_comment_bonus *= 0.50
        if is_girls_like:
            single_comment_bonus *= 0.50
    single_comment_bonus = clamp(single_comment_bonus, -0.020, 0.0)

    # -----------------------------------------------------
    # ライン連動補正
    #   後ろの選手が「番手・目標」チェックありなら、
    #   その前のライン先頭を少し格上げする。
    #   例：42で2が「小原君」なら、4を少し救う。
    # -----------------------------------------------------
    line_cushion_bonus = 0.0

    try:
        gid = car_to_group.get(int(no), None)
        members = line_def.get(gid, []) if gid is not None else []

        # 自分がそのラインの先頭かどうか
        is_line_head = bool(members and int(members[0]) == int(no))

        if is_line_head:
            behind_members = [int(x) for x in members[1:]]

            has_target_behind = any(
                bool(target_comment_map.get(int(x), False))
                for x in behind_members
            )

            if has_target_behind:
                # 番手・後位が前を指名しているなら、先頭車を少し救う
                line_cushion_bonus = 0.040

                # H主導ラインの先頭なら、ライン成立度を少し上乗せ
                try:
                    h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                    if h_line and int(h_line[0]) == int(no):
                        line_cushion_bonus += 0.020
                except Exception:
                    pass

    except Exception:
        line_cushion_bonus = 0.0

    line_cushion_bonus = clamp(line_cushion_bonus, 0.0, 0.060)

    # -----------------------------------------------------
    # 競り補正
    #   ライン入力は崩さず、競り当事者を減点する。
    #   ・自分が競りチェックあり
    #   ・または他車から競り相手として指定されている
    #   このどちらかなら競り当事者として扱う。
    #   さらに、競り相手同士で基礎点が低い側は追加減点する。
    # -----------------------------------------------------
    seri_penalty = 0.0

    is_seri_involved = bool(is_seri_comment or seri_opponents)

    if is_seri_involved:
        seri_penalty = -0.100

        try:
            my_base = float(prof_base.get(int(no), 0.0))
            opp_bases = [
                float(prof_base.get(int(x), 0.0))
                for x in seri_opponents
                if int(x) in prof_base
            ]
            if opp_bases:
                best_opp = max(opp_bases)
                # 弱い側はより競り負け・脚消耗しやすいので追加減点
                if my_base + 1e-9 < best_opp:
                    seri_penalty -= 0.050
                else:
                    seri_penalty -= 0.020
        except Exception:
            pass

        # 番手で競る場合は、ライン連動が壊れやすい
        if role == "second":
            seri_penalty -= 0.030

        # ガールズは基本的に競りの意味が薄いので弱め
        if is_girls_like:
            seri_penalty *= 0.50

    seri_penalty = clamp(seri_penalty, -0.180, 0.0)

    # -----------------------------------------------------
    # 3番手以降の追走信頼補正
    # -----------------------------------------------------
    line_follow_trust_bonus = _line_follow_trust_bonus_for_car(no, role, is_girls_like)

    # =====================================================
    # 環境・個人補正（既存）
    # =====================================================
    wind     = _wind_func(eff_wind_dir, float(eff_wind_speed or 0.0), role, float(prof_escape[no]))
    bank_b   = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no], bank_length)
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv    = extra_bonus.get(no, 0.0)
    stab     = stability_score(no)  # 安定度
    h_bonus  = h_home_bonus(no, role, H_Z)

    l200 = l200_adjust(
        role, straight_length, bank_length, race_class,
        float(prof_escape[no]), float(prof_sashi[no]), float(prof_oikomi[no]),
        is_wet=st.session_state.get("is_wet", False)
    )

    # =====================================================
    # 合計スコア
    # =====================================================
    total_raw = (
        prof_base[no]
        + wind
        + cf["spread"] * level_rating_scale * tens_corr.get(no, 0.0)
        + bank_b
        + length_b
        + laps_adj
        + indiv
        + stab
        + h_bonus
        + l200
        + jiryoku_comment_bonus
        + jizai_comment_bonus
        + single_comment_bonus
        + line_cushion_bonus
        + seri_penalty
        + line_follow_trust_bonus
    )

    rows.append([
        no, role,
        round(prof_base[no], 3),
        round(wind, 3),
        round(cf["spread"] * level_rating_scale * tens_corr.get(no, 0.0), 3),
        round(bank_b, 3),
        round(length_b, 3),
        round(laps_adj, 3),
        round(indiv, 3),
        round(stab, 3),
        round(h_bonus, 3),
        round(l200, 3),
        round(jiryoku_comment_bonus, 3),
        round(jizai_comment_bonus, 3),
        round(single_comment_bonus, 3),
        round(line_cushion_bonus, 3),
        round(seri_penalty, 3),
        round(line_follow_trust_bonus, 3),
        float(total_raw)
    ])

df = pd.DataFrame(rows, columns=[
    "車番", "役割", "脚質基準(会場)", "風補正", "得点補正", "バンク補正",
    "周長補正", "周回補正", "個人補正", "安定度", "H補正", "ラスト200",
    "自力コメント補正", "自在コメント補正", "単騎コメント補正", "ライン連動補正", "競り補正", "後位信頼補正",
    "合計_SBなし_raw",
])

# ===== [PATCH] dfの型を確定させ、SBなし母集団(v_wo/v_final)を必ず作る =====
# 1) dfが空のときも落とさない
if df is None or len(df) == 0:
    st.warning("DEBUG: df（SBなし内訳）が空です。rowsが生成されていない可能性。")
    v_wo = {int(no): 0.0 for no in active_cars}
else:
    # 2) 車番を必ずintにする（★最重要：ここがズレると全部emptyになる）
    df["車番"] = df["車番"].astype(int)

    # 3) v_wo を df から必ず生成（全車キー保証）
    v_wo = {int(r["車番"]): float(r["合計_SBなし_raw"]) for _, r in df.iterrows()}
    for no in active_cars:
        ino = int(no)
        if ino not in v_wo:
            v_wo[ino] = 0.0

# 4) v_final は最低でも v_wo を引き継ぐ（KOが走らない/空でも落ちない）
v_final = dict(v_wo)

# 5) df_sorted_pure をここで確定（アンカー選定が安定）
df_sorted_pure = pd.DataFrame({
    "車番": sorted([int(k) for k in v_final.keys()]),
    "合計_SBなし": [float(v_final[int(k)]) for k in sorted([int(k) for k in v_final.keys()])]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)


    


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
# ❸ 役割の日本語化（lineの並びから）
def _role_jp(no: int, line_def: dict) -> str:
    r = role_in_line(no, line_def)
    return {"head":"先頭","second":"番手","thirdplus":"三番手","single":"単騎"}.get(r, "単騎")


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

# 印（◎→▲→偏差値補完）
mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0 * (df["合計_SBなし_raw"] - mu)

# --- SBなし(母集団) を df から「全車ぶん必ず」作る（None防止） ---
sb_map = {int(r["車番"]): float(r.get("合計_SBなし", 0.0)) for _, r in df.iterrows()}

# df が空 / sb_map が空のときは、全車0で母集団を作る（5車・欠番・SB未入力でも止めない）
if not sb_map:
    sb_map = {int(no): 0.0 for no in active_cars}
    

# === [PATCH-A] 安定度をENVから分離し、各柱をレース内z化（SD固定） ===
SD_FORM = 0.28
SD_ENV  = 0.20
SD_STAB = 0.12
SD_L200 = float(globals().get("SD_L200", 0.22))  # ← 追加。まず0.22〜0.30で様子見

# 安定度（raw）と、ENVのベース（= 合計_SBなし_raw から安定度だけ除いたもの）
STAB_RAW = {int(df.loc[i, "車番"]): float(df.loc[i, "安定度"]) for i in df.index}
ENV_BASE = {
    int(df.loc[i, "車番"]): (
        float(df.loc[i, "合計_SBなし_raw"])
        - float(df.loc[i, "安定度"])
        - float(df.loc[i, "ラスト200"])
    )
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

# 0) SBなし(母集団) を df から確実に作る（全車）
sb_map = {int(k): float(v) for k, v in zip(df["車番"].astype(int), df["合計_SBなし"].astype(float))}

# ★必須：dfが空でも全車0で母集団を作る
if not sb_map:
    sb_map = {int(no): 0.0 for no in active_cars}

# 1) key 欠損チェック
missing = [int(n) for n in active_cars if int(n) not in sb_map]
if missing:
    st.error(f"SBなし(母集団) が欠損してる車番: {missing} / sb_map.keys={sorted(sb_map.keys())}")
    # st.stop()

# 2) 値が None/NaN チェック
bad = [
    int(n) for n in active_cars
    if (int(n) in sb_map) and (
        sb_map[int(n)] is None or
        (isinstance(sb_map[int(n)], float) and np.isnan(sb_map[int(n)]))
    )
]
if bad:
    st.error(f"SBなし(母集団) の値が None/NaN: {bad} / values={[sb_map[int(n)] for n in bad]}")
    # st.stop()

# 3) KO入力に使う母集団（全車）
v_wo = dict(sb_map)

# 4) 以降 KO
_is_girls = is_girls_like
head_scale = KO_HEADCOUNT_SCALE.get(int(n_cars), 1.0)
ko_scale_raw = (KO_GIRLS_SCALE if _is_girls else 1.0) * head_scale
KO_SCALE_MAX = 0.45
ko_scale = min(ko_scale_raw, KO_SCALE_MAX)

if ko_scale > 0.0 and line_def and len(line_def) >= 1 and v_wo:
    # --- KO順序（_ko_order が落ちる/不正でも必ずフォールバックで作る） ---
    try:
        ko_order = _ko_order(
            v_wo, line_def, S, B,
            line_factor=line_factor_eff,
            gap_delta=KO_GAP_DELTA
        )
    except Exception as e:
        # Streamlitで原因を見たいならコメント解除
        # st.warning(f"_ko_order fallback: {type(e).__name__}: {e}")
        ko_order = None

    # ★重要：ko_order が None/空/欠損でも「全車」を必ず含める
    ko_order = [int(c) for c in (ko_order or []) if int(c) in v_wo]
    rest = [int(c) for c in v_wo.keys() if int(c) not in set(ko_order)]
    rest = sorted(rest, key=lambda c: float(v_wo[int(c)]), reverse=True)
    ko_order = ko_order + rest  # ← 全車を必ず含める（ここが最重要）

    # ここ以降は ko_order が必ず全車になるので安全
    vals = [float(v_wo[c]) for c in v_wo.keys()]
    mu0  = float(np.mean(vals))
    sd0  = float(np.std(vals) + 1e-12)
    KO_STEP_SIGMA_LOCAL = max(0.25, KO_STEP_SIGMA * 0.7)
    step = KO_STEP_SIGMA_LOCAL * sd0
    # ★new_scores は「全車のベース」から開始して KO で上書き
    new_scores = dict(v_wo)

    for rank, car in enumerate(ko_order, start=1):
        rank_adjust = step * (len(ko_order) - rank)
        blended = (1.0 - ko_scale) * float(v_wo[int(car)]) + ko_scale * (
            mu0 + rank_adjust - (len(ko_order)/2.0 - 0.5)*step
        )
        new_scores[int(car)] = float(blended)

    v_final = dict(new_scores)

else:
    # KOしない時も「全車保持」
    if v_wo:
        ko_order = sorted(v_wo.keys(), key=lambda c: float(v_wo[c]), reverse=True)
        v_final = dict(v_wo)
    else:
        ko_order = []
        v_final = {}

# --- 純SBなしランキング（KOまで／格上げ前）
df_sorted_pure = (pd.DataFrame({
    "車番": sorted([int(k) for k in v_final.keys()]),
    "合計_SBなし": [round(float(v_final[int(c)]), 6) for c in sorted([int(k) for k in v_final.keys()])]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True))


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


# === [PATCH-1] ENV/FORM をレース内で z 化し、目標SDを掛ける（anchor_score の前に置く） ===
SD_FORM = 0.28   # Balanced 既定
SD_ENV  = 0.20

# ENV = v_final（風・会場・周回疲労・個人補正・安定度 等を含む“Form以外”）
# ENV = v_final を int キー前提に揃える
_env_arr = np.array([float(v_final.get(int(n), np.nan)) for n in active_cars], dtype=float)

_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask]))
    sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0

_den = sd_env if sd_env > 1e-12 else 1.0
ENV_Z = {int(n): (float(v_final.get(int(n), mu_env)) - mu_env) / _den for n in active_cars}


# FORM = form_T_map（T=50, SD=10）→ z 化
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}


# --- ここで必ず定義してから使う（NameError防止） ---
line_sb_enable = bool(globals().get("line_sb_enable", (race_class != "ガールズ")))

def _pos_idx(no: int) -> int:
    g = car_to_group.get(no)
    if g is None or g not in line_def:
        return 4  # 単騎/不明は最後方（POS_BONUS[4]）

    grp = line_def[g]  # 例: [5,2,6] みたいな並び
    try:
        return max(0, grp.index(no))
    except ValueError:
        return 4  # グループに居ないなら最後方扱い


bonus_init, _ = compute_lineSB_bonus(
    line_def, S, B,
    line_factor=line_factor_eff,
    exclude=None, cap=cap_SB_eff,
    enable=line_sb_enable
)

def anchor_score(no: int) -> float:
    role = role_in_line(no, line_def)
    sb = float(
        bonus_init.get(car_to_group.get(no, None), 0.0)
        * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    )
    pos_term = (POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)) if line_sb_enable else 0.0
    env_term  = SD_ENV  * float(ENV_Z.get(int(no), 0.0))
    form_term = SD_FORM * float(FORM_Z.get(int(no), 0.0))
    stab_term = SD_STAB * float(STAB_Z.get(int(no), 0.0))
    l200_term = SD_L200 * float(L200_Z.get(int(no), 0.0))
    tiny      = SMALL_Z_RATING * float(zt_map.get(int(no), 0.0))
    return env_term + form_term + stab_term + l200_term + sb + pos_term + tiny



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

# ===== ANCHOR 選定（SBなし母集団ベース）+ 安全弁 + DEBUG =====
ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)

# --- DEBUG（必要ならOFFにできる） ---
DBG_ANCHOR = bool(globals().get("DBG_ANCHOR", True))

# df_sorted_pure が空なら、active_cars を母集団として使う（落下防止）
df_pure_empty = (df_sorted_pure is None) or (len(df_sorted_pure) == 0)

if df_pure_empty:
    base_order = [int(x) for x in list(active_cars)[:]]  # 1..7
else:
    # 念のため int 化
    base_order = df_sorted_pure["車番"].astype(int).tolist()

# rank_pure（SBなしランキング順位）
rank_pure = {int(no): i + 1 for i, no in enumerate(base_order)}

# 候補プール：C の中で SBなし上位K位
cand_pool = [int(c) for c in C if rank_pure.get(int(c), 999) <= ANCHOR_CAND_SB_TOPK]

# もし空なら、SBなし上位K位から直接作る
if not cand_pool:
    cand_pool = [int(no) for no in base_order[:min(ANCHOR_CAND_SB_TOPK, len(base_order))]]

# 最終フォールバック（どれも無い場合）
fallback_no = int(active_cars[0]) if active_cars else 1

# anchor_no_pre（まずは候補プール内で anchor_score 最大）
if cand_pool:
    anchor_no_pre = max(cand_pool, key=lambda x: anchor_score(int(x)))
else:
    anchor_no_pre = fallback_no

anchor_no = anchor_no_pre

# 同点圏（TIE_EPSILON以内）なら ratings_rank2 で決める
top2 = sorted(cand_pool, key=lambda x: anchor_score(int(x)), reverse=True)[:2]
if len(top2) >= 2:
    s1 = float(anchor_score(int(top2[0])))
    s2 = float(anchor_score(int(top2[1])))
    if (s1 - s2) < TIE_EPSILON:
        better_by_rating = min(top2, key=lambda x: ratings_rank2.get(int(x), 999))
        anchor_no = int(better_by_rating)

# SBなし上位N位縛り
if rank_pure.get(int(anchor_no), 999) > ANCHOR_REQUIRE_TOP_SB:
    pool = [int(c) for c in cand_pool if rank_pure.get(int(c), 999) <= ANCHOR_REQUIRE_TOP_SB]
    if pool:
        anchor_no = max(pool, key=lambda x: anchor_score(int(x)))
    else:
        anchor_no = int(base_order[0]) if base_order else fallback_no

    st.caption(
        f"※ ◎は『SBなし 上位{ANCHOR_REQUIRE_TOP_SB}位以内』縛りで {anchor_no_pre}→{anchor_no} に調整。"
    )



# ===== confidence 算出（anchor_score のギャップ/分散）=====
role_map = {int(no): role_in_line(int(no), line_def) for no in active_cars}

cand_scores = [float(anchor_score(int(no))) for no in C] if len(C) >= 2 else [0.0, 0.0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf_gap = float(cand_scores_sorted[0] - cand_scores_sorted[1]) if len(cand_scores_sorted) >= 2 else 0.0

# v_final が空のときは spread=0 で落ちないように（confidenceは混戦寄りになる）
spread = float(np.std(list(v_final.values()))) if isinstance(v_final, dict) and len(v_final) >= 2 else 0.0
norm = conf_gap / (spread if spread > 1e-6 else 1.0)
confidence = "優位" if norm >= 1.0 else ("互角" if norm >= 0.5 else "混戦")

# ===== 格上げ（v_final が空でも落ちないように）=====
if not isinstance(v_final, dict) or len(v_final) == 0:
    # downstream を落とさないための最小母集団（全車0）
    v_final = {int(no): 0.0 for no in active_cars}

score_adj_map = apply_anchor_line_bonus(v_final, car_to_group, role_map, int(anchor_no), confidence)

df_sorted_wo = pd.DataFrame({
    "車番": [int(c) for c in active_cars],
    "合計_SBなし": [
        round(float(score_adj_map.get(int(c), v_final.get(int(c), float("-inf")))), 6)
        for c in active_cars
    ]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

velobi_wo = list(zip(
    df_sorted_wo["車番"].astype(int).tolist(),
    df_sorted_wo["合計_SBなし"].round(3).tolist()
))
# ==============================
# ★ レース内T偏差値 → 印 → 買い目 → note出力（2車系対応＋会場個性浸透版）
# ==============================

HEN_DEC_PLACES = 1

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

# 偏差値母集団は「SBなし（KO適用後＆格上げ前後どちらか）」に固定
SB_BASE_MAP = {int(i): float(score_adj_map.get(int(i), v_final.get(int(i), np.nan))) for i in USED_IDS}



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



# ==============================
# 偏差値テーブル（SBなし母集団）＋欠損ガード
# ==============================
race_z = (xs_race_t - 50.0) / 10.0

# --- SBなし(母集団) を map として確定（KO入力もここを使う） ---
# USED_IDS と xs_base_raw は「同じ順番」で対応している前提
sb_map = {}
for cid, x in zip(USED_IDS, xs_base_raw):
    try:
        if x is None:
            continue
        xf = float(x)
        if not np.isfinite(xf):
            continue
        sb_map[int(cid)] = xf
    except Exception:
        pass

# --- 欠損チェック（None連発の犯人特定） ---
missing = [int(n) for n in active_cars if int(n) not in sb_map]
if missing:
    st.error(f"SBなし(母集団) が欠損してる車番: {missing} / sb_map.keys={sorted(sb_map.keys())}")


# zipで短くなってる可能性チェック
if len(xs_base_raw) != len(USED_IDS):
    st.error("xs_base_raw と USED_IDS の長さが一致していません。zip が途中で切れて欠損になります。")


# --- 表（hen_df）を sb_map から作る：Noneは明示的にNoneで残す ---
hen_df = pd.DataFrame({
    "車": USED_IDS,
    "SBなし(母集団)": [sb_map.get(int(cid), None) for cid in USED_IDS],
    "偏差値T(レース内)": [race_t[int(cid)] for cid in USED_IDS],
}).sort_values(["偏差値T(レース内)", "車"], ascending=[False, True]).reset_index(drop=True)

st.markdown("### 偏差値（レース内T＝平均50・SD10｜SBなしと同一母集団）")
st.caption(f"μ={mu_sb if np.isfinite(mu_sb) else 'nan'} / σ={sd_sb:.6f} / 有効件数k={k_finite}")
st.dataframe(hen_df, use_container_width=True)

# 7) 印（◎〇▲）＝ T↓ → SBなし↓ → 車番↑（βは除外）
if "select_beta" not in globals():
    def select_beta(cars): return None
if "enforce_alpha_eligibility" not in globals():
    def enforce_alpha_eligibility(m): return m

# ===== βラベル付与（単なる順位ラベル） =====
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


# =========================
#  Tesla369｜出力統合・最終ブロック（安定版・重複なし / 3車ライン厚め対応）
# =========================

# ---------- 基本ヘルパ ----------
def _t369_norm(s) -> str:
    return (str(s) if s is not None else "").replace("　", " ").strip()

def _t369_safe_mean(xs, default: float = 0.0) -> float:
    try:
        return sum(xs) / len(xs) if xs else default
    except Exception:
        return default

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


def _t369_line_core_strength(
    mem,
    scores_map,
    singleton_scale: float = 0.70,
    default_score: float = 50.0,
) -> float:
    """
    ライン人数に左右されない流れ用ライン強度。

    ・先頭=1.00、番手=0.72、3番手以降=0.55（既存の位置係数）
    ・重み合計で割るため、2車/3車/4車で人数そのものの加点は発生しない
    ・単騎は従来の抑制係数0.70を維持
    """
    members = []
    for x in (mem or []):
        try:
            members.append(int(x))
        except Exception:
            continue
    if not members:
        return 0.0

    def _score(car: int) -> float:
        try:
            return float((scores_map or {}).get(int(car), float(default_score)))
        except Exception:
            return float(default_score)

    if len(members) == 1:
        return _score(members[0]) * float(singleton_scale)

    weights = [1.00, 0.72] + [0.55] * max(0, len(members) - 2)
    weighted_sum = sum(_score(car) * w for car, w in zip(members, weights))
    weight_total = sum(weights)
    return weighted_sum / weight_total if weight_total > 0.0 else 0.0



def _t369_two_car_equivalent_strength(
    mem,
    scores_map,
    default_score: float = 0.0,
) -> float:
    """ライン／単騎を同じ2車換算で比較する勢力値。

    ・2車以上のライン：ライン内スコア上位2車の合計
    ・単騎：本人スコア×2
    ・3番手以降はライン人数加点に使わない
    """
    members = []
    for value in (mem or []):
        try:
            car = int(value)
        except Exception:
            continue
        if car not in members:
            members.append(car)

    if not members:
        return 0.0

    def _score(car: int) -> float:
        for key in (car, str(car)):
            try:
                if key in (scores_map or {}):
                    value = float((scores_map or {}).get(key, default_score) or default_score)
                    return value if math.isfinite(value) else float(default_score)
            except Exception:
                continue
        return float(default_score)

    values = sorted((_score(car) for car in members), reverse=True)
    if len(values) == 1:
        return float(values[0]) * 2.0
    return float(values[0]) + float(values[1])


def _build_line_two_car_strength_map(lines, scores_map):
    """ラインキー（例: '127'）→2車換算勢力の辞書を作る。"""
    out = {}
    for line in _normalize_lines(lines):
        key = "".join(str(int(car)) for car in line)
        if not key:
            continue
        out[key] = _t369_two_car_equivalent_strength(
            line,
            scores_map,
            default_score=0.0,
        )
    return out


# ---------- 流れ指標（簡潔・安定版） ----------
# ---------- 流れ指標（簡潔・安定版） ----------
def compute_flow_indicators(lines_str, marks, scores):
    parts = [_t369_norm(p) for p in str(lines_str).split() if _t369_norm(p)]
    lines = [[int(ch) for ch in p if ch.isdigit()] for p in parts if any(ch.isdigit() for ch in p)]
    if not lines:
        return {
            "VTX": 0.0, "FR": 0.0, "U": 0.0,
            "note": "【流れ未循環】ラインなし → ケン",
            "waves": {}, "vtx_bid": "", "lines": [], "dbg": {},
            "FR_line": [], "VTX_line": [], "U_line": []
        }

    buckets = _t369_buckets(lines)
    bucket_to_members = {buckets[ln[0]]: ln for ln in lines}

    def mean(xs, d=0.0):
        try:
            return sum(xs) / len(xs) if xs else d
        except Exception:
            return d

    def avg_score(mem):
        # v259: ライン人数の単純加点を排除し、位置係数で正規化した同一強度を使用
        return _t369_line_core_strength(mem, scores)

    muA = mean([avg_score(ln) for ln in lines], 50.0) / 100.0
    star_id = marks.get("◎", -999)
    none_id = marks.get("無", -999)

    def est(mem):
        A = max(10.0, min(avg_score(mem), 90.0)) / 100.0
        if star_id in mem:
            phi0, d = -0.8, +1
        elif none_id in mem:
            phi0, d = +0.8, -1
        else:
            phi0, d = +0.2, +1
        phi = phi0 + 1.2 * (A - muA)
        return A, phi, d

    def S_end(A, phi, t=0.9, f=0.9, gamma=0.12):
        return A * math.exp(-gamma * t) * (
            2 * math.pi * f * math.cos(2 * math.pi * f * t + phi)
            - gamma * math.sin(2 * math.pi * f * t + phi)
        )

    waves = {}
    for bid, mem in bucket_to_members.items():
        A, phi, d = est(mem)
        waves[bid] = {"A": A, "phi": phi, "d": d, "S": S_end(A, phi, t=0.9)}

    def I(bi, bj):
        if not bi or not bj or bi not in waves or bj not in waves:
            return 0.0
        return math.cos(waves[bi]["phi"] - waves[bj]["phi"])

    # ★v259 順流/逆流：人数ではなく正規化ライン強度で決める
    def line_strength(bid: str) -> float:
        mem = bucket_to_members.get(bid, [])
        return float(_t369_line_core_strength(mem, scores))

    all_buckets = list(bucket_to_members.keys())
    b_star = max(all_buckets, key=lambda bid: (line_strength(bid), bid))
    cand_buckets = [bid for bid in all_buckets if bid != b_star]
    b_none = min(cand_buckets, key=lambda bid: (line_strength(bid), bid)) if cand_buckets else ""

    # --- VTX ---
    vtx_list = []
    for bid, mem in bucket_to_members.items():
        if bid in (b_star, b_none):
            continue
        if waves.get(bid, {}).get("S", -1e9) < -0.02:
            continue
        wA = 0.5 + 0.5 * waves[bid]["A"]
        v = (0.6 * abs(I(bid, b_star)) + 0.4 * abs(I(bid, b_none))) * wA
        vtx_list.append((v, bid))
    vtx_list.sort(reverse=True, key=lambda x: x[0])
    VTX = vtx_list[0][0] if vtx_list else 0.0
    VTX_bid = vtx_list[0][1] if vtx_list else ""

    # --- FR ---
    ws, wn = waves.get(b_star, {}), waves.get(b_none, {})

    def S_point(w, t=0.95, f=0.9, gamma=0.12):
        if not w:
            return 0.0
        A, phi = w.get("A", 0.0), w.get("phi", 0.0)
        return A * math.exp(-gamma * t) * (
            2 * math.pi * f * math.cos(2 * math.pi * f * t + phi)
            - gamma * math.sin(2 * math.pi * f * t + phi)
        )

    blend_star = 0.6 * S_point(ws) + 0.4 * ws.get("S", 0.0)
    blend_none = 0.6 * S_point(wn) + 0.4 * wn.get("S", 0.0)

    def sig(x, k=3.0):
        try:
            return 1.0 / (1.0 + math.exp(-k * x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    sd_raw = (sig(-blend_star, 3.0) - 0.5) * 2.0
    nu_raw = (sig(blend_none, 3.0) - 0.5) * 2.0
    sd = max(0.0, sd_raw)
    nu = max(0.05, nu_raw)
    FR = sd * nu

    # --- U ---
    vtx_vals = [v for v, _ in vtx_list] or [0.0]
    vtx_mu = _t369_safe_mean(vtx_vals, 0.0)
    vtx_sd = (_t369_safe_mean([(x - vtx_mu) ** 2 for x in vtx_vals], 0.0)) ** 0.5
    vtx_hi = max(0.60, vtx_mu + 0.35 * vtx_sd)
    VTX_high = 1.0 if VTX >= vtx_hi else 0.0

    S_max = max(1e-6, max(abs(w["S"]) for w in waves.values()))
    S_noneN = max(0.0, wn.get("S", 0.0)) / S_max
    U_raw = sig(I(b_none, b_star), k=2.0)
    U = max(0.05, (0.6 * U_raw + 0.4 * S_noneN) * (1.0 if VTX_high > 0 else 0.8))

    def label(bid):
        mem = bucket_to_members.get(bid, [])
        return "".join(map(str, mem)) if mem else "—"

    note = "\n".join([
        f"【順流】◎ライン {label(b_star)}：失速危険 {'高' if FR >= 0.15 else ('中' if FR >= 0.05 else '低')}",
        f"【渦】候補ライン：{label(VTX_bid)}（VTX={VTX:.2f}）",
        f"【逆流】無ライン {label(b_none)}：U={U:.2f}（※判定基準内）",
    ])

    dbg = {
        "blend_star": blend_star,
        "blend_none": blend_none,
        "sd": sd,
        "nu": nu,
        "vtx_hi": vtx_hi,
        "line_strength_method": "normalized_role_weighted_average",
        "line_strengths": {
            label(bid): round(float(line_strength(bid)), 6)
            for bid in all_buckets
        },
    }

    # ★パッチ2：内部で使ったラインを返す
    def members_of(bid: str) -> list[int]:
        return list(bucket_to_members.get(bid, []) or [])

    FR_line = members_of(b_star)
    VTX_line = members_of(VTX_bid)
    U_line = members_of(b_none)

    return {
        "VTX": VTX,
        "FR": FR,
        "U": U,
        "note": note,
        "waves": waves,
        "vtx_bid": VTX_bid,
        "lines": lines,
        "dbg": dbg,
        "FR_line": FR_line,
        "VTX_line": VTX_line,
        "U_line": U_line,
    }


# === v2.3: 相手4枠ロジック（3車厚め“強制保証”＋3番手保証(帯)＋U高域でも最大2枚まで許容）===


# === /v2.3 ===


# === PATCH（generate_tesla_bets の直前に挿入）==============================
# 前提：ファイル上部に import re があるならここでは不要（無ければ追加）
# 前提：typing を上で import 済みならここでは不要（無ければ追加）

# 軸選定用（generate_tesla_bets から呼ばれる）
# ---- 相手4枠ロジック v2.3（3車厚め“強制保証”＋3番手保証(帯)＋U高域でも最大2枚許容）----
# === /PATCH ==============================================================


# ======================= T369｜FREE-ONLY 完全置換ブロック（精簡版） =======================

# ---- 小ヘルパ（ローカル名で衝突回避） -----------------------------------------
def _free_fmt_nums(arr):
    if isinstance(arr, list):
        return "".join(str(x) for x in arr) if arr else "—"
    return "—"

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

# ============================================================
# /T369｜FREE-ONLY 出力一括ブロック（券種コード完全撤去 + 0.000連発対策 + KO統一）
# ============================================================

def _normalize_lines(_lines):
    """
    入力 lines を必ず [[2,4],[5,7,1]...] の形にする
    - "24" / 24 / [24] / [2,4] どれでもOK（数字だけ抜いて桁分解）
    """
    out = []
    for ln in (_lines or []):
        if ln is None:
            continue
        s = "".join(ch for ch in str(ln) if ch.isdigit())
        if not s:
            continue
        out.append([int(ch) for ch in s])
    return out

# --- v282：ラインFRは「ライン上位2車合計／単騎2倍」の2車換算勢力で作る ---
def _build_line_fr_map_v282(lines, scores_map, FRv):
    normalized_lines = _normalize_lines(lines)
    FRv = float(FRv or 0.0)
    if not normalized_lines:
        return {}

    strength_map = _build_line_two_car_strength_map(normalized_lines, scores_map)
    total = sum(float(value or 0.0) for value in strength_map.values())
    sum_target = FRv if FRv > 0.0 else 1.0

    if total <= 0.0:
        equal_share = sum_target / len(normalized_lines)
        return {
            "".join(map(str, line)): equal_share
            for line in normalized_lines
        }

    return {
        key: sum_target * (float(value or 0.0) / total)
        for key, value in strength_map.items()
    }

# ---------- 3) 安全ラッパ（券種なし：flowだけ） ----------
def _safe_flow(lines_str, marks, scores):
    try:
        fr = compute_flow_indicators(lines_str, marks, scores)
        return fr if isinstance(fr, dict) else {}
    except Exception:
        return {}


def _v265_compact_line_zone_representatives(zones, fmt_line=None):
    """
    順流域・渦域・逆流域を各1代表へ圧縮し、余剰ラインを
    「その他（3列目候補）」として返す。

    前段のゾーン判定・v235の3枠補完で確定した並び順を尊重し、
    各ゾーンの先頭だけを代表に残す。余剰ラインは流れ比率へ加算しない。
    """
    zone_names = ("順流域", "渦域", "逆流域")
    compacted = {name: list((zones or {}).get(name, []) or []) for name in zone_names}
    other_items = []

    for zone_name in zone_names:
        items = list(compacted.get(zone_name, []) or [])
        if not items:
            compacted[zone_name] = []
            continue

        compacted[zone_name] = [items[0]]
        origin = zone_name.replace("域", "")
        for item in items[1:]:
            copied = dict(item or {})
            copied["origin_zone"] = origin
            other_items.append(copied)

    def _line_key(item):
        return "".join(ch for ch in str((item or {}).get("line", "")) if ch.isdigit())

    representative_keys = {
        _line_key(item)
        for zone_name in zone_names
        for item in (compacted.get(zone_name, []) or [])
        if _line_key(item)
    }

    unique_other = []
    seen = set()
    for item in other_items:
        key = _line_key(item)
        if not key or key in representative_keys or key in seen:
            continue
        seen.add(key)
        unique_other.append(item)

    def _fmt(item):
        line = (item or {}).get("line", [])
        if callable(fmt_line):
            try:
                return str(fmt_line(line))
            except Exception:
                pass
        return "".join(ch for ch in str(line) if ch.isdigit())

    unique_other.sort(
        key=lambda item: (
            -float((item or {}).get("fr", 0.0) or 0.0),
            _fmt(item),
        )
    )
    return compacted, unique_other

# ===================== 4) 出力本体（券種なし・一括置換） =====================
try:
    import math

    # --- note_sections を必ず用意 ---
    ns = globals().get("note_sections", None)
    if not isinstance(ns, list):
        ns = []
        globals()["note_sections"] = ns
    note_sections = ns

    # ---- flow 作成 ----
    _flow = _safe_flow(
        globals().get("lines_str", ""),
        globals().get("marks", {}),
        globals().get("scores", {}),
    )
    globals()["_flow"] = _flow  # 後段参照用に保持

    # ---- 値の確定 ----
    FRv = float(_flow.get("FR", 0.0) or 0.0)
    VTXv = float(_flow.get("VTX", 0.0) or 0.0)
    Uv = float(_flow.get("U", 0.0) or 0.0)

    all_lines = list(_flow.get("lines") or [])
    all_lines = _normalize_lines(all_lines)  # ここで必ず正規化
    globals()["all_lines"] = all_lines

    # ---- レース名 ----
    venue = str(globals().get("track") or globals().get("place") or "").strip()
    race_no = str(globals().get("race_no") or "").strip()
    if venue or race_no:
        _rn = race_no if (race_no.endswith("R") or race_no == "") else f"{race_no}R"
        note_sections.append(f"{venue}{_rn}")
        note_sections.append("")

    # =========================================================
    # KO母集団スコア（v_final > v_wo > scores）で統一
    # =========================================================
    def _as_int_float_map(m):
        out = {}
        if not isinstance(m, dict):
            return out
        for k, v in m.items():
            try:
                kk = int(k)
                vv = float(v)
                if math.isfinite(vv):
                    out[kk] = vv
            except Exception:
                pass
        return out

    v_final_map = _as_int_float_map(globals().get("v_final"))
    v_wo_map = _as_int_float_map(globals().get("v_wo"))
    scores_map = _as_int_float_map(globals().get("scores"))

    score_map = dict(v_final_map or v_wo_map or scores_map or {})

    # active_cars を必ず含める（欠けを防ぐ）
    active_cars = [int(x) for x in (globals().get("active_cars") or []) if str(x).isdigit()]
    for n in active_cars:
        score_map.setdefault(int(n), 0.0)

   

        # =========================================================
    # KO母集団スコア補正：ライン3番手以降・H0/B0の過大評価抑制
    # ※脚質名に依存しない版。「追」ではなく「マーク」扱いでも効く。
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _H = globals().get("H", {})
        _B = globals().get("B", {})

        for _n in list(score_map.keys()):
            _car = int(_n)

            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"

            _h_val = float(_H.get(_car, _H.get(str(_car), 0)) or 0)
            _b_val = float(_B.get(_car, _B.get(str(_car), 0)) or 0)

            # 例：364 の 4番 = thirdplus、H0、B0 → 必ず減点
            if _role == "thirdplus":
                if _h_val == 0 and _b_val == 0:
                    score_map[_n] = float(score_map[_n]) - 0.15
                else:
                    score_map[_n] = float(score_map[_n]) - 0.08

    except Exception as _e:
        note_sections.append(f"※KO母集団補正エラー：{_e}")

    score_map_before_last_half = dict(score_map)
    globals()["score_map_before_last_half"] = dict(score_map_before_last_half)

    # =========================================================
    # ラスト半周補正：自力粘り・番手差し
    # ※既存のKO母集団スコアに後付けする
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _H = globals().get("H", {})
        _B = globals().get("B", {})
        _kaku = globals().get("kaku", {})
        _tenscore = globals().get("tenscore", globals().get("tenscores", {}))

        # 競走得点の取り出し
        def _get_num_from_map(_mp, _car, _default=0.0):
            try:
                if isinstance(_mp, dict):
                    return float(_mp.get(int(_car), _mp.get(str(_car), _default)) or _default)
            except Exception:
                pass
            return float(_default)

        _race_scores = []
        for _n in active_cars:
            _v = _get_num_from_map(_tenscore, _n, 0.0)
            if _v > 0:
                _race_scores.append(_v)

        _race_avg_tenscore = float(np.mean(_race_scores)) if _race_scores else 0.0
        _last_half_bonus_map = {}
        _last_half_reason_map = {}
        
                # -------------------------------------------------
        # ラスト半周補正用：レース内順位マップ
        # 上位1/3判定用。7車なら3位以内。
        # -------------------------------------------------
        _active_list = [int(x) for x in active_cars]
        _top_third_limit = int(math.ceil(len(_active_list) / 3.0)) if _active_list else 3
        _top_third_limit = max(1, _top_third_limit)

        # 競走得点順位
        _race_score_rank_map = {}
        _ten_pairs = []
        for _n in _active_list:
            _v = _get_num_from_map(_tenscore, _n, 0.0)
            _ten_pairs.append((int(_n), float(_v)))

        _ten_pairs_sorted = sorted(_ten_pairs, key=lambda x: (-x[1], x[0]))
        for _idx, (_car2, _v2) in enumerate(_ten_pairs_sorted, start=1):
            _race_score_rank_map[int(_car2)] = int(_idx)

        # KO順位・展開順位
        # この時点の score_map_before_last_half は「ラスト半周補正前」のスコア
        _ko_score_rank_map = {}
        _ko_pairs_sorted = sorted(
            [(int(k), float(v)) for k, v in score_map_before_last_half.items()],
            key=lambda x: (-x[1], x[0])
        )
        for _idx, (_car2, _v2) in enumerate(_ko_pairs_sorted, start=1):
            _ko_score_rank_map[int(_car2)] = int(_idx)

        _tenkai_score_rank_map = dict(_ko_score_rank_map)

        # 順流・渦・逆流の複数上位は次段階用
        _scenario_top_count_map = globals().get("scenario_top_count_map", {})
        if not isinstance(_scenario_top_count_map, dict):
            _scenario_top_count_map = {}

        for _n in list(score_map.keys()):
            _car = int(_n)

            # ライン内の役割
            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"

            # 同ライン先頭の競走得点
            _leader = _car
            try:
                if isinstance(_line_def, dict):
                    for _gid, _mem in _line_def.items():
                        _mem2 = [int(x) for x in _mem]
                        if _car in _mem2 and _mem2:
                            _leader = int(_mem2[0])
                            break
            except Exception:
                _leader = _car

            _car_ten = _get_num_from_map(_tenscore, _car, 0.0)
            _leader_ten = _get_num_from_map(_tenscore, _leader, _car_ten)

            _h_val = _get_num_from_map(_H, _car, 0.0)
            _b_val = _get_num_from_map(_B, _car, 0.0)

            # kakuは現在の入力仕様では使わない。
            # 関数互換用に空文字で渡す。
            _style = ""

            # H主導ラインの3番手以降かどうか
            _is_h_lead_thirdplus = False
            try:
                _h_members = []
                if home_top_gid is not None and isinstance(_line_def, dict):
                    _h_members = [int(x) for x in _line_def.get(home_top_gid, [])]

                if (
                    len(_h_members) >= 3
                    and _role == "thirdplus"
                    and _car in _h_members[2:]
                ):
                    _is_h_lead_thirdplus = True

            except Exception:
                _is_h_lead_thirdplus = False

            # ---------------------------------------------
            # ラスト半周用：個人成績率
            # x1 / x2 / x3 / x_out から
            # 1着率・2着内率・3着内率を作る
            # ---------------------------------------------
            _p1_rate = None
            _p2_rate = None
            _p3_rate = None

            try:
                _x1 = globals().get("x1", {})
                _x2 = globals().get("x2", {})
                _x3 = globals().get("x3", {})
                _xo = globals().get("x_out", {})

                _n1 = float(_x1.get(_car, _x1.get(str(_car), 0)) or 0)
                _n2 = float(_x2.get(_car, _x2.get(str(_car), 0)) or 0)
                _n3 = float(_x3.get(_car, _x3.get(str(_car), 0)) or 0)
                _no = float(_xo.get(_car, _xo.get(str(_car), 0)) or 0)

                _total = _n1 + _n2 + _n3 + _no

                if _total > 0:
                    _p1_rate = _n1 / _total
                    _p2_rate = (_n1 + _n2) / _total
                    _p3_rate = (_n1 + _n2 + _n3) / _total

            except Exception:
                _p1_rate = None
                _p2_rate = None
                _p3_rate = None

            _bonus, _reasons = calc_last_half_role_bonus(
                role=_role,
                kaku=_style,
                tenscore=_car_ten,
                leader_tenscore=_leader_ten,
                race_avg_tenscore=_race_avg_tenscore,
                h_count=_h_val,
                b_count=_b_val,
                race_score_rank=_race_score_rank_map.get(_car),
                ko_score_rank=_ko_score_rank_map.get(_car),
                tenkai_score_rank=_tenkai_score_rank_map.get(_car),
                top_third_limit=_top_third_limit,
                scenario_top_count=int(_scenario_top_count_map.get(_car, 0) or 0),
                p1_rate=_p1_rate,
                p2_rate=_p2_rate,
                p3_rate=_p3_rate,
            )

            _last_half_bonus_map[_car] = float(_bonus)
            _last_half_reason_map[_car] = list(_reasons)

            score_map[_car] = float(score_map.get(_car, 0.0)) + float(_bonus)

    


            # -------------------------------------------------
        # H主導ライン3番手以降：3着内率40%以上なら最低4番手評価まで床上げ
        # -------------------------------------------------
        THIRDPLUS_TOP3_RATE_MIN = 0.40
        THIRDPLUS_FLOOR_RANK = 4
        THIRDPLUS_FLOOR_EPS = 0.001

        def _normalize_rate_0to1(v):
            try:
                x = float(v)
                if x > 1.0:
                    x = x / 100.0
                return x
            except Exception:
                return None

        def _get_top3_rate_for_car(_car_no):
            """
            車番ごとの3着内率を取得する。
            変数名が多少違っても拾えるように、候補名とglobals内のdictを探す。
            値は 0.40 / 40.0 のどちらでも対応。
            """
            _car_no = int(_car_no)

            # よくありそうな名前を優先
            _candidate_names = [
                "top3_rate_map",
                "in3_rate_map",
                "pTop3_map",
                "ptop3_map",
                "car_top3_rate_map",
                "car_in3_rate_map",
                "top3_map",
                "in3_map",
                "P_TOP3_MAP",
                "IN3_RATE_MAP",
            ]

            for _name in _candidate_names:
                _obj = globals().get(_name, None)
                if isinstance(_obj, dict):
                    _v = _obj.get(_car_no, _obj.get(str(_car_no), None))
                    _r = _normalize_rate_0to1(_v)
                    if _r is not None:
                        return _r

            # 名前が違う場合の保険：globals内の「top3 / in3 / 3着」系dictを探索
            try:
                for _name, _obj in globals().items():
                    _lname = str(_name).lower()
                    if not isinstance(_obj, dict):
                        continue

                    if not (
                        "top3" in _lname
                        or "in3" in _lname
                        or "p_top3" in _lname
                        or "3着" in str(_name)
                        or "三着" in str(_name)
                    ):
                        continue

                    _v = _obj.get(_car_no, _obj.get(str(_car_no), None))
                    _r = _normalize_rate_0to1(_v)
                    if _r is not None:
                        return _r
            except Exception:
                pass

            return None



        globals()["last_half_bonus_map"] = _last_half_bonus_map
        globals()["last_half_reason_map"] = _last_half_reason_map
        globals()["score_map_last_half_applied"] = dict(score_map)

    except Exception as _e:
        note_sections.append(f"※ラスト半周補正エラー：{_e}")

    # =========================================================
    # 会場成績 × 最終ホームライン補正（買い目用スコア）
    # H1番手ラインはイン減速で減点、H2番手ラインは外スピードで加点
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _car_to_group = globals().get("car_to_group", {})
        _track = globals().get("track", st.session_state.get("track", ""))
        _venue_profile = globals().get("venue_profile", st.session_state.get("venue_profile", "unknown"))
        _home_top_gid = globals().get("home_top_gid", None)
        _home_second_gid = globals().get("home_second_gid", None)

        _home_flow_bonus_map = {}
        _home_flow_reason_map = {}
        _before_home_flow_map = dict(score_map)

        for _n in list(score_map.keys()):
            _car = int(_n)
            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"
            _gid = _car_to_group.get(_car, None) if isinstance(_car_to_group, dict) else None

            _hf_bonus, _hf_reason = home_flow_adjust_by_venue(
                no=_car,
                role=_role,
                gid=_gid,
                home_top_gid=_home_top_gid,
                home_second_gid=_home_second_gid,
                track_name=_track,
                venue_profile=_venue_profile,
            )

            _home_flow_bonus_map[_car] = float(_hf_bonus)
            _home_flow_reason_map[_car] = str(_hf_reason)

            score_map[_car] = float(score_map.get(_car, 0.0)) + float(_hf_bonus)

        globals()["home_flow_bonus_map"] = dict(_home_flow_bonus_map)
        globals()["home_flow_reason_map"] = dict(_home_flow_reason_map)
        globals()["score_map_before_home_flow"] = dict(_before_home_flow_map)
        globals()["score_map_home_flow_applied"] = dict(score_map)

    except Exception as _e:
        note_sections.append(f"※会場×最終H補正エラー：{_e}")

    # =========================================================
    # v178：開催場決まり手補正（常時適用・雨天補正とは別枠）
    # 入力された1着/2着決まり手率を、役割別の小幅ptへ変換して加算。
    # =========================================================
    try:
        _vk_stats = globals().get("VENUE_KIMARITE_STATS", st.session_state.get("VENUE_KIMARITE_STATS", {}))
        _line_def_for_vk = globals().get("line_def", {})
        _before_vk_score_map = dict(score_map)

        score_map, _vk_role_bonus_map, _vk_reliability, _vk_detail, _vk_reason_map = _apply_venue_kimarite_to_score_map(
            score_map=score_map,
            line_def=_line_def_for_vk,
            stats=_vk_stats,
        )

        globals()["score_map_before_venue_kimarite"] = dict(_before_vk_score_map)
        globals()["score_map_venue_kimarite_applied"] = dict(score_map)
        globals()["venue_kimarite_role_bonus_map"] = dict(_vk_role_bonus_map)
        globals()["venue_kimarite_reliability"] = float(_vk_reliability)
        globals()["venue_kimarite_detail"] = dict(_vk_detail or {})
        globals()["venue_kimarite_reason_map"] = dict(_vk_reason_map)

    except Exception as _e:
        note_sections.append(f"※開催場決まり手補正エラー：{_e}")

    # 0/None/NaN の床値補完
    vals_pos = [
        float(v) for v in score_map.values()
        if isinstance(v, (int, float)) and float(v) > 0.0 and math.isfinite(float(v))
    ]

    _floor = min(vals_pos) if vals_pos else 1e-6

    for k in list(score_map.keys()):
        try:
            v = float(score_map[k])
            if (not math.isfinite(v)) or v <= 0.0:
                score_map[k] = float(_floor)
        except Exception:
            score_map[k] = float(_floor)

    globals()["score_map"] = score_map  # 後段参照用に保持

    # =========================================================
    # v282：ライン／単騎を2車換算して line_fr_map を毎回再構築
    # ・2車以上のライン＝ライン内KO使用スコア上位2車の合計
    # ・単騎＝本人のKO使用スコア×2
    # =========================================================
    try:
        line_two_car_strength_map = _build_line_two_car_strength_map(
            all_lines,
            score_map,
        )
        line_fr_map = _build_line_fr_map_v282(
            all_lines,
            score_map,
            FRv if FRv > 0.0 else 1.0,
        )
    except Exception:
        line_two_car_strength_map = {}
        line_fr_map = {}

    globals()["LINE_TWO_CAR_STRENGTH_MAP"] = dict(line_two_car_strength_map)
    globals()["line_fr_map"] = dict(line_fr_map)

    def _line_key(ln):
        try:
            if not ln:
                return ""
            return "".join(str(int(x)) for x in ln if str(x).isdigit())
        except Exception:
            return "".join(ch for ch in str(ln) if ch.isdigit())

    def _lfr(ln):
        try:
            return float(line_fr_map.get(_line_key(ln), 0.0) or 0.0)
        except Exception:
            return 0.0
    # =========================================================
    # 展開評価（share_pct は「順流ライン」基準）
    # =========================================================
    FR_line = _flow.get("FR_line") or []
    VTX_line = _flow.get("VTX_line") or []
    U_line = _flow.get("U_line") or []

    FR_line = _normalize_lines([FR_line])[0] if FR_line else []
    VTX_line = _normalize_lines([VTX_line])[0] if VTX_line else []
    U_line = _normalize_lines([U_line])[0] if U_line else []

    globals()["FR_line"] = FR_line
    globals()["VTX_line"] = VTX_line
    globals()["U_line"] = U_line

    # =========================================================
    # 渦ラインを必ず埋める（空なら自動選定）
    # ルール：FR_line / U_line 以外で、想定FRが最大のラインを渦にする
    # =========================================================
    if (not VTX_line) or (_lfr(VTX_line) <= 0.0):
        _cand = []
        for ln in (all_lines or []):
            if not ln:
                continue
            if ln == FR_line or ln == U_line:
                continue
            _cand.append(ln)
        if _cand:
            VTX_line = max(_cand, key=lambda x: _lfr(x))
            globals()["VTX_line"] = VTX_line

    axis_line = FR_line if FR_line else (all_lines[0] if all_lines else [])
    axis_line_fr = float(line_fr_map.get(_line_key(axis_line), 0.0) or 0.0)
    total_fr = sum(float(v or 0.0) for v in line_fr_map.values()) if isinstance(line_fr_map, dict) else 0.0
    share_pct = (axis_line_fr / total_fr * 100.0) if (total_fr > 1e-12 and axis_line) else None

    note_sections.append(f"展開評価：{infer_eval_with_share(FRv, VTXv, Uv, share_pct)}")
    note_sections.append("")

    # ---- 時刻・クラス ----
    race_time = str(globals().get("race_time", "") or "")
    race_class = str(globals().get("race_class", "") or "")
    hdr = f"{race_time}　{race_class}".strip()
    if hdr:
        note_sections.append(hdr)

        # ---- ライン表示 ----
    line_inputs = globals().get("line_inputs", [])
    if isinstance(line_inputs, list) and any(str(x).strip() for x in line_inputs):
        _lines = [str(x).strip() for x in line_inputs if str(x).strip()]
        note_sections.append("ライン　" + "　".join(_lines))

        # H：最終ホーム想定ライン
        try:
            note_sections.append(f"最終ホーム想定　{home_line_text}")
            note_sections.append(f"H主導ライン　{home_top_line}")
        except Exception:
            pass

    note_sections.append("")

    # =========================================================
    # ライン想定FR（順流/渦/逆流 + その他）表示  ※区分け復活
    # =========================================================
    def _fmt_line(ln):
        try:
            f = globals().get("_free_fmt_nums")
            if callable(f):
                return f(ln)
        except Exception:
            pass
        return "".join(map(str, ln)) if isinstance(ln, (list, tuple)) and ln else "—"

        # =========================================================
    # ライン評価グループ（順流域／渦域／逆流域）
    # =========================================================
    def _fmt_line(ln):
        try:
            f = globals().get("_free_fmt_nums")
            if callable(f):
                return f(ln)
        except Exception:
            pass
        return "".join(map(str, ln)) if isinstance(ln, (list, tuple)) and ln else "—"

    def _same_line(a, b):
        return tuple(int(x) for x in (a or [])) == tuple(int(x) for x in (b or []))

    try:
        h_line_members = line_def.get(home_top_gid, []) if home_top_gid is not None else []
    except Exception:
        h_line_members = []

    valid_lines = [ln for ln in (all_lines or []) if ln]
    line_items = []

    for ln in valid_lines:
        fr = float(_lfr(ln))
        line_items.append({
            "line": ln,
            "fr": fr,
            "is_fr": _same_line(ln, FR_line),
            "is_vtx": _same_line(ln, VTX_line),
            "is_u": _same_line(ln, U_line),
            "is_h": _same_line(ln, h_line_members),
        })

    line_items = sorted(line_items, key=lambda x: (-x["fr"], _fmt_line(x["line"])))

    if line_items:
        top_fr = float(line_items[0]["fr"])

        # FR差による範囲判定
        # 7車以下はやや狭め、8・9車は広め
        if int(n_cars) >= 8:
            upper_gap = 0.080
            middle_ratio = 0.45
            h_gap = 0.150
        else:
            upper_gap = 0.050
            middle_ratio = 0.45
            h_gap = 0.090

        zones = {
            "順流域": [],
            "渦域": [],
            "逆流域": [],
        }

        for item in line_items:
            ln = item["line"]
            fr = float(item["fr"])
            gap = top_fr - fr
            ratio = (fr / top_fr) if top_fr > 1e-12 else 0.0

            tags = []
            if item["is_fr"]:
                tags.append("◎")
            if item["is_h"]:
                tags.append("H主導")
            if item["is_vtx"]:
                tags.append("旧渦")
            if item["is_u"]:
                tags.append("旧逆流")

            # 順流域：
            # FRトップ、またはFRトップとの差が小さいライン
            if item["is_fr"] or gap <= upper_gap:
                zone = "順流域"

            # H主導ラインは、FR2位級なら実質上位へ寄せる
            elif item["is_h"] and (gap <= h_gap or ratio >= 0.55):
                zone = "順流域"
                tags.append("実質上位")

            # 中位以上の別線は渦域
            elif ratio >= middle_ratio:
                zone = "渦域"

            # 低FR・単騎・押上げ側は逆流域
            else:
                zone = "逆流域"

            sort_score = fr + (0.030 if item["is_h"] else 0.0)

            zones[zone].append({
                "line": ln,
                "fr": fr,
                "tags": tags,
                "sort_score": sort_score,
            })

        for z in zones:
            zones[z] = sorted(
                zones[z],
                key=lambda x: (-x["sort_score"], -x["fr"], _fmt_line(x["line"]))
            )

        # =====================================================
        # v164: 順流域は必ず代表1ラインだけにする
        # 目的：157 と 24 のように複数ラインが同じ順流域へ入り、
        #       KO隊列で 15724 を1塊のように混ぜてしまう現象を防ぐ。
        #       単騎も1ラインとして扱う。
        # =====================================================
        try:
            jun_items = list(zones.get("順流域", []))
            if len(jun_items) > 1:
                # ◎ラインを最優先。なければ現在のソート順トップを順流代表にする。
                fr_items = [x for x in jun_items if "◎" in x.get("tags", [])]
                keep_jun = fr_items[0] if fr_items else jun_items[0]
                overflow = [x for x in jun_items if x is not keep_jun]

                zones["順流域"] = [keep_jun]

                # 余った順流候補は、まず渦域へ1本、残りは逆流域へ回す。
                # 既に渦域がある場合は、渦域へ追加してFR順で再ソートする。
                if overflow:
                    zones.setdefault("渦域", [])
                    zones.setdefault("逆流域", [])

                    if not zones.get("渦域"):
                        zones["渦域"].append(overflow[0])
                        zones["逆流域"].extend(overflow[1:])
                    else:
                        zones["渦域"].extend(overflow)

                    for _z in ("渦域", "逆流域"):
                        zones[_z] = sorted(
                            zones.get(_z, []),
                            key=lambda x: (-x["sort_score"], -x["fr"], _fmt_line(x["line"]))
                        )
        except Exception:
            pass

                # =====================================================
        # 全ラインが順流域に吸収された場合の強制分割
        # 目的：順流・渦・逆流メインが全部同じになるのを防ぐ
        # =====================================================
        try:
            if (
                len(zones.get("順流域", [])) >= 3
                and len(zones.get("渦域", [])) == 0
                and len(zones.get("逆流域", [])) == 0
            ):
                all_top_items = list(zones["順流域"])

                # まずFR順で並べる
                all_top_items = sorted(
                    all_top_items,
                    key=lambda x: (-float(x["fr"]), _fmt_line(x["line"]))
                )

                # ◎ラインは順流域に残す
                fr_items = [x for x in all_top_items if "◎" in x.get("tags", [])]

                if fr_items:
                    keep_jun = fr_items[0]
                else:
                    keep_jun = all_top_items[0]

                rest = [x for x in all_top_items if x is not keep_jun]

                # 残りの中でFR最上位を渦域へ
                rest = sorted(
                    rest,
                    key=lambda x: (-float(x["fr"]), _fmt_line(x["line"]))
                )

                keep_vtx = rest[0] if rest else None
                rest2 = [x for x in rest if x is not keep_vtx]

                zones["順流域"] = [keep_jun]
                zones["渦域"] = [keep_vtx] if keep_vtx is not None else []
                zones["逆流域"] = rest2

        except Exception:
            pass

        


        # =====================================================
        # v235: 順流・渦・逆流は必ず3枠に割り振る
        # 目的：ライン評価グループで逆流域が空なのに、流れ比率だけ逆流100%になる矛盾を防ぐ。
        # ・旧逆流タグを持つラインは逆流域の補完候補として最優先
        # ・旧渦タグを持つラインは渦域の補完候補として最優先
        # ・3ライン以上ある場合、表示上も内部比率上も3枠を空にしない
        # =====================================================
        try:
            _zone_names = ["順流域", "渦域", "逆流域"]
            for _z in _zone_names:
                zones.setdefault(_z, [])

            def _move_one_zone(_from, _to, _prefer_tag=None):
                try:
                    _items = list(zones.get(_from, []) or [])
                    if len(_items) <= 1:
                        return False
                    _idx = None
                    if _prefer_tag:
                        for _i, _it in enumerate(_items):
                            if _prefer_tag in (_it.get("tags", []) or []):
                                _idx = _i
                                break
                    if _idx is None:
                        # FRが低いものほど逆流/補完側へ回しやすい。
                        _idx = min(range(len(_items)), key=lambda i: (float(_items[i].get("fr", 0.0) or 0.0), _fmt_line(_items[i].get("line"))))
                    _item = _items.pop(_idx)
                    zones[_from] = _items
                    zones.setdefault(_to, [])
                    zones[_to].append(_item)
                    zones[_to] = sorted(
                        zones.get(_to, []),
                        key=lambda x: (-x.get("sort_score", 0.0), -float(x.get("fr", 0.0) or 0.0), _fmt_line(x.get("line")))
                    )
                    return True
                except Exception:
                    return False

            _all_zone_count = sum(len(zones.get(_z, []) or []) for _z in _zone_names)
            if _all_zone_count >= 3:
                # 逆流域が空なら、旧逆流タグを持つ渦域ラインを最優先で逆流域へ戻す。
                if not zones.get("逆流域"):
                    if not _move_one_zone("渦域", "逆流域", "旧逆流"):
                        _move_one_zone("順流域", "逆流域", "旧逆流")

                # 渦域が空なら、旧渦タグを持つ逆流域ラインを最優先で渦域へ戻す。
                if not zones.get("渦域"):
                    if not _move_one_zone("逆流域", "渦域", "旧渦"):
                        _move_one_zone("順流域", "渦域", "旧渦")

                # 順流域が空になる異常時だけ、最大FRのラインを順流域へ補完する。
                if not zones.get("順流域"):
                    _donors = [z for z in ("渦域", "逆流域") if len(zones.get(z, []) or []) > 1]
                    if _donors:
                        _from = max(_donors, key=lambda z: max(float(x.get("fr", 0.0) or 0.0) for x in zones.get(z, []) or []))
                        _items = list(zones.get(_from, []) or [])
                        _idx = max(range(len(_items)), key=lambda i: float(_items[i].get("fr", 0.0) or 0.0))
                        _item = _items.pop(_idx)
                        zones[_from] = _items
                        zones["順流域"] = [_item]

            # 3枠確定後のFR比率を保存。以後の流れ想定比率はこの表示分類を優先する。
            _zone_fr = {
                "順流": sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("順流域", []) or [])),
                "渦":   sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("渦域", []) or [])),
                "逆流": sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("逆流域", []) or [])),
            }
            _zone_total = sum(_zone_fr.values())
            if _zone_total > 0:
                globals()["FLOW_RATIO_MAP_BY_ZONE"] = {
                    "順流": _zone_fr["順流"] / _zone_total,
                    "逆流": _zone_fr["逆流"] / _zone_total,
                    "渦": _zone_fr["渦"] / _zone_total,
                }
        except Exception:
            pass


        # =====================================================
        # v265: 各流れは代表1ラインだけ。余剰は「その他（3列目候補）」へ。
        # ・その他は流れ比率・比率順位へ加算しない
        # ・KO隊列では「その他」として末尾へ残し、3列目補強候補には使える
        # =====================================================
        _other_line_items = []
        try:
            zones, _other_line_items = _v265_compact_line_zone_representatives(
                zones,
                fmt_line=_fmt_line,
            )

            _zone_fr = {
                "順流": sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("順流域", []) or [])),
                "渦":   sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("渦域", []) or [])),
                "逆流": sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("逆流域", []) or [])),
            }
            _zone_total = sum(_zone_fr.values())
            if _zone_total > 0:
                globals()["FLOW_RATIO_MAP_BY_ZONE"] = {
                    "順流": _zone_fr["順流"] / _zone_total,
                    "逆流": _zone_fr["逆流"] / _zone_total,
                    "渦": _zone_fr["渦"] / _zone_total,
                }

            globals()["OTHER_LINE_ITEMS_FOR_THIRD"] = [
                {
                    "line": list(item.get("line", []) or []),
                    "fr": float(item.get("fr", 0.0) or 0.0),
                    "origin_zone": str(item.get("origin_zone", "") or ""),
                    "tags": list(item.get("tags", []) or []),
                }
                for item in (_other_line_items or [])
            ]
        except Exception:
            _other_line_items = []
            globals()["OTHER_LINE_ITEMS_FOR_THIRD"] = []

        # KO隊列用：ラインごとの新ゾーン分類を保存
        _LINE_ZONE_MAP = {}

        _zone_to_short = {
            "順流域": "順流",
            "渦域": "渦",
            "逆流域": "逆流",
        }

        for zone_name, items in zones.items():
            short_zone = _zone_to_short.get(zone_name, "その他")
            for item in items:
                try:
                   key = "".join(ch for ch in str(item["line"]) if ch.isdigit())
                   if key:
                       _LINE_ZONE_MAP[key] = short_zone
                except Exception:
                   pass

        # 代表外ラインは旧FR/VTX/Uの保険判定へ戻さず、明示的に「その他」とする。
        for item in (_other_line_items or []):
            try:
                key = "".join(ch for ch in str(item.get("line", "")) if ch.isdigit())
                if key:
                    _LINE_ZONE_MAP[key] = "その他"
            except Exception:
                pass

        globals()["LINE_ZONE_MAP"] = _LINE_ZONE_MAP

        # st.write("DEBUG LINE_ZONE_MAP", _LINE_ZONE_MAP)

        note_sections.append("【ライン評価グループ】")

        for zone_name in ["順流域", "渦域", "逆流域"]:
            items = zones.get(zone_name, [])
            if not items:
                note_sections.append(f"{zone_name}：—")
                continue
            parts = []
            for item in items:
                tag_txt = ""
                if item["tags"]:
                    tag_txt = "・" + "・".join(item["tags"])

                parts.append(
                    f"{_fmt_line(item['line'])}［FR={item['fr']:.3f}{tag_txt}］"
                )

            note_sections.append(f"{zone_name}：" + "／".join(parts))

        if _other_line_items:
            other_parts = []
            for item in _other_line_items:
                tags = []
                origin = str(item.get("origin_zone", "") or "")
                if origin:
                    tags.append(f"元:{origin}")
                tags.extend(str(x) for x in (item.get("tags", []) or []) if str(x))
                tag_txt = ("・" + "・".join(tags)) if tags else ""
                other_parts.append(
                    f"{_fmt_line(item.get('line', []))}［FR={float(item.get('fr', 0.0) or 0.0):.3f}{tag_txt}］"
                )
            note_sections.append("その他（3列目候補）：" + "／".join(other_parts))
        else:
            note_sections.append("その他（3列目候補）：—")

    else:
        note_sections.append("【ライン評価グループ】")
        note_sections.append("順流域：—")
        note_sections.append("渦域：—")
        note_sections.append("逆流域：—")
        note_sections.append("その他（3列目候補）：—")

    note_sections.append("")

        # =========================================================
    # ラスト半周補正 表示
    # =========================================================
    try:
        _lh_bonus_map = globals().get("last_half_bonus_map", {})
        _lh_reason_map = globals().get("last_half_reason_map", {})
        _before_map = globals().get("score_map_before_last_half", {})
        _after_map = globals().get("score_map_last_half_applied", {})

        if isinstance(_lh_bonus_map, dict) and _lh_bonus_map:
            note_sections.append("【ラスト半周補正】")

            _lh_pairs = sorted(
                [(int(k), float(v)) for k, v in _lh_bonus_map.items()],
                key=lambda t: t[0]
            )

            for _car, _bonus in _lh_pairs:
                _before = float(_before_map.get(_car, 0.0) or 0.0)
                _after = float(_after_map.get(_car, _before + _bonus) or 0.0)

                _reasons = _lh_reason_map.get(_car, [])
                if not isinstance(_reasons, list):
                    _reasons = [_reasons]

                _reason_txt = "／".join(str(x) for x in _reasons if str(x).strip())
                if not _reason_txt:
                    _reason_txt = "補正なし"

                note_sections.append(
                    f"{_car}：展開={_before:.6f} ／ 補正={_bonus:+.3f} ／ 最終={_after:.6f}［{_reason_txt}］"
                )

            note_sections.append("")

    except Exception as _e:
        note_sections.append(f"※ラスト半周補正表示エラー：{_e}")
        note_sections.append("")
    # =========================================================
    # 会場×最終Hライン補正 表示
    # =========================================================
    try:
        _hf_bonus_map = globals().get("home_flow_bonus_map", {})
        _hf_reason_map = globals().get("home_flow_reason_map", {})
        _hf_before_map = globals().get("score_map_before_home_flow", {})
        _hf_after_map = globals().get("score_map_home_flow_applied", {})

        if isinstance(_hf_bonus_map, dict) and _hf_bonus_map:
            note_sections.append("【会場×最終Hライン補正】")
            note_sections.append(
                f"会場判定={globals().get('venue_profile', 'unknown')} ／ "
                f"補正倍率={float(globals().get('venue_home_flow_mult', 1.0)):.2f} ／ "
                f"必要オッズ倍率={float(globals().get('venue_min_odds_mult', 1.0)):.2f}"
            )

            _hf_pairs = sorted(
                [(int(k), float(v)) for k, v in _hf_bonus_map.items()],
                key=lambda t: t[0]
            )

            for _car, _bonus in _hf_pairs:
                _before = float(_hf_before_map.get(_car, 0.0) or 0.0)
                _after = float(_hf_after_map.get(_car, _before + _bonus) or 0.0)
                _reason_txt = str(_hf_reason_map.get(_car, ""))
                note_sections.append(
                    f"{_car}：補正前={_before:.6f} ／ H補正={_bonus:+.3f} ／ 補正後={_after:.6f}［{_reason_txt}］"
                )

            note_sections.append("")

    except Exception as _e:
        note_sections.append(f"※会場×最終H補正表示エラー：{_e}")
        note_sections.append("")

    # =========================================================
    # KO使用スコア（降順）
    # =========================================================
    _sc_pairs = sorted(
        [(int(k), float(v)) for k, v in (score_map or {}).items()],
        key=lambda t: (-t[1], t[0])
    )
    globals()["KO_SCORE_ORDER_FOR_TIE"] = [int(n) for n, _sc in _sc_pairs]
    globals()["KO_SCORE_MAP_FOR_SANTEN"] = {int(n): float(_sc) for n, _sc in _sc_pairs}

    note_sections.append("【KO使用スコア（降順）】")

    
    if _sc_pairs:
        for i, (n, sc) in enumerate(_sc_pairs, start=1):
            note_sections.append(f"{i}位：{n} (スコア={sc:.6f})")
    else:
        note_sections.append("—")
    note_sections.append("")

    # =========================================================
    # 最終ジャン想定隊列 → KO（6パターン）
    #   ワープ禁止：全体再ソート禁止
    #   距離：隣同士の交換のみ + 交換コスト
    #   重要：1パス中に同一車が何回も抜けない
    # =========================================================
    def _append_ko_queue_predictions(note_sections, all_lines, score_map, FR_line, VTX_line, U_line, _lfr):
        def _digits_of_line(ln):
            s = "".join(ch for ch in str(ln) if ch.isdigit())
            return [int(ch) for ch in s] if s else []

        def _norm_line(ln):
            return "".join(ch for ch in str(ln) if ch.isdigit())

        _PATTERNS = [
            ("順流→渦→逆流", ["順流", "渦", "逆流"]),
            ("順流→逆流→渦", ["順流", "逆流", "渦"]),
            ("渦→順流→逆流", ["渦", "順流", "逆流"]),
            ("渦→逆流→順流", ["渦", "逆流", "順流"]),
            ("逆流→順流→渦", ["逆流", "順流", "渦"]),
            ("逆流→渦→順流", ["逆流", "渦", "順流"]),
        ]

        def _infer_line_zone(ln):
            s = _norm_line(ln)

            # 新方式：ライン評価グループを優先
            try:
                zmap = globals().get("LINE_ZONE_MAP", {})
                if isinstance(zmap, dict) and s in zmap:
                    return zmap.get(s, "その他")
            except Exception:
                pass

            # 保険：旧方式
            if s and FR_line and s == _norm_line(FR_line):
                return "順流"
            if VTX_line and s == _norm_line(VTX_line):
                return "渦"
            if s and U_line and s == _norm_line(U_line):
                return "逆流"

            return "その他"

        def _queue_for_pattern(lines, svr_order):
            lines = list(lines or [])
            bucket = {"順流": [], "渦": [], "逆流": [], "その他": []}
            for ln in lines:
                bucket[_infer_line_zone(ln)].append(ln)

            queue = []
            for z in (svr_order or ["順流", "渦", "逆流"]):
                xs = sorted(bucket.get(z, []), key=lambda x: _lfr(x), reverse=True)
                for ln in xs:
                    queue.extend(_digits_of_line(ln))

            xs = sorted(bucket.get("その他", []), key=lambda x: _lfr(x), reverse=True)
            for ln in xs:
                queue.extend(_digits_of_line(ln))

            if not queue:
                for ln in lines:
                    queue.extend(_digits_of_line(ln))
            return queue

        def _build_car_zone_map(lines):
            m = {}
            for ln in (lines or []):
                z = _infer_line_zone(ln)
                for c in _digits_of_line(ln):
                    m[int(c)] = z
            return m

        _car_zone_map = _build_car_zone_map(all_lines)

        _car_line_size = {}
        _car_line_pos = {}

        for ln in (all_lines or []):
            ds = _digits_of_line(ln)
            sz = len(ds)

            for idx, c in enumerate(ds):
                _car_line_size[int(c)] = sz if sz > 0 else 1
                _car_line_pos[int(c)] = int(idx)

        def _pos_adj_for_car(car):
            """
            位置補正は隊列全体の何番目かではなく、
            その車が所属ライン内で何番手かを見る。
            単騎は番手利を与えない。
            """
            car = int(car)
            sz = int(_car_line_size.get(car, 1) or 1)
            pos = int(_car_line_pos.get(car, 0) or 0)

            # 単騎は位置補正なし
            if sz <= 1:
                return 0.0

            # ライン先頭
            if pos == 0:
                return -0.040

            # ライン2番手
            if pos == 1:
                return +0.020

            # 3番手以降
            return 0.0

        _FR_K_MAIN = 0.18
        _FR_K_SUB = 0.06
        _FR_BONUS_CAP = 0.06

        def _fr_bonus_for_car(car, main_zone):
            z = _car_zone_map.get(int(car), "その他")
            z_fr = {
                "順流": float(_lfr(FR_line) if FR_line else 0.0),
                "渦":   float(_lfr(VTX_line) if VTX_line else 0.0),
                "逆流": float(_lfr(U_line) if U_line else 0.0),
            }.get(z, 0.0)

            k = _FR_K_MAIN if z == main_zone else _FR_K_SUB
            sz = float(_car_line_size.get(int(car), 1) or 1.0)

            bonus = (k * z_fr) / sz
            if bonus > _FR_BONUS_CAP:
                bonus = _FR_BONUS_CAP
            if bonus < 0.0:
                bonus = 0.0
            return bonus

        def _run_ko(q, main_zone):
            # ======================================================
            # 距離ベース（B）＋ KO閾値（C）
            # ======================================================
            q = [int(x) for x in (q or []) if str(x).isdigit()]

            seen = set()
            order = []
            for c in q:
                if c not in seen:
                    seen.add(c)
                    order.append(c)

        def _run_ko(q, main_zone):
            # ======================================================
            # 距離ベース（B）＋ KO閾値（C）
            # ======================================================
            q = [int(x) for x in (q or []) if str(x).isdigit()]

            seen = set()
            order = []
            for c in q:
                if c not in seen:
                    seen.add(c)
                    order.append(c)


            tail = [int(c) for c in score_map.keys() if int(c) not in seen]
            tail.sort(key=lambda c: float(score_map.get(int(c), 0.0)), reverse=True)
            order.extend(tail)

            straight_m = float(globals().get("straight_length", 60.0) or 60.0)
            style = float(globals().get("style", 0.0) or 0.0)
            wind_ms = float(globals().get("wind_speed", 0.0) or 0.0)
            race_class = str(globals().get("race_class", "Ａ級") or "Ａ級")

            CLASS_SPREAD = {"Ｓ級": 1.00, "Ａ級": 0.90, "Ａ級チャレンジ": 0.80, "ガールズ": 0.85}
            spread = float(CLASS_SPREAD.get(race_class, 0.90))

            def _final_at(car, i):
                base = float(score_map.get(int(car), 0.0))
                return base + _pos_adj_for_car(int(car)) + _fr_bonus_for_car(int(car), main_zone)
            
            # ====== PATCH: venue-aware pass_m / available_m + speed-based MAX_PASSES ======
            pass_m = 14.0 + 0.35 * straight_m
            pass_m *= (1.0 + 0.25 * max(0.0, style))
            pass_m *= (1.0 + 0.03 * max(0.0, wind_ms - 3))

            # 会場カント（薄く：外回しロス増）
            bank_angle = float(globals().get("bank_angle", 30.0) or 30.0)
            pass_m *= (1.0 + 0.10 * max(0.0, (bank_angle - 30.0) / 10.0))  # 36°で+6%程度

            # クリップ
            if pass_m < 18.0:
                pass_m = 18.0
            if pass_m > 55.0:
                pass_m = 55.0

            # ---- available_m: bank_len を “差分だけ” 反映して飽和を減らす ----
            bank_len = float(globals().get("bank_length", 400.0) or 400.0)
            base_bank = 400.0
            # bank_len差分の反映を少し強める（500が1回に張り付くのを緩和）
            bank_term = 0.20 * base_bank + 0.30 * (bank_len - base_bank)
            available_m = float(straight_m) + bank_term

            # ---- スコア分布（sigma）----
            vals = [float(score_map.get(int(c), 0.0)) for c in order]
            if len(vals) >= 2:
                mu = sum(vals) / float(len(vals))
                var = sum((v - mu) ** 2 for v in vals) / float(len(vals))
                sigma = max(var ** 0.5, 1e-6)
            else:
                mu = (vals[0] if vals else 0.0)
                sigma = 1e-6

            # ---- クラス別の代表速度（終盤の代表値）----
            VREF_KMH = {"Ｓ級": 67.0, "Ａ級": 64.0, "Ａ級チャレンジ": 62.0, "ガールズ": 63.0}
            v_ref = float(VREF_KMH.get(race_class, 64.0)) / 3.6  # m/s

            # ---- スコア→速度：zで圧縮（暴走防止）----
            # 333/335は終盤時間が短く gain_m が出にくいので少し強める
            if bank_len <= 335:
                k_speed = float(globals().get("ko_k_speed_333", 0.014) or 0.014)
            elif bank_len >= 500:
                k_speed = float(globals().get("ko_k_speed_500", 0.012) or 0.012)
            else:
                k_speed = float(globals().get("ko_k_speed", 0.011) or 0.011)
            def _v_from_score(sc: float) -> float:
                z = (float(sc) - float(mu)) / float(sigma)
                if z > 2.0:
                    z = 2.0
                if z < -2.0:
                    z = -2.0
                return float(v_ref) * (1.0 + float(k_speed) * z)

            # ---- 終盤時間 & 相対距離（抜ける回数の根拠）----
            t_final = float(available_m) / max(float(v_ref), 1e-6)

            top_scores = sorted(vals, reverse=True)
            if len(top_scores) >= 3:
                v_fast = _v_from_score(top_scores[0])
                v_mid  = _v_from_score(top_scores[2])
            else:
                v_fast = _v_from_score(mu + sigma)
                v_mid  = _v_from_score(mu)

            gain_m = max(0.0, (float(v_fast) - float(v_mid)) * float(t_final))

            MAX_PASSES = int(gain_m // max(pass_m, 1e-9))
            if MAX_PASSES < 1:
                MAX_PASSES = 1

            # 333/335は最大2、その他は最大3（過剰シャッフル防止）
            cap = 2 if bank_len <= 335 else 3
            if MAX_PASSES > cap:
                MAX_PASSES = cap

            # ---- PASS_DELTAの正規化：available_m依存を弱めて安定化 ----
            base_k = float(globals().get("ko_base_k", 0.040) or 0.040)  # 0.025〜0.060
            score_per_m = base_k * sigma * (1.0 / max(spread, 1e-6)) / max(pass_m, 1e-6)

            PASS_DELTA = score_per_m * pass_m
            cross_mul = 0.35 if bank_len <= 335 else 0.30
            CROSS_DELTA = score_per_m * (cross_mul * pass_m)
            fatigue_delta = 0.35 * PASS_DELTA
            # ====== /PATCH ======

            overtake_cnt = {int(c): 0 for c in order}

            for _ in range(MAX_PASSES):
                swapped = False
                n = len(order)
                moved_this_pass = set()

                for i in range(n - 1):
                    a = order[i]
                    b = order[i + 1]

                    if b in moved_this_pass:
                        continue

                    sa = _final_at(a, i)
                    sb = _final_at(b, i + 1)

                    need = PASS_DELTA + fatigue_delta * float(overtake_cnt.get(b, 0))

                    za = _car_zone_map.get(int(a), "その他")
                    zb = _car_zone_map.get(int(b), "その他")
                    if za != zb:
                        need += CROSS_DELTA

                    if sb >= sa + need:
                        order[i], order[i + 1] = b, a
                        overtake_cnt[b] = overtake_cnt.get(b, 0) + 1
                        moved_this_pass.add(b)
                        swapped = True

                if not swapped:
                    break

            globals()["_overtake_available_m"] = float(available_m)
            globals()["_overtake_pass_m"] = float(pass_m)
            globals()["_overtake_max_passes"] = int(MAX_PASSES)
            globals()["_overtake_pass_delta"] = float(PASS_DELTA)
            globals()["_overtake_cross_delta"] = float(CROSS_DELTA)

            # 任意：調整が速くなるデバッグ（欲しければ d 表示にも足せる）
            globals()["_overtake_gain_m"] = float(gain_m)
            globals()["_overtake_t_final"] = float(t_final)
            globals()["_overtake_v_ref"] = float(v_ref)

            return order

            globals()["_overtake_available_m"] = float(available_m)
            globals()["_overtake_pass_m"] = float(pass_m)
            globals()["_overtake_max_passes"] = int(MAX_PASSES)
            globals()["_overtake_pass_delta"] = float(PASS_DELTA)
            globals()["_overtake_cross_delta"] = float(CROSS_DELTA)

            return order

        outs = {}
        for pname, svr in _PATTERNS:
            q = _queue_for_pattern(all_lines, svr)
            main_zone = (svr[0] if (svr and len(svr) >= 1) else "順流")
            outs[pname] = _run_ko(q, main_zone)

        def _fmt_seq(seq, max_n=None):
            xs = [int(x) for x in (seq or []) if str(x).isdigit()]
            if max_n is None:
                max_n = int(globals().get("n_cars", len(xs)))
            xs = xs[:max_n]
            return " → ".join(str(x) for x in xs) if xs else "（なし）"

        out_j = outs.get("順流→渦→逆流") or []
        out_v = outs.get("渦→順流→逆流") or []
        out_u = outs.get("逆流→順流→渦") or []

        

                       # ======================================================
        # 表示用ガード：
        # 1) KO隊列結果がスコア下位を頭に置きすぎる場合だけ補正
        # 2) 主戦ライン先頭が同ライン低スコア車より後ろに落ちるのを防ぐ
        # ※ _run_ko本体は触らない
        # ======================================================
        def _digits_line(x):
            return [int(ch) for ch in str(x) if ch.isdigit()]

        def _display_score_guard(seq, main_line=None):
            xs = [int(x) for x in (seq or []) if str(x).isdigit()]
            if not xs:
                return xs

            score_order = sorted(
                [int(k) for k in score_map.keys()],
                key=lambda c: (-float(score_map.get(c, 0.0)), c)
            )
            score_rank = {c: i + 1 for i, c in enumerate(score_order)}

            # 1) 先頭ガード
            # 先頭がKOスコア5位以下なら、スコア上位3台のうち
            # 元の隊列内で一番前にいる車を先頭へ上げる
            head = xs[0]
            if score_rank.get(head, 99) >= 5:
                candidates = [c for c in score_order[:3] if c in xs]
                if candidates:
                    best = min(candidates, key=lambda c: xs.index(c))
                    xs.remove(best)
                    xs.insert(0, best)

                        # 2) 主戦ライン先頭ガード
            # 例：364なら3がライン先頭。
            # 3よりスコアが低い同ライン車（例：6）が3より前にいるなら、
            # 3をその車の前まで戻す。
            line_members = _digits_line(main_line)
            if len(line_members) >= 2:
                line_head = line_members[0]

                if line_head in xs:
                    line_head_score = float(score_map.get(line_head, 0.0))
                    line_head_idx = xs.index(line_head)

                    lower_mates_before = []
                    for m in line_members[1:]:
                        if m in xs:
                            m_score = float(score_map.get(m, 0.0))
                            if m_score < line_head_score and xs.index(m) < line_head_idx:
                                lower_mates_before.append(m)

                    if lower_mates_before:
                        target_idx = min(xs.index(m) for m in lower_mates_before)
                        xs.remove(line_head)
                        xs.insert(target_idx, line_head)

                        # 3) 最下位スコア車の早出しガード
            # KOスコア最下位の車が3番手以内に残るのを防ぐ
            n_score = len(score_order)

            for bad in list(xs):
                if score_rank.get(bad, 99) == n_score and xs.index(bad) <= 2:
                    xs.remove(bad)

                    # スコア5位以内の車が並んだ最後の直後へ送る
                    insert_pos = 0
                    for i, c in enumerate(xs):
                        if score_rank.get(c, 99) <= 5:
                            insert_pos = i + 1

                    xs.insert(insert_pos, bad)

                        # 4) KO上位車の沈みすぎガード
            # KOスコア上位3車が沈みすぎるのを防ぐ
            # 1位は頭候補、2〜3位は3番手以内を目安に戻す
            for good in score_order[:3]:
                if good not in xs:
                    continue

                r = score_rank.get(good, 99)

                # KO2〜3位が4番手以下なら、3番手以内へ戻す
                if r in (2, 3) and xs.index(good) >= 3:
                    xs.remove(good)
                    target_pos = min(2, len(xs))
                    xs.insert(target_pos, good)

                # KO1位が3番手以下なら、2番手以内へ戻す
                elif r == 1 and xs.index(good) >= 2:
                    xs.remove(good)
                    target_pos = min(1, len(xs))
                    xs.insert(target_pos, good)

            return xs

        # ======================================================
        # v195：戦法別シナリオ補正
        # 順流・渦・逆流は「同じ全体順位の別名」ではなく、
        # それぞれの流域ラインが主役になった場合の着順予想として組み立てる。
        #
        # 重要：
        # ・無条件にライン先頭を1着固定するのではなく、ライン内のKO/役割で頭候補を選ぶ。
        # ・ただし、その戦法のシナリオでは主役ラインのいずれかが1着候補になる前提を守る。
        # ・逆流域がLINE_ZONE_MAP上で空でも、旧U_lineがあれば逆流シナリオの主役ラインとして使う。
        # ======================================================
        def _scenario_line_digits(_ln):
            try:
                if isinstance(_ln, (list, tuple)):
                    return [int(x) for x in _ln if str(x).isdigit()]
            except Exception:
                pass
            return [int(ch) for ch in str(_ln) if ch.isdigit()]

        def _scenario_line_key(_ln):
            return "".join(str(int(x)) for x in _scenario_line_digits(_ln))

        def _scenario_same_line(_a, _b):
            return _scenario_line_key(_a) == _scenario_line_key(_b) and bool(_scenario_line_key(_a))

        def _scenario_lines_for_zone(_zone_name):
            """LINE_ZONE_MAPから該当ゾーンのラインを取得。line_defを優先して実ライン順を復元する。"""
            out_lines = []
            seen_keys = set()
            try:
                zmap = globals().get("LINE_ZONE_MAP", {}) or {}
                _line_def_local = globals().get("line_def", {}) or {}
                if isinstance(_line_def_local, dict):
                    for _gid, _mem in _line_def_local.items():
                        _xs = _scenario_line_digits(_mem)
                        _key = _scenario_line_key(_xs)
                        if not _key or _key in seen_keys:
                            continue
                        if str(zmap.get(_key, "")) == str(_zone_name):
                            seen_keys.add(_key)
                            out_lines.append(_xs)
                # line_defに無いキーがあれば保険で拾う
                if isinstance(zmap, dict):
                    for _key, _z in zmap.items():
                        if str(_z) != str(_zone_name):
                            continue
                        if str(_key) in seen_keys:
                            continue
                        _xs = _scenario_line_digits(_key)
                        if _xs:
                            seen_keys.add(str(_key))
                            out_lines.append(_xs)
            except Exception:
                out_lines = []
            try:
                out_lines = sorted(out_lines, key=lambda _ln: float(_lfr(_ln)), reverse=True)
            except Exception:
                pass
            return out_lines

        def _scenario_main_line(_style_name):
            """
            各戦法の主役ライン。

            v270-R2:
            表示済みのライン評価グループ（LINE_ZONE_MAP）を唯一の基準にする。
            旧FR_line／VTX_line／U_lineは、該当ゾーンを取得できない場合だけ
            フォールバックとして使う。これにより、表示上は「渦域=37」なのに
            「渦メイン=625」となるラベル逆転を防ぐ。

            2ライン戦などでは、順流・渦・逆流が同じラインを重複して主役に
            しない従来条件を維持する。
            """
            try:
                style = str(_style_name or "")

                def _first_distinct(_lines, _used_keys):
                    for _ln in (_lines or []):
                        _key = _scenario_line_key(_ln)
                        if _key and _key not in _used_keys:
                            return _scenario_line_digits(_ln)
                    return []

                if style == "順流":
                    # 現在のライン評価グループを最優先。
                    _current = _first_distinct(_scenario_lines_for_zone("順流"), set())
                    if _current:
                        return _current
                    # LINE_ZONE_MAPで取得できない場合だけ旧FR_lineへ戻る。
                    return _scenario_line_digits(FR_line) if FR_line else []

                if style == "渦":
                    _jun = _scenario_main_line("順流")
                    _used_keys = {
                        _scenario_line_key(_jun)
                    } if _scenario_line_key(_jun) else set()

                    _current = _first_distinct(_scenario_lines_for_zone("渦"), _used_keys)
                    if _current:
                        return _current

                    # 現在の渦域が取得できない場合だけ旧VTX_lineへ戻る。
                    if VTX_line:
                        _key = _scenario_line_key(VTX_line)
                        if _key and _key not in _used_keys:
                            return _scenario_line_digits(VTX_line)
                    return []

                if style == "逆流":
                    _jun = _scenario_main_line("順流")
                    _vtx = _scenario_main_line("渦")
                    _used_keys = {
                        _key for _key in (
                            _scenario_line_key(_jun),
                            _scenario_line_key(_vtx),
                        ) if _key
                    }

                    _current = _first_distinct(_scenario_lines_for_zone("逆流"), _used_keys)
                    if _current:
                        return _current

                    # 現在の逆流域が取得できない場合だけ旧U_lineへ戻る。
                    if U_line:
                        _key = _scenario_line_key(U_line)
                        if _key and _key not in _used_keys:
                            return _scenario_line_digits(U_line)
                    return []
            except Exception:
                return []
            return []

        def _scenario_queue_for_main(_main_line, _zone_order):
            """主役ラインを先頭に置いた仮想隊列。残りはゾーン順＋FR順で並べる。"""
            main = _scenario_line_digits(_main_line)
            main_key = _scenario_line_key(main)
            queue = []
            seen_cars = set()

            for c in main:
                if int(c) not in seen_cars:
                    seen_cars.add(int(c))
                    queue.append(int(c))

            # all_linesが取れる場合はライン単位、無い場合は既存のSTYLE_SEQ_MAP相当で補完。
            used_line_keys = {main_key} if main_key else set()
            try:
                lines_src = list(all_lines or [])
            except Exception:
                lines_src = []

            bucket = {"順流": [], "渦": [], "逆流": [], "その他": []}
            for ln in lines_src:
                key = _scenario_line_key(ln)
                if key and key in used_line_keys:
                    continue
                z = _infer_line_zone(ln)
                # U_lineは逆流シナリオでは既に主役として使うため、渦側に重複させない。
                bucket.setdefault(z, []).append(ln)

            for z in (_zone_order or []):
                xs = sorted(bucket.get(z, []), key=lambda ln: float(_lfr(ln)), reverse=True)
                for ln in xs:
                    key = _scenario_line_key(ln)
                    if key and key in used_line_keys:
                        continue
                    used_line_keys.add(key)
                    for c in _scenario_line_digits(ln):
                        if int(c) not in seen_cars:
                            seen_cars.add(int(c))
                            queue.append(int(c))

            for z in ["順流", "渦", "逆流", "その他"]:
                if z in (_zone_order or []):
                    continue
                xs = sorted(bucket.get(z, []), key=lambda ln: float(_lfr(ln)), reverse=True)
                for ln in xs:
                    key = _scenario_line_key(ln)
                    if key and key in used_line_keys:
                        continue
                    used_line_keys.add(key)
                    for c in _scenario_line_digits(ln):
                        if int(c) not in seen_cars:
                            seen_cars.add(int(c))
                            queue.append(int(c))

            # 保険：score_mapに存在する車を全て補完
            try:
                tail = sorted([int(c) for c in score_map.keys() if int(c) not in seen_cars], key=lambda c: float(score_map.get(c, 0.0)), reverse=True)
                queue.extend(tail)
            except Exception:
                pass
            return queue

        def _scenario_best_head_from_main_line(_main_line):
            """主役ライン内で最も頭に置きやすい車を選ぶ。ライン先頭固定ではない。"""
            main = _scenario_line_digits(_main_line)
            if not main:
                return None
            def _role_bonus(_car):
                try:
                    pos = main.index(int(_car))
                except Exception:
                    pos = 0
                # 先頭と番手を主に見る。3番手以降は頭固定しにくいが、KOが抜けていれば上がれる。
                if len(main) <= 1:
                    return 0.00
                if pos == 0:
                    return 0.035
                if pos == 1:
                    return 0.025
                return -0.015
            try:
                return max(main, key=lambda c: float(score_map.get(int(c), 0.0)) + _role_bonus(c))
            except Exception:
                return int(main[0])

        def _scenario_force_main_head(_seq, _main_line):
            """
            シナリオの前提として、主役ラインのいずれかを1着候補へ置く。

            v196:
            1着候補だけを先頭へ上げても、同ライン相手が後方へ沈むと
            「そのラインが主役になった展開」として買目妙味が効かない。
            そのため、主役ラインの残りも2〜4番手以内へ保護する。
            ただしライン丸ごと無条件固定ではなく、KO/既存順位を見て並べる。
            """
            xs = [int(x) for x in (_seq or []) if str(x).isdigit()]
            main = _scenario_line_digits(_main_line)
            if not xs or not main:
                return xs

            main_set = {int(c) for c in main}
            head = int(xs[0]) if int(xs[0]) in main_set else _scenario_best_head_from_main_line(main)
            if head is None or int(head) not in xs:
                return xs
            head = int(head)

            # 主役ライン内の残りは、KOスコア＋既存順位で2〜4番手へ寄せる。
            # 2車ラインなら相手を2番手へ、3車以上なら最大2車までを上位保護する。
            main_rest = [int(c) for c in main if int(c) != head and int(c) in xs]
            try:
                rank_now = {int(c): i for i, c in enumerate(xs)}
                main_rest = sorted(
                    main_rest,
                    key=lambda c: (float(score_map.get(int(c), 0.0)), -int(rank_now.get(int(c), 99))),
                    reverse=True,
                )
            except Exception:
                pass

            protect_count = 1 if len(main) <= 2 else 2
            protected = [head] + main_rest[:protect_count]

            out = []
            seen = set()
            for c in protected:
                c = int(c)
                if c in xs and c not in seen:
                    out.append(c)
                    seen.add(c)

            for c in xs:
                c = int(c)
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out

        def _make_style_scenario_seq(_style_name, _fallback_seq):
            main_line = _scenario_main_line(_style_name)
            if not main_line:
                # v197:
                # シナリオ主役ラインが存在しない流れは、
                # fallbackで無理に買目考察を作らない。
                # 例：2ライン戦で渦=旧逆流ラインの場合、逆流は空扱い。
                return []

            if _style_name == "順流":
                zone_order = ["順流", "渦", "逆流"]
                fallback_main = FR_line
            elif _style_name == "渦":
                zone_order = ["渦", "順流", "逆流"]
                fallback_main = VTX_line
            else:
                zone_order = ["逆流", "順流", "渦"]
                fallback_main = U_line

            q = _scenario_queue_for_main(main_line, zone_order)
            seq = _run_ko(q, _style_name)
            seq = _display_score_guard(seq, main_line or fallback_main)
            seq = _scenario_force_main_head(seq, main_line)
            return [int(x) for x in (seq or []) if str(x).isdigit()]

        # 旧表示順は保持しておく。以後の買目考察にはシナリオ補正版を使う。
        out_j_raw = _display_score_guard(out_j, FR_line)
        out_v_raw = _display_score_guard(out_v, VTX_line)
        out_u_raw = _display_score_guard(out_u, U_line)

        out_j = _make_style_scenario_seq("順流", out_j_raw)
        out_v = _make_style_scenario_seq("渦", out_v_raw)
        out_u = _make_style_scenario_seq("逆流", out_u_raw)

        globals()["STYLE_BASE_SEQ_MAP"] = {
            "順流": [int(x) for x in (out_j_raw or []) if str(x).isdigit()],
            "渦":   [int(x) for x in (out_v_raw or []) if str(x).isdigit()],
            "逆流": [int(x) for x in (out_u_raw or []) if str(x).isdigit()],
        }
        globals()["STYLE_SCENARIO_MAIN_LINE_MAP"] = {
            "順流": _scenario_main_line("順流"),
            "渦":   _scenario_main_line("渦"),
            "逆流": _scenario_main_line("逆流"),
        }

        # ======================================================
        # H主導ライン3番手以降：
        # 3着内率40%以上なら、
        # 「その戦法の表示1着候補ライン」と同じ場合だけ4番手以内へ移動
        # ======================================================
        try:
            def _display_promote_gid(_car_no):
                try:
                    _car_no = int(_car_no)
                    if isinstance(line_def, dict):
                        for _gid, _mem in line_def.items():
                            _mem2 = [int(x) for x in _mem]
                            if _car_no in _mem2:
                                return _gid
                except Exception:
                    pass
                return None

            def _display_promote_top3_rate(_car_no):
                try:
                    _car_no = int(_car_no)

                    _x1 = globals().get("x1", {})
                    _x2 = globals().get("x2", {})
                    _x3 = globals().get("x3", {})
                    _xo = globals().get("x_out", {})

                    n1 = float(_x1.get(_car_no, _x1.get(str(_car_no), 0)) or 0)
                    n2 = float(_x2.get(_car_no, _x2.get(str(_car_no), 0)) or 0)
                    n3 = float(_x3.get(_car_no, _x3.get(str(_car_no), 0)) or 0)
                    no = float(_xo.get(_car_no, _xo.get(str(_car_no), 0)) or 0)

                    total = n1 + n2 + n3 + no
                    if total <= 0:
                        return None

                    return float((n1 + n2 + n3) / total)

                except Exception:
                    return None

            def _display_promote_to_top4(_seq, _target_car):
                try:
                    _target_car = int(_target_car)
                    _xs = [int(x) for x in (_seq or []) if str(x).isdigit()]

                    if _target_car not in _xs:
                        return _xs

                    _idx = _xs.index(_target_car)

                    # すでに4番手以内なら何もしない
                    if _idx <= 3:
                        return _xs

                    _xs.pop(_idx)
                    _xs.insert(3, _target_car)

                    return _xs

                except Exception:
                    return _seq

            # H主導ラインの3番手以降で、3着内率40%以上の車だけ対象
            _promote_targets = []

            if home_top_gid is not None and isinstance(line_def, dict):
                _h_members = [int(x) for x in line_def.get(home_top_gid, [])]

                if len(_h_members) >= 3:
                    for _car3 in _h_members[2:]:
                        _p3 = _display_promote_top3_rate(_car3)

                        if _p3 is not None and float(_p3) >= 0.40:
                            _promote_targets.append(int(_car3))

            # 各戦法の「表示上の1着候補ライン」と同じ場合だけ、4番手以内へ移動
            for _car3 in _promote_targets:
                _target_gid = _display_promote_gid(_car3)

                if _target_gid is None:
                    continue

                # 順流
                if out_j:
                    _jun_head = int(out_j[0])
                    _jun_gid = _display_promote_gid(_jun_head)
                    if _target_gid == _jun_gid:
                        out_j = _display_promote_to_top4(out_j, _car3)

                # 渦
                if out_v:
                    _vtx_head = int(out_v[0])
                    _vtx_gid = _display_promote_gid(_vtx_head)
                    if _target_gid == _vtx_gid:
                        out_v = _display_promote_to_top4(out_v, _car3)

                # 逆流
                if out_u:
                    _u_head = int(out_u[0])
                    _u_gid = _display_promote_gid(_u_head)
                    if _target_gid == _u_gid:
                        out_u = _display_promote_to_top4(out_u, _car3)

        except Exception as _e:
            note_sections.append(f"※H主導3番手以降・戦法別4番手以内補正エラー：{_e}")
            note_sections.append("")

        # ======================================================
        # 戦法別評価順を保存
        # v195以降：STYLE_SEQ_MAPは、流域ライン主役のシナリオ補正版を保存する。
        # 後段の「戦法別想定決着率」「2車複候補」「買目考察」はこの補正版を使う。
        # 元の全体KO寄り順位は STYLE_BASE_SEQ_MAP に保持済み。
        # ======================================================
        globals()["STYLE_SCENARIO_SEQ_MAP"] = {
            "順流": [int(x) for x in (out_j or []) if str(x).isdigit()],
            "渦":   [int(x) for x in (out_v or []) if str(x).isdigit()],
            "逆流": [int(x) for x in (out_u or []) if str(x).isdigit()],
        }
        globals()["STYLE_SEQ_MAP"] = dict(globals().get("STYLE_SCENARIO_SEQ_MAP", {}) or {})

        # ======================================================
        # 戦法別着順予想を全表示
        # ※ここでは推奨戦法がまだ確定していないため、強調はしない。
        #   推奨戦法は現行サマリーで表示する。
        # ======================================================
        try:
            def _fmt_seq_full(_seq):
                _xs = [int(x) for x in (_seq or []) if str(x).isdigit()]
                return " → ".join(str(x) for x in _xs) if _xs else "該当なし"

            note_sections.append("【順流メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_j))
            note_sections.append("")
            note_sections.append("【渦メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_v))
            note_sections.append("")
            note_sections.append("【逆流メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_u))
            note_sections.append("")
        except Exception as _e:
            note_sections.append(f"※戦法別着順予想表示エラー：{_e}")
            note_sections.append("")


    _append_ko_queue_predictions(note_sections, all_lines, score_map, FR_line, VTX_line, U_line, _lfr)
    # ここまでで note_sections を確実に保持

        # =========================================================
    # ＜短評＞（KOの成否に関係なく表示）※完全tryゼロ
    # =========================================================
    lines_out = ["＜短評＞"]

    # レースFR：flowのFR（過去出力と同じ定義）
    raceFR = float(_flow.get("FR", 0.0) or 0.0) if isinstance(_flow, dict) else 0.0
    if raceFR != raceFR:  # NaN
        raceFR = 0.0

    # flowが0なら「混戦度」= 1 - 最大取り分（line_fr_mapがあれば）
    if raceFR <= 0.0 and isinstance(line_fr_map, dict) and line_fr_map:
        vals = []
        for v in line_fr_map.values():
            s = str(v).strip()
            fv = float(s) if s not in ("", "None", "nan", "NaN") else 0.0
            if fv > 0.0 and fv == fv:
                vals.append(fv)

        total = sum(vals)
        if total > 1e-12:
            max_share = max(fv / total for fv in vals)
            raceFR = 1.0 - max_share
            if raceFR < 0.0:
                raceFR = 0.0
            if raceFR > 1.0:
                raceFR = 1.0

    # レースFR表示
    lines_out.append(f"・レースFR={raceFR:.3f}［{_band3_fr(raceFR)}］")

    # 混戦度表示
    _compact_label = globals().get("race_compact_label", "未判定")
    _compact_gap = globals().get("race_compact_gap", None)

    if _compact_gap is not None:
        lines_out.append(
            f"・順当度：{_compact_label}［上位差={float(_compact_gap):.2f}］"
        )
    else:
        lines_out.append(
            f"・順当度：{_compact_label}"
        )

    # VTX/U はラインFR（ズレ防止）
    _vtx_fr = float(_lfr(VTX_line) if VTX_line else 0.0)
    _u_fr = float(_lfr(U_line) if U_line else 0.0)

    

    lines_out.append(f"・VTXラインFR={_vtx_fr:.3f}［{_band3_vtx(_vtx_fr)}］")
    lines_out.append(f"・逆流ラインFR={_u_fr:.3f}［{_band3_u(_u_fr)}］")

    # 内訳要約（flow dbg）
    dbg = _flow.get("dbg", {}) if isinstance(_flow, dict) else {}

    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star", 0.0) or 0.0)
        bn = float(dbg.get("blend_none", 0.0) or 0.0)
        sd = float(dbg.get("sd", 0.0) or 0.0)
        nu = float(dbg.get("nu", 0.0) or 0.0)

        star_txt = "先頭負担:強" if bs <= -0.60 else (
                   "先頭負担:中" if bs <= -0.30 else
                   "先頭負担:小")

        none_txt = "無印押上げ:強" if bn >= 1.20 else (
                   "無印押上げ:中" if bn >= 0.60 else
                   "無印押上げ:小")

        sd_txt = "ライン偏差:大" if sd >= 0.60 else (
                 "ライン偏差:中" if sd >= 0.30 else
                 "ライン偏差:小")

        nu_txt = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"

        lines_out.append(
            f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}"
        )

    # =========================================================
    # ＜短評＞（KOの成否に関係なく表示）
    # =========================================================
    lines_out = ["＜短評＞"]

    raceFR = float(_flow.get("FR", 0.0) or 0.0) if isinstance(_flow, dict) else 0.0
    if raceFR != raceFR:
        raceFR = 0.0

    if raceFR <= 0.0 and isinstance(line_fr_map, dict) and line_fr_map:
        vals = []
        for v in line_fr_map.values():
            s = str(v).strip()
            fv = float(s) if s not in ("", "None", "nan", "NaN") else 0.0
            if fv > 0.0 and fv == fv:
                vals.append(fv)

        total = sum(vals)
        if total > 1e-12:
            max_share = max(fv / total for fv in vals)
            raceFR = 1.0 - max_share
            raceFR = max(0.0, min(1.0, raceFR))

        lines_out.append(f"・レースFR={raceFR:.3f}［{_band3_fr(raceFR)}］")

    # レースレベル表示
    try:
        lines_out.append(
            f"・レースレベル：{race_level_label}［平均得点={race_level_avg:.2f}／得点差={race_level_spread:.2f}］"
        )
    except Exception:
        pass

    _vtx_fr = float(_lfr(VTX_line) if VTX_line else 0.0)
    _u_fr = float(_lfr(U_line) if U_line else 0.0)

        # 混戦度表示
    _compact_label = globals().get("race_compact_label", "未判定")
    _compact_gap = globals().get("race_compact_gap", None)

    if _compact_gap is not None:
        lines_out.append(
            f"・順当度：{_compact_label}［上位差={float(_compact_gap):.2f}］"
        )
    else:
        lines_out.append(
            f"・順当度：{_compact_label}"
        )

    lines_out.append(f"・VTXラインFR={_vtx_fr:.3f}［{_band3_vtx(_vtx_fr)}］")
    lines_out.append(f"・逆流ラインFR={_u_fr:.3f}［{_band3_u(_u_fr)}］")

    bs = 0.0
    bn = 0.0
    sd = 0.0
    nu = 1.0

    dbg = _flow.get("dbg", {}) if isinstance(_flow, dict) else {}
    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star", 0.0) or 0.0)
        bn = float(dbg.get("blend_none", 0.0) or 0.0)
        sd = float(dbg.get("sd", 0.0) or 0.0)
        nu = float(dbg.get("nu", 1.0) or 1.0)

    star_txt = "先頭負担:強" if bs <= -0.60 else ("先頭負担:中" if bs <= -0.30 else "先頭負担:小")
    none_txt = "無印押上げ:強" if bn >= 1.20 else ("無印押上げ:中" if bn >= 0.60 else "無印押上げ:小")
    sd_txt = "ライン偏差:大" if sd >= 0.60 else ("ライン偏差:中" if sd >= 0.30 else "ライン偏差:小")
    nu_txt = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"

    lines_out.append(f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}")

    # =========================================================
    # 推奨戦法（優先順位固定・上書き禁止）
    # =========================================================

    try:
        recommend_style = None
        recommend_reason = []
        confidence = "C"

        tenkai_txt = str(
            globals().get("展開評価", "")
            or globals().get("tenkai_eval", "")
            or ""
        ).strip()

        fr_diff = abs(_vtx_fr - _u_fr)

                # =====================================================
        # 現在のライン評価グループでH主導ラインを判定する
        #   旧FR_line / 旧VTX_line / 旧U_line ではなく、
        #   LINE_ZONE_MAP を優先する
        # =====================================================

        def _norm_line_key_for_recommend(ln):
            try:
                if isinstance(ln, (list, tuple)):
                    return "".join(str(int(x)) for x in ln if str(x).isdigit())
            except Exception:
                pass
            return "".join(ch for ch in str(ln) if ch.isdigit())

        def _current_zone_for_line(ln):
            key = _norm_line_key_for_recommend(ln)

            try:
                zmap = globals().get("LINE_ZONE_MAP", {})
                if isinstance(zmap, dict) and key in zmap:
                    return zmap.get(key, "その他")
            except Exception:
                pass

            # 保険：LINE_ZONE_MAPが無い場合だけ旧方式へフォールバック
            if key and key == _norm_line_key_for_recommend(FR_line):
                return "順流"
            if key and key == _norm_line_key_for_recommend(VTX_line):
                return "渦"
            if key and key == _norm_line_key_for_recommend(U_line):
                return "逆流"

            return "その他"

        def _style_fr_for_recommend(style_name):
            if style_name == "順流":
                return float(_lfr(FR_line) if FR_line else 0.0)
            if style_name == "渦":
                return float(_lfr(VTX_line) if VTX_line else 0.0)
            if style_name == "逆流":
                return float(_lfr(U_line) if U_line else 0.0)
            return 0.0

        # =====================================================
        # 1. 展開評価（最優先）
        # =====================================================

        if "混戦" in tenkai_txt:
            recommend_style = "渦"
            recommend_reason = ["展開=混戦"]

        elif "差し" in tenkai_txt:
            recommend_style = "渦"
            recommend_reason = ["展開=差し寄り"]

        elif "先行" in tenkai_txt or "逃げ" in tenkai_txt:
            recommend_style = "順流"
            recommend_reason = ["展開=先行寄り"]

        # =====================================================
        # 2. 短評（ここで確定させる）
        # =====================================================

        if recommend_style is None:

            if bn >= 0.50:
                recommend_style = "渦"
                recommend_reason = ["無印押上げ=中以上"]

            elif sd >= 0.60:
                recommend_style = "順流"
                recommend_reason = ["ライン偏差=大"]

            elif bs <= -0.60 and bn >= 0.50:
                recommend_style = "逆流"
                recommend_reason = ["先頭負担強＋押上げ中以上"]

        # =====================================================
        # 3. FR差（ここは最後）
        # =====================================================

        if recommend_style is None:

            if fr_diff >= 0.02:

                if _u_fr > _vtx_fr:
                    recommend_style = "逆流"
                    recommend_reason = ["逆流FR優勢"]

                else:
                    recommend_style = "順流"
                    recommend_reason = ["VTX優勢"]

        # =====================================================
        # 4. 最終安全側
        # =====================================================

        if recommend_style is None:
            recommend_style = "渦"
            recommend_reason = ["標準判定"]

               
                
                # =====================================================
        # H：推奨理由への反映
        #   旧分類ではなく、現在のライン評価グループで判定
        # =====================================================
        try:
            if home_top_line == "主導なし":
                recommend_reason.append("H主導ラインなし")
            else:
                h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                h_zone = _current_zone_for_line(h_line)

                if h_zone in ("順流", "渦", "逆流"):
                    recommend_reason.append(f"H主導={h_zone}ライン")
                else:
                    recommend_reason.append("H主導=その他ライン")
        except Exception:
            pass

                
               
                # =====================================================
        # 信頼度
        # =====================================================
        if bn >= 0.50:
            confidence = "B"

        elif fr_diff >= 0.02:
            confidence = "A"

        elif fr_diff >= 0.01:
            confidence = "B"

        else:
            confidence = "C"

                # =====================================================
        # H：低信頼時の推奨戦法切り替え
        #   旧分類ではなく、現在のライン評価グループで判定
        #   ※ガールズはライン戦ではないため、H主導で戦法を切り替えない
        # =====================================================
        h_style = None
        h_changed = False

        try:
            if home_top_line != "主導なし":
                h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                h_zone = _current_zone_for_line(h_line)

                if h_zone in ("順流", "渦", "逆流"):
                    h_style = h_zone
                    h_fr = float(_lfr(h_line) if h_line else 0.0)
                else:
                    h_style = None
                    h_fr = 0.0

                cur_fr = _style_fr_for_recommend(recommend_style)

                if not is_girls_like:
                    if (
                        h_style is not None
                        and h_style != recommend_style
                        and confidence in ("B", "C")
                        and h_fr >= cur_fr - 0.01
                    ):
                        recommend_reason.append(f"H主導により{h_style}寄せ")
                        recommend_style = h_style
                        h_changed = True
                        confidence = "B"
                else:
                    recommend_reason.append("ガールズ/アドバンスのためH主導による戦法変更なし")
        except Exception:
            pass

        # =====================================================
        # H：信頼度への反映
        #   旧分類ではなく、現在のライン評価グループで判定
        #   ※ガールズはライン戦ではないため、H主導で信頼度も上下させない
        # =====================================================
        try:
            if not is_girls_like:
                if home_top_line != "主導なし":
                    h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                    h_zone = _current_zone_for_line(h_line)

                    h_match = (
                        h_zone in ("順流", "渦", "逆流")
                        and h_zone == recommend_style
                    )

                    h_conflict = (
                        h_zone in ("順流", "渦", "逆流")
                        and h_zone != recommend_style
                    )

                    if h_match:
                        if confidence == "C":
                            confidence = "B"
                        elif confidence == "B":
                            confidence = "A"

                    elif h_conflict:
                        if confidence == "A":
                            confidence = "B"
                        elif confidence == "B":
                            confidence = "C"

        except Exception:
            pass

        # Hで戦法変更した場合は、過信防止で信頼度AをBに抑える
        try:
            if h_changed and confidence == "A":
                confidence = "B"
        except Exception:
            pass

        # H反映チェック表示
        try:
            if h_style is not None:
                if h_changed:
                    recommend_reason.append("H反映=戦法変更あり")
                else:
                    recommend_reason.append("H反映=戦法変更なし")
        except Exception:
            pass

        # =====================================================
        # v250: 最終推奨流れを「流れ想定比率」の単独1位へ統一
        #   ・KO、H主導、ライン評価などは比率を作る材料として扱う。
        #   ・比率算出後にKO多数決で別流れへ上書きしない。
        #   ・単独1位なら順流／逆流／渦のいずれでもその流れを採用。
        #   ・同率時だけ、直前までの推奨流れを維持する。
        # =====================================================
        try:
            if not is_girls_like:
                _final_ratio_map = globals().get("FLOW_RATIO_MAP_BY_ZONE", {}) or {}
                _ratio_style, _ratio_top_styles, _ratio_reason = (
                    _select_recommended_style_by_flow_ratio(
                        recommend_style,
                        _final_ratio_map,
                    )
                )

                if _ratio_style and _ratio_style != recommend_style:
                    recommend_reason.append(
                        f"{_ratio_reason}により{_ratio_style}へ統一"
                    )
                    recommend_style = _ratio_style
                elif _ratio_top_styles:
                    recommend_reason.append(
                        f"{_ratio_reason}=" + "／".join(_ratio_top_styles)
                    )
        except Exception:
            pass

                # =====================================================
        # ガールズ補正
        #   ガールズはライン戦ではないため、
        #   無印押上げだけで渦に寄せすぎない
        # =====================================================
        try:
            if is_girls_like and recommend_style == "渦":
                recommend_style = "順流"
                recommend_reason.append("ガールズ/アドバンスのため渦寄せを順流扱いに補正")
        except Exception:
            pass

                # =====================================================
        # 信頼度の最終補正：展開評価・順当度・上位差を統合
        # =====================================================
        try:
            compact_label = str(globals().get("race_compact_label", ""))
            compact_gap = globals().get("race_compact_gap", None)

            def _down_conf(conf):
                if conf == "A":
                    return "B"
                if conf == "B":
                    return "C"
                return "C"

            conf_down_reasons = []

            # 波乱気味＋上位差小は、信頼度を1段階下げる
            if "波乱気味" in compact_label and compact_gap is not None:
                if float(compact_gap) < 1.0:
                    old_conf = confidence
                    confidence = _down_conf(confidence)
                    if confidence != old_conf:
                        conf_down_reasons.append(
                            f"波乱気味＋上位差小={float(compact_gap):.2f}"
                        )

            # 混戦＋波乱気味はB以上を出しすぎない
            if "混戦" in tenkai_txt and "波乱気味" in compact_label:
                if confidence in ("A", "B"):
                    old_conf = confidence
                    confidence = "C"
                    if confidence != old_conf:
                        conf_down_reasons.append("混戦＋波乱気味")

            # レースFRが不利域なら、AはBへ落とす
            if raceFR >= 0.65 and confidence == "A":
                confidence = "B"
                conf_down_reasons.append(f"レースFR不利域={raceFR:.3f}")

            # ライン偏差大なら、B以上を1段階下げる
            if sd >= 0.60:
                old_conf = confidence
                confidence = _down_conf(confidence)
                if confidence != old_conf:
                    conf_down_reasons.append("ライン偏差大")

            if conf_down_reasons:
                recommend_reason.append(
                    "信頼度補正：" + "／".join(conf_down_reasons)
                )

        except Exception:
            pass

        # =====================================================
        # 推奨戦法と着順予想を現行サマリーへ渡す
        # =====================================================
        try:
            _style_seq_map_for_display = globals().get("STYLE_SEQ_MAP", {}) or {}
            _recommended_seq = _style_seq_map_for_display.get(recommend_style, []) or []
            if not _recommended_seq:
                _fallback_map = {
                    "順流": [int(x) for x in (out_j or []) if str(x).isdigit()],
                    "渦":   [int(x) for x in (out_v or []) if str(x).isdigit()],
                    "逆流": [int(x) for x in (out_u or []) if str(x).isdigit()],
                }
                _recommended_seq = _fallback_map.get(recommend_style, []) or []

            if _recommended_seq:
                globals()["RECOMMENDED_STYLE"] = recommend_style
                globals()["RECOMMENDED_STYLE_SEQ"] = _recommended_seq
        except Exception:
            pass

        # 推奨理由は短評内に残す
        lines_out.append(
            f"・推奨理由：{'／'.join(recommend_reason)}"
        )

    except Exception as _e:
        lines_out.append(f"・推奨戦法判定不可：{_e}")

    note_sections.extend(lines_out)
    note_sections.append("")
    globals()["note_sections"] = note_sections

except Exception as _e:
    try:
        ns = globals().get("note_sections", None)
        if not isinstance(ns, list):
            ns = []
            globals()["note_sections"] = ns

        ns.append("")
        ns.append("＜短評＞")
        ns.append(f"・出力生成中に例外が発生しました: {_e}")
        ns.append("判定：混戦")

    except Exception:
        pass

# =========================
# note用コピーエリア：全体妙味＋現行買い目
# =========================

note_text = "\n".join(note_sections)

st.markdown("### 📋 note用（コピーエリア）")

# -----------------------------------------
# 全体妙味判定用：◎〇△× 車番入力
# ※公開コピーには、市場名・外部名は出さない
# ※入力印は市場人気として妙味評価へ反映する
# -----------------------------------------
# 全体妙味判定用の市場印は、計算反映前に snapshot へ固定済み。
# ここでは再入力させず、反映済み値だけを使う。
market_honmei_raw = snapshot.get("market_honmei_raw", "—")
market_taikou_raw = snapshot.get("market_taikou_raw", "—")
market_tan_raw = snapshot.get("market_tan_raw", "—")
market_batsu_raw = snapshot.get("market_batsu_raw", "—")

def _to_car_int_or_none(v):
    try:
        s = str(v).strip()
        if not s or s == "—":
            return None
        x = int(s)
        return x if 1 <= x <= 9 else None
    except Exception:
        return None


market_honmei = _to_car_int_or_none(market_honmei_raw)
market_taikou = _to_car_int_or_none(market_taikou_raw)
market_tan = _to_car_int_or_none(market_tan_raw)
market_batsu = _to_car_int_or_none(market_batsu_raw)

# 車番→印。
# v20: 原則として、入力画面の「車番ごとの外部印」をそのまま使う。
# 以前は ◎/〇/△/× から車番へ圧縮した値だけで復元していたため、
# 車番別市場印と旧raw値が食い違わないよう、有効印を車番単位で固定する。
_market_mark_snapshot = snapshot.get("market_mark_by_car", {})
_VALID_MARKS_LOCAL = {"◎", "〇", "○", "△", "▲", "×"}

def _normalize_market_mark_local(_mark):
    _mk = str(_mark or "").strip()
    if _mk == "○":
        return "〇"
    if _mk == "▲":
        return "△"
    return _mk

market_mark_map = {}
# v47: market_mark_by_car は全車ぶん「—」を持つことがある。
# その場合に「dictがあるからfallbackしない」と、市場印が空扱いになって妙味ptが10のままになる。
# 先に車番ごとの有効印を拾い、その後で旧raw値も必ず補完する。
if isinstance(_market_mark_snapshot, dict) and _market_mark_snapshot:
    for _car, _mark in _market_mark_snapshot.items():
        try:
            _ci = int(_car)
        except Exception:
            continue
        _mk = _normalize_market_mark_local(_mark)
        if _mk in _VALID_MARKS_LOCAL:
            market_mark_map[_ci] = _mk

# v190:
# 反映後の計算では snapshot に保存した市場印だけを使う。
# ここで st.session_state の現在値を再取得すると、
# 「反映ボタンを押した固定値」と「画面上の未反映値」が混ざり、
# 市場印が空扱い/旧R扱いになって妙味ptが10.0に張り付く原因になる。

# 旧snapshot用・または market_mark_by_car が「—」だけだった時の補完。
# ここは setdefault ではなく、有効な旧rawがある場合は上書きする。
# ただし上の車番別radioが有効なら同じ内容になる。
for _car, _mark in [
    (market_honmei, "◎"),
    (market_taikou, "〇"),
    (market_tan, "△"),
    (market_batsu, "×"),
]:
    if _car is None:
        continue
    try:
        market_mark_map[int(_car)] = _mark
    except Exception:
        pass


def _uniq_keep(seq):
    out = []
    seen = set()
    for x in seq:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
    return out


def _find_line_members_of_car(line_def_obj, car):
    try:
        car = int(car)
        if isinstance(line_def_obj, dict):
            for _, mem in line_def_obj.items():
                mm = [int(x) for x in (mem or []) if str(x).isdigit()]
                if car in mm:
                    return mm
    except Exception:
        pass
    return []


def _find_line_members_of_car_from_note_text(note_text_obj, car):
    """
    line_def が globals に無い/取れない場合の保険。
    note本文の「ライン　73　16　524」から評価1の所属ラインを復元する。
    """
    try:
        car = int(car)
        txt = str(note_text_obj or "")
        m = re.search(r"^ライン\s+(.+)$", txt, flags=re.MULTILINE)
        if not m:
            return []
        part = m.group(1).strip()
        # 全角スペース・半角スペースで分割。数字以外は落とす。
        chunks = re.split(r"[\s　]+", part)
        for ch in chunks:
            nums = [int(x) for x in re.findall(r"\d", ch)]
            if car in nums:
                return nums
    except Exception:
        pass
    return []



def _parse_line_members_from_note_text(note_text_obj):
    """
    note本文の「ライン　416　27　3　5」から、ライン配列を復元する。
    返り値例：[[4,1,6], [2,7], [3], [5]]
    """
    try:
        txt = str(note_text_obj or "")
        m = re.search(r"^ライン\s+(.+)$", txt, flags=re.MULTILINE)
        if not m:
            return []
        part = m.group(1).strip()
        chunks = re.split(r"[\s　]+", part)
        out = []
        for ch in chunks:
            nums = []
            for x in re.findall(r"\d", ch):
                xi = int(x)
                if 1 <= xi <= 9 and xi not in nums:
                    nums.append(xi)
            if nums:
                out.append(nums)
        return out
    except Exception:
        return []


def _line_members_list_from_line_def(line_def_obj):
    """
    globals の line_def からライン配列を作る保険。
    """
    try:
        if isinstance(line_def_obj, dict):
            out = []
            for _, mem in line_def_obj.items():
                nums = []
                for x in (mem or []):
                    if str(x).isdigit():
                        xi = int(x)
                        if 1 <= xi <= 9 and xi not in nums:
                            nums.append(xi)
                if nums:
                    out.append(nums)
            return out
    except Exception:
        pass
    return []


def _rank_lines_by_order(line_members_list, order_seq):
    """
    各ラインを、推奨順/KO順で一番早く出る車を代表順位として並べる。
    評価順そのものではなく、ライン単位で列へ割り振るための基礎。
    """
    order = [int(x) for x in (order_seq or []) if str(x).isdigit()]
    pos = {car: i for i, car in enumerate(order)}

    def key(mem):
        best = min([pos.get(int(c), 999) for c in mem] or [999])
        # 同順位の保険として、ライン先頭の推奨位置も見る
        head_pos = pos.get(int(mem[0]), 999) if mem else 999
        return (best, head_pos, len(mem))

    return sorted([list(map(int, mem)) for mem in (line_members_list or []) if mem], key=key)



def _calc_overall_myoumi_score_label(col1_cars, col2_cars, role1, mark_map):
    """
    全体妙味を点数化する。

    基本思想：
    ・信頼度ではなく、市場印とのズレによる配当妙味を見る。
    ・軸候補と相手候補を分けて市場人気の偏りを評価する。
    ・1列目に市場印が付くほど人気寄りで妙味は下がりやすい。
    ・2列目だけの市場印は、相手人気として軽く減点する。
    ・評価1が無印なら、市場からのズレを妙味として加点する。

    妙味点 = 10
      - 1列目印減点
      - 2列目専用印減点
      + 評価1印補正
    """
    try:
        col1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
        col2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
        r1 = int(role1)
        mark_map = {int(k): str(v) for k, v in (mark_map or {}).items()}

        head_penalty = {"◎": 4.0, "〇": 3.0, "△": 1.5, "×": 0.75, "無印": 0.0}
        tail_penalty = {"◎": 2.0, "〇": 1.5, "△": 0.75, "×": 0.40, "無印": 0.0}

        score = 10.0
        role_marks = []

        # 1列目候補は頭として市場に売れやすいため強めに減点
        for car in col1:
            mk = mark_map.get(int(car), "無印")
            role_marks.append(mk)
            score -= head_penalty.get(mk, 0.0)

        # 2列目だけの候補は相手人気なので軽めに減点
        col1_set = set(col1)
        for car in col2:
            if int(car) in col1_set:
                continue
            mk = mark_map.get(int(car), "無印")
            role_marks.append(mk)
            score -= tail_penalty.get(mk, 0.0)

        # 評価1の印による補正。信頼度ではなく、市場とのズレを表す。
        r1_mark = mark_map.get(r1, "無印")
        r1_bonus_map = {"無印": 1.0, "×": 0.5, "△": 0.0, "〇": -0.5, "◎": -1.0}
        score += r1_bonus_map.get(r1_mark, 0.0)

        score = max(0.0, min(10.0, float(score)))

        # 表示ランクだけを調整。
        # 6.6点付近の荒れ寄りを最上位表示へ上げすぎない。
        # 8.0以上はズレすぎの荒領域として扱う。
        if score >= 8.0:
            label = "荒"
        elif score >= 6.8:
            label = "AA"
        elif score >= 5.5:
            label = "A"
        elif score >= 4.5:
            label = "B"
        elif score >= 3.5:
            label = "C"
        else:
            label = "低"

        return label, round(score, 1), role_marks

    except Exception:
        return "C", None, []



def _myoumi_eval1_bonus(car: int, role1: int, mark_map: dict) -> float:
    """
    評価1を1列目に置く場合だけ、市場ズレを内部基準に反映する。
    評価1が無印・×なら妙味を上げ、◎なら市場評価と一致しているので下げる。
    """
    try:
        if int(car) != int(role1):
            return 0.0
        mk = {int(k): str(v) for k, v in (mark_map or {}).items()}.get(int(role1), "無印")
        return {"無印": 1.0, "×": 0.5, "△": 0.0, "〇": -0.5, "◎": -1.0}.get(mk, 0.0)
    except Exception:
        return 0.0


def _myoumi_market_pair_penalty(marks) -> float:
    """
    2車系用の本線ペア追加減点。
    ◎〇のように市場評価ど真ん中の組み合わせは、的中候補ではあっても妙味候補ではないため強く落とす。
    """
    ms = {str(x or "無印") for x in marks}
    if "◎" in ms and "〇" in ms:
        return 1.5
    if "◎" in ms and "△" in ms:
        return 0.8
    if "〇" in ms and "△" in ms:
        return 0.6
    if "◎" in ms and "×" in ms:
        return 0.3
    if "〇" in ms and "×" in ms:
        return 0.2
    return 0.0


def _has_valid_market_marks_for_myoumi(mark_map: dict) -> bool:
    """市場印が1つ以上反映されているか。未反映時に妙味10.0張り付きを防ぐための保険。"""
    try:
        valid = {"◎", "〇", "○", "△", "▲", "×"}
        for v in (mark_map or {}).values():
            mk = str(v or "").strip()
            if mk in valid:
                return True
        return False
    except Exception:
        return False

def _resolve_market_mark_for_car_myoumi(car: int, mark_map: dict) -> str:
    """
    妙味計算専用の市場印取得。

    v190:
    反映済み snapshot から作った mark_map だけを信用する。
    st.session_state をここで再検索すると、未反映の画面値や別Rのradio値を拾い、
    反映ボタンの固定計算とズレるため使わない。
    """
    try:
        c = int(car)
    except Exception:
        return "無印"

    valid = {"◎", "〇", "○", "△", "▲", "×"}

    def norm(v):
        mk = str(v or "").strip()
        if mk == "○":
            mk = "〇"
        if mk == "▲":
            mk = "△"
        if mk in valid:
            return mk
        return "無印"

    try:
        mm = {int(k): norm(v) for k, v in (mark_map or {}).items()}
        return mm.get(c, "無印")
    except Exception:
        return "無印"


def _myoumi_score_2kei(a: int, b: int, role1: int, mark_map: dict) -> float:
    """
    加重2車複の妙味点計算用。
    a-b の入力順を保持する。
    実オッズではなく、外部印との被りから見た内部妙味pt。

    v50方針：
    ・2車複は「軸の印」より「相手の印の薄さ」を重視する。
      軸が◎でも、相手が無印なら妙味は残す。
    ・ただし、相手が△/〇/◎なら市場にも拾われているので明確に下げる。
    ・市場印を mark_map だけに頼らず、session_state からも再取得する。
    """
    mm = {int(k): str(v) for k, v in (mark_map or {}).items()}

    # v190:
    # 市場印が1つも反映されていない状態を「全車無印＝全買い目が超妙味」と誤解しない。
    # 反映失敗・未入力時は中立値に落とし、A++張り付きを防ぐ。
    if not _has_valid_market_marks_for_myoumi(mm):
        return 7.0

    ma = _resolve_market_mark_for_car_myoumi(int(a), mm)
    mb = _resolve_market_mark_for_car_myoumi(int(b), mm)

    # 2車複では、軸が市場◎でも「相手が売れていない」なら妙味は残る。
    # そのため head 減点は軽め、tail 減点は強めにする。
    head_penalty = {"◎": 0.8, "〇": 0.55, "△": 0.30, "×": 0.15, "無印": 0.0}
    tail_penalty = {"◎": 2.4, "〇": 1.7, "△": 1.2, "×": 0.45, "無印": 0.0}

    # 相手側の市場軽視を妙味として見る。
    tail_bonus = {"無印": 1.0, "×": 0.55, "△": 0.00, "〇": -0.35, "◎": -0.70}

    score = 9.2
    score -= head_penalty.get(ma, 0.0)
    score -= tail_penalty.get(mb, 0.0)
    score += tail_bonus.get(mb, 0.0)

    # ◎×△、◎×〇などは市場にも相手が見えているので追加で落とす。
    score -= _myoumi_market_pair_penalty([ma, mb]) * 0.75

    # 評価1が市場無印なら少しだけ上げ、◎なら少しだけ下げる。
    # 軸印だけで妙味を殺さないため係数は小さくする。
    score += 0.30 * _myoumi_eval1_bonus(int(a), int(role1), mm)

    return round(max(0.0, min(10.0, score)), 1)


# ==============================
# 全体妙味表示
# ==============================

def _display_overall_myoumi_label(label: str) -> str:
    """
    v119: 全体妙味の内部判定は旧ロジックのまま残し、表示だけA/B/Cへ丸める。
    旧「低」→ A
    旧「C」 → A
    旧「B」 → B
    旧「A」 → B
    旧「AA」→ C
    旧「荒」→ C
    """
    s = str(label or "").strip()
    if s in ("低", "C"):
        return "A"
    if s in ("B", "A"):
        return "B"
    if s in ("AA", "荒"):
        return "C"
    if s in ("A", "B", "C"):
        return s
    return "B"


def _display_overall_myoumi_labels_in_text(text: str) -> str:
    """本文内に残る全体妙味表記も、表示だけA/B/Cへ統一する。"""
    def repl(m):
        return "全体妙味：" + _display_overall_myoumi_label(m.group(1))
    return re.sub(r"全体妙味：(AA|A|B|C|荒|低)", repl, str(text))

# -----------------------------------------
# 推奨戦法とメイン着順予想を箱で強調表示
# -----------------------------------------
try:
    _rec_style = globals().get("RECOMMENDED_STYLE", "")
    _rec_seq = globals().get("RECOMMENDED_STYLE_SEQ", [])

    _rec_seq = [int(x) for x in (_rec_seq or []) if str(x).isdigit()]

    if _rec_style and _rec_seq:
        _rec_display_seq = " → ".join(str(int(x)) for x in _rec_seq)

        # v163: note用コピーエリア上部の青網掛けボックスは表示しない。
        # 推奨戦法・メイン着順は note_text 本文側に残す。
        pass

except Exception as _e:
    st.caption(f"推奨戦法表示生成不可：{_e}")


# -----------------------------------------
# 全体妙味表示（旧フォーメーション生成とは分離）
# -----------------------------------------
def _calc_current_overall_myoumi_label(note_body, rec_seq, line_def_obj, mark_map):
    """現行の全体妙味A/B/C表示に必要な軸・相手だけを組み立てる。"""
    default = ("C", None, [])
    try:
        rec_order = [int(x) for x in (rec_seq or []) if str(x).isdigit()]
        if len(rec_order) < 3:
            return default

        role1 = int(rec_order[0])
        line_members_all = _parse_line_members_from_note_text(note_body)
        if not line_members_all:
            line_members_all = _line_members_list_from_line_def(line_def_obj or {})
        ranked_lines = _rank_lines_by_order(line_members_all, rec_order)

        line_from_text = _find_line_members_of_car_from_note_text(note_body, role1)
        line_from_global = _find_line_members_of_car(line_def_obj or {}, role1)
        if line_from_text and role1 in [int(x) for x in line_from_text]:
            axis_line = [int(x) for x in line_from_text]
        else:
            axis_line = [int(x) for x in (line_from_global or [])]

        for members in ranked_lines:
            members = [int(x) for x in members]
            if role1 in members:
                axis_line = members
                break

        rec_pos = {int(car): idx for idx, car in enumerate(rec_order)}
        axis_others = [int(x) for x in axis_line if int(x) != role1]
        axis_others = sorted(
            axis_others,
            key=lambda car: (
                rec_pos.get(int(car), 999),
                axis_line.index(int(car)) if int(car) in axis_line else 999,
                int(car),
            ),
        )

        col1 = [role1]
        col2 = []
        if axis_others:
            col2.append(int(axis_others[0]))

        for members in ranked_lines:
            members = [int(x) for x in members]
            if not members or role1 in members:
                continue
            representative = next((int(car) for car in rec_order if int(car) in members), int(members[0]))
            if representative not in col1 and representative not in col2:
                col2.append(representative)
            if len(col2) >= 3:
                break

        for car in axis_others[1:]:
            if car not in col1 and car not in col2:
                col2.append(car)

        col2 = _uniq_keep(col2[:4])
        return _calc_overall_myoumi_score_label(col1, col2, role1, mark_map or {})
    except Exception:
        return default


overall_myoumi_label, _, _ = _calc_current_overall_myoumi_label(
    note_text,
    globals().get("RECOMMENDED_STYLE_SEQ", []),
    globals().get("line_def", {}),
    market_mark_map,
)


# -----------------------------------------
# note上部に実戦用サマリーを差し込む# -----------------------------------------
# note上部に実戦用サマリーを差し込む
# 詳細部は行単位で保存する
# v94: noteコピペ用は「最終推奨」中心に圧縮する。
#      旧買い目ブロック・会場H詳細ログはnoteへ出さない。
# -----------------------------------------
def _is_car_seri_involved_for_axis(_car):
    try:
        _car = int(_car)
        _seri_comment = globals().get("seri_comment", {}) or {}
        _seri_target = globals().get("seri_target", {}) or {}

        if bool(_seri_comment.get(_car, _seri_comment.get(str(_car), False))):
            return True

        for _src, _dst in (_seri_target or {}).items():
            try:
                if _dst is None or str(_dst).strip() in ("", "None", "—"):
                    continue
                if int(_dst) == _car:
                    return True
            except Exception:
                continue

        return False
    except Exception:
        return False


def _axis_line_follow_summary(_axis):
    try:
        _axis = int(_axis)
        _line_def = globals().get("line_def", {}) or {}
        _trust = globals().get("line_follow_trust", {}) or {}

        _members = []
        for _gid, _mem in (_line_def or {}).items():
            _mm = [int(x) for x in (_mem or []) if str(x).isdigit()]
            if _axis in _mm:
                _members = _mm
                break

        if len(_members) < 3:
            return "ライン後位：3番手以降なし", "normal"

        _thirds = [int(x) for x in _members[2:]]
        _labels = [str(_trust.get(int(x), _trust.get(str(int(x)), "通常")) or "通常") for x in _thirds]

        if any(x in ("流動", "単騎寄り") for x in _labels):
            return f"ライン後位：流動リスクあり（{','.join(str(x) for x in _thirds)}）", "weak"
        if any(x == "地区まとめ" for x in _labels):
            return f"ライン後位：地区まとめで結束弱め（{','.join(str(x) for x in _thirds)}）", "district"
        if any(x == "明確追走" for x in _labels):
            return f"ライン後位：明確追走あり（{','.join(str(x) for x in _thirds)}）", "strong"
        return f"ライン後位：通常追走（{','.join(str(x) for x in _thirds)}）", "normal"

    except Exception:
        return "ライン後位：未判定", "unknown"


def _make_axis_trust_judgement(seq):
    """
    評価1を安心軸にできるかの判定。
    材料：
    ・KO/H補正後の score_map の1位-2位差
    ・評価1/評価2の競り関与
    ・自力/自力自在/自在コメント
    ・3番手以降の追走信頼
    """
    try:
        xs = [int(x) for x in (seq or []) if str(x).isdigit()]
        if len(xs) < 2:
            return {
                "type": "未判定",
                "gap": None,
                "cap": "通常",
                "reasons": ["評価順不足"],
                "line_note": "ライン後位：未判定",
                "line_level": "unknown",
            }

        A, B = int(xs[0]), int(xs[1])
        score_map = globals().get("score_map", {}) or {}
        s1 = float(score_map.get(A, score_map.get(str(A), 0.0)) or 0.0)
        s2 = float(score_map.get(B, score_map.get(str(B), 0.0)) or 0.0)
        gap = s1 - s2

        jiryoku = globals().get("jiryoku_comment", {}) or {}
        jiryoku_jizai = globals().get("jiryoku_jizai_comment", {}) or {}
        jizai = globals().get("jizai_comment", {}) or {}
        single_comment = globals().get("single_comment", {}) or {}
        line_def = globals().get("line_def", {}) or {}

        a_seri = _is_car_seri_involved_for_axis(A)
        b_seri = _is_car_seri_involved_for_axis(B)
        a_jiryoku = bool(jiryoku.get(A, jiryoku.get(str(A), False)))
        a_jiryoku_jizai = bool(jiryoku_jizai.get(A, jiryoku_jizai.get(str(A), False)))
        a_jizai = bool(jizai.get(A, jizai.get(str(A), False)))
        a_single_comment = bool(single_comment.get(A, single_comment.get(str(A), False)))
        if a_jiryoku_jizai or (a_jiryoku and a_jizai):
            a_move_style = "自力自在"
        elif a_jiryoku:
            a_move_style = "自力"
        elif a_jizai:
            a_move_style = "自在"
        else:
            a_move_style = ""
        a_role = role_in_line(A, line_def) if isinstance(line_def, dict) else "single"
        line_note, line_level = _axis_line_follow_summary(A)

        reasons = []
        reasons.append(f"KO差={gap:.3f}")

        if a_seri:
            axis_type = "二強型・見送り寄り"
            cap = "ステップ1まで"
            reasons.append("評価1が競り関与")
        elif gap >= 0.220 and (a_move_style in ("自力", "自力自在", "自在") or a_role == "head"):
            axis_type = "1軸型"
            cap = "ステップ3まで"
            reasons.append("評価1が評価2を明確に上回る")
            if a_move_style:
                reasons.append(f"評価1に{a_move_style}コメント")
            elif a_role == "head":
                reasons.append("評価1がライン先頭")
            if a_single_comment and not a_move_style:
                reasons.append("評価1は単騎コメントのみ")
        elif gap >= 0.160 and b_seri and not a_seri:
            axis_type = "1軸寄り"
            cap = "ステップ2まで"
            reasons.append("評価2が競り関与")
        elif gap <= 0.080:
            axis_type = "混戦寄り"
            cap = "ステップ1まで"
            reasons.append("評価1・2のKO差が小さい")
        else:
            axis_type = "評価1・2二強型"
            cap = "ステップ2まで"
            reasons.append("評価1・2を並列評価")

        if line_level in ("weak", "district") and axis_type in ("1軸型", "1軸寄り"):
            # 評価1本人は軸でも、ライン丸抱えの信頼は落とす
            reasons.append("ライン後位の結束に不安")
            if axis_type == "1軸型":
                cap = "ステップ2まで"

        return {
            "type": axis_type,
            "gap": gap,
            "cap": cap,
            "reasons": reasons,
            "line_note": line_note,
            "line_level": line_level,
        }

    except Exception as e:
        return {
            "type": "未判定",
            "gap": None,
            "cap": "通常",
            "reasons": [f"判定不可:{e}"],
            "line_note": "ライン後位：未判定",
            "line_level": "unknown",
        }


def _parse_santan_reference_triplet(ref_text):
    """3単参考「A→B→C」を3車のtupleへ変換。厳密に3車取れない場合はNone。"""
    try:
        nums = [int(x) for x in re.findall(r"\d+", str(ref_text or ""))]
        if len(nums) != 3 or len(set(nums)) != 3:
            return None
        return tuple(nums)
    except Exception:
        return None


def _v262_unique_flow_sequence(seq, active_cars=None):
    """流れ別着順を、出走車だけ・重複なしで正規化する。"""
    active = None
    try:
        if active_cars is not None:
            active = {int(x) for x in active_cars if str(x).isdigit()}
    except Exception:
        active = None

    out = []
    seen = set()
    for x in (seq or []):
        try:
            car = int(x)
        except Exception:
            continue
        if active is not None and car not in active:
            continue
        if car in seen:
            continue
        seen.add(car)
        out.append(car)
    return out


def _v262_ranked_flows(flow_ratio_map, style_seq_map, active_cars=None, preferred_style=""):
    """
    v261互換：順流・逆流・渦を比率降順に並べる。
    同率時は現在の推奨流れ、その後は順流・逆流・渦の固定順を優先する。
    """
    style_order = ["順流", "逆流", "渦"]
    ratios = dict(flow_ratio_map or {})
    seq_map = dict(style_seq_map or {})
    preferred = str(preferred_style or "")
    rows = []
    for idx, style in enumerate(style_order):
        try:
            ratio = float(ratios.get(style, 0.0) or 0.0)
        except Exception:
            ratio = 0.0
        rows.append({
            "style": style,
            "ratio": ratio,
            "seq": _v262_unique_flow_sequence(seq_map.get(style, []) or [], active_cars),
            "fixed_index": idx,
        })
    rows.sort(key=lambda row: (
        -float(row.get("ratio", 0.0) or 0.0),
        0 if str(row.get("style", "")) == preferred else 1,
        int(row.get("fixed_index", 99)),
    ))
    return rows


def _v262_select_second_third_flow_five_plan(
    flow_ratio_map,
    style_seq_map,
    active_cars=None,
    preferred_style="",
):
    """
    v261の3連複候補選別を5車の役割順で返す。

    ・比率1位の流れは軸候補の生成から除外。
    ・比率2位・3位の順位合計が最小の車を1（同点は2位流れ順位→3位流れ順位）。
    ・残る2～5は、比率2位流れ→3位流れの同順位を交互に見て4車選ぶ。
    ・流れ1位の車が2位・3位流れ側にも評価されていれば、4・5等のヒモには入り得る。
    """
    ranked = [
        row for row in _v262_ranked_flows(
            flow_ratio_map,
            style_seq_map,
            active_cars=active_cars,
            preferred_style=preferred_style,
        )
        if row.get("seq")
    ]
    if len(ranked) < 3:
        return None

    excluded = ranked[0]
    primary, secondary = ranked[1], ranked[2]
    seq1 = list(primary.get("seq") or [])
    seq2 = list(secondary.get("seq") or [])
    candidates = []
    for seq in (seq1, seq2):
        for car in seq:
            car = int(car)
            if car not in candidates:
                candidates.append(car)
    if active_cars is not None:
        for car in _v262_unique_flow_sequence(active_cars):
            if car not in candidates:
                candidates.append(int(car))
    if len(candidates) < 5:
        return None

    miss1 = len(seq1) + len(candidates) + 1
    miss2 = len(seq2) + len(candidates) + 1
    pos1 = {int(car): idx + 1 for idx, car in enumerate(seq1)}
    pos2 = {int(car): idx + 1 for idx, car in enumerate(seq2)}
    axis = min(
        candidates,
        key=lambda car: (
            int(pos1.get(int(car), miss1)) + int(pos2.get(int(car), miss2)),
            int(pos1.get(int(car), miss1)),
            int(pos2.get(int(car), miss2)),
            int(car),
        ),
    )

    opponents = []
    max_len = max(len(seq1), len(seq2))
    for idx in range(max_len):
        for seq in (seq1, seq2):
            if idx >= len(seq):
                continue
            car = int(seq[idx])
            if car == int(axis) or car in opponents:
                continue
            opponents.append(car)
            if len(opponents) >= 4:
                break
        if len(opponents) >= 4:
            break

    if len(opponents) < 4:
        for car in candidates:
            car = int(car)
            if car == int(axis) or car in opponents:
                continue
            opponents.append(car)
            if len(opponents) >= 4:
                break
    if len(opponents) != 4:
        return None

    return {
        "cars": (int(axis),) + tuple(int(x) for x in opponents),
        "axis": int(axis),
        "opponents": tuple(int(x) for x in opponents),
        "styles": (str(primary.get("style")), str(secondary.get("style"))),
        "ratios": (float(primary.get("ratio", 0.0)), float(secondary.get("ratio", 0.0))),
        "excluded_style": str(excluded.get("style", "")),
        "excluded_ratio": float(excluded.get("ratio", 0.0)),
        "ranked_flows": tuple(ranked),
    }



def _v264_best_live_line_groups(line_sources, active_cars=None):
    """実行中ライン情報から、出走車の被覆が最も大きいライン集合を1組選ぶ。"""
    try:
        active = {int(x) for x in (active_cars or []) if str(x).isdigit()}
    except Exception:
        active = set()

    best_groups = []
    best_key = (-1, -1, -1)
    for source in (line_sources or []):
        try:
            raw_groups = list(source.values()) if isinstance(source, dict) else list(source or [])
        except Exception:
            raw_groups = []

        groups = []
        covered = set()
        seen = set()
        for raw in raw_groups:
            try:
                cars = []
                for x in (raw or []):
                    if not str(x).isdigit():
                        continue
                    car = int(x)
                    if active and car not in active:
                        continue
                    if car not in cars:
                        cars.append(car)
            except Exception:
                cars = []
            if not cars:
                continue
            key = tuple(cars)
            if key in seen:
                continue
            seen.add(key)
            groups.append(cars)
            covered.update(cars)

        score_key = (len(covered), sum(1 for g in groups if len(g) >= 2), len(groups))
        if score_key > best_key:
            best_key = score_key
            best_groups = groups

    covered = {int(x) for g in best_groups for x in g}
    for car in sorted(active):
        if int(car) not in covered:
            best_groups.append([int(car)])
    return best_groups


def _v264_line_diverse_five_plan(plan, line_sources, active_cars=None, ko_score_map=None):
    """
    v266：比率2位・3位流れ用 12-123-12345 の5車を、表示どおりに選ぶ。

    厳守事項
    ・比率1位の除外流れ代表ラインはA～Eへ入れない。
    ・A/Bは使用する2流れの代表ラインから1車ずつ。物理ラインも必ず別。
    ・Cも使用する2流れの代表ライン内から選ぶ。
    ・「その他」ラインはD/E（3列目）だけで補完できる。
    ・KO使用スコアは「その他」候補のD/E順位付けだけに使い、
      A/B/Cや使用流れ内の並びを上書きしない。
    ・条件を満たす5車を作れない場合はNoneを返し、矛盾した7点を生成しない。
    """
    if not plan:
        return None

    try:
        active = []
        for x in (active_cars or []):
            car = int(x)
            if car not in active:
                active.append(car)
    except Exception:
        active = []
    if len(active) < 5:
        return None

    line_groups = _v264_best_live_line_groups(line_sources, active_cars=active)
    if not line_groups:
        return None

    zone_map = globals().get("LINE_ZONE_MAP", {}) or {}
    if not isinstance(zone_map, dict) or not zone_map:
        # ライン分類が取れない状態で、スコア順へ黙ってフォールバックしない。
        return None

    def _line_key(group):
        return "".join(str(int(x)) for x in (group or []) if str(x).isdigit())

    records = []
    covered = set()
    for group in line_groups:
        cars = []
        for x in (group or []):
            try:
                car = int(x)
            except Exception:
                continue
            if car in active and car not in cars:
                cars.append(car)
        if not cars:
            continue
        key = _line_key(cars)
        zone = str(zone_map.get(key, "その他") or "その他")
        records.append({"key": key, "cars": cars, "zone": zone})
        covered.update(cars)

    # ライン未所属車は単騎の「その他」として扱う。
    for car in active:
        if car not in covered:
            records.append({"key": str(car), "cars": [car], "zone": "その他"})

    ranked_flows = list(plan.get("ranked_flows", tuple()) or tuple())
    style_to_seq = {}
    for row in ranked_flows:
        style = str((row or {}).get("style", "") or "")
        seq = _v262_unique_flow_sequence((row or {}).get("seq", []) or [], active)
        if style and seq:
            style_to_seq[style] = seq

    styles = tuple(str(x) for x in (plan.get("styles", tuple()) or tuple()))
    if len(styles) != 2 or styles[0] == styles[1]:
        return None
    excluded_style = str(plan.get("excluded_style", "") or "")

    def _representative_record(style):
        matches = [rec for rec in records if rec.get("zone") == style]
        if not matches:
            return None
        # v265で各流れ1代表のはず。複数残っても最初の代表だけを採用する。
        return matches[0]

    rec1 = _representative_record(styles[0])
    rec2 = _representative_record(styles[1])
    if rec1 is None or rec2 is None or rec1.get("key") == rec2.get("key"):
        return None

    def _ordered_line_cars(style, rec):
        line_set = set(int(x) for x in (rec.get("cars") or []))
        ordered = [int(x) for x in style_to_seq.get(style, []) if int(x) in line_set]
        for car in (rec.get("cars") or []):
            car = int(car)
            if car in active and car not in ordered:
                ordered.append(car)
        return ordered

    order1 = _ordered_line_cars(styles[0], rec1)
    order2 = _ordered_line_cars(styles[1], rec2)
    if not order1 or not order2:
        return None

    # A/B：使用2流れの代表ライン先頭を1車ずつ。
    selected = [int(order1[0]), int(order2[0])]
    source_by_car = {
        int(order1[0]): f"{styles[0]}:{rec1.get('key')}",
        int(order2[0]): f"{styles[1]}:{rec2.get('key')}",
    }

    # C以降の使用流れ候補は、各流れ内順位を交互にたどる。
    # KOスコアでは並べ替えない。
    merged_core = []
    max_len = max(len(order1), len(order2))
    for idx in range(1, max_len):
        for style, rec, order in ((styles[0], rec1, order1), (styles[1], rec2, order2)):
            if idx >= len(order):
                continue
            car = int(order[idx])
            if car in selected or car in merged_core:
                continue
            merged_core.append(car)
            source_by_car[car] = f"{style}:{rec.get('key')}"

    # Cは必ず使用2流れの代表ライン内から。
    if not merged_core:
        return None
    selected.append(int(merged_core.pop(0)))

    # D/Eはまず使用2流れ内の残りを採用。
    for car in merged_core:
        if car not in selected:
            selected.append(int(car))
        if len(selected) >= 5:
            break

    # 足りない場合だけ「その他」ラインをD/Eへ補完。
    score_map = {}
    for k, v in (ko_score_map or {}).items():
        try:
            score_map[int(k)] = float(v)
        except Exception:
            pass

    other_candidates = []
    for rec in records:
        if str(rec.get("zone")) != "その他":
            continue
        for car in (rec.get("cars") or []):
            car = int(car)
            if car in selected or car in other_candidates:
                continue
            other_candidates.append(car)
            source_by_car[car] = f"その他:{rec.get('key')}"
    other_candidates.sort(
        key=lambda car: (float(score_map.get(int(car), 0.0)), -int(car)),
        reverse=True,
    )
    for car in other_candidates:
        if len(selected) >= 5:
            break
        selected.append(int(car))

    # 除外流れ代表ラインを使わなければ5車に届かない場合は、生成自体を中止する。
    if len(selected) != 5 or len(set(selected)) != 5:
        return None

    excluded_cars = {
        int(car)
        for rec in records
        if str(rec.get("zone")) == excluded_style
        for car in (rec.get("cars") or [])
    }
    if excluded_cars.intersection(selected):
        return None

    # A/B/Cへ「その他」が混入していないことを最終監査。
    if any(str(source_by_car.get(int(car), "")).startswith("その他:") for car in selected[:3]):
        return None
    if not str(source_by_car.get(selected[0], "")).startswith(f"{styles[0]}:"):
        return None
    if not str(source_by_car.get(selected[1], "")).startswith(f"{styles[1]}:"):
        return None

    out = dict(plan)
    out["cars"] = tuple(int(x) for x in selected)
    out["axis"] = int(selected[0])
    out["opponents"] = tuple(int(x) for x in selected[1:])
    out["line_diversified"] = True
    out["strict_flow_consistency"] = True
    out["axis_source_style"] = styles[0]
    out["second_source_style"] = styles[1]
    out["source_by_car"] = dict(source_by_car)
    out["selected_line_keys"] = (str(rec1.get("key")), str(rec2.get("key")))
    out["excluded_flow_cars_used"] = tuple()
    out["other_third_cars"] = tuple(
        int(car) for car in selected[3:]
        if str(source_by_car.get(int(car), "")).startswith("その他:")
    )
    out["first_column_lines_separated"] = str(rec1.get("key")) != str(rec2.get("key"))
    out["second_column_core_only"] = True
    return out


def _v267_select_three_flow_line_five_plan(
    flow_ratio_map,
    style_seq_map,
    line_sources,
    active_cars=None,
    preferred_style="",
    ko_score_map=None,
):
    """
    v267：男子の3連単非該当時に使う 12-123-12345 の5車を、
    三流れ・三ラインの役割から作る。

    A：比率1位流れの代表ライン先頭
    B：比率2位流れの代表ライン先頭
    C：比率3位流れの代表ライン先頭
    D/E：三代表ラインの後位、または「その他（3列目候補）」だけ

    重要：
    ・比率1位を除外しない。
    ・A/B/CはKOスコア順へ置き換えない。
    ・D/Eだけ、役割制約を守った候補内でKOスコアを補助順位に使う。
    ・D/Eは可能な限り別ラインに分散する。
    ・条件を満たせなければNoneを返し、説明と異なる7点を作らない。
    """
    try:
        active = []
        for x in (active_cars or []):
            car = int(x)
            if car not in active:
                active.append(car)
    except Exception:
        active = []
    if len(active) < 5:
        return None

    ranked = [
        row for row in _v262_ranked_flows(
            flow_ratio_map,
            style_seq_map,
            active_cars=active,
            preferred_style=preferred_style,
        )
        if row.get("seq")
    ]
    if len(ranked) < 3:
        return None
    ranked = ranked[:3]

    line_groups = _v264_best_live_line_groups(line_sources, active_cars=active)
    if not line_groups:
        return None

    zone_map = globals().get("LINE_ZONE_MAP", {}) or {}
    if not isinstance(zone_map, dict) or not zone_map:
        return None

    def _line_key(group):
        return "".join(str(int(x)) for x in (group or []) if str(x).isdigit())

    records = []
    covered = set()
    for group in line_groups:
        cars = []
        for x in (group or []):
            try:
                car = int(x)
            except Exception:
                continue
            if car in active and car not in cars:
                cars.append(car)
        if not cars:
            continue
        key = _line_key(cars)
        zone = str(zone_map.get(key, "その他") or "その他")
        records.append({"key": key, "cars": cars, "zone": zone})
        covered.update(cars)

    for car in active:
        if car not in covered:
            records.append({"key": str(car), "cars": [car], "zone": "その他"})

    style_to_seq = {
        str(row.get("style", "")): _v262_unique_flow_sequence(row.get("seq", []) or [], active)
        for row in ranked
    }

    def _representative_record(style):
        matches = [rec for rec in records if str(rec.get("zone", "")) == str(style)]
        return matches[0] if matches else None

    def _ordered_line_cars(style, rec):
        line_set = {int(x) for x in (rec.get("cars") or [])}
        ordered = [int(x) for x in style_to_seq.get(style, []) if int(x) in line_set]
        for car in (rec.get("cars") or []):
            car = int(car)
            if car in active and car not in ordered:
                ordered.append(car)
        return ordered

    flow_roles = []
    representative_line_keys = set()
    representative_cars = []
    source_by_car = {}

    for flow_rank, row in enumerate(ranked, start=1):
        style = str(row.get("style", "") or "")
        rec = _representative_record(style)
        if rec is None:
            return None
        line_key = str(rec.get("key", "") or "")
        if not line_key or line_key in representative_line_keys:
            return None
        ordered = _ordered_line_cars(style, rec)
        if not ordered:
            return None
        rep_car = int(ordered[0])
        if rep_car in representative_cars:
            return None
        representative_line_keys.add(line_key)
        representative_cars.append(rep_car)
        source_by_car[rep_car] = f"{style}:{line_key}（代表）"
        flow_roles.append({
            "rank": flow_rank,
            "style": style,
            "ratio": float(row.get("ratio", 0.0) or 0.0),
            "record": rec,
            "line_key": line_key,
            "ordered": ordered,
            "representative": rep_car,
        })

    # A/B/Cは三流れの代表3車で固定。ここへKO順位を介入させない。
    selected = list(representative_cars)

    score_map = {}
    for k, v in (ko_score_map or {}).items():
        try:
            score_map[int(k)] = float(v)
        except Exception:
            pass

    support_candidates = []
    seen_support = set()

    # 三代表ラインの後位候補。
    for role in flow_roles:
        for depth, car in enumerate(role["ordered"][1:], start=1):
            car = int(car)
            if car in selected or car in seen_support:
                continue
            seen_support.add(car)
            support_candidates.append({
                "car": car,
                "line_key": role["line_key"],
                "style": role["style"],
                "kind": "代表ライン後位",
                "flow_rank": int(role["rank"]),
                "depth": int(depth),
                "score": float(score_map.get(car, 0.0)),
            })

    # 「その他」はD/Eだけの候補。A/B/Cには絶対に入れない。
    for rec in records:
        if str(rec.get("zone", "")) != "その他":
            continue
        other_cars = []
        for x in (rec.get("cars") or []):
            try:
                car = int(x)
            except Exception:
                continue
            if car in active and car not in selected and car not in seen_support:
                other_cars.append(car)
        other_cars.sort(key=lambda car: (-float(score_map.get(int(car), 0.0)), int(car)))
        for depth, car in enumerate(other_cars, start=1):
            seen_support.add(car)
            support_candidates.append({
                "car": int(car),
                "line_key": str(rec.get("key", "") or str(car)),
                "style": "その他",
                "kind": "その他",
                "flow_rank": 99,
                "depth": int(depth),
                "score": float(score_map.get(int(car), 0.0)),
            })

    if len(support_candidates) < 2:
        return None

    # D/Eは制約済み候補内でKOスコアを補助順位にする。
    # 同点・スコア欠落時は、代表ライン後位→流れ順位→ライン内順位を優先。
    support_candidates.sort(key=lambda item: (
        -float(item.get("score", 0.0)),
        0 if str(item.get("kind")) == "代表ライン後位" else 1,
        int(item.get("flow_rank", 99)),
        int(item.get("depth", 99)),
        int(item.get("car", 99)),
    ))

    d = support_candidates[0]
    e_pool = [
        item for item in support_candidates[1:]
        if str(item.get("line_key", "")) != str(d.get("line_key", ""))
    ]
    e = e_pool[0] if e_pool else support_candidates[1]

    selected.extend([int(d["car"]), int(e["car"])])
    if len(selected) != 5 or len(set(selected)) != 5:
        return None

    for item in (d, e):
        car = int(item["car"])
        source_by_car[car] = (
            f"{item['style']}:{item['line_key']}（{item['kind']}）"
        )

    styles = tuple(str(role["style"]) for role in flow_roles)
    ratios = tuple(float(role["ratio"]) for role in flow_roles)
    line_keys = tuple(str(role["line_key"]) for role in flow_roles)

    return {
        "cars": tuple(int(x) for x in selected),
        "axis": int(selected[0]),
        "opponents": tuple(int(x) for x in selected[1:]),
        "styles": styles,
        "ratios": ratios,
        "ranked_flows": tuple(ranked),
        "source_by_car": dict(source_by_car),
        "representative_cars": tuple(int(x) for x in selected[:3]),
        "representative_line_keys": line_keys,
        "support_cars": tuple(int(x) for x in selected[3:]),
        "support_line_keys": (str(d["line_key"]), str(e["line_key"])),
        "other_third_cars": tuple(
            int(item["car"]) for item in (d, e)
            if str(item.get("kind")) == "その他"
        ),
        "first_column_lines_separated": line_keys[0] != line_keys[1],
        "second_column_three_flows": len(set(styles)) == 3,
        "second_column_three_lines": len(set(line_keys)) == 3,
        "support_lines_separated": str(d["line_key"]) != str(e["line_key"]),
        "selection_rule": "A/B/C=三流れ代表、D/E=代表ライン後位またはその他",
    }


def _v262_trio_key_from_row(row):
    """既存の加重3連複評価行から車番3車のキーを取り出す。"""
    try:
        cars = [int(x) for x in ((row or {}).get("cars") or []) if str(x).isdigit()]
        if len(cars) == 3 and len(set(cars)) == 3:
            return tuple(sorted(cars))
    except Exception:
        pass
    try:
        cars = [int((row or {}).get(k)) for k in ("a", "b", "c")]
        if len(set(cars)) == 3:
            return tuple(sorted(cars))
    except Exception:
        pass
    try:
        nums = [int(x) for x in re.findall(r"\d+", str((row or {}).get("disp", "")))]
        if len(nums) == 3 and len(set(nums)) == 3:
            return tuple(sorted(nums))
    except Exception:
        pass
    return tuple()


def _v262_trio_row_map(trio_rows):
    """同じ3車の評価行が複数ある場合は、既存評価の高い行を残す。"""
    row_map = {}
    for row in (trio_rows or []):
        key = _v262_trio_key_from_row(row)
        if not key:
            continue
        old = row_map.get(key)
        score = (
            float((row or {}).get("total_pt", 0.0) or 0.0),
            float((row or {}).get("hit_score", 0.0) or 0.0),
            float((row or {}).get("myoumi_score", 0.0) or 0.0),
        )
        old_score = (
            float((old or {}).get("total_pt", 0.0) or 0.0),
            float((old or {}).get("hit_score", 0.0) or 0.0),
            float((old or {}).get("myoumi_score", 0.0) or 0.0),
        ) if old is not None else None
        if old is None or old_score is None or score > old_score:
            row_map[key] = row
    return row_map


def _v262_rows_for_12_123_12345(trio_rows, five_cars):
    """実車番5車を3連複12-123-12345の7点へ展開する。"""
    try:
        a, b, c, d, e = [int(x) for x in (five_cars or [])]
    except Exception:
        return []
    if len({a, b, c, d, e}) != 5:
        return []
    combos = (
        (a, b, c),
        (a, b, d),
        (a, b, e),
        (a, c, d),
        (a, c, e),
        (b, c, d),
        (b, c, e),
    )
    row_map = _v262_trio_row_map(trio_rows)
    out = []
    for combo in combos:
        key = tuple(sorted(combo))
        row = row_map.get(key)
        if row is None:
            return []
        cloned = dict(row or {})
        cloned["cars"] = key
        cloned["a"], cloned["b"], cloned["c"] = key
        cloned["disp"] = "-".join(str(x) for x in key)
        out.append(cloned)
    return out


def _v262_form_12_123_12345(five_cars):
    try:
        a, b, c, d, e = [int(x) for x in (five_cars or [])]
    except Exception:
        return ""
    if len({a, b, c, d, e}) != 5:
        return ""
    return f"{a}{b}-{a}{b}{c}-{a}{b}{c}{d}{e}"


def _v262_build_santan_plus_trio_plan(
    first,
    second,
    candidate_rows,
    all_trio_rows,
    protected_third=None,
):
    """
    3連単該当レースを4点＋2点へ再構成する。

    3連単：AB-ABC-ABC
      A→B→C / A→C→B / B→A→C / B→C→A
    3連複：A-B-DE
      A-B-D / A-B-E

    Cは直後同ライン保護車を優先。存在しない2車ライン等では、
    既存のライン3連複候補で評価最上位の第三車を使う。
    """
    try:
        first, second = int(first), int(second)
    except Exception:
        return None
    if first == second:
        return None

    thirds = []
    source_rows = list(candidate_rows or [])
    # candidate_rowsで不足した場合も、同じA・Bを含む既存評価行だけから補う。
    for rows in (source_rows, list(all_trio_rows or [])):
        for row in rows:
            key = _v262_trio_key_from_row(row)
            if len(key) != 3 or first not in key or second not in key:
                continue
            rest = [int(x) for x in key if int(x) not in {first, second}]
            if len(rest) == 1 and rest[0] not in thirds:
                thirds.append(rest[0])
            if len(thirds) >= 3:
                break
        if len(thirds) >= 3:
            break
    if len(thirds) < 3:
        return None

    core = None
    try:
        p = int(protected_third)
        if p in thirds and p not in {first, second}:
            core = p
    except Exception:
        core = None
    if core is None:
        core = int(thirds[0])

    support = [int(x) for x in thirds if int(x) != int(core)][:2]
    if len(support) != 2:
        return None
    d, e = support

    row_map = _v262_trio_row_map(all_trio_rows)
    support_rows = []
    for third in (d, e):
        row = row_map.get(tuple(sorted((first, second, int(third)))))
        if row is None:
            return None
        cloned = dict(row or {})
        key = tuple(sorted((first, second, int(third))))
        cloned["cars"] = key
        cloned["a"], cloned["b"], cloned["c"] = key
        cloned["disp"] = "-".join(str(x) for x in key)
        support_rows.append(cloned)

    tickets = (
        f"{first}→{second}→{core}",
        f"{first}→{core}→{second}",
        f"{second}→{first}→{core}",
        f"{second}→{core}→{first}",
    )
    return {
        "santan_form": f"{first}{second}-{first}{second}{core}-{first}{second}{core}",
        "santan_tickets": tickets,
        "core_third": int(core),
        "support_thirds": (int(d), int(e)),
        "support_trio_form": f"{first}-{second}-{d}{e}",
        "support_trio_rows": tuple(support_rows),
    }


def _rows_average_strength_key(rows):
    """
    同じ点数の候補群を、既存の加重評価だけで比較する。

    比較順：平均総合点 → 平均的中点 → 平均妙味点。
    サイドバーの得点・S/H/B・決まり手・着順成績・コメント等は、
    既存の加重評価へ反映済みなので、ここでは新しい閾値を設けない。
    """
    vals = []
    for row in list(rows or []):
        try:
            vals.append((
                float((row or {}).get("total_pt", 0.0) or 0.0),
                float((row or {}).get("hit_score", 0.0) or 0.0),
                float((row or {}).get("myoumi_score", 0.0) or 0.0),
            ))
        except Exception:
            pass
    if not vals:
        return tuple()
    n = float(len(vals))
    return (
        sum(v[0] for v in vals) / n,
        sum(v[1] for v in vals) / n,
        sum(v[2] for v in vals) / n,
    )


def _choose_final_trio_structure_by_sidebar_power(base_structure, line_rows, nonline_rows):
    """
    v270-R2 診断用の加重比較。

    ライン主体／非ライン主体の候補評価を比較して表示理由へ残すだけで、
    展開構造・券種・元5車を上書きしない。返却するstructureは旧呼び出しとの
    互換用であり、最終判定には使用しない。
    """
    base_structure = str(base_structure or "非ライン主体")
    line_key = _rows_average_strength_key(line_rows)
    nonline_key = _rows_average_strength_key(nonline_rows)

    if line_key and nonline_key:
        if line_key > nonline_key:
            return {
                "structure": "ライン主体",
                "reason": "最終加重3連複評価でライン主体が上位",
                "line_power_key": line_key,
                "nonline_power_key": nonline_key,
            }
        if nonline_key > line_key:
            return {
                "structure": "非ライン主体",
                "reason": "最終加重3連複評価で非ライン主体が上位",
                "line_power_key": line_key,
                "nonline_power_key": nonline_key,
            }
        return {
            "structure": base_structure,
            "reason": "最終加重3連複評価が同点のため従来構造を維持",
            "line_power_key": line_key,
            "nonline_power_key": nonline_key,
        }

    if line_key:
        return {
            "structure": "ライン主体",
            "reason": "ライン主体候補のみ生成可能",
            "line_power_key": line_key,
            "nonline_power_key": nonline_key,
        }
    if nonline_key:
        return {
            "structure": "非ライン主体",
            "reason": "非ライン主体候補のみ生成可能",
            "line_power_key": line_key,
            "nonline_power_key": nonline_key,
        }

    return {
        "structure": base_structure,
        "reason": "最終加重3連複評価を取得できないため従来構造を維持",
        "line_power_key": line_key,
        "nonline_power_key": nonline_key,
    }


def _decide_ticket_from_structure_and_santan_refs(
    structure,
    trio_rows,
    pair_rows=None,
    protected_third_candidates=None,
):
    """
    v270-R2 券種判定（実オッズ・新規数値閾値は使わない）。

    ・非ライン主体：着順を固定できないため3連単対象外。
      旧2車複／3連複平均比較で券種を変えず、後段の元5車三連複7点へ送る。
    ・ライン主体：採用3連複3点の3単参考を確認
      - 3点すべての1着・2着が同一 → 3着2車へ1・2着折り返し4点の候補
      - 1・2着候補の直後に続く同ライン車が3着候補内にいる場合、その1車を必ず保護
      - それ以外 → 後段の元5車三連複7点へ送る
    """
    structure = str(structure or "")
    rows = list(trio_rows or [])
    pair_rows = list(pair_rows or [])
    protected_third_set = {
        int(x) for x in (protected_third_candidates or [])
        if str(x).isdigit()
    }

    if structure != "ライン主体":
        # 加重2車複／3連複の比較値は診断用に保持するが、券種は変えない。
        # 非ライン主体では着順固定ができないため、後段の本来のv270
        # 「元5車を一車も切らない三連複7点」へ必ず送る。
        trio_key = _rows_average_strength_key(rows)
        pair_key = _rows_average_strength_key(pair_rows)
        return {
            "recommended_ticket": "3連複",
            "ticket_reason": "非ライン主体で着順を固定できないため3連単対象外。元5車三連複7点へ送る",
            "santan_form": "",
            "santan_tickets": tuple(),
            "santan_common_first_second": tuple(),
            "pair_power_key": pair_key,
            "trio_power_key": trio_key,
        }

    refs = []
    ref_rows = []
    for row in rows:
        parsed = _parse_santan_reference_triplet((row or {}).get("santan_ref", ""))
        if parsed is None:
            return {
                "recommended_ticket": "3連複",
                "ticket_reason": "ライン主体だが、3単参考3点の共通1・2着を確認できないため順不同を優先",
                "santan_form": "",
                "santan_tickets": tuple(),
                "santan_common_first_second": tuple(),
                "pair_power_key": tuple(),
                "trio_power_key": _rows_average_strength_key(rows),
            }
        refs.append(parsed)
        ref_rows.append(row)

    if len(refs) == 3:
        first_second = refs[0][:2]
        same_first_second = all(ref[:2] == first_second for ref in refs)
        if same_first_second:
            # rowsは既存の加重3連複総合点順。
            # まず3着候補を順位どおり重複なしで並べる。
            ranked_thirds = []
            for ref in refs:
                third = int(ref[2])
                if third not in ranked_thirds:
                    ranked_thirds.append(third)

            original_top_thirds = list(ranked_thirds[:2])
            protected_available = [
                third for third in ranked_thirds
                if third in protected_third_set and third not in first_second
            ]
            protection_applied = False

            # v255：ライン保護は「残り同ライン車なら誰でもよい」ではなく、
            # 1・2着候補の直後に続く最優先同ライン車1車を固定する。
            # 残る1枠だけを、4番手以降の同ライン車と他ライン候補を含む
            # 既存の加重3連複総合点順から選ぶ。新しい数値閾値は置かない。
            if protected_available:
                protected_third = int(protected_available[0])
                other_thirds = [
                    int(third) for third in ranked_thirds
                    if int(third) != protected_third
                    and int(third) not in first_second
                ]
                top_thirds = [protected_third]
                if other_thirds:
                    top_thirds.append(int(other_thirds[0]))
                protection_applied = top_thirds != original_top_thirds
            else:
                top_thirds = list(original_top_thirds)

            if len(top_thirds) == 2 and len(set(top_thirds)) == 2:
                first, second = int(first_second[0]), int(first_second[1])
                c, d = int(top_thirds[0]), int(top_thirds[1])
                tickets = (
                    f"{first}→{second}→{c}",
                    f"{first}→{second}→{d}",
                    f"{second}→{first}→{c}",
                    f"{second}→{first}→{d}",
                )
                form = f"{first}{second}-{first}{second}-{c}{d}"
                if protected_available:
                    if protection_applied:
                        reason = "ライン主体で3単参考3点の1・2着が共通。直後の同ライン車を3着に固定保護し、残る1枠を4番手以降と他ライン候補の加重評価で選んで1・2着折り返し"
                    else:
                        reason = "ライン主体で3単参考3点の1・2着が共通。直後の同ライン車が3着上位内にあり、残る1枠と合わせて1・2着折り返し"
                else:
                    reason = "ライン主体で3単参考3点の1・2着が共通。3着上位2車へ1・2着折り返し"
                return {
                    "recommended_ticket": "3連単",
                    "ticket_reason": reason,
                    "santan_form": form,
                    "santan_tickets": tickets,
                    "santan_common_first_second": (first, second),
                    "santan_protected_third": int(protected_available[0]) if protected_available else None,
                    "santan_line_protection_applied": bool(protection_applied),
                    "pair_power_key": tuple(),
                    "trio_power_key": _rows_average_strength_key(rows),
                }

    return {
        "recommended_ticket": "3連複",
        "ticket_reason": "ライン主体だが、3単参考3点の1着または2着が割れるため順不同を優先",
        "santan_form": "",
        "santan_tickets": tuple(),
        "santan_common_first_second": tuple(),
        "pair_power_key": tuple(),
        "trio_power_key": _rows_average_strength_key(rows),
    }



def _win_ai_confidence_profile(mark_map, active_cars=None):
    """
    WINTICKET AIの4印を券種信頼度用に正規化する。

    妙味計算の点数は流用せず、◎→〇→△→×という順位情報だけを見る。
    4印が1車ずつすべて揃った場合のみ complete=True とし、
    部分入力時は従来券種判定へ戻す。
    """
    active = None
    try:
        if active_cars is not None:
            active = {int(x) for x in active_cars if str(x).isdigit()}
    except Exception:
        active = None

    aliases = {"○": "〇", "▲": "△"}
    normalized = {}
    try:
        raw = dict(mark_map or {})
    except Exception:
        raw = {}

    # {車番:印} と {印:車番} の両形式へ対応。
    for k, v in raw.items():
        try:
            if str(k).strip() in {"◎", "〇", "○", "△", "▲", "×"}:
                car = int(v)
                mark = aliases.get(str(k).strip(), str(k).strip())
            else:
                car = int(k)
                mark = aliases.get(str(v).strip(), str(v).strip())
        except Exception:
            continue
        if active is not None and car not in active:
            continue
        if mark in {"◎", "〇", "△", "×"}:
            normalized[int(car)] = mark

    ordered = []
    complete = True
    for mark in ("◎", "〇", "△", "×"):
        cars = sorted(int(car) for car, mk in normalized.items() if mk == mark)
        if len(cars) != 1:
            complete = False
            continue
        ordered.append(int(cars[0]))

    if len(ordered) != 4 or len(set(ordered)) != 4:
        complete = False

    return {
        "complete": bool(complete),
        "ordered": tuple(ordered),
        "top2": tuple(ordered[:2]) if complete else tuple(),
        "top4": tuple(ordered) if complete else tuple(),
        "mark_map": normalized,
    }


def _decide_ticket_with_win_ai_confidence(
    structure,
    trio_rows,
    pair_rows=None,
    protected_third_candidates=None,
    *,
    market_mark_map=None,
    active_cars=None,
    all_trio_rows=None,
    line_pair=None,
    line_trio_rows=None,
    line_form="",
    line_protected_third=None,
    line_length=0,
    is_girls_only=False,
    structure_explainable=True,
):
    """
    v270-R：券種判定の優先順位を一本化する。

    1) 展開構造は、流れ・展開評価・ライン構成で先に確定する。
    2) 3連単は、v252の「採用3点の3単参考で1・2着が共通」を必須とする。
    3) さらに、A・B直後の同ライン3着候補が、A-B-CDE候補内で
       総合点の単独1位である場合だけ着順固定を許可する。
       新しい数値閾値は置かず、2位との差が正であることだけを見る。
    4) 採用ラインは3車以上に限定する。
    5) AI印は結果の補助表示だけに使い、券種・構造・買い目を昇格／降格させない。
    6) ガールズは3連単へ昇格させず、後段の三連複7点へ送る。
    """
    base = _decide_ticket_from_structure_and_santan_refs(
        structure,
        trio_rows,
        pair_rows,
        protected_third_candidates=protected_third_candidates,
    )
    result = dict(base or {})
    profile = _win_ai_confidence_profile(market_mark_map, active_cars=active_cars)

    result["win_confidence_complete"] = bool(profile.get("complete"))
    result["win_top2"] = tuple(profile.get("top2", tuple()) or tuple())
    result["win_top4"] = tuple(profile.get("top4", tuple()) or tuple())
    result["win_confidence_action"] = "AI印は補助参照（券種変更なし）"
    result["structure_override"] = ""
    result["structure_reason_override"] = ""

    try:
        normalized_line_length = int(line_length)
    except Exception:
        normalized_line_length = 0
    result["santan_line_length"] = normalized_line_length
    result["santan_line_length_ok"] = bool(normalized_line_length >= 3)
    result["structure_explainable"] = bool(structure_explainable)

    # AI4印は説明用。既存の展開・ポイント判定を上書きしない。
    def _ai_action(first_second=tuple(), trifecta_kept=False):
        if not profile.get("complete"):
            return "4印未完了（券種変更なし）"
        pair = {int(x) for x in (first_second or []) if str(x).isdigit()}
        top2 = {int(x) for x in (profile.get("top2", tuple()) or tuple())}
        top4 = {int(x) for x in (profile.get("top4", tuple()) or tuple())}

        # 非ライン主体などで比較対象となる共通1・2着が存在しない場合、
        # AI印を「不一致」とは判定しない。4印を確認した事実だけを表示する。
        if len(pair) != 2:
            return "AI4印確認済み（券種変更なし）"
        if pair == top2:
            return "AI◎〇一致（判定補強・昇格なし）" if trifecta_kept else "AI◎〇一致（参考・昇格なし）"
        if pair.issubset(top4):
            return "AI上位4車内（判定補助・昇格なし）"
        return "AI印不一致（参考・券種変更なし）"

    # ガールズは後段で従来の元5車三連複7点へ統一する。
    if bool(is_girls_only):
        if str(result.get("recommended_ticket", "")) == "3連単":
            result.update({
                "recommended_ticket": "3連複",
                "ticket_reason": "ガールズは着順固定を行わず、従来の元5車三連複7点へ送る",
                "santan_form": "",
                "santan_tickets": tuple(),
                "santan_common_first_second": tuple(),
                "selected_trio_rows": list(trio_rows or []),
                "selected_trio_form": str(line_form or ""),
            })
        result["win_confidence_action"] = _ai_action(tuple(), False)
        return result

    # v252判定で3連単に届いていない場合、AIで昇格させない。
    if str(result.get("recommended_ticket", "")) != "3連単":
        result["win_confidence_action"] = _ai_action(tuple(), False)
        return result

    common_pair = tuple(result.get("santan_common_first_second", tuple()) or tuple())

    # 展開としてライン主体を説明できない場合は3連複へ落とす。
    if str(structure or "") != "ライン主体" or not bool(structure_explainable):
        result.update({
            "recommended_ticket": "3連複",
            "ticket_reason": "3単参考は揃うが、採用展開をライン主体として説明できないため順不同を優先",
            "santan_form": "",
            "santan_tickets": tuple(),
            "santan_common_first_second": tuple(),
            "selected_trio_rows": list(trio_rows or []),
            "selected_trio_form": str(line_form or ""),
        })
        result["win_confidence_action"] = _ai_action(common_pair, False)
        return result

    # 2車ラインでは3連単にしない。
    if normalized_line_length < 3:
        result.update({
            "recommended_ticket": "3連複",
            "ticket_reason": f"3単参考は揃うが、採用ラインが{normalized_line_length}車のため3連単対象外。順不同を優先",
            "santan_form": "",
            "santan_tickets": tuple(),
            "santan_common_first_second": tuple(),
            "selected_trio_rows": list(trio_rows or []),
            "selected_trio_form": str(line_form or ""),
        })
        result["win_confidence_action"] = _ai_action(common_pair, False)
        return result

    # A・B直後の同ライン3着候補が、候補3点の総合点で単独1位か確認する。
    try:
        protected = int(line_protected_third)
    except Exception:
        protected = None

    candidate_rows = list(line_trio_rows or trio_rows or [])
    ranked = []
    for row in candidate_rows:
        parsed = _parse_santan_reference_triplet((row or {}).get("santan_ref", ""))
        if parsed is None:
            continue
        try:
            total_pt = float((row or {}).get("total_pt", 0.0) or 0.0)
        except Exception:
            total_pt = 0.0
        ranked.append((int(parsed[2]), total_pt, row))

    protected_rows = [item for item in ranked if protected is not None and int(item[0]) == int(protected)]
    protected_total = protected_rows[0][1] if protected_rows else None
    other_totals = [item[1] for item in ranked if protected is None or int(item[0]) != int(protected)]
    protected_is_clear_top = bool(
        protected_total is not None
        and other_totals
        and float(protected_total) > max(float(x) for x in other_totals) + 1e-12
    )

    if not protected_is_clear_top:
        result.update({
            "recommended_ticket": "3連複",
            "ticket_reason": (
                "3単参考の1・2着は共通だが、直後の同ライン3着候補が候補内の総合点単独1位ではない。"
                "3着候補のポイント差が明確でないため順不同を優先"
            ),
            "santan_form": "",
            "santan_tickets": tuple(),
            "santan_common_first_second": tuple(),
            "selected_trio_rows": list(trio_rows or []),
            "selected_trio_form": str(line_form or ""),
            "santan_protected_third": protected,
            "santan_core_third_clear_top": False,
        })
        result["win_confidence_action"] = _ai_action(common_pair, False)
        return result

    result["selected_trio_rows"] = list(trio_rows or [])
    result["selected_trio_form"] = str(line_form or "")
    result["santan_protected_third"] = protected
    result["santan_core_third_clear_top"] = True
    result["ticket_reason"] = (
        f"{str(result.get('ticket_reason', '') or '')}／"
        "直後の同ライン3着候補が候補内の総合点単独1位で、展開・着順・ポイントが一致"
    ).strip("／")
    result["win_confidence_action"] = _ai_action(common_pair, True)
    return result



# =========================================================
# v281：各流れ1位候補で採用流れ選定
#       → 採用流れ1・2位のAI評価が低い方を最終軸
#       → 最終軸の同ライン車を必須保護
#       → 採用流れ上位でヒモ4車を完成
# 買い目は三連複1車軸－4車－4車の6点だけ。
# =========================================================
_V281_STYLES = ("順流", "渦", "逆流")
_V281_STYLE_ORDER = {style: idx for idx, style in enumerate(_V281_STYLES)}
_V281_AI_MARK_RANK = {"◎": 0, "〇": 1, "△": 2, "×": 3, "": 4}


def _v281_normalize_mark(mark):
    """AI印を ◎／〇／△／×／無印へ正規化する。"""
    mk = str(mark or "").strip()
    if mk == "○":
        return "〇"
    if mk == "▲":
        return "△"
    return mk if mk in {"◎", "〇", "△", "×"} else ""


def _v281_normalize_ratio_map(flow_ratio_map):
    clean = {}
    for style in _V281_STYLES:
        try:
            value = float((flow_ratio_map or {}).get(style, 0.0) or 0.0)
            clean[style] = value if math.isfinite(value) and value >= 0.0 else 0.0
        except Exception:
            clean[style] = 0.0
    total = sum(clean.values())
    if total <= 0.0:
        return {style: 1.0 / len(_V281_STYLES) for style in _V281_STYLES}
    return {style: float(clean[style]) / float(total) for style in _V281_STYLES}


def _v281_unique_sequence(seq):
    out = []
    seen = set()
    for value in (seq or []):
        try:
            car = int(value)
        except Exception:
            continue
        if car <= 0 or car in seen:
            continue
        seen.add(car)
        out.append(car)
    return out


def _v281_map_float(mapping, car, default=0.0):
    for key in (car, str(car)):
        try:
            if key in (mapping or {}):
                value = float((mapping or {}).get(key, default) or default)
                return value if math.isfinite(value) else float(default)
        except Exception:
            pass
    return float(default)


def _v281_mark_for_car(mark_map, car):
    for key in (car, str(car)):
        try:
            if key in (mark_map or {}):
                return _v281_normalize_mark((mark_map or {}).get(key, ""))
        except Exception:
            pass
    return ""


def _v281_ai_rank(mark):
    return int(_V281_AI_MARK_RANK.get(_v281_normalize_mark(mark), 4))


def _v281_find_axis_line(line_def_obj, axis):
    """反映済みラインから最終軸を含むラインを入力順のまま返す。"""
    try:
        axis = int(axis)
    except Exception:
        return []

    if isinstance(line_def_obj, dict):
        line_values = list(line_def_obj.values())
    elif isinstance(line_def_obj, (list, tuple)):
        line_values = list(line_def_obj)
    else:
        line_values = []

    for members in line_values:
        line = _v281_unique_sequence(members)
        if axis in line:
            return line
    return []


def _v281_build_fixed_flow_plan(
    style_seq_map,
    flow_ratio_map,
    mark_map,
    ko_map,
    line_def_obj,
    line_strength_map=None,
):
    """
    1) 順流・渦・逆流の各着順予想1位を流れ選定候補にする。
    2) 各候補が属するライン／単騎を2車換算する。
       ・2車以上のライン＝ライン内KO使用スコア上位2車の合計
       ・単騎＝本人のKO使用スコア×2
    3) 2車換算勢力が最上位の流れを採用する。
    4) 採用流れの1位・2位から、AI評価が低い方を最終軸にする。
    5) 最終軸の同ライン車を軸以外すべて先にヒモへ確保する。
    6) 残枠を採用流れの着順予想上位から補充する。
    7) 三連複1車軸－4車－4車の6点を生成する。
    """
    ratios = _v281_normalize_ratio_map(flow_ratio_map)
    seq_map = {
        style: _v281_unique_sequence((style_seq_map or {}).get(style, []) or [])
        for style in _V281_STYLES
    }

    def _line_key(line):
        return "".join(str(int(car)) for car in (line or []))

    def _line_label(line):
        members = [int(car) for car in (line or [])]
        if not members:
            return "—"
        digits = "".join(str(car) for car in members)
        return f"単騎{digits}" if len(members) == 1 else digits

    def _candidate_line(car):
        line = _v281_find_axis_line(line_def_obj, car)
        return line if line else [int(car)]

    def _candidate_strength(line):
        key = _line_key(line)
        try:
            if key and key in (line_strength_map or {}):
                value = float((line_strength_map or {}).get(key, 0.0) or 0.0)
                if math.isfinite(value):
                    return value
        except Exception:
            pass
        return _t369_two_car_equivalent_strength(
            line,
            ko_map,
            default_score=0.0,
        )

    flow_candidates = []
    for style in _V281_STYLES:
        seq = list(seq_map.get(style, []) or [])
        if not seq:
            continue
        car = int(seq[0])
        candidate_line = _candidate_line(car)
        rec = {
            "style": style,
            "car": car,
            "score": _v281_map_float(ko_map, car, 0.0),
            "strength": float(_candidate_strength(candidate_line)),
            "line": tuple(candidate_line),
            "line_label": _line_label(candidate_line),
            "ratio": float(ratios.get(style, 0.0) or 0.0),
            "mark": _v281_mark_for_car(mark_map, car),
            "sequence": tuple(seq),
        }
        flow_candidates.append(rec)

    if not flow_candidates:
        return None

    # 採用流れは個人KOではなく、ライン／単騎の2車換算勢力で決める。
    # 完全同値時だけ、流れ想定比率→順流・渦・逆流の固定順→車番で決める。
    candidate_rows = sorted(
        flow_candidates,
        key=lambda row: (
            -float(row.get("strength", 0.0) or 0.0),
            -float(row.get("ratio", 0.0) or 0.0),
            int(_V281_STYLE_ORDER.get(str(row.get("style", "")), 99)),
            int(row.get("car", 99) or 99),
        ),
    )
    flow_selector_row = dict(candidate_rows[0])
    flow_selector_car = int(flow_selector_row.get("car"))
    flow_selector_score = float(flow_selector_row.get("score", 0.0) or 0.0)
    flow_selector_strength = float(flow_selector_row.get("strength", 0.0) or 0.0)
    flow_selector_line = tuple(flow_selector_row.get("line", tuple()) or tuple())
    flow_selector_line_label = str(flow_selector_row.get("line_label", "—") or "—")
    adopted_style = str(flow_selector_row.get("style", ""))
    adopted_styles = (adopted_style,)
    adopted_sequence = list(seq_map.get(adopted_style, []) or [])

    base_result = {
        "flow_candidates": tuple(flow_candidates),
        "flow_selector_candidates": tuple(candidate_rows),
        "flow_selector_car": flow_selector_car,
        "flow_selector_score": flow_selector_score,
        "flow_selector_strength": flow_selector_strength,
        "flow_selector_line": flow_selector_line,
        "flow_selector_line_label": flow_selector_line_label,
        "adopted_style": adopted_style,
        "adopted_styles": adopted_styles,
        "adopted_sequence": tuple(adopted_sequence),
    }

    if len(adopted_sequence) < 2:
        return {
            **base_result,
            "status": "insufficient_axis_candidates",
            "axis_pair": tuple(),
            "axis": 0,
            "axis_score": 0.0,
            "axis_mark": "",
            "axis_flow_rank": 0,
            "axis_line": tuple(),
            "same_line_himo": tuple(),
            "himo": tuple(),
            "ticket_groups": tuple(),
            "ticket_count": 0,
            "ticket_family": "3連複1車軸－4車－4車",
            "ticket_reason": f"{adopted_style}着順予想の上位2車を取得できないため生成不可",
        }

    axis_pair = []
    for idx, car in enumerate(adopted_sequence[:2], start=1):
        car = int(car)
        mark = _v281_mark_for_car(mark_map, car)
        axis_pair.append({
            "car": car,
            "rank": idx,
            "mark": mark,
            "ai_rank": _v281_ai_rank(mark),
            "score": _v281_map_float(ko_map, car, 0.0),
        })

    # AI評価が低い方を採用。入力仕様上、◎／〇／△／×は重複しない。
    axis_row = max(axis_pair, key=lambda row: int(row.get("ai_rank", 4)))
    axis = int(axis_row.get("car"))
    axis_score = float(axis_row.get("score", 0.0) or 0.0)
    axis_mark = str(axis_row.get("mark", "") or "")
    axis_flow_rank = int(axis_row.get("rank", 0) or 0)

    axis_line = _v281_find_axis_line(line_def_obj, axis)
    same_line_himo = [int(car) for car in axis_line if int(car) != axis]

    # 同ライン車は全車必須。ヒモ4枠を超える場合は黙って切らず生成停止する。
    if len(same_line_himo) > 4:
        return {
            **base_result,
            "status": "too_many_same_line_himo",
            "axis_pair": tuple(axis_pair),
            "axis": axis,
            "axis_score": axis_score,
            "axis_mark": axis_mark,
            "axis_flow_rank": axis_flow_rank,
            "axis_line": tuple(axis_line),
            "same_line_himo": tuple(same_line_himo),
            "himo": tuple(),
            "ticket_groups": tuple(),
            "ticket_count": 0,
            "ticket_family": "3連複1車軸－4車－4車",
            "ticket_reason": f"軸{axis}の同ライン車が4車を超えるため、必須保護を維持したまま6点生成できない",
        }

    himo = []
    for car in same_line_himo:
        if car != axis and car not in himo:
            himo.append(int(car))

    flow_added_himo = []
    for car in adopted_sequence:
        car = int(car)
        if car == axis or car in himo:
            continue
        himo.append(car)
        flow_added_himo.append(car)
        if len(himo) >= 4:
            break

    # 同ライン必須＋採用流れ上位で4車を作る。不足時は別流れから補完しない。
    if len(himo) < 4:
        return {
            **base_result,
            "status": "insufficient_himo",
            "axis_pair": tuple(axis_pair),
            "axis": axis,
            "axis_score": axis_score,
            "axis_mark": axis_mark,
            "axis_flow_rank": axis_flow_rank,
            "axis_line": tuple(axis_line),
            "same_line_himo": tuple(same_line_himo),
            "flow_added_himo": tuple(flow_added_himo),
            "himo": tuple(himo),
            "ticket_groups": tuple(),
            "ticket_count": 0,
            "ticket_family": "3連複1車軸－4車－4車",
            "ticket_reason": f"同ライン必須車と{adopted_style}着順予想上位を合わせてもヒモが4車未満のため生成不可",
        }

    tickets = []
    for a, b in combinations(himo[:4], 2):
        trio = tuple(sorted((axis, int(a), int(b))))
        tickets.append("-".join(str(x) for x in trio))

    candidate_text = "・".join(
        f"{rec['style']}={rec['line_label']}（2車換算={float(rec['strength']):.6f}・1着候補={int(rec['car'])}）"
        for rec in flow_candidates
    )
    axis_pair_text = "・".join(
        f"{adopted_style}{int(rec['rank'])}位={int(rec['car'])}（AI{str(rec['mark'] or '無印')}）"
        for rec in axis_pair
    )
    if same_line_himo:
        line_text = "".join(str(x) for x in axis_line)
        same_line_text = "・".join(str(x) for x in same_line_himo)
        line_reason = f"自ライン{line_text}の軸以外［{same_line_text}］を先に必須確保"
    else:
        line_reason = "軸は単騎のため自ライン必須車なし"

    added_text = "・".join(str(x) for x in flow_added_himo) if flow_added_himo else "なし"
    reason = (
        f"各流れ勢力［{candidate_text}］から2車換算スコア最上位の{flow_selector_line_label}で{adopted_style}を採用。"
        f"評価軸候補［{axis_pair_text}］のうちAI評価が低い{axis}を最終軸に選択。"
        f"{line_reason}し、{adopted_style}着順予想上位から［{added_text}］を補充"
    )

    return {
        **base_result,
        "status": "ok",
        "axis_pair": tuple(axis_pair),
        "axis": axis,
        "axis_score": axis_score,
        "axis_mark": axis_mark,
        "axis_flow_rank": axis_flow_rank,
        "axis_line": tuple(axis_line),
        "same_line_himo": tuple(same_line_himo),
        "flow_added_himo": tuple(flow_added_himo),
        "himo": tuple(himo[:4]),
        "ticket_groups": (("【3連複】", tuple(tickets)),),
        "ticket_count": len(tickets),
        "ticket_family": "3連複1車軸－4車－4車",
        "ticket_reason": reason,
    }


def _v281_format_fixed_flow_block(plan):
    if not isinstance(plan, dict) or not plan:
        return []

    flow_candidate_parts = []
    for rec in (plan.get("flow_candidates", tuple()) or tuple()):
        try:
            style = str(rec.get("style", ""))
            car = int(rec.get("car"))
            score = float(rec.get("score", 0.0) or 0.0)
            strength = float(rec.get("strength", 0.0) or 0.0)
            line_label = str(rec.get("line_label", "—") or "—")
            mark = str(rec.get("mark", "") or "無印")
            flow_candidate_parts.append(
                f"{style}={car}（勢力={line_label}:{strength:.6f}・KO={score:.6f}・AI{mark}）"
            )
        except Exception:
            pass

    adopted_style = str(plan.get("adopted_style", "") or "未判定")
    flow_selector_car = int(plan.get("flow_selector_car", 0) or 0)
    flow_selector_strength = float(plan.get("flow_selector_strength", 0.0) or 0.0)
    flow_selector_line_label = str(plan.get("flow_selector_line_label", "—") or "—")

    axis_pair_parts = []
    for rec in (plan.get("axis_pair", tuple()) or tuple()):
        try:
            rank = int(rec.get("rank", 0) or 0)
            car = int(rec.get("car"))
            mark = str(rec.get("mark", "") or "無印")
            axis_pair_parts.append(f"{rank}位={car}（AI{mark}）")
        except Exception:
            pass

    axis = int(plan.get("axis", 0) or 0)
    axis_score = float(plan.get("axis_score", 0.0) or 0.0)
    axis_mark = str(plan.get("axis_mark", "") or "無印")
    axis_flow_rank = int(plan.get("axis_flow_rank", 0) or 0)
    axis_line = [int(x) for x in (plan.get("axis_line", tuple()) or tuple())]
    same_line_himo = [int(x) for x in (plan.get("same_line_himo", tuple()) or tuple())]
    himo = [int(x) for x in (plan.get("himo", tuple()) or tuple())]

    if axis_line:
        line_label = "".join(str(x) for x in axis_line)
        same_line_label = "・".join(str(x) for x in same_line_himo) if same_line_himo else "なし"
        same_line_display = f"ライン{line_label}／必須ヒモ={same_line_label}"
    else:
        same_line_display = "単騎／必須ヒモなし"

    out = [
        "【流れ選定候補】" + (" ／ ".join(flow_candidate_parts) if flow_candidate_parts else "生成不可"),
        (
            f"【採用流れ】{adopted_style}（勢力={flow_selector_line_label}:"
            f"{flow_selector_strength:.6f}・1着候補={flow_selector_car}）"
        ),
        "【評価軸候補】" + (" ／ ".join(axis_pair_parts) if axis_pair_parts else "生成不可"),
        f"【最終軸】{axis}（{adopted_style}・{axis_flow_rank}位・AI{axis_mark}・KO使用スコア={axis_score:.6f}）",
        f"【自ライン優先】{same_line_display}",
        "【ヒモ4車】" + ("・".join(str(x) for x in himo[:4]) if len(himo) >= 4 else "不足"),
        "",
        f"【推奨車券】{plan.get('ticket_family', '3連複1車軸－4車－4車')}・{int(plan.get('ticket_count', 0) or 0)}点",
        f"【選定理由】{plan.get('ticket_reason', '')}",
    ]
    for label, items in (plan.get("ticket_groups", tuple()) or tuple()):
        out.append(f"{label}" + "　".join(str(x) for x in items))
    return out


def _make_note_final_summary_block(rec_style, rec_seq, mark_map=None):
    """note貼り付け用の現行v270サマリーを生成する。

    旧期待値推奨、34-12切替、三展開合成フォメ、VeloBi列フォメは参照しない。
    """
    try:
        xs = []
        seen = set()
        for x in (rec_seq or []):
            if str(x).isdigit():
                c = int(x)
                if c not in seen:
                    seen.add(c)
                    xs.append(c)

        def _pair_display_from_line(s):
            """旧妙味ブロックの行から車番ペアだけを抜く。例：'7-2　9.1pt［通過］' → '7-2'"""
            try:
                m = re.search(r"([1-9])\s*[-=]\s*([1-9])", str(s))
                if not m:
                    return None
                a, b = int(m.group(1)), int(m.group(2))
                if a == b:
                    return None
                return f"{a}-{b}"
            except Exception:
                return None

        def _add_pair(out, pair_keys, a, b):
            try:
                a, b = int(a), int(b)
                if a == b:
                    return
                key = tuple(sorted((a, b)))
                if key in pair_keys:
                    return
                pair_keys.add(key)
                out.append(f"{a}-{b}")
            except Exception:
                pass

        def _safe_col_text(name, fallback):
            try:
                v = str(globals().get(name, "") or "").strip()
                return v if v else fallback
            except Exception:
                return fallback

        def _axis_pair_line_tail_candidates(_a, _b):
            """
            A-Bが同一ラインで並んでいる場合、Bの後ろの3番手以降を
            3着・相手拡張候補として保護する。

            目的：
            ・個人評価が低い3番手でも、A→Bのライン決着では市場上位に残ることがある。
            ・「地区まとめ」は結束弱めだが、ライン残り候補から即消ししない。
            ・「流動」「単騎寄り」は固定ラインとして扱いにくいため保護しない。
            """
            out = []
            try:
                _a, _b = int(_a), int(_b)
                _line_def = globals().get("line_def", {}) or {}
                _trust = globals().get("line_follow_trust", {}) or {}
                _single_comment = globals().get("single_comment", {}) or {}

                for _gid, _mem in (_line_def or {}).items():
                    xs = [int(x) for x in (_mem or []) if str(x).isdigit()]
                    if _a not in xs or _b not in xs:
                        continue
                    ia, ib = xs.index(_a), xs.index(_b)

                    # A→Bの順で隣接しているラインだけを保護対象にする。
                    if ib != ia + 1:
                        continue

                    for x in xs[ib + 1:]:
                        xi = int(x)
                        label = str(_trust.get(xi, _trust.get(str(xi), "通常")) or "通常")
                        single_flag = bool(_single_comment.get(xi, _single_comment.get(str(xi), False)))
                        if label in ("流動", "単騎寄り") or single_flag:
                            continue
                        if xi not in out:
                            out.append(xi)
                    break
            except Exception:
                pass
            return out

        def _merge_car_text(*seqs):
            out = []
            for seq in seqs:
                for x in (seq or []):
                    try:
                        xi = int(x)
                    except Exception:
                        continue
                    if xi not in out:
                        out.append(xi)
            return "".join(str(x) for x in out)

        def _flow_ratio_map_for_trio():
            """
            流れ想定比率。

            v235:
            表示上のライン評価グループ（順流域／渦域／逆流域）で確定した
            3枠のFR比率を最優先で使う。
            これにより、逆流域が空なのに逆流100%などの矛盾を防ぐ。

            フォールバックとして compute_flow_indicators の FR/VTX/U を使う。
            """
            try:
                _zone_ratio = globals().get("FLOW_RATIO_MAP_BY_ZONE", None)
                if isinstance(_zone_ratio, dict):
                    _jr = float(_zone_ratio.get("順流", 0.0) or 0.0)
                    _ur = float(_zone_ratio.get("逆流", 0.0) or 0.0)
                    _vr = float(_zone_ratio.get("渦", 0.0) or 0.0)
                    _zt = _jr + _ur + _vr
                    if _zt > 0:
                        return {"順流": _jr / _zt, "逆流": _ur / _zt, "渦": _vr / _zt}
            except Exception:
                pass

            try:
                _flow = globals().get("_flow", {}) or {}
                _fr = float(_flow.get("FR", 0.0) or 0.0)
                _vtx = float(_flow.get("VTX", 0.0) or 0.0)
                _u = float(_flow.get("U", 0.0) or 0.0)
                _total = _fr + _u + _vtx
                if _total <= 0:
                    return {"順流": 1.0/3.0, "逆流": 1.0/3.0, "渦": 1.0/3.0}
                return {
                    "順流": _fr / _total,
                    "逆流": _u / _total,
                    "渦": _vtx / _total,
                }
            except Exception:
                return {"順流": 1.0/3.0, "逆流": 1.0/3.0, "渦": 1.0/3.0}

        def _fmt_flow_ratio_line(_ratio_map):
            try:
                return (
                    "流れ想定比率】"
                    f"順流{float(_ratio_map.get('順流', 0.0))*100:.0f}%／"
                    f"逆流{float(_ratio_map.get('逆流', 0.0))*100:.0f}%／"
                    f"渦{float(_ratio_map.get('渦', 0.0))*100:.0f}%"
                )
            except Exception:
                return "流れ想定比率】順流—%／逆流—%／渦—%"

        def _fmt_trio_form(_axis, _cols):
            try:
                _axis = int(_axis)
                _cols = [int(x) for x in (_cols or []) if str(x).isdigit() and int(x) != _axis]
                _out = []
                for _x in _cols:
                    if _x not in _out:
                        _out.append(_x)
                return f"{_axis}-{''.join(str(x) for x in _out)}-{''.join(str(x) for x in _out)}"
            except Exception:
                return "該当なし"

        def _trio_form_ticket_count(_cols):
            try:
                _n = len([x for x in (_cols or [])])
                return int(_n * (_n - 1) / 2) if _n >= 2 else 0
            except Exception:
                return 0

        lines = []

        # v193:
        # 買目考察は「推奨戦法1本」ではなく、順流・逆流・渦を全て並列表示する。
        # 会場判定 good/middle/bad による買目切替は廃止し、各流れごとに
        # 「総合評価B以上・総合pt上位2点」の2車複購入候補だけを表示する。
        style_seq_map = globals().get("STYLE_SEQ_MAP", {}) or {}
        if not isinstance(style_seq_map, dict):
            style_seq_map = {}

        flow_items = []
        # v198:
        # 順流・逆流・渦の表示枠は常に残す。
        # 2ライン戦などで独立した逆流シナリオが成立しない場合は、
        # 逆流を削除せず「該当なし」と表示する。
        for _style_name in ["順流", "逆流", "渦"]:
            _seq = style_seq_map.get(_style_name, []) or []
            _flow_xs = []
            _seen_flow = set()
            for _x in (_seq or []):
                if str(_x).isdigit():
                    _c = int(_x)
                    if _c not in _seen_flow:
                        _seen_flow.add(_c)
                        _flow_xs.append(_c)
            flow_items.append((_style_name, _flow_xs))

        # 保険：STYLE_SEQ_MAP が未生成の場合のみ、従来の推奨1本を表示対象にする。
        if not any(_seq for _, _seq in flow_items) and len(xs) >= 3:
            flow_items = [(str(rec_style or "推奨"), list(xs))]

        # v194: 詳細考察の前に、各流れで選ばれた2車複だけを一覧表示するための保持。
        flow_buy_summary = []
        # v201: 採用2点の総合ptも上部サマリーで使うため保持する。
        flow_buy_pt_summary = []

        # v199:
        # 「買目採用」は従来通り各流れの総合B以上・総合pt上位2点。
        # ただし判断材料として、まず各流れの総合B以上候補を全点表示し、
        # その候補同士で複数流れに重複する買目をイチオシとして抽出する。
        flow_b_candidate_summary = []
        # v204: サマリーの本線/抑えは「採用2点」ではなく、総合B以上候補全体から作る。
        # そのため、総合B以上候補のptも保持する。
        flow_b_candidate_pt_summary = []
        # v220: 各流れの「的中順単騎評価」を後で流れ想定比率で加重し、2車複・3連複の共通土台にする。
        flow_hit_avg_summary = []
        # v223: 妙味順単騎評価も同じく流れ比率で加重し、2車複の妙味期待側へ反映する。
        flow_myoumi_avg_summary = []
        # v220: 全21通り2車複評価も流れ別に保持し、加重単騎評価で2車複サマリーを再構成する。
        flow_all_pair_pt_summary = []

        def _append_one_flow_bet_review(_style_name, _seq):
            try:
                _xs = []
                _seen = set()
                for _x in (_seq or []):
                    if str(_x).isdigit():
                        _c = int(_x)
                        if _c not in _seen:
                            _seen.add(_c)
                            _xs.append(_c)

                lines.append(f"【買目考察｜{_style_name}】")
                lines.append("")

                if len(_xs) < 2:
                    flow_buy_summary.append((_style_name, []))
                    flow_buy_pt_summary.append((_style_name, []))
                    flow_b_candidate_summary.append((_style_name, []))
                    flow_b_candidate_pt_summary.append((_style_name, []))
                    flow_hit_avg_summary.append((_style_name, []))
                    flow_myoumi_avg_summary.append((_style_name, []))
                    flow_all_pair_pt_summary.append((_style_name, []))
                    lines.append("該当なし")
                    lines.append("")
                    return

                _A = int(_xs[0])
                _long_span_all_cars = [int(x) for x in _xs if str(x).isdigit()]

                # v196: 流れ別シナリオの主役ラインを、買目評価側にも渡す。
                # ここを使って、順流/渦/逆流ごとの妙味ptに差を付ける。
                try:
                    _scenario_main_line_map = globals().get("STYLE_SCENARIO_MAIN_LINE_MAP", {}) or {}
                    _scenario_main_line = [int(x) for x in (_scenario_main_line_map.get(str(_style_name), []) or []) if str(x).isdigit()]
                except Exception:
                    _scenario_main_line = []
                _scenario_main_set = {int(x) for x in (_scenario_main_line or [])}

                lines.append(f"推奨流れ【{_style_name}】：")
                lines.append(" → ".join(str(int(x)) for x in _xs))
                lines.append("")

                _axis_judge = _make_axis_trust_judgement(_xs)
                _axis_type = str(_axis_judge.get("type", "未判定"))
                _axis_reasons = [str(x) for x in (_axis_judge.get("reasons", []) or [])]
                _axis_line_note = str(_axis_judge.get("line_note", "ライン後位：未判定"))

                lines.append("【軸判定】")
                lines.append(_axis_type)
                if _axis_reasons:
                    lines.append("理由：" + "／".join(_axis_reasons[:4]))
                lines.append(_axis_line_note)
                lines.append("")

                def _longspan_velobi_rank(_car_no):
                    try:
                        return [int(x) for x in _xs].index(int(_car_no)) + 1
                    except Exception:
                        return 99

                def _longspan_velobi_point(_car_no):
                    try:
                        _rank = _longspan_velobi_rank(_car_no)
                        return {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}.get(_rank, 0)
                    except Exception:
                        return 0

                def _longspan_win_point(_car_no):
                    try:
                        _car_no = int(_car_no)
                        _mk = str((mark_map or {}).get(_car_no, (mark_map or {}).get(str(_car_no), "")) or "").strip()
                        _mk = _mk.replace("○", "〇")
                        return {"◎": 4, "〇": 3, "△": 2, "×": 1}.get(_mk, 0)
                    except Exception:
                        return 0

                def _longspan_hit_score_one(_car_no):
                    try:
                        _v = float(_longspan_velobi_point(_car_no))
                        _w = float(_longspan_win_point(_car_no))
                        _bonus = 0.0
                        if _v >= 4 and _w >= 3:
                            _bonus = 1.5
                        elif _v >= 3 and _w >= 2:
                            _bonus = 1.0
                        return 0.6 * _v + 0.4 * _w + _bonus
                    except Exception:
                        return 0.0

                def _longspan_hit_score_pair(_a, _b):
                    try:
                        return float(_longspan_hit_score_one(_a)) + float(_longspan_hit_score_one(_b))
                    except Exception:
                        return 0.0

                def _longspan_hit_rank(_a, _b):
                    try:
                        _s = _longspan_hit_score_pair(_a, _b)
                        if _s >= 10.0:
                            return "A"
                        if _s >= 8.0:
                            return "B"
                        if _s >= 6.0:
                            return "C"
                        return "D"
                    except Exception:
                        return "D"

                def _longspan_myoumi_rank(_score):
                    try:
                        _score = float(_score)
                        if _score >= 10.0:
                            return "A++"
                        if _score >= 9.4:
                            return "A+"
                        if _score >= 8.4:
                            return "A"
                        if _score >= 7.0:
                            return "B"
                        if _score >= 5.5:
                            return "C"
                        return "D"
                    except Exception:
                        return "D"

                def _longspan_myoumi_core_rank(_myoumi_rank):
                    _r = str(_myoumi_rank)
                    if _r in ("A++", "A+", "A"):
                        return "A"
                    if _r in ("B", "C", "D"):
                        return _r
                    return "D"

                def _longspan_total_rank(_hit_rank, _myoumi_rank):
                    _mr = _longspan_myoumi_core_rank(_myoumi_rank)
                    _table = {
                        ("A", "A"): "A", ("A", "B"): "A", ("A", "C"): "B", ("A", "D"): "C",
                        ("B", "A"): "A", ("B", "B"): "B", ("B", "C"): "B", ("B", "D"): "C",
                        ("C", "A"): "B", ("C", "B"): "C", ("C", "C"): "C", ("C", "D"): "D",
                        ("D", "A"): "C", ("D", "B"): "D", ("D", "C"): "D", ("D", "D"): "D",
                    }
                    return _table.get((str(_hit_rank), _mr), "D")

                def _longspan_total_score(_hit_score, _myoumi_score, _hit_rank, _myoumi_rank, _total_rank):
                    """
                    v207:
                    総合ptは、的中点と妙味点のバランスを見るため、加重平均ではなく幾何平均にする。
                    10点満点には丸めず、現状上限（的中12.0・妙味10.8）から最大約11.4点のまま扱う。

                    旧式：0.55 * 的中点 + 0.45 * 妙味点 + ランクボーナス
                    新式：sqrt(的中点 * 妙味点)

                    ※総合ランク表、的中期待ランク、妙味期待ランクは触らない。
                    ※v205の「イチオシ/本線への妙味期待ランク併記」も維持する。
                    """
                    try:
                        _hs = max(0.0, float(_hit_score))
                    except Exception:
                        _hs = 0.0
                    try:
                        _ms = max(0.0, float(_myoumi_score))
                    except Exception:
                        _ms = 0.0
                    try:
                        return round(math.sqrt(_hs * _ms), 1)
                    except Exception:
                        return 0.0

                def _scenario_myoumi_bonus_2kei(_a, _b, _base_score):
                    """
                    v196:
                    市場印だけの妙味ptだと、流れ別シナリオにしても妙味順位がほぼ変わらない。
                    そこで、その流れの主役ラインが絡む2車複へ小幅補正を入れる。

                    ・主役ライン内の2車複：強めに加点
                    ・主役ライン頭候補×高評価別線：中加点
                    ・主役ライン残り×高評価別線：小加点
                    ・主役ライン非関与：微減点

                    これは実オッズではなく、流れ別の仮説妙味を買目表へ反映するための内部pt。
                    """
                    try:
                        a, b = int(_a), int(_b)
                        base = float(_base_score)
                    except Exception:
                        return float(_base_score or 0.0)

                    if not _scenario_main_set:
                        return round(max(0.0, min(10.0, base)), 1)

                    in_a = a in _scenario_main_set
                    in_b = b in _scenario_main_set
                    bonus = 0.0

                    # 主役ライン内決着。2車複では頭裏の順序ブレを吸収できるので最優先で妙味を残す。
                    if in_a and in_b:
                        bonus += 1.25
                    elif in_a or in_b:
                        other = b if in_a else a
                        main_car = a if in_a else b
                        r_other = _longspan_velobi_rank(other)
                        r_main = _longspan_velobi_rank(main_car)

                        # 主役ラインが勝った時に、全体上位が2着へ突っ込む形。
                        if r_other <= 2:
                            bonus += 0.85
                        elif r_other <= 4:
                            bonus += 0.55
                        else:
                            bonus += 0.25

                        # 主役ライン内でも頭候補に近い車を少し優先。
                        if r_main <= 2:
                            bonus += 0.25
                    else:
                        # その流れの主役ラインが絡まない買い目は、比較上少しだけ下げる。
                        bonus -= 0.25

                    return round(max(0.0, min(10.0, base + bonus)), 1)

                def _longspan_pair_sort_key(_row):
                    _rank_order = {"A++": 6, "A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    try:
                        _total_pt = float(_row.get("total_pt", 0.0))
                    except Exception:
                        _total_pt = 0.0
                    try:
                        _hit_score = float(_row.get("hit_score", 0.0))
                    except Exception:
                        _hit_score = 0.0
                    try:
                        _myoumi_score = float(_row.get("myoumi_score", 0.0))
                    except Exception:
                        _myoumi_score = 0.0
                    return (
                        _total_pt,
                        _rank_order.get(str(_row.get("total_rank")), 0),
                        _hit_score,
                        _myoumi_score,
                        _rank_order.get(str(_row.get("hit_rank")), 0),
                        _rank_order.get(str(_row.get("myoumi_rank")), 0),
                    )

                # v203: 会場成績を2車複BOX評価の内部ptへ小幅反映する。
                # 的中率 → 的中期待係数、回収率 → 妙味期待係数。
                # 買目採用ルールは変えず、総合pt計算前の材料だけを補正する。
                try:
                    _venue_hit_coef = float(globals().get("venue_hit_expect_coef", st.session_state.get("venue_hit_expect_coef", 1.00)) or 1.00)
                except Exception:
                    _venue_hit_coef = 1.00
                try:
                    _venue_myoumi_coef = float(globals().get("venue_myoumi_expect_coef", st.session_state.get("venue_myoumi_expect_coef", 1.00)) or 1.00)
                except Exception:
                    _venue_myoumi_coef = 1.00

                _long_span_pairs = []
                _long_span_keys = set()
                for _a, _b in combinations(_long_span_all_cars, 2):
                    try:
                        _a_i, _b_i = int(_a), int(_b)
                        if _a_i == _b_i:
                            continue
                        _key = tuple(sorted((_a_i, _b_i)))
                        if _key in _long_span_keys:
                            continue
                        _long_span_keys.add(_key)

                        _order_pair = sorted([_a_i, _b_i], key=lambda z: _longspan_velobi_rank(z))
                        _score_head, _score_tail = int(_order_pair[0]), int(_order_pair[1])
                        try:
                            _base_sc = float(_myoumi_score_2kei(_score_head, _score_tail, int(_A), mark_map or {}))
                        except Exception:
                            _base_sc = 0.0
                        _sc = _scenario_myoumi_bonus_2kei(_key[0], _key[1], _base_sc)
                        # v203: 回収率が低い開催では妙味期待を少し弱め、回収率が高い開催では少し強める。
                        _sc = round(max(0.0, min(10.8, float(_sc) * float(_venue_myoumi_coef))), 2)

                        _disp = f"{_key[0]}-{_key[1]}"
                        _hit_score = _longspan_hit_score_pair(_key[0], _key[1])
                        # v203: 的中率が低い開催では的中期待を少し弱め、的中率が高い開催では少し強める。
                        _hit_score = round(max(0.0, min(12.0, float(_hit_score) * float(_venue_hit_coef))), 2)
                        if _hit_score >= 10.0:
                            _hit_rank = "A"
                        elif _hit_score >= 8.0:
                            _hit_rank = "B"
                        elif _hit_score >= 6.0:
                            _hit_rank = "C"
                        else:
                            _hit_rank = "D"
                        _myoumi_rank = _longspan_myoumi_rank(_sc)
                        _total_rank = _longspan_total_rank(_hit_rank, _myoumi_rank)
                        _total_pt = _longspan_total_score(_hit_score, _sc, _hit_rank, _myoumi_rank, _total_rank)
                        _long_span_pairs.append({
                            "disp": _disp,
                            "a": _key[0],
                            "b": _key[1],
                            "score_head": _score_head,
                            "score_tail": _score_tail,
                            "hit_rank": _hit_rank,
                            "hit_score": round(float(_hit_score), 2),
                            "myoumi_rank": _myoumi_rank,
                            "myoumi_score": round(float(_sc), 2),
                            "total_rank": _total_rank,
                            "total_pt": _total_pt,
                        })
                    except Exception:
                        pass

                if not _long_span_pairs:
                    flow_buy_summary.append((_style_name, []))
                    flow_buy_pt_summary.append((_style_name, []))
                    flow_b_candidate_summary.append((_style_name, []))
                    flow_b_candidate_pt_summary.append((_style_name, []))
                    flow_hit_avg_summary.append((_style_name, []))
                    flow_myoumi_avg_summary.append((_style_name, []))
                    flow_all_pair_pt_summary.append((_style_name, []))
                    lines.append("該当なし")
                    lines.append("")
                    return

                _sorted_pairs = sorted(_long_span_pairs, key=_longspan_pair_sort_key, reverse=True)
                _nifuku_buy_base = [
                    _row for _row in _sorted_pairs
                    if str(_row.get("total_rank", "")).strip() in ("A", "B")
                ]
                # v210: 流れ内の候補母集団は固定ptで切らず、総合B以上候補を保持する。
                #        冒頭サマリー側で、レース内の上位割合（本線30%・抑え50%）に絞る。
                _nifuku_display_base = list(_nifuku_buy_base or [])
                _b_candidate_disp = [str(_row.get("disp")) for _row in (_nifuku_display_base or []) if _row.get("disp")]
                flow_b_candidate_summary.append((_style_name, list(_b_candidate_disp)))
                flow_b_candidate_pt_summary.append((_style_name, [
                    {
                        "disp": str(_row.get("disp")),
                        "a": int(_row.get("a")),
                        "b": int(_row.get("b")),
                        "hit_score": float(_row.get("hit_score", 0.0) or 0.0),
                        "myoumi_score": float(_row.get("myoumi_score", 0.0) or 0.0),
                        "total_pt": float(_row.get("total_pt", 0.0) or 0.0),
                        "hit_rank": str(_row.get("hit_rank", "") or ""),
                        "myoumi_rank": str(_row.get("myoumi_rank", "") or ""),
                        "total_rank": str(_row.get("total_rank", "") or ""),
                    }
                    for _row in (_nifuku_display_base or []) if _row.get("disp")
                ]))

                flow_all_pair_pt_summary.append((_style_name, [
                    {
                        "disp": str(_row.get("disp")),
                        "a": int(_row.get("a")),
                        "b": int(_row.get("b")),
                        "hit_score": float(_row.get("hit_score", 0.0) or 0.0),
                        "myoumi_score": float(_row.get("myoumi_score", 0.0) or 0.0),
                        "total_pt": float(_row.get("total_pt", 0.0) or 0.0),
                        "hit_rank": str(_row.get("hit_rank", "") or ""),
                        "myoumi_rank": str(_row.get("myoumi_rank", "") or ""),
                        "total_rank": str(_row.get("total_rank", "") or ""),
                    }
                    for _row in (_sorted_pairs or []) if _row.get("disp")
                ]))

                _nifuku_buy = list(_nifuku_display_base or [])[:2]
                _nifuku_buy_disp = [str(_row.get("disp")) for _row in _nifuku_buy if _row.get("disp")]
                flow_buy_summary.append((_style_name, list(_nifuku_buy_disp)))
                flow_buy_pt_summary.append((_style_name, [
                    {"disp": str(_row.get("disp")), "total_pt": float(_row.get("total_pt", 0.0) or 0.0)}
                    for _row in _nifuku_buy if _row.get("disp")
                ]))

                lines.append("【総合評価2車複推奨】")
                lines.append("2車複購入候補（総合B以上・流れ内上位2点）")
                lines.append("　".join(_nifuku_buy_disp) if _nifuku_buy_disp else "該当なし")
                lines.append("")

                def _longspan_trimmed_avg(_vals):
                    try:
                        _vals = sorted([float(v) for v in (_vals or [])])
                        if len(_vals) >= 3:
                            _vals = _vals[1:-1]
                        if not _vals:
                            return 0.0
                        return round(sum(_vals) / len(_vals), 2)
                    except Exception:
                        return 0.0

                def _longspan_car_average_rows(_pairs, _cars):
                    _avg_rows = []
                    try:
                        for _car in [int(x) for x in (_cars or []) if str(x).isdigit()]:
                            _hit_vals = []
                            _myoumi_vals = []
                            _total_vals = []
                            for _row in (_pairs or []):
                                try:
                                    if int(_row.get("a")) == _car or int(_row.get("b")) == _car:
                                        _hit_vals.append(float(_row.get("hit_score", 0.0)))
                                        _myoumi_vals.append(float(_row.get("myoumi_score", 0.0)))
                                        _total_vals.append(float(_row.get("total_pt", 0.0)))
                                except Exception:
                                    pass
                            if _hit_vals and _myoumi_vals and _total_vals:
                                _hit_avg = _longspan_trimmed_avg(_hit_vals)
                                _myoumi_avg = _longspan_trimmed_avg(_myoumi_vals)
                                _avg_rows.append({
                                    "car": _car,
                                    "hit_avg": _hit_avg,
                                    "myoumi_avg": _myoumi_avg,
                                    "total_avg": _longspan_trimmed_avg(_total_vals),
                                })
                    except Exception:
                        _avg_rows = []
                    return _avg_rows

                def _longspan_car_average_line(_avg_rows, _key):
                    try:
                        _rows = sorted(_avg_rows or [], key=lambda r: (float(r.get(_key, 0.0)), -_longspan_velobi_rank(r.get("car"))), reverse=True)
                        return " → ".join(f"{int(r.get('car'))}（{float(r.get(_key, 0.0)):.1f}）" for r in _rows)
                    except Exception:
                        return ""

                _car_avg_rows = _longspan_car_average_rows(_sorted_pairs, _long_span_all_cars)
                flow_hit_avg_summary.append((_style_name, [
                    {"car": int(_r.get("car")), "hit_avg": float(_r.get("hit_avg", 0.0) or 0.0)}
                    for _r in (_car_avg_rows or []) if str(_r.get("car", "")).isdigit()
                ]))
                flow_myoumi_avg_summary.append((_style_name, [
                    {"car": int(_r.get("car")), "myoumi_avg": float(_r.get("myoumi_avg", 0.0) or 0.0)}
                    for _r in (_car_avg_rows or []) if str(_r.get("car", "")).isdigit()
                ]))
                _hit_avg_line = _longspan_car_average_line(_car_avg_rows, "hit_avg")
                _myoumi_avg_line = _longspan_car_average_line(_car_avg_rows, "myoumi_avg")

                lines.append("車番別平均評価（極端値除外）")
                if _hit_avg_line:
                    lines.append(f"的中順単騎評価：{_hit_avg_line}")
                if _myoumi_avg_line:
                    lines.append(f"妙味順単騎評価：{_myoumi_avg_line}")
                lines.append("")

                def _longspan_display_width(_text):
                    import unicodedata
                    _s = str(_text)
                    _w = 0
                    for _ch in _s:
                        _w += 2 if unicodedata.east_asian_width(_ch) in ("F", "W", "A") else 1
                    return _w

                def _longspan_pad_center(_text, _width):
                    _txt = str(_text)
                    _pad = max(0, int(_width) - _longspan_display_width(_txt))
                    _left = _pad // 2
                    _right = _pad - _left
                    return (("　" * (_left // 2)) + (" " * (_left % 2)) +
                            _txt +
                            ("　" * (_right // 2)) + (" " * (_right % 2)))

                _col_w = {
                    "disp": 10,
                    "hit": 10,
                    "myoumi": 10,
                    "total": 10,
                    "pt": 8,
                }
                _sep = ""
                lines.append(_sep.join([
                    _longspan_pad_center("買い目", _col_w["disp"]),
                    _longspan_pad_center("的中期待", _col_w["hit"]),
                    _longspan_pad_center("妙味期待", _col_w["myoumi"]),
                    _longspan_pad_center("総合評価", _col_w["total"]),
                    _longspan_pad_center("総合pt", _col_w["pt"]),
                ]))
                for _row in _sorted_pairs:
                    _disp_cell = _longspan_pad_center(_row.get("disp"), _col_w["disp"])
                    _hit_cell = _longspan_pad_center(_row.get("hit_rank"), _col_w["hit"])
                    _myoumi_cell = _longspan_pad_center(_row.get("myoumi_rank"), _col_w["myoumi"])
                    _total_cell = _longspan_pad_center(_row.get("total_rank"), _col_w["total"])
                    _pt_cell = _longspan_pad_center(f"{float(_row.get('total_pt', 0.0)):.1f}", _col_w["pt"])
                    lines.append(_sep.join([_disp_cell, _hit_cell, _myoumi_cell, _total_cell, _pt_cell]))

                lines.append("")
            except Exception as _e:
                flow_buy_summary.append((_style_name, []))
                flow_buy_pt_summary.append((_style_name, []))
                flow_b_candidate_summary.append((_style_name, []))
                flow_b_candidate_pt_summary.append((_style_name, []))
                flow_hit_avg_summary.append((_style_name, []))
                flow_myoumi_avg_summary.append((_style_name, []))
                flow_all_pair_pt_summary.append((_style_name, []))
                lines.append(f"【買目考察｜{_style_name}】")
                lines.append(f"生成不可（{_e}）")
                lines.append("")

        def _fmt_flow_buy_pairs(_pairs):
            _pairs = [str(x) for x in (_pairs or []) if str(x).strip()]
            return "　".join(_pairs) if _pairs else "該当なし"

        def _flow_summary_label(_style_name):
            # 「渦」は1文字なので、順流/逆流と縦位置が近くなるよう全角空白を足す。
            return "渦　" if str(_style_name) == "渦" else str(_style_name)

        if flow_items:
            # v194: まず詳細を一度組み立て、その過程で flow_buy_summary に各流れの購入候補を保持する。
            _main_lines_ref = lines
            _detail_lines = []
            lines = _detail_lines

            for _i, (_style_name, _seq) in enumerate(flow_items):
                if _i > 0:
                    lines.append("＊＊＊＊")
                    lines.append("")
                _append_one_flow_bet_review(_style_name, _seq)

            lines = _main_lines_ref

            # v210: 冒頭サマリーは固定pt足切りではなく、レース内の順位割合で整理する。
            # 表示順は、
            # 1) 本線：総合B以上候補のうち、総合pt上位30%
            # 2) 抑え：総合B以上候補のうち、総合pt上位50%以内（本線以外）
            # 3) ベスト10内重複：各流れの総合B以上候補・総合pt上位10内で複数流れに重複したもの
            # 4) 流れ別：総合B以上候補のうち、本線/抑えの表示対象に入ったものだけ表示
            # ※各流れ採用2点をサマリーの母集団には使わない。
            _NIFUKU_MAIN_PERCENT = 0.30
            _NIFUKU_DISPLAY_PERCENT = 0.50
            # v222: 流れ加重後は総合B以上が増えやすいので、表示点数に上限を置く。
            # 2車複は購入主役ではなく、3連複の骨格確認用サマリーとして使う。
            _NIFUKU_MAIN_MAX = 3
            _NIFUKU_DISPLAY_MAX = 5
            _summary_map = {}
            _summary_pt_map = {}
            _b_candidate_map = {}
            _overall_pairs = []
            _overall_pair_rows = []
            _overall_seen = set()

            def _pair_key_from_disp(_p):
                try:
                    _m = re.search(r"([1-9])\s*[-=]\s*([1-9])", str(_p))
                    if not _m:
                        return None
                    _a, _b = int(_m.group(1)), int(_m.group(2))
                    return tuple(sorted((_a, _b)))
                except Exception:
                    return None

            def _fmt_flow_buy_pairs(_pairs):
                _pairs = [str(x) for x in (_pairs or []) if str(x).strip()]
                return "　".join(_pairs) if _pairs else "該当なし"

            def _same_pair_list(_a, _b):
                try:
                    _ka = [_pair_key_from_disp(x) for x in (_a or [])]
                    _kb = [_pair_key_from_disp(x) for x in (_b or [])]
                    _ka = [x for x in _ka if x]
                    _kb = [x for x in _kb if x]
                    return _ka == _kb
                except Exception:
                    return False

            def _weighted_car_score_map_from_flows(_summary, _value_key):
                """v223: 各流れの車番別平均評価×流れ比率を車番ごとに合算する共通関数。"""
                _ratio = _flow_ratio_map_for_trio()
                _per_car = {}
                try:
                    for _style_name, _rows in (_summary or []):
                        _w = float(_ratio.get(str(_style_name), 0.0) or 0.0)
                        for _r in (_rows or []):
                            try:
                                _car = int(_r.get("car"))
                                _v = float(_r.get(str(_value_key), 0.0) or 0.0)
                                _per_car[_car] = _per_car.get(_car, 0.0) + _v * _w
                            except Exception:
                                pass
                except Exception:
                    _per_car = {}
                return _per_car

            def _weighted_car_hit_map_from_flows():
                return _weighted_car_score_map_from_flows(flow_hit_avg_summary, "hit_avg")

            def _weighted_car_myoumi_map_from_flows():
                return _weighted_car_score_map_from_flows(flow_myoumi_avg_summary, "myoumi_avg")

            def _overall_myoumi_core_rank(_myoumi_rank):
                _r = str(_myoumi_rank)
                if _r in ("A++", "A+", "A"):
                    return "A"
                if _r in ("B", "C", "D"):
                    return _r
                return "D"

            def _overall_hit_rank_from_score(_score):
                try:
                    _s = float(_score)
                    if _s >= 10.0:
                        return "A"
                    if _s >= 8.0:
                        return "B"
                    if _s >= 6.0:
                        return "C"
                    return "D"
                except Exception:
                    return "D"

            def _overall_myoumi_rank_from_score(_score):
                try:
                    _score = float(_score)
                    if _score >= 10.0:
                        return "A++"
                    if _score >= 9.4:
                        return "A+"
                    if _score >= 8.4:
                        return "A"
                    if _score >= 7.0:
                        return "B"
                    if _score >= 5.5:
                        return "C"
                    return "D"
                except Exception:
                    return "D"

            def _overall_total_rank_from_ranks(_hit_rank, _myoumi_rank):
                _mr = _overall_myoumi_core_rank(_myoumi_rank)
                _table = {
                    ("A", "A"): "A", ("A", "B"): "A", ("A", "C"): "B", ("A", "D"): "C",
                    ("B", "A"): "A", ("B", "B"): "B", ("B", "C"): "B", ("B", "D"): "C",
                    ("C", "A"): "B", ("C", "B"): "C", ("C", "C"): "C", ("C", "D"): "D",
                    ("D", "A"): "C", ("D", "B"): "D", ("D", "C"): "D", ("D", "D"): "D",
                }
                return _table.get((str(_hit_rank), _mr), "D")

            def _overall_total_score(_hit_score, _myoumi_score):
                """
                v243:
                総合点は「的中点」と「妙味点」の単純平均。
                順位判定は丸め前内部値で行い、表示時だけ小数第1位へ整形する。
                """
                try:
                    return (max(0.0, float(_hit_score)) + max(0.0, float(_myoumi_score))) / 2.0
                except Exception:
                    return 0.0

            def _make_weighted_overall_pair_rows(_weighted_car_hit_map, _weighted_car_myoumi_map):
                """
                v231:
                全21通り2車複を、流れ加重の的中単騎評価＋妙味単騎評価から再評価する。
                ・的中点：2車の加重的中単騎評価の平均
                ・妙味点：2車の加重妙味単騎評価の平均
                ・総合点：的中点と妙味点の単純平均
                ・ABCDランクは出さず、小数点第一位の数値で表示する。
                """
                # 全通りのキーは既存評価表から取得する。欠ける場合に備え、全車からも補完する。
                _keys = set()
                try:
                    for _style_name, _rows in (flow_all_pair_pt_summary or []):
                        for _row in (_rows or []):
                            _key = _pair_key_from_disp((_row or {}).get("disp"))
                            if _key:
                                _keys.add(_key)
                except Exception:
                    pass
                try:
                    _cars = sorted(set(int(c) for c in list(_weighted_car_hit_map.keys()) + list(_weighted_car_myoumi_map.keys())))
                    for _a, _b in combinations(_cars, 2):
                        _keys.add(tuple(sorted((int(_a), int(_b)))))
                except Exception:
                    pass

                _out = []
                for _key in sorted(_keys):
                    try:
                        _a, _b = int(_key[0]), int(_key[1])
                        _ha = _weighted_car_hit_map.get(_a, None)
                        _hb = _weighted_car_hit_map.get(_b, None)
                        _ma = _weighted_car_myoumi_map.get(_a, None)
                        _mb = _weighted_car_myoumi_map.get(_b, None)
                        if _ha is None or _hb is None:
                            continue
                        if _ma is None or _mb is None:
                            continue
                        # v231: 単騎評価から2車複へ変換するため、2車の平均で同じスケールを維持する。
                        _hit_score = round(max(0.0, min(12.0, (float(_ha) + float(_hb)) / 2.0)), 2)
                        _myoumi_score = round(max(0.0, min(10.8, (float(_ma) + float(_mb)) / 2.0)), 2)
                        _hit_rank = ""
                        _myoumi_rank = ""
                        _total_rank = ""
                        _total_pt = _overall_total_score(_hit_score, _myoumi_score)
                        _out.append({
                            "disp": f"{_a}-{_b}",
                            "a": _a,
                            "b": _b,
                            "hit_score": round(float(_hit_score), 2),
                            "myoumi_score": round(float(_myoumi_score), 2),
                            "hit_rank": _hit_rank,
                            "myoumi_rank": _myoumi_rank,
                            "total_rank": _total_rank,
                            "total_pt": _total_pt,
                        })
                    except Exception:
                        pass
                try:
                    return sorted(list(_out or []), key=lambda _r: (
                        float((_r or {}).get("total_pt", 0.0) or 0.0),
                        float((_r or {}).get("hit_score", 0.0) or 0.0),
                        float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                    ), reverse=True)
                except Exception:
                    return list(_out or [])

            def _select_axis3_nifuku_rows(_rows, _weighted_car_hit_map, _weighted_car_myoumi_map):
                """v223: 全通り評価から、◎軸-相手3車の2車複3点へ絞る。"""
                try:
                    _rows = list(_rows or [])
                    if not _rows:
                        return None, [], []
                    _top_rows = _rows[:10]
                    _conn_count = {}
                    _conn_pt = {}
                    for _r in _top_rows:
                        try:
                            a, b = int(_r.get("a")), int(_r.get("b"))
                            pt = float(_r.get("total_pt", 0.0) or 0.0)
                            for c, other in ((a, b), (b, a)):
                                _conn_count[c] = _conn_count.get(c, 0) + 1
                                _conn_pt[c] = _conn_pt.get(c, 0.0) + pt
                        except Exception:
                            pass

                    _hit_rows = sorted(
                        _weighted_car_hit_map.items(),
                        key=lambda kv: (float(kv[1]), float(_weighted_car_myoumi_map.get(int(kv[0]), 0.0)), int(_conn_count.get(int(kv[0]), 0)), -int(kv[0])),
                        reverse=True,
                    )
                    _axis_candidates = [int(c) for c, _ in _hit_rows[:3]] or [int(_rows[0].get("a"))]
                    _axis = max(
                        _axis_candidates,
                        key=lambda c: (
                            int(_conn_count.get(int(c), 0)),
                            float(_conn_pt.get(int(c), 0.0)),
                            float(_weighted_car_hit_map.get(int(c), 0.0)),
                            float(_weighted_car_myoumi_map.get(int(c), 0.0)),
                            -int(c),
                        )
                    )

                    _axis_rows = []
                    for _r in _rows:
                        try:
                            a, b = int(_r.get("a")), int(_r.get("b"))
                            if int(_axis) in (a, b):
                                _axis_rows.append(_r)
                        except Exception:
                            pass
                    _axis_rows = sorted(_axis_rows, key=lambda _r: (
                        float((_r or {}).get("total_pt", 0.0) or 0.0),
                        float((_r or {}).get("hit_score", 0.0) or 0.0),
                        float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                    ), reverse=True)
                    _main = _axis_rows[:3]
                    return int(_axis), _main, _axis_rows
                except Exception:
                    return None, [], []


            _weighted_car_hit_map = _weighted_car_hit_map_from_flows()
            _weighted_car_myoumi_map = _weighted_car_myoumi_map_from_flows()
            _nifuku_axis = None
            _nifuku_axis_rows_all = []

            for _style_name, _rows in (flow_buy_pt_summary or []):
                _summary_pt_map[str(_style_name)] = list(_rows or [])

            # v204/v205:
            # 全体の本線/抑えは、各流れの「上位2点」ではなく、
            # 総合B以上候補全体を重複除外して作る。
            # 同じ買目が複数流れに出た場合、表示ptと妙味期待は最もptが高い流れの値を採用する。
            for _style_name, _rows in (flow_b_candidate_pt_summary or []):
                for _row in (_rows or []):
                    try:
                        _key = _pair_key_from_disp(_row.get("disp"))
                        if not _key:
                            continue
                        _disp = f"{_key[0]}-{_key[1]}"
                        _pt = float(_row.get("total_pt", 0.0) or 0.0)
                        _myoumi_rank = str(_row.get("myoumi_rank", "") or "")
                        if _key in _overall_seen:
                            for _old in _overall_pair_rows:
                                try:
                                    if _pair_key_from_disp(_old.get("disp")) == _key and _pt > float(_old.get("total_pt", 0.0) or 0.0):
                                        _old["total_pt"] = _pt
                                        _old["myoumi_rank"] = _myoumi_rank
                                except Exception:
                                    pass
                            continue
                        _overall_seen.add(_key)
                        _overall_pairs.append(_disp)
                        _overall_pair_rows.append({"disp": _disp, "total_pt": _pt, "myoumi_rank": _myoumi_rank})
                    except Exception:
                        pass

            # v220: 2車複サマリーは、流れ別候補の最大ptではなく、
            #       流れ配分込みの車番別平均評価で的中期待を再計算した全通り評価から作る。
            _weighted_all_pair_rows = _make_weighted_overall_pair_rows(_weighted_car_hit_map, _weighted_car_myoumi_map)
            if _weighted_all_pair_rows:
                # v225:
                # 2車複は軸を先に決めない。
                # 流れ加重的中単騎評価＋流れ加重妙味単騎評価から作った
                # 全21通りの加重2車複評価表そのものを母集団にし、
                # 最終本線は総合pt上位3点を採用する。
                _nifuku_axis = None
                _nifuku_axis_rows_all = []
                _overall_pair_rows = list(_weighted_all_pair_rows or [])
                _overall_pairs = [str(_r.get("disp")) for _r in (_overall_pair_rows or []) if _r.get("disp")]
                _overall_seen = {_pair_key_from_disp(_p) for _p in (_overall_pairs or []) if _pair_key_from_disp(_p)}

            # 流れ別表示用には総合B以上候補の車券名だけを保持する。
            for _style_name, _pairs in (flow_b_candidate_summary or []):
                _b_candidate_map[str(_style_name)] = list(_pairs or [])

            _candidate_pair_styles = {}
            _candidate_pair_order = []
            _candidate_pair_best_row = {}
            for _style_name, _rows in (flow_b_candidate_pt_summary or []):
                for _row in (_rows or []):
                    _key = _pair_key_from_disp((_row or {}).get("disp"))
                    if not _key:
                        continue
                    if _key not in _candidate_pair_styles:
                        _candidate_pair_styles[_key] = []
                        _candidate_pair_order.append(_key)
                    if str(_style_name) not in _candidate_pair_styles[_key]:
                        _candidate_pair_styles[_key].append(str(_style_name))
                    try:
                        _pt = float((_row or {}).get("total_pt", 0.0) or 0.0)
                        _old = _candidate_pair_best_row.get(_key)
                        if _old is None or _pt > float((_old or {}).get("total_pt", 0.0) or 0.0):
                            _candidate_pair_best_row[_key] = {
                                "disp": f"{_key[0]}-{_key[1]}",
                                "total_pt": _pt,
                                "myoumi_rank": str((_row or {}).get("myoumi_rank", "") or ""),
                            }
                    except Exception:
                        pass

            # v210: イチオシの判定前に、本線/抑えの表示対象キーを先に決める。
            def _sort_rows_by_pt_desc(_rows):
                try:
                    return sorted(list(_rows or []), key=lambda _r: float((_r or {}).get("total_pt", 0.0) or 0.0), reverse=True)
                except Exception:
                    return list(_rows or [])

            _overall_sorted_rows = _sort_rows_by_pt_desc(_overall_pair_rows)
            try:
                import math as _math
                _n_all = len(_overall_sorted_rows or [])
                _main_n = max(1, int(_math.ceil(_n_all * _NIFUKU_MAIN_PERCENT))) if _n_all > 0 else 0
                _main_n = min(_main_n, int(_NIFUKU_MAIN_MAX)) if _main_n > 0 else 0
                _display_n = max(_main_n, int(_math.ceil(_n_all * _NIFUKU_DISPLAY_PERCENT))) if _n_all > 0 else 0
                _display_n = min(_display_n, int(_NIFUKU_DISPLAY_MAX)) if _display_n > 0 else 0
                _display_n = max(_display_n, _main_n)
            except Exception:
                _n_all = len(_overall_sorted_rows or [])
                _main_n = min(int(_NIFUKU_MAIN_MAX), _n_all)
                _display_n = min(int(_NIFUKU_DISPLAY_MAX), _n_all)
            _overall_main_rows = list(_overall_sorted_rows[:_main_n])
            _overall_sub_rows = list(_overall_sorted_rows[_main_n:_display_n])
            _display_pair_key_set = set()
            for _r in list(_overall_main_rows) + list(_overall_sub_rows):
                try:
                    _k = _pair_key_from_disp((_r or {}).get("disp"))
                    if _k:
                        _display_pair_key_set.add(_k)
                except Exception:
                    pass

            # v211: 「イチオシ」は廃止。
            #       代わりに、各流れの総合B以上候補・総合pt上位10内で
            #       複数流れに重複した買目を「ベスト10内重複」として表示する。
            _best10_pair_styles = {}
            _best10_pair_order = []
            _best10_pair_best_row = {}
            for _style_name, _rows in (flow_b_candidate_pt_summary or []):
                try:
                    _top10_rows = _sort_rows_by_pt_desc(_rows or [])[:10]
                except Exception:
                    _top10_rows = list(_rows or [])[:10]
                for _row in (_top10_rows or []):
                    _key = _pair_key_from_disp((_row or {}).get("disp"))
                    if not _key:
                        continue
                    if _key not in _best10_pair_styles:
                        _best10_pair_styles[_key] = []
                        _best10_pair_order.append(_key)
                    if str(_style_name) not in _best10_pair_styles[_key]:
                        _best10_pair_styles[_key].append(str(_style_name))
                    try:
                        _pt = float((_row or {}).get("total_pt", 0.0) or 0.0)
                        _old = _best10_pair_best_row.get(_key)
                        if _old is None or _pt > float((_old or {}).get("total_pt", 0.0) or 0.0):
                            _best10_pair_best_row[_key] = {
                                "disp": f"{_key[0]}-{_key[1]}",
                                "total_pt": _pt,
                                "myoumi_rank": str((_row or {}).get("myoumi_rank", "") or ""),
                            }
                    except Exception:
                        pass

            _best10_overlap_parts = []
            for _key in _best10_pair_order:
                _styles = _best10_pair_styles.get(_key, []) or []
                if len(_styles) >= 2:
                    _best = _best10_pair_best_row.get(_key, {}) or {}
                    try:
                        _best_pt = float(_best.get("total_pt", 0.0) or 0.0)
                    except Exception:
                        _best_pt = 0.0
                    _pt_txt = f"／{_best_pt:.1f}"
                    _myoumi_rank = str(_best.get("myoumi_rank", "") or "").strip()
                    _myoumi_txt = f" 妙味期待{_myoumi_rank}" if _myoumi_rank else ""
                    _best10_overlap_parts.append(f"{_key[0]}-{_key[1]}（{'・'.join(_styles)}{_pt_txt}{_myoumi_txt}）")

            def _fmt_overall_rows_with_pt(_rows, include_myoumi=False):
                _parts = []
                for _r in (_rows or []):
                    try:
                        _disp = str(_r.get("disp", "")).strip()
                        if not _disp:
                            continue
                        _pt = float(_r.get("total_pt", 0.0) or 0.0)
                        if include_myoumi:
                            _myoumi_rank = str(_r.get("myoumi_rank", "") or "").strip()
                            _myoumi_txt = f" 妙味期待{_myoumi_rank}" if _myoumi_rank else ""
                            _parts.append(f"{_disp}（{_pt:.1f}{_myoumi_txt}）")
                        else:
                            _parts.append(f"{_disp}（{_pt:.1f}）")
                    except Exception:
                        pass
                return "　".join(_parts) if _parts else "該当なし"

            def _sort_rows_by_pt_desc(_rows):
                try:
                    return sorted(list(_rows or []), key=lambda _r: float((_r or {}).get("total_pt", 0.0) or 0.0), reverse=True)
                except Exception:
                    return list(_rows or [])

            # v225: 最終2車複は◎軸流しではなく、加重2車複評価表の総合pt上位3点。
            _overall_sorted_rows = _sort_rows_by_pt_desc(_overall_pair_rows)
            _overall_main_rows = list(_overall_sorted_rows[:3])
            _overall_sub_rows = []
            _display_pair_key_set = set()
            for _r in list(_overall_main_rows) + list(_overall_sub_rows):
                try:
                    _k = _pair_key_from_disp((_r or {}).get("disp"))
                    if _k:
                        _display_pair_key_set.add(_k)
                except Exception:
                    pass

            def _make_weighted_overall_trio_rows(_pair_rows, _weighted_car_hit_map, _weighted_car_myoumi_map):
                """
                v241:
                加重2車複評価表を土台に、7車立て全35通りの3連複を評価する。
                ・3連複的中点：内包する2車複3本の的中点平均
                ・3連複妙味点：内包する2車複3本の妙味点平均
                ・3連複総合点：的中点と妙味点の平均（内部値で保持、表示は小数1桁）
                ・3単参考順：該当3車を流れ加重的中単騎評価順に並べる
                """
                try:
                    _pair_map = {}
                    _cars = set()
                    for _r in (_pair_rows or []):
                        try:
                            _a, _b = int(_r.get("a")), int(_r.get("b"))
                            _k = tuple(sorted((_a, _b)))
                            _pair_map[_k] = _r
                            _cars.add(_a); _cars.add(_b)
                        except Exception:
                            pass
                    try:
                        _cars.update(int(c) for c in (_weighted_car_hit_map or {}).keys())
                        _cars.update(int(c) for c in (_weighted_car_myoumi_map or {}).keys())
                    except Exception:
                        pass
                    _cars = sorted(int(c) for c in _cars if str(c).isdigit())
                    _out = []
                    for _a, _b, _c in combinations(_cars, 3):
                        try:
                            _ks = [tuple(sorted((_a, _b))), tuple(sorted((_a, _c))), tuple(sorted((_b, _c)))]
                            _prs = [_pair_map.get(_k) for _k in _ks]
                            if any(_x is None for _x in _prs):
                                continue
                            _hit_score = sum(float((_x or {}).get("hit_score", 0.0) or 0.0) for _x in _prs) / 3.0
                            _myoumi_score = sum(float((_x or {}).get("myoumi_score", 0.0) or 0.0) for _x in _prs) / 3.0
                            _total_pt = (max(0.0, float(_hit_score)) + max(0.0, float(_myoumi_score))) / 2.0
                            _cars3 = [int(_a), int(_b), int(_c)]
                            _santan_order = sorted(
                                _cars3,
                                key=lambda _x: (
                                    -float((_weighted_car_hit_map or {}).get(int(_x), 0.0) or 0.0),
                                    -float((_weighted_car_myoumi_map or {}).get(int(_x), 0.0) or 0.0),
                                    int(_x),
                                )
                            )
                            _out.append({
                                "disp": f"{_a}-{_b}-{_c}",
                                "cars": tuple(_cars3),
                                "a": int(_a),
                                "b": int(_b),
                                "c": int(_c),
                                "hit_score": float(_hit_score),
                                "myoumi_score": float(_myoumi_score),
                                "total_pt": float(_total_pt),
                                "santan_ref": "→".join(str(x) for x in _santan_order),
                            })
                        except Exception:
                            pass
                    return sorted(list(_out or []), key=lambda _r: (
                        float((_r or {}).get("total_pt", 0.0) or 0.0),
                        float((_r or {}).get("hit_score", 0.0) or 0.0),
                        float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                        str((_r or {}).get("disp", "")),
                    ), reverse=True)
                except Exception:
                    return []

            _weighted_trio_rows = _make_weighted_overall_trio_rows(
                _overall_sorted_rows,
                _weighted_car_hit_map,
                _weighted_car_myoumi_map,
            )

            # v256 ハイブリッド：
            # 1) 流れ想定比率の単独1位として確定したRECOMMENDED_STYLEの評価1位を軸Aにする。
            # 2) 2車複はv247を維持する。
            #    ・Aの同ライン相手のうち妙味点基準以上を妙味点順で先行採用。
            #    ・3点未満は、基準未満の同ライン相手を除外し、他ラインから総合点順で補完。
            # 3) 3連複は、ライン主体型と非ライン主体型を両方作り、展開構成で片方だけ採用する。
            #    ・ライン主体：A-直近ライン相手B-CDE（3点）
            #    ・非ライン主体：A-XYZ-XYZ（軸ライン以外の3車、3点）
            # 4) 構成判定は、流れ想定比率とその流れの着順予想を整合させる。
            #    ・軸流域が比率単独1位で、その流れの上位3車に軸Aと直近ライン相手Bが共存
            #      → 展開評価が優位、または初日互角の場合だけライン主体
            #    ・流れ着順予想が取得できない場合のみ、v248の優位／初日互角判定を補助使用
            #    ・単騎軸、軸流域が首位でない、上位3車にBがいない → 非ライン主体
            # 5) 新しい点差閾値は追加しない。
            def _select_v256_flow_axis_structure_bets(
                _pair_rows,
                _trio_rows,
                _hit_map,
                _myoumi_map,
            ):
                try:
                    _pair_rows = list(_pair_rows or [])
                    _trio_rows = list(_trio_rows or [])
                    _hit_map = dict(_hit_map or {})
                    _myoumi_map = dict(_myoumi_map or {})
                    if not _pair_rows or not _trio_rows:
                        return None

                    def _pair_cars(_row):
                        try:
                            return int(_row.get("a")), int(_row.get("b"))
                        except Exception:
                            _k = _pair_key_from_disp((_row or {}).get("disp"))
                            return (int(_k[0]), int(_k[1])) if _k else (None, None)

                    def _trio_cars(_row):
                        try:
                            _cars = [int(x) for x in ((_row or {}).get("cars") or []) if str(x).isdigit()]
                            if len(_cars) == 3:
                                return tuple(sorted(_cars))
                        except Exception:
                            pass
                        try:
                            return tuple(sorted((int(_row.get("a")), int(_row.get("b")), int(_row.get("c")))))
                        except Exception:
                            return tuple()

                    def _trio_row_sort_key(_row):
                        return (
                            float((_row or {}).get("total_pt", 0.0) or 0.0),
                            float((_row or {}).get("hit_score", 0.0) or 0.0),
                            float((_row or {}).get("myoumi_score", 0.0) or 0.0),
                            str((_row or {}).get("disp", "")),
                        )

                    # 全ペアに実在する車番を把握する。
                    _active_cars = set()
                    for _r in _pair_rows:
                        _a, _b = _pair_cars(_r)
                        if _a is not None:
                            _active_cars.add(int(_a))
                        if _b is not None:
                            _active_cars.add(int(_b))
                    if len(_active_cars) < 4:
                        return None

                    # 流れ想定比率の単独1位として確定した流れの評価1位を軸にする。
                    _recommended_style = str(globals().get("RECOMMENDED_STYLE", "") or "")
                    _recommended_seq = globals().get("RECOMMENDED_STYLE_SEQ", []) or []
                    if not _recommended_seq:
                        try:
                            _recommended_seq = (globals().get("STYLE_SEQ_MAP", {}) or {}).get(_recommended_style, []) or []
                        except Exception:
                            _recommended_seq = []

                    _axis = None
                    for _x in (_recommended_seq or []):
                        try:
                            _c = int(_x)
                        except Exception:
                            continue
                        if _c in _active_cars:
                            _axis = _c
                            break

                    # 推奨流れの並びが取得できない例外時だけ、加重的中単騎1位へフォールバック。
                    if _axis is None and _hit_map:
                        _axis = max(
                            _active_cars,
                            key=lambda _c: (
                                float(_hit_map.get(int(_c), 0.0) or 0.0),
                                -int(_c),
                            ),
                        )
                    if _axis is None:
                        return None
                    _axis = int(_axis)

                    def _partner_of_axis(_row):
                        _a, _b = _pair_cars(_row)
                        if _a is None or _b is None or _axis not in (int(_a), int(_b)):
                            return None
                        return int(_b) if int(_a) == _axis else int(_a)

                    def _pair_sort_key(_row):
                        _p = _partner_of_axis(_row)
                        return (
                            float((_row or {}).get("total_pt", 0.0) or 0.0),
                            float((_row or {}).get("hit_score", 0.0) or 0.0),
                            float((_row or {}).get("myoumi_score", 0.0) or 0.0),
                            -int(_p) if _p is not None else -99,
                        )

                    _axis_pair_rows = [
                        _r for _r in _pair_rows
                        if _partner_of_axis(_r) is not None
                    ]
                    _axis_pair_rows = sorted(_axis_pair_rows, key=_pair_sort_key, reverse=True)
                    if len(_axis_pair_rows) < 3:
                        return None

                    def _line_sources_v250():
                        _src = []
                        try:
                            _x = globals().get("lines_live", None)
                            if _x:
                                _src.append(_x)
                        except Exception:
                            pass
                        try:
                            _x = globals().get("line_def_live", None)
                            if isinstance(_x, dict) and _x:
                                _src.append(list(_x.values()))
                        except Exception:
                            pass
                        return _src

                    def _axis_line_order_v250(_axis_no):
                        _axis_no = int(_axis_no)
                        for _lines in _line_sources_v250():
                            for _ln in (_lines or []):
                                try:
                                    _cars = [int(x) for x in (_ln or []) if str(x).isdigit()]
                                except Exception:
                                    _cars = []
                                if _axis_no not in _cars:
                                    continue
                                _idx = _cars.index(_axis_no)
                                _mates = [c for c in _cars if c != _axis_no]
                                # 軸に近いライン相手から並べる。同距離ならライン表記の前側を先にする。
                                _mates.sort(key=lambda c: (abs(_cars.index(c) - _idx), _cars.index(c)))
                                return list(_cars), _mates
                        return [], []

                    _trio_map = {}
                    for _r in _trio_rows:
                        _cars3 = _trio_cars(_r)
                        if len(_cars3) == 3:
                            _trio_map[tuple(sorted(_cars3))] = _r

                    _axis_line, _line_mates = _axis_line_order_v250(_axis)
                    _line_anchor = int(_line_mates[0]) if _line_mates else None
                    _line_mate_set = {int(_c) for _c in (_line_mates or [])}
                    # v255：A・Bの次に続く最優先同ライン車だけを3連単の保護枠にする。
                    # 例：ライン1234、A/B=1・2なら3。4以降は他ライン候補との比較対象。
                    _line_protected_third = int(_line_mates[1]) if len(_line_mates) >= 2 else None

                    # -------------------------------------------------
                    # 2車複：v247仕様をそのまま維持
                    # -------------------------------------------------
                    try:
                        _same_line_myoumi_min = float(
                            globals().get("NIFUKU_SAME_LINE_MYOUMI_MIN", 7.0)
                        )
                    except Exception:
                        _same_line_myoumi_min = 7.0

                    _same_line_qualified = []
                    _same_line_rejected = set()

                    for _r in _axis_pair_rows:
                        _p = _partner_of_axis(_r)
                        if _p is None or int(_p) not in _line_mate_set:
                            continue
                        _p = int(_p)
                        _my = float((_r or {}).get("myoumi_score", 0.0) or 0.0)
                        if _my >= _same_line_myoumi_min:
                            _same_line_qualified.append(_r)
                        else:
                            _same_line_rejected.add(_p)

                    _same_line_qualified = sorted(
                        _same_line_qualified,
                        key=lambda _r: (
                            float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                            float((_r or {}).get("total_pt", 0.0) or 0.0),
                            float((_r or {}).get("hit_score", 0.0) or 0.0),
                            -int(_partner_of_axis(_r)) if _partner_of_axis(_r) is not None else -99,
                        ),
                        reverse=True,
                    )

                    _main_pair_rows = []
                    _selected_pair_partners = set()

                    for _r in _same_line_qualified:
                        _p = _partner_of_axis(_r)
                        if _p is None or int(_p) in _selected_pair_partners:
                            continue
                        _main_pair_rows.append(_r)
                        _selected_pair_partners.add(int(_p))
                        if len(_main_pair_rows) >= 3:
                            break

                    if len(_main_pair_rows) < 3:
                        for _r in _axis_pair_rows:
                            _p = _partner_of_axis(_r)
                            if _p is None:
                                continue
                            _p = int(_p)
                            if _p in _selected_pair_partners:
                                continue
                            # 同ラインは妙味基準だけで判定し、総合点補完から除外する。
                            if _p in _line_mate_set:
                                continue
                            _main_pair_rows.append(_r)
                            _selected_pair_partners.add(_p)
                            if len(_main_pair_rows) >= 3:
                                break

                    if not _main_pair_rows:
                        return None

                    _main_pair_rows = list(_main_pair_rows[:3])
                    _pair_partners = [int(_partner_of_axis(_r)) for _r in _main_pair_rows]
                    if len(set(_pair_partners)) != len(_pair_partners):
                        return None

                    # -------------------------------------------------
                    # ライン主体候補：A-B-CDE
                    # -------------------------------------------------
                    _line_third_candidates = []
                    _line_trio_rows = []
                    _line_form = ""

                    if _line_anchor is not None:
                        # 同ラインの3車目以降を先に3列目へ置く。
                        for _c in _line_mates[1:]:
                            _c = int(_c)
                            if _c not in (_axis, _line_anchor) and _c not in _line_third_candidates:
                                _line_third_candidates.append(_c)

                        # 残りは軸絡み総合点順。
                        for _r in _axis_pair_rows:
                            _c = _partner_of_axis(_r)
                            if _c is None:
                                continue
                            _c = int(_c)
                            if _c in (_axis, _line_anchor) or _c in _line_third_candidates:
                                continue
                            _line_third_candidates.append(_c)
                            if len(_line_third_candidates) >= 3:
                                break

                        if len(_line_third_candidates) < 3:
                            for _c in sorted(_active_cars):
                                _c = int(_c)
                                if _c in (_axis, _line_anchor) or _c in _line_third_candidates:
                                    continue
                                _line_third_candidates.append(_c)
                                if len(_line_third_candidates) >= 3:
                                    break

                        _line_third_candidates = _line_third_candidates[:3]
                        if len(set(_line_third_candidates)) == 3:
                            for _c in _line_third_candidates:
                                _r = _trio_map.get(tuple(sorted((_axis, _line_anchor, int(_c)))))
                                if _r is not None:
                                    _line_trio_rows.append(_r)
                            if len(_line_trio_rows) == 3:
                                _line_trio_rows = sorted(_line_trio_rows, key=_trio_row_sort_key, reverse=True)
                                _line_form = f"{_axis}-{_line_anchor}-{''.join(str(x) for x in sorted(_line_third_candidates))}"
                            else:
                                _line_trio_rows = []

                    # -------------------------------------------------
                    # 非ライン主体候補：A-XYZ-XYZ
                    # 軸の同ライン相手は候補から外し、他ライン側3車だけで組む。
                    # まず2車複本線に採用済みの他ライン車を使い、不足分を軸絡み総合点順で補う。
                    # -------------------------------------------------
                    _nonline_candidates = []

                    for _p in _pair_partners:
                        _p = int(_p)
                        if _p == _axis or _p in _line_mate_set or _p in _nonline_candidates:
                            continue
                        _nonline_candidates.append(_p)

                    for _r in _axis_pair_rows:
                        _p = _partner_of_axis(_r)
                        if _p is None:
                            continue
                        _p = int(_p)
                        if _p == _axis or _p in _line_mate_set or _p in _nonline_candidates:
                            continue
                        _nonline_candidates.append(_p)
                        if len(_nonline_candidates) >= 3:
                            break

                    if len(_nonline_candidates) < 3:
                        for _c in sorted(_active_cars):
                            _c = int(_c)
                            if _c == _axis or _c in _line_mate_set or _c in _nonline_candidates:
                                continue
                            _nonline_candidates.append(_c)
                            if len(_nonline_candidates) >= 3:
                                break

                    _nonline_candidates = _nonline_candidates[:3]
                    _nonline_trio_rows = []
                    _nonline_form = ""
                    if len(set(_nonline_candidates)) == 3:
                        for _x, _y in combinations(_nonline_candidates, 2):
                            _r = _trio_map.get(tuple(sorted((_axis, int(_x), int(_y)))))
                            if _r is not None:
                                _nonline_trio_rows.append(_r)
                        if len(_nonline_trio_rows) == 3:
                            _nonline_trio_rows = sorted(_nonline_trio_rows, key=_trio_row_sort_key, reverse=True)
                            _nonline_form = _fmt_trio_form(int(_axis), sorted(_nonline_candidates))
                        else:
                            _nonline_trio_rows = []

                    # -------------------------------------------------
                    # 3連複想定構成の自動判定
                    # 新しい数値閾値は置かず、既存の展開評価と比率順位を使う。
                    # -------------------------------------------------
                    _ratio_map = _flow_ratio_map_for_trio()
                    _ratio_map = {
                        "順流": float((_ratio_map or {}).get("順流", 0.0) or 0.0),
                        "逆流": float((_ratio_map or {}).get("逆流", 0.0) or 0.0),
                        "渦": float((_ratio_map or {}).get("渦", 0.0) or 0.0),
                    }

                    _axis_line_key = "".join(str(int(x)) for x in (_axis_line or []) if str(x).isdigit())
                    _axis_zone = ""
                    try:
                        _axis_zone = str((globals().get("LINE_ZONE_MAP", {}) or {}).get(_axis_line_key, "") or "")
                    except Exception:
                        _axis_zone = ""

                    _max_ratio = max(_ratio_map.values()) if _ratio_map else 0.0
                    _max_styles = [
                        _s for _s, _v in _ratio_map.items()
                        if abs(float(_v) - float(_max_ratio)) <= 1e-12
                    ]
                    _axis_zone_is_unique_top = bool(
                        _axis_zone in _ratio_map
                        and len(_max_styles) == 1
                        and _max_styles[0] == _axis_zone
                    )

                    _tenkai_eval = ""
                    try:
                        for _s in reversed(list(globals().get("note_sections", []) or [])):
                            _txt = str(_s).strip()
                            if _txt.startswith("展開評価："):
                                _tenkai_eval = _txt.split("：", 1)[1].strip()
                                break
                    except Exception:
                        _tenkai_eval = ""

                    _day_label = str(globals().get("day_label", "") or "")

                    # v250：比率単独1位の流れと、その流れの着順予想を直接つなぐ。
                    # 三連複のライン主体A-B-CDEはBを固定するため、
                    # 上位3車に「軸A」と「直近ライン相手B」が共存することを採用根拠にする。
                    _style_seq_map = globals().get("STYLE_SEQ_MAP", {}) or {}
                    try:
                        _axis_zone_seq = [
                            int(_x) for _x in (_style_seq_map.get(_axis_zone, []) or [])
                            if str(_x).isdigit() and int(_x) in _active_cars
                        ]
                    except Exception:
                        _axis_zone_seq = []
                    _axis_zone_top3 = list(_axis_zone_seq[:3])
                    _axis_zone_seq_available = bool(_axis_zone_seq)
                    _flow_prediction_supports_line = bool(
                        _line_anchor is not None
                        and int(_axis) in _axis_zone_top3
                        and int(_line_anchor) in _axis_zone_top3
                    )

                    _structure = "非ライン主体"
                    _structure_reason = ""
                    _structure_explainable = False

                    # v270-R：展開評価を先に固定する。
                    # 混戦を、流れ上位や加重比較だけでライン主体へ上書きしない。
                    if _line_form and _axis_zone_is_unique_top and _flow_prediction_supports_line:
                        if _tenkai_eval == "優位":
                            _structure = "ライン主体"
                            _structure_reason = "展開評価=優位／軸流域=比率単独1位／流れ予想上位3車に軸と直近ライン相手"
                            _structure_explainable = True
                        elif _tenkai_eval == "互角" and _day_label == "初日":
                            _structure = "ライン主体"
                            _structure_reason = "初日補正／展開評価=互角／軸流域=比率単独1位／流れ予想上位3車に軸と直近ライン相手"
                            _structure_explainable = True
                        elif _tenkai_eval == "混戦":
                            _structure_reason = "展開評価=混戦のため非ライン主体。単一ラインの着順固定を行わない"
                        else:
                            _structure_reason = "ライン支持はあるが、展開評価が3連単を説明できる条件に未達"
                    elif not _axis_zone_is_unique_top:
                        _structure_reason = "軸流域が比率単独1位ではないため非ライン主体"
                    elif not _flow_prediction_supports_line:
                        _structure_reason = "採用流れの上位3車に軸と直近ライン相手が共存しないため非ライン主体"
                    elif not _line_form:
                        _structure_reason = "ライン主体候補を生成できないため非ライン主体"

                    # 選ばれた構成が生成不能な場合だけ、生成可能な候補へ切り替える。
                    # このフォールバックは説明可能なライン主体とは扱わず、3連単を許可しない。
                    if _structure == "ライン主体" and not _line_trio_rows:
                        _structure = "非ライン主体"
                        _structure_reason = "ライン主体生成不可のため非ライン主体"
                        _structure_explainable = False
                    if _structure == "非ライン主体" and not _nonline_trio_rows:
                        if _line_trio_rows:
                            _structure = "ライン主体"
                            _structure_reason = "非ライン主体生成不可のためライン候補を三連複として使用"
                            _structure_explainable = False
                        else:
                            return None

                    # 加重3連複比較は診断値として残すが、展開構造を上書きしない。
                    _base_structure = str(_structure)
                    _power_decision = _choose_final_trio_structure_by_sidebar_power(
                        _base_structure,
                        _line_trio_rows,
                        _nonline_trio_rows,
                    )
                    _power_reason = str(_power_decision.get("reason", "") or "")
                    _line_power_key = tuple(_power_decision.get("line_power_key", tuple()) or tuple())
                    _nonline_power_key = tuple(_power_decision.get("nonline_power_key", tuple()) or tuple())
                    if _power_reason:
                        _structure_reason = f"{_structure_reason}／加重比較は参考:{_power_reason}" if _structure_reason else f"加重比較は参考:{_power_reason}"

                    if _structure == "ライン主体":
                        _main_trio_rows = list(_line_trio_rows)
                        _form = str(_line_form)
                        _third_candidates = tuple(_line_third_candidates)
                        _trio_mode = "line_axis"
                    else:
                        _main_trio_rows = list(_nonline_trio_rows)
                        _form = str(_nonline_form)
                        _third_candidates = tuple(_nonline_candidates)
                        _trio_mode = "nonline_box"

                    # v270-R2：
                    # ・ライン主体だけ、共通1・2着と同ライン3着候補の明確さを確認する。
                    # ・ライン保護はA・B直後の同ライン車1車だけを固定対象にする。
                    # ・非ライン主体は着順固定を行わず、元5車三連複7点へ送る。
                    # ・加重2車複／3連複比較は診断値に限定し、券種を変えない。
                    # 新しい数値閾値や実オッズは使わない。
                    _protected_santan_thirds = (
                        (int(_line_protected_third),)
                        if _structure == "ライン主体"
                        and _line_protected_third is not None
                        and int(_line_protected_third) in set(_line_third_candidates or [])
                        else tuple()
                    )
                    # v270-R2：AI印は補助表示だけに使用し、券種・構造・買い目を変更しない。
                    # 3連系のA・Bと3着候補は、必ず既存のライン主体候補から取得する。
                    # 流れ上位2車や非ライン候補から新しい3連単・3連複を生成しない。
                    _line_pair_for_confidence = (
                        (int(_axis), int(_line_anchor))
                        if _line_anchor is not None
                        else tuple()
                    )
                    _ticket_decision = _decide_ticket_with_win_ai_confidence(
                        _structure,
                        _main_trio_rows,
                        _main_pair_rows,
                        protected_third_candidates=_protected_santan_thirds,
                        market_mark_map=market_mark_map,
                        active_cars=_active_cars,
                        all_trio_rows=_trio_rows,
                        line_pair=_line_pair_for_confidence,
                        line_trio_rows=_line_trio_rows,
                        line_form=_line_form,
                        line_protected_third=_line_protected_third,
                        line_length=len(_axis_line or []),
                        is_girls_only=(race_class == "ガールズ"),
                        structure_explainable=bool(_structure_explainable),
                    )
                    _structure_override = str(_ticket_decision.get("structure_override", "") or "")
                    _structure_reason_override = str(
                        _ticket_decision.get("structure_reason_override", "") or ""
                    )
                    if _structure_override in {"ライン主体", "非ライン主体"}:
                        _structure = _structure_override
                    if _structure_reason_override:
                        _structure_reason = _structure_reason_override

                    _selected_trio_rows = list(_ticket_decision.get("selected_trio_rows", []) or [])
                    _selected_trio_form = str(_ticket_decision.get("selected_trio_form", "") or "")
                    if _selected_trio_rows:
                        _main_trio_rows = _selected_trio_rows
                    if _selected_trio_form:
                        _form = _selected_trio_form
                    _recommended_ticket = str(_ticket_decision.get("recommended_ticket", "3連複") or "3連複")
                    _ticket_reason_core = str(_ticket_decision.get("ticket_reason", "") or "")
                    _ticket_reason = "／".join(
                        x for x in (_structure_reason, _ticket_reason_core) if str(x).strip()
                    )
                    _santan_form = str(_ticket_decision.get("santan_form", "") or "")
                    _santan_tickets = tuple(_ticket_decision.get("santan_tickets", tuple()) or tuple())
                    _santan_common_first_second = tuple(
                        _ticket_decision.get("santan_common_first_second", tuple()) or tuple()
                    )
                    _pair_power_key = tuple(_ticket_decision.get("pair_power_key", tuple()) or tuple())
                    _trio_power_key = tuple(_ticket_decision.get("trio_power_key", tuple()) or tuple())
                    _win_confidence_action = str(
                        _ticket_decision.get("win_confidence_action", "v255維持") or "v255維持"
                    )

                    # =================================================
                    # v270-R 券種別の統合買い目構成
                    # 1) 3連単該当：説明可能な単一展開だけを4点＋3連複2点へ展開。
                    # 2) 3連単非該当の男子：v267の三流れ・三ライン分散5車を一車も切らず、
                    #    三連複12-123-12345の7点へ展開。
                    # 3) ガールズ：従来の比率2位・3位流れ5車をライン分散補正し、
                    #    同じく三連複12-123-12345の7点へ展開。
                    # 4) AI印・加重比較は、買い目骨格と券種を上書きしない。
                    # =================================================
                    _composition_label = ""
                    _composition_detail = ""
                    _supplement_trio_rows = []
                    _supplement_trio_form = ""
                    _five_car_form = tuple()
                    _nifuku_form = ""  # 旧戻り値との互換用。v270-Rでは使用しない。

                    if _recommended_ticket == "3連単" and len(_santan_common_first_second) == 2:
                        _first = int(_santan_common_first_second[0])
                        _second = int(_santan_common_first_second[1])
                        _candidate_rows_for_santan = list(
                            _ticket_decision.get("selected_trio_rows", []) or _line_trio_rows or _main_trio_rows
                        )
                        _santan_unified_plan = _v262_build_santan_plus_trio_plan(
                            _first,
                            _second,
                            _candidate_rows_for_santan,
                            _trio_rows,
                            protected_third=_line_protected_third,
                        )
                        if _santan_unified_plan:
                            _santan_form = str(_santan_unified_plan.get("santan_form", "") or "")
                            _santan_tickets = tuple(
                                _santan_unified_plan.get("santan_tickets", tuple()) or tuple()
                            )
                            _supplement_trio_rows = list(
                                _santan_unified_plan.get("support_trio_rows", tuple()) or tuple()
                            )
                            _supplement_trio_form = str(
                                _santan_unified_plan.get("support_trio_form", "") or ""
                            )
                            _composition_label = "3連単4点＋3連複2点"
                            _composition_detail = (
                                f"採用展開:{_recommended_style}／"
                                f"3連単{_santan_form}／3連複{_supplement_trio_form}"
                            )
                            _ticket_reason = (
                                f"{_ticket_reason}／採用展開を一つに固定し、"
                                "中心ラインの着順を説明できる範囲だけ3連単。別線侵入は3連複で順不同にする"
                            )
                            _main_pair_rows = []
                        else:
                            # 3連単4点＋3連複2点を一貫して生成できなければ、元5車7点へ落とす。
                            _recommended_ticket = "3連複"
                            _santan_form = ""
                            _santan_tickets = tuple()
                            _ticket_reason = f"{_ticket_reason}／3連単統合構成を生成できないため元5車三連複7点へ変更"

                    if _recommended_ticket != "3連単":
                        if race_class == "ガールズ":
                            _girls_base_plan = _v262_select_second_third_flow_five_plan(
                                _ratio_map,
                                _style_seq_map,
                                active_cars=_active_cars,
                                preferred_style=_recommended_style,
                            )
                            _flow_five_plan = _v264_line_diverse_five_plan(
                                _girls_base_plan,
                                _line_sources_v250(),
                                active_cars=_active_cars,
                                ko_score_map=globals().get("KO_SCORE_MAP_FOR_SANTEN", {}) or {},
                            )
                        else:
                            _flow_five_plan = _v267_select_three_flow_line_five_plan(
                                _ratio_map,
                                _style_seq_map,
                                _line_sources_v250(),
                                active_cars=_active_cars,
                                preferred_style=_recommended_style,
                                ko_score_map=globals().get("KO_SCORE_MAP_FOR_SANTEN", {}) or {},
                            )

                        if _flow_five_plan:
                            _five_car_form = tuple(
                                int(x) for x in (_flow_five_plan.get("cars", tuple()) or tuple())
                            )
                            _seven_trio_rows = _v262_rows_for_12_123_12345(
                                _trio_rows,
                                _five_car_form,
                            )
                            _seven_trio_form = _v262_form_12_123_12345(_five_car_form)
                        else:
                            _five_car_form = tuple()
                            _seven_trio_rows = []
                            _seven_trio_form = ""

                        if (
                            len(_five_car_form) == 5
                            and len(set(_five_car_form)) == 5
                            and len(_seven_trio_rows) == 7
                            and bool(_seven_trio_form)
                        ):
                            _recommended_ticket = "3連複"
                            _main_pair_rows = []
                            _main_trio_rows = list(_seven_trio_rows)
                            _form = str(_seven_trio_form)
                            _third_candidates = tuple(int(x) for x in _five_car_form[2:])
                            _composition_label = "元5車を一車も切らない三連複7点"

                            if race_class == "ガールズ":
                                _trio_mode = "girls_v270_five_trio7"
                                _structure = "比率2位・3位流れ5車・ライン分散"
                                _styles2 = tuple(_flow_five_plan.get("styles", tuple()) or tuple())
                                _ratios2 = tuple(_flow_five_plan.get("ratios", tuple()) or tuple())
                                _excluded_style = str(_flow_five_plan.get("excluded_style", "") or "")
                                _excluded_ratio = float(_flow_five_plan.get("excluded_ratio", 0.0) or 0.0)
                                _source_by_car = dict(_flow_five_plan.get("source_by_car", {}) or {})
                                _source_text = "・".join(
                                    f"{int(_car)}={_source_by_car.get(int(_car), '使用流れ内') }"
                                    for _car in _five_car_form
                                )
                                _composition_detail = (
                                    f"除外流れ:{_excluded_style}{_excluded_ratio*100:.0f}%／"
                                    f"使用:{_styles2[0]}{float(_ratios2[0])*100:.0f}%＋"
                                    f"{_styles2[1]}{float(_ratios2[1])*100:.0f}%／"
                                    f"元5車:{''.join(str(x) for x in _five_car_form)}／"
                                    f"出所:{_source_text}／三連複{_form}"
                                ) if len(_styles2) == 2 and len(_ratios2) == 2 else (
                                    f"元5車:{''.join(str(x) for x in _five_car_form)}／"
                                    f"出所:{_source_text}／三連複{_form}"
                                )
                                _ticket_reason = (
                                    "ガールズ用の従来ロジックで比率2位・3位流れの元5車を確定し、"
                                    "ライン分散を確認したうえで一車も切らず、12-123-12345の三連複7点へ展開"
                                )
                            else:
                                _trio_mode = "v270_three_flow_five_trio7"
                                _structure = "三流れ・三ライン分散"
                                _styles3 = tuple(_flow_five_plan.get("styles", tuple()) or tuple())
                                _ratios3 = tuple(_flow_five_plan.get("ratios", tuple()) or tuple())
                                _source_by_car = dict(_flow_five_plan.get("source_by_car", {}) or {})
                                _source_text = "・".join(
                                    f"{int(_car)}={_source_by_car.get(int(_car), '不明')}"
                                    for _car in _five_car_form
                                )
                                _composition_detail = (
                                    f"使用:{_styles3[0]}{float(_ratios3[0])*100:.0f}%＋"
                                    f"{_styles3[1]}{float(_ratios3[1])*100:.0f}%＋"
                                    f"{_styles3[2]}{float(_ratios3[2])*100:.0f}%／"
                                    f"元5車:{''.join(str(x) for x in _five_car_form)}／"
                                    f"出所:{_source_text}／三連複{_form}"
                                ) if len(_styles3) == 3 and len(_ratios3) == 3 else (
                                    f"元5車:{''.join(str(x) for x in _five_car_form)}／"
                                    f"出所:{_source_text}／三連複{_form}"
                                )
                                _ticket_reason = (
                                    f"{_ticket_reason_core or '3連単条件未達'}／"
                                    "v267の三流れ・三ライン分散で元5車を確定し、一車も切らず、"
                                    "12-123-12345の三連複7点へ展開"
                                )
                        else:
                            _recommended_ticket = "未判定"
                            _main_pair_rows = []
                            _main_trio_rows = []
                            _form = ""
                            _third_candidates = tuple()
                            _composition_label = "三連複7点生成不可"
                            _composition_detail = "元5車または12-123-12345の7点を、確定仕様どおりに生成できませんでした"
                            _ticket_reason = "矛盾した買い目へフォールバックせず、生成を停止"

                    return {
                        "recommended_style": _recommended_style,
                        "axis": int(_axis),
                        "axis_zone": _axis_zone,
                        "flow_ratio_map": dict(_ratio_map),
                        "tenkai_eval": _tenkai_eval,
                        "day_label": _day_label,
                        "axis_zone_seq": tuple(_axis_zone_seq),
                        "axis_zone_top3": tuple(_axis_zone_top3),
                        "flow_prediction_supports_line": bool(_flow_prediction_supports_line),
                        "structure": _structure,
                        "base_structure": _base_structure,
                        "structure_reason": _structure_reason,
                        "structure_explainable": bool(_structure_explainable),
                        "line_power_key": _line_power_key,
                        "nonline_power_key": _nonline_power_key,
                        "recommended_ticket": _recommended_ticket,
                        "ticket_reason": _ticket_reason,
                        "pair_power_key": _pair_power_key,
                        "trio_power_key": _trio_power_key,
                        "santan_form": _santan_form,
                        "santan_tickets": _santan_tickets,
                        "santan_common_first_second": _santan_common_first_second,
                        "win_confidence_complete": bool(_ticket_decision.get("win_confidence_complete", False)),
                        "win_top2": tuple(_ticket_decision.get("win_top2", tuple()) or tuple()),
                        "win_top4": tuple(_ticket_decision.get("win_top4", tuple()) or tuple()),
                        "win_confidence_action": str(_win_confidence_action or _ticket_decision.get("win_confidence_action", "v255維持") or "v255維持"),
                        "composition_label": _composition_label,
                        "composition_detail": _composition_detail,
                        "supplement_trio_rows": tuple(_supplement_trio_rows),
                        "supplement_trio_form": _supplement_trio_form,
                        "five_car_form": tuple(_five_car_form),
                        "nifuku_form": _nifuku_form,
                        "pair_partners": tuple(_pair_partners),
                        "same_line_myoumi_min": float(_same_line_myoumi_min),
                        "same_line_qualified": tuple(
                            int(_partner_of_axis(_r)) for _r in _same_line_qualified
                            if _partner_of_axis(_r) is not None
                        ),
                        "same_line_rejected": tuple(sorted(_same_line_rejected)),
                        "trio_mode": _trio_mode,
                        "line_anchor": _line_anchor,
                        "line_protected_third": _line_protected_third,
                        "third_candidates": _third_candidates,
                        "line_form": _line_form,
                        "nonline_form": _nonline_form,
                        "pair_rows": _main_pair_rows,
                        "trio_rows": _main_trio_rows,
                        "form": _form,
                    }
                except Exception:
                    return None

            _v256_bets = _select_v256_flow_axis_structure_bets(
                _overall_sorted_rows,
                _weighted_trio_rows,
                _weighted_car_hit_map,
                _weighted_car_myoumi_map,
            )
            if _v256_bets:
                _overall_main_rows = list(_v256_bets.get("pair_rows", []) or [])
                _overall_sub_rows = []
                _trio_main_rows = list(_v256_bets.get("trio_rows", []) or [])
                _final_trio_form = str(_v256_bets.get("form", "") or "")
                _trio_structure_label = str(_v256_bets.get("structure", "未判定") or "未判定")
                _recommended_ticket = str(_v256_bets.get("recommended_ticket", "未判定") or "未判定")
                _ticket_reason = str(_v256_bets.get("ticket_reason", "") or "")
                _santan_form = str(_v256_bets.get("santan_form", "") or "")
                _santan_tickets = tuple(_v256_bets.get("santan_tickets", tuple()) or tuple())
                _composition_label = str(_v256_bets.get("composition_label", "") or "")
                _composition_detail = str(_v256_bets.get("composition_detail", "") or "")
                _supplement_trio_rows = list(_v256_bets.get("supplement_trio_rows", tuple()) or tuple())
                _supplement_trio_form = str(_v256_bets.get("supplement_trio_form", "") or "")
                _five_car_form = tuple(_v256_bets.get("five_car_form", tuple()) or tuple())
                _nifuku_form = str(_v256_bets.get("nifuku_form", "") or "")
                _line_trio_form = str(_v256_bets.get("line_form", "") or "")
                _nonline_trio_form = str(_v256_bets.get("nonline_form", "") or "")
                _win_confidence_complete = bool(_v256_bets.get("win_confidence_complete", False))
                _win_top2 = tuple(_v256_bets.get("win_top2", tuple()) or tuple())
                _win_top4 = tuple(_v256_bets.get("win_top4", tuple()) or tuple())
                _win_confidence_action = str(_v256_bets.get("win_confidence_action", "v255維持") or "v255維持")
            else:
                # 推奨流れ軸や評価表が取得できない例外時だけ、v241の全体上位3点へ戻す。
                _overall_main_rows = list(_overall_sorted_rows[:3])
                _overall_sub_rows = []
                _trio_main_rows = list(_weighted_trio_rows[:3])
                _final_trio_form = ""
                _trio_structure_label = "未判定"
                _recommended_ticket = "未判定"
                _ticket_reason = "構成判定を生成できないため券種判定なし"
                _santan_form = ""
                _santan_tickets = tuple()
                _composition_label = ""
                _composition_detail = ""
                _supplement_trio_rows = []
                _supplement_trio_form = ""
                _five_car_form = tuple()
                _nifuku_form = ""
                _line_trio_form = ""
                _nonline_trio_form = ""
                _win_confidence_complete = False
                _win_top2 = tuple()
                _win_top4 = tuple()
                _win_confidence_action = "判定なし"

            # v282：各流れの代表ライン／単騎を2車換算し、勢力最上位の流れを採用する。
            # 採用流れ1・2位のAI低評価側を軸にし、最終軸の同ライン車を必須保護する。
            _v281_fixed_plan = _v281_build_fixed_flow_plan(
                globals().get("STYLE_SEQ_MAP", {}) or {},
                _flow_ratio_map_for_trio(),
                mark_map or {},
                globals().get("KO_SCORE_MAP_FOR_SANTEN", {}) or {},
                globals().get("line_def", {}) or {},
                globals().get("LINE_TWO_CAR_STRENGTH_MAP", {}) or {},
            )

            def _fmt_trio_summary_rows(_rows, include_santan_ref=True):
                _out = []
                for _r in (_rows or []):
                    try:
                        _disp = str((_r or {}).get("disp", "")).strip()
                        if not _disp:
                            continue
                        _pt = float((_r or {}).get("total_pt", 0.0) or 0.0)
                        _ref = str((_r or {}).get("santan_ref", "") or "").strip()
                        _ref_txt = f"／3単参考 {_ref}" if include_santan_ref and _ref else ""
                        _out.append(f"{_disp}（{_pt:.1f}{_ref_txt}）")
                    except Exception:
                        pass
                return _out if _out else ["該当なし"]

            def _fmt_santan_summary_rows(_tickets, _rows):
                """3連単4点を表示し、同じ3車の既存3連複総合点を添える。"""
                _combo_pt = {}
                for _r in (_rows or []):
                    try:
                        _key = _v262_trio_key_from_row(_r)
                        if len(_key) != 3:
                            continue
                        _combo_pt[_key] = max(
                            float((_r or {}).get("total_pt", 0.0) or 0.0),
                            float(_combo_pt.get(_key, 0.0) or 0.0),
                        )
                    except Exception:
                        pass

                _out = []
                for _ticket in (_tickets or []):
                    try:
                        _parsed = _parse_santan_reference_triplet(_ticket)
                        if _parsed is None:
                            continue
                        _pt = _combo_pt.get(tuple(sorted(int(x) for x in _parsed)))
                        if _pt is None:
                            _out.append(str(_ticket))
                        else:
                            _out.append(f"{_ticket}（参考元3連複総合 {_pt:.1f}）")
                    except Exception:
                        pass
                return _out if _out else ["該当なし"]

            def _make_flow_weighted_trio_lines():
                """
                v220:
                2車複サマリーにも使った流れ配分込みの車番別平均評価を、
                そのまま3連複 A-BCD-BCD の軸・ヒモ決定にも使う。
                """
                try:
                    _per_car = dict(_weighted_car_hit_map or {})
                    _per_myoumi = dict(_weighted_car_myoumi_map or {})
                    if not _per_car:
                        return []
                    _weighted_rows = sorted(_per_car.items(), key=lambda kv: (float(kv[1]), -int(kv[0])), reverse=True)
                    _weighted_line = " → ".join(f"{int(c)}（{float(v):.1f}）" for c, v in _weighted_rows)
                    _myoumi_rows = sorted(_per_myoumi.items(), key=lambda kv: (float(kv[1]), -int(kv[0])), reverse=True)
                    _myoumi_line = " → ".join(f"{int(c)}（{float(v):.1f}）" for c, v in _myoumi_rows)

                    # 表示対象2車複（本線＋抑え）との接続。
                    _display_rows = list(_overall_main_rows or []) + list(_overall_sub_rows or [])
                    _conn = {}
                    _main_conn = {}
                    _overlap_conn = {}
                    for _r in (_display_rows or []):
                        _k = _pair_key_from_disp((_r or {}).get("disp"))
                        if not _k:
                            continue
                        a, b = int(_k[0]), int(_k[1])
                        _conn.setdefault(a, set()).add(b)
                        _conn.setdefault(b, set()).add(a)
                    for _r in (_overall_main_rows or []):
                        _k = _pair_key_from_disp((_r or {}).get("disp"))
                        if not _k:
                            continue
                        a, b = int(_k[0]), int(_k[1])
                        _main_conn.setdefault(a, set()).add(b)
                        _main_conn.setdefault(b, set()).add(a)
                    for _key in (_best10_pair_order or []):
                        try:
                            _styles = _best10_pair_styles.get(_key, []) or []
                            if len(_styles) >= 2:
                                a, b = int(_key[0]), int(_key[1])
                                _overlap_conn.setdefault(a, set()).add(b)
                                _overlap_conn.setdefault(b, set()).add(a)
                        except Exception:
                            pass

                    _top_for_axis = [int(c) for c, _ in _weighted_rows[:3]]
                    if not _top_for_axis:
                        _base = [f"流れ加重的中単騎評価】{_weighted_line}"]
                        if _myoumi_line:
                            _base.append(f"流れ加重妙味単騎評価】{_myoumi_line}")
                        return _base
                    _axis = int(_nifuku_axis) if _nifuku_axis is not None else max(
                        _top_for_axis,
                        key=lambda c: (len(_conn.get(int(c), set())), float(_per_car.get(int(c), 0.0)), -int(c))
                    )

                    def _add_unique(_lst, _x):
                        try:
                            _x = int(_x)
                            if _x != int(_axis) and _x not in _lst:
                                _lst.append(_x)
                        except Exception:
                            pass

                    # v240 本線ヒモ：
                    # 三連複は「軸-ヒモ-ヒモ」の3点フォーメーションで作る。
                    # ヒモ3車は、まず軸が含まれる実ラインの相手を優先し、
                    # 不足分を「軸と組み合わせた加重2車複総合点上位」で補完する。
                    # 例：軸7・ライン726・軸絡み総合上位5-7なら 7-265-265。
                    def _axis_line_mates(_axis_no):
                        _mates = []
                        try:
                            _axis_no = int(_axis_no)
                            _line_sources = []
                            try:
                                _line_sources.append(globals().get("lines_live", None))
                            except Exception:
                                pass
                            try:
                                _line_sources.append(lines_live)
                            except Exception:
                                pass
                            try:
                                _line_def = globals().get("line_def_live", None)
                                if isinstance(_line_def, dict):
                                    _line_sources.append(list(_line_def.values()))
                            except Exception:
                                pass
                            for _lines in (_line_sources or []):
                                if not _lines:
                                    continue
                                for _ln in (_lines or []):
                                    try:
                                        _cars = [int(x) for x in (_ln or []) if str(x).isdigit()]
                                    except Exception:
                                        _cars = []
                                    if _axis_no not in _cars:
                                        continue
                                    for _c in _cars:
                                        if _c != _axis_no and _c not in _mates:
                                            _mates.append(_c)
                                    if _mates:
                                        return _mates
                        except Exception:
                            pass
                        return _mates

                    def _axis_pair_score_partners(_axis_no):
                        _partners = []
                        try:
                            _axis_no = int(_axis_no)
                            _rows = sorted(list(_overall_pair_rows or []), key=lambda _r: (
                                float((_r or {}).get("total_pt", 0.0) or 0.0),
                                float((_r or {}).get("hit_score", 0.0) or 0.0),
                                float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                            ), reverse=True)
                            for _r in (_rows or []):
                                try:
                                    a, b = int(_r.get("a")), int(_r.get("b"))
                                    if _axis_no not in (a, b):
                                        continue
                                    _p = b if a == _axis_no else a
                                    if _p != _axis_no and _p not in _partners:
                                        _partners.append(_p)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        return _partners

                    _main_cols = []
                    for _c in _axis_line_mates(_axis):
                        _add_unique(_main_cols, _c)
                    for _c in _axis_pair_score_partners(_axis):
                        _add_unique(_main_cols, _c)
                    # 保険：ライン・軸絡み2車複で不足する場合だけ、加重的中単騎順で補完する。
                    for c, _v in _weighted_rows:
                        _add_unique(_main_cols, c)
                    _main_cols = _main_cols[:3]

                    # 広め列：本線列に、加重単騎評価の次点を1車追加（最大4車）。
                    # 追加車は、最後の保護相手（例：ベスト10内重複相手）の手前へ入れる。
                    # 例：本線 5-627-627 → 広め 5-6237-6237。
                    _wide_cols = list(_main_cols)
                    for c, _v in _weighted_rows:
                        try:
                            c = int(c)
                            if c == int(_axis) or c in _wide_cols:
                                continue
                            if len(_wide_cols) >= 3:
                                _wide_cols.insert(len(_wide_cols) - 1, c)
                            else:
                                _wide_cols.append(c)
                            break
                        except Exception:
                            pass
                    _wide_cols = _wide_cols[:4]

                    _out = [f"流れ加重的中単騎評価】{_weighted_line}"]
                    if _myoumi_line:
                        _out.append(f"流れ加重妙味単騎評価】{_myoumi_line}")
                    if len(_main_cols) >= 2:
                        _out.append("流れ加重3連複】")
                        _out.append(f"本線】{_fmt_trio_form(_axis, _main_cols)}（{_trio_form_ticket_count(_main_cols)}点）")
                        if len(_wide_cols) > len(_main_cols):
                            _out.append(f"広め】{_fmt_trio_form(_axis, _wide_cols)}（{_trio_form_ticket_count(_wide_cols)}点）")
                    return _out
                except Exception:
                    return []

            _v281_fixed_lines = _v281_format_fixed_flow_block(_v281_fixed_plan)

            lines.append(_fmt_flow_ratio_line(_flow_ratio_map_for_trio()))
            lines.append("")

            if _v281_fixed_lines:
                lines.extend(_v281_fixed_lines)
                lines.append("")

            # v231:
            # 加重2車複評価表はABCDを出さず、的中点・妙味点・総合点を小数点第一位で表示する。
            def _fmt_weighted_pair_table(_rows, _limit=21):
                """
                v237:
                旧ヴェロビ表に近い全角スペース主体の整形。
                罫線なし。半角スペースの大量挿入は使わない。
                """
                try:
                    _rows = list(_rows or [])[:int(_limit)]
                    if not _rows:
                        return ["該当なし"]

                    def _fmt_num(_v):
                        try:
                            return f"{float(_v):.1f}"
                        except Exception:
                            return "-"

                    def _cell_left(_text, _chars):
                        _s = str(_text)
                        return _s + ("　" * max(0, int(_chars) - len(_s)))

                    def _cell_right(_text, _chars):
                        _s = str(_text)
                        return ("　" * max(0, int(_chars) - len(_s))) + _s

                    # 旧表の見た目に寄せる。
                    # 買い目は3桁固定、数値は小数1桁の3桁固定。
                    _out = []
                    _out.append("　買い目　　 的中点　 妙味点　 総合点")
                    for _r in _rows:
                        try:
                            _disp = str(_r.get("disp", "")).strip()
                            if not _disp:
                                continue
                            _hit = _fmt_num(_r.get("hit_score", 0.0))
                            _myo = _fmt_num(_r.get("myoumi_score", 0.0))
                            _tot = _fmt_num(_r.get("total_pt", 0.0))
                            _out.append(
                                "　" + _cell_left(_disp, 3) + "　　　　" +
                                _cell_right(_hit, 3) + "　　　" +
                                _cell_right(_myo, 3) + "　　　" +
                                _cell_right(_tot, 3)
                            )
                        except Exception:
                            pass
                    return _out if _out else ["該当なし"]
                except Exception:
                    return ["該当なし"]

            def _fmt_weighted_trio_table(_rows, _limit=35):
                """v241: 加重3連複評価表。罫線なし・全角スペース主体。"""
                try:
                    _rows = list(_rows or [])[:int(_limit)]
                    if not _rows:
                        return ["該当なし"]

                    def _fmt_num(_v):
                        try:
                            return f"{float(_v):.1f}"
                        except Exception:
                            return "-"

                    def _cell_left(_text, _chars):
                        _s = str(_text)
                        return _s + ("　" * max(0, int(_chars) - len(_s)))

                    def _cell_right(_text, _chars):
                        _s = str(_text)
                        return ("　" * max(0, int(_chars) - len(_s))) + _s

                    _out = []
                    _out.append("　買い目　　 3単参考　 的中点　 妙味点　 総合点")
                    for _r in _rows:
                        try:
                            _disp = str((_r or {}).get("disp", "")).strip()
                            if not _disp:
                                continue
                            _ref = str((_r or {}).get("santan_ref", "") or "").strip()
                            _hit = _fmt_num((_r or {}).get("hit_score", 0.0))
                            _myo = _fmt_num((_r or {}).get("myoumi_score", 0.0))
                            _tot = _fmt_num((_r or {}).get("total_pt", 0.0))
                            _out.append(
                                "　" + _cell_left(_disp, 5) + "　　　" +
                                _cell_left(_ref, 5) + "　　" +
                                _cell_right(_hit, 3) + "　　　" +
                                _cell_right(_myo, 3) + "　　　" +
                                _cell_right(_tot, 3)
                            )
                        except Exception:
                            pass
                    return _out if _out else ["該当なし"]
                except Exception:
                    return ["該当なし"]

            _fw_trio_lines = _make_flow_weighted_trio_lines()

            # v281: 旧買い目判定・A～E振り分けの表示は廃止。
            # 固定の三連複1車軸4車流し・6点だけを上部に表示する。
            # 旧計算値は互換性のため内部に残すが、note本文へは出力しない。

            # v227: 検証に必要な総合加重単騎評価だけ残す。
            if _fw_trio_lines:
                _score_lines = [
                    str(x) for x in _fw_trio_lines
                    if str(x).startswith("流れ加重的中単騎評価】")
                    or str(x).startswith("流れ加重妙味単騎評価】")
                ]
                if _score_lines:
                    lines.append("【総合加重単騎評価】")
                    lines.extend(_score_lines)
                    lines.append("")
                    lines.append("【加重2車複評価表】")
                    lines.extend(_fmt_weighted_pair_table(_overall_sorted_rows, _limit=21))
                    lines.append("")
                    lines.append("【加重3連複評価表】")
                    lines.extend(_fmt_weighted_trio_table(_weighted_trio_rows, _limit=35))
                    lines.append("")
                    lines.append("")
        else:
            lines.append("【買目考察】")
            lines.append("")
            lines.append("生成不可")
            lines.append("")
        # 旧版の補助文字列は表示しない。
        return "\n".join(lines).strip()
    except Exception as e:
        return f"note最終推奨サマリー生成不可：{e}"



try:
    _rec_style = globals().get("RECOMMENDED_STYLE", "")
    _rec_seq = globals().get("RECOMMENDED_STYLE_SEQ", [])
    _rec_seq = [int(x) for x in (_rec_seq or []) if str(x).isdigit()]

    _summary_core = _make_note_final_summary_block(
        _rec_style,
        _rec_seq,
        market_mark_map,
    )

    # v282：2車換算勢力・同ライン保護・AI妙味軸の三連複6点を展開評価の直後へ表示する。
    _current_summary = f"{_summary_core}\n"

    _m_tenkai = re.search(r"^展開評価：[^\n]*$", note_text, flags=re.MULTILINE)
    if _m_tenkai:
        note_text = note_text.replace(
            _m_tenkai.group(0),
            _m_tenkai.group(0) + "\n" + _current_summary,
            1,
        )
    else:
        note_text = _current_summary + "\n" + note_text


except Exception as _e:
    st.caption(f"note上部サマリー生成不可：{_e}")



# -----------------------------------------
# noteコピー表示整理（表示だけ。計算・順位・買い目生成には触らない）
# 削除対象：
# ・ラスト半周補正ブロック
# ・会場×最終Hライン補正ブロック
# 残す対象：
# ・上部サマリー
# ・ライン評価グループ
# ・KO使用スコア
# ・順流/渦/逆流メイン着順予想
# ・短評
# -----------------------------------------
def _clean_note_copy_display_only(text: str) -> str:
    try:
        lines = str(text).splitlines()
        out = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            s = line.strip()

            # 1) ラスト半周補正ブロックを削除
            if s == "【ラスト半周補正】":
                i += 1
                # 次の空行まで飛ばす
                while i < n and lines[i].strip() != "":
                    i += 1
                # 空行も1つ飛ばす
                while i < n and lines[i].strip() == "":
                    i += 1
                continue

            # 1-b) 会場×最終Hライン補正ブロックを削除
            if s == "【会場×最終Hライン補正】":
                i += 1
                # 次の空行まで飛ばす
                while i < n and lines[i].strip() != "":
                    i += 1
                # 空行も1つ飛ばす
                while i < n and lines[i].strip() == "":
                    i += 1
                continue

            out.append(line)
            i += 1

        # 連続空行を最大2行に抑える
        cleaned = []
        blank = 0
        for line in out:
            if line.strip() == "":
                blank += 1
                if blank <= 2:
                    cleaned.append(line)
            else:
                blank = 0
                cleaned.append(line)

        return "\n".join(cleaned).strip() + "\n"

    except Exception:
        return text



# -----------------------------------------
# 短評をアプリ向け定型コメントへ置換（表示だけ。計算・順位・買い目生成には触らない）
# -----------------------------------------
def _replace_tanpyou_with_simple_comment(text: str) -> str:
    try:
        txt = str(text)

        # 固定型の最終軸と採用流れを取得
        m_axis = re.search(r"【最終軸】([1-9])（(順流|渦|逆流)・", txt)
        axis = m_axis.group(1).strip() if m_axis else "未判定"
        axis_style = m_axis.group(2).strip() if m_axis else "未判定"

        # 順当度を旧短評から取得
        m_jundo = re.search(r"・順当度：([^［\n]+)", txt)
        jundo = m_jundo.group(1).strip() if m_jundo else "未判定"

        line1 = "・固定型：同ライン保護・AI低評価軸の三連複1車軸4車流し・6点。"
        if axis != "未判定" and axis_style != "未判定":
            line2 = f"・最終軸は{axis}、採用流れは{axis_style}。"
        else:
            line2 = "・最終軸と採用流れは未判定。"

        if jundo and jundo != "未判定":
            line3 = f"・展開は{jundo}。"
        else:
            line3 = "・展開は未判定。"

        new_block = "＜短評＞\n" + "\n".join([line1, line2, line3])

        if "＜短評＞" in txt:
            txt = re.sub(r"＜短評＞[\s\S]*$", new_block + "\n", txt)
        else:
            txt = txt.rstrip() + "\n\n" + new_block + "\n"

        return txt
    except Exception:
        return text


note_text = _clean_note_copy_display_only(note_text)
note_text = re.sub(r"^全体妙味：[^\n]*\n?", "", note_text, flags=re.MULTILINE)
note_text = re.sub(r"^【全体分類】[^\n]*\n?", "", note_text, flags=re.MULTILINE)
note_text = _replace_tanpyou_with_simple_comment(note_text)

st.text_area("ここを選択してコピー", note_text, height=620)
# =========================


# =========================
#  一括置換ブロック ここまで
# =========================
