"""
بوت إشارات تلغرام: Wyckoff -> Volume Profile -> SMC -> Sweep -> MSS -> Order Flow -> Entry -> Target
يرسل إشارات فقط (لا ينفّذ صفقات).

التثبيت:   pip install yfinance pandas numpy requests
المتغيرات: TELEGRAM_TOKEN  (من @BotFather)
           TELEGRAM_CHAT_ID (رقم المحادثة أو القناة)
           SYMBOL (اختياري، الافتراضي GC=F = عقود الذهب الآجلة)
التشغيل:   python signal_bot.py          # تشغيل مستمر
           python signal_bot.py --test   # رسالة اختبار
           python signal_bot.py --once   # فحص واحد ثم خروج
"""
import os
import sys
import json
import time
import logging

import numpy as np
import pandas as pd
import requests

# ---------------- الإعدادات ----------------
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SYMBOL = os.getenv("SYMBOL", "GC=F")

HTF_INTERVAL, HTF_PERIOD = "1h", "60d"    # فريم الاتجاه (Wyckoff)
LTF_INTERVAL, LTF_PERIOD = "15m", "30d"   # فريم التنفيذ (SMC)
CHECK_SECONDS = 120                       # كل كم ثانية يفحص

RANGE_BARS = 48          # عدد شموع H1 لتعريف النطاق
RANGE_ATR_MULT = 12      # أقصى عرض للنطاق بوحدات ATR
SWEEP_WINDOW = 20        # أقصى عمر للـSweep (بالشموع)
MIN_RR = 2.0             # أدنى نسبة عائد/مخاطرة
VOL_SPIKE = 1.3          # حجم شمعة الـSweep مقارنة بالمتوسط
MIN_LOWER_WICK = 0.4     # أدنى نسبة للذيل السفلي من مدى الشمعة
STATE_FILE = "sent_signals.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bot")


# ---------------- البيانات ----------------
def fetch(interval, period):
    import yfinance as yf
    df = yf.download(SYMBOL, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df.iloc[:-1]  # حذف الشمعة غير المغلقة


# ---------------- أدوات ----------------
def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low,
                    (df.High - pc).abs(),
                    (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def swing_points(df, n=3):
    hi, lo = df.High.values, df.Low.values
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if hi[i] == hi[i - n:i + n + 1].max():
            highs.append(i)
        if lo[i] == lo[i - n:i + n + 1].min():
            lows.append(i)
    return highs, lows


def invert(df):
    """قلب الأسعار لاكتشاف سيناريو البيع بنفس منطق الشراء."""
    d = df.copy()
    d["Open"], d["Close"] = -df.Open, -df.Close
    d["High"], d["Low"] = -df.Low, -df.High
    return d


# ---------------- 1) Wyckoff (فلتر تجميع) ----------------
def wyckoff_ok(htf):
    if len(htf) < RANGE_BARS + 50:
        return False
    w = htf.iloc[-RANGE_BARS:]
    a = atr(htf).iloc[-1]
    rng = w.High.max() - w.Low.min()
    if rng <= 0 or np.isnan(a):
        return False
    pos = (w.Close.iloc[-1] - w.Low.min()) / rng
    prior_down = htf.Close.iloc[-RANGE_BARS - 30] > htf.Close.iloc[-RANGE_BARS]
    return rng <= RANGE_ATR_MULT * a and pos <= 0.6 and prior_down


# ---------------- 2) Volume Profile ----------------
def volume_profile(df, bins=40, va=0.7):
    lo, hi = df.Low.min(), df.High.max()
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    tp = ((df.High + df.Low + df.Close) / 3).values
    idx = np.clip(np.searchsorted(edges, tp) - 1, 0, bins - 1)
    np.add.at(vol, idx, df.Volume.values)
    poc = int(vol.argmax())
    lo_i = hi_i = poc
    tot, acc = vol.sum(), vol[poc]
    while acc < va * tot and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    poc_price = (edges[poc] + edges[poc + 1]) / 2
    return poc_price, edges[lo_i], edges[hi_i + 1]  # POC, VAL, VAH


# ---------------- المحرك (شراء) ----------------
def detect(ltf, htf):
    """يبحث عن سيناريو شراء كامل. يرجع dict أو None."""
    if len(ltf) < 120 or not wyckoff_ok(htf):
        return None

    A = atr(ltf).iloc[-1]
    if np.isnan(A) or A <= 0:
        return None
    poc, val, vah = volume_profile(ltf.iloc[-300:])
    highs, lows = swing_points(ltf)
    O, H, L, C, V = (ltf[c].values for c in ["Open", "High", "Low", "Close", "Volume"])
    last = len(ltf) - 1

    for s in range(last - 1, last - SWEEP_WINDOW, -1):
        # قاع سيولة لم يُكسر قبل شمعة الـSweep
        cand = [i for i in lows if s - 60 < i < s - 3 and L[i + 1:s].min() > L[i]]
        if not cand:
            continue
        swing_low = L[cand[-1]]

        # 4) Liquidity Sweep: كسر القاع بذيل وإغلاق فوقه
        if not (L[s] < swing_low and C[s] > swing_low):
            continue
        sweep_low = L[s]

        # 2) السعر قرب منطقة قيمة مهمة (VAL أو POC)
        near_value = (val <= sweep_low <= poc) or \
            min(abs(sweep_low - val), abs(sweep_low - poc)) <= A
        if not near_value:
            continue

        # 5) MSS: إغلاق فوق آخر قمة سوينغ قبل الـSweep
        prior_highs = [i for i in highs if i < s]
        if not prior_highs:
            continue
        mss_level = H[prior_highs[-1]]
        m = next((k for k in range(s + 1, last + 1) if C[k] > mss_level), None)
        if m is None:
            continue

        # 6) Order Flow (تقريبي): امتصاص + دلتا إيجابية
        rng_s = H[s] - L[s]
        if rng_s <= 0:
            continue
        lower_wick = (min(O[s], C[s]) - L[s]) / rng_s
        avg_vol = V[max(0, s - 20):s].mean()
        absorption = lower_wick >= MIN_LOWER_WICK and V[s] >= VOL_SPIKE * avg_vol
        body = (C[s:m + 1] - O[s:m + 1]) / np.maximum(H[s:m + 1] - L[s:m + 1], 1e-9)
        delta = float((body * V[s:m + 1]).sum())
        if not (absorption and delta > 0):
            continue

        # 7) الدخول بعد Retracement (منطقة OTE بين 50% و70.5%)
        top = H[s:last + 1].max()
        leg = top - sweep_low
        if leg < A:
            continue
        z_hi = top - 0.5 * leg
        z_lo = top - 0.705 * leg
        entry = top - 0.618 * leg
        sl = sweep_low - 0.2 * A
        if not (L[last] <= z_hi and C[last] > sl):
            continue  # لم يحدث retracement بعد أو الإشارة فسدت

        # 8) الهدف: أقرب سيولة علوية (قمة سوينغ فوق قمة الحركة)
        targets = [H[i] for i in highs if H[i] > top]
        if not targets:
            continue
        tp = min(targets)
        risk = entry - sl
        rr = (tp - entry) / risk if risk > 0 else 0
        if rr < MIN_RR:
            continue

        return dict(key=str(ltf.index[s]), entry=entry, sl=sl, tp=tp, rr=rr,
                    z_lo=z_lo, z_hi=z_hi, poc=poc, val=val, vah=vah,
                    sweep_low=sweep_low)
    return None


def flip(sig):
    """تحويل نتيجة السيناريو المقلوب إلى بيع."""
    if not sig:
        return None
    out = dict(sig)
    for k in ("entry", "sl", "tp", "poc", "val", "vah", "sweep_low"):
        out[k] = -sig[k]
    out["z_lo"], out["z_hi"] = -sig["z_hi"], -sig["z_lo"]
    return out


# ---------------- تلغرام ----------------
def send(text):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


def format_signal(side, s):
    icon = "🟢" if side == "BUY" else "🔴"
    return (
        f"{icon} إشارة {side} | {SYMBOL}\n"
        f"الدخول (Limit): {s['entry']:.2f}\n"
        f"منطقة OTE: {s['z_lo']:.2f} - {s['z_hi']:.2f}\n"
        f"وقف الخسارة: {s['sl']:.2f}\n"
        f"الهدف (السيولة المقابلة): {s['tp']:.2f}\n"
        f"R:R = {s['rr']:.1f}\n\n"
        f"✔ Wyckoff: نطاق {'تجميع' if side == 'BUY' else 'توزيع'}\n"
        f"✔ Volume Profile: POC {s['poc']:.2f} | VAL {s['val']:.2f} | VAH {s['vah']:.2f}\n"
        f"✔ Sweep عند {s['sweep_low']:.2f} ثم MSS\n"
        f"✔ Order Flow: امتصاص + دلتا (تقريبي)\n\n"
        f"⚠ ليست نصيحة مالية. اختبر الاستراتيجية قبل الاعتماد عليها."
    )


# ---------------- الحالة ----------------
def load_sent():
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_sent(sent):
    with open(STATE_FILE, "w") as f:
        json.dump(sorted(sent)[-500:], f)


# ---------------- الحلقة الرئيسية ----------------
def scan_once(sent):
    htf = fetch(HTF_INTERVAL, HTF_PERIOD)
    ltf = fetch(LTF_INTERVAL, LTF_PERIOD)
    results = {"BUY": detect(ltf, htf),
               "SELL": flip(detect(invert(ltf), invert(htf)))}
    for side, sig in results.items():
        if not sig:
            continue
        key = f"{side}:{sig['key']}"
        if key in sent:
            continue
        send(format_signal(side, sig))
        sent.add(key)
        save_sent(sent)
        log.info("signal sent %s", key)


def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("ضع TELEGRAM_TOKEN و TELEGRAM_CHAT_ID أولاً")
    if "--test" in sys.argv:
        send("✅ البوت يعمل")
        return
    sent = load_sent()
    while True:
        try:
            scan_once(sent)
        except Exception as e:
            log.warning("error: %s", e)
        if "--once" in sys.argv:
            break
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
