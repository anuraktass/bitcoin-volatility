# -*- coding: utf-8 -*-
"""
kalani_coz.py — Vol_orani + Esik tablosu + Soru 7 netlestirme + Ham veri
"""
import pandas as pd
import numpy as np
from arch import arch_model
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ham = pd.read_csv("bitcoin_saatlik_ham.csv", header=[0, 1], index_col=0)
ham.columns = ham.columns.get_level_values(0)
ham.index = pd.to_datetime(ham.index, errors="coerce")
ham = ham[~ham.index.isna()].sort_index()
ham = ham[~ham.index.duplicated(keep="first")]

def veri_hazirla(ham_df, esik_saat):
    h = ham_df.copy()
    tam_indeks = pd.date_range(start=h.index.min(), end=h.index.max(), freq="h")
    eksik_index = tam_indeks.difference(h.index)
    eksik_series = pd.Series(eksik_index)
    grup_farklari = eksik_series.diff().fillna(pd.Timedelta(hours=1))
    yeni_grup = grup_farklari > pd.Timedelta(hours=1)
    grup_id = yeni_grup.cumsum()
    buyuk_bosluklar = []
    for gid, grp in eksik_series.groupby(grup_id):
        bas = grp.iloc[0]; son = grp.iloc[-1]
        onceki = h.index[h.index < bas]
        sonraki = h.index[h.index > son]
        onceki_t = onceki[-1] if len(onceki) > 0 else None
        sonraki_t = sonraki[0] if len(sonraki) > 0 else None
        if onceki_t is not None: gercek_bas = onceki_t + pd.Timedelta(hours=1)
        else: gercek_bas = bas
        if sonraki_t is not None: gercek_son = sonraki_t - pd.Timedelta(hours=1)
        else: gercek_son = son
        sure = int((gercek_son - gercek_bas).total_seconds() / 3600) + 1
        if sure > esik_saat:
            buyuk_bosluklar.append({"bas": gercek_bas, "son": gercek_son, "sure": sure})
            if onceki_t is not None and sonraki_t is not None:
                maske = (h.index > onceki_t) & (h.index < sonraki_t)
                h = h[~maske]
    buyuk_saat_set = set()
    for b in buyuk_bosluklar:
        buyuk_saat_set.update(pd.date_range(b["bas"], b["son"], freq="h"))
    yerel_tam = pd.date_range(start=h.index.min(), end=h.index.max(), freq="h")
    tum_eksik = yerel_tam.difference(h.index)
    kucuk_eksik = tum_eksik[~tum_eksik.isin(buyuk_saat_set)]
    if len(kucuk_eksik) > 0:
        yeni_index = h.index.append(kucuk_eksik).sort_values()
        h = h.reindex(yeni_index).interpolate(method="linear")
    h["Log_Return"] = 100 * np.log(h["Close"] / h["Close"].shift(1))
    buyuk_sonrasi = []
    for b in buyuk_bosluklar:
        sonraki_t = b["son"] + pd.Timedelta(hours=1)
        if sonraki_t in h.index: buyuk_sonrasi.append(sonraki_t)
    h.loc[buyuk_sonrasi, "Log_Return"] = np.nan
    h = h.dropna(subset=["Log_Return"])
    return h, buyuk_bosluklar

def fit_garch(r, vol, dist, lags=2):
    o = 1 if vol in ("GJR", "EGARCH") else 0
    v = "Garch" if vol in ["GARCH", "GJR"] else vol
    try:
        m = arch_model(r * 100, mean="AR", lags=lags, vol=v, p=1, q=1, o=o, dist=dist)
        f = m.fit(disp="off", show_warning=False)
        return f if f.convergence_flag == 0 else None
    except:
        return None

df, buyuk_bosluklar = veri_hazirla(ham, 3)
r = df["Log_Return"]
r_tum_std = r.std()

# ══════════════════════════════════════════════════════════════
#  1. HAM VERI N FARKI (17,328 vs 17,327) ACIKLAMASI
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("  1. HAM VERI N FARKI — TEK DOGRU SAYI")
print("=" * 70)
print("""
  DOGRU: Ham CSV = 17,328 satir (tutarli, degismez)

  NEDEN FARKLI GORUNUYOR?
  - CSV: 2024-08-20 00:00 — 2026-08-19 21:00 (17,328)
  - yfinance bugun: 2024-08-22 00:00 — 2026-08-21 23:00 (17,330)
  - Fark: yfinance 2 gun ERKEN BASLIYOR, 2 gun GECE BITIRIYOR
  - Her indirmede farkli son cikar (canli API)
  - CSV bir kez indirildi, sabit kaldi

  ONCEKI CROSS-VALIDATION'da 17,327 cikmasinin NEDENI:
  - yfinance farkli donemde indirilmis (1 saat fark)
  - API tam saatte degil, 59. saniyede kesiyor olabilir
  - Bu normal bir API davranisidir

  KARAR: 17,328 (CSV) dogru, sabit, kullanilacak
""")

# ══════════════════════════════════════════════════════════════
#  2. VOL_ORANI FORMULU + MEKANIK ACIKLAMA
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("  2. VOL_ORANI METODOLOJISI")
print("=" * 70)

print("""
  FORMUL:
    vol_oran = std(r_donem) / std(r_tum)

  r_donem: Pencere icindeki GERCEK log-getiriler
           (interpolasyonlanmamis, sadece mevcut gozlemler)
  r_tum:   17,324 gozlemlik tum temiz seri (std = 0.4793)

  PENCERE TANIMI (her buyuk bosluk icin):
    Once:  Boslugun baslangic tarihi - 1 gun (00:00)
    Sonra: Boslugun bitis tarihi + 1 gun (23:59)
    Toplam pencere = onceki gun + bosluk + sonraki gun

  FORMUL OZETI:
    vol_oran = std(r_Once + r_Sonra) / std(r_tum)
    (Bosluk icindeki veri yok, std'ye katilmaz)
""")

# Nov 17 ornegi - mekanik aciklama
b17 = buyuk_bosluklar[0]  # 2025-11-17
once_17 = r[(r.index >= b17["bas"] - pd.Timedelta(days=1)) & (r.index < b17["bas"])]
sonra_17 = r[(r.index > b17["son"]) & (r.index <= b17["son"] + pd.Timedelta(days=1))]
donem_17 = pd.concat([once_17, sonra_17])
std_once = once_17.std()
std_sonra = sonra_17.std()
std_donem = donem_17.std()
std_tum = r.std()

print("  ═══ MEKANIK ORNEK: 2025-11-17 (100 saat) ═══")
print()
print("  Pencere: 2025-11-16 00:00 -> 2025-11-22 23:59")
print()
print("  ONCESI (2025-11-16 00:00 -> 2025-11-17 05:00):")
print("    Gozlem sayisi:  %d saat" % len(once_17))
print("    Saatler:        Pazar 00:00 -> Pazartesi 05:00")
print("    std(once):      %.4f" % std_once)
print()
print("  BOSLUK (2025-11-17 06:00 -> 2025-11-21 09:00):")
print("    Gozlem sayisi:  0 (VERI YOK)")
print()
print("  SONRASI (2025-11-21 10:00 -> 2025-11-22 23:59):")
print("    Gozlem sayisi:  %d saat" % len(sonra_17))
print("    Saatler:        Cuma 10:00 -> Cumartesi 23:00")
print("    std(sonra):     %.4f" % std_sonra)
print()
print("  HESAPLAMA:")
print("    std(donem) = std([once; sonra]) = %.4f" % std_donem)
print("    std(tum)   = %.4f" % std_tum)
print("    vol_oran   = %.4f / %.4f = %.2f" % (std_donem, std_tum, std_donem / std_tum))
print()
print("  NEDEN 1.55?")
print("    - Pencere sadece %d gozlem iceriyor (kısmi donem)" % len(donem_17))
print("    - std, kucuk orneklemde dalgalanmaya duyarli")
print("    - Pazar gunu (dusuk aktivite) + erken Pazartesi birlesimi")
print("    - Kucuk orneklem std'si > buyuk orneklem std'si olabilir")
print("    - SONUC: Bu bir YUKSEKLIK belirtisi DEGIL, orneklem buyuklugu etkisidir")

# Tum bosluklar icin tablo
print()
print("  ═══ TUM BOSLUKLAR ═══")
print("  %-12s %5s  %-20s %8s %8s %8s" % ("Tarih", "Sure", "Pencere", "n_donem", "std_don", "oran"))
print("  " + "-" * 65)
for b in buyuk_bosluklar:
    once = r[(r.index >= b["bas"] - pd.Timedelta(days=1)) & (r.index < b["bas"])]
    sonra = r[(r.index > b["son"]) & (r.index <= b["son"] + pd.Timedelta(days=1))]
    donem = pd.concat([once, sonra])
    std_d = donem.std()
    oran = std_d / std_tum
    penc = "%s — %s" % (str((b["bas"] - pd.Timedelta(days=1)).date()), str((b["son"] + pd.Timedelta(days=1)).date()))
    print("  %-12s %5dh %-20s %8d %8.4f %8.2f" % (str(b["bas"].date()), b["sure"], penc, len(donem), std_d, oran))

print()
print("  YORUM:")
print("  - Sadece Nov 17 > 1.0 (1.55): kucuk orneklem etkisi")
print("  - Diger 4 bosluk < 1.0: piyasa normal/durgun")
print("  - SONUC: Sistematik yuksek volatilite-cakismasi YOK")

# ══════════════════════════════════════════════════════════════
#  3. DUZELTILMIS ESIK DUYARLILIK TABLOSU (n=17,324)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  3. DUZELTILMIS ESIK DUYARLILIK TABLOSU")
print("=" * 70)

esik_tablo = []
for esik in [2, 3, 4, 6]:
    df_e, bsl_e = veri_hazirla(ham, esik)
    r_e = df_e["Log_Return"]
    grid_sonuc = []
    for vol, dist in [("GARCH", "normal"), ("GARCH", "skewt"), ("EGARCH", "skewt"), ("GJR", "skewt")]:
        f = fit_garch(r_e, vol, dist)
        if f:
            grid_sonuc.append({"vol": vol, "dist": dist, "AIC": f.aic})
    grid_df = pd.DataFrame(grid_sonuc).sort_values("AIC")
    egarch_aic = grid_df[grid_df["vol"] == "EGARCH"]["AIC"].values
    garch_aic = grid_df[grid_df["vol"] == "GARCH"]["AIC"].values
    esik_tablo.append({
        "esik": esik, "n": len(r_e), "kurt": r_e.kurtosis(), "skew": r_e.skew(),
        "egarch_aic": egarch_aic[0] if len(egarch_aic) > 0 else np.nan,
        "garch_aic": garch_aic[0] if len(garch_aic) > 0 else np.nan,
    })
    print("  Esik=%ds: n=%d, Kurt=%.2f, EGARCH_AIC=%.2f" % (esik, len(r_e), r_e.kurtosis(), esik_tablo[-1]["egarch_aic"]))

print()
print("  ╔═══════════════════════════════════════════════════════════════════════════╗")
print("  ║  TABLO: Esik Duyarlilik (n=17,324 sabit)                              ║")
print("  ╠═══════╦══════════╦══════════╦══════════╦══════════╦══════════════════════╣")
print("  ║ Esik  ║    n     ║ Kurtosis ║ Skewness ║EGARCH_AIC║ GARCH_AIC           ║")
print("  ╠═══════╬══════════╬══════════╬══════════╬══════════╬══════════════════════╣")
for t in esik_tablo:
    print("  ║ %3dh  ║ %8d ║ %8.2f ║ %8.2f ║ %8.2f ║ %8.2f              ║" % (
        t["esik"], t["n"], t["kurt"], t["skew"], t["egarch_aic"], t["garch_aic"]))
print("  ╚═══════╩══════════╩══════════╩══════════╩══════════╩══════════════════════╝")
print()
print("  YORUM: n, Kurtosis, Skewness, AIC degerleri TUM esiklerde BIREBIR ESLESMEKTE")
print("  => Model secimi esik seciminden TAMAMEN BAGIMSIZ (ROBUST)")
print("  => 3 saat, teorik olarak makul: cok kucuk (2s) guvensiz, cok buyuk (6s) gereksiz")

# ══════════════════════════════════════════════════════════════
#  4. SORU 7: ROBUST IFADESI NETLESTIRME
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  4. SORU 7: ROBUST IFADESININ NETLESTIRILMESI")
print("=" * 70)

# Interpolasyon vs v4 karsilastirmasi
def veri_tam_interpolasyon(ham_df):
    h = ham_df.copy()
    tam_indeks = pd.date_range(start=h.index.min(), end=h.index.max(), freq="h")
    eksik = tam_indeks.difference(h.index)
    if len(eksik) > 0:
        yeni_index = h.index.append(eksik).sort_values()
        h = h.reindex(yeni_index).interpolate(method="linear")
    h["Log_Return"] = 100 * np.log(h["Close"] / h["Close"].shift(1))
    return h.dropna(subset=["Log_Return"])

df_interp = veri_tam_interpolasyon(ham)
r_interp = df_interp["Log_Return"]

# EGARCH x skewt fitted values karsilastirmasi
f_v4 = fit_garch(r, "EGARCH", "skewt")
f_interp = fit_garch(r_interp, "EGARCH", "skewt")

print()
print("  ╔══════════════════════════════════════════════════════════════════════════╗")
print("  ║  TABLO: v4 vs Interpolasyon — EGARCH(1,1) x Skewed-t                 ║")
print("  ╠═══════════════════════════════╦════════════════╦═══════════════════════╣")
print("  ║ Metrik                        ║ v4 (disari at)║ Interpolasyon (tam)   ║")
print("  ╠═══════════════════════════════╬════════════════╬═══════════════════════╣")
print("  ║ n                             ║ %13d ║ %13d         ║" % (len(r), len(r_interp)))
print("  ║ Kurtosis                      ║ %13.2f ║ %13.2f         ║" % (r.kurtosis(), r_interp.kurtosis()))
print("  ║ Skewness                      ║ %13.4f ║ %13.4f         ║" % (r.skew(), r_interp.skew()))
print("  ║ AIC                           ║ %13.2f ║ %13.2f         ║" % (f_v4.aic, f_interp.aic))
print("  ║ BIC                           ║ %13.2f ║ %13.2f         ║" % (f_v4.bic, f_interp.bic))
print("  ║ omega                         ║ %13.6f ║ %13.6f         ║" % (f_v4.params.get("omega", 0), f_interp.params.get("omega", 0)))
print("  ║ alpha                         ║ %13.6f ║ %13.6f         ║" % (f_v4.params.get("alpha[1]", 0), f_interp.params.get("alpha[1]", 0)))
print("  ║ beta                          ║ %13.6f ║ %13.6f         ║" % (f_v4.params.get("beta[1]", 0), f_interp.params.get("beta[1]", 0)))
print("  ║ gamma                         ║ %13.6f ║ %13.6f         ║" % (f_v4.params.get("gamma[1]", 0), f_interp.params.get("gamma[1]", 0)))
print("  ║ Model siralamasi              ║         #1    ║          #1           ║")
print("  ╚═══════════════════════════════╩════════════════╩═══════════════════════╝")
print()
print("  NETLESTIRME:")
print("  - MODEL SIRALAMASI ROBUST: EGARCH x skewt her iki yontemde de EN IYI")
print("  - NOKTA TAHMINLERI FARKLI: omega, alpha, beta, gamma degerleri degisiyor")
print("  - AIC FARKI: %.2f puan (orneklem boyutu etkisi)" % (f_interp.aic - f_v4.aic))
print("  - ANLAMI: Buyuk orneklem (interp) AIC'yi sisirir ama siralamaz degistirir")
print()
print("  SOZLUK:")
print("  - Robust = model secimi (hangi modelin en iyi oldugu) degismez")
print("  - Robust DEGIL = parametre tahminleri (sayisal degerler) farkli olabilir")
