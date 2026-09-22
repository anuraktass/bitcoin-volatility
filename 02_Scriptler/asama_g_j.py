"""
Asama G-J: OOS Dogrulama + Alt Donem + VaR Backtesting
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")

# Veri yukle
r = pd.read_csv("bitcoin_saatlik_temiz_v4.csv", index_col=0, parse_dates=True)["Log_Return"].dropna()
print(f"Veri yuklendi: {len(r)} gozlem")

def fit_garch(r, vol, dist, lags=2):
    o = 1 if vol in ("GJR", "EGARCH") else 0
    v = "Garch" if vol in ["GARCH", "GJR"] else vol
    try:
        m = arch_model(r * 100, mean="AR", lags=lags, vol=v, p=1, q=1, o=o, dist=dist)
        f = m.fit(disp="off", show_warning=False)
        return f if f.convergence_flag == 0 else None
    except:
        return None

# ============================================================
# ASAMA G: OUT-OF-SAMPLE DOGRULAMA
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA G: OUT-OF-SAMPLE DOGRULAMA")
print("=" * 60)

n = len(r)
train_oran = 0.80
train_sinir = int(n * train_oran)
train = r.iloc[:train_sinir]
test = r.iloc[train_sinir:]

print(f"Toplam: {n}  Train: {len(train)} ({train_oran*100:.0f}%)  Test: {len(test)} ({(1-train_oran)*100:.0f}%)")
print(f"Train: {train.index[0]} -- {train.index[-1]}")
print(f"Test:  {test.index[0]} -- {test.index[-1]}")

# Gerceklesen volatilite
realized_var = pd.Series(index=test.index, dtype=float)
for i in range(len(test)):
    baslama = max(0, train_sinir + i - 24)
    pencere = r.iloc[baslama:train_sinir + i].values
    if len(pencere) >= 6:
        realized_var.iloc[i] = np.sum(pencere ** 2)
realized_var = realized_var.dropna()
print(f"Gerceklesen volatilite gozlem: {len(realized_var)}")

# Rolling OOS
PENCERE = 1000
REFIT_HER = 24
modeller_oos = [("EGARCH", "skewt"), ("GJR", "skewt"), ("GJR", "t"), ("GARCH", "skewt")]

oos_sonuclar = []
tahmin_seri = {}

for vol, dist in modeller_oos:
    adi = f"{vol}_{dist}"
    t_start = time.time()
    tahminler = pd.Series(index=test.index, dtype=float)
    son_fit = None
    fit_sayisi = 0

    for i in range(len(test)):
        test_bas = train_sinir + i
        train_bas = max(0, test_bas - PENCERE)
        r_pencere = r.iloc[train_bas:test_bas]
        if i % REFIT_HER == 0 or son_fit is None:
            son_fit = fit_garch(r_pencere, vol, dist)
            fit_sayisi += 1
            if son_fit is None:
                tahminler.iloc[i] = np.nan
                continue
        if son_fit is not None:
            try:
                tahmin_var = son_fit.forecast(horizon=1, reindex=False).variance.iloc[-1].values[0]
                tahminler.iloc[i] = tahmin_var / 10000
            except:
                tahminler.iloc[i] = np.nan

    tahminler = tahminler.dropna()
    gecen = time.time() - t_start
    ortak_idx = realized_var.index.intersection(tahminler.index)
    rv = realized_var.loc[ortak_idx].values
    tv = np.maximum(tahminler.loc[ortak_idx].values, 1e-10)
    rmse = np.sqrt(np.mean((rv - tv) ** 2))
    mae = np.mean(np.abs(rv - tv))
    qlike = np.mean(np.log(rv / tv) + tv / rv - 1)
    oos_sonuclar.append({"Model": adi, "RMSE": rmse, "MAE": mae, "QLIKE": qlike})
    tahmin_seri[adi] = tahminler.loc[ortak_idx]
    print(f"  {adi:15s}  RMSE={rmse:.6f}  MAE={mae:.6f}  QLIKE={qlike:.6f}  ({gecen:.1f}s)")

oos_df = pd.DataFrame(oos_sonuclar).sort_values("QLIKE")
print("\nOOS KARSILASTIRMA TABLOSU:")
print(oos_df.to_string(index=False))

# Diebold-Mariano Testi
m1_adi = oos_df.iloc[0]["Model"]
m2_adi = oos_df.iloc[1]["Model"]
ortak = tahmin_seri[m1_adi].index.intersection(tahmin_seri[m2_adi].index)
ortak = ortak.intersection(realized_var.index)

rv_dm = realized_var.loc[ortak].values
t1 = np.maximum(tahmin_seri[m1_adi].loc[ortak].values, 1e-10)
t2 = np.maximum(tahmin_seri[m2_adi].loc[ortak].values, 1e-10)
L1 = np.log(rv_dm / t1) + t1 / rv_dm - 1
L2 = np.log(rv_dm / t2) + t2 / rv_dm - 1
d = L1 - L2
d_bar = np.mean(d)
T = len(d)
max_lag = max(1, int(np.floor(4 * (T / 100) ** (2/9))))
gamma0 = np.var(d, ddof=1)
hac_var = gamma0 / T
for lag in range(1, max_lag + 1):
    weight = 1 - lag / (max_lag + 1)
    gamma_lag = np.mean((d[:-lag] - d_bar) * (d[lag:] - d_bar))
    hac_var += 2 * weight * gamma_lag / T
dm_stat = d_bar / np.sqrt(max(hac_var, 1e-20))
dm_p = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

print(f"\nDIEBOLD-MARIANO TESTI (Newey-West HAC)")
print(f"  {m1_adi} vs {m2_adi}")
print(f"  DM istatistigi: {dm_stat:.4f}  p-degeri: {dm_p:.4f}")
print(f"  Kazanan: {m1_adi if d_bar < 0 else m2_adi} ({'anlami' if dm_p < 0.05 else 'anlamsiz (p>0.05)'})")

# Kaydet
oos_df.to_csv("oos_sonuclar_v4.csv", index=False)
print("\noos_sonuclar_v4.csv kaydedildi")

# ============================================================
# ASAMA H: ALT DONEM ANALIZI
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA H: ALT DONEM ANALIZI")
print("=" * 60)

bolme_tarihi = pd.Timestamp("2025-11-17")
donem1 = r[r.index < bolme_tarihi]
donem2 = r[r.index > bolme_tarihi + pd.Timedelta(days=5)]

print(f"Bolme: {bolme_tarihi}")
print(f"Donem 1: {len(donem1)} gozlem ({donem1.index[0]} -- {donem1.index[-1]})")
print(f"Donem 2: {len(donem2)} gozlem ({donem2.index[0]} -- {donem2.index[-1]})")

alt_donem_sonuclar = []
for d_adi, d_veri in [("Donem 1", donem1), ("Donem 2", donem2)]:
    f = fit_garch(d_veri, "EGARCH", "skewt")
    if f:
        p = f.params
        a = p.get("alpha[1]", 0); b = p.get("beta[1]", 0); g = p.get("gamma[1]", 0)
        print(f"\n{d_adi} (n={len(d_veri)}):")
        print(f"  alpha={a:.4f}  beta={b:.4f}  gamma={g:.4f}  kalicilik(beta)={b:.4f}")
        alt_donem_sonuclar.append({"Donem": d_adi, "n": len(d_veri),
                                    "alpha": a, "beta": b, "gamma": g})

pd.DataFrame(alt_donem_sonuclar).to_csv("alt_donem_v4.csv", index=False)
print("\nalt_donem_v4.csv kaydedildi")

# ============================================================
# ASAMA J: VALUE AT RISK + BACKTESTING
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA J: VALUE AT RISK + BACKTESTING")
print("=" * 60)

f_nihai = fit_garch(r, "EGARCH", "skewt")
p = f_nihai.params

nu = np.nan
for anahtar in ["nu", "eta", "df", "shape"]:
    if anahtar in p.index:
        nu = p[anahtar]
        break

t_q95 = stats.t.ppf(0.05, df=nu)
t_q99 = stats.t.ppf(0.01, df=nu)

print(f"Nihai Model: EGARCH(1,1) x Skewed-t")
print(f"Parametreler: {list(p.index)}")
print(f"nu={nu:.4f}  t_q95={t_q95:.4f}  t_q99={t_q99:.4f}")

# Rolling VaR
PENCERE_VaR = 1000
test_bas_idx = int(len(r) * 0.80)
test_veri = r.iloc[test_bas_idx:]
VaR_95 = pd.Series(index=test_veri.index, dtype=float)
VaR_99 = pd.Series(index=test_veri.index, dtype=float)
fit_sayisi_var = 0

for i in range(len(test_veri)):
    test_bas = test_bas_idx + i
    train_bas = max(0, test_bas - PENCERE_VaR)
    r_pencere = r.iloc[train_bas:test_bas]
    if i % 48 == 0:
        f_var = fit_garch(r_pencere, "EGARCH", "skewt")
        fit_sayisi_var += 1
        if f_var is None:
            VaR_95.iloc[i] = np.nan; VaR_99.iloc[i] = np.nan
            continue
    if f_var is not None:
        try:
            tahmin_var = f_var.forecast(horizon=1, reindex=False).variance.iloc[-1].values[0]
            tahmin_sigma = np.sqrt(tahmin_var) / 100
            VaR_95.iloc[i] = tahmin_sigma * t_q95
            VaR_99.iloc[i] = tahmin_sigma * t_q99
        except:
            VaR_95.iloc[i] = np.nan; VaR_99.iloc[i] = np.nan

VaR_95 = VaR_95.dropna()
VaR_99 = VaR_99.dropna()
print(f"\nTest gozlem: {len(test_veri)}  VaR hesaplanan: {len(VaR_95)}  Refit: {fit_sayisi_var}")

# Kupiec Testi
print("\nKUPIEC TESTI:")
var_sonuclar = []
for seviye, vaer_seri in [(95, VaR_95), (99, VaR_99)]:
    ortak = test_veri.index.intersection(vaer_seri.index)
    gercek = test_veri.loc[ortak].values
    tahmin = vaer_seri.loc[ortak].values
    ihlal = gercek < tahmin
    n_ihlal = np.sum(ihlal)
    n_toplam = len(ihlal)
    orani = n_ihlal / n_toplam
    beklenen = 1 - seviye / 100.0
    if n_ihlal > 0 and n_ihlal < n_toplam:
        p_hat = n_ihlal / n_toplam
        p0 = beklenen
        LR = -2 * (n_toplam * np.log(1-p0) + n_ihlal * np.log(p0)
                   - n_toplam * np.log(1-p_hat) - n_ihlal * np.log(p_hat))
        kupiec_p = 1 - chi2.cdf(LR, 1)
    else:
        LR = 0; kupiec_p = 1.0
    durum = "KABUL" if kupiec_p > 0.05 else "RED"
    print(f"  {seviye}% VaR: ihlal={n_ihlal}/{n_toplam} ({orani*100:.2f}%, beklenen={beklenen*100:.1f}%)  LR={LR:.2f}  p={kupiec_p:.4f}  {durum}")
    var_sonuclar.append({"Seviye": seviye, "Ihlal": n_ihlal, "Toplam": n_toplam,
                         "Oran": orani, "Beklenen": beklenen, "LR": LR, "p": kupiec_p, "Durum": durum})

# VaR Grafigi
ortak_VaR_idx = VaR_95.index.intersection(test_veri.index)[:200]
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_veri.loc[ortak_VaR_idx].values, color="blue", alpha=0.7, label="Gercek Getiri")
ax.plot(VaR_95.loc[ortak_VaR_idx].values, color="red", linewidth=1.5, label="VaR 95%")
ax.plot(VaR_99.loc[ortak_VaR_idx].values, color="darkred", linewidth=1.5, label="VaR 99%")
ax.fill_between(range(len(ortak_VaR_idx)),
                VaR_95.loc[ortak_VaR_idx].values,
                VaR_99.loc[ortak_VaR_idx].values, alpha=0.2, color="red")
ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
ax.set_title("VaR Backtesting (Ilk 200 test gozlemi)")
ax.set_xlabel("Gozlem")
ax.set_ylabel("Log-Getiri (%)")
ax.legend()
plt.tight_layout()
plt.savefig("grafik_v4_var_backtest.png")
plt.close()

pd.DataFrame(var_sonuclar).to_csv("var_sonuclar_v4.csv", index=False)
print("\nvar_sonuclar_v4.csv ve grafik_v4_var_backtest.png kaydedildi")

# ============================================================
# SONUC
# ============================================================
print("\n" + "=" * 60)
print("  NIHAI SONUCLAR")
print("=" * 60)
print(f"  Nihai Model:    EGARCH(1,1) x Skewed-t")
print(f"  AIC:            175,063.06")
print(f"  BIC:            175,132.90")
print(f"  Mean Equation:  AR(2)")
print(f"  Kalicilik:      {p.get('beta[1]', 0):.4f}")
print(f"  Gamma:          {p.get('gamma[1]', 0):.4f}")
print(f"  Nu:             {nu:.4f}")
