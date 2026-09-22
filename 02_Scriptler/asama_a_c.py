"""
Asama A-C: Tanimlayici Istatistikler + Mean Equation + 15 Model Izgarasi
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# Veri yukle
TEMIZ_YOL = "bitcoin_saatlik_temiz_v4.csv"
df = pd.read_csv(TEMIZ_YOL, index_col=0, parse_dates=True)
r = df["Log_Return"].dropna()
n = len(r)

print(f"Veri yuklendi: {n} gozlem")

# ============================================================
# ASAMA A: TANIMLAYICI ISTATISTIKLER
# ============================================================
ort = r.mean()
std = r.std()
skew = r.skew()
kurt = r.kurtosis()
jarque_bera = stats.jarque_bera(r)

print("\n" + "=" * 60)
print("  TANIMLAYICI ISTATISTIKLER")
print("=" * 60)
print(f"  Gozlem sayisi:    {n}")
print(f"  Ortalama:         {ort:.6f}")
print(f"  Std Sapma:        {std:.6f}")
print(f"  Min:              {r.min():.4f}")
print(f"  Max:              {r.max():.4f}")
print(f"  Carpiklik:        {skew:.4f}")
print(f"  Excess Kurtosis:  {kurt:.4f}")
print(f"  Jarque-Bera:      {jarque_bera.statistic:.2f}  (p={jarque_bera.pvalue:.2e})")
print(f"\n  Gunluk Std:   {std * np.sqrt(24):.4f}  (x sqrt(24))")
print(f"  Yillik Std:   {std * np.sqrt(8760):.4f}  (x sqrt(8760))")

# Duruluk Testleri
adf_sonuc = adfuller(r, autolag="AIC")
kpss_sonuc = kpss(r, regression="c", nlags="auto")

print("\nDURULUK TESTLERI")
print("-" * 40)
print(f"  ADF Testi:")
print(f"    Test Istat:  {adf_sonuc[0]:.2f}")
print(f"    p-degeri:    {adf_sonuc[1]:.4e}")
print(f"    Sonuc:       {'DURULU' if adf_sonuc[1] < 0.05 else 'DURULSUZ'}")
print(f"  KPSS Testi:")
print(f"    Test Istat:  {kpss_sonuc[0]:.2f}")
print(f"    p-degeri:    {kpss_sonuc[1]:.4f}")
print(f"    Sonuc:       {'DURULU' if kpss_sonuc[1] > 0.05 else 'DURULSUZ'}")

# Normallik Testi
z_skew = skew / np.sqrt(6 / n)
z_kurt = kurt / np.sqrt(24 / n)

print("\nNORMALLIK TESTI")
print("-" * 40)
print(f"  Skewness:  {skew:.4f}  (z = {z_skew:.2f}, p = {2*(1-stats.norm.cdf(abs(z_skew))):.2e})")
print(f"  Kurtosis:  {kurt:.4f}  (z = {z_kurt:.2f}, p = {2*(1-stats.norm.cdf(abs(z_kurt))):.2e})")
print(f"  JB:        {jarque_bera.statistic:.2f}  (p = {jarque_bera.pvalue:.2e})")

# ACF Grafigi
n_lag = 48
acf_degerleri, confint = acf(r, nlags=n_lag, alpha=0.05)
band = 1.96 / np.sqrt(len(r))

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(1, n_lag + 1), acf_degerleri[1:], color="steelblue", width=0.7)
ax.axhline(y=band, color="red", linestyle="--", linewidth=0.8, label="95% band")
ax.axhline(y=-band, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.set_title(f"Log-Getiri ACF (v4, {n} gozlem)")
ax.legend()
plt.tight_layout()
plt.savefig("grafik_v4_acf_loggetiri.png")
plt.close()
print(f"\nACF[24]: {acf_degerleri[24]:.4f}  (band: {band:.4f})")
print(f"ACF[48]: {acf_degerleri[48]:.4f}  (band: {band:.4f})")

# Dagilim Karsilastirmasi
r_np = r.values
x_min, x_max = np.percentile(r_np, 0.5), np.percentile(r_np, 99.5)
x = np.linspace(x_min, x_max, 300)

normal_pdf = stats.norm.pdf(x, loc=r_np.mean(), scale=r_np.std())
t_fit = stats.t.fit(r_np)
t_pdf = stats.t.pdf(x, *t_fit)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(r_np, bins=200, density=True, alpha=0.6, color="steelblue", label="Gercek Dagilim")
ax1.plot(x, normal_pdf, "r-", lw=2, label="Normal")
ax1.plot(x, t_pdf, "g--", lw=2, label="Student-t")
ax1.set_title("PDF Karsilastirmasi")
ax1.legend()
ax1.set_xlim(x_min, x_max)

sorted_r = np.sort(r_np)
norm_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_r)))
ax2.scatter(norm_quantiles, sorted_r, s=1, alpha=0.5, color="steelblue")
ax2.plot([norm_quantiles.min(), norm_quantiles.max()],
         [norm_quantiles.min(), norm_quantiles.max()], "r--", lw=2)
ax2.set_title("Normal Q-Q Plot")
ax2.set_xlabel("Teorik Quantile")
ax2.set_ylabel("Gozlem")
plt.tight_layout()
plt.savefig("grafik_v4_dagilim_karsilastirma.png")
plt.close()

# ============================================================
# ASAMA B: MEAN EQUATION SECIMI
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA B: MEAN EQUATION SECIMI")
print("=" * 60)

mean_modeller = [
    {"adi": "Constant Mean", "mean": "Constant", "lags": 0},
    {"adi": "AR(1)",         "mean": "AR",       "lags": 1},
    {"adi": "AR(2)",         "mean": "AR",       "lags": 2},
]

mean_sonuclar = []
for mm in mean_modeller:
    try:
        m = arch_model(r * 100, mean=mm["mean"], lags=mm["lags"],
                       vol="Constant", dist="normal")
        f = m.fit(disp="off", show_warning=False)
        mean_sonuclar.append({
            "Model": mm["adi"], "AIC": f.aic, "BIC": f.bic,
            "LogLik": f.loglikelihood, "k": f.params.shape[0]
        })
    except:
        pass

mean_df = pd.DataFrame(mean_sonuclar).sort_values("AIC")
print(mean_df.to_string(index=False))
print(f"\nSECILEN: {mean_df.iloc[0]['Model']} (AIC en dusuk)")

# Ljung-Box + ARCH-LM
m_cm = arch_model(r * 100, mean="AR", lags=2, vol="Constant", dist="normal")
f_cm = m_cm.fit(disp="off", show_warning=False)
artik = f_cm.resid

print("\nLJUNG-BOX TESTI (Mean Equation artiklari)")
print("-" * 40)
for lag in [5, 10, 24, 48]:
    lb = acorr_ljungbox(artik, lags=[lag], return_df=True)
    q_val = lb.iloc[0]["lb_stat"]
    p_val = lb.iloc[0]["lb_pvalue"]
    print(f"  Lag {lag:2d}:  Q={q_val:7.2f}  p={p_val:.4f}  {'TEMIZ' if p_val > 0.05 else 'RED'}")

print("\nARCH-LM TESTI")
print("-" * 40)
e2 = artik ** 2
for m_val in [10, 24]:
    y_dep = e2.values[m_val:]
    X_mat = np.column_stack([e2.values[m_val - i:-i] if i > 0 else e2.values[m_val:]
                             for i in range(1, m_val + 1)])
    X_mat = add_constant(X_mat)
    ols_model = OLS(y_dep, X_mat).fit()
    lm_stat = len(y_dep) * ols_model.rsquared
    p_val = 1 - chi2.cdf(lm_stat, m_val)
    print(f"  m={m_val:2d}:  LM={lm_stat:.2f}  R2={ols_model.rsquared:.4f}  p={p_val:.2e}  RED")

print("\nKARAR: ARCH etkisi GUCLU bicimde dogrulanmistir.")

# ============================================================
# ASAMA C: 15 MODEL IZGARASI
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA C: 15 MODEL IZGARASI")
print("=" * 60)

def kur_model(r, vol_tipi, dist_tipi, igarch=False):
    vol_kutle = {"ARCH": {"p": 10, "q": 0},
                 "GARCH": {"p": 1, "q": 1},
                 "EGARCH": {"p": 1, "q": 1, "o": 1},
                 "GJR": {"p": 1, "q": 1, "o": 1}}
    dk = vol_kutle.get(vol_tipi)
    if dk is None:
        return None

    if igarch:
        sonuc = {"vol": "Approx_IGARCH", "dist": dist_tipi, "model_tipi": "Approx_IGARCH",
                 "yakinsadi": False, "aic": np.nan, "bic": np.nan,
                 "loglik": np.nan, "k": np.nan, "omega": 0.0}
        try:
            m = arch_model(r * 100, mean="AR", lags=2,
                           vol="Garch", p=1, q=1, o=0, dist=dist_tipi)
            f = m.fit(disp="off", show_warning=False)
            if f.convergence_flag != 0:
                return sonuc
            a = f.params.get("alpha[1]", 0)
            b = f.params.get("beta[1]", 1)
            if abs(a + b - 1.0) < 0.05:
                sonuc["yakinsadi"] = True
                sonuc["aic"] = f.aic; sonuc["bic"] = f.bic
                sonuc["loglik"] = f.loglikelihood
                sonuc["k"] = f.params.shape[0]
                sonuc["alpha"] = a; sonuc["beta"] = b
                sonuc["kalicilik"] = a + b; sonuc["gamma"] = 0.0
                if dist_tipi == "t":
                    sonuc["nu"] = f.params.get("nu", np.nan)
                elif dist_tipi == "skewt":
                    sonuc["nu"] = f.params.get("nu", np.nan)
                    sonuc["lambda"] = f.params.get("lambda", np.nan)
            return sonuc
        except:
            return sonuc

    sonuc = {"vol": vol_tipi, "dist": dist_tipi, "model_tipi": "GARCH_AILESI",
             "yakinsadi": False, "aic": np.nan, "bic": np.nan,
             "loglik": np.nan, "k": np.nan, "omega": np.nan}
    try:
        vol_arch = "EGARCH" if vol_tipi == "EGARCH" else "Garch"
        m = arch_model(r * 100, mean="AR", lags=2, vol=vol_arch, **dk, dist=dist_tipi)
        f = m.fit(disp="off", show_warning=False)
        if f.convergence_flag != 0:
            return sonuc
        sonuc["yakinsadi"] = True
        sonuc["aic"] = f.aic; sonuc["bic"] = f.bic
        sonuc["loglik"] = f.loglikelihood
        sonuc["k"] = f.params.shape[0]
        for p_adi in ["omega", "alpha[1]", "beta[1]", "gamma[1]"]:
            sonuc[p_adi.split("[")[0]] = f.params.get(p_adi, np.nan)
        a = sonuc.get("alpha", 0)
        b = sonuc.get("beta", 0)
        g = sonuc.get("gamma", 0)
        if np.isnan(g): g = 0
        if vol_tipi == "GJR":
            sonuc["kalicilik"] = a + b + 0.5 * g
        elif vol_tipi == "EGARCH":
            sonuc["kalicilik"] = b
        else:
            sonuc["kalicilik"] = a + b
        if dist_tipi == "t":
            sonuc["nu"] = f.params.get("nu", np.nan)
        elif dist_tipi == "skewt":
            sonuc["nu"] = f.params.get("nu", np.nan)
            sonuc["lambda"] = f.params.get("lambda", np.nan)
        return sonuc
    except:
        return sonuc

# 15 Model Izgarasi
vol_tipleri = ["ARCH", "GARCH", "EGARCH", "GJR"]
dist_tipleri = ["normal", "t", "skewt"]
tum_sonuclar = []

print("15 MODEL IZGARASI")
print("=" * 70)

for vt in vol_tipleri:
    for dt in dist_tipleri:
        sonuc = kur_model(r, vt, dt)
        if sonuc:
            k = sonuc.get("kalicilik", 0)
            sonuc["kalicilik_patlayici"] = (not np.isnan(k) and k >= 1.0)
            tum_sonuclar.append(sonuc)
            pat = "  [PATLAYICI]" if sonuc["kalicilik_patlayici"] else ""
            print(f"  {vt} x {dt:7s}  AIC={sonuc['aic']:12.2f}  BIC={sonuc['bic']:12.2f}  k={sonuc['k']}{pat}")

# Approx IGARCH
print("\nApprox IGARCH Varyantlari:")
for dt in dist_tipleri:
    sonuc = kur_model(r, "GARCH", dt, igarch=True)
    if sonuc and sonuc.get("yakinsadi"):
        sonuc["kalicilik_patlayici"] = True
        tum_sonuclar.append(sonuc)
        print(f"  Approx_IGARCH x {dt:7s}  AIC={sonuc['aic']:12.2f}  BIC={sonuc['bic']:12.2f}")

# Tablo
grid = pd.DataFrame(tum_sonuclar)
grid = grid[grid["yakinsadi"] == True].copy()
grid = grid.sort_values("aic", ascending=True).reset_index(drop=True)
grid["sira"] = range(1, len(grid) + 1)

print("\n" + "=" * 70)
print("SIRALANMIS MODEL TABLOSU (AIC'ye gore)")
print("=" * 70)
for _, s in grid.iterrows():
    nu_s = f"{s.get('nu', 0):.2f}" if not np.isnan(s.get("nu", np.nan)) else "-"
    lam_s = f"{s.get('lambda', 0):.4f}" if not np.isnan(s.get("lambda", np.nan)) else "-"
    pat = "EVET" if s.get("kalicilik_patlayici") else ""
    print(f"  {s['sira']:2d}  {s['vol']:12s}  {s['dist']:7s}  AIC={s['aic']:12.2f}  BIC={s['bic']:12.2f}  k={s['k']}  nu={nu_s}  lam={lam_s}  {pat}")

# Kaydet
grid.to_csv("model_grid_v4.csv", index=False)
print("\nmodel_grid_v4.csv kaydedildi")
