"""
Asama D-F: Dagilim Etkisi + Tansal Kontrol + Nihai Model Secimi
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Veri yukle
r = pd.read_csv("bitcoin_saatlik_temiz_v4.csv", index_col=0, parse_dates=True)["Log_Return"].dropna()
grid = pd.read_csv("model_grid_v4.csv")

print(f"Veri yuklendi: {len(r)} gozlem, {len(grid)} model")

# ============================================================
# ASAMA D: DAGILIM ETKISI VE HABER ETKI EGRILERI
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA D: DAGILIM ETKISI")
print("=" * 60)

print("\nHer volatilite yapisinda Normal'den Skewed-t'ye AIC iyilesmesi:")
print(f"  {'Vol':10s} {'AIC_norm':12s} {'AIC_skewt':12s} {'Fark':12s}  Yorum")
print("  " + "-" * 60)
for vol in ["ARCH", "GARCH", "EGARCH", "GJR"]:
    norm_aic = grid[(grid["vol"] == vol) & (grid["dist"] == "normal")]["aic"].values
    skewt_aic = grid[(grid["vol"] == vol) & (grid["dist"] == "skewt")]["aic"].values
    if len(norm_aic) > 0 and len(skewt_aic) > 0:
        fark = norm_aic[0] - skewt_aic[0]
        yorum = "COK GUCLU" if fark > 10 else "Guclu" if fark > 2 else "Zayif"
        print(f"  {vol:10s} {norm_aic[0]:12.2f} {skewt_aic[0]:12.2f} {fark:12.2f}  {yorum}")

# Haber Etki Egrileri
model_sozluk = {}
for vol, dt in [("EGARCH", "normal"), ("GJR", "normal"), ("GARCH", "normal"),
                ("EGARCH", "t"), ("GJR", "t")]:
    try:
        vol_map = {"GARCH": "Garch", "GJR": "Garch", "EGARCH": "EGARCH"}
        m = arch_model(r * 100, mean="AR", lags=2, vol=vol_map[vol], p=1, q=1, o=1, dist=dt)
        f = m.fit(disp="off", show_warning=False)
        model_sozluk[f"{vol}_{dt}"] = f
    except:
        pass

e_values = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (vol_adi, ekler) in enumerate([("EGARCH", "EGARCH"), ("GJR", "GJR-GARCH"), ("GARCH", "GARCH")]):
    ax = axes[idx]
    for dt, renk, styl in [("normal", "blue", "-"), ("t", "red", "--")]:
        anahtar = f"{vol_adi}_{dt}"
        if anahtar not in model_sozluk:
            continue
        f = model_sozluk[anahtar]
        a = f.params.get("alpha[1]", 0)
        b = f.params.get("beta[1]", 0)
        g = f.params.get("gamma[1]", 0)
        h2 = f.conditional_volatility.iloc[-1] ** 2
        sigma_artik = e_values * 0.479
        if vol_adi == "EGARCH":
            nic = np.exp(a * (np.abs(sigma_artik / 0.479) - np.sqrt(2/np.pi))
                         + g * (sigma_artik / 0.479)) * h2
        elif vol_adi == "GJR":
            nic = np.where(sigma_artik >= 0, (a + 0) * sigma_artik**2 + b * h2,
                           (a + g) * sigma_artik**2 + b * h2)
        else:
            nic = a * sigma_artik**2 + b * h2
        ax.plot(e_values, nic, color=renk, linestyle=styl, linewidth=2, label=f"{ekler} ({dt})")
    ax.set_xlabel("Artik deger (e_t / std)")
    ax.set_ylabel("Kosullu varyans (h_t)")
    ax.set_title(f"{ekler} Haber Etki Egrisi")
    ax.legend()
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("grafik_v4_haber_etki.png")
plt.close()
print("\nHaber etki grafikleri kaydedildi")

# Gamma karsilastirmasi
print("\nGAMMA (asimetri) katsayisi karsilastirmasi:")
print(f"  {'Model':18s} {'Dist':8s} {'gamma':>8s} {'p-degeri':>10s} Durum")
print("  " + "-" * 55)
for anahtar, f in model_sozluk.items():
    if "gamma[1]" in f.params.index:
        gamma = f.params["gamma[1]"]
        pval = f.pvalues["gamma[1]"]
        durum = "ANLAMLI" if pval < 0.05 else "ANLAMSIZ"
        vol_dist = anahtar.split("_")
        print(f"  {vol_dist[0]:18s} {vol_dist[1]:8s} {gamma:8.4f} {pval:10.4f} {durum}")

# ============================================================
# ASAMA E: TANSAL KONTROL
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA E: TANSAL KONTROL")
print("=" * 60)

tum_diag = []
for _, satir in grid.iterrows():
    vol = satir["vol"]
    dist = satir["dist"]
    if vol in ("IGARCH", "Approx_IGARCH"):
        continue
    try:
        if vol == "GJR":
            m = arch_model(r * 100, mean="AR", lags=2, vol="Garch", p=1, q=1, o=1, dist=dist)
        elif vol == "EGARCH":
            m = arch_model(r * 100, mean="AR", lags=2, vol="EGARCH", p=1, q=1, dist=dist)
        else:
            m = arch_model(r * 100, mean="AR", lags=2, vol="Garch", p=1, q=1, o=0, dist=dist)
        f = m.fit(disp="off", show_warning=False)
        std_resid = f.std_resid.dropna()
        if len(std_resid) < 50:
            continue
        lb24 = acorr_ljungbox(std_resid, lags=[24], return_df=True)
        lb48 = acorr_ljungbox(std_resid, lags=[min(48, len(std_resid)//2-1)], return_df=True)
        lb24_p = lb24.iloc[0]["lb_pvalue"]
        lb48_p = lb48.iloc[0]["lb_pvalue"]
        e2 = std_resid ** 2
        m_val = 24
        y_dep = e2.values[m_val:]
        X_mat = np.column_stack([e2.values[m_val - i:-i] if i > 0 else e2.values[m_val:]
                                 for i in range(1, m_val + 1)])
        X_mat = add_constant(X_mat)
        ols_m = OLS(y_dep, X_mat).fit()
        lm_stat = len(y_dep) * ols_m.rsquared
        arch_lm_p = 1 - chi2.cdf(lm_stat, m_val)
        temiz = (lb24_p > 0.05) and (lb48_p > 0.05) and (arch_lm_p > 0.05)
        tum_diag.append({"vol": vol, "dist": dist, "LB24-p": lb24_p, "LB48-p": lb48_p,
                         "ARCH-LM-p": arch_lm_p, "R2": ols_m.rsquared, "temiz": temiz})
    except:
        pass

diag_df = pd.DataFrame(tum_diag)
print("\nTANISAL KONTROL")
print("=" * 75)
print(f"  {'Vol':8s} {'Dist':8s} {'LB24-p':>10s} {'LB48-p':>10s} {'LM-p':>10s} {'R2':>8s} Temiz")
print("  " + "-" * 65)
for _, s in diag_df.iterrows():
    print(f"  {s['vol']:8s} {s['dist']:8s} {s['LB24-p']:10.4f} {s['LB48-p']:10.4f} {s['ARCH-LM-p']:10.4f} {s['R2']:8.4f} {'HAYIR' if not s['temiz'] else 'EVET'}")

print(f"\nTemiz model sayisi: {diag_df['temiz'].sum()} / {len(diag_df)}")

# Kaydet
diag_df.to_csv("diagnostic_tests_v4.csv", index=False)
print("diagnostic_tests_v4.csv kaydedildi")

# ============================================================
# ASAMA F: NIHAI MODEL SECIMI
# ============================================================
print("\n" + "=" * 60)
print("  ASAMA F: NIHAI MODEL SECIMI")
print("=" * 60)

secim = grid.merge(diag_df[["vol", "dist", "temiz"]], on=["vol", "dist"], how="left")
secim["aic_sira"] = secim["aic"].rank(ascending=True).astype(int)
secim["bic_sira"] = secim["bic"].rank(ascending=True).astype(int)
secim["tanisal_puan"] = secim["temiz"].fillna(False).astype(int) * 3
secim["duruluk_puan"] = (~secim["kalicilik_patlayici"].fillna(False)).astype(int) * 3
secim["toplam_puan"] = (secim["aic_sira"] + secim["bic_sira"]
                        + (6 - secim["tanisal_puan"]) + (6 - secim["duruluk_puan"]))
secim = secim.sort_values("toplam_puan").reset_index(drop=True)

print("\nNIHAI MODEL SECIMI -- 4 KRITER PUAN TABLOSU")
print("=" * 90)
print(f"  {'Sira':4s} {'Vol':10s} {'Dist':8s} {'AIC':>12s} {'BIC':>12s} {'A_sira':>6s} {'B_sira':>6s} {'Tanisal':>7s} {'Duruluk':>7s}  Toplam")
print("  " + "-" * 85)
for idx, s in secim.iterrows():
    print(f"  {idx+1:4d} {s['vol']:10s} {s['dist']:8s} {s['aic']:12.2f} {s['bic']:12.2f}"
          f"  {int(s['aic_sira']):5d}  {int(s['bic_sira']):5d}"
          f"  {'TEMIZ' if s.get('temiz', False) else 'HAYIR':>7s}"
          f"  {'DURGAN' if not s.get('kalicilik_patlayici', True) else 'PATLAYICI':>7s}"
          f"  {int(s['toplam_puan'])}")

en_iyi = secim.iloc[0]
print(f"\n{'=' * 60}")
print(f"NIHAI SECIM: {en_iyi['vol']} x {en_iyi['dist']}")
print(f"AIC: {en_iyi['aic']:.2f}  BIC: {en_iyi['bic']:.2f}")
print(f"Kalicilik: {en_iyi.get('kalicilik', 0):.4f}")

# Kaydet
secim.to_csv("model_select_v4.csv", index=False)
print("\nmodel_select_v4.csv kaydedildi")
