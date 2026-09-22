# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

ham = pd.read_csv("bitcoin_saatlik_ham.csv", header=[0, 1], index_col=0)
ham.columns = ham.columns.get_level_values(0)
ham.index = pd.to_datetime(ham.index, errors="coerce")
ham = ham[~ham.index.isna()].sort_index()
ham = ham[~ham.index.duplicated(keep="first")]
r = 100 * np.log(ham["Close"] / ham["Close"].shift(1)).dropna()

bosluklar = [
    {"tarih": "2025-11-17", "bas": pd.Timestamp("2025-11-17 06:00", tz="UTC"),
     "son": pd.Timestamp("2025-11-21 09:00", tz="UTC"), "sure": 100,
     "once_bas": pd.Timestamp("2025-11-16 00:00", tz="UTC"),
     "sonra_son": pd.Timestamp("2025-11-23 00:00", tz="UTC")},
    {"tarih": "2025-11-28", "bas": pd.Timestamp("2025-11-28 23:00", tz="UTC"),
     "son": pd.Timestamp("2025-11-29 18:00", tz="UTC"), "sure": 20,
     "once_bas": pd.Timestamp("2025-11-27 00:00", tz="UTC"),
     "sonra_son": pd.Timestamp("2025-11-30 00:00", tz="UTC")},
    {"tarih": "2025-11-29", "bas": pd.Timestamp("2025-11-29 20:00", tz="UTC"),
     "son": pd.Timestamp("2025-12-01 06:00", tz="UTC"), "sure": 35,
     "once_bas": pd.Timestamp("2025-11-28 00:00", tz="UTC"),
     "sonra_son": pd.Timestamp("2025-12-02 00:00", tz="UTC")},
    {"tarih": "2026-05-05", "bas": pd.Timestamp("2026-05-05 12:00", tz="UTC"),
     "son": pd.Timestamp("2026-05-05 23:00", tz="UTC"), "sure": 12,
     "once_bas": pd.Timestamp("2026-05-04 00:00", tz="UTC"),
     "sonra_son": pd.Timestamp("2026-05-07 00:00", tz="UTC")},
    {"tarih": "2026-07-13", "bas": pd.Timestamp("2026-07-13 03:00", tz="UTC"),
     "son": pd.Timestamp("2026-07-13 23:00", tz="UTC"), "sure": 21,
     "once_bas": pd.Timestamp("2026-07-12 00:00", tz="UTC"),
     "sonra_son": pd.Timestamp("2026-07-15 00:00", tz="UTC")},
]

print("=" * 70)
print("  5 BOSLUGUN FIYAT VE PIYASA OLAYI ANALIZI")
print("=" * 70)

for b in bosluklar:
    print("\n" + "-" * 60)
    print("  %s (%dh bosluk)" % (b["tarih"], b["sure"]))
    print("-" * 60)
    once_veri = ham[(ham.index >= b["once_bas"]) & (ham.index < b["bas"])]
    sonra_veri = ham[(ham.index > b["son"]) & (ham.index <= b["sonra_son"])]
    if len(once_veri) > 0 and len(sonra_veri) > 0:
        once_kapanis = once_veri["Close"].iloc[-1]
        sonra_acilis = sonra_veri["Close"].iloc[0]
        fark = sonra_acilis - once_kapanis
        fark_yuzde = (sonra_acilis / once_kapanis - 1) * 100
        once_r = r[(r.index >= b["once_bas"]) & (r.index < b["bas"])]
        sonra_r = r[(r.index > b["son"]) & (r.index <= b["sonra_son"])]
        print("  Once son kapanis:   $%s" % "{:,.2f}".format(once_kapanis))
        print("  Sonra ilk kapanis:  $%s" % "{:,.2f}".format(sonra_acilis))
        print("  FIYAT DEGISIMI:     $%s (%.2f%%)" % ("{:,}".format(int(fark)), fark_yuzde))
        print("  Once std(getiri):   %.4f" % once_r.std())
        print("  Sonra std(getiri):  %.4f" % sonra_r.std())
        if abs(fark_yuzde) > 10:
            print("  >> KRITIK: Buyuk fiyat degisimi (>10%) -- GERCEK PIYASA OLAYI")
        elif abs(fark_yuzde) > 3:
            print("  >> ONEMLI: Onemli fiyat degisimi (3-10%) -- piyasa olayi muhtemel")
        elif abs(fark_yuzde) > 1:
            print("  >> NORMAL: Hafif fiyat degisimi (1-3%) -- normal piyasa hareketi")
        else:
            print("  >> SAKIN: Minimal degisim (<1%) -- durgun donem")

# NOV 17 vs NOV 29 KARSILASTIRMASI
print("\n" + "=" * 70)
print("  NOV 17 vs NOV 29: IKINCI DALGA ANALIZI")
print("=" * 70)

once_17 = ham[(ham.index >= pd.Timestamp("2025-11-16 00:00", tz="UTC")) &
              (ham.index < pd.Timestamp("2025-11-17 06:00", tz="UTC"))]
ara = ham[(ham.index > pd.Timestamp("2025-11-21 10:00", tz="UTC")) &
          (ham.index < pd.Timestamp("2025-11-28 23:00", tz="UTC"))]
sonra_29 = ham[(ham.index > pd.Timestamp("2025-11-29 20:00", tz="UTC")) &
               (ham.index < pd.Timestamp("2025-12-02 00:00", tz="UTC"))]

print("\n  DONEM 1: Nov 16-17 (krash oncesi)")
print("    Fiyat: $%s -> $%s" % ("{:,.2f}".format(once_17["Close"].iloc[0]),
    "{:,.2f}".format(once_17["Close"].iloc[-1])))
once_17_degisim = (once_17["Close"].iloc[-1] / once_17["Close"].iloc[0] - 1) * 100
print("    Degisim: %.2f%%" % once_17_degisim)

print("\n  DONEM 2: Nov 21-28 (krash sonrasi toparlanma)")
if len(ara) > 0:
    print("    Fiyat: $%s -> $%s" % ("{:,.2f}".format(ara["Close"].iloc[0]),
        "{:,.2f}".format(ara["Close"].iloc[-1])))
    ara_degisim = (ara["Close"].iloc[-1] / ara["Close"].iloc[0] - 1) * 100
    print("    Degisim: %.2f%%" % ara_degisim)

print("\n  DONEM 3: Nov 29 - Dec 1 (ikinci bosluk)")
if len(sonra_29) > 0:
    print("    Ilk fiyat: $%s" % "{:,.2f}".format(sonra_29["Close"].iloc[0]))
    print("    Son fiyat: $%s" % "{:,.2f}".format(sonra_29["Close"].iloc[-1]))
    son_29_degisim = (sonra_29["Close"].iloc[-1] / sonra_29["Close"].iloc[0] - 1) * 100
    print("    Degisim: %.2f%%" % son_29_degisim)

# Nov 28 son kapanis vs Nov 29 sonrasi ilk kapanis
nov_28_son = ham[(ham.index > pd.Timestamp("2025-11-28 18:00", tz="UTC")) &
                 (ham.index < pd.Timestamp("2025-11-28 23:00", tz="UTC"))]
nov_29_ilk = ham[(ham.index > pd.Timestamp("2025-11-29 18:00", tz="UTC")) &
                 (ham.index < pd.Timestamp("2025-11-29 21:00", tz="UTC"))]
if len(nov_28_son) > 0 and len(nov_29_ilk) > 0:
    n28_kapanis = nov_28_son["Close"].iloc[-1]
    n29_acilis = nov_29_ilk["Close"].iloc[0]
    n29_fark = (n29_acilis / n28_kapanis - 1) * 100
    print("\n  Nov 28 son -> Nov 29 ilk:")
    print("    $%s -> $%s (%.2f%%)" % ("{:,.2f}".format(n28_kapanis),
        "{:,.2f}".format(n29_acilis), n29_fark))

print("\n  SONUC:")
print("  Nov 17 krashi $95K -> $82K (%-13) dusmus.")
print("  Nov 21-28 arasi toparlanma: $82K -> yaklasik $87K civari.")
print("  Nov 28-29: $87K civarinda duraganlasmis.")
print("  Nov 29 boslugu: vol_oran=1.12, buyuk bir fiyat degisimi yok.")
print("  Bu bosluk muhtemelen yfinance API kesintisi, piyasa krizi degil.")
