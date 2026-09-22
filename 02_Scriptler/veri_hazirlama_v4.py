"""
Bitcoin Saatlik Veri Hazirlama (v4)
- Buyuk bosluklar (>3 saat): disari atildi
- Kucuk bosluklar (<=3 saat): reindex + lineer interpolasyon
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yfinance as yf
import os

HAM_YOL = "bitcoin_saatlik_ham.csv"
TEMIZ_YOL = "bitcoin_saatlik_temiz_v4.csv"
BUYUK_BOSLUK_ESIK_SAAT = 3

def veri_indir():
    print("yfinance ile indiriliyor...")
    data = yf.download("BTC-USD", interval="1h", period="730d")
    data.to_csv(HAM_YOL)
    print(f"Ham veri indirildi: {data.shape}")
    return data

def veri_temizle_v4():
    if not os.path.exists(HAM_YOL):
        veri_indir()

    df = pd.read_csv(HAM_YOL, index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"Ham veri: {df.shape[0]} satir")
    print(f"Tarih: {df.index.min()} -- {df.index.max()}")

    # Tarihleri timezone'dan arindir
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Bosluklari tespit et
    beklenen_freq = pd.Timedelta(hours=1)
    farklar = df.index.to_series().diff()
    buyuk_boslk = farklar[farklar > beklenen_freq * BUYUK_BOSLUK_ESIK_SAAT]

    print(f"\nBuyuk bosluk (> {BUYUK_BOSLUK_ESIK_SAAT} saat): {len(buyuk_boslk)} adet")
    for t, d in buyuk_boslk.items():
        print(f"  {t}: {d}")

    # Buyuk bosluklari belirle - sadece buyuk bosluklari kes
    cikarilacak = set()
    for t, d in buyuk_boslk.items():
        idx = df.index.get_loc(t)
        # Buyuk bosluktan onceki gozlemi cikar
        if idx > 0:
            cikarilacak.add(df.index[idx - 1])
        # Buyuk bosluktan sonraki gozlemi cikar
        cikarilacak.add(t)

    df = df.drop(cikarilacak)
    print(f"Buyuk bosluk sinirlarindan cikarilan: {len(cikarilacak)} gozlem")

    # Retsiz gozlemleri temizle
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Reindex (kucuk bosluklari doldur)
    tam_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1h")
    df = df.reindex(tam_index)

    bos_sayisi = df["Close"].isna().sum()
    print(f"Reindex sonrasi bos gozlem: {bos_sayisi}")

    if bos_sayisi > 0:
        df["Close"] = df["Close"].interpolate(method="linear")
        print("Lineer interpolasyon uygulandi")

    # Log-return hesapla
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["Log_Return"])

    print(f"\nTemiz veri: {df.shape[0]} satir x {df.shape[1]} sutun")
    print(f"Tarih: {df.index.min()} -- {df.index.max()}")
    print(f"Log-Return std: {df['Log_Return'].std():.6f}")

    df.to_csv(TEMIZ_YOL)
    print(f"\nKaydedildi: {TEMIZ_YOL}")
    return df

if __name__ == "__main__":
    veri_temizle_v4()
