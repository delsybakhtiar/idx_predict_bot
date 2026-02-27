"""
================================================================================
QUANTITATIVE STOCK PREDICTION FRAMEWORK + TELEGRAM ALERT
Version: 2.1.0 (Cron-ready)
Focus: Auto-run Daily Analysis & Notification
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime
from collections import Counter

import requests
import numpy as np
import pandas as pd
import yfinance as yf

from dotenv import load_dotenv

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

import pandas_ta as ta


# Load environment variables
load_dotenv(".env")


def html_escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            print("⚠️ WARNING: Telegram credentials not found in .env file.")
            print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            return

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message: str) -> bool:
        if not self.enabled:
            return False

        url = f"{self.base_url}/sendMessage"
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": str(message),
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            print("✅ Telegram Message Sent Successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False


class StockPredictor:
    def __init__(self, ticker: str = "TLKM.JK", lookback_period: str = "3mo"):
        self.ticker = ticker
        self.lookback_period = lookback_period

        self.df_raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self._X_test = None

    def fetch_data(self):
        print(f"[INFO] Fetching data for {self.ticker} ({self.lookback_period})...")

        df = yf.download(
            tickers=self.ticker,
            period=self.lookback_period,
            progress=False,
        )

        if df is None or len(df) == 0:
            raise ValueError("Data kosong dari yfinance. Cek ticker/koneksi.")

        # Normalize MultiIndex columns: ('Close','TLKM.JK') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Kolom OHLCV tidak lengkap. Missing: {missing}. Kolom tersedia: {list(df.columns)}"
            )

        self.df_raw = df
        return self

    def generate_features(self):
        if self.df_raw is None or self.df_raw.empty:
            raise ValueError("Data belum di-fetch atau kosong.")

        df = self.df_raw.copy()

        df["RSI_14"] = ta.rsi(df["Close"], length=14)

        macd = ta.macd(df["Close"])
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)

        bb = ta.bbands(df["Close"], length=20)
        if bb is not None and not bb.empty:
            df = pd.concat([df, bb], axis=1)

        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        # Placeholder external features
        n = len(df)
        df["Foreign_Net_Buy"] = np.random.normal(0, 500_000_000, n)
        df["Sentiment_Score"] = np.random.uniform(-1, 1, n)
        df["PBV_Ratio"] = np.random.normal(3.5, 0.5, n)

        self.df_raw = df
        return self

    def prepare_data(self):
        if self.df_raw is None or self.df_raw.empty:
            raise ValueError("Data belum siap untuk diproses.")

        df = self.df_raw.copy()
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        feature_candidates = [
            "Open", "High", "Low", "Close", "Volume",
            "RSI_14", "MACD_12_26_9",
            "Foreign_Net_Buy", "Sentiment_Score", "PBV_Ratio",
            "ATR_14",
        ]
        available_features = [c for c in feature_candidates if c in df.columns]
        use_cols = available_features + ["Target"]

        self.df = df[use_cols].dropna()

        if len(self.df) < 25:
            raise ValueError(
                f"Data setelah feature engineering terlalu sedikit: {len(self.df)} baris. "
                f"Coba perpanjang lookback_period (mis. '6mo')."
            )

        X = self.df.drop("Target", axis=1)
        y = self.df["Target"].astype(int)

        split_idx = int(len(X) * 0.8)
        if split_idx < 10 or (len(X) - split_idx) < 3:
            raise ValueError(
                f"Data split tidak cukup (train/test). Total={len(X)}, split={split_idx}."
            )

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self._X_test = X_test_scaled
        return X_train_scaled, y_train.values, X_test_scaled, y_test.values, list(X.columns)

    def train(self, X_train, y_train):
        counts = Counter(y_train)
        if len(counts) < 2:
            raise ValueError(f"Kelas target tidak lengkap (hanya satu kelas): {counts}")

        min_class = min(counts.values())
        cv = 2 if min_class < 2 else min(5, min_class)
        if cv < 2:
            raise ValueError(f"Data training terlalu sedikit / kelas tidak seimbang: {counts}")

        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        model = xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )

        search = RandomizedSearchCV(
            model,
            param_dist,
            n_iter=8,
            scoring="accuracy",
            random_state=42,
            cv=cv,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return self

    def predict_latest(self):
        if self.model is None:
            raise ValueError("Model belum dilatih.")
        if self._X_test is None or len(self._X_test) == 0:
            raise ValueError("X_test tidak tersedia untuk prediksi.")

        proba_up = float(self.model.predict_proba(self._X_test[-1:])[0][1])
        pred_class = int(self.model.predict(self._X_test[-1:])[0])
        return pred_class, proba_up


def run_pipeline(ticker: str = "TLKM.JK") -> bool:
    bot = TelegramNotifier()
    predictor = StockPredictor(ticker=ticker, lookback_period="3mo")

    print("--- STARTING ANALYSIS ---")

    try:
        predictor.fetch_data().generate_features()
        X_train, y_train, X_test, y_test, features = predictor.prepare_data()
        predictor.train(X_train, y_train)

        pred_class, confidence = predictor.predict_latest()

        current_price = float(predictor.df_raw["Close"].iloc[-1])
        trend = "UP" if pred_class == 1 else "DOWN"
        action = "BUY" if pred_class == 1 else "HOLD/SELL"
        date_str = predictor.df_raw.index[-1].strftime("%d/%m/%Y")

        # Escape only user/data fields, not HTML tags
        alert_msg = (
            "🤖 <b>IDX SIGNAL REPORT</b>\n\n"
            f"📈 <b>Ticker</b>: {html_escape(ticker)}\n"
            f"📅 <b>Date</b>: {html_escape(date_str)}\n"
            f"💹 <b>Price</b>: {current_price:.2f}\n"
            f"🎯 <b>Direction</b>: {html_escape(trend)} ({confidence*100:.2f}%)\n"
            f"⚡ <b>Action</b>: {html_escape(action)}\n\n"
            "<i>Note: Model generated from multivariate analysis. Use responsibly.</i>"
        )

        bot.send_message(alert_msg)
        print(alert_msg)
        return True

    except Exception as e:
        error_msg = (
            "🔴 ERROR IN PREDICTION SCRIPT\n"
            f"Error: {str(e)}\n"
            "Check logs."
        )
        try:
            bot.send_message(error_msg)
        except Exception:
            pass

        print(f"Script Error: {e}")
        return False


if __name__ == "__main__":
    ok = run_pipeline("TLKM.JK")
    sys.exit(0 if ok else 1)
