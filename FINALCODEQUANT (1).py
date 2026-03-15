
# ============================================================
# MERGED: folder/index builder + Quadrant v5.3 strategy
# ============================================================

import pandas as pd
import numpy as np
import os
import shutil
import warnings
import time
from pathlib import Path
import pyarrow.parquet as pq
from collections import defaultdict
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import yfinance as yf

warnings.filterwarnings('ignore')
np.random.seed(42)
print('Libraries loaded')


# ============================================================
# PERFORMANCE REPORTING REQUIREMENTS (KRITI)
# Adds: CAGR, Ann Std (daily returns), Max DD, Info Ratio vs Nifty 500,
# Up/Down capture vs Nifty 500, and rolling outperformance stats (1/3/5Y).
# Rolling windows are overlapping and calculated daily (by trading-day count).
# ============================================================

def _perf_cagr(prices: pd.Series) -> float:
    prices = prices.dropna()
    if len(prices) < 2:
        return float('nan')
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years <= 0 or prices.iloc[0] <= 0:
        return float('nan')
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1.0 / years) - 1.0)


def _perf_max_drawdown(prices: pd.Series) -> float:
    prices = prices.dropna()
    if prices.empty:
        return float('nan')
    dd = prices / prices.cummax() - 1.0
    return float(dd.min())


def _perf_ann_std(daily_rets: pd.Series, trading_days: int = 252) -> float:
    daily_rets = daily_rets.dropna()
    if daily_rets.empty:
        return float('nan')
    return float(daily_rets.std(ddof=0) * np.sqrt(trading_days))


def _perf_information_ratio(port_rets: pd.Series, bench_rets: pd.Series, trading_days: int = 252) -> float:
    df = pd.DataFrame({'p': port_rets, 'b': bench_rets}).dropna()
    if df.empty:
        return float('nan')
    active = df['p'] - df['b']
    te = active.std(ddof=0)
    if te < 1e-12:
        return float('nan')
    return float(np.sqrt(trading_days) * active.mean() / te)


def _perf_capture_ratios(port_rets: pd.Series,
                        bench_rets: pd.Series,
                        method: str = "geomean",      # "geomean" (standard) or "total"
                        freq: str | None = "M",       # "M" monthly, None = use as-is
                        auto_percent_fix: bool = True,
                        debug: bool = False):

    df = pd.concat([port_rets.rename("p"), bench_rets.rename("b")], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan

    # Ensure datetime index for resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # OPTIONAL: convert daily -> monthly compounded returns
    if freq is not None:
        df = df.resample(freq).apply(lambda x: (1.0 + x).prod() - 1.0).dropna()
        if df.empty:
            return np.nan, np.nan

    # Optional: auto-fix percent vs decimal (do it per-column)
    if auto_percent_fix:
        for c in ["p", "b"]:
            q = df[c].abs().quantile(0.99)
            if q > 1.5:   # likely % units like 2 = 2%
                df[c] = df[c] / 100.0

    up = df["b"] > 0
    dn = df["b"] < 0

    def total_compound(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        if np.any(x <= -1.0):
            return np.nan
        return float(np.expm1(np.log1p(x).sum()))

    def geo_mean(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        if np.any(x <= -1.0):
            return np.nan
        return float(np.expm1(np.log1p(x).mean()))

    f = geo_mean if method == "geomean" else total_compound

    p_up = f(df.loc[up, "p"].to_numpy())
    b_up = f(df.loc[up, "b"].to_numpy())
    up_cap = np.nan if (not np.isfinite(b_up) or abs(b_up) < 1e-12) else (p_up / b_up) * 100.0

    p_dn = f(df.loc[dn, "p"].to_numpy())
    b_dn = f(df.loc[dn, "b"].to_numpy())
    dn_cap = np.nan if (not np.isfinite(b_dn) or abs(b_dn) < 1e-12) else (p_dn / b_dn) * 100.0

    if debug:
        print("[DEBUG capture]")
        print("n periods:", len(df), "up:", int(up.sum()), "down:", int(dn.sum()))
        print("p_up:", p_up, "b_up:", b_up, "up_cap:", up_cap)
        print("p_dn:", p_dn, "b_dn:", b_dn, "dn_cap:", dn_cap)

    return float(up_cap) if np.isfinite(up_cap) else np.nan, float(dn_cap) if np.isfinite(dn_cap) else np.nan

def _perf_download_benchmark(start_date, end_date, tickers=None) -> tuple[pd.Series, str]:
    """
    Robustly download a 1D daily close series from yfinance, even if yfinance returns:
    - Single-index columns (usual case)
    - MultiIndex columns (multiple tickers or different group_by defaults)
    """
    if tickers is None:
        # Most common for Nifty 500 on Yahoo is ^CRSLDX.
        tickers = ["^CRSLDX", "NIFTY500.NS", "^NIFTY500", "^NSE500"]

    last_err = None

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                actions=False,
                interval="1d",
            )
            if data is None or data.empty:
                continue

            close = None

            # Case 1: normal columns
            if "Close" in data.columns:
                close = data["Close"]

            # Case 2: MultiIndex columns
            if close is None and isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    close = data.xs("Close", axis=1, level=0)
                elif "Close" in data.columns.get_level_values(-1):
                    close = data.xs("Close", axis=1, level=-1)

            if close is None:
                continue

            # If yfinance returned a DataFrame, pick first column
            if isinstance(close, pd.DataFrame):
                if close.shape[1] == 0:
                    continue
                close = close.iloc[:, 0]

            # Force 1D Series
            close = pd.Series(close).dropna().copy()
            close.index = pd.to_datetime(close.index)
            close = close.sort_index()
            close.name = t

            if len(close) >= 2:
                return close, t

        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not download benchmark from yfinance. Tried: {tickers}. Last error: {last_err}"
    )


def _perf_rolling_outperformance(eq: pd.Series, bench: pd.Series, years_list=(1, 3, 5), trading_days: int = 252):
    df = pd.DataFrame({'eq': eq, 'b': bench}).dropna()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    stats_rows = []

    for y in years_list:
        w = int(trading_days * y)
        port_ann = (df['eq'] / df['eq'].shift(w)) ** (1.0 / y) - 1.0
        bench_ann = (df['b'] / df['b'].shift(w)) ** (1.0 / y) - 1.0
        excess = port_ann - bench_ann

        col = f'excess_ann_{y}y'
        out[col] = excess

        stats_rows.append({
            'window': f'{y} Year',
            'average_outperformance_ann': float(excess.mean(skipna=True)),
            'worst_underperformance_ann': float(excess.min(skipna=True)),
        })

    stats_df = pd.DataFrame(stats_rows)
    return out, stats_df


def _perf_period_report(eq: pd.Series, bench: pd.Series, label: str, trading_days: int = 252):
    df = pd.DataFrame({'eq': eq, 'b': bench}).dropna()
    if len(df) < 2:
        return None

    eq2 = df['eq']
    b2 = df['b']
    pr = eq2.pct_change().dropna()
    br = b2.pct_change().dropna()

    up_cap, dn_cap = _perf_capture_ratios(pr, br)

    return {
        'period': label,
        'start': df.index[0].date().isoformat(),
        'end': df.index[-1].date().isoformat(),
        'CAGR': _perf_cagr(eq2),
        'AnnStd_DailyRets': _perf_ann_std(pr, trading_days),
        'MaxDrawdown': _perf_max_drawdown(eq2),
        'InfoRatio_vs_Nifty500': _perf_information_ratio(pr, br, trading_days),
        'UpCapture_vs_Nifty500': up_cap,
        'DownCapture_vs_Nifty500': dn_cap,
    }


def _perf_run_reporting(eq: pd.Series,
                        output_prefix: str = "quadrant_v5",
                        trading_days: int = 252,
                        benchmark_tickers=None,
                        periods=None):
    # periods: list of (name, start, end)
    if periods is None:
        periods = [
            ("2010-2020", "2010-01-01", "2020-12-31"),
            ("2020-2025", "2020-01-01", "2025-12-31"),
        ]

    eq = eq.dropna().copy()
    # Ensure 1D Series
    if isinstance(eq, pd.DataFrame):
        eq = eq.iloc[:, 0]
    eq = pd.Series(eq).astype(float)
    eq.index = pd.to_datetime(eq.index)

    # Save strategy equity curve (uses existing OUTPUT_FILE if present)
    out_eq_csv = globals().get('OUTPUT_FILE', f"{output_prefix}_equity.csv")
    pd.DataFrame({'date': eq.index, 'equity': np.asarray(eq.values).reshape(-1)}).to_csv(out_eq_csv, index=False)

    # Download benchmark for full span of the strategy
    start_all = eq.index.min().date().isoformat()
    end_all = (eq.index.max() + pd.Timedelta(days=1)).date().isoformat()

    bench_all, used_ticker = _perf_download_benchmark(start_all, end_all, tickers=benchmark_tickers)
    # Ensure benchmark is 1D before saving / aligning
    if isinstance(bench_all, pd.DataFrame):
        bench_all = bench_all.iloc[:, 0]
    bench_all = pd.Series(bench_all).astype(float)
    pd.DataFrame({'date': bench_all.index, 'close': np.asarray(bench_all.values).reshape(-1)}).to_csv(
    f"{output_prefix}_nifty500_{used_ticker.replace('^','')}.csv",
    index=False
)

    # Align once for rolling calculations
    aligned = pd.DataFrame({'eq': eq, 'b': bench_all}).dropna()
    if aligned.empty:
        raise RuntimeError("No overlapping dates between strategy equity curve and benchmark.")

    # --- Period summary (core + capture) ---
    rows = []
    for name, s, e in periods:
        s = pd.to_datetime(s)
        e = pd.to_datetime(e)
        eq_p = aligned.loc[(aligned.index >= s) & (aligned.index <= e), 'eq']
        b_p = aligned.loc[(aligned.index >= s) & (aligned.index <= e), 'b']
        rep = _perf_period_report(eq_p, b_p, name, trading_days)
        if rep is not None:
            rows.append(rep)
        else:
            print(f"[PERF REPORT] Skipping period {name}: insufficient overlap.")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{output_prefix}_performance_summary.csv", index=False)

    # --- Rolling outperformance (full sample + per period stats) ---
    roll_series_all, roll_stats_all = _perf_rolling_outperformance(aligned['eq'], aligned['b'], years_list=(1, 3, 5), trading_days=trading_days)
    roll_series_all.reset_index(names='date').to_csv(f"{output_prefix}_rolling_outperformance_daily.csv", index=False)

    roll_stats_all['period'] = 'FULL_SAMPLE'
    roll_stats_rows = [roll_stats_all]

    for name, s, e in periods:
        s = pd.to_datetime(s)
        e = pd.to_datetime(e)
        sub = aligned.loc[(aligned.index >= s) & (aligned.index <= e)]
        rs, st = _perf_rolling_outperformance(sub['eq'], sub['b'], years_list=(1, 3, 5), trading_days=trading_days)
        if not st.empty:
            st['period'] = name
            roll_stats_rows.append(st)

    roll_stats_df = pd.concat(roll_stats_rows, ignore_index=True)
    roll_stats_df = roll_stats_df[['period', 'window', 'average_outperformance_ann', 'worst_underperformance_ann']]
    roll_stats_df.to_csv(f"{output_prefix}_rolling_outperformance_stats.csv", index=False)

    # Console print (keeps your existing prints intact)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + "="*60)
    print("PERFORMANCE REPORTING REQUIREMENTS (KRITI)")
    print("="*60)
    print(f"Benchmark ticker used: {used_ticker}")
    print("\nCore + Capture Metrics (by period):")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No period summary generated (no overlap).")

    print("\nRolling Outperformance Stats (annualized excess return):")
    print(roll_stats_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  - {out_eq_csv}")
    print(f"  - {output_prefix}_performance_summary.csv")
    print(f"  - {output_prefix}_rolling_outperformance_daily.csv")
    print(f"  - {output_prefix}_rolling_outperformance_stats.csv")


# ============================================================
# USER INPUT
# ============================================================
PARQUET_FILE = r'C:/Users/surface/OneDrive/Desktop/code/kriti 5.0/nse_prices_2021_2024_test.parquet'
OUTPUT_BASE = os.getcwd()   # Creates folders in current directory
BASE_VALUE = 100

GICS_MAP = {
    'Energy': 10,
    'Materials': 15,
    'Industrials': 20,
    'Consumer Discretionary': 25,
    'Consumer Staples': 30,
    'Health Care': 35,
    'Financials': 40,
    'Information Technology': 45,
    'Communication Services': 50,
    'Utilities': 55,
    'Real Estate': 60
}

# ============================================================
# STEP 1 — Split by FID
# ============================================================
def step1_split_by_fid(df, dataset_dir):
    print("\nSTEP 1: Splitting by FID")

    os.makedirs(dataset_dir, exist_ok=True)

    for fid, group in df.groupby('fid'):
        group = group.sort_values('tradedate').reset_index(drop=True)
        filepath = os.path.join(dataset_dir, f"{fid}.csv")
        group.to_csv(filepath, index=False)

    print(f"Saved {df['fid'].nunique()} stock files to '{dataset_dir}'")


# ============================================================
# STEP 2 — Organize by Sector
# ============================================================
def step2_organize_by_sector(dataset_dir, df):
    print("\nSTEP 2: Organizing into sector folders")

    fid_sector = df.drop_duplicates('fid')[['fid', 'gics_sector']]
    fid_sector = fid_sector.set_index('fid')['gics_sector'].to_dict()

    for fid, sector in fid_sector.items():

        if isinstance(sector, str) and sector in GICS_MAP:
            sector_code = str(GICS_MAP[sector])
        elif pd.notna(sector):
            try:
                sector_code = str(int(float(sector)))
            except:
                sector_code = 'unknown'
        else:
            sector_code = 'unknown'

        sector_dir = os.path.join(dataset_dir, sector_code)
        os.makedirs(sector_dir, exist_ok=True)

        src = os.path.join(dataset_dir, f"{fid}.csv")
        dst = os.path.join(sector_dir, f"{fid}.csv")

        if os.path.exists(src):
            shutil.move(src, dst)

    print("Stocks organized into sector folders.")


# ============================================================
# STEP 3 — MCAP Weighted Indices
# ============================================================
def step3_create_mcap_indices(dataset_dir, mcap_index_dir):
    print("\nSTEP 3: Creating MCAP Indices")

    os.makedirs(mcap_index_dir, exist_ok=True)

    sector_folders = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d != 'unknown'
    ]

    for sector in sorted(sector_folders):

        sector_path = os.path.join(dataset_dir, sector)
        csv_files = [
            os.path.join(sector_path, f)
            for f in os.listdir(sector_path)
            if f.endswith('.csv')
        ]

        if not csv_files:
            continue

        li = []

        for filename in csv_files:
            try:
                stock_df = pd.read_csv(filename)
                stock_df['tradedate'] = pd.to_datetime(stock_df['tradedate'])

                k = (stock_df['open'] + stock_df['high'] +
                     stock_df['low'] + stock_df['close']) / 4

                stock_df['shares'] = stock_df['mcap'] / k

                stock_df['w_open'] = stock_df['open'] * stock_df['shares']
                stock_df['w_high'] = stock_df['high'] * stock_df['shares']
                stock_df['w_low'] = stock_df['low'] * stock_df['shares']
                stock_df['w_close'] = stock_df['close'] * stock_df['shares']

                li.append(stock_df)

            except Exception as e:
                print(f"Skipping {filename}: {e}")

        if not li:
            continue

        full_df = pd.concat(li, ignore_index=True)

        index_df = full_df.groupby('tradedate').agg({
            'w_open': 'sum',
            'w_high': 'sum',
            'w_low': 'sum',
            'w_close': 'sum',
            'mcap': 'sum'
        }).reset_index().sort_values('tradedate')

        divisor = index_df.iloc[0]['mcap'] / BASE_VALUE

        index_df['open'] = index_df['w_open'] / divisor
        index_df['high'] = index_df['w_high'] / divisor
        index_df['low'] = index_df['w_low'] / divisor
        index_df['close'] = index_df['w_close'] / divisor
        index_df['previous'] = index_df['close'].shift(1)

        index_df['fid'] = f'MCAP_INDEX_{sector}'
        index_df['gics_sector'] = int(sector)

        index_df['tradedate'] = index_df['tradedate'].dt.strftime('%Y-%m-%d')

        out_file = os.path.join(mcap_index_dir, f"sector_index_{sector}.csv")
        index_df.to_csv(out_file, index=False)

        print(f"Sector {sector} MCAP index created.")


# ============================================================
# STEP 4 — Volatility Smart Beta
# ============================================================
def step4_create_volatility_indices(dataset_dir, smart_beta_dir):
    print("\nSTEP 4: Creating Volatility Smart Beta Indices")

    os.makedirs(smart_beta_dir, exist_ok=True)

    sector_folders = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d != 'unknown'
    ]

    for sector in sorted(sector_folders):

        sector_path = os.path.join(dataset_dir, sector)
        csv_files = [
            os.path.join(sector_path, f)
            for f in os.listdir(sector_path)
            if f.endswith('.csv')
        ]

        stock_data = {}
        volatilities = {}

        for file in csv_files:
            try:
                stock_id = os.path.basename(file).replace('.csv', '')
                df = pd.read_csv(file)
                df['tradedate'] = pd.to_datetime(df['tradedate'])

                df = df.sort_values('tradedate')
                df['returns'] = np.log(df['close'] / df['close'].shift(1))

                vol = df['returns'].std() * np.sqrt(252)

                if not np.isnan(vol) and vol > 0:
                    volatilities[stock_id] = vol
                    stock_data[stock_id] = df

            except:
                continue

        if not volatilities:
            continue

        static_weights = pd.Series({s: 1/v for s, v in volatilities.items()})

        closes_df = pd.concat(
            [df.set_index('tradedate')['close'].rename(s)
             for s, df in stock_data.items()],
            axis=1
        ).sort_index()

        rets_df = np.log(closes_df / closes_df.shift(1))

        daily_weights = closes_df.notna().multiply(static_weights, axis=1)
        daily_weights = daily_weights.div(daily_weights.sum(axis=1), axis=0)

        index_close = BASE_VALUE * np.exp(
            (rets_df.fillna(0) * daily_weights).sum(axis=1).cumsum()
        )

        index_df = pd.DataFrame({
            'tradedate': closes_df.index.strftime('%Y-%m-%d'),
            'close': index_close,
            'fid': f'VOL_INDEX_{sector}',
            'gics_sector': int(sector)
        })

        out_file = os.path.join(smart_beta_dir, f"volatility_{sector}.csv")
        index_df.to_csv(out_file, index=False)

        print(f"Sector {sector} Smart Beta index created.")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():

    print("Loading parquet file...")
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()

    dataset_dir = os.path.join(OUTPUT_BASE, 'dataset')
    mcap_index_dir = os.path.join(OUTPUT_BASE, 'mcap_index')
    smart_beta_dir = os.path.join(OUTPUT_BASE, 'smart_beta_index')

    # Remove old folders if they exist
    for d in [dataset_dir, mcap_index_dir, smart_beta_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    step1_split_by_fid(df, dataset_dir)
    step2_organize_by_sector(dataset_dir, df)
    step3_create_mcap_indices(dataset_dir, mcap_index_dir)
    step4_create_volatility_indices(dataset_dir, smart_beta_dir)

    print("\n✅ PROCESS COMPLETE")
    print("Created folders:")
    print(" - dataset/")
    print(" - mcap_index/")
    print(" - smart_beta_index/")



# ============================================================
# MERGE GLUE (added)
# ============================================================
# Save folder-builder main before the strategy script overwrites `main`
folder_main = main

# Prevent the strategy script's `if __name__ == "__main__": main()` from auto-running
__MERGED_ORIGINAL_NAME__ = __name__
__name__ = "quadrant_v5_3_strategy"

# =============================================================================
# CONFIGURATION v5.0 (Defensive Features + Sector ATR Stops)
# =============================================================================

DATASET_FOLDER = "dataset"
MCAP_FOLDER = "mcap_index"
SMARTBETA_FOLDER = "smart_beta_index"

OUTPUT_FILE = 'quadrant_v5_strategy_results.csv'
TRADE_LOG_FILE = 'quadrant_v5_trade_log.csv'

TARGET_QUADRANTS = ['HH', 'HL', 'LH']
MAX_TOTAL_STOCKS_NORMAL = 100
MAX_TOTAL_STOCKS_DEFENSIVE = 25
MIN_HOLDINGS_THRESHOLD = 20
REBALANCE_MONTHS = 3

S_RANGE = list(range(1, 11, 2)) + list(range(15, 101, 5))
L_RANGE = list(range(5, 51, 5)) + list(range(65, 991, 25))
BAND_PCT = 0.01
WARMUP_DAYS = 252

SECTOR_CAP_ENABLED = True
MAX_SECTOR_ALLOCATION = 0.20

MIN_EXIT_SIGNALS_REQUIRED = 2
# v5.2: EMA is one of the counted exit signals (no immediate exit)

FEATURE_LOOKBACK = 180
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

INITIAL_CAPITAL = 5000000
COMMISSION_PCT = 0.00268

MARKET_BREADTH_ENABLED = True
BREADTH_SMA_PERIOD = 200
STRONG_MARKET_THRESHOLD = 0.50

BREADTH_SMA_CROSSOVER_ENABLED = True
BREADTH_FAST_SMA = 30
BREADTH_SLOW_SMA = 150
REGIME_CROSSOVER_STD = 1.0

BREADTH_DIVERGENCE_ENABLED = False
DIVERGENCE_LOOKBACK = 20
TOP_PERCENTILE = 0.10

# =============================================================================
# ATR STOP LOSS — SECTOR-SPECIFIC MULTIPLIERS (v5.0)
# =============================================================================
# Rationale:
#   High-vol sectors (Energy, InfoTech, ConsumerDisc) need wider stops to
#   avoid getting shaken out by normal volatility.
#   Low-vol sectors (Utilities, ConsumerStaples, HealthCare) can use tighter
#   stops since their moves are more meaningful when they happen.
ATR_STOP_LOSS_ENABLED = True
ATR_PERIOD = 14
ATR_MULTIPLIER_DEFAULT = 3     # Fallback if sector not in map

SECTOR_ATR_MULTIPLIERS = {
    10: 2.7,    # Energy — high commodity vol, wide stops
    15: 3.0,    # Materials — cyclical, moderate-wide
    20: 2.2,    # Industrials — moderate
    25: 2.2,    # ConsumerDisc — high beta, wider
    30: 1.8,    # ConsumerStaples — low vol, tight stops
    35: 2.0,    # HealthCare — moderate-low
    40: 2.5,    # Financials — moderate
    45: 2.2,    # InfoTech — high vol, wide stops
    50: 2.3,    # CommServices — moderate
    55: 1.6,    # Utilities — very low vol, tight stops
    60: 2.8,    # RealEstate — moderate-low
}

# =============================================================================
# PORTFOLIO OPTIMIZATION — Multi-Objective Genetic Algorithm (v5.3)
# =============================================================================
GA_OPTIMIZER_ENABLED = True
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 40
GA_ELITE_PCT = 0.05
GA_MUTATION_STD = 0.05
GA_CROSSOVER_RATE = 0.7
KELLY_MULTIPLIER = 0.80         # Half-Kelly fix 0.80
HRP_MAX_CLUSTER_WEIGHT = 0.35
HRP_PENALTY_MULT = 2.0
MPT_SHARPE_FLOOR_PCT = 0.60
MPT_PENALTY_MULT = 1.0
RISK_FREE_RATE = 0.065 / 252   # ~6.5% annual (India T-bill), daily
OPT_LOOKBACK = 180
VOL_LOOKBACK = 180
MIN_VOL_FLOOR = 0.10
MAX_VOL_CAP = 0.80

# Exhaustion Exit
EXHAUSTION_ENABLED = False
EXHAUSTION_VOLUME_MULTIPLE = 2.5
EXHAUSTION_LOOKBACK = 20

# OBV Divergence Exit
OBV_DIVERGENCE_ENABLED = True
OBV_LOOKBACK = 120
OBV_PRICE_THRESHOLD = 0.05
OBV_DIVERGENCE_THRESHOLD = -0.02

# CMF Exit
CMF_EXIT_ENABLED = False
CMF_LOOKBACK = 20
CMF_EXIT_THRESHOLD = -0.20

# Liquidity Exit
LIQUIDITY_EXIT_ENABLED = True
SLIPPAGE_ADV_RATIO = 0.05
SLIPPAGE_LOOKBACK = 200
MIN_ADV_VALUE = 500000

EMERGENCY_MAX_PER_STOCK_PCT = 0.10

# v5.2: Emergency Rebalance Cooldown
EMERGENCY_COOLDOWN_DAYS = 15   # Min days between emergency rebals when < 20 holdings

# =============================================================================
# FEATURE CONFIGURATION (v5.0)
# =============================================================================
# NORMAL REGIME: per-quadrant features (unchanged from v4)
#   HH: inertia, comparative_rs, vpt_slope (momentum)
#   HL: gpr, max_dd_duration, downside_ratio (risk-adjusted recovery)
#   LH: nvi_slope, bb_bandwidth, obv_divergence (stealth accumulation)
#
# DEFENSIVE REGIME: different feature set emphasizing quality & safety
#   DEF: gpr, downside_ratio, comparative_rs
#   Rationale — in bear markets, you want stocks that:
#     - Have high gain-to-pain ratio (gpr): efficient returns per unit pain
#     - Have LOW downside deviation (downside_ratio): less left-tail risk
#     - Still have relative strength vs sector (comparative_rs): least weak


NORMAL_WEIGHTS = {
    'HH': {'inertia': 0.50, 'comparative_rs': 0.30, 'vpt_slope': 0.20},
    'HL': {'gpr': 0.20 , 'max_dd_duration': 0.40, 'downside_ratio': 0.40},
    'LH': {'nvi_slope': 0.40, 'bb_bandwidth': 0.20, 'obv_divergence': 0.40}
}

DEFENSIVE_WEIGHTS = {
    'DEF': {'inertia': 0.20, 'vpt_slope': 0.50, 'comparative_rs': 0.30}
}

INVERT_FEATURES = ['max_dd_duration', 'downside_ratio', 'bb_bandwidth']

SECTOR_NAMES = {
    10: 'Energy', 15: 'Materials', 20: 'Industrials',
    25: 'ConsumerDisc', 30: 'ConsumerStaples', 35: 'HealthCare',
    40: 'Financials', 45: 'InfoTech', 50: 'CommServices',
    55: 'Utilities', 60: 'RealEstate'
}

class WalkForwardEMABandCalculator:
    """
    Walk-forward EMA band signal calculator.
    
    v4.1 FIX: All signals are LAGGED by 1 day to prevent look-ahead bias.
    Signal at index i is based on data up to i-1 (previous close).
    Decision on day T uses signal computed from day T-1's close.
    """
    
    def __init__(self, df_close):
        self.close = df_close
        self.stocks = df_close.columns.tolist()
        self.rules = [(s, l) for s in S_RANGE for l in L_RANGE if s < l]
        print(f'  EMA Band Rules: {len(self.rules)} combinations')
        print(f'    S_RANGE: {S_RANGE[:5]}...{S_RANGE[-3:]} ({len(S_RANGE)} values)')
        print(f'    L_RANGE: {L_RANGE[:5]}...{L_RANGE[-3:]} ({len(L_RANGE)} values)')
        print(f'    BAND_PCT: {BAND_PCT:.1%}')
        self._precompute()
    
    def _compute_stock_signals(self, prices_series):
        """Compute walk-forward EMA band signals for a single stock."""
        prices = prices_series.values if hasattr(prices_series, 'values') else prices_series
        n = len(prices)
        
        if n < WARMUP_DAYS + 50:
            return np.zeros(n, dtype=np.int8)
        
        price_series = pd.Series(prices)
        num_rules = len(self.rules)
        signals = np.zeros((n, num_rules), dtype=np.int8)
        
        for i, (s, l) in enumerate(self.rules):
            ema_s = price_series.ewm(span=s, adjust=False).mean().values
            ema_l = price_series.ewm(span=l, adjust=False).mean().values
            
            sig = np.zeros(n, dtype=np.int8)
            with np.errstate(invalid='ignore'):
                sig[ema_s > ema_l * (1 + BAND_PCT)] = 1
                sig[ema_s < ema_l * (1 - BAND_PCT)] = -1
            
            # Signal persistence
            mask = sig != 0
            idx = np.where(mask, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            signals[:, i] = sig[idx]
        
        # Market returns
        mkt_ret = np.zeros(n)
        mkt_ret[1:] = np.diff(prices) / prices[:-1]
        
        # Walk-forward rule selection
        sig_active = np.vstack([np.zeros((1, num_rules)), signals[:-1]])
        rule_rets = mkt_ret[:, None] * (sig_active == 1).astype(float)
        cum_rets = np.cumsum(rule_rets, axis=0)
        best_idx = np.argmax(cum_rets, axis=1)
        
        sel_idx = np.roll(best_idx, 1)
        sel_idx[0] = 0
        sel_idx[:WARMUP_DAYS] = -1
        
        # Raw signals (using today's data)
        raw_signals = np.zeros(n, dtype=np.int8)
        for t in range(n):
            ridx = sel_idx[t]
            if ridx == -1:
                raw_signals[t] = 0
            else:
                raw_signals[t] = signals[t, ridx]
        
        # v4.1 LOOK-AHEAD FIX: Lag all signals by 1 day
        # Signal available on day T is computed from day T-1's close
        lagged_signals = np.zeros(n, dtype=np.int8)
        lagged_signals[1:] = raw_signals[:-1]
        
        return lagged_signals
    
    def _precompute(self):
        """Pre-compute signals for all stocks."""
        print('  Computing walk-forward EMA band signals (lagged T-1)...')
        start = time.time()
        
        self.signals = pd.DataFrame(index=self.close.index, columns=self.stocks, dtype=np.int8)
        self.signals[:] = 0
        
        total = len(self.stocks)
        for i, stock in enumerate(self.stocks):
            if (i + 1) % 50 == 0 or i == total - 1:
                print(f'    [{i+1}/{total}] stocks...', end='\r')
            
            prices = self.close[stock].dropna()
            if len(prices) < WARMUP_DAYS + 50:
                continue
            
            stock_sigs = self._compute_stock_signals(prices)
            aligned = pd.Series(stock_sigs, index=prices.index, dtype=np.int8)
            self.signals[stock] = aligned.reindex(self.close.index).fillna(0).astype(np.int8)
        
        print(f'\n  EMA signals computed in {time.time() - start:.1f}s (lagged by 1 day)')
    
    def get_signal(self, stock, idx):
        try:
            if stock not in self.signals.columns:
                return 0
            sig = self.signals[stock].iloc[idx]
            return int(sig) if not np.isnan(sig) else 0
        except:
            return 0
    
    def check_entry_condition(self, stock, idx):
        """Entry: signal = +1 (already lagged — safe to use at idx)"""
        sig = self.get_signal(stock, idx)
        return sig == 1, sig
    
    def check_exit_condition(self, stock, idx):
        """Exit: signal = -1 (already lagged — safe to use at idx)"""
        sig = self.get_signal(stock, idx)
        return sig == -1, sig

class AdvancedBreadthCalculator:
    """
    Market Breadth Calculator v4.1:
    - Breadth on HH+HL+LH pool only
    - SMA Crossover with 1 std dev threshold for regime detection
    - All indicators LAGGED by 1 day (look-ahead fix)
    """
    
    def __init__(self, df_close, pool_stocks=None):
        self.close = df_close
        self.pool_stocks = pool_stocks
        self._precompute()
    
    def _precompute(self):
        if self.pool_stocks:
            cols = [s for s in self.pool_stocks if s in self.close.columns]
            print(f'  Computing Market Breadth on HH+HL+LH pool ({len(cols)} stocks)...')
            pool_close = self.close[cols]
        else:
            print(f'  Computing Market Breadth on ALL stocks...')
            pool_close = self.close
        
        self.sma = pool_close.rolling(BREADTH_SMA_PERIOD, min_periods=30).mean()
        above_sma_raw = pool_close > self.sma
        valid_counts = above_sma_raw.notna().sum(axis=1)
        above_counts = above_sma_raw.sum(axis=1)
        breadth_raw = above_counts / (valid_counts + 1e-10)
        
        # v4.1 LOOK-AHEAD FIX: Lag breadth by 1 day
        # On day T, we only know breadth from day T-1's close
        self.breadth_series = breadth_raw.shift(1)
        
        print(f'  Computing Breadth SMA Crossover ({BREADTH_FAST_SMA}/{BREADTH_SLOW_SMA}) with {REGIME_CROSSOVER_STD} std dev (lagged)...')
        self.breadth_fast_sma = self.breadth_series.rolling(BREADTH_FAST_SMA, min_periods=5).mean()
        self.breadth_slow_sma = self.breadth_series.rolling(BREADTH_SLOW_SMA, min_periods=15).mean()
        self.breadth_std = self.breadth_series.rolling(BREADTH_SLOW_SMA, min_periods=15).std()
        self.sigma_threshold = self.breadth_std * REGIME_CROSSOVER_STD
        
        # Returns also lagged for divergence
        self.returns = pool_close.pct_change()
        print('  Breadth indicators ready (lagged T-1)')
    
    def get_breadth(self, idx):
        try:
            val = self.breadth_series.iloc[idx]
            if np.isnan(val):
                return 0.5, True
            return float(val), val > STRONG_MARKET_THRESHOLD
        except:
            return 0.5, True
    
    def check_regime_signal(self, idx):
        """
        DEFENSIVE = fast < slow - 1 std dev
        NORMAL    = fast > slow + 1 std dev
        HOLD      = between bands
        All based on lagged breadth, safe to use at idx.
        """
        if not BREADTH_SMA_CROSSOVER_ENABLED:
            return 'NORMAL', 0, 0, 0, 0
        try:
            fast = self.breadth_fast_sma.iloc[idx]
            slow = self.breadth_slow_sma.iloc[idx]
            sigma = self.sigma_threshold.iloc[idx]
            if np.isnan(fast) or np.isnan(slow) or np.isnan(sigma):
                return 'NORMAL', 0, 0, 0, 0
            gap = fast - slow
            if fast < (slow - sigma):
                return 'DEFENSIVE', float(gap), float(fast), float(slow), float(sigma)
            elif fast > (slow + sigma):
                return 'NORMAL', float(gap), float(fast), float(slow), float(sigma)
            else:
                return 'HOLD', float(gap), float(fast), float(slow), float(sigma)
        except:
            return 'NORMAL', 0, 0, 0, 0
    
    def check_breadth_divergence(self, idx, holdings_returns):
        if not BREADTH_DIVERGENCE_ENABLED or idx < DIVERGENCE_LOOKBACK + 10:
            return False, 0, 0, 0
        try:
            breadth_now = self.breadth_series.iloc[idx]
            breadth_prev = self.breadth_series.iloc[idx - DIVERGENCE_LOOKBACK]
            if np.isnan(breadth_now) or np.isnan(breadth_prev):
                return False, 0, 0, 0
            breadth_change = breadth_now - breadth_prev
            # Use returns up to idx-1 (not today) for divergence
            if not holdings_returns:
                period_returns = self.returns.iloc[idx - DIVERGENCE_LOOKBACK - 1:idx - 1].sum()
                valid_returns = period_returns.dropna().sort_values(ascending=False)
                if len(valid_returns) == 0:
                    return False, 0, 0, 0
                top_n = max(1, int(len(valid_returns) * TOP_PERCENTILE))
                top_return = valid_returns.iloc[:top_n].mean()
            else:
                sorted_h = sorted(holdings_returns.items(), key=lambda x: x[1], reverse=True)
                top_n = max(1, int(len(sorted_h) * TOP_PERCENTILE))
                top_return = np.mean([r for _, r in sorted_h[:top_n]])
            has_divergence = (top_return > 0.02) and (breadth_change < -0.05)
            divergence_score = (top_return - breadth_change) if breadth_change < 0 else 0
            return has_divergence, float(top_return), float(breadth_change), float(divergence_score)
        except:
            return False, 0, 0, 0
    
    def get_full_status(self, idx, holdings_returns=None):
        breadth_pct, is_strong = self.get_breadth(idx)
        regime_signal, gap, fast_sma, slow_sma, sigma = self.check_regime_signal(idx)
        has_div, top_ret, breadth_chg, div_score = self.check_breadth_divergence(idx, holdings_returns or {})
        return {
            'breadth_pct': breadth_pct, 'is_strong': is_strong,
            'regime_signal': regime_signal, 'breadth_fast_sma': fast_sma,
            'breadth_slow_sma': slow_sma, 'sigma_threshold': sigma, 'sma_gap': gap,
            'has_divergence': has_div, 'top_10pct_return': top_ret,
            'breadth_change': breadth_chg, 'divergence_score': div_score
        }

class VolatilityCalculator:
    """
    v5.0: Sector-specific ATR stop loss multipliers.
    High-vol sectors get wider stops; low-vol sectors get tighter stops.
    """
    
    def __init__(self, df_close, df_high, df_low, stock_sectors):
        self.close = df_close
        self.high = df_high
        self.low = df_low
        self.stock_sectors = stock_sectors
        self.returns = df_close.pct_change()
        self._precompute()
    
    def _precompute(self):
        print('  Computing ATR and volatility...')
        tr = pd.DataFrame(index=self.close.index)
        for col in self.close.columns:
            tr1 = self.high[col] - self.low[col]
            tr2 = (self.high[col] - self.close[col].shift(1)).abs()
            tr3 = (self.low[col] - self.close[col].shift(1)).abs()
            tr[col] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = tr.rolling(ATR_PERIOD, min_periods=5).mean()
        self.volatility = self.returns.rolling(VOL_LOOKBACK, min_periods=20).std() * np.sqrt(252)
        
        # Log sector multiplier map
        print('  Sector ATR Multipliers:')
        for sec_code, mult in sorted(SECTOR_ATR_MULTIPLIERS.items()):
            print(f'    {SECTOR_NAMES.get(sec_code, sec_code):18s}: {mult:.1f}x ATR')
        print(f'    {"Default":18s}: {ATR_MULTIPLIER_DEFAULT:.1f}x ATR')
        print('  ATR and volatility ready')
    
    def _get_sector_multiplier(self, stock):
        """Get the ATR multiplier for a stock's sector."""
        sec = self.stock_sectors.get(stock, 0)
        return SECTOR_ATR_MULTIPLIERS.get(sec, ATR_MULTIPLIER_DEFAULT)
    
    def get_atr_stop_price(self, stock, idx, entry_price):
        """Stop = entry - (ATR * sector_multiplier). Sector-specific stops."""
        try:
            atr = self.atr[stock].iloc[idx]
            if np.isnan(atr) or entry_price <= 0:
                return entry_price * 0.90
            multiplier = self._get_sector_multiplier(stock)
            stop = entry_price - (atr * multiplier)
            # Floor at 80% of entry, ceiling at 95%
            return max(entry_price * 0.80, min(entry_price * 0.95, stop))
        except:
            return entry_price * 0.90
    
    def check_atr_stop_loss(self, stock, idx, entry_price, stop_price):
        try:
            curr = self.close[stock].iloc[idx]
            if np.isnan(curr):
                return False, 0, stop_price
            return curr < stop_price, float(curr), stop_price
        except:
            return False, 0, stop_price
    
    def get_volatility(self, stock, idx):
        try:
            vol = self.volatility[stock].iloc[idx]
            if np.isnan(vol):
                return 0.30
            return max(MIN_VOL_FLOOR, min(MAX_VOL_CAP, float(vol)))
        except:
            return 0.30
    
    def get_inverse_vol_weights(self, stocks, idx):
        # Legacy fallback — equal weight (GA optimizer handles allocation now)
        if not stocks:
            return {}
        return {s: 1.0 / len(stocks) for s in stocks}

class QuadrantFeatureCalculator:
    """
    v5.0: Added defensive feature set (gpr, downside_ratio, comparative_rs).
    Normal features unchanged. All use data up to idx-1 (no look-ahead).
    """
    def __init__(self, df_close, df_high, df_low, df_volume, stock_sectors):
        self.close = df_close
        self.high = df_high
        self.low = df_low
        self.volume = df_volume
        self.stock_sectors = stock_sectors
        self.returns = df_close.pct_change()
        self._precompute()
    
    def _precompute(self):
        print('  Pre-computing quadrant indicators...')
        pc = self.close.diff()
        self.obv = (self.volume * np.sign(pc).fillna(0)).cumsum()
        self.vpt = (self.volume * self.close.pct_change().fillna(0)).cumsum()
        vol_chg = self.volume.diff()
        self.nvi = (1 + self.returns.where(vol_chg < 0, 0).fillna(0)).cumprod() * 100
        bb_mid = self.close.rolling(BOLLINGER_PERIOD).mean()
        bb_std = self.close.rolling(BOLLINGER_PERIOD).std()
        self.bb_bandwidth = (2 * BOLLINGER_STD * bb_std) / (bb_mid + 1e-10)
        print('  Quadrant indicators ready')
    
    def _linreg(self, y):
        y = y.dropna()
        if len(y) < 10:
            return 0, 0
        try:
            x = np.arange(len(y))
            slope, _, r, _, _ = stats.linregress(x, y.values)
            return slope, r ** 2
        except:
            return 0, 0
    
    def calc_hh_features(self, idx, stocks, sec_map):
        stocks = [s for s in stocks if s in self.close.columns]
        if not stocks or idx < FEATURE_LOOKBACK + 1:
            return pd.DataFrame()
        end = idx
        start = end - FEATURE_LOOKBACK
        results = []
        sec_med = {}
        for sec, ss in sec_map.items():
            rets = []
            for s in [x for x in ss if x in self.close.columns]:
                try:
                    p0, p1 = self.close[s].iloc[start], self.close[s].iloc[end - 1]
                    if p0 > 0:
                        rets.append((p1 / p0) - 1)
                except:
                    pass
            sec_med[sec] = np.median(rets) if rets else 0
        for stock in stocks:
            try:
                cl = self.close[stock].iloc[start:end]
                vpt = self.vpt[stock].iloc[start:end]
                slope, r2 = self._linreg(cl)
                inertia = (slope / (cl.mean() + 1e-10) * 100) * r2
                ret = (cl.iloc[-1] / cl.iloc[0] - 1) if cl.iloc[0] > 0 else 0
                comp_rs = ret - sec_med.get(self.stock_sectors.get(stock, 0), 0)
                vslope, _ = self._linreg(vpt)
                vpt_norm = vslope / (self.volume[stock].iloc[start:end].mean() + 1e-10) * 1000
                results.append({'stock': stock, 'inertia': inertia, 'comparative_rs': comp_rs, 'vpt_slope': vpt_norm})
            except:
                pass
        return pd.DataFrame(results).set_index('stock') if results else pd.DataFrame()
    
    def calc_hl_features(self, idx, stocks):
        stocks = [s for s in stocks if s in self.close.columns]
        if not stocks or idx < FEATURE_LOOKBACK + 1:
            return pd.DataFrame()
        end = idx
        start = end - FEATURE_LOOKBACK
        results = []
        for stock in stocks:
            try:
                ret = self.returns[stock].iloc[start:end].dropna()
                cl = self.close[stock].iloc[start:end]
                if len(ret) < 20:
                    continue
                gpr = ret[ret > 0].sum() / (abs(ret[ret < 0].sum()) + 1e-10)
                uw = cl < cl.cummax()
                dd_dur = (uw.astype(int).groupby((~uw).cumsum()).cumsum()).max()
                ds = ret[ret < 0].std() if len(ret[ret < 0]) > 0 else 0
                dr = ds / (ret.std() + 1e-10)
                results.append({'stock': stock, 'gpr': gpr, 'max_dd_duration': dd_dur, 'downside_ratio': dr})
            except:
                pass
        return pd.DataFrame(results).set_index('stock') if results else pd.DataFrame()
    
    def calc_lh_features(self, idx, stocks):
        stocks = [s for s in stocks if s in self.close.columns]
        if not stocks or idx < FEATURE_LOOKBACK + 1:
            return pd.DataFrame()
        end = idx
        start = end - FEATURE_LOOKBACK
        results = []
        for stock in stocks:
            try:
                nvi = self.nvi[stock].iloc[start:end]
                cl = self.close[stock].iloc[start:end]
                obv = self.obv[stock].iloc[start:end]
                bw = self.bb_bandwidth[stock].iloc[max(start, end-20):end]
                ns, _ = self._linreg(nvi)
                os_val, _ = self._linreg(obv)
                ps, _ = self._linreg(cl)
                div = os_val / (obv.abs().mean() + 1e-10) - ps / (cl.mean() + 1e-10)
                results.append({'stock': stock, 'nvi_slope': ns / (nvi.mean() + 1e-10) * 100, 'bb_bandwidth': bw.mean(), 'obv_divergence': div})
            except:
                pass
        return pd.DataFrame(results).set_index('stock') if results else pd.DataFrame()
    
    def calc_defensive_features(self, idx, stocks, sec_map):
        """
        DEFENSIVE features: gpr, downside_ratio, comparative_rs
        Computed for ALL stocks regardless of quadrant.
        Combines HL risk features with HH relative strength.
        """
        stocks = [s for s in stocks if s in self.close.columns]
        if not stocks or idx < FEATURE_LOOKBACK + 1:
            return pd.DataFrame()
        end = idx
        start = end - FEATURE_LOOKBACK
        
        # Sector median returns for comparative_rs
        sec_med = {}
        for sec, ss in sec_map.items():
            rets = []
            for s in [x for x in ss if x in self.close.columns]:
                try:
                    p0, p1 = self.close[s].iloc[start], self.close[s].iloc[end - 1]
                    if p0 > 0:
                        rets.append((p1 / p0) - 1)
                except:
                    pass
            sec_med[sec] = np.median(rets) if rets else 0
        
        results = []
        for stock in stocks:
            try:
                ret = self.returns[stock].iloc[start:end].dropna()
                cl = self.close[stock].iloc[start:end]
                if len(ret) < 20:
                    continue
                
                # GPR: gain-to-pain ratio
                gpr = ret[ret > 0].sum() / (abs(ret[ret < 0].sum()) + 1e-10)
                
                # Downside ratio
                ds = ret[ret < 0].std() if len(ret[ret < 0]) > 0 else 0
                dr = ds / (ret.std() + 1e-10)
                
                # Comparative RS vs sector
                stock_ret = (cl.iloc[-1] / cl.iloc[0] - 1) if cl.iloc[0] > 0 else 0
                comp_rs = stock_ret - sec_med.get(self.stock_sectors.get(stock, 0), 0)
                
                results.append({
                    'stock': stock,
                    'gpr': gpr,
                    'downside_ratio': dr,
                    'comparative_rs': comp_rs
                })
            except:
                pass
        return pd.DataFrame(results).set_index('stock') if results else pd.DataFrame()
    
    def get_features(self, idx, quadrant, stocks, sec_map=None):
        if quadrant == 'HH':
            return self.calc_hh_features(idx, stocks, sec_map or {})
        elif quadrant == 'HL':
            return self.calc_hl_features(idx, stocks)
        elif quadrant == 'LH':
            return self.calc_lh_features(idx, stocks)
        elif quadrant == 'DEF':
            return self.calc_defensive_features(idx, stocks, sec_map or {})
        return pd.DataFrame()

class QuadrantRanker:
    """
    v5.0: Separate feature sets for normal vs defensive regimes.
    Normal: HH/HL/LH features with fixed weights per quadrant.
    Defensive: DEF features (gpr, downside_ratio, comparative_rs) for ALL stocks.
    """
    
    def __init__(self, feature_calc, ema_calc, stock_sectors):
        self.fc = feature_calc
        self.ema = ema_calc
        self.ss = stock_sectors
    
    def _norm(self, s):
        r = s.max() - s.min()
        return pd.Series(50, index=s.index) if r < 1e-10 else ((s - s.min()) / r) * 100
    
    def rank_quadrant(self, idx, q, stocks, sec_map=None, weights=None):
        f = self.fc.get_features(idx, q, stocks, sec_map)
        if f.empty:
            return []
        w = weights or NORMAL_WEIGHTS.get(q, {})
        n = pd.DataFrame(index=f.index)
        for c in f.columns:
            v = self._norm(f[c])
            n[c] = 100 - v if c in INVERT_FEATURES else v
        sc = sum(n[c] * wt for c, wt in w.items() if c in n.columns)
        if isinstance(sc, (int, float)):
            return []
        return [(s, float(v)) for s, v in sc.sort_values(ascending=False).items()]
    
    def select_top_stocks_normal(self, idx, stock_sectors, quadrant_sectors, n_stocks=100):
        """NORMAL REGIME: HH/HL/LH features with per-quadrant weights."""
        all_ranked = []
        qcounts = {'HH': 0, 'HL': 0, 'LH': 0}
        sec_map = defaultdict(list)
        for s, sec in stock_sectors.items():
            sec_map[sec].append(s)
        for q in TARGET_QUADRANTS:
            eligible = [s for s, sec in stock_sectors.items() if sec in quadrant_sectors.get(q, [])]
            if eligible:
                ranked = self.rank_quadrant(idx, q, eligible, sec_map, weights=NORMAL_WEIGHTS.get(q))
                qcounts[q] = len(ranked)
                all_ranked.extend([(s, sc, q) for s, sc in ranked])
        all_ranked.sort(key=lambda x: x[1], reverse=True)
        max_per_sec = int(n_stocks * MAX_SECTOR_ALLOCATION) if SECTOR_CAP_ENABLED else n_stocks
        sec_count = defaultdict(int)
        selected = []
        ema_rej = 0
        sec_cap = 0
        for stock, score, q in all_ranked:
            is_long, sig = self.ema.check_entry_condition(stock, idx)
            if not is_long:
                ema_rej += 1
                continue
            sec = self.ss.get(stock, 0)
            if SECTOR_CAP_ENABLED and sec_count[sec] >= max_per_sec:
                sec_cap += 1
                continue
            selected.append((stock, score, q))
            sec_count[sec] += 1
            if len(selected) >= n_stocks:
                break
        sector_counts_str = ', '.join([f"{SECTOR_NAMES.get(s, s)}:{c}" for s, c in sorted(sec_count.items(), key=lambda x: -x[1]) if c > 0])
        return {
            'stocks': selected,
            'by_quadrant': {'HH': [(s, sc) for s, sc, q in selected if q == 'HH'],
                           'HL': [(s, sc) for s, sc, q in selected if q == 'HL'],
                           'LH': [(s, sc) for s, sc, q in selected if q == 'LH']},
            'eligible_counts': qcounts, 'ema_rejected': ema_rej,
            'sector_capped': sec_cap, 'sector_counts': dict(sec_count),
            'sector_counts_str': sector_counts_str, 'regime': 'NORMAL'
        }
    
    def select_top_stocks_defensive(self, idx, stock_sectors, n_stocks=30):
        """
        DEFENSIVE REGIME: DEF features (gpr, downside_ratio, comparative_rs)
        ranked across ALL stocks regardless of quadrant. Pick top 30.
        """
        all_stocks = list(stock_sectors.keys())
        sec_map = defaultdict(list)
        for s, sec in stock_sectors.items():
            sec_map[sec].append(s)
        
        # Use 'DEF' quadrant -> calc_defensive_features with DEFENSIVE_WEIGHTS
        ranked = self.rank_quadrant(idx, 'DEF', all_stocks, sec_map, weights=DEFENSIVE_WEIGHTS.get('DEF'))
        
        max_per_sec = int(n_stocks * MAX_SECTOR_ALLOCATION) if SECTOR_CAP_ENABLED else n_stocks
        sec_count = defaultdict(int)
        selected = []
        ema_rej = 0
        sec_cap = 0
        for stock, score in ranked:
            is_long, sig = self.ema.check_entry_condition(stock, idx)
            if not is_long:
                ema_rej += 1
                continue
            sec = self.ss.get(stock, 0)
            if SECTOR_CAP_ENABLED and sec_count[sec] >= max_per_sec:
                sec_cap += 1
                continue
            selected.append((stock, score, 'DEF'))
            sec_count[sec] += 1
            if len(selected) >= n_stocks:
                break
        sector_counts_str = ', '.join([f"{SECTOR_NAMES.get(s, s)}:{c}" for s, c in sorted(sec_count.items(), key=lambda x: -x[1]) if c > 0])
        return {
            'stocks': selected,
            'by_quadrant': {'DEF': [(s, sc) for s, sc, q in selected]},
            'eligible_counts': {'ALL': len(ranked)}, 'ema_rejected': ema_rej,
            'sector_capped': sec_cap, 'sector_counts': dict(sec_count),
            'sector_counts_str': sector_counts_str, 'regime': 'DEFENSIVE'
        }

class GeneticPortfolioOptimizer:
    """
    Multi-Objective Genetic Algorithm for Portfolio Optimization.
    
    Pipeline:
      1. HRP clustering -> detect correlated groups
      2. MPT tangency portfolio -> Sharpe benchmark
      3. Kelly criterion fitness -> log-growth maximization
      4. Genetic algorithm -> population x generations search
      5. Half-Kelly sizing -> cash allocation
    """
    
    def __init__(self, df_close, stock_sectors):
        self.close = df_close
        self.returns = df_close.pct_change()
        self.stock_sectors = stock_sectors
        print('  GA Portfolio Optimizer initialized')
        print(f'    Population: {GA_POPULATION_SIZE} | Generations: {GA_GENERATIONS}')
        print(f'    Kelly Mult: {KELLY_MULTIPLIER} | HRP Cluster Cap: {HRP_MAX_CLUSTER_WEIGHT:.0%}')
        print(f'    MPT Sharpe Floor: {MPT_SHARPE_FLOOR_PCT:.0%} of tangency')
    
    def _get_returns_matrix(self, stocks, idx):
        """Get return matrix for stocks using lookback window ending at idx-1."""
        start = max(0, idx - OPT_LOOKBACK)
        end = idx
        cols = [s for s in stocks if s in self.returns.columns]
        if not cols or end - start < 30:
            return None, cols
        ret_matrix = self.returns[cols].iloc[start:end].dropna(axis=1, how='all').fillna(0)
        valid = ret_matrix.std() > 1e-10
        ret_matrix = ret_matrix.loc[:, valid]
        return ret_matrix, ret_matrix.columns.tolist()
    
    def _compute_hrp_clusters(self, ret_matrix):
        """
        HRP clustering: Distance = sqrt((1-corr)/2), Ward linkage.
        Returns cluster labels (0-indexed) for each asset.
        """
        n = ret_matrix.shape[1]
        if n <= 2:
            return np.arange(n)
        
        corr = ret_matrix.corr().values
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1, 1)
        
        dist = np.sqrt((1 - corr) / 2.0)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        dist = np.maximum(dist, 0)
        
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method='ward')
            n_clusters = max(2, min(10, int(np.sqrt(n))))
            labels = fcluster(link, t=n_clusters, criterion='maxclust')
            return labels - 1
        except Exception:
            stocks = ret_matrix.columns.tolist()
            sectors = [self.stock_sectors.get(s, 0) for s in stocks]
            unique_secs = list(set(sectors))
            return np.array([unique_secs.index(s) for s in sectors])
    
    def _compute_mpt_benchmark_sharpe(self, mean_rets, cov_matrix):
        """Tangency portfolio Sharpe via closed-form: w* = inv(Sigma) @ (mu - rf)."""
        try:
            n = len(mean_rets)
            excess = mean_rets - RISK_FREE_RATE
            cov_reg = cov_matrix + np.eye(n) * 1e-8
            cov_inv = np.linalg.inv(cov_reg)
            raw_w = cov_inv @ excess
            denom = np.sum(raw_w)
            if abs(denom) < 1e-10:
                return 0.0
            w_tan = raw_w / denom
            port_ret = w_tan @ mean_rets
            port_var = w_tan @ cov_matrix @ w_tan
            sharpe = (port_ret - RISK_FREE_RATE) / np.sqrt(max(port_var, 1e-10)) * np.sqrt(252)
            return max(sharpe, 0)
        except Exception:
            return 0.0
    
    def _fitness(self, weights, ret_matrix, mean_rets, cov_matrix, cluster_labels, mpt_sharpe):
        """
        Multi-objective fitness:
          A. Kelly Growth: E[ln(1 + w'r)] annualized
          B. HRP Penalty: cluster > 35% -> penalize
          C. MPT Penalty: Sharpe < 60% of tangency -> penalize
        """
        w = np.maximum(weights, 0)
        w_sum = w.sum()
        if w_sum < 1e-10:
            return -999.0
        w = w / w_sum
        
        # A. Kelly Growth
        port_rets = ret_matrix.values @ w
        log_growth = np.mean(np.log1p(np.clip(port_rets, -0.99, None)))
        growth_score = log_growth * 252
        
        # B. HRP Cluster Penalty
        n_cl = cluster_labels.max() + 1
        cl_w = np.zeros(n_cl)
        for i in range(len(w)):
            cl_w[cluster_labels[i]] += w[i]
        penalty_hrp = max(0, cl_w.max() - HRP_MAX_CLUSTER_WEIGHT) * HRP_PENALTY_MULT
        
        # C. MPT Sharpe Penalty
        port_ret = w @ mean_rets
        port_var = w @ cov_matrix @ w
        current_sharpe = (port_ret - RISK_FREE_RATE) / np.sqrt(max(port_var, 1e-10)) * np.sqrt(252)
        sharpe_floor = mpt_sharpe * MPT_SHARPE_FLOOR_PCT
        penalty_mpt = max(0, sharpe_floor - current_sharpe) * MPT_PENALTY_MULT
        
        return growth_score - penalty_hrp - penalty_mpt
    
    def _run_ga(self, ret_matrix, mean_rets, cov_matrix, cluster_labels, mpt_sharpe):
        """Genetic Algorithm: evolve portfolio weights."""
        n_assets = len(mean_rets)
        pop_size = GA_POPULATION_SIZE
        n_elite = max(2, int(pop_size * GA_ELITE_PCT))
        
        # Initialize: random Dirichlet (sum=1, non-negative)
        population = np.random.dirichlet(np.ones(n_assets), size=pop_size)
        
        best_fitness = -999
        best_weights = np.ones(n_assets) / n_assets
        
        for gen in range(GA_GENERATIONS):
            fitnesses = np.array([
                self._fitness(ind, ret_matrix, mean_rets, cov_matrix, cluster_labels, mpt_sharpe)
                for ind in population
            ])
            
            gen_best = np.argmax(fitnesses)
            if fitnesses[gen_best] > best_fitness:
                best_fitness = fitnesses[gen_best]
                best_weights = population[gen_best].copy()
            
            ranked = np.argsort(fitnesses)[::-1]
            elite = population[ranked[:n_elite]]
            
            new_pop = list(elite)
            while len(new_pop) < pop_size:
                top_30 = ranked[:max(2, int(pop_size * 0.3))]
                p1 = population[np.random.choice(top_30)]
                p2 = population[np.random.choice(top_30)]
                
                if np.random.random() < GA_CROSSOVER_RATE:
                    alpha = np.random.uniform(0.3, 0.7)
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()
                
                if np.random.random() < 0.8:
                    child = child + np.random.normal(0, GA_MUTATION_STD, n_assets)
                
                child = np.maximum(child, 0)
                cs = child.sum()
                child = child / cs if cs > 1e-10 else np.ones(n_assets) / n_assets
                new_pop.append(child)
            
            population = np.array(new_pop[:pop_size])
        
        best_weights = np.maximum(best_weights, 0)
        ws = best_weights.sum()
        best_weights = best_weights / ws if ws > 1e-10 else np.ones(n_assets) / n_assets
        return best_weights, best_fitness
    
    def _kelly_fraction(self, weights, mean_rets, cov_matrix):
        """Continuous Kelly: f* = (E[Rp]-Rf)/Var(Rp), then Half-Kelly, clamped [0.2, 1.0]."""
        port_ret = weights @ mean_rets
        port_var = weights @ cov_matrix @ weights
        if port_var < 1e-10:
            return 1.0
        kelly_f = (port_ret - RISK_FREE_RATE) / port_var
        return float(np.clip(kelly_f * KELLY_MULTIPLIER, 0.2, 1.0))
    
    def optimize(self, stocks, idx):
        """
        Main entry: returns (weights_dict, kelly_fraction, info_dict).
        weights_dict: {stock: weight} summing to 1
        kelly_fraction: fraction of capital to invest (rest is cash)
        """
        if not stocks or len(stocks) < 2:
            if stocks:
                return {stocks[0]: 1.0}, 1.0, {'method': 'single_stock'}
            return {}, 1.0, {'method': 'empty'}
        
        ret_matrix, valid_stocks = self._get_returns_matrix(stocks, idx)
        if ret_matrix is None or len(valid_stocks) < 2:
            w = {s: 1.0/len(stocks) for s in stocks}
            return w, 1.0, {'method': 'fallback_equal', 'reason': 'insufficient_data'}
        
        n = len(valid_stocks)
        mean_rets = ret_matrix.mean().values
        cov_matrix = ret_matrix.cov().values + np.eye(n) * 1e-8
        
        cluster_labels = self._compute_hrp_clusters(ret_matrix)
        mpt_sharpe = self._compute_mpt_benchmark_sharpe(mean_rets, cov_matrix)
        
        try:
            ga_weights, ga_fitness = self._run_ga(ret_matrix, mean_rets, cov_matrix, cluster_labels, mpt_sharpe)
        except Exception as e:
            w = {s: 1.0/len(stocks) for s in stocks}
            return w, 1.0, {'method': 'fallback_ga_error', 'error': str(e)}
        
        kelly_frac = self._kelly_fraction(ga_weights, mean_rets, cov_matrix)
        
        weights_dict = {}
        for i, stock in enumerate(valid_stocks):
            if ga_weights[i] > 1e-6:
                weights_dict[stock] = float(ga_weights[i])
        
        total_w = sum(weights_dict.values())
        if total_w > 1e-10:
            weights_dict = {s: w/total_w for s, w in weights_dict.items()}
        
        # Cluster stats
        n_cl = int(cluster_labels.max() + 1)
        cl_w = {}
        for i, stock in enumerate(valid_stocks):
            cl = int(cluster_labels[i])
            cl_w[cl] = cl_w.get(cl, 0) + float(ga_weights[i])
        
        port_ret = float(ga_weights @ mean_rets * 252)
        port_vol = float(np.sqrt(ga_weights @ cov_matrix @ ga_weights) * np.sqrt(252))
        port_sharpe = (port_ret - RISK_FREE_RATE*252) / (port_vol + 1e-10)
        
        info = {
            'method': 'genetic_algorithm',
            'n_stocks_optimized': n, 'n_stocks_input': len(stocks),
            'ga_fitness': float(ga_fitness),
            'mpt_benchmark_sharpe': float(mpt_sharpe),
            'portfolio_sharpe': float(port_sharpe),
            'portfolio_return_ann': port_ret, 'portfolio_vol_ann': port_vol,
            'kelly_fraction': float(kelly_frac),
            'n_clusters': n_cl,
            'max_cluster_weight': float(max(cl_w.values())) if cl_w else 0,
            'n_nonzero_weights': sum(1 for w in ga_weights if w > 1e-4),
            'top_5_weights': sorted(weights_dict.items(), key=lambda x: -x[1])[:5]
        }
        return weights_dict, kelly_frac, info

class VolumeExitCalculator:
    def __init__(self, df_close, df_high, df_low, df_volume):
        self.close = df_close
        self.high = df_high
        self.low = df_low
        self.volume = df_volume
        self._precompute()
    
    def _precompute(self):
        print('  Computing volume exit indicators...')
        self.avg_vol = self.volume.rolling(EXHAUSTION_LOOKBACK, min_periods=5).mean()
        self.vol_ratio = self.volume / (self.avg_vol + 1e-10)
        pc = self.close.diff()
        self.obv = (self.volume * np.sign(pc).fillna(0)).cumsum()
        hl = self.high - self.low + 1e-10
        mfm = ((self.close - self.low) - (self.high - self.close)) / hl
        self.cmf = (mfm * self.volume).rolling(CMF_LOOKBACK, min_periods=5).sum() / (self.volume.rolling(CMF_LOOKBACK, min_periods=5).sum() + 1e-10)
        self.adv = self.volume.rolling(SLIPPAGE_LOOKBACK, min_periods=5).mean()
        self.adv_val = self.adv * self.close
        print('  Volume indicators ready')
    
    def check_exhaustion(self, stock, idx):
        if not EXHAUSTION_ENABLED or stock not in self.vol_ratio.columns:
            return False, 0
        try:
            r = self.vol_ratio[stock].iloc[idx]
            return (r >= EXHAUSTION_VOLUME_MULTIPLE, float(r)) if not np.isnan(r) else (False, 0)
        except:
            return False, 0
    
    def check_obv_div(self, stock, idx):
        if not OBV_DIVERGENCE_ENABLED or idx < OBV_LOOKBACK:
            return False, 0, 0
        try:
            p0, p1 = self.close[stock].iloc[idx - OBV_LOOKBACK], self.close[stock].iloc[idx]
            o0, o1 = self.obv[stock].iloc[idx - OBV_LOOKBACK], self.obv[stock].iloc[idx]
            if p0 <= 0 or np.isnan(p0):
                return False, 0, 0
            pc = (p1 / p0) - 1
            oc = (o1 - o0) / (abs(o0) + 1e-10)
            return (pc > OBV_PRICE_THRESHOLD and oc < OBV_DIVERGENCE_THRESHOLD), float(pc), float(oc)
        except:
            return False, 0, 0
    
    def check_cmf(self, stock, idx):
        if not CMF_EXIT_ENABLED or stock not in self.cmf.columns:
            return False, 0
        try:
            v = self.cmf[stock].iloc[idx]
            return (v < CMF_EXIT_THRESHOLD, float(v)) if not np.isnan(v) else (False, 0)
        except:
            return False, 0
    
    def check_liq(self, stock, idx, qty):
        if not LIQUIDITY_EXIT_ENABLED:
            return False, 0
        try:
            adv = self.adv[stock].iloc[idx]
            av = self.adv_val[stock].iloc[idx]
            if np.isnan(adv) or av < MIN_ADV_VALUE:
                return True, 1.0
            r = qty / (adv + 1e-10)
            return r > SLIPPAGE_ADV_RATIO, float(r)
        except:
            return False, 0
    
    def get_all_exit_signals(self, stock, idx, qty, entry, stop, vol_calc, ema_calc):
        sigs = {}
        triggered = []
        
        ema_exit, ema_sig = ema_calc.check_exit_condition(stock, idx)
        sigs['ema'] = {'trigger': ema_exit, 'signal': ema_sig}
        if ema_exit:
            triggered.append('EMA_BAND')
        
        if ATR_STOP_LOSS_ENABLED and stop > 0:
            atr_t, curr, st = vol_calc.check_atr_stop_loss(stock, idx, entry, stop)
            sigs['atr'] = {'trigger': atr_t, 'price': curr, 'stop': st}
            if atr_t:
                triggered.append('ATR_STOP')
        else:
            sigs['atr'] = {'trigger': False}
        
        exh, exh_r = self.check_exhaustion(stock, idx)
        sigs['exh'] = {'trigger': exh, 'val': exh_r}
        if exh:
            triggered.append('EXHAUSTION')
        
        obv, _, _ = self.check_obv_div(stock, idx)
        sigs['obv'] = {'trigger': obv}
        if obv:
            triggered.append('OBV_DIV')
        
        cmf, cmf_v = self.check_cmf(stock, idx)
        sigs['cmf'] = {'trigger': cmf, 'val': cmf_v}
        if cmf:
            triggered.append('CMF')
        
        liq, liq_r = self.check_liq(stock, idx, qty)
        sigs['liq'] = {'trigger': liq}
        if liq:
            triggered.append('LIQUIDITY')
        
        sigs['triggered'] = triggered
        sigs['count'] = len(triggered)
        
        # v5.2: EMA is just one of the signals — need MIN_EXIT_SIGNALS_REQUIRED (2) to exit
        if len(triggered) >= MIN_EXIT_SIGNALS_REQUIRED:
            sigs['should_exit'] = True
            sigs['reason'] = f"MULTI ({'+'.join(triggered)})"
        else:
            sigs['should_exit'] = False
            sigs['reason'] = None
        
        return sigs

def load_sector_data_fast(mcap_folder, sb_folder):
    mp, sp = Path(mcap_folder), Path(sb_folder)
    if not mp.exists() or not sp.exists():
        return {}
    sd = {}
    for f in mp.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            df['tradedate'] = pd.to_datetime(df['tradedate'])
            if 'gics_sector' in df.columns:
                for sec in df['gics_sector'].unique():
                    sdf = df[df['gics_sector'] == sec].sort_values('tradedate')
                    o4 = (sdf['open'] + sdf['high'] + sdf['low'] + sdf['close']) / 4
                    r = o4.pct_change()
                    cs = r.rolling(252).apply(lambda x: x.sum() / (x.std() * np.sqrt(252) + 1e-10), raw=True)
                    sd[int(sec)] = {'mcap': pd.Series(cs.values, index=sdf['tradedate'].values)}
        except:
            pass
    for f in sp.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            df['tradedate'] = pd.to_datetime(df['tradedate'])
            if 'gics_sector' in df.columns:
                for sec in df['gics_sector'].unique():
                    sdf = df[df['gics_sector'] == sec].sort_values('tradedate')
                    o4 = (sdf['open'] + sdf['high'] + sdf['low'] + sdf['close']) / 4
                    r = o4.pct_change()
                    cs = r.rolling(252).apply(lambda x: x.sum() / (x.std() * np.sqrt(252) + 1e-10), raw=True)
                    if int(sec) in sd:
                        sd[int(sec)]['sb'] = pd.Series(cs.values, index=sdf['tradedate'].values)
        except:
            pass
    return {s: d for s, d in sd.items() if 'mcap' in d and 'sb' in d}

def classify_sectors(sd, date):
    date = pd.to_datetime(date)
    res = {}
    for sec, d in sd.items():
        try:
            mi, si = d['mcap'].index <= date, d['sb'].index <= date
            if mi.any() and si.any():
                m, s = d['mcap'][mi].iloc[-1], d['sb'][si].iloc[-1]
                if not np.isnan(m) and not np.isnan(s):
                    res[sec] = (m, s)
        except:
            pass
    if not res:
        return {q: list(SECTOR_NAMES.keys()) for q in ['HH', 'HL', 'LH', 'LL']}
    mm, sm = np.median([v[0] for v in res.values()]), np.median([v[1] for v in res.values()])
    quads = {'HH': [], 'HL': [], 'LH': [], 'LL': []}
    for sec, (m, s) in res.items():
        quads[('H' if m >= mm else 'L') + ('H' if s >= sm else 'L')].append(sec)
    return quads

def get_gics(f):
    try:
        return int(f)
    except:
        return {'energy': 10, 'materials': 15, 'industrials': 20, 'consumer_discretionary': 25, 'consumer_staples': 30, 'health_care': 35, 'financials': 40, 'information_technology': 45, 'communication_services': 50, 'utilities': 55, 'real_estate': 60}.get(f.lower().replace(' ', '_').replace('-', '_'))

def main():
    print('='*60)
    print('LOADING DATA')
    print('='*60)

    sector_data = load_sector_data_fast(MCAP_FOLDER, SMARTBETA_FOLDER)
    print(f'Loaded {len(sector_data)} sectors')

    if not os.path.exists(DATASET_FOLDER):
        raise FileNotFoundError(f'Dataset not found: {DATASET_FOLDER}')

    stock_sectors = {}
    all_dfs = {}

    for subfolder in os.listdir(DATASET_FOLDER):
        path = os.path.join(DATASET_FOLDER, subfolder)
        if not os.path.isdir(path):
            continue
        gics = get_gics(subfolder)
        if gics is None:
            continue
        cnt = 0
        for f in os.listdir(path):
            if not f.endswith('.csv'):
                continue
            try:
                df = pd.read_csv(os.path.join(path, f))
                df.columns = [c.lower().strip() for c in df.columns]
                if 'date' in df.columns:
                    df.rename(columns={'date': 'tradedate'}, inplace=True)
                df['tradedate'] = pd.to_datetime(df['tradedate'])
                df = df.dropna(subset=['tradedate']).sort_values('tradedate')
                if len(df) >= 500:
                    sid = f'{gics}_{f}'
                    all_dfs[sid] = df.set_index('tradedate')
                    stock_sectors[sid] = gics
                    cnt += 1
            except:
                pass
        print(f'  {subfolder}: {cnt} stocks')

    print(f'\nTotal: {len(all_dfs)} stocks')

    print('\nCreating aligned DataFrames...')
    all_dates = sorted(set().union(*[set(df.index) for df in all_dfs.values()]))
    print(f'Date range: {all_dates[0]} to {all_dates[-1]}')

    df_close = pd.DataFrame(index=all_dates)
    df_high = pd.DataFrame(index=all_dates)
    df_low = pd.DataFrame(index=all_dates)
    df_volume = pd.DataFrame(index=all_dates)

    for sid, df in all_dfs.items():
        df_close[sid] = df['close'] if 'close' in df.columns else np.nan
        df_high[sid] = df.get('high', df.get('close', np.nan))
        df_low[sid] = df.get('low', df.get('close', np.nan))
        df_volume[sid] = df.get('traded_volume', df.get('volume', 1000000))

    df_close = df_close.ffill().bfill()
    df_high = df_high.ffill().bfill()
    df_low = df_low.ffill().bfill()
    df_volume = df_volume.ffill().bfill()
    print(f'Shape: {df_close.shape}')

    print('\nInitializing calculators...')
    feature_calc = QuadrantFeatureCalculator(df_close, df_high, df_low, df_volume, stock_sectors)
    volume_calc = VolumeExitCalculator(df_close, df_high, df_low, df_volume)
    vol_calc = VolatilityCalculator(df_close, df_high, df_low, stock_sectors)

    print('  Building HH+HL+LH stock pool for breadth...')
    _init_date = all_dates[min(WARMUP_DAYS, len(all_dates)-1)]
    _init_qsecs = classify_sectors(sector_data, _init_date)
    _pool_sectors = set()
    for q in TARGET_QUADRANTS:
        _pool_sectors.update(_init_qsecs.get(q, []))
    _pool_stocks = [s for s, sec in stock_sectors.items() if sec in _pool_sectors]
    print(f'  Pool: {len(_pool_stocks)} stocks from sectors {[SECTOR_NAMES.get(s,s) for s in sorted(_pool_sectors)]}')

    breadth_calc = AdvancedBreadthCalculator(df_close, pool_stocks=_pool_stocks)
    ema_calc = WalkForwardEMABandCalculator(df_close)
    ranker = QuadrantRanker(feature_calc, ema_calc, stock_sectors)

    if GA_OPTIMIZER_ENABLED:
        ga_optimizer = GeneticPortfolioOptimizer(df_close, stock_sectors)
    else:
        ga_optimizer = None

    print(f'\n  Normal features:')
    for q, w in NORMAL_WEIGHTS.items():
        w_str = ', '.join([f'{f}={v:.2f}' for f, v in w.items()])
        print(f'    {q}: {w_str}')
    print(f'  Defensive features:')
    for q, w in DEFENSIVE_WEIGHTS.items():
        w_str = ', '.join([f'{f}={v:.2f}' for f, v in w.items()])
        print(f'    {q}: {w_str}')

    print('\n' + '='*60)
    print('RUNNING SIMULATION v4.1 (Look-Ahead Fixed + Efficient Rebal)')
    print('='*60)
    print(f'EMA Rules: {len(ema_calc.rules)} | Band: {BAND_PCT:.1%}')
    print(f'Breadth SMA: {BREADTH_FAST_SMA}/{BREADTH_SLOW_SMA} with {REGIME_CROSSOVER_STD} std dev')
    print(f'ATR Stop: {SECTOR_ATR_MULTIPLIERS}x | Look-ahead: signals lagged T-1, exec at T open')

    cash = float(INITIAL_CAPITAL)
    holdings = {}
    equity_curve = [float(INITIAL_CAPITAL)]
    dates_out = [all_dates[0]]
    trade_log = []
    exit_log = []
    breadth_log = []

    start_idx = WARMUP_DAYS
    force_rebal = False
    sim_start = time.time()
    last_month = None
    last_qtr = None

    current_regime = 'NORMAL'
    prev_regime = 'NORMAL'
    regime_switch_pending = False

    # v4.1: Pending rebalance target (computed EOD, executed next day open)
    pending_rebal = None  # Will hold {'new_pf': ..., 'sel': ..., 'rtype': ...}

    # v5.2: Emergency rebalance cooldown tracker
    last_emergency_idx = -999

    for i in range(start_idx, len(all_dates)):
        date = all_dates[i]
        date_prev = all_dates[i-1]

        curr_month = date.month if hasattr(date, 'month') else (i // 21)
        is_month_start = (curr_month != last_month)
        curr_qtr = (curr_month - 1) // 3
        is_qtr_start = is_month_start and (curr_qtr != last_qtr)

        # ================================================================
        # v4.1: EXECUTE PENDING REBALANCE (from yesterday's decision)
        # Uses today's OPEN price (approximated by previous close)
        # ================================================================
        if pending_rebal is not None:
            pr = pending_rebal
            new_pf = pr['new_pf']
            sel = pr['sel']
            rtype = pr['rtype']
            is_emergency = pr.get('is_emergency', False)

            print(f'\n[EXEC {rtype}] {date} (decided {date_prev})')
            if is_emergency:
                last_emergency_idx = i

            # Use previous day's close as proxy for today's open
            exec_price_col = date_prev

            # === EFFICIENT REBALANCE: only trade the diff ===
            # Step 1: Identify KEEP, SELL, BUY
            old_stocks = set(holdings.keys())
            new_stocks = set(new_pf.keys())

            to_sell = old_stocks - new_stocks   # stocks to exit entirely
            to_keep = old_stocks & new_stocks   # stocks in both
            to_buy = new_stocks - old_stocks    # stocks to enter

            # Step 2: GA Optimizer for portfolio weights (or equal weight fallback)
            all_new_stocks = list(new_pf.keys())
            kelly_frac = 1.0

            if GA_OPTIMIZER_ENABLED and ga_optimizer is not None and len(all_new_stocks) >= 2:
                target_wts, kelly_frac, opt_info = ga_optimizer.optimize(all_new_stocks, i - 1)
                for s in all_new_stocks:
                    if s not in target_wts:
                        target_wts[s] = 0
                tw = sum(target_wts.values())
                target_wts = {s: w/tw for s, w in target_wts.items()} if tw > 0 else {s: 1/len(all_new_stocks) for s in all_new_stocks}
                if opt_info.get('method') == 'genetic_algorithm':
                    print(f'  [GA-OPT] Sharpe: {opt_info["portfolio_sharpe"]:.2f} | '
                          f'MPT bench: {opt_info["mpt_benchmark_sharpe"]:.2f} | '
                          f'Kelly: {kelly_frac:.1%} | '
                          f'Clusters: {opt_info["n_clusters"]} (max: {opt_info["max_cluster_weight"]:.1%}) | '
                          f'Nonzero: {opt_info["n_nonzero_weights"]}/{opt_info["n_stocks_optimized"]}')
            else:
                target_wts = {s: 1/len(all_new_stocks) for s in all_new_stocks}
                opt_info = {'method': 'equal_weight'}

            # Sector cap enforcement (post-optimization safety net)
            if SECTOR_CAP_ENABLED:
                sw = defaultdict(float)
                for s in all_new_stocks:
                    sw[stock_sectors.get(s, 0)] += target_wts.get(s, 0)
                for sec, w in sw.items():
                    if w > MAX_SECTOR_ALLOCATION:
                        sc = MAX_SECTOR_ALLOCATION / w
                        for s in all_new_stocks:
                            if stock_sectors.get(s, 0) == sec:
                                target_wts[s] *= sc
                tw = sum(target_wts.values())
                target_wts = {s: w/tw for s, w in target_wts.items()} if tw > 0 else target_wts

            # Step 3: SELL stocks not in new portfolio
            for stock in to_sell:
                h = holdings[stock]
                price = float(df_close.at[exec_price_col, stock]) if stock in df_close.columns else 0
                if price > 0:
                    sale = h['qty'] * price
                    cash += sale - (sale * COMMISSION_PCT)
                    trade_log.append({
                        'Date': date, 'Stock': stock, 'Action': 'SELL', 'Reason': rtype,
                        'Qty': h['qty'], 'Price': price, 'PnL': (price/h['entry_price']-1),
                        'Sector': SECTOR_NAMES.get(stock_sectors.get(stock, 0), 'Unknown'),
                        'Sector_Counts': sel.get('sector_counts_str', ''),
                        'Breadth': pr.get('bpct', 0), 'Regime': current_regime,
                        'Divergence': pr.get('has_divergence', False)
                    })
                del holdings[stock]

            # Step 4: Compute total equity after sells
            total_eq = cash + sum(
                holdings[s]['qty'] * float(df_close.at[exec_price_col, s])
                for s in holdings if s in df_close.columns
                and not np.isnan(df_close.at[exec_price_col, s]))

            # Kelly fraction scales invested capital; rest stays as cash buffer
            avail = total_eq * 0.98 * kelly_frac

            # Emergency spike guard
            if is_emergency:
                max_alloc = total_eq * EMERGENCY_MAX_PER_STOCK_PCT
                print(f'  [SPIKE_GUARD] Emergency: max {EMERGENCY_MAX_PER_STOCK_PCT:.0%}/stock')
            else:
                max_alloc = avail

            # Step 5: ADJUST kept stocks (resize positions if needed)
            for stock in to_keep:
                h = holdings[stock]
                price = float(df_close.at[exec_price_col, stock])
                if np.isnan(price) or price <= 0:
                    continue
                current_value = h['qty'] * price
                target_value = avail * target_wts.get(stock, 0)
                target_value = min(target_value, max_alloc)
                target_qty = int(target_value / price)
                diff_qty = target_qty - h['qty']

                if diff_qty > 5:  # Need to buy more (only if meaningful)
                    add_cost = diff_qty * price
                    add_fee = add_cost * COMMISSION_PCT
                    if cash >= add_cost + add_fee:
                        cash -= add_cost + add_fee
                        holdings[stock]['qty'] += diff_qty
                        trade_log.append({
                            'Date': date, 'Stock': stock, 'Action': 'ADD', 'Reason': rtype,
                            'Qty': diff_qty, 'Price': price,
                            'Sector': SECTOR_NAMES.get(stock_sectors.get(stock, 0), 'Unknown'),
                            'Breadth': pr.get('bpct', 0), 'Regime': current_regime
                        })
                elif diff_qty < -5:  # Need to sell some
                    sell_qty = abs(diff_qty)
                    sale = sell_qty * price
                    cash += sale - (sale * COMMISSION_PCT)
                    holdings[stock]['qty'] -= sell_qty
                    trade_log.append({
                        'Date': date, 'Stock': stock, 'Action': 'TRIM', 'Reason': rtype,
                        'Qty': sell_qty, 'Price': price,
                        'Sector': SECTOR_NAMES.get(stock_sectors.get(stock, 0), 'Unknown'),
                        'Breadth': pr.get('bpct', 0), 'Regime': current_regime
                    })
                # Update quadrant tag and stop
                holdings[stock]['quadrant'] = new_pf[stock]
                holdings[stock]['stop_price'] = vol_calc.get_atr_stop_price(stock, i - 1, holdings[stock]['entry_price'])

            # Step 6: BUY new stocks
            for stock in to_buy:
                if stock not in df_close.columns:
                    continue
                price = float(df_close.at[exec_price_col, stock])
                if np.isnan(price) or price <= 0:
                    continue
                alloc = min(avail * target_wts.get(stock, 0), max_alloc)
                qty = int(alloc / price)
                if qty <= 0:
                    continue
                cost = qty * price
                fee = cost * COMMISSION_PCT
                if cash >= cost + fee:
                    cash -= cost + fee
                    stop = vol_calc.get_atr_stop_price(stock, i - 1, price)
                    holdings[stock] = {'qty': qty, 'entry_price': price, 'stop_price': stop, 'quadrant': new_pf[stock]}
                    trade_log.append({
                        'Date': date, 'Stock': stock, 'Action': 'BUY', 'Reason': rtype,
                        'Qty': qty, 'Price': price, 'Stop': stop, 'Weight': target_wts.get(stock, 0),
                        'Sector': SECTOR_NAMES.get(stock_sectors.get(stock, 0), 'Unknown'),
                        'Sector_Counts': sel.get('sector_counts_str', ''),
                        'EMA_Signal': ema_calc.get_signal(stock, i),
                        'Breadth': pr.get('bpct', 0), 'Regime': current_regime,
                        'Divergence': pr.get('has_divergence', False)
                    })

            kept = len(to_keep)
            print(f'  Sold: {len(to_sell)} | Kept: {kept} | Bought: {len(to_buy)} | Holdings: {len(holdings)} | Cash: Rs {cash:,.0f}')
            pending_rebal = None

        # ================================================================
        # Breadth & regime (all signals already lagged in calculators)
        # ================================================================
        holdings_returns = {}
        for stock, h in holdings.items():
            try:
                curr_price = df_close.at[date_prev, stock]  # Use prev close
                holdings_returns[stock] = (curr_price / h['entry_price']) - 1
            except:
                pass

        breadth_status = breadth_calc.get_full_status(i, holdings_returns)
        bpct = breadth_status['breadth_pct']
        regime_signal = breadth_status['regime_signal']
        has_divergence = breadth_status['has_divergence']

        # Regime detection
        if regime_signal == 'DEFENSIVE' and current_regime == 'NORMAL':
            regime_switch_pending = True
            prev_regime = current_regime
            current_regime = 'DEFENSIVE'
            print(f'  [REGIME] {date}: NORMAL -> DEFENSIVE, rebal queued')
        elif regime_signal == 'NORMAL' and current_regime == 'DEFENSIVE':
            regime_switch_pending = True
            prev_regime = current_regime
            current_regime = 'NORMAL'
            print(f'  [REGIME] {date}: DEFENSIVE -> NORMAL, rebal queued')

        breadth_log.append({
            'Date': date, 'Breadth_Pct': bpct,
            'Is_Strong': breadth_status['is_strong'],
            'Breadth_Fast_SMA': breadth_status['breadth_fast_sma'],
            'Breadth_Slow_SMA': breadth_status['breadth_slow_sma'],
            'Sigma_Threshold': breadth_status['sigma_threshold'],
            'SMA_Gap': breadth_status['sma_gap'],
            'Regime_Signal': regime_signal,
            'Current_Regime': current_regime,
            'Has_Divergence': has_divergence,
            'Top10_Return': breadth_status['top_10pct_return'],
            'Breadth_Change': breadth_status['breadth_change'],
            'Divergence_Score': breadth_status['divergence_score']
        })

        # ================================================================
        # Daily Exit Checks (signals already lagged in EMA calc)
        # Execute exits at previous close (proxy for today's open)
        # ================================================================
        before = len(holdings)
        exits = []

        for stock in list(holdings.keys()):
            h = holdings[stock]
            sigs = volume_calc.get_all_exit_signals(
                stock, i, h['qty'], h['entry_price'], h['stop_price'], vol_calc, ema_calc)

            if sigs['should_exit']:
                price = float(df_close.at[date_prev, stock]) if stock in df_close.columns else 0
                if price > 0:
                    sale = h['qty'] * price
                    cash += sale - (sale * COMMISSION_PCT)
                    pnl = (price / h['entry_price'] - 1) if h['entry_price'] > 0 else 0
                    exits.append({
                        'Date': date, 'Stock': stock, 'Reason': sigs['reason'],
                        'Signals': '+'.join(sigs['triggered']), 'Count': sigs['count'],
                        'Qty': h['qty'], 'Price': price, 'Entry': h['entry_price'],
                        'PnL': pnl, 'Quadrant': h['quadrant'],
                        'Sector': SECTOR_NAMES.get(stock_sectors.get(stock, 0), 'Unknown'),
                        'Breadth': bpct, 'Regime': current_regime
                    })
                del holdings[stock]

        after = len(holdings)
        for e in exits:
            e['Before'] = before
            e['After'] = after
            exit_log.append(e)

        if after < MIN_HOLDINGS_THRESHOLD:
            force_rebal = True

        # ================================================================
        # DECIDE REBALANCE (decision made EOD, execution queued for T+1)
        # ================================================================
        do_rebal = False
        rtype = None
        is_emergency = False

        if regime_switch_pending:
            do_rebal = True
            rtype = f'REGIME_SWITCH ({prev_regime}->{current_regime})'
            regime_switch_pending = False
        elif force_rebal:
            # v5.2: Cooldown — skip if we just did an emergency rebal recently
            days_since_last = i - last_emergency_idx
            if days_since_last >= EMERGENCY_COOLDOWN_DAYS:
                do_rebal = True
                rtype = 'EMERGENCY'
                is_emergency = True
            else:
                pass  # Cooldown active, suppress emergency rebal
            force_rebal = False
        elif is_qtr_start:
            do_rebal = True
            rtype = 'QUARTERLY'

        if do_rebal:
            # Compute new portfolio target using today's signals (already lagged)
            qsecs = classify_sectors(sector_data, date_prev)

            if current_regime == 'DEFENSIVE':
                sel = ranker.select_top_stocks_defensive(idx=i, stock_sectors=stock_sectors, n_stocks=MAX_TOTAL_STOCKS_DEFENSIVE)
            else:
                sel = ranker.select_top_stocks_normal(idx=i, stock_sectors=stock_sectors, quadrant_sectors=qsecs, n_stocks=MAX_TOTAL_STOCKS_NORMAL)

            new_pf = {s[0]: s[2] for s in sel['stocks']}

            # Queue for T+1 execution
            pending_rebal = {
                'new_pf': new_pf, 'sel': sel, 'rtype': rtype,
                'is_emergency': is_emergency, 'bpct': bpct,
                'has_divergence': has_divergence
            }
            print(f'  [QUEUED] {date}: {rtype} -> {len(new_pf)} stocks (exec T+1)')

        if is_month_start:
            last_month = curr_month
        if is_qtr_start:
            last_qtr = curr_qtr

        # Equity valuation at today's close
        curr_val = sum(
            holdings[s]['qty'] * float(df_close.at[date, s])
            for s in holdings if s in df_close.columns
            and not np.isnan(df_close.at[date, s]))
        equity_curve.append(cash + curr_val)
        dates_out.append(date)

    print(f'\nSimulation complete in {time.time() - sim_start:.1f}s')

    print('\n' + '='*60)
    print('RESULTS v4.1 (Look-Ahead Fixed + Efficient Rebal)')
    print('='*60)

    eq = pd.Series(equity_curve, index=[dates_out[0]] + dates_out[1:])
    rets = eq.pct_change().dropna()
    years = (dates_out[-1] - dates_out[0]).days / 365.25
    cagr = _perf_cagr(eq)
    dd = eq / eq.cummax() - 1
    max_dd = dd.min()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    print(f'Initial: Rs {INITIAL_CAPITAL:,.0f}')
    print(f'Final:   Rs {equity_curve[-1]:,.0f}')
    print(f'Sharpe:  {sharpe:.2f}')
    print(f'Calmar:  {calmar:.2f}')


    if breadth_log:
        bdf = pd.DataFrame(breadth_log)
        print(f'\nBreadth Stats (HH+HL+LH Pool):')
        print(f'  Avg: {bdf["Breadth_Pct"].mean():.1%}')
        def_days = (bdf['Current_Regime'] == 'DEFENSIVE').sum()
        norm_days = (bdf['Current_Regime'] == 'NORMAL').sum()
        total_days = norm_days + def_days
        switches = (bdf['Current_Regime'] != bdf['Current_Regime'].shift(1)).sum()
        print(f'\nRegime Stats:')
        print(f'  Normal: {norm_days} days ({norm_days/max(total_days,1)*100:.1f}%)')
        print(f'  Defensive: {def_days} days ({def_days/max(total_days,1)*100:.1f}%)')
        print(f'  Switches: {switches}')

    if trade_log:
        tdf = pd.DataFrame(trade_log)
        print(f'\nTrade Stats:')
        print(f'  Total trades: {len(tdf)}')
        for action in ['BUY', 'SELL', 'ADD', 'TRIM']:
            cnt = len(tdf[tdf['Action'] == action])
            if cnt > 0:
                print(f'  {action}: {cnt}')
        # Estimate commission savings
        adds = len(tdf[tdf['Action'] == 'ADD'])
        trims = len(tdf[tdf['Action'] == 'TRIM'])
        print(f'  Efficient rebal (ADD+TRIM instead of SELL+BUY): {adds + trims} trades')

    if exit_log:
        edf = pd.DataFrame(exit_log)
        print(f'\nExit Stats: {len(edf)} total')

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    axes[0].plot(dates_out, equity_curve, 'b-', lw=1)
    axes[0].set_title(f'Equity Curve v4.1 (CAGR: {cagr*100:.1f}%, Max DD: {max_dd*100:.1f}%)')
    axes[0].set_ylabel('Portfolio Value (Rs)')
    axes[0].grid(True, alpha=0.3)
    if breadth_log:
        bdf = pd.DataFrame(breadth_log)
        is_def = bdf['Current_Regime'] == 'DEFENSIVE'
        if is_def.any():
            ylim = axes[0].get_ylim()
            axes[0].fill_between(bdf['Date'], ylim[0], ylim[1], where=is_def.values, color='orange', alpha=0.15, label='Defensive')
            axes[0].set_ylim(ylim)
            axes[0].legend(loc='upper left')

    axes[1].fill_between(dates_out[1:], dd.values[1:], color='red', alpha=0.3)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown')
    axes[1].grid(True, alpha=0.3)

    if breadth_log:
        bdf = pd.DataFrame(breadth_log)
        axes[2].plot(bdf['Date'], bdf['Breadth_Pct'], 'g-', lw=0.5, alpha=0.7, label='Breadth (HH+HL+LH)')
        axes[2].plot(bdf['Date'], bdf['Breadth_Fast_SMA'], 'b-', lw=1, label=f'{BREADTH_FAST_SMA}d SMA')
        axes[2].plot(bdf['Date'], bdf['Breadth_Slow_SMA'], 'r-', lw=1, label=f'{BREADTH_SLOW_SMA}d SMA')
        upper_band = bdf['Breadth_Slow_SMA'] + bdf['Sigma_Threshold']
        lower_band = bdf['Breadth_Slow_SMA'] - bdf['Sigma_Threshold']
        axes[2].fill_between(bdf['Date'], lower_band, upper_band, color='gray', alpha=0.15, label='1 std band')
        is_def = bdf['Current_Regime'] == 'DEFENSIVE'
        if is_def.any():
            axes[2].fill_between(bdf['Date'], 0, 1, where=is_def.values, color='red', alpha=0.15, label='Defensive')
        axes[2].set_title('Market Breadth & Regime Switch (Lagged T-1)')
        axes[2].set_ylabel('% Above SMA')
        axes[2].legend(loc='lower left', fontsize=8)
        axes[2].grid(True, alpha=0.3)

    if breadth_log:
        axes[3].plot(bdf['Date'], bdf['Divergence_Score'], 'purple', lw=0.5, alpha=0.7)
        axes[3].fill_between(bdf['Date'], 0, bdf['Divergence_Score'], where=bdf['Has_Divergence'], color='red', alpha=0.3, label='Divergence')
        axes[3].set_title('Breadth Divergence Score')
        axes[3].set_ylabel('Divergence Score')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quadrant_v4_1_results.png', dpi=100)
    plt.show()

    if trade_log:
        pd.DataFrame(trade_log).to_csv(TRADE_LOG_FILE, index=False)
        print(f'Trade log: {TRADE_LOG_FILE}')
    if exit_log:
        pd.DataFrame(exit_log).to_csv('exit_log_v4.csv', index=False)
        print(f'Exit log: exit_log_v4.csv')
    if breadth_log:
        pd.DataFrame(breadth_log).to_csv('breadth_log_v4.csv', index=False)
        print(f'Breadth log: breadth_log_v4.csv')

    if trade_log:
        print('\nLast 10 buys:')
        tdf = pd.DataFrame(trade_log)
        buys = tdf[tdf['Action'] == 'BUY'][['Date', 'Stock', 'Sector', 'Regime', 'Breadth']].tail(10)
        print(buys.to_string())

    if breadth_log:
        bdf = pd.DataFrame(breadth_log)
        switches = bdf[bdf['Current_Regime'] != bdf['Current_Regime'].shift(1)][['Date', 'Breadth_Pct', 'Breadth_Fast_SMA', 'Breadth_Slow_SMA', 'Sigma_Threshold', 'Current_Regime']]
        if len(switches) > 0:
            print(f'\nRegime Switches ({len(switches)}):')
            print(switches.to_string())

        div_events = bdf[bdf['Has_Divergence']][['Date', 'Breadth_Pct', 'Top10_Return', 'Breadth_Change', 'Divergence_Score']]
        if len(div_events) > 0:
            print(f'\nDivergence Events ({len(div_events)}):')
            print(div_events.head(20).to_string())

    # ============================================================
    # KRITI: Performance Reporting Requirements (vs Nifty 500)
    # ============================================================
    try:
        _perf_run_reporting(
            eq,
            output_prefix="quadrant_v5",
            trading_days=252,
            benchmark_tickers=None,
            periods=[
                ("2010-2020", "2010-01-01", "2020-12-31"),
            ],
        )
    except Exception as _e:
        print(f"[PERF REPORT] Could not generate Nifty 500 performance report: {_e}")
        print("[PERF REPORT] If the benchmark ticker fails, edit benchmark_tickers in _perf_run_reporting().")

if __name__ == "__main__":
    main()


# ============================================================
# MERGE GLUE (added)
# ============================================================
# Capture strategy main (strategy script overwrote `main`)
strategy_main = main

# Restore the original module name
__name__ = __MERGED_ORIGINAL_NAME__

def merged_main():
    # 1) Build folders / indices (script 1)
    folder_main()

    # 2) Point strategy to the folders created by script 1 (in OUTPUT_BASE)
    global DATASET_FOLDER, MCAP_FOLDER, SMARTBETA_FOLDER
    DATASET_FOLDER = os.path.join(OUTPUT_BASE, 'dataset')
    MCAP_FOLDER = os.path.join(OUTPUT_BASE, 'mcap_index')
    SMARTBETA_FOLDER = os.path.join(OUTPUT_BASE, 'smart_beta_index')

    # 3) Run strategy (script 2)
    strategy_main()

if __name__ == "__main__":
    merged_main()
