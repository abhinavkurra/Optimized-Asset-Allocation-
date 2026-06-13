# =============================================================================
# Post-Modern Portfolio Theory (PMPT) Optimization - Post 1
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# --- 1. CONSTANTS & DIRECTORY SETUP -------------------------------------------
TRADABLE_TICKERS = [
    'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 
    'CIPLA.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 'MARUTI.NS', 
    'GC=F', 'CL=F'
]
BENCHMARK = '^NSEI'
ALL_TICKERS = TRADABLE_TICKERS + [BENCHMARK]

MAR = 0.06                # 6% Minimum Acceptable Return (annualised)
TRADING_DAYS = 252        
WEIGHT_THRESHOLD = 0.001  
N_RANDOM_PORTFOLIOS = 5000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Create output directory
OUTPUT_DIR = "PMPT_Post1_Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"All outputs will be saved to: {os.path.abspath(OUTPUT_DIR)}\n")

# --- 2. DATA FETCHING & PREPROCESSING -----------------------------------------
print("Downloading 10 years of price data...")
raw_data = yf.download(ALL_TICKERS, period="10y", auto_adjust=True, progress=False)

if isinstance(raw_data.columns, pd.MultiIndex):
    prices = raw_data['Close']
else:
    prices = raw_data

# Clean data and calculate daily returns
prices = prices.ffill().dropna()
returns = prices.pct_change().dropna()

# Separate tradable assets from the benchmark
tradable_returns = returns[TRADABLE_TICKERS]
benchmark_returns = returns[BENCHMARK]

# --- 3. DELIVERABLE 2: METRIC COMPARISON TABLE (MPT vs PMPT) ------------------
print("\n[Output 2] Generating Metric Comparison Table...")
metrics = []
daily_mar = MAR / TRADING_DAYS

for ticker in TRADABLE_TICKERS:
    asset_ret = tradable_returns[ticker]
    
    # MPT Metrics
    ann_ret = asset_ret.mean() * TRADING_DAYS
    ann_vol = asset_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - MAR) / ann_vol
    
    # PMPT Metrics
    downside_diff = asset_ret - daily_mar
    downside_only = np.minimum(downside_diff, 0)
    downside_var = np.mean(downside_only ** 2)
    ann_downside_dev = np.sqrt(downside_var * TRADING_DAYS)
    sortino = (ann_ret - MAR) / ann_downside_dev
    
    metrics.append({
        'Asset': ticker,
        'Exp Return': round(ann_ret, 4),
        'Total Volatility': round(ann_vol, 4),
        'Downside Dev': round(ann_downside_dev, 4),
        'Sharpe Ratio': round(sharpe, 4),
        'Sortino Ratio': round(sortino, 4)
    })

comparison_df = pd.DataFrame(metrics).sort_values(by='Sortino Ratio', ascending=False)
print("\n", comparison_df.to_string(index=False), "\n")

# Save table to CSV
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'deliverable_2_metrics.csv'), index=False)

# --- 4. OPTIMIZATION CLASS ----------------------------------------------------
class LongOnlyPMPTOptimizer:
    def __init__(self, daily_returns: pd.DataFrame, mar: float = 0.06, t_days: int = 252):
        self.returns = daily_returns
        self.mar = mar
        self.T = t_days
        self.tickers = list(daily_returns.columns)
        self.n = len(self.tickers)
        self.mu = self.returns.mean() * self.T

    def port_return(self, w: np.ndarray) -> float:
        return float(np.dot(w, self.mu))

    def port_downside_dev(self, w: np.ndarray) -> float:
        port_daily_ret = self.returns.dot(w)
        daily_mar = self.mar / self.T
        downside_only = np.minimum(port_daily_ret - daily_mar, 0)
        return float(np.sqrt(np.mean(downside_only ** 2) * self.T))

    def sortino(self, w: np.ndarray) -> float:
        ret = self.port_return(w)
        dd = self.port_downside_dev(w)
        return (ret - self.mar) / dd if dd > 1e-9 else -np.inf

    def _neg_sortino(self, w: np.ndarray) -> float:
        return -self.sortino(w)

    def optimise(self) -> dict:
        w0 = np.full(self.n, 1.0 / self.n)
        bounds = [(0.0, 1.0)] * self.n
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        res = minimize(
            fun=self._neg_sortino, x0=w0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        
        w = np.where(res.x < WEIGHT_THRESHOLD, 0.0, res.x)
        self.optimal_weights = w / w.sum()
        
        return {
            'return': self.port_return(self.optimal_weights),
            'downside_dev': self.port_downside_dev(self.optimal_weights),
            'sortino': self.sortino(self.optimal_weights)
        }

    def simulate(self, n: int) -> pd.DataFrame:
        records = []
        for _ in range(n):
            w = np.random.dirichlet(np.ones(self.n))
            records.append({
                'Return': self.port_return(w),
                'Downside_Dev': self.port_downside_dev(w),
                'Sortino': self.sortino(w)
            })
        return pd.DataFrame(records)

# Execute Optimization
optimizer = LongOnlyPMPTOptimizer(tradable_returns, mar=MAR)
opt_summary = optimizer.optimise()
sim_df = optimizer.simulate(N_RANDOM_PORTFOLIOS)

# --- 5. VISUALIZATIONS --------------------------------------------------------

# DELIVERABLE 1: Return Distribution (Fat Tails)
print("[Output 1] Plotting Return Distribution...")
plt.figure(figsize=(10, 6))
asset = "TATASTEEL.NS"
asset_ret = tradable_returns[asset]

count, bins, _ = plt.hist(asset_ret, bins=100, density=True, alpha=0.6, color='gray', label='Actual Returns')
mu, std = norm.fit(asset_ret)
plt.plot(bins, norm.pdf(bins, mu, std), linewidth=2, color='blue', label='Normal Curve (MPT)')

# FIX: Shade downside using the correctly calculated daily MAR
downside_bins = bins[bins < daily_mar]
plt.fill_between(downside_bins, 0, norm.pdf(downside_bins, mu, std), color='red', alpha=0.3, label=f'Downside Risk (< MAR)')

# FIX: Plot the vertical line exactly at the daily MAR
plt.axvline(daily_mar, color='black', linestyle='dashed', linewidth=2, label=f'Daily MAR ({(daily_mar*100):.3f}%)')

plt.title(f"{asset}: Actual Returns vs. Normal Distribution (Fat Tails Proof)")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deliverable_1_fat_tails.png'), dpi=150)
plt.close() # Close to prevent overlapping plots

# DELIVERABLE 3: PMPT Efficient Frontier
# DELIVERABLE 3: PMPT Efficient Frontier
print("[Output 3] Plotting PMPT Efficient Frontier...")
fig_ef = go.Figure()

# Plot the random portfolios
fig_ef.add_trace(go.Scatter(
    x=sim_df['Downside_Dev'] * 100, y=sim_df['Return'] * 100,
    mode='markers',
    marker=dict(color=sim_df['Sortino'], colorscale='Viridis', size=4, opacity=0.6, colorbar=dict(title='Sortino')),
    name='Random Portfolios'
))

# Plot the Max Sortino optimal point
fig_ef.add_trace(go.Scatter(
    x=[opt_summary['downside_dev'] * 100], y=[opt_summary['return'] * 100],
    mode='markers',
    marker=dict(symbol='star', size=20, color='red', line=dict(width=1, color='white')),
    name=f"Max Sortino"
))

# FIX: Add a custom, properly aligned annotation box pointing to the star
fig_ef.add_annotation(
    x=opt_summary['downside_dev'] * 100,
    y=opt_summary['return'] * 100,
    text=f"<b>⭐ Max Sortino: {opt_summary['sortino']:.2f}</b>",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="white",
    ax=60,   # Shifts the text 60 pixels to the right
    ay=-50,  # Shifts the text 50 pixels up
    font=dict(size=12, color="white"),
    bgcolor="rgba(200, 0, 0, 0.8)", # Semi-transparent red background
    bordercolor="white",
    borderwidth=1,
    borderpad=5
)

fig_ef.update_layout(
    title='<b>Post-Modern Efficient Frontier</b><br><sup>Optimised for Downside Deviation</sup>',
    xaxis_title='Downside Deviation (%)', yaxis_title='Expected Return (%)',
    template='plotly_dark', width=900, height=600,
    showlegend=False # Legend isn't needed now that we have a bold annotation
)
fig_ef.write_html(os.path.join(OUTPUT_DIR, 'deliverable_3_pmpt_frontier.html'))
# DELIVERABLE 4: Underwater Drawdown Curve
print("[Output 4] Plotting Underwater Drawdown Curve...")
port_daily_ret = tradable_returns.dot(optimizer.optimal_weights)
cum_ret = (1 + port_daily_ret).cumprod()
running_max = cum_ret.cummax()
drawdown = (cum_ret - running_max) / running_max

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown * 100,
    fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
    line=dict(color='red', width=1.5),
    name='Drawdown'
))

fig_dd.update_layout(
    title='<b>Underwater Drawdown Curve</b><br><sup>Max Sortino Portfolio</sup>',
    xaxis_title='Date', yaxis_title='Drawdown (%)',
    template='plotly_dark', width=1000, height=500
)
fig_dd.write_html(os.path.join(OUTPUT_DIR, 'deliverable_4_drawdown.html'))

# DELIVERABLE 5: Downside Correlation Heatmap
print("[Output 5] Plotting Downside Correlation Heatmap...")
# Filter returns for days where Nifty 50 was negative
downside_days = returns[returns[BENCHMARK] < 0]
downside_corr = downside_days[TRADABLE_TICKERS].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(downside_corr, dtype=bool))
sns.heatmap(
    downside_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
    annot=True, fmt=".2f", square=True, linewidths=0.5
)
plt.title(f"Downside Correlation (Filtered for {BENCHMARK} < 0)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deliverable_5_downside_correlation.png'), dpi=150)
plt.close()