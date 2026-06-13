# =============================================================================
# Long-Only Modern Portfolio Optimization (MPO) for Indian Equities
# Senior Quantitative Financial Developer Implementation
# =============================================================================

# --- 1. IMPORTS --------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import zscore
import logging

# Suppress yfinance and other verbose logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)

# --- 2. CONSTANTS -------------------------------------------------------------
TICKERS = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS',
    'SBIN.NS', 'ICICIBANK.NS', 'TCS.NS', 'LT.NS'
]
START_DATE        = '2022-01-01'
END_DATE          = '2026-01-01'
RISK_FREE_RATE    = 0.06          # 6% ansnualised (Indian T-bill proxy)
TRADING_DAYS      = 252           # Standard annualisation factor
WEIGHT_THRESHOLD  = 0.001         # Drop weights below 0.1%
RANDOM_SEED       = 42
N_RANDOM_PORTFOLIOS = 5_000       # Monte-Carlo frontier portfolios for visualisation

np.random.seed(RANDOM_SEED)

# =============================================================================
# --- 3. DATA FETCHING ---------------------------------------------------------
# =============================================================================
print("=" * 65)
print("  Long-Only Modern Portfolio Optimisation — Indian Equities")
print("=" * 65)
print(f"\n[1/5] Downloading price data ({START_DATE} → {END_DATE}) …")

raw = yf.download(
    tickers    = TICKERS,
    start      = START_DATE,
    end        = END_DATE,
    auto_adjust= True,          # 'Adj Close' equivalent with auto-adjust
    progress   = False,
)

# yfinance returns a MultiIndex; extract the 'Close' level
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw['Close']
else:
    prices = raw

# ── Data quality: keep only tickers actually downloaded ──────────────────────
prices = prices[[c for c in TICKERS if c in prices.columns]]

# Forward-fill isolated gaps (e.g. exchange holidays), then drop leading NaNs
prices = prices.ffill().dropna()

print(f"    ✓ {prices.shape[1]} tickers | {prices.shape[0]} trading days loaded")
print(f"    Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

# =============================================================================
# --- 4. OPTIMIZATION CLASS ----------------------------------------------------
# =============================================================================

class LongOnlyOptimizer:
    """
    Encapsulates all quantitative logic for a Long-Only Mean-Variance
    portfolio optimisation using the Sharpe Ratio as the objective.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted closing prices (rows = dates, cols = tickers).
    risk_free_rate : float
        Annualised risk-free rate (decimal form).
    trading_days : int
        Number of trading days used for annualisation (default 252).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float = 0.06,
        trading_days: int = 252,
    ):
        self.prices         = prices
        self.rf             = risk_free_rate
        self.T              = trading_days
        self.tickers        = list(prices.columns)
        self.n              = len(self.tickers)

        # ── Daily simple returns: r_t = (P_t / P_{t-1}) - 1 ─────────────────
        self.daily_returns  = prices.pct_change().dropna()

        # ── Annualised expected returns vector μ (n × 1) ──────────────────────
        # μ_i = mean(r_i) × T
        self.mu             = self.daily_returns.mean() * self.T

        # ── Annualised covariance matrix Σ (n × n) ────────────────────────────
        # Σ = Cov(R_daily) × T
        # Uses Bessel-corrected sample covariance (ddof=1 by default in pandas)
        self.cov            = self.daily_returns.cov() * self.T

        # ── Correlation matrix (for EDA heatmap) ─────────────────────────────
        self.corr           = self.daily_returns.corr()

        # Placeholder for results
        self.optimal_weights = None
        self.result          = None

    # ── Portfolio statistics ──────────────────────────────────────────────────

    def portfolio_return(self, w: np.ndarray) -> float:
        """
        E[R_p] = wᵀ μ
        Dot product of weight vector and expected returns vector.
        """
        return float(np.dot(w, self.mu))

    def portfolio_volatility(self, w: np.ndarray) -> float:
        """
        σ_p = sqrt(wᵀ Σ w)
        Portfolio standard deviation via the quadratic form of the cov matrix.
        """
        variance = np.dot(w, np.dot(self.cov.values, w))
        return float(np.sqrt(variance))

    def sharpe_ratio(self, w: np.ndarray) -> float:
        """
        SR = (E[R_p] - R_f) / σ_p
        Risk-adjusted excess return per unit of total risk.
        """
        ret = self.portfolio_return(w)
        vol = self.portfolio_volatility(w)
        return (ret - self.rf) / vol if vol > 1e-9 else -np.inf

    def _neg_sharpe(self, w: np.ndarray) -> float:
        """Objective function: minimise −SR  ≡  maximise SR."""
        return -self.sharpe_ratio(w)

    # ── Core optimisation ─────────────────────────────────────────────────────

    def optimise(self) -> dict:
        """
        Solve the Sharpe-maximisation problem via SLSQP:

            max  SR(w)  =  (wᵀμ − Rf) / sqrt(wᵀΣw)
            s.t. Σᵢ wᵢ = 1          (budget / full-investment constraint)
                 0 ≤ wᵢ ≤ 1         (long-only bounds)

        SLSQP (Sequential Least-Squares Quadratic Programming) handles
        both equality constraints and box bounds simultaneously.
        """
        print("\n[2/5] Running SLSQP optimisation …")

        # Equal-weight initialisation — inside the feasible region
        w0 = np.full(self.n, 1.0 / self.n)

        # Box bounds: each weight in [0, 1]
        bounds = [(0.0, 1.0)] * self.n

        # Equality constraint: Σ wᵢ = 1
        constraints = {
            'type': 'eq',
            'fun' : lambda w: np.sum(w) - 1.0
        }

        result = minimize(
            fun         = self._neg_sharpe,
            x0          = w0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'ftol': 1e-12, 'maxiter': 1_000, 'disp': False},
        )

        if not result.success:
            raise RuntimeError(f"Optimisation failed: {result.message}")

        self.result = result
        raw_w       = result.x

        # ── Post-processing: zero out negligible weights ───────────────────
        cleaned_w = np.where(raw_w < WEIGHT_THRESHOLD, 0.0, raw_w)
        # Re-normalise so weights still sum to exactly 1
        cleaned_w /= cleaned_w.sum()

        self.optimal_weights = cleaned_w
        print("    ✓ Optimisation converged successfully")
        return self._build_summary(cleaned_w)

    # ── Result summary ────────────────────────────────────────────────────────

    def _build_summary(self, w: np.ndarray) -> dict:
        ret = self.portfolio_return(w)
        vol = self.portfolio_volatility(w)
        sr  = self.sharpe_ratio(w)

        allocation = {
            ticker: round(float(weight), 6)
            for ticker, weight in zip(self.tickers, w)
            if weight >= WEIGHT_THRESHOLD
        }

        summary = {
            'expected_annual_return' : ret,
            'annual_volatility'      : vol,
            'sharpe_ratio'           : sr,
            'allocation'             : allocation,
        }
        return summary

    # ── Monte-Carlo efficient frontier (for visualisation) ────────────────────

    def simulate_random_portfolios(self, n: int = N_RANDOM_PORTFOLIOS) -> pd.DataFrame:
        """
        Generate n random long-only portfolios to visualise the feasible set.
        Uses Dirichlet sampling to ensure uniform coverage of the simplex.
        """
        print(f"\n[3/5] Simulating {n:,} random portfolios for frontier plot …")
        records = []
        for _ in range(n):
            w   = np.random.dirichlet(np.ones(self.n))
            ret = self.portfolio_return(w)
            vol = self.portfolio_volatility(w)
            sr  = self.sharpe_ratio(w)
            records.append({'Return': ret, 'Volatility': vol, 'Sharpe': sr})
        print("    ✓ Simulation complete")
        return pd.DataFrame(records)


# =============================================================================
# --- 5. EXECUTION -------------------------------------------------------------
# =============================================================================

optimizer = LongOnlyOptimizer(
    prices         = prices,
    risk_free_rate = RISK_FREE_RATE,
    trading_days   = TRADING_DAYS,
)

summary       = optimizer.optimise()
sim_df        = optimizer.simulate_random_portfolios(N_RANDOM_PORTFOLIOS)
opt_w         = optimizer.optimal_weights
opt_ret       = summary['expected_annual_return']
opt_vol       = summary['annual_volatility']
opt_sr        = summary['sharpe_ratio']
allocation    = summary['allocation']

# ── Pretty-print results ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  OPTIMAL PORTFOLIO — Maximum Sharpe Ratio")
print("=" * 65)
print(f"  Expected Annual Return : {opt_ret * 100:>8.2f}%")
print(f"  Annual Volatility      : {opt_vol * 100:>8.2f}%")
print(f"  Maximum Sharpe Ratio   : {opt_sr:>8.4f}")
print("-" * 65)
print("  Asset Allocation (weights ≥ 0.1%):")
for ticker, weight in sorted(allocation.items(), key=lambda x: -x[1]):
    bar = "█" * int(weight * 40)
    print(f"    {ticker:<18}  {weight * 100:>6.2f}%  {bar}")
print("=" * 65)

# =============================================================================
# --- 6. VISUALISATIONS --------------------------------------------------------
# =============================================================================
print("\n[4/5] Generating visualisations …")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — EDA: Seaborn Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig_hm, ax_hm = plt.subplots(figsize=(9, 7))

mask = np.triu(np.ones_like(optimizer.corr, dtype=bool))   # Upper-triangle mask
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    optimizer.corr,
    mask       = mask,
    cmap       = cmap,
    vmax       = 1.0, vmin = -1.0, center = 0,
    annot      = True, fmt = ".2f", annot_kws = {"size": 10},
    linewidths = 0.5,
    square     = True,
    ax         = ax_hm,
    cbar_kws   = {"shrink": 0.75},
)
ax_hm.set_title(
    "Pairwise Correlation — Daily Returns\n"
    f"(Indian Equities · {START_DATE} → {END_DATE})",
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("    ✓ Correlation heatmap saved → correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 — Plotly: Efficient Frontier + Optimal Portfolio
# ─────────────────────────────────────────────────────────────────────────────
fig_ef = go.Figure()

# Scatter of random portfolios coloured by Sharpe ratio
fig_ef.add_trace(go.Scatter(
    x    = sim_df['Volatility'] * 100,
    y    = sim_df['Return']    * 100,
    mode = 'markers',
    marker = dict(
        color     = sim_df['Sharpe'],
        colorscale= 'Viridis',
        size      = 3,
        opacity   = 0.55,
        colorbar  = dict(title='Sharpe Ratio', thickness=14),
    ),
    name       = 'Random Portfolios',
    hovertemplate =
        'Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>',
))

# Optimal portfolio star
fig_ef.add_trace(go.Scatter(
    x    = [opt_vol * 100],
    y    = [opt_ret * 100],
    mode = 'markers',
    marker = dict(symbol='star', size=20, color='#FF4B4B', line=dict(width=1.5, color='white')),
    name       = f'Max Sharpe ({opt_sr:.3f})',
    hovertemplate =
        f'<b>Optimal Portfolio</b><br>'
        f'Volatility: {opt_vol*100:.2f}%<br>'
        f'Return: {opt_ret*100:.2f}%<br>'
        f'Sharpe: {opt_sr:.4f}<extra></extra>',
))

fig_ef.update_layout(
    title    = dict(text='<b>Efficient Frontier — Monte-Carlo Simulation</b><br>'
                        '<sup>Long-Only Indian Equities Portfolio</sup>',
                    x=0.5),
    xaxis    = dict(title='Annual Volatility (%)', gridcolor='#2a2a3e'),
    yaxis    = dict(title='Expected Annual Return (%)', gridcolor='#2a2a3e'),
    legend   = dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.3)'),
    template = 'plotly_dark',
    width    = 900, height = 560,
)
fig_ef.write_html('efficient_frontier.html')
fig_ef.show()
print("    ✓ Efficient frontier chart saved → efficient_frontier.html")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — Plotly: Optimal Allocation Donut Chart
# ─────────────────────────────────────────────────────────────────────────────
# Build short ticker labels (strip ".NS" suffix)
alloc_labels  = [t.replace('.NS', '') for t in allocation.keys()]
alloc_values  = list(allocation.values())

fig_donut = go.Figure(go.Pie(
    labels       = alloc_labels,
    values       = alloc_values,
    hole         = 0.55,
    textinfo     = 'label+percent',
    hovertemplate= '%{label}<br>Weight: %{value:.2%}<extra></extra>',
    marker       = dict(line=dict(color='#1a1a2e', width=2)),
    pull         = [0.04 if v == max(alloc_values) else 0 for v in alloc_values],
))

fig_donut.update_layout(
    title    = dict(text='<b>Optimal Portfolio Allocation</b><br>'
                        f'<sup>Max Sharpe Ratio = {opt_sr:.4f}  |  '
                        f'E[R] = {opt_ret*100:.2f}%  |  σ = {opt_vol*100:.2f}%</sup>',
                    x=0.5),
    annotations = [dict(
        text=f'<b>{opt_sr:.3f}</b><br>Sharpe', x=0.5, y=0.5,
        font_size=16, showarrow=False, font_color='white'
    )],
    template = 'plotly_dark',
    showlegend = True,
    legend   = dict(orientation='v', x=1.02, y=0.5),
    width = 750, height = 520,
)
fig_donut.write_html('optimal_allocation.html')
fig_donut.show()
print("    ✓ Allocation donut chart saved → optimal_allocation.html")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4 — Plotly: Historical Cumulative Return of Optimal Portfolio ($1 → $?)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Building cumulative performance chart …")

# Portfolio daily return = Σ (wᵢ × rᵢ)   [dot product row-wise]
port_daily_ret  = optimizer.daily_returns.dot(opt_w)   # Series of daily P&L
port_cum_ret    = (1 + port_daily_ret).cumprod()        # $1 grows to ...

# Individual asset cumulative returns (for context)
asset_cum_ret   = (1 + optimizer.daily_returns).cumprod()

fig_perf = go.Figure()

# Individual asset lines (thin, low-opacity)
palette = px.colors.qualitative.Set2
for i, ticker in enumerate(optimizer.tickers):
    short = ticker.replace('.NS', '')
    fig_perf.add_trace(go.Scatter(
        x    = asset_cum_ret.index,
        y    = asset_cum_ret[ticker],
        mode = 'lines',
        name = short,
        line = dict(width=1.2, color=palette[i % len(palette)]),
        opacity   = 0.45,
        hovertemplate = f'{short}<br>Date: %{{x|%d %b %Y}}<br>Value: $%{{y:.3f}}<extra></extra>',
    ))

# Optimal portfolio — thick highlight
fig_perf.add_trace(go.Scatter(
    x    = port_cum_ret.index,
    y    = port_cum_ret.values,
    mode = 'lines',
    name = '⭐ Optimal Portfolio',
    line = dict(width=3.5, color='#FF4B4B'),
    hovertemplate = 'Optimal Portfolio<br>Date: %{x|%d %b %Y}<br>Value: $%{y:.3f}<extra></extra>',
))

# $1 baseline
fig_perf.add_hline(
    y=1.0, line_dash='dot', line_color='grey', line_width=1.2,
    annotation_text='$1 Invested', annotation_position='top left'
)

final_val = port_cum_ret.iloc[-1]
fig_perf.update_layout(
    title = dict(
        text = (f'<b>Historical Cumulative Performance — $1 Invested</b><br>'
                f'<sup>Optimal Portfolio  |  '
                f'Final Value: <b>${final_val:.2f}</b>  |  '
                f'Total Return: <b>{(final_val-1)*100:.1f}%</b>  |  '
                f'{START_DATE} → {END_DATE}</sup>'),
        x = 0.5
    ),
    xaxis    = dict(title='Date', gridcolor='#2a2a3e', rangeslider=dict(visible=True)),
    yaxis    = dict(title='Portfolio Value (USD)', gridcolor='#2a2a3e'),
    legend   = dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.35)', font=dict(size=11)),
    template = 'plotly_dark',
    hovermode= 'x unified',
    width    = 1050, height = 580,
)
fig_perf.write_html('cumulative_performance.html')
fig_perf.show()
print("    ✓ Performance chart saved → cumulative_performance.html")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  All outputs generated successfully.")
print("  Files: correlation_heatmap.png | efficient_frontier.html")
print("         optimal_allocation.html | cumulative_performance.html")
print("=" * 65)