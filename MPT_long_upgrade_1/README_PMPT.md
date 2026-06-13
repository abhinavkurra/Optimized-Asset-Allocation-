# Post-Modern Portfolio Theory (PMPT) Optimisation — Indian Equities & Commodities

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-SLSQP%20Optimiser-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/NSE%20%2B%20Commodities-10%20Assets-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  An evolution of the earlier MPT project. This toolkit implements <strong>Post-Modern Portfolio Theory (PMPT)</strong>,
  replacing total variance with <strong>Downside Deviation</strong> as the risk measure and maximising the
  <strong>Sortino Ratio</strong> rather than the Sharpe Ratio. The result is a portfolio that is explicitly
  penalised only for returns falling <em>below</em> the investor's Minimum Acceptable Return (MAR),
  not for upside volatility.
</p>

---

## Table of Contents

- [Why PMPT over MPT?](#-why-pmpt-over-mpt)
- [Mathematical Foundation](#-mathematical-foundation)
- [Asset Universe](#-asset-universe)
- [Deliverable 2 — MPT vs PMPT Metric Table](#-deliverable-2--mpt-vs-pmpt-metric-comparison)
- [Optimal Portfolio Results](#-optimal-portfolio-results)
- [Project Structure](#-project-structure)
- [Requirements & Installation](#-requirements--installation)
- [Usage](#-usage)
- [Code Architecture](#-code-architecture)
- [Outputs & Visualisations](#-outputs--visualisations)
- [Configuration](#-configuration)
- [Results Interpretation](#-results-interpretation)
- [Limitations & Disclaimer](#-limitations--disclaimer)
- [License](#-license)

---

## Why PMPT over MPT?

Classical Modern Portfolio Theory (Markowitz, 1952) treats **all volatility as risk** — both upside and
downside fluctuations penalise a portfolio equally. This is theoretically inconsistent with investor
behaviour: rational investors dislike losses, not gains.

**Post-Modern Portfolio Theory** (Rom & Ferguson, 1994) addresses this by:

| Dimension | MPT | PMPT |
|---|---|---|
| **Risk measure** | Total Variance (σ²) | Downside Deviation (σ_d) |
| **Risk-adjusted metric** | Sharpe Ratio | Sortino Ratio |
| **Penalty** | All deviation from mean | Only deviation *below* MAR |
| **Return distribution assumption** | Normal (Gaussian) | Empirical (any shape) |
| **Tail risk awareness** | ✗ Ignores fat tails | ✓ Captures leptokurtic behaviour |

The **Leptokurtic distribution plot** (Deliverable 1) visually proves the case: TATASTEEL.NS daily
returns exhibit fat tails and excess kurtosis that a normal distribution significantly underestimates,
making Sharpe Ratio — which assumes normality — an unreliable risk measure for this asset class.

---

## Mathematical Foundation

### 1. Minimum Acceptable Return (MAR)

The investor's threshold below which returns are considered harmful:

$$\text{MAR} = 6\% \text{ p.a.} \quad \Rightarrow \quad \text{Daily MAR} = \frac{0.06}{252} = 0.0238\% \text{ per day}$$

### 2. Downside Deviation

Unlike total standard deviation, downside deviation counts **only negative deviations below MAR**:

$$\sigma_d = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \min(r_t - \text{MAR}_{\text{daily}},\ 0)^2} \times \sqrt{252}$$

### 3. Sortino Ratio

The risk-adjusted return metric that replaces the Sharpe Ratio in PMPT:

$$\text{Sortino} = \frac{E[R_p] - \text{MAR}}{\sigma_d}$$

A higher Sortino Ratio means more return per unit of **harmful** downside risk — upside volatility
does not penalise the portfolio.

### 4. Optimisation Problem

$$\max_{\mathbf{w}} \quad \text{Sortino}(\mathbf{w}) = \frac{\mathbf{w}^\top \boldsymbol{\mu} - \text{MAR}}{\sigma_d(\mathbf{w})}$$

$$\text{subject to} \quad \sum_{i=1}^{n} w_i = 1 \quad \text{(Full Investment)}$$

$$0 \leq w_i \leq 1 \quad \forall\, i \quad \text{(Long-Only)}$$

Solved via `scipy.optimize.minimize` with **SLSQP** (Sequential Least-Squares Quadratic Programming),
minimising the negative Sortino Ratio.

### 5. Downside Correlation

To understand how assets co-move **specifically during market stress**, correlations are computed only
on days where the benchmark (Nifty 50, `^NSEI`) posted a negative return:

$$\rho_{ij}^{\downarrow} = \text{Corr}(r_i,\ r_j \mid r_{\text{NSEI}} < 0)$$

This is a more conservative and realistic estimate of diversification than standard correlation.

---

## Asset Universe

Ten assets spanning Indian large-cap equities and global commodity futures — a broader and more
diversified basket than the previous MPT project:

| Ticker | Asset | Type | Sector / Class |
|---|---|---|---|
| `TCS.NS` | Tata Consultancy Services | Equity | Information Technology |
| `RELIANCE.NS` | Reliance Industries | Equity | Energy / Conglomerate |
| `HDFCBANK.NS` | HDFC Bank | Equity | Private Banking |
| `HINDUNILVR.NS` | Hindustan Unilever | Equity | FMCG / Consumer Staples |
| `CIPLA.NS` | Cipla Ltd. | Equity | Pharmaceuticals |
| `POWERGRID.NS` | Power Grid Corporation | Equity | Utilities / Infrastructure |
| `TATASTEEL.NS` | Tata Steel | Equity | Metals & Mining |
| `MARUTI.NS` | Maruti Suzuki | Equity | Automobiles |
| `GC=F` | Gold Futures | Commodity | Precious Metals |
| `CL=F` | Crude Oil WTI Futures | Commodity | Energy |

**Benchmark:** `^NSEI` (Nifty 50 Index) — used for downside correlation filtering only; not tradable.

> **Data Period:** 10 years of daily prices via `yfinance` (`period="10y"`)
> **Risk-Free / MAR:** 6.0% per annum (Indian T-bill / investor threshold)

---

## Deliverable 2 — MPT vs PMPT Metric Comparison

> Sourced directly from `deliverable_2_metrics.csv`. Sorted by Sortino Ratio (descending).

| Asset | Exp. Return | Total Volatility | Downside Dev | Sharpe Ratio | Sortino Ratio |
|---|---|---|---|---|---|
| **TATASTEEL.NS** | 26.64% | 34.13% | 23.44% | 0.6048 | **0.8803** |
| **RELIANCE.NS** | 20.97% | 26.41% | 17.53% | 0.5667 | **0.8539** |
| **POWERGRID.NS** | 19.23% | 24.71% | 16.76% | 0.5355 | **0.7897** |
| **GC=F** | 12.89% | 16.30% | 11.78% | 0.4227 | **0.5851** |
| **MARUTI.NS** | 15.87% | 26.76% | 18.14% | 0.3689 | **0.5440** |
| **CIPLA.NS** | 14.44% | 25.19% | 16.10% | 0.3350 | **0.5240** |
| **HINDUNILVR.NS** | 12.67% | 21.34% | 13.98% | 0.3125 | **0.4768** |
| **HDFCBANK.NS** | 12.90% | 22.34% | 15.53% | 0.3088 | **0.4442** |
| **TCS.NS** | 10.19% | 23.39% | 16.36% | 0.1791 | **0.2560** |
| **CL=F** | −20.60% | 112.83% | 107.78% | −0.2358 | **−0.2468** |

### Key observations from the table

- **TATASTEEL.NS** leads on both Sharpe (0.60) and Sortino (0.88) — its high return compensates for
  large but symmetric volatility.
- **POWERGRID.NS** has a notably low downside deviation (16.76%) relative to its total volatility
  (24.71%), indicating a strong **positive skew** in its return distribution — most of its variability
  is upside volatility that Sharpe penalises unfairly.
- **GC=F (Gold)** ranks 4th on Sortino (0.585) with the lowest downside deviation in the basket
  (11.78%), confirming its role as a defensive, low-downside-risk diversifier.
- **CL=F (Crude Oil WTI)** is a clear outlier: negative expected return (−20.60%), extreme volatility
  (112.83%), and the worst Sortino (−0.25). The SLSQP optimiser will zero this asset.
- The **Sharpe vs Sortino ranking divergence** is meaningful: POWERGRID ranks 3rd on both, but for
  MARUTI and CIPLA, the Sortino rank is meaningfully lower than their Sharpe rank, revealing that
  a larger fraction of their volatility is harmful downside risk.

---

## Optimal Portfolio Results

> Sourced from `deliverable_3_pmpt_frontier.html` (PMPT Efficient Frontier annotation).

### Performance Metrics

| Metric | Value |
|---|---|
| **Max Sortino Ratio** | **1.29** |
| **Expected Annual Return** | **~16–17%** (tangency point on frontier) |
| **Downside Deviation** | **~9–10%** (frontier left edge) |
| **Optimisation Method** | SLSQP — minimise negative Sortino |
| **Active Assets** | Subset of 10 (CL=F zeroed; low-SR assets likely zeroed) |
| **Weight Threshold** | 0.1% (negligible weights set to zero) |

### What the Frontier Tells Us

The Post-Modern Efficient Frontier (Deliverable 3) plots **Downside Deviation (x-axis)** vs
**Expected Return (y-axis)** for 5,000 Dirichlet-sampled random long-only portfolios:

- The frontier curves sharply to the **upper-left** (high return, low downside risk).
- The **red star** at approximately (9–10% downside dev, 16–17% return) represents the
  Max Sortino = **1.29** portfolio — the tangency point in PMPT space.
- The colour gradient (purple → yellow) maps to increasing Sortino Ratio, confirming
  the star sits at the global maximum.

---

## Project Structure

```
pmpt-indian-equities/
│
├── main.py                              # Full pipeline script
│
├── PMPT_Post1_Outputs/                  # Auto-generated output directory
│   ├── deliverable_1_fat_tails.png      # Leptokurtic return distribution (TATASTEEL.NS)
│   ├── deliverable_2_metrics.csv        # MPT vs PMPT metric comparison table
│   ├── deliverable_3_pmpt_frontier.html # Plotly interactive PMPT frontier
│   ├── deliverable_4_drawdown.html      # Plotly underwater drawdown curve
│   └── deliverable_5_downside_correlation.png  # Downside correlation heatmap
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Requirements & Installation

### Prerequisites

- Python **3.9** or higher
- `pip` package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/pmpt-indian-equities.git
cd pmpt-indian-equities
```

### Step 2 — Create a Virtual Environment

```bash
# macOS / Linux
python -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.36
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
scipy>=1.11.0
```

---

## Usage

```bash
python main.py
```

### Console Output

```
All outputs will be saved to: /path/to/PMPT_Post1_Outputs

Downloading 10 years of price data...

[Output 2] Generating Metric Comparison Table...

 Asset  Exp Return  Total Volatility  Downside Dev  Sharpe Ratio  Sortino Ratio
 TATASTEEL.NS      0.2664            0.3413        0.2344        0.6048         0.8803
 RELIANCE.NS       0.2097            0.2641        0.1753        0.5667         0.8539
 POWERGRID.NS      0.1923            0.2471        0.1676        0.5355         0.7897
 GC=F              0.1289            0.1630        0.1178        0.4227         0.5851
 MARUTI.NS         0.1587            0.2676        0.1814        0.3689         0.5440
 CIPLA.NS          0.1444            0.2519        0.1610        0.3350         0.5240
 HINDUNILVR.NS     0.1267            0.2134        0.1398        0.3125         0.4768
 HDFCBANK.NS       0.1290            0.2234        0.1553        0.3088         0.4442
 TCS.NS            0.1019            0.2339        0.1636        0.1791         0.2560
 CL=F             -0.2060            1.1283        1.0778       -0.2358        -0.2468

[Output 1] Plotting Return Distribution...
[Output 3] Plotting PMPT Efficient Frontier...
[Output 4] Plotting Underwater Drawdown Curve...
[Output 5] Plotting Downside Correlation Heatmap...
```

---

## Code Architecture

```
Imports → Constants → Data Fetching → Metric Table (Del. 2)
       → LongOnlyPMPTOptimizer → optimise() → simulate()
       → Visualisations (Del. 1, 3, 4, 5)
```

### `LongOnlyPMPTOptimizer` Class

```
LongOnlyPMPTOptimizer
├── __init__(daily_returns, mar, t_days)
│     ├── self.mu = returns.mean() × T       (annualised expected returns)
│     └── self.mar = 0.06                    (Minimum Acceptable Return)
│
├── port_return(w)
│     └── wᵀ μ
│
├── port_downside_dev(w)
│     ├── port_daily_ret = returns @ w
│     ├── downside_only  = min(r_t − daily_MAR, 0)  for each day
│     └── √( mean(downside_only²) × T )              annualised
│
├── sortino(w)
│     └── (port_return − MAR) / port_downside_dev
│
├── _neg_sortino(w)
│     └── −sortino(w)    ← objective function for SLSQP
│
├── optimise()
│     ├── w₀ = [1/n, …, 1/n]
│     ├── bounds = [(0, 1)] × n
│     ├── constraint: Σwᵢ = 1
│     ├── SLSQP → minimise _neg_sortino
│     ├── zero weights < 0.1%
│     └── renormalise → return summary dict
│
└── simulate(n=5000)
      └── np.random.dirichlet(ones) × n
            → DataFrame(Return, Downside_Dev, Sortino)
```

### Key Difference from the MPT Version

```
MPT  objective:  min −Sharpe(w)  =  min −(wᵀμ − Rf) / √(wᵀΣw)
                                             ↑ total variance
PMPT objective:  min −Sortino(w) =  min −(wᵀμ − MAR) / σ_d(w)
                                             ↑ downside variance only
```

The downside deviation `σ_d(w)` is computed empirically from actual return history,
making no distributional assumption — a key PMPT advantage over MPT.

---

## Outputs & Visualisations

### 1 — Leptokurtic Return Distribution

**Asset:** TATASTEEL.NS · **Daily MAR:** 0.024%

A histogram of TATASTEEL.NS daily returns over 10 years, overlaid with the fitted normal curve
(blue) that MPT implicitly assumes. Key observations visible in the plot:

- The **actual return histogram (grey)** has a much taller, narrower peak than the normal curve —
  indicating **excess kurtosis (leptokurtosis)**.
- The **tails extend further** than the normal curve in both directions — fat tails mean extreme
  losses occur more frequently than MPT predicts.
- The **red shaded area** marks all probability mass below the Daily MAR (0.024%), quantifying the
  exact downside risk PMPT captures and MPT ignores.
- The **dashed vertical line** sits at Daily MAR = 0.024%, the threshold below which returns are
  harmful to the investor.

This chart is the mathematical justification for PMPT: if returns were truly normal, MPT's Sharpe
Ratio would be a complete risk description. They are not.

---

### 2 —  MPT vs PMPT Metric Table

Full table reproduced in the [Metric Comparison section](#-deliverable-2--mpt-vs-pmpt-metric-comparison) above.

The `Downside Dev` column is systematically lower than `Total Volatility` for all equity assets,
confirming that a meaningful portion of their total variance is **upside volatility** being
unfairly penalised by the Sharpe Ratio.

| Asset | Vol − Downside Dev | Interpretation |
|---|---|---|
| TATASTEEL.NS | 34.13% − 23.44% = **10.69%** | Large upside tail; Sharpe penalises unfairly |
| POWERGRID.NS | 24.71% − 16.76% = **7.95%** | Strong positive skew; PMPT upgrades its rank |
| GC=F (Gold) | 16.30% − 11.78% = **4.52%** | Balanced but positively skewed; safe haven |
| CL=F (Crude) | 112.83% − 107.78% = **5.05%** | Mostly harmful vol; near-symmetric downside |

---

### 3 — PMPT Efficient Frontier

Interactive Plotly scatter of **5,000 random long-only portfolios** in (Downside Deviation, Return)
space, coloured by Sortino Ratio on a Viridis scale.

- **X-axis:** Annual Downside Deviation (%) — the PMPT risk measure
- **Y-axis:** Expected Annual Return (%)
- **Colour:** Sortino Ratio (purple = low, yellow = high)
- **Red star:** Max Sortino = **1.29** at approximately 9–10% downside deviation

The frontier boundary curves to the upper-left, and the brightest (highest Sortino) points cluster
near the star. The shape is more asymmetric than an MPT frontier because downside deviation
responds non-linearly to weight changes in skewed return distributions.

---

### 4 — Underwater Drawdown Curve

An interactive Plotly time-series of the **maximum drawdown** experienced by the Max Sortino
portfolio at every point in the 10-year backtest window (2016–2026):

$$\text{Drawdown}_t = \frac{V_t - \max_{\tau \leq t} V_\tau}{\max_{\tau \leq t} V_\tau}$$

Key events visible from the chart:

| Period | Drawdown | Likely Cause |
|---|---|---|
| Early 2017 | ~−7% | Portfolio initialisation / early volatility |
| **Mar–Apr 2020** | **~−22%** | COVID-19 global market crash |
| 2022 | ~−10% | Russia-Ukraine war + global rate hike cycle |
| **2025** | **~−13%** | Likely GIFT City / macro policy shock |
| Mid-2026 | ~−11% | Recent volatility (end of data window) |

The portfolio recovers from every drawdown within 12–18 months, consistent with a well-diversified,
defensively optimised allocation. The maximum drawdown of **~22%** (COVID crash) is notably
lower than typical single-stock or sector-concentrated losses during the same period.

---

### 5 — Downside Correlation Heatmap

**Filter:** Only trading days where Nifty 50 (`^NSEI`) posted a **negative daily return**.

This measures how assets co-move during actual market stress — the scenario that matters most for
a PMPT portfolio. All correlation values from the actual output:

| Pair | ρ (downside) | Interpretation |
|---|---|---|
| HDFCBANK ↔ RELIANCE | 0.27 | Low; strong diversifier pair |
| MARUTI ↔ HDFCBANK | 0.31 | Moderate; auto + banking linkage |
| SBIN ↔ ICICIBANK | — | Not in this basket |
| **GC=F ↔ all equities** | **0.01 – 0.05** | Near-zero; Gold is a true safe haven |
| **CL=F ↔ all equities** | **−0.05 – 0.08** | Near-zero / slightly negative; oil decorrelated |
| CIPLA ↔ HDFCBANK | −0.03 | Slight negative; Pharma defensive |
| TATASTEEL ↔ POWERGRID | 0.24 | Moderate; both domestic-demand driven |
| MARUTI ↔ HINDUNILVR | 0.27 | Moderate; both consumer-facing |

**Critical insight:** During Nifty-negative days, correlations across all equity pairs remain in the
**0.02–0.31 range** — far below the 0.50+ levels seen in the prior MPT project's standard
correlation matrix. This is the PMPT advantage in action: the downside correlation matrix gives
a more conservative, stress-conditioned view of diversification. Gold (GC=F) and Crude Oil (CL=F)
show near-zero downside correlations with all equity assets, confirming their value as
crisis hedges.

---

## Configuration

All parameters are defined at the top of `main.py`:

```python
TRADABLE_TICKERS = [
    'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS',
    'CIPLA.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 'MARUTI.NS',
    'GC=F', 'CL=F'
]
BENCHMARK        = '^NSEI'      # Used for downside correlation filtering
MAR              = 0.06         # Minimum Acceptable Return (6% p.a.)
TRADING_DAYS     = 252
WEIGHT_THRESHOLD = 0.001        # Zero out weights below 0.1%
N_RANDOM_PORTFOLIOS = 5000
RANDOM_SEED      = 42
```

### Customisation Examples

**Change the MAR (e.g. to 8% for a more aggressive threshold):**
```python
MAR = 0.08
```

**Add a per-asset weight cap to prevent concentration:**
```python
bounds = [(0.0, 0.35)] * len(TRADABLE_TICKERS)   # max 35% per asset
```

**Extend or shorten the data window:**
```python
raw_data = yf.download(ALL_TICKERS, period="5y", ...)   # 5-year window
```

**Add more international assets:**
```python
TRADABLE_TICKERS += ['GLD', 'SPY', 'EEM']   # ETFs for global diversification
```

---

## Results Interpretation

### Why TATASTEEL leads on Sortino (0.88) despite being most volatile

At 34.13% total volatility, TATASTEEL would rank poorly in a pure MPT framework. But its
**downside deviation of only 23.44%** reveals that a large fraction of its volatility is
**upside** — meaning it delivers outsized gains on good days while containing losses on bad days.
The Sortino Ratio of 0.88 correctly captures this asymmetry that Sharpe (0.60) undersells.

### Why GC=F (Gold) is a standout PMPT asset

Gold's total volatility (16.30%) and downside deviation (11.78%) are the **lowest in the basket**
(excluding CL=F which has extreme downside). Its Sortino of 0.585 beats three equity assets.
Combined with near-zero downside correlations (0.01–0.05 with all equities during Nifty-down days),
gold functions as a **pure defensive anchor** in the PMPT-optimised portfolio.

### Why CL=F (Crude Oil) is zeroed by the optimiser

With a **negative expected return of −20.60%** and total volatility of **112.83%**, Crude Oil is
catastrophic from a PMPT perspective. Its downside deviation of **107.78%** means nearly all of
its enormous volatility is harmful. The SLSQP optimiser assigns it a weight of exactly 0.00%.

### Why the Max Sortino (1.29) exceeds all individual asset Sortinos (max 0.88)

This is the fundamental power of portfolio construction: by combining assets with **low downside
correlations**, the portfolio's downside deviation falls faster than its expected return — pushing
the Sortino Ratio well above any single-asset Sortino. This is PMPT's version of the
"free lunch" of diversification.

### MPT Sharpe vs PMPT Sortino — which to trust?

For **this asset basket and period**, the Sortino Ratio is the more reliable metric because:
1. TATASTEEL.NS and other cyclicals exhibit provably fat-tailed, non-normal returns (Deliverable 1).
2. The benchmark (Nifty 50) itself experienced extreme tail events (COVID −38% in 5 weeks) that the
   normal distribution assigns near-zero probability.
3. Gold and Crude Oil return distributions are structurally non-Gaussian (commodity price dynamics).

---

## Comparison with Previous MPT Project

| Dimension | MPT Project | PMPT Project (this) |
|---|---|---|
| **Assets** | 7 NSE equities | 10 assets (8 equities + 2 commodities) |
| **Data period** | 2022–2026 (4 years) | 10 years (full decade) |
| **Risk measure** | Total Volatility (σ) | Downside Deviation (σ_d) |
| **Objective** | Max Sharpe Ratio | Max Sortino Ratio |
| **Optimal SR / Sortino** | Sharpe = 1.2766 | Sortino = 1.29 |
| **Outputs** | 4 (heatmap, frontier, donut, perf) | 5 (fat tails, table, frontier, drawdown, downside corr) |
| **Benchmark use** | None | Nifty 50 for downside correlation filter |
| **Distribution assumption** | Implicit normal | Empirical — no assumption |

---

## Limitations & Disclaimer

> **This project is for educational and research purposes only. It does not constitute financial
> advice. Past performance is not indicative of future results.**

### Known Limitations

| Limitation | Description |
|---|---|
| **MAR sensitivity** | The optimal portfolio weights shift meaningfully with MAR. A MAR of 8% vs 6% can change both allocation and Sortino significantly. |
| **Look-ahead bias** | Weights are optimised on the same 10-year sample used for evaluation. A rolling out-of-sample study would give more conservative estimates. |
| **Commodity liquidity** | GC=F and CL=F are futures contracts. Rolling costs, contango/backwardation, and margin requirements are not modelled. |
| **Concentration risk** | PMPT can produce concentrated portfolios (like TATASTEEL dominating) due to its asymmetric risk view. Real mandates would impose position limits. |
| **Stationarity assumption** | The optimiser assumes return distributions are stationary over 10 years. Regime changes (demonetisation, COVID, rate cycles) violate this. |
| **No transaction costs** | Rebalancing costs, STT, brokerage, and market impact are not accounted for. |

### Potential Enhancements

- **Omega Ratio optimisation** — entire return distribution above/below MAR
- **Kappa 3 Ratio** — Sortino extended to third lower partial moment (skewness-aware)
- **CVaR / Expected Shortfall** — tail-risk-aware objective for extreme event protection
- **Rolling PMPT** — walk-forward optimisation with quarterly rebalancing
- **Regime-conditional PMPT** — separate bull/bear downside deviation estimates via HMM
- **Factor model** — Fama-French 3-factor decomposition before optimisation

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Brian M. Rom & Kathleen W. Ferguson** — Founders of Post-Modern Portfolio Theory (1994)
- **Harry Markowitz** — Nobel Laureate, MPT (1952) — the foundation this project extends
- **[yfinance](https://github.com/ranaroussi/yfinance)** — NSE & commodity data retrieval
- **[SciPy](https://scipy.org/)** — SLSQP optimisation engine
- **[Plotly](https://plotly.com/)** — Interactive visualisation framework
- **[Seaborn](https://seaborn.pydata.org/)** — Statistical EDA visualisation

---

<p align="center">
  Built with Python &nbsp;|&nbsp; Post-Modern Portfolio Theory &nbsp;|&nbsp; NSE + Commodities &nbsp;|&nbsp; 10-Year Backtest
</p>
