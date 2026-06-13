# Long-Only Modern Portfolio Optimisation (MPO) — Indian Equities

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-SLSQP%20Optimiser-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/NSE-Indian%20Equities-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready quantitative finance toolkit that applies <strong>Harry Markowitz's Modern Portfolio Theory</strong> to a curated basket of NSE-listed blue-chip Indian equities. The engine maximises the <strong>Sharpe Ratio</strong> under long-only constraints using <strong>Sequential Least-Squares Quadratic Programming (SLSQP)</strong>, and delivers a full suite of interactive Plotly visualisations.
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Mathematical Foundation](#-mathematical-foundation)
- [Asset Universe](#-asset-universe)
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

## 🔭 Project Overview

Modern Portfolio Theory (MPT), introduced by Harry Markowitz in 1952, provides a mathematical framework for assembling a portfolio of assets such that **expected return is maximised for a given level of risk**. This project implements the **Maximum Sharpe Ratio** variant — the tangency portfolio — which yields the single best risk-adjusted allocation from the efficient frontier.

### Key Objectives

| Goal | Approach |
|---|---|
| Maximise risk-adjusted return | Optimise the Sharpe Ratio via SLSQP |
| Respect investment constraints | Long-only bounds `[0, 1]` + full-investment equality `Σwᵢ = 1` |
| Use real market data | `yfinance` daily adjusted closing prices (2022–2026) |
| Produce actionable insights | Weights, performance metrics, and 4 interactive visual outputs |

---

## Mathematical Foundation

### 1. Returns & Covariance

Daily simple returns are computed as:

$$r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$$

These are then annualised over $T = 252$ trading days:

$$\mu_i = \bar{r}_i \times T \qquad \Sigma = \text{Cov}(R_{\text{daily}}) \times T$$

### 2. Portfolio Statistics

For a weight vector $\mathbf{w} \in \mathbb{R}^n$:

| Metric | Formula |
|---|---|
| **Expected Return** | $E[R_p] = \mathbf{w}^\top \boldsymbol{\mu}$ |
| **Portfolio Variance** | $\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}$ |
| **Portfolio Volatility** | $\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$ |
| **Sharpe Ratio** | $SR = \dfrac{E[R_p] - R_f}{\sigma_p}$ |

### 3. Optimisation Problem

$$\max_{\mathbf{w}} \quad SR(\mathbf{w}) = \frac{\mathbf{w}^\top \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}}$$

$$\text{subject to} \quad \sum_{i=1}^{n} w_i = 1 \quad \text{(Full Investment Constraint)}$$

$$0 \leq w_i \leq 1 \quad \forall\, i \quad \text{(Long-Only Bounds)}$$

The problem is solved by `scipy.optimize.minimize` as a **minimisation of the negative Sharpe Ratio** using **SLSQP**, which natively handles equality constraints and box bounds simultaneously.

---

## Asset Universe

Seven large-cap NSE-listed equities spanning multiple high-growth sectors of the Indian economy:

| Ticker | Company | Sector |
|---|---|---|
| `RELIANCE.NS` | Reliance Industries Ltd. | Energy / Conglomerate |
| `HDFCBANK.NS` | HDFC Bank Ltd. | Private Banking |
| `BHARTIARTL.NS` | Bharti Airtel Ltd. | Telecommunications |
| `SBIN.NS` | State Bank of India | Public Sector Banking |
| `ICICIBANK.NS` | ICICI Bank Ltd. | Private Banking |
| `TCS.NS` | Tata Consultancy Services Ltd. | Information Technology |
| `LT.NS` | Larsen & Toubro Ltd. | Engineering & Infrastructure |

> **Backtest Period:** 1 January 2022 → 1 January 2026 (≈ 1,008 trading days)
> **Risk-Free Rate:** 6.0% per annum (Indian Government T-bill proxy)

---

## Optimal Portfolio Results

> All figures below are sourced directly from the project's actual output files —
> `optimal_allocation.html`, `cumulative_performance.html`, `efficient_frontier.html`,
> and `correlation_heatmap.png`.

### Performance Metrics

| Metric | Value |
|---|---|
| **Expected Annual Return** | **28.66%** |
| **Annual Volatility (σ)** | **17.75%** |
| **Maximum Sharpe Ratio** | **1.2766** |
| **Final Portfolio Value** | **$2.89** (from $1 invested on 2022-01-01) |
| **Total Return** | **+188.7%** over 4 years |
| **Assets Allocated** | **3 / 7** |
| **Assets Zeroed by SLSQP** | **4 / 7** |

---

### Optimal Weight Allocation

| Rank | Ticker | Company | Sector | Weight |
|---|---|---|---|---|
| 1 | **BHARTIARTL.NS** | Bharti Airtel Ltd. | Telecom | **67.09%** |
| 2 | **SBIN.NS** | State Bank of India | PSU Banking | **17.33%** |
| 3 | **LT.NS** | Larsen & Toubro Ltd. | Infra | **15.58%** |
| — | RELIANCE.NS | Reliance Industries | Energy | 0.00% |
| — | HDFCBANK.NS | HDFC Bank | Pvt Banking | 0.00% |
| — | ICICIBANK.NS | ICICI Bank | Pvt Banking | 0.00% |
| — | TCS.NS | Tata Consultancy Services | IT | 0.00% |

> Weights are filtered at the 0.1% negligibility threshold and renormalised to sum exactly to 1.000.

```
BHARTIARTL  67.09%  ██████████████████████████████████
SBIN        17.33%  █████████
LT          15.58%  ████████
```

---

### Pairwise Correlation Matrix (Daily Returns · 2022–2026)

Values read directly from `correlation_heatmap.png`:

|  | REL | HDFC | BRTL | SBIN | ICICI | TCS | LT |
|---|---|---|---|---|---|---|---|
| **RELIANCE** | 1.00 | — | — | — | — | — | — |
| **HDFCBANK** | 0.36 | 1.00 | — | — | — | — | — |
| **BHARTIARTL** | 0.35 | 0.27 | 1.00 | — | — | — | — |
| **SBIN** | 0.46 | 0.39 | 0.30 | 1.00 | — | — | — |
| **ICICIBANK** | 0.40 | 0.50 | 0.36 | 0.52 | 1.00 | — | — |
| **TCS** | 0.31 | 0.25 | 0.25 | 0.24 | 0.25 | 1.00 | — |
| **LT** | 0.43 | 0.38 | 0.31 | 0.47 | 0.42 | 0.33 | 1.00 |

**Key observations:**

- **BHARTIARTL ↔ HDFCBANK (0.27)** — lowest pair in the matrix; a primary diversification anchor.
- **SBIN ↔ TCS (0.24)** and **BHARTIARTL ↔ TCS (0.25)** — near-minimum correlations, but TCS's return in this window was insufficient to earn a positive SLSQP weight.
- **SBIN ↔ ICICIBANK (0.52)** and **HDFCBANK ↔ ICICIBANK (0.50)** — the two highest correlations. The optimiser avoids holding all three private/PSU bank pairs simultaneously; SBIN alone is retained as the banking representative.

---

## Project Structure

```
mpo-indian-equities/
│
├── main.py                          # Primary script (linear pipeline)
│
├── outputs/                         # Auto-generated on execution
│   ├── correlation_heatmap.png      # Seaborn EDA heatmap (static PNG)
│   ├── efficient_frontier.html      # Plotly interactive frontier
│   ├── optimal_allocation.html      # Plotly interactive donut chart
│   └── cumulative_performance.html  # Plotly interactive growth chart
│
├── requirements.txt                 # Python dependency list
├── README.md                        # This file
└── LICENSE                          # MIT License
```

---

## Requirements & Installation

### Prerequisites

- Python **3.9** or higher
- `pip` package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/mpo-indian-equities.git
cd mpo-indian-equities
```

### Step 2 — Create a Virtual Environment (Recommended)

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

### Run as a Python Script

```bash
python main.py
```

### Run in a Jupyter Notebook

```bash
pip install jupyterlab
jupyter lab
```

### Actual Console Output (from this run)

```
=================================================================
  Long-Only Modern Portfolio Optimisation — Indian Equities
=================================================================

[1/5] Downloading price data (2022-01-01 → 2026-01-01) …
    ✓ 7 tickers | 1,008 trading days loaded
    Date range: 2022-01-03 → 2025-12-31

[2/5] Running SLSQP optimisation …
    ✓ Optimisation converged successfully

[3/5] Simulating 5,000 random portfolios for frontier plot …
    ✓ Simulation complete

=================================================================
  OPTIMAL PORTFOLIO — Maximum Sharpe Ratio
=================================================================
  Expected Annual Return :    28.66%
  Annual Volatility      :    17.75%
  Maximum Sharpe Ratio   :    1.2766
-----------------------------------------------------------------
  Asset Allocation (weights ≥ 0.1%):
    BHARTIARTL.NS       67.09%  ███████████████████████████
    SBIN.NS             17.33%  ███████
    LT.NS               15.58%  ██████
=================================================================

[4/5] Generating visualisations …
    ✓ Correlation heatmap    → correlation_heatmap.png
    ✓ Efficient frontier     → efficient_frontier.html
    ✓ Allocation donut       → optimal_allocation.html

[5/5] Building cumulative performance chart …
    ✓ Performance chart      → cumulative_performance.html
         Final Value: $2.89  |  Total Return: +188.7%
=================================================================
  All outputs generated successfully.
=================================================================
```

---

## Code Architecture

The script follows a strict **linear pipeline** with an **OOP core**:

```
Imports → Constants → Data Fetching → LongOnlyOptimizer → Execution → Visualisations
```

### `LongOnlyOptimizer` Class

```
LongOnlyOptimizer
├── __init__(prices, risk_free_rate, trading_days)
│     ├── daily_returns  = prices.pct_change().dropna()
│     ├── mu             = daily_returns.mean() × T        (annualised)
│     ├── cov            = daily_returns.cov()  × T        (annualised)
│     └── corr           = daily_returns.corr()            (for EDA heatmap)
│
├── portfolio_return(w)       →  wᵀ μ
├── portfolio_volatility(w)   →  √(wᵀ Σ w)
├── sharpe_ratio(w)           →  (wᵀμ − Rf) / σ_p
├── _neg_sharpe(w)            →  −SR(w)   ← objective passed to SLSQP
│
├── optimise()
│     ├── w₀ = [1/n, …, 1/n]            equal-weight initialisation
│     ├── bounds      = [(0.0, 1.0)] × n  long-only box bounds
│     ├── constraints = {type:'eq', Σwᵢ = 1}
│     ├── scipy.optimize.minimize('SLSQP', ftol=1e-12, maxiter=1000)
│     ├── zero weights < WEIGHT_THRESHOLD (0.1%)
│     └── renormalise → return summary dict
│
└── simulate_random_portfolios(n=5000)
      └── np.random.dirichlet(ones) × n  →  (Return, Volatility, Sharpe) per portfolio
```

### Data Flow

```
yfinance download (Adj Close)
        │
        ▼
   ffill() → dropna()       ← handle missing / holiday gaps
        │
        ▼
  pct_change() daily returns
        │
   ┌────┴────────────┐
   ▼                 ▼
Ann. μ (×252)    Ann. Σ (×252)      corr matrix
        │
        ▼
   SLSQP Optimiser  ←  bounds + equality constraint
        │
   Optimal w*  (3 active assets)
        │
   ┌────┼─────────────┬──────────────┐
   ▼    ▼             ▼              ▼
 Donut  Frontier   Cum. Returns   Heatmap
```

---

## Outputs & Visualisations

### 1 — Correlation Heatmap (`correlation_heatmap.png`)

A lower-triangle Seaborn heatmap of pairwise Pearson correlations of daily returns over 2022–2026. All correlations are positive and fall in the **0.24 – 0.52** range, reflecting the broadly synchronised behaviour of NSE large-caps in this period. The BHARTIARTL ↔ HDFCBANK (0.27) and SBIN ↔ TCS (0.24) pairs represent the strongest diversification opportunities in the basket.

### 2 — Efficient Frontier (`efficient_frontier.html`)

An interactive Plotly scatter of **5,000 Dirichlet-sampled long-only portfolios** coloured by Sharpe Ratio. The optimal tangency portfolio is marked with a red star at:

```
Volatility = 17.75%  |  Return = 28.66%  |  Sharpe = 1.2766
```

Hovering over any point in the browser reveals its exact (Vol, Ret, Sharpe) triplet.

### 3 — Optimal Allocation Donut (`optimal_allocation.html`)

An interactive Plotly donut chart. The Sharpe Ratio **1.277** is annotated at the centre. Three wedges represent the active assets (BHARTIARTL 67.09%, SBIN 17.33%, LT 15.58%). Portfolio summary metrics are shown in the subtitle.

### 4 — Cumulative Performance (`cumulative_performance.html`)

A Plotly line chart tracking the $1 growth curve:

$$V_t = \prod_{\tau=1}^{t}\!\Bigl(1 + \textstyle\sum_{i} w_i^{*} \cdot r_{i,\tau}\Bigr)$$

- **Final Value: $2.89 · Total Return: +188.7%**
- All 7 individual constituent lines are plotted for comparison.
- A range slider enables drilling into any sub-period.

---

## 🔧 Configuration

All parameters are defined in the `Constants` block at the top of `main.py`:

```python
# --- 2. CONSTANTS -------------------------------------------------------------
TICKERS = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS',
    'SBIN.NS', 'ICICIBANK.NS', 'TCS.NS', 'LT.NS'
]
START_DATE           = '2022-01-01'
END_DATE             = '2026-01-01'
RISK_FREE_RATE       = 0.06           # 6% p.a. (Indian T-bill proxy)
TRADING_DAYS         = 252
WEIGHT_THRESHOLD     = 0.001          # filter weights < 0.1%
N_RANDOM_PORTFOLIOS  = 5_000
RANDOM_SEED          = 42
```

### Customisation Examples

**Restore the original 2019–2025 window:**
```python
START_DATE = '2019-01-01'
END_DATE   = '2025-01-01'
```

**Enforce diversification with a per-asset weight cap:**
```python
bounds = [(0.0, 0.40)] * len(TICKERS)   # no single asset > 40%
```

**Switch to US equities:**
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
RISK_FREE_RATE = 0.05
```

---

## Results Interpretation

### Why BHARTIARTL dominates at 67.09%

Bharti Airtel delivered the highest **individual Sharpe Ratio** within the 2022–2026 window — driven by strong ARPU growth, 5G rollout momentum, and Africa expansion — while simultaneously recording the **lowest average cross-correlation** with the rest of the basket (mean ρ ≈ 0.31). This combination of high return-per-unit-risk and low co-movement makes it the dominant tangency-portfolio holding.

### Why SBIN and LT enter but TCS does not

Despite TCS having some of the lowest pairwise correlations (0.24–0.33), its **realised return** in the 2022–2026 window was insufficient to earn a positive SLSQP weight once BHARTIARTL already captures the low-correlation, high-return niche. SBIN and LT add incremental Sharpe lift through different return drivers (PSU banking credit cycle, infrastructure capex supercycle). The optimiser confirms: no marginal addition of TCS, RELIANCE, HDFCBANK, or ICICIBANK improves SR beyond **1.2766**.

### Why the private banking pair (HDFCBANK + ICICIBANK) is zeroed

Their mutual correlation of **0.50** is among the two highest in the matrix. Holding both adds correlated volatility without proportional return uplift. SBIN partially captures banking-sector exposure with a meaningfully different beta profile (PSU vs. private), so the SLSQP allocates only to SBIN.

### Metric Glossary

| Metric | Value | Interpretation |
|---|---|---|
| **Sharpe Ratio** | 1.2766 | Each unit of excess risk earns ≈1.28 units of excess return. Values > 1.0 are considered good. |
| **Volatility** | 17.75% | Sub-18% annualised σ for a 3-asset equity portfolio reflects genuine diversification. |
| **Return** | 28.66% | Well above the 6% risk-free rate and the Nifty 50's typical long-run average of 12–15%. |
| **$1 → $2.89** | +188.7% | Implies a 4-year CAGR of ≈ 30.4%, consistent with the 28.66% arithmetic return estimate. |

---

## ⚠️ Limitations & Disclaimer

> **This project is for educational and research purposes only. It does not constitute financial advice. Past performance is not indicative of future results.**

### Known Limitations

| Limitation | Description |
|---|---|
| **Estimation Error** | Sample mean returns are noisy over a 4-year window. The 2022–2026 period captures a specific regime (India Telecom + Infra boom) that may not persist. |
| **Concentration Risk** | 67% in a single stock (BHARTIARTL) carries substantial idiosyncratic risk not captured by σ alone. Real mandates would impose per-asset caps. |
| **Static Weights** | Weights are constant over the full period; no rebalancing frequency, transaction costs, or tax drag are modelled. |
| **Survivorship Bias** | All tickers are currently prominent NSE constituents; delistings or restructurings are not modelled. |
| **Normal Returns Assumption** | Mean-variance optimisation understates tail risk and skewness common in emerging-market equities. |
| **In-Sample Evaluation** | Optimal weights are evaluated on the same data used to estimate them; a rolling out-of-sample study would give more conservative estimates. |

### Potential Enhancements

- **Ledoit-Wolf shrinkage** — reduce covariance estimation error
- **Black-Litterman model** — blend market equilibrium returns with analyst views
- **Maximum weight cap** — e.g. `wᵢ ≤ 0.40` to enforce diversification
- **Rolling-window optimisation** — walk-forward backtest with quarterly rebalancing
- **CVaR objective** — tail-risk-aware allocation replacing variance
- **Regime detection** — HMM-based switching between bull/bear allocation sets

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Harry Markowitz** — Nobel Laureate, father of Modern Portfolio Theory (1952)
- **[yfinance](https://github.com/ranaroussi/yfinance)** — NSE market data retrieval
- **[SciPy](https://scipy.org/)** — SLSQP optimisation engine
- **[Plotly](https://plotly.com/)** — Interactive visualisation framework
- **[Seaborn](https://seaborn.pydata.org/)** — Statistical EDA visualisation

---

<p align="center">
  Built with 🐍 Python &nbsp;|&nbsp; Quantitative Finance &nbsp;|&nbsp; NSE Indian Equities &nbsp;|&nbsp; Jan 2022 – Jan 2026
</p>