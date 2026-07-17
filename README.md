# Regulating the Retail Options Boom: Evidence from India's 2024–25 Equity Index Derivatives Reforms

Anklesh Rawat and Shreshtha Rawat — SSRN working paper (replication package).

## Research question
Did SEBI's October 2024 – April 2025 index-derivatives curbs (weekly-expiry
rationalization, 3× minimum contract size, expiry-day ELM, upfront premium,
intraday position limits) reduce retail derivatives activity and participation,
and did activity substitute into the cash market?

## Headline results
- **Turnover:** trend-adjusted DiD (index options vs. stock options, NSE+BSE)
  shows a −0.573 log-point on-impact drop in relative notional ADT (p < 0.001);
  average Nov-2024–May-2025 shortfall ≈ −43% vs. counterfactual. Unadjusted
  full-window DiD ≈ 0: the aggregate effect attenuates by FY2026.
- **Composition:** premium fell far less than notional (−13% vs −29% YoY);
  premium-to-notional trend reversed (+5.9 log pts/month, p < 0.001) — the
  deep-OTM weekly lottery end of the market was eliminated.
- **Participation:** unique traders −20% YoY; cohort DiD −46.8 log points for
  sub-₹10-lakh cohorts (permutation p = 0.10, second-most extreme of 20).
- **Substitution:** none into cash (−9.6% YoY); large NSE→BSE migration
  (NSE-only DiD −31% vs ≈0 market-wide).

## Contents
- `paper/manuscript.tex` — full manuscript (compiles with pdflatex; figures
  via `\graphicspath{{../figures/}}`). Upload to Overleaf together with
  `figures/*.pdf`.
- `code/01_extract_data.py` — builds the monthly panel from SEBI bulletin
  Excel/PDF annexures; hand-codes SEBI July-2025 study tables 8/11/12.
- `code/02_analysis.py` — DiD, event study, ITS, cohort DiD + permutation,
  robustness (placebo dates, window drops, alternative control, NSE-only).
- `code/03_trend_adjusted.py` — trend-adjusted DiD, horizon gaps, magnitudes,
  summary statistics.
- `code/04_figures.py` — all six figures (PDF + PNG).
- `data/` — raw SEBI source documents (bulletins, studies) and derived CSVs
  (`monthly_market_panel.csv` is the main analysis file).
- `results/` — `results.json` (all estimates), event-study/diff series CSVs.
- `figures/` — publication figures.

## Data sources (all free and public, downloaded 16 July 2026)
- SEBI Monthly Bulletin annexure tables (Apr-2023, Apr-2024, Mar-2025,
  Apr-2025, Apr-2026 issues), sebi.gov.in → Reports & Statistics → Publications.
- SEBI research studies: 25 Jan 2023, 23 Sep 2024, 7 Jul 2025 (Comparative
  study of growth in EDS vis-à-vis Cash Market).
- Panel validated against SEBI's published window aggregates (0.2–4%).

## Reproduce
```bash
pip install pandas numpy statsmodels matplotlib openpyxl pdfplumber
python code/01_extract_data.py
python code/02_analysis.py
python code/03_trend_adjusted.py
python code/04_figures.py
cd paper && pdflatex manuscript.tex && pdflatex manuscript.tex
```

## AI-use disclosure
A large language model (Claude, Anthropic) assisted with extraction code,
statistical programming, and drafting. The authors specified the design,
verified all numbers against primary SEBI documents, and take full
responsibility for the content. (Disclosure also appears in the manuscript,
per SSRN policy.)
