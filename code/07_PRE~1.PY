#!/usr/bin/env python3
"""
Figure 9 -- log premium-to-notional ratio of index options over time,
with the fitted pre-reform trend and the three reform-wave lines.

This visualises the central "composition" result (Section 4.4): the ratio
declines through the boom (cheaper, shorter-dated, deeper-OTM contracts) and
reverses sharply after the reforms.

Reads the derived monthly panel and writes:
    figures/fig9_premium_notional.pdf   (used by manuscript.tex)
    figures/fig9_premium_notional.png

Run from the repository root:
    python code/07_premium_notional_figure.py

No new dependencies beyond what 04_figures.py already uses
(pandas, numpy, matplotlib).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# CONFIG -- only touch these if the auto-detection below picks the wrong
# columns (the script prints what it chose).
# ---------------------------------------------------------------------------
PANEL_CSV = "data/monthly_market_panel.csv"   # main analysis file
DATE_COL  = None   # e.g. "date" / "month"; None => auto-detect
PREM_COL  = None   # index-options PREMIUM ADT column; None => auto-detect
NOTL_COL  = None   # index-options NOTIONAL ADT column; None => auto-detect

OUT_PDF = "figures/fig9_premium_notional.pdf"
OUT_PNG = "figures/fig9_premium_notional.png"

# (date, short label). Dates from Section 2.2 / Table 1.
REFORM_WAVES = [
    ("2024-11-20", "Wave 1"),
    ("2025-01-02", "Wave 2"),
    ("2025-04-01", "Wave 3"),
]
PRE_END = "2024-10-31"   # last pre-reform month used to fit the trend


# ---------------------------------------------------------------------------
def _auto(colnames, must, prefer):
    """Pick a column whose lowercased name contains every token in `must`,
    ranked by how many `prefer` tokens it also contains."""
    cands = [c for c in colnames if all(t in c.lower() for t in must)]
    if not cands:
        return None
    cands.sort(key=lambda c: sum(t in c.lower() for t in prefer), reverse=True)
    return cands[0]


def main():
    if not os.path.exists(PANEL_CSV):
        raise SystemExit(f"Cannot find {PANEL_CSV}. Run from the repo root.")

    df = pd.read_csv(PANEL_CSV)
    cols = list(df.columns)

    date_col = DATE_COL or _auto(cols, ["date"], []) or _auto(cols, ["month"], [])
    # index-options premium and notional ADT columns
    prem_col = PREM_COL or _auto(cols, ["prem"], ["index", "idx", "io", "opt", "adt"])
    notl_col = NOTL_COL or _auto(cols, ["not"],  ["index", "idx", "io", "opt", "adt"])

    if not all([date_col, prem_col, notl_col]):
        raise SystemExit(
            "Could not auto-detect the columns. Available columns:\n  "
            + "\n  ".join(cols)
            + "\n\nSet DATE_COL / PREM_COL / NOTL_COL at the top of this script."
        )
    print(f"Using date={date_col!r}, premium={prem_col!r}, notional={notl_col!r}")
    print("If those are not the INDEX-OPTIONS premium/notional ADT columns, "
          "set them manually in CONFIG.")

    d = df[[date_col, prem_col, notl_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.dropna().sort_values(date_col)
    # premium reporting starts Apr 2023; keep months where both are positive
    d = d[(d[prem_col] > 0) & (d[notl_col] > 0)]
    d["logratio"] = np.log(d[prem_col].astype(float) / d[notl_col].astype(float))

    # pre-reform linear trend, extrapolated across the full window
    pre = d[d[date_col] <= pd.Timestamp(PRE_END)]
    t_all = mdates.date2num(d[date_col])
    t_pre = mdates.date2num(pre[date_col])
    slope, intercept = np.polyfit(t_pre, pre["logratio"], 1)
    trend = intercept + slope * t_all

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(d[date_col], d["logratio"], color="#1f4e79", lw=1.8,
            marker="o", ms=3, label="Log premium/notional ratio")
    ax.plot(d[date_col], trend, color="#888888", ls="--", lw=1.4,
            label="Pre-reform trend (extrapolated)")
    for wd, lab in REFORM_WAVES:
        ax.axvline(pd.Timestamp(wd), color="#c00000", ls=":", lw=1.1)
    ax.text(pd.Timestamp(REFORM_WAVES[0][0]), ax.get_ylim()[1],
            " reforms", color="#c00000", va="top", ha="left", fontsize=8)

    ax.set_ylabel(r"$\ln(\mathrm{premium\ ADT} / \mathrm{notional\ ADT})$")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
