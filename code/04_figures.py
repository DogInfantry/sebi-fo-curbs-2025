"""
04_figures.py -- publication-quality figures (PDF + PNG).
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings, os
warnings.filterwarnings("ignore")

DATA = "/sessions/serene-great-cannon/mnt/outputs/data/"
OUT = "/sessions/serene-great-cannon/mnt/outputs/results/"
FIG = "/sessions/serene-great-cannon/mnt/outputs/figures/"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#1a1a1a", "#8a8a8a", "#b30000", "#4477aa"

panel = pd.read_csv(DATA + "monthly_market_panel.csv", index_col=0)
panel.index = pd.PeriodIndex(panel.index, freq="M").to_timestamp()

TREAT = pd.Timestamp("2024-11-01")
EVENTS = [("2024-11-20", "Wave 1:\nweekly expiry,\nELM"),
          ("2025-01-02", "Wave 2:\ncontract size"),
          ("2025-04-01", "Wave 3:\nintraday limits")]

def vlines(ax, labels=True, ymax=0.97):
    for i, (d, lab) in enumerate(EVENTS):
        ax.axvline(pd.Timestamp(d), color=C3, lw=0.9,
                   ls=["-", "--", ":"][i], alpha=0.85)

def save(fig, name):
    fig.savefig(FIG + name + ".pdf")
    fig.savefig(FIG + name + ".png")
    plt.close(fig)
    print("saved", name)

# ---------------------------------------------------------- Figure 1: levels
fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
ax = axes[0]
ax.plot(panel.index, panel["idxopt_notional_adt"] / 1e7, color=C1, lw=1.4,
        label="Index options (notional)")
ax.set_yscale("log")
ax.set_ylabel("ADT, ₹ crore × 10⁷ (log scale)")
vlines(ax)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("(a) Index options, notional average daily turnover", fontsize=10, loc="left")

ax = axes[1]
ax.plot(panel.index, panel["idxopt_prem_adt"] / 1e3, color=C1, lw=1.4,
        label="Index options (premium)")
ax.plot(panel.index, panel["stkopt_prem_adt"] / 1e3 * 8, color=C2, lw=1.2, ls="--",
        label="Stock options (premium, ×8)")
ax.set_ylabel("ADT, ₹ '000 crore")
vlines(ax)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("(b) Options premium average daily turnover, NSE+BSE", fontsize=10, loc="left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
for a in axes:
    a.axvline(TREAT, color=C3, lw=0, alpha=0)  # keep range
fig.autofmt_xdate(rotation=0, ha="center")
save(fig, "fig1_series")

# ---------------------------------------------- Figure 2: DiD gap (money fig)
d = pd.read_csv(OUT + "diff_series_notional.csv", index_col=0)
d.index = pd.PeriodIndex(d.index, freq="M").to_timestamp()
pre = d[d.index < TREAT]
b = np.polyfit(pre["t"], pre["d"], 1)
d["cf"] = b[1] + b[0] * d["t"]

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(d.index, d["d"], color=C1, lw=1.5, marker="o", ms=2.6,
        label="ln(index options) − ln(stock options), notional ADT")
ax.plot(d.index, d["cf"], color=C4, lw=1.2, ls="--",
        label="Pre-period linear trend (extrapolated)")
ax.fill_between(d.index, d["d"], d["cf"],
                where=(d.index >= TREAT), color=C3, alpha=0.14,
                label="Post-treatment gap")
vlines(ax)
ax.set_ylabel("Log difference")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
save(fig, "fig2_did_gap")

# ------------------------------------------- Figure 3: event-study coefficients
es = pd.read_csv(OUT + "event_study_notional.csv")
es["month"] = pd.PeriodIndex(es["month"], freq="M").to_timestamp()
# de-trend using pre-period slope of the diff (rel<0)
pre_es = es[es["rel"] < 0]
bb = np.polyfit(pre_es["rel"], pre_es["coef"], 1)
es["coef_detr"] = es["coef"] - (bb[1] + bb[0] * es["rel"])

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.axhline(0, color="k", lw=0.7)
ax.axvline(TREAT - pd.Timedelta(days=15), color=C3, lw=0.9, alpha=0.8)
ax.scatter(es["month"], es["coef_detr"], s=14, color=C1, zorder=3)
ax.plot(es["month"], es["coef_detr"], lw=0.8, color=C1, alpha=0.6)
ax.set_ylabel("Detrended gap relative to Oct-2024 (log points)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
save(fig, "fig3_event_study")

# ---------------------------------------------------- Figure 4: cohort growth
cg = pd.read_csv(OUT + "cohort_growth.csv")
x = np.arange(len(cg))
fig, ax = plt.subplots(figsize=(6.4, 3.7))
w = 0.38
ax.bar(x - w/2, cg["g_pre"], w, color=C2, label="Pre: Dec22–May23 → Dec23–May24")
ax.bar(x + w/2, cg["g_post"], w, color=C3, label="Post: Dec23–May24 → Dec24–May25")
ax.axhline(0, color="k", lw=0.7)
labels = ["<₹10k", "₹10k–1L", "₹1L–10L",
          "₹10L–1Cr", "₹1Cr–10Cr", ">₹10Cr"]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Log growth in unique traders (%)")
ax.set_xlabel("Six-month turnover bucket")
ax.legend(frameon=False, fontsize=8.5)
save(fig, "fig4_cohorts")

# ------------------------------------------- Figure 5: participation and P&L
t12 = pd.read_csv(DATA + "sebi_t12_quarterly.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
ax = axes[0]
ax.bar(t12["quarter"], t12["traders_lakh"], color=C4, width=0.6)
ax.set_ylabel("Unique individual traders (lakh)")
ax.set_title("(a) Participation", fontsize=10, loc="left")
ax.tick_params(axis="x", labelsize=8)
ax = axes[1]
ax.bar(t12["quarter"], -t12["net_profit_cr"] / 1000, color=C3, width=0.6)
ax.set_ylabel("Aggregate net loss (₹ '000 crore)")
ax.set_title("(b) Net losses", fontsize=10, loc="left")
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
save(fig, "fig5_quarterly")

# ------------------------------------------- Figure 6: substitution (indexed)
base = slice("2024-05-01", "2024-10-31")
fig, ax = plt.subplots(figsize=(6.4, 3.7))
for c, lab, col, ls in [("idxopt_notional_adt", "Index options (notional)", C1, "-"),
                        ("idxopt_prem_adt", "Index options (premium)", C4, "-"),
                        ("cash_adt", "Cash market", C3, "--"),
                        ("stkopt_notional_adt", "Stock options (notional)", C2, ":")]:
    s = panel[c] / panel.loc[base, c].mean() * 100
    ax.plot(panel.index, s, color=col, ls=ls, lw=1.3, label=lab)
ax.axhline(100, color="k", lw=0.6, alpha=0.6)
vlines(ax)
ax.set_xlim(pd.Timestamp("2023-10-01"), panel.index.max())
ax.set_ylabel("ADT, index (May–Oct 2024 = 100)")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
save(fig, "fig6_substitution")
print("all figures done")
