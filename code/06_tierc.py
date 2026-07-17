#!/usr/bin/env python3
"""
06_tierc.py -- Tier C extensions (revision round 1, July 2026).

C1: split post-period at the January 2026 lot-size recalibration
    (still a 2x2 design: one treated group, one control group; the
    post period is split into two phases of differing intensity).
C2: continuous cohort dose-response + Spearman rank test (SEBI Table 8).
C3: Newey-West lag sensitivity for the main collapsed-series estimates.
C4: NSE vs BSE index-options share figure (fig8).

Validation: reproduces the paper's baseline estimates from results.json
before computing anything new. Run from the repo root:
    python code/06_tierc.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
OUT = {}

# ---------------------------------------------------------------- load
mkt = pd.read_csv(os.path.join(DATA, "monthly_market_panel.csv"), parse_dates=["month"])
mkt = mkt.sort_values("month").reset_index(drop=True)
mkt["t"] = np.arange(len(mkt))
POST0 = pd.Timestamp("2024-11-01")   # first reform wave
RECAL = pd.Timestamp("2026-01-01")   # NSE lot-size recalibration series

def nw(y, X, lags=6):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m

# collapsed series
d_not = np.log(mkt["idxopt_notional_adt"]) - np.log(mkt["stkopt_notional_adt"])
prem = mkt.dropna(subset=["idxopt_prem_adt", "stkopt_prem_adt"]).copy()
d_prem = np.log(prem["idxopt_prem_adt"]) - np.log(prem["stkopt_prem_adt"])

post = (mkt["month"] >= POST0).astype(float)
post_p = (prem["month"] >= POST0).astype(float)

# ------------------------------------------------------- validation
base = json.load(open(os.path.join(RES, "results.json")))
m_val = nw(d_not, post.rename("Post"))
beta_val = m_val.params["Post"]
ref = base["D1_notional"]["beta_diff"]
assert abs(beta_val - ref) < 1e-6, f"validation FAILED: {beta_val} vs {ref}"
OUT["validation"] = {"D1_notional_reproduced": beta_val, "reference": ref}

# trend-adjusted validation
t = mkt["t"].astype(float)
t0 = int(mkt.loc[mkt["month"] == POST0, "t"].iloc[0])
X_ta = pd.DataFrame({"t": t, "Post": post, "PostTrend": (t - t0 + 1) * post})
m_ta = nw(d_not, X_ta)
OUT["validation"]["TA_notional_reproduced"] = m_ta.params["Post"]

# ---------------------------------------------------------------- C1
phase1 = ((mkt["month"] >= POST0) & (mkt["month"] < RECAL)).astype(float)
phase2 = (mkt["month"] >= RECAL).astype(float)
X1 = pd.DataFrame({"Post1": phase1, "Post2": phase2})
m1 = nw(d_not, X1)
OUT["C1_split_unadjusted_notional"] = {
    "beta_post1_nov24_dec25": m1.params["Post1"], "se_post1": m1.bse["Post1"], "p_post1": m1.pvalues["Post1"],
    "beta_post2_jan26_mar26": m1.params["Post2"], "se_post2": m1.bse["Post2"], "p_post2": m1.pvalues["Post2"],
    "n_months_post2": int(phase2.sum()),
}
X1t = pd.DataFrame({"t": t, "Post1": phase1, "Post2": phase2, "PostTrend": (t - t0 + 1) * post})
m1t = nw(d_not, X1t)
OUT["C1_split_trendadj_notional"] = {
    "beta_post1": m1t.params["Post1"], "se_post1": m1t.bse["Post1"], "p_post1": m1t.pvalues["Post1"],
    "beta_post2": m1t.params["Post2"], "se_post2": m1t.bse["Post2"], "p_post2": m1t.pvalues["Post2"],
}
# premium
ph1p = ((prem["month"] >= POST0) & (prem["month"] < RECAL)).astype(float)
ph2p = (prem["month"] >= RECAL).astype(float)
m1p = nw(d_prem, pd.DataFrame({"Post1": ph1p, "Post2": ph2p}))
OUT["C1_split_unadjusted_premium"] = {
    "beta_post1": m1p.params["Post1"], "se_post1": m1p.bse["Post1"], "p_post1": m1p.pvalues["Post1"],
    "beta_post2": m1p.params["Post2"], "se_post2": m1p.bse["Post2"], "p_post2": m1p.pvalues["Post2"],
}

# ---------------------------------------------------------------- C2
coh = pd.read_csv(os.path.join(DATA, "sebi_t8_cohorts.csv"))
# log growth across the two transitions
coh["g_pre"] = np.log(coh["traders_2324"]) - np.log(coh["traders_2223"])
coh["g_post"] = np.log(coh["traders_2425"]) - np.log(coh["traders_2324"])
coh["swing"] = coh["g_post"] - coh["g_pre"]
# bucket upper-threshold midpoint in log rupees (geometric midpoints; top bucket open)
# thresholds (Rs): <10k, 10k-1L, 1L-10L, 10L-1Cr, 1Cr-10Cr, >10Cr
log_mid = np.log([1e4/np.sqrt(10), np.sqrt(1e4*1e5), np.sqrt(1e5*1e6),
                  np.sqrt(1e6*1e7)*np.sqrt(10), np.sqrt(1e7*1e8)*10, 1e8*10*np.sqrt(10)])
# NOTE: buckets are <10k, 10k-1L(=1e5), 1L-10L(=1e6), 10L(=1e6)-1Cr(=1e7)... use consistent rupee values:
# <1e4, 1e4-1e5, 1e5-1e6, 1e6-1e7? No: 10L = 1e6, 1Cr = 1e7, 10Cr = 1e8.
log_mid = np.log([np.sqrt(1e3*1e4), np.sqrt(1e4*1e5), np.sqrt(1e5*1e6),
                  np.sqrt(1e6*1e7), np.sqrt(1e7*1e8), np.sqrt(1e8*1e9)])
coh["log_mid"] = log_mid
# stacked dose-response: g_cw = a + b*logmid + c*Post + d*(logmid x Post)
stack = pd.DataFrame({
    "g": pd.concat([coh["g_pre"], coh["g_post"]], ignore_index=True),
    "logmid": pd.concat([coh["log_mid"], coh["log_mid"]], ignore_index=True),
    "Post": [0.0]*6 + [1.0]*6,
})
stack["inter"] = stack["logmid"] * stack["Post"]
m2 = sm.OLS(stack["g"], sm.add_constant(stack[["logmid", "Post", "inter"]])).fit(cov_type="HC1")
rho_sw, p_sw = spearmanr(coh["log_mid"], coh["swing"])
rho_po, p_po = spearmanr(coh["log_mid"], coh["g_post"])
OUT["C2_dose_response"] = {
    "slope_logmid_x_post": m2.params["inter"], "se": m2.bse["inter"], "p_HC1": m2.pvalues["inter"],
    "interpretation": "extra log-points of participation growth per log-rupee of cohort size after the reform",
    "spearman_swing_vs_size": {"rho": rho_sw, "p": p_sw, "n": 6},
    "spearman_postgrowth_vs_size": {"rho": rho_po, "p": p_po, "n": 6},
    "swings_by_bucket": dict(zip(coh["bucket"], coh["swing"].round(4))),
}

# ---------------------------------------------------------------- C3
lags_grid = [2, 4, 6, 8, 12]
sens = {}
for L in lags_grid:
    mu = nw(d_not, post.rename("Post"), lags=L)
    mt = nw(d_not, X_ta, lags=L)
    sens[str(L)] = {
        "unadj_beta": mu.params["Post"], "unadj_se": mu.bse["Post"],
        "ta_beta": mt.params["Post"], "ta_se": mt.bse["Post"], "ta_p": mt.pvalues["Post"],
    }
OUT["C3_nw_lag_sensitivity_notional"] = sens

# ---------------------------------------------------------------- C4
nse = pd.read_csv(os.path.join(DATA, "monthly_nse_panel.csv"), parse_dates=["month"]).sort_values("month")
bse = pd.read_csv(os.path.join(DATA, "monthly_bse_panel.csv"), parse_dates=["month"]).sort_values("month")
mrg = nse.merge(bse, on="month", suffixes=("_nse", "_bse"))
for col in ["idxopt_notional", "idxopt_prem"]:
    mrg[f"bse_share_{col}"] = mrg[f"{col}_bse"] / (mrg[f"{col}_nse"] + mrg[f"{col}_bse"])
share = mrg[["month", "bse_share_idxopt_notional", "bse_share_idxopt_prem"]]
share.to_csv(os.path.join(RES, "bse_share_series.csv"), index=False)
OUT["C4_bse_share"] = {
    "notional_share_oct2024": float(share.loc[share["month"] == "2024-10-01", "bse_share_idxopt_notional"].iloc[0]),
    "notional_share_mar2026": float(share.loc[share["month"] == "2026-03-01", "bse_share_idxopt_notional"].iloc[0]),
    "premium_share_oct2024": float(share.loc[share["month"] == "2024-10-01", "bse_share_idxopt_prem"].iloc[0]),
    "premium_share_mar2026": float(share.loc[share["month"] == "2026-03-01", "bse_share_idxopt_prem"].iloc[0]),
}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(share["month"], 100 * share["bse_share_idxopt_notional"], color="#1f4e79", lw=1.8, label="Notional")
ax.plot(share["month"], 100 * share["bse_share_idxopt_prem"], color="#c0504d", lw=1.8, ls="--", label="Premium")
for dt, lab in [("2024-11-20", "Wave 1"), ("2025-01-01", "Wave 2"),
                ("2025-04-01", "Wave 3"), ("2026-01-01", "Lot recalib.")]:
    ax.axvline(pd.Timestamp(dt), color="grey", lw=0.8, ls=":")
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1], " " + lab, rotation=90, va="top", fontsize=7, color="grey")
ax.set_ylabel("BSE share of index-options ADT (%)")
ax.set_xlabel("")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig8_bse_share.{ext}"), dpi=200)

# ---------------------------------------------------------------- save
def _f(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    raise TypeError
with open(os.path.join(RES, "tierc_results.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=_f)
print(json.dumps(OUT, indent=2, default=_f))

# ---------------------------------------------------------------- C5 (round 2)
# Curvature robustness for the pre-trend (referee round 2):
# quadratic and piecewise-linear pre-trend specifications.
Xq = pd.DataFrame({"t": t, "t2": t**2, "Post": post, "PT": (t - t0 + 1) * post})
mq = nw(d_not, Xq)
kink = 15  # pre-sample midpoint (Jul-2023)
Xp = pd.DataFrame({"t": t, "tk": np.maximum(t - kink, 0), "Post": post, "PT": (t - t0 + 1) * post})
mp = nw(d_not, Xp)
C5 = {
    "quad_beta": float(mq.params["Post"]), "quad_se": float(mq.bse["Post"]), "quad_p": float(mq.pvalues["Post"]),
    "piecewise_beta": float(mp.params["Post"]), "piecewise_se": float(mp.bse["Post"]), "piecewise_p": float(mp.pvalues["Post"]),
    "note": "On-impact break robust to curvature; concave-quadratic extrapolation unstable at multi-month horizons (see paper, Robustness).",
}
try:
    existing = json.load(open(os.path.join(RES, "tierc_results.json")))
except FileNotFoundError:
    existing = OUT
existing["C5_pretrend_curvature_notional"] = C5
with open(os.path.join(RES, "tierc_results.json"), "w") as fh:
    json.dump(existing, fh, indent=2, default=_f)
print("C5:", C5)
