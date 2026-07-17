"""
03_trend_adjusted.py
Trend-adjusted DiD (differential linear trend allowed), horizon-specific
effects, and summary statistics for the manuscript.
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import json, warnings, os
warnings.filterwarnings("ignore")
DATA = "/sessions/serene-great-cannon/mnt/outputs/data/"
OUT  = "/sessions/serene-great-cannon/mnt/outputs/results/"
TREAT = pd.Period("2024-11", "M")

panel = pd.read_csv(DATA + "monthly_market_panel.csv", index_col=0)
panel.index = pd.PeriodIndex(panel.index, freq="M")
res = json.load(open(OUT + "results.json"))

def diff_series(tcol, ccol):
    df = panel[[tcol, ccol]].dropna()
    d = (np.log(df[tcol]) - np.log(df[ccol])).rename("d").to_frame()
    d["t"] = range(len(d))
    d["post"] = (d.index >= TREAT).astype(int)
    fp = d.loc[d.post == 1, "t"].min()
    d["t_post"] = np.maximum(d.t - fp + 1, 0) * d.post
    return d

for label, tcol, ccol in [
        ("notional", "idxopt_notional_adt", "stkopt_notional_adt"),
        ("premium", "idxopt_prem_adt", "stkopt_prem_adt")]:
    d = diff_series(tcol, ccol)
    # trend-adjusted DiD: level shift + slope change relative to pre-trend
    m = smf.ols("d ~ t + post + t_post", data=d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6})
    # implied gap at 6 and 12 months after treatment
    g6  = m.params["post"] + 6 * m.params["t_post"]
    g12 = m.params["post"] + 12 * m.params["t_post"]
    res[f"TA_{label}"] = dict(
        level=float(m.params["post"]), level_se=float(m.bse["post"]),
        level_p=float(m.pvalues["post"]),
        slope=float(m.params["t_post"]), slope_se=float(m.bse["t_post"]),
        slope_p=float(m.pvalues["t_post"]),
        pretrend=float(m.params["t"]), pretrend_se=float(m.bse["t"]),
        gap6=float(g6), gap12=float(g12), n=int(m.nobs))
    print(f"[TA {label}] level={m.params['post']:.3f} (p={m.pvalues['post']:.4f}) "
          f"slope={m.params['t_post']:.4f} (p={m.pvalues['t_post']:.4f}) "
          f"pretrend={m.params['t']:.4f} | gap@6m={g6:.3f} gap@12m={g12:.3f}")

    # short-run average effect Nov-24..May-25 relative to pre-mean, detrended
    pre = d[d.index < TREAT]
    tr = smf.ols("d ~ t", data=pre).fit()
    d["cf"] = tr.params["Intercept"] + tr.params["t"] * d["t"]
    win = d[(d.index >= TREAT) & (d.index <= pd.Period("2025-05", "M"))]
    sr = float((win["d"] - win["cf"]).mean())
    res[f"TA_{label}"]["short_run_nov_may"] = sr
    res[f"TA_{label}"]["short_run_pct"] = float(np.expm1(sr) * 100)
    print(f"   short-run (Nov24-May25) vs extrapolated trend: {sr:.3f} "
          f"({np.expm1(sr)*100:.1f}%)")
    d.to_csv(OUT + f"diff_series_{label}.csv")

# ---------------- raw magnitudes for the text -------------------------------
w_post = [pd.Period(x, "M") for x in
          ["2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05"]]
w_pre = [p - 12 for p in w_post]
mag = {}
for c in ["idxopt_prem", "idxopt_notional", "stkopt_prem", "stkopt_notional",
          "idx_fut", "stk_fut", "cash_turnover"]:
    a = panel.loc[w_post, c].sum() / panel.loc[w_post, "days"].sum()
    b = panel.loc[w_pre, c].sum() / panel.loc[w_pre, "days"].sum()
    mag[c] = dict(adt_post=float(a), adt_pre=float(b),
                  yoy_pct=float((a / b - 1) * 100))
    print(f"[mag {c}] Dec24-May25 ADT={a:,.0f} vs Dec23-May24 {b:,.0f} "
          f"({(a/b-1)*100:+.1f}%)")
res["magnitudes"] = mag

# ---------------- summary statistics table ----------------------------------
sumcols = {"idxopt_prem_adt": "Index options ADT (premium)",
           "idxopt_notional_adt": "Index options ADT (notional)",
           "stkopt_prem_adt": "Stock options ADT (premium)",
           "stkopt_notional_adt": "Stock options ADT (notional)",
           "idx_fut_adt": "Index futures ADT",
           "stk_fut_adt": "Stock futures ADT",
           "cash_adt": "Cash market ADT"}
stats = []
for c, name in sumcols.items():
    s = panel[c].dropna()
    pre = s[s.index < TREAT]; post = s[s.index >= TREAT]
    stats.append(dict(var=name, n=len(s), mean=s.mean(), sd=s.std(),
                      p25=s.quantile(.25), med=s.median(), p75=s.quantile(.75),
                      pre_mean=pre.mean(), post_mean=post.mean()))
pd.DataFrame(stats).to_csv(OUT + "summary_stats.csv", index=False)

json.dump(res, open(OUT + "results.json", "w"), indent=1)
print("\nupdated ->", OUT + "results.json")
