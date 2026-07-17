"""
02_analysis.py
Econometric analysis: SEBI's 2024-25 index-derivatives curbs.

Designs
-------
D1  Difference-in-differences: index options (treated) vs stock options
    (control), monthly log ADT, market-wide (NSE+BSE). Treatment from
    Nov-2024 (first measures effective 20 Nov 2024).
D2  Event-study version of D1 (monthly interactions).
D3  Interrupted time series on log index-options ADT (level + slope break),
    Newey-West HAC standard errors.
D4  Cohort difference-in-differences on unique-trader counts across
    turnover buckets (SEBI Jul-2025 study, Table 8).
D5  Cash-market substitution test.
Robustness: placebo date, drop transition months, drop Jane-Street window,
alternative control (stock futures), NSE-only series.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json, warnings
warnings.filterwarnings("ignore")

DATA = "/sessions/serene-great-cannon/mnt/outputs/data/"
OUT  = "/sessions/serene-great-cannon/mnt/outputs/results/"
import os; os.makedirs(OUT, exist_ok=True)

TREAT_START = pd.Period("2024-11", "M")   # first wave effective 20 Nov 2024

panel = pd.read_csv(DATA + "monthly_market_panel.csv", index_col=0)
panel.index = pd.PeriodIndex(panel.index, freq="M")

res = {}

# ============================================================ D1: 2x2 DiD
def build_long(treat_col, ctrl_col, val_label):
    df = panel[[treat_col, ctrl_col]].dropna().copy()
    long = df.stack().reset_index()
    long.columns = ["month", "group", "adt"]
    long["treated"] = (long["group"] == treat_col).astype(int)
    long["post"] = (long["month"] >= TREAT_START).astype(int)
    long["ln_adt"] = np.log(long["adt"])
    long["t"] = long["month"].astype(str)
    return long

def did_2x2(long, label, hac_lags=6):
    """TWFE 2x2 DiD; SEs via HAC on the differenced series (Donald-Lang
    style collapse for 2 groups) and conventional OLS for reference."""
    m = smf.ols("ln_adt ~ treated*post + C(t)", data=long).fit()
    beta = m.params["treated:post"]
    # collapse: monthly diff (treated - control), regress on post
    wide = long.pivot(index="month", columns="treated", values="ln_adt")
    diff = (wide[1] - wide[0]).rename("d").to_frame()
    diff["post"] = (diff.index >= TREAT_START).astype(int)
    md = smf.ols("d ~ post", data=diff).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    out = dict(beta_twfe=float(beta),
               beta_diff=float(md.params["post"]),
               se_hac=float(md.bse["post"]),
               t_hac=float(md.tvalues["post"]),
               p_hac=float(md.pvalues["post"]),
               n_months=int(len(diff)),
               pct_effect=float(np.expm1(md.params["post"]) * 100))
    res[label] = out
    print(f"[{label}] DiD beta={out['beta_diff']:.4f} "
          f"(HAC se={out['se_hac']:.4f}, t={out['t_hac']:.2f}, "
          f"p={out['p_hac']:.4f}) -> {out['pct_effect']:.1f}%  N={out['n_months']}m")
    return md, diff

print("=" * 70)
long_not  = build_long("idxopt_notional_adt", "stkopt_notional_adt", "notional")
long_prem = build_long("idxopt_prem_adt",     "stkopt_prem_adt",     "premium")
md_not,  diff_not  = did_2x2(long_not,  "D1_notional")
md_prem, diff_prem = did_2x2(long_prem, "D1_premium")

# ============================================================ D2: event study
def event_study(long, label, ref_period=pd.Period("2024-10", "M")):
    long = long.copy()
    long["rel"] = long["month"].apply(lambda m: (m - ref_period).n)
    # interact treated with each month (omit rel=0 i.e. Oct-2024)
    rows = []
    wide = long.pivot(index="month", columns="treated", values="ln_adt")
    d = (wide[1] - wide[0])
    base = d.loc[ref_period]
    for m, v in d.items():
        rows.append(dict(month=str(m), rel=(m - ref_period).n,
                         coef=float(v - base)))
    es = pd.DataFrame(rows)
    es.to_csv(OUT + f"event_study_{label}.csv", index=False)
    return es

es_not = event_study(long_not, "notional")
es_prem = event_study(long_prem, "premium")

# pre-trend test: regression of diff on time, pre-period only
for label, diff in [("notional", diff_not), ("premium", diff_prem)]:
    pre = diff[diff.index < TREAT_START].copy()
    pre["trend"] = range(len(pre))
    mp = smf.ols("d ~ trend", data=pre).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    res[f"pretrend_{label}"] = dict(slope=float(mp.params["trend"]),
                                    se=float(mp.bse["trend"]),
                                    p=float(mp.pvalues["trend"]),
                                    n=int(len(pre)))
    print(f"[pretrend {label}] slope={mp.params['trend']:.4f} "
          f"(p={mp.pvalues['trend']:.3f}, n={len(pre)})")

# ============================================================ D3: ITS
def its(series, label, hac_lags=6):
    df = panel[[series]].dropna().copy()
    df["ln"] = np.log(df[series])
    df["t"] = range(len(df))
    df["post"] = (df.index >= TREAT_START).astype(int)
    first_post_t = df.loc[df["post"] == 1, "t"].min()
    df["t_post"] = np.maximum(df["t"] - first_post_t + 1, 0) * df["post"]
    df["moy"] = df.index.month
    m = smf.ols("ln ~ t + post + t_post + C(moy)", data=df).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    res[f"ITS_{label}"] = dict(level=float(m.params["post"]),
                               level_se=float(m.bse["post"]),
                               level_p=float(m.pvalues["post"]),
                               slope=float(m.params["t_post"]),
                               slope_se=float(m.bse["t_post"]),
                               slope_p=float(m.pvalues["t_post"]),
                               trend_pre=float(m.params["t"]),
                               n=int(m.nobs), r2=float(m.rsquared))
    print(f"[ITS {label}] level={m.params['post']:.3f} (p={m.pvalues['post']:.4f}) "
          f"slope={m.params['t_post']:.4f} (p={m.pvalues['t_post']:.4f}) "
          f"pre-trend={m.params['t']:.4f} n={int(m.nobs)}")
    return m

print("=" * 70)
its("idxopt_notional_adt", "idxopt_notional")
its("idxopt_prem_adt", "idxopt_premium")
its("stkopt_notional_adt", "stkopt_notional")   # falsification: control series
its("cash_adt", "cash")                          # D5 substitution
its("stk_fut_adt", "stk_fut")

# ============================================================ D4: cohort DiD
print("=" * 70)
t8 = pd.read_csv(DATA + "sebi_t8_cohorts.csv")
# exposure: small-ticket cohorts (< Rs 10 lakh turnover) most exposed to the
# 3x increase in minimum contract size
t8["exposed"] = [1, 1, 1, 0, 0, 0]
g = []
for _, r in t8.iterrows():
    g.append(dict(bucket=r["bucket"], exposed=r["exposed"], period="pre",
                  dln=np.log(r["traders_2324"] / r["traders_2223"])))
    g.append(dict(bucket=r["bucket"], exposed=r["exposed"], period="post",
                  dln=np.log(r["traders_2425"] / r["traders_2324"])))
gd = pd.DataFrame(g)
gd["post"] = (gd["period"] == "post").astype(int)
mc = smf.ols("dln ~ exposed*post", data=gd).fit(cov_type="HC1")
res["D4_cohort"] = dict(beta=float(mc.params["exposed:post"]),
                        se=float(mc.bse["exposed:post"]),
                        p=float(mc.pvalues["exposed:post"]),
                        n=int(mc.nobs))
print(f"[D4 cohort DiD] beta={mc.params['exposed:post']:.4f} "
      f"(se={mc.bse['exposed:post']:.4f}, p={mc.pvalues['exposed:post']:.4f})")
# permutation test over exposure assignments (choose 3 of 6 as exposed)
from itertools import combinations
obs = mc.params["exposed:post"]
perms = []
for c in combinations(range(6), 3):
    tt = t8.copy(); tt["exposed"] = [1 if i in c else 0 for i in range(6)]
    gg = []
    for _, r in tt.iterrows():
        gg.append(dict(exposed=r["exposed"], post=0,
                       dln=np.log(r["traders_2324"] / r["traders_2223"])))
        gg.append(dict(exposed=r["exposed"], post=1,
                       dln=np.log(r["traders_2425"] / r["traders_2324"])))
    ggd = pd.DataFrame(gg)
    mm = smf.ols("dln ~ exposed*post", data=ggd).fit()
    perms.append(mm.params["exposed:post"])
p_perm = np.mean([abs(x) >= abs(obs) for x in perms])
res["D4_cohort"]["p_perm"] = float(p_perm)
print(f"[D4 permutation] p = {p_perm:.3f} over {len(perms)} assignments")

# per-bucket growth table
t8["g_pre"] = np.log(t8["traders_2324"] / t8["traders_2223"]) * 100
t8["g_post"] = np.log(t8["traders_2425"] / t8["traders_2324"]) * 100
t8.to_csv(OUT + "cohort_growth.csv", index=False)
print(t8[["bucket", "g_pre", "g_post"]].round(1).to_string(index=False))

# ============================================================ robustness
print("=" * 70)
def did_window(diff, label, drop=None, treat=TREAT_START, sample_end=None):
    d = diff.copy()
    if drop is not None:
        d = d[~d.index.isin(drop)]
    if sample_end is not None:
        d = d[d.index <= sample_end]
    d["post"] = (d.index >= treat).astype(int)
    if d["post"].nunique() < 2:
        return
    m = smf.ols("d ~ post", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    res[f"rob_{label}"] = dict(beta=float(m.params["post"]),
                               se=float(m.bse["post"]),
                               p=float(m.pvalues["post"]), n=int(m.nobs))
    print(f"[rob {label}] beta={m.params['post']:.4f} "
          f"(se={m.bse['post']:.4f}, p={m.pvalues['post']:.4f}, n={int(m.nobs)})")

# placebo: fake treatment Nov-2023, estimated on pre-period only
did_window(diff_not[diff_not.index < TREAT_START], "placebo_nov23_notional",
           treat=pd.Period("2023-11", "M"))
did_window(diff_prem[diff_prem.index < TREAT_START], "placebo_nov23_premium",
           treat=pd.Period("2023-11", "M"))
# drop transition months Nov-Dec 2024
trans = [pd.Period("2024-11", "M"), pd.Period("2024-12", "M")]
did_window(diff_not, "droptrans_notional", drop=trans)
did_window(diff_prem, "droptrans_premium", drop=trans)
# drop Jane Street window Jul-Aug 2025
js = [pd.Period("2025-07", "M"), pd.Period("2025-08", "M")]
did_window(diff_not, "dropJS_notional", drop=js)
did_window(diff_prem, "dropJS_premium", drop=js)
# end sample Mar-2025 (exclude May-2025 expiry standardisation + later)
did_window(diff_not, "endMar25_notional", sample_end=pd.Period("2025-03", "M"))
did_window(diff_prem, "endMar25_premium", sample_end=pd.Period("2025-03", "M"))
# alternative control: stock futures
long_alt = build_long("idxopt_notional_adt", "stk_fut_adt", "alt")
did_2x2(long_alt, "rob_ctrl_stkfut_notional")
# NSE-only
nsep = pd.read_csv(DATA + "monthly_nse_panel.csv", index_col=0)
nsep.index = pd.PeriodIndex(nsep.index, freq="M")
_sav = panel
globals()["panel"] = nsep
long_nse = build_long("idxopt_notional_adt", "stkopt_notional_adt", "nse")
did_2x2(long_nse, "rob_nseonly_notional")
globals()["panel"] = _sav

# ============================================================ save
with open(OUT + "results.json", "w") as f:
    json.dump(res, f, indent=1)
print("\nsaved ->", OUT + "results.json")
