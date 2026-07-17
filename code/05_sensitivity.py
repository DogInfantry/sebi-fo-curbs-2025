"""
05_sensitivity.py
Sensitivity of the short-run DiD effect to the assumed counterfactual
differential trend, in the spirit of Rambachan & Roth (2023).

The trend-adjusted estimate extrapolates the pre-period differential trend
(theta_hat = 0.022 log points/month). Here we recompute the average
Nov-2024..May-2025 effect under every assumed counterfactual trend theta in
a grid, and report the breakdown value theta* at which the effect is zero.
"""
import numpy as np, pandas as pd, json
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

OUT = "/sessions/serene-great-cannon/mnt/outputs/results/"
FIG = "/sessions/serene-great-cannon/mnt/outputs/figures/"
TREAT = pd.Period("2024-11", "M")

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

d = pd.read_csv(OUT + "diff_series_notional.csv", index_col=0)
d.index = pd.PeriodIndex(d.index, freq="M")

pre = d[d.index < TREAT]
fit = smf.ols("d ~ t", data=pre).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
theta_hat, theta_se = fit.params["t"], fit.bse["t"]
t0 = pre["t"].max()                      # Oct-2024
anchor = fit.params["Intercept"] + theta_hat * t0   # fitted level at Oct-24

win = d[(d.index >= TREAT) & (d.index <= pd.Period("2025-05", "M"))]

def effect(theta):
    cf = anchor + theta * (win["t"] - t0)
    return float((win["d"] - cf).mean())

grid = np.linspace(-0.16, 0.06, 1101)
eff = np.array([effect(th) for th in grid])
curve = pd.DataFrame({"theta": grid, "effect": eff})
curve.to_csv(OUT + "sensitivity_curve.csv", index=False)

theta_star = float(grid[np.argmin(np.abs(eff))])
e_hat = effect(theta_hat)
e_zero = effect(0.0)
M = (theta_star - theta_hat) / theta_se

print(f"theta_hat (pre-trend)     = {theta_hat:.4f} (se {theta_se:.4f})")
print(f"effect at theta_hat       = {e_hat:.3f}  ({np.expm1(e_hat)*100:.1f}%)")
print(f"effect at theta = 0       = {e_zero:.3f}  ({np.expm1(e_zero)*100:.1f}%)")
print(f"breakdown theta*          = {theta_star:.4f}")
print(f"theta* in SEs above trend = {M:.1f}")
print(f"theta* / theta_hat        = {theta_star/theta_hat:.2f}")

res = json.load(open(OUT + "results.json"))
res["sensitivity"] = dict(theta_hat=theta_hat, theta_se=theta_se,
                          effect_at_theta_hat=e_hat, effect_at_zero=e_zero,
                          theta_star=theta_star, M_se=M,
                          ratio=theta_star / theta_hat)
json.dump(res, open(OUT + "results.json", "w"), indent=1)

# ------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(grid, eff, color="#1a1a1a", lw=1.5)
ax.axhline(0, color="k", lw=0.7)
ymin = eff.min() * 1.12
for x, lab, c, dy in [(0.0, "no trend\nadjustment", "#8a8a8a", 0.10),
                      (theta_hat, "estimated\npre-trend $\\hat{\\gamma}$", "#4477aa", 0.10),
                      (theta_star, "breakdown $\\gamma^{*}$", "#b30000", 0.02)]:
    ax.axvline(x, color=c, lw=1.0, ls="--")
    ax.annotate(lab, xy=(x, ymin), xytext=(x + 0.003, ymin + dy),
                fontsize=8, color=c)
ax.scatter([0.0, theta_hat], [e_zero, e_hat], zorder=3,
           color=["#8a8a8a", "#4477aa"], s=25)
ax.set_xlabel("Assumed counterfactual differential trend (log points/month)")
ax.set_ylabel("Mean effect, Nov-2024–May-2025 (log points)")
fig.savefig(FIG + "fig7_sensitivity.pdf")
fig.savefig(FIG + "fig7_sensitivity.pdf")
fig.savefig(FIG + "fig7_sensitivity.png")
print("saved fig7_sensitivity")
