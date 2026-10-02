"""Marginal priors of the independent model, its per-benchmark hyperpriors integrated out.

With ``joint=False`` every benchmark gets its own hyperparameters (a mean and a spread for L, k,
alpha, xi and s) and is their only member, so the data reach them only through that one child.
Integrating them out leaves the posterior of every other quantity unchanged and removes six to
ten uninformed parameters per benchmark whose geometry NUTS cannot cross (on the 2026-10-01 run:
step size 0.001 to 0.007, trees at the depth limit on nearly every draw, bulk ESS 7 to 89).

Each marginal p(x) = integral of f(x | mu, sigma) p(mu) p(sigma) is computed once by quadrature:
sigma on Gauss-Legendre nodes of its CDF (a half-normal), mu = x - sigma z on Gauss-Legendre nodes
of z, so the integrand stays smooth when sigma is small and f is a spike around mu = x. The log
density is tabulated on a grid of an unconstrained coordinate y (logit or log) and read back in
the model by cubic Hermite interpolation, linear in y beyond the table. The clamps of the
hierarchical model (sigma below sqrt(mu(1-mu)) for the Beta, below mu for xi's Gamma) are applied
at every node, so the marginal is the one the hierarchical model implies.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy import special, stats

N_SIGMA, N_Z, N_GRID = 96, 160, 1601
Z_MAX = 12.0


@dataclass(frozen=True)
class LogDensityTable:
    """log p(x) on a uniform grid of y = g(x), with its slopes, for Hermite interpolation."""
    y0: float
    h: float
    f: np.ndarray       # log density of x (not of y) at each knot
    m: np.ndarray       # d f / d y at each knot


def _beta_ms(m, s):
    k = m * (1 - m) / s**2 - 1
    return m * k, (1 - m) * k


def _gamma_ms(m, s):
    return (m / s) ** 2, m / s**2         # shape, rate


def _gl(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def _marginal(x, sigma_scale, mu_logpdf, mu_lo, mu_hi, child_logpdf):
    """log integral of f(x | mu, sigma) p(mu) p(sigma) dmu dsigma for each x (1-D array)."""
    u, wu = _gl(N_SIGMA, 0.0, 1.0)
    sig = sigma_scale * np.sqrt(2.0) * special.erfinv(u)          # half-normal quantiles
    zt, wt = _gl(N_Z, -1.0, 1.0)
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        # z range: mu = xi - sig z inside the support of mu, and |z| <= Z_MAX
        z_lo = np.maximum(-Z_MAX, (xi - mu_hi) / sig)
        z_hi = np.minimum(Z_MAX, (xi - mu_lo) / sig)
        ok = z_hi > z_lo
        z = 0.5 * (z_hi - z_lo)[:, None] * zt[None, :] + 0.5 * (z_hi + z_lo)[:, None]
        wz = 0.5 * (z_hi - z_lo)[:, None] * wt[None, :]
        mu = xi - sig[:, None] * z
        with np.errstate(all="ignore"):
            lg = (child_logpdf(xi, mu, sig[:, None]) + mu_logpdf(mu)
                  + np.log(sig)[:, None] + np.log(wz) + np.log(wu)[:, None])
        lg = np.where(ok[:, None] & np.isfinite(lg), lg, -np.inf)
        out[i] = special.logsumexp(lg)
    return out


def _table(logit: bool, x_lo: float, x_hi: float, **kw) -> LogDensityTable:
    if logit:
        y = np.linspace(special.logit(x_lo), special.logit(x_hi), N_GRID)
        x = special.expit(y)
    else:
        y = np.linspace(np.log(x_lo), np.log(x_hi), N_GRID)
        x = np.exp(y)
    f = _marginal(x, **kw)
    f = np.where(np.isfinite(f), f, f[np.isfinite(f)].min())
    return LogDensityTable(float(y[0]), float(y[1] - y[0]), f, np.gradient(f, y))


def _beta_child(clamp):
    def lp(x, mu, sig):
        s = np.minimum(sig, np.sqrt(mu * (1 - mu)) - 1e-4) if clamp else sig
        a, b = _beta_ms(mu, s)
        return stats.beta.logpdf(x, a, b)
    return lp


def _gamma_child(clamp):
    def lp(x, mu, sig):
        s = np.minimum(sig, mu - 1e-6) if clamp else sig
        a, r = _gamma_ms(mu, s)
        return stats.gamma.logpdf(x, a, scale=1.0 / r)
    return lp


def _gamma_mu(mean, sd):
    a, r = _gamma_ms(mean, sd)
    return lambda mu: stats.gamma.logpdf(mu, a, scale=1.0 / r)


@cache
def L_raw_table(mu_raw: float, sd_raw: float) -> LogDensityTable:
    """L_raw ~ Beta(L_raw_mu, min(L_raw_sigma, sqrt(mu(1-mu)) - 1e-4)),
    L_raw_mu ~ Beta(mu_raw, sd_raw), L_raw_sigma ~ HalfNormal(sd_raw)."""
    a, b = _beta_ms(mu_raw, sd_raw)
    return _table(True, 1e-14, 1 - 1e-14, sigma_scale=sd_raw,
                  mu_logpdf=lambda mu: stats.beta.logpdf(mu, a, b), mu_lo=0.0, mu_hi=1.0,
                  child_logpdf=_beta_child(clamp=True))


@cache
def gamma_table(mean: float, sd: float, sigma_scale: float, clamp: bool,
                x_lo: float, x_hi: float) -> LogDensityTable:
    """x ~ Gamma(x_mu, x_sigma [clamped below x_mu]), x_mu ~ Gamma(mean, sd),
    x_sigma ~ HalfNormal(sigma_scale)."""
    return _table(False, x_lo, x_hi, sigma_scale=sigma_scale, mu_logpdf=_gamma_mu(mean, sd),
                  mu_lo=0.0, mu_hi=mean + 40 * sd, child_logpdf=_gamma_child(clamp))


@cache
def neg_skew_table(mu0: float, sd0: float, sigma_scale: float) -> LogDensityTable:
    """For q = -s > 0: s ~ TruncatedNormal(s_mu, s_sigma, upper=0), s_mu ~ Normal(mu0, sd0),
    s_sigma ~ HalfNormal(sigma_scale)."""
    def child(q, mu, sig):          # density of q = -s, mu is the mean of s
        return stats.norm.logpdf(-q, mu, sig) - stats.norm.logcdf(-mu / sig)
    def mu_lp(mu):
        return stats.norm.logpdf(-mu, mu0, sd0)   # integration variable is -mu (mean of q)
    return _table(False, 1e-4, 40.0, sigma_scale=sigma_scale, mu_logpdf=mu_lp,
                  mu_lo=-mu0 - 40 * sd0, mu_hi=-mu0 + 40 * sd0,
                  child_logpdf=lambda q, nm, sig: child(q, -nm, sig))


def hermite_logp(y, table: LogDensityTable):
    """log p(x) at y = g(x), a pytensor expression: cubic Hermite inside the table, linear in
    y outside it."""
    import pytensor.tensor as pt
    n = table.f.size
    f = pt.constant(table.f)
    m = pt.constant(table.m)
    u = (y - table.y0) / table.h
    i = pt.cast(pt.clip(pt.floor(u), 0, n - 2), "int64")
    t = u - i
    f0, f1, m0, m1 = f[i], f[i + 1], m[i] * table.h, m[i + 1] * table.h
    t2, t3 = t * t, t * t * t
    inside = ((2 * t3 - 3 * t2 + 1) * f0 + (t3 - 2 * t2 + t) * m0
              + (-2 * t3 + 3 * t2) * f1 + (t3 - t2) * m1)
    below = table.f[0] + table.m[0] * (y - table.y0)
    above = table.f[-1] + table.m[-1] * (y - (table.y0 + (n - 1) * table.h))
    return pt.switch(u < 0, below, pt.switch(u > n - 1, above, inside))
