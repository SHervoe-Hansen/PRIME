"""
prime -- Protonation-number Reweighting in Multi-pH Ensembles
=============================================================

Reference implementation accompanying

    S. Hervø-Hansen and N. Matubayasi,
    "Protonation-number Reweighting for pH-Dependent Free Energy Calculations".

PRIME is the multistate Bennett acceptance ratio (MBAR) estimator specialized
to the semi-grand canonical ensemble of protons. Because the K pH ensembles
differ only through the term beta*mu_H+ * N(n), the potential energy cancels
identically from the MBAR equations and the sum over sampled configurations
collapses onto the discrete protonation number N. The protonation number is a
sufficient statistic for the reweighting problem, so the resulting histogram
compression is exact rather than approximate: unlike WHAM, no binning bias is
incurred, while the statistical efficiency of a binless estimator is retained.

The only input is, for every pH window, the sequence of protonation numbers
N(t) recorded along the trajectory (one integer per frame). Nothing else of
the simulation is needed: no energies, no coordinates, no site assignments.

Contents
--------
1. Histograms            build_histograms, block_histograms, partial_histograms,
                         histogram_moments, block_means
2. Correlation           statistical_inefficiency, block_length, blocking_scan
3. Estimator             solve_prime
4. Uncertainty           compute_prime_errors, resample_histograms, bootstrap,
                         estimator_profiles, transition_profiles
5. Diagnostics           overlap_matrix, connectivity_diagnostics, support_connected
6. Weights of N          log_weights, reweight, distribution,
                         stepwise_pka_from_weights, edge_weights
7. TI reference          compute_TI_trapezoid, compute_TI_hermite, fourth_cumulant,
                         ti_bias_trapezoid, ti_bias_hermite
8. Consistency test      stepwise_pka, consistency_test, format_consistency (Sec. IV F)
9. Residual              residual_decomposition                           (Sec. S10)
10. Coarsening           coarsening_rms                                   (Sec. IV D)
11. Prediction           predict, variance_maxima                         (Sec. IV E)
12. Object interface     PRIME

Conventions
-----------
* Free energies are returned as reduced (dimensionless) relative semi-grand
  potentials g_k = beta * (Upsilon_k - Upsilon_1), anchored at g[0] = 0,
  except in the TI routines where a thermal energy kT may be supplied.
* Histograms H have shape (K, Nmax + 1): one row per pH window, one column per
  protonation number N = 0, ..., Nmax. Build them with ``n_sites`` set to the
  number of titratable sites, so that the last column is the physical bound
  of N even when that bound was never sampled; the prediction rule needs it.
* The pH-independent weights of the protonation numbers, c(N) of Eq. 22 of
  the paper, are handled as ``lnc``, an array over N = 0, ..., Nmax of ln c(N),
  NaN where N was never sampled. Everything evaluable at an arbitrary pH
  (free energy, <N>, Var(N), Pr(N | pH), stepwise pKa) is a function of
  ``lnc`` alone.
* All Boltzmann sums are evaluated in log space. For large numbers of
  titratable sites the sums themselves exceed the range of double-precision
  arithmetic, whereas their logarithms do not.
* Uncertainties from the trajectory are obtained by a block bootstrap of the
  histograms: every window is cut into consecutive blocks, the histogram of
  each block is stored once (``block_histograms``), and a resample is the sum
  of as many blocks drawn with replacement. A block must be longer than the
  correlation time of N(t); ``block_length`` gives the minimum from the
  statistical inefficiency.
* Nothing here is specific to one simulation package or one molecule. The two
  states of the paper (native monomer and fibril) are analysed by calling the
  routines once per state; per-molecule comparisons divide the results of a
  state by its number of molecules, which scales g and the moments of N by
  1/n and the TI errors likewise.

Numerical note
--------------
The exponent range of IEEE-754 double precision is exceeded once an argument
of exp() reaches ~709. On systems of the size of an amyloid fibril of several molecules
the relative semi-grand potentials reach several thousand in reduced units, so
every Boltzmann sum here is evaluated through `scipy.special.logsumexp`, which
subtracts the running maximum internally. The same kernel is shared by the
solver, the covariance analysis and the weights so that
all three remain numerically consistent.
"""

from __future__ import annotations

import multiprocessing
import warnings
from collections import namedtuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp
from scipy.stats import chi2

__all__ = [
    # histograms
    "build_histograms", "block_histograms", "partial_histograms", "histogram_moments",
    "block_means",
    # correlation
    "statistical_inefficiency", "block_length", "blocking_scan",
    # estimator and uncertainty
    "solve_prime", "compute_prime_errors", "resample_histograms", "bootstrap",
    "estimator_profiles", "transition_profiles",
    # diagnostics
    "overlap_matrix", "connectivity_diagnostics", "support_connected",
    # weights of the protonation numbers
    "log_weights", "reweight", "distribution", "stepwise_pka_from_weights",
    "edge_weights",
    # TI
    "compute_TI_trapezoid", "compute_TI_hermite", "fourth_cumulant",
    "ti_bias_trapezoid", "ti_bias_hermite",
    # analyses of the paper
    "stepwise_pka", "consistency_test", "format_consistency", "residual_decomposition",
    "coarsening_rms",
    "predict", "variance_maxima",
    # object interface
    "PRIME", "PRIMEWarning",
]

__author__ = "Stefan Hervø-Hansen"
__version__ = "1.0.0"

LN10 = np.log(10.0)


class PRIMEWarning(UserWarning):
    """Warnings of the solver (singular Hessian, non-convergence)."""


# ======================================================================
# 1. Histograms
# ======================================================================
def _records(data, pHs=None):
    """Normalize the accepted input forms into a list of integer arrays and a pH array.

    Accepted forms of ``data``:
      * ``{pH: {"N": array}}`` or ``{pH: array}``; ``pHs`` selects and orders the
        windows (all keys, in their order, if omitted);
      * a sequence of arrays, one per window, in the order of ``pHs`` (required).
    """
    if isinstance(data, dict):
        keys = list(data.keys()) if pHs is None else list(pHs)
        arrays = []
        for key in keys:
            value = data[key]
            if isinstance(value, dict):
                value = value["N"]
            arrays.append(np.asarray(value, dtype=np.int64).ravel())
        return arrays, np.asarray(keys, dtype=float)
    if pHs is None:
        raise ValueError("pHs is required when the records are given as a sequence of arrays.")
    arrays = [np.asarray(value, dtype=np.int64).ravel() for value in data]
    if len(arrays) != len(pHs):
        raise ValueError("one array of protonation numbers per pH window is required.")
    return arrays, np.asarray(pHs, dtype=float)


def _n_columns(arrays, n_sites):
    """Number of histogram columns, Nmax + 1."""
    sampled = max(int(a.max()) for a in arrays if a.size)
    if n_sites is None:
        return sampled + 1
    if sampled > n_sites:
        raise ValueError(f"a protonation number {sampled} exceeds n_sites = {n_sites}.")
    return int(n_sites) + 1


def build_histograms(data, pHs=None, n_sites=None):
    """Compress protonation-number trajectories onto a common coordinate.

    Each window's records are binned on the shared protonation-number
    coordinate N = 0, ..., Nmax, with protonation numbers not sampled in a
    given window assigned zero counts, so that rows are directly summable.

    Parameters
    ----------
    data : dict or sequence
        ``data[pH]["N"]`` or ``data[pH]`` gives the protonation numbers sampled
        in the window at that pH, or a sequence of such arrays in the order of
        ``pHs``.
    pHs : array_like, shape (K,), optional
        Simulated pH of each of the K windows (selects and orders the dict keys).
    n_sites : int, optional
        Number of titratable sites. Sets Nmax = n_sites, so that the physical
        bounds of N are the first and the last column even when they were not
        sampled. Without it Nmax is the largest sampled N.

    Returns
    -------
    H : ndarray of int64, shape (K, Nmax + 1)
        Per-window protonation-number histograms. The aggregated histogram
        H_tot(N) of the estimator is ``H.sum(axis=0)``.
    """
    arrays, _ = _records(data, pHs)
    nb = _n_columns(arrays, n_sites)
    H = np.zeros((len(arrays), nb), dtype=np.int64)
    for i, a in enumerate(arrays):
        H[i] = np.bincount(a, minlength=nb)
    return H


def block_histograms(data, pHs=None, n_blocks=None, block_length=None, n_sites=None):
    """Histograms of consecutive blocks of every window, for the block bootstrap.

    Every window is cut into ``n_blocks`` consecutive blocks of (nearly) equal
    length, or into blocks of at least ``block_length`` frames (then the
    shortest window sets the number of blocks), and the histogram of each
    block is stored. Resampling blocks rather than frames keeps the
    within-block correlation, and a resample of a window costs a sum of
    ``n_blocks`` histograms instead of a pass over the frames. The blocks must
    be longer than the correlation time of N(t); see ``block_length``.

    Returns
    -------
    h_blk : ndarray of int64, shape (K, n_blocks, Nmax + 1)
        ``h_blk.sum(axis=1)`` is the histogram matrix of ``build_histograms``.
    """
    arrays, _ = _records(data, pHs)
    if (n_blocks is None) == (block_length is None):
        raise ValueError("give either n_blocks or block_length.")
    if n_blocks is None:
        n_blocks = min(len(a) for a in arrays) // int(block_length)
    nb = _n_columns(arrays, n_sites)
    h_blk = np.zeros((len(arrays), n_blocks, nb), dtype=np.int64)
    for k, a in enumerate(arrays):
        if len(a) < n_blocks:
            raise ValueError(f"window {k} has {len(a)} frames, fewer than n_blocks = {n_blocks}.")
        for b, chunk in enumerate(np.array_split(a, n_blocks)):
            h_blk[k, b] = np.bincount(chunk, minlength=nb)
    return h_blk


def partial_histograms(h_blk, n_parts):
    """Histograms of consecutive parts of the trajectories (halves, quarters).

    Returns a list of ``n_parts`` histogram matrices, each from a contiguous
    group of blocks, for convergence checks: profiles from the first and the
    second half, or from the four quarters, of every window.
    """
    K, B, nb = h_blk.shape
    edges = np.linspace(0, B, n_parts + 1).astype(int)
    return [h_blk[:, edges[i]:edges[i + 1], :].sum(axis=1) for i in range(n_parts)]


WindowMoments = namedtuple("WindowMoments", "M mean var kappa4")


def histogram_moments(H):
    """Per-window number of frames, <N>, Var(N) and the fourth cumulant of N.

    The moments of a discrete variable are exact functions of its histogram,
    so the TI estimators and their biases can be evaluated from H alone.
    """
    H = np.asarray(H, dtype=float)
    N = np.arange(H.shape[1], dtype=float)
    M = H.sum(axis=1)
    mean = (H * N[None, :]).sum(axis=1) / M
    d = N[None, :] - mean[:, None]
    mu2 = (H * d ** 2).sum(axis=1) / M
    mu4 = (H * d ** 4).sum(axis=1) / M
    return WindowMoments(M, mean, mu2, mu4 - 3.0 * mu2 ** 2)


def block_means(h_blk):
    """Mean protonation number of every block, shape (K, n_blocks): the time
    series of block averages of N(t) in each window."""
    h_blk = np.asarray(h_blk, dtype=float)
    N = np.arange(h_blk.shape[2], dtype=float)
    return (h_blk * N).sum(axis=2) / h_blk.sum(axis=2)


# ======================================================================
# 2. Correlation of the records and block length
# ======================================================================
def statistical_inefficiency(x, max_lag=None):
    """Statistical inefficiency g = 1 + 2*sum_t C(t), Chodera-style truncation.

    The effective number of independent samples is M/g, so the true standard
    error exceeds the naive one by sqrt(g).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    v = np.dot(x, x) / n
    if v == 0:
        return 1.0
    max_lag = min(n // 4, 2000) if max_lag is None else max_lag
    g, t = 1.0, 1
    while t < max_lag:
        c = np.dot(x[:n - t], x[t:]) / ((n - t) * v)
        if c <= 0:
            break
        g += 2.0 * c * (1.0 - t / n)
        t += 1
    return max(1.0, g)


def block_length(data, pHs=None, factor=3):
    """Minimum block length for the bootstrap, ``factor`` times the largest
    statistical inefficiency of N(t) over the windows, in frames.

    Returns (block_length, g) with g the inefficiency of every window.
    """
    arrays, _ = _records(data, pHs)
    g = np.array([statistical_inefficiency(a) for a in arrays])
    return int(factor * np.ceil(g.max())), g


def blocking_scan(x, lengths=None, min_blocks=4):
    """Flyvbjerg-Petersen scan: standard error of the mean of x from blocks of
    increasing length.

    The error rises with the block length until the blocks are longer than
    the slowest correlated mode, then plateaus; the plateau is the block length
    the bootstrap needs. A curve that keeps rising up to the longest blocks is
    a drift, or a mode slower than the trajectory. ``x`` may be the records of
    a window or its block means (``block_means``), in which case the lengths
    count blocks.

    Returns (lengths, se), with se NaN where fewer than ``min_blocks`` blocks fit.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lengths is None:
        lengths = [2 ** i for i in range(int(np.log2(max(n // min_blocks, 1))) + 1)]
    lengths = np.asarray(lengths, dtype=int)
    se = np.full(len(lengths), np.nan)
    for i, L in enumerate(lengths):
        nb = n // L
        if nb < min_blocks:
            continue
        bm = x[:nb * L].reshape(nb, L).mean(axis=1)
        se[i] = np.sqrt(bm.var(ddof=1) / nb)
    return lengths, se


# ======================================================================
# 3. Shared numerically stable assignment kernel and the estimator
# ======================================================================
def _prime_log_probs(g, a, logM, N):
    """Log posterior assignment probabilities ln Pr(k | N).

        exponent[k, N] = ln M_k + g_k - a_k N,      a_k = ln(10) pH_k
        ln D(N)        = logsumexp_k exponent[k, N]
        ln Pr(k | N)   = exponent[k, N] - ln D(N)

    Returns
    -------
    log_Pr : ndarray, shape (K, Nmax+1); finite everywhere and <= 0
    log_denom : ndarray, shape (Nmax+1,); the Boltzmann denominator ln D(N)
    """
    expo = logM[:, None] + g[:, None] - np.outer(a, N)
    log_denom = logsumexp(expo, axis=0)
    return expo - log_denom[None, :], log_denom


def solve_prime(H, pHs, M=None, tol=1e-10, max_iter=10000, verbose=False,
                mix=0.1, method="newton", g_init=None):
    """Solve the PRIME equations for the relative semi-grand potentials.

    The negative log-likelihood is strictly convex whenever the pH ensembles
    form a connected overlap network, so the solution is unique. A safeguarded
    Newton iteration with backtracking line search is used by default; a
    stabilized log-space fixed-point path is retained for cross-checking.
    Without an initial guess the iteration is seeded from the thermodynamic-
    integration estimate, which requires no inter-window overlap and therefore
    remains usable when the reweighting problem is stiff.

    Parameters
    ----------
    H : ndarray, shape (K, Nmax + 1)
        Per-window histograms from `build_histograms`. Only the aggregated
        histogram enters the estimator; the full matrix is used for the
        log-likelihood in the line search and for the TI seed.
    pHs : array_like, shape (K,)
    M : array_like, shape (K,), optional
        Configurations per window. Defaults to ``H.sum(axis=1)``.
    tol : float
        Convergence tolerance on max|delta g| per iteration.
    verbose : bool
        Print the progress of the iteration.
    method : {'newton', 'fixedpoint'}
    g_init : array_like, shape (K,), optional
        Initial guess; defaults to the internal TI seed.

    Returns
    -------
    g : ndarray, shape (K,)
        Relative reduced semi-grand potentials, anchored at ``g[0] = 0``.
        A singular Hessian (disconnected overlap network) and a failure to
        converge are reported through ``PRIMEWarning``.
    """
    pHs = np.asarray(pHs, dtype=float)
    H = np.asarray(H, dtype=float)
    M = H.sum(axis=1) if M is None else np.asarray(M, dtype=float)

    if np.any(M <= 0):
        raise ValueError("Every window must contain at least one configuration "
                         "(all M_k > 0) for the potentials to be defined.")

    N = np.arange(H.shape[1], dtype=float)
    Htot = H.sum(axis=0)
    a = LN10 * pHs
    logM = np.log(M)
    with np.errstate(divide="ignore"):
        logHtot = np.log(Htot)          # -inf on unsampled N; drops out
    aN = np.outer(a, N)

    g = (_ti_seed(H, N, M, pHs) if g_init is None
         else np.asarray(g_init, dtype=float).copy())
    g = g - g[0]

    def neg_log_likelihood(logPr):
        return -np.sum(H * logPr)

    for it in range(max_iter):
        g_old = g.copy()
        grad_red = None

        if method == "fixedpoint":
            _, log_denom = _prime_log_probs(g, a, logM, N)
            terms = logHtot[None, :] - aN - log_denom[None, :]
            g_new = -logsumexp(terms, axis=1)
            g_new -= g_new[0]
            g = (1.0 - mix) * g + mix * g_new
            g -= g[0]

        elif method == "newton":
            logPr, _ = _prime_log_probs(g, a, logM, N)
            Pr = np.exp(logPr)
            PrH = Pr * Htot[None, :]

            grad = M - PrH.sum(axis=1)
            Hess = np.diag(PrH.sum(axis=1)) - Pr @ PrH.T
            Hess_red, grad_red = Hess[1:, 1:], grad[1:]

            try:
                delta = np.linalg.solve(Hess_red, grad_red)
            except np.linalg.LinAlgError:
                delta, ridge = _ridge_solve(Hess_red, grad_red)
                warnings.warn(f"iteration {it}: singular Hessian, ridge {ridge:.1e}; the overlap "
                              "network is disconnected and the affected potentials are unresolved.",
                              PRIMEWarning, stacklevel=2)

            nll0 = neg_log_likelihood(logPr)
            step, g_next = 1.0, g
            while step > mix * 1e-6:
                cand = g.copy()
                cand[1:] += step * delta
                cand -= cand[0]
                logPr_c, _ = _prime_log_probs(cand, a, logM, N)
                if neg_log_likelihood(logPr_c) <= nll0:
                    g_next = cand
                    break
                step *= 0.5
            g = g_next

        else:
            raise ValueError("method must be 'newton' or 'fixedpoint'")

        delta_g = np.max(np.abs(g - g_old))
        if verbose and (it % 100 == 0 or delta_g < tol):
            msg = f"iter {it:5d}  max|dg| = {delta_g:.3e}"
            if grad_red is not None:
                msg += f"   ||grad||inf = {np.max(np.abs(grad_red)):.3e}"
            print(msg)
        if delta_g < tol:
            if verbose:
                print(f"Converged in {it} iterations.")
            break
    else:
        warnings.warn("PRIME did not reach the convergence tolerance. Seed g_init with a TI "
                      "estimate and/or inspect the overlap network with connectivity_diagnostics().",
                      PRIMEWarning, stacklevel=2)

    return g


def _ti_seed(H, N, M, pHs):
    """Thermodynamic-integration initial guess from the histograms.

    Uses d(beta DeltaUpsilon)/d(pH) = ln(10) <N>(pH) with cumulative
    trapezoidal integration. TI requires no inter-window overlap, so this
    remains well defined even when the reweighting problem is stiff.
    """
    Nmean = (H * N[None, :]).sum(axis=1) / M
    order = np.argsort(pHs)
    pH_s, Nmean_s = pHs[order], Nmean[order]
    integ = np.concatenate(
        [[0.0], np.cumsum(0.5 * (Nmean_s[1:] + Nmean_s[:-1]) * np.diff(pH_s))])
    g = np.empty(len(pHs))
    g[order] = LN10 * integ
    return g


def _ridge_solve(A, b, base=1e-10, factor=10.0, max_tries=12):
    """Levenberg-style fallback for a singular reduced Hessian.

    A singular reduced Hessian indicates that one or more relative potentials
    are not identifiable, i.e. the overlap network is disconnected. The ridge
    permits the iteration to proceed, but the affected potentials should be
    treated as unresolved; the remedy is finer pH spacing, not more ridge.
    """
    n = A.shape[0]
    scale = np.trace(A) / max(n, 1)
    ridge = base * scale if scale > 0 else base
    for _ in range(max_tries):
        try:
            return np.linalg.solve(A + ridge * np.eye(n), b), ridge
        except np.linalg.LinAlgError:
            ridge *= factor
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x, ridge


# ======================================================================
# 4. Uncertainty: asymptotic covariance and block bootstrap
# ======================================================================
def compute_prime_errors(g, H, pHs, M=None):
    """Asymptotic maximum-likelihood uncertainties of the PRIME potentials.

    The observed information matrix is built from the converged assignment
    probabilities. Because window 0 is the reference (g[0] = 0), the
    covariance follows from the (K-1) x (K-1) active block.

    Note that the observed-information approximation treats the numbers of
    configurations per window as random. In simulation they are fixed by
    design, so the resulting intervals are conservative. They also treat the
    frames as independent; for correlated records use the block bootstrap.

    Returns
    -------
    sigma_g : ndarray, shape (K,); standard errors, sigma_g[0] = 0
    Sigma : ndarray, shape (K-1, K-1); covariance of g_2, ..., g_K
    """
    g = np.asarray(g, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    H = np.asarray(H, dtype=float)
    M = H.sum(axis=1) if M is None else np.asarray(M, dtype=float)

    N = np.arange(H.shape[1], dtype=float)
    Htot = H.sum(axis=0)

    logPr, _ = _prime_log_probs(g, LN10 * pHs, np.log(M), N)
    Pr = np.exp(logPr)

    PrH = Pr * Htot[None, :]
    Hessian = np.diag(PrH.sum(axis=1)) - Pr @ PrH.T

    try:
        Sigma = np.linalg.inv(Hessian[1:, 1:])
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "The active Hessian is singular. The overlap network is "
            "disconnected; inspect it with connectivity_diagnostics()."
        ) from exc

    sigma_g = np.zeros(len(pHs), dtype=float)
    sigma_g[1:] = np.sqrt(np.diag(Sigma))
    return sigma_g, Sigma


def resample_histograms(h_blk, rng, n=1, aligned=False):
    """Block-bootstrap resamples of the histogram matrix.

    Every window is rebuilt from ``n_blocks`` of its blocks drawn with
    replacement. With ``aligned=True`` the same block indices are drawn for
    all windows, which keeps the coupling in time between the replicas of a
    pH replica-exchange simulation; by default every window is drawn
    independently.

    Returns
    -------
    Hs : ndarray of int64, shape (n, K, Nmax + 1)
    """
    h_blk = np.asarray(h_blk)
    K, B, nb = h_blk.shape
    if aligned:
        draws = np.repeat(rng.integers(0, B, size=(n, 1, B)), K, axis=1)
    else:
        draws = rng.integers(0, B, size=(n, K, B))
    counts = np.zeros((n, K, B), dtype=np.int64)
    np.add.at(counts, (np.arange(n)[:, None, None], np.arange(K)[None, :, None], draws), 1)
    return np.einsum("nkb,kbm->nkm", counts, h_blk)


def estimator_profiles(H, pHs, **solver):
    """Free-energy profiles of the three estimators from one histogram matrix.

    Returns an array of shape (3, K): PRIME, trapezoidal TI and Hermite TI,
    all relative to the first window. Ready for ``bootstrap``.
    """
    H = np.asarray(H, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    mom = histogram_moments(H)
    gp = solve_prime(H, pHs, mom.M, **solver)
    gt, _ = compute_TI_trapezoid(pHs, mom.mean, mom.var, mom.M)
    gh, _ = compute_TI_hermite(pHs, mom.mean, mom.var, mom.M)
    return np.stack([gp - gp[0], gt - gt[0], gh - gh[0]])


def transition_profiles(Hs, pHs, nmol=(1, 1), **solver):
    """Free energy of the transition A -> B per molecule from the three estimators,

        ddY(pH) = dY_B(pH) / n_B - dY_A(pH) / n_A ,

    with ``Hs = [H_A, H_B]`` and ``nmol = (n_A, n_B)`` the numbers of molecules
    in the two simulated systems. Returns (3, K): PRIME, trapezoidal TI,
    Hermite TI, relative to the first window. Ready for ``bootstrap`` with
    ``h_blk = [h_blk_A, h_blk_B]``.
    """
    (H_A, H_B), (n_A, n_B) = Hs, nmol
    return estimator_profiles(H_B, pHs, **solver) / n_B - estimator_profiles(H_A, pHs, **solver) / n_A


_BOOT = {}                                             # context of a parallel bootstrap, inherited by forked workers


def _boot_one(seed):
    """One resample of every system and the value of the function on it (None on failure)."""
    blocks, pHs, func, kwargs, several, aligned = _BOOT["ctx"]
    rng = np.random.default_rng(seed)
    Hs = [resample_histograms(h, rng, aligned=aligned)[0] for h in blocks]
    if not all(support_connected(Hr) for Hr in Hs):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PRIMEWarning)
        try:
            value = np.asarray(func(Hs if several else Hs[0], pHs, **kwargs), dtype=float)
        except (np.linalg.LinAlgError, ValueError):
            return None
    return value if np.all(np.isfinite(value)) else None


def _boot_chunk(seeds):
    return [_boot_one(s) for s in seeds]


def bootstrap(h_blk, pHs, func, n_boot=200, seed=0, ci=0.95, aligned=False, nprocs=1, **kwargs):
    """Block bootstrap of any function of the histograms.

    Parameters
    ----------
    h_blk : ndarray (K, n_blocks, Nmax + 1), or a sequence of them
        Block histograms of one system, or of several systems that ``func``
        combines (for example the native and the fibril state).
    pHs : array_like, shape (K,)
    func : callable
        ``func(H, pHs, **kwargs)`` for one system, ``func([H_1, H_2, ...], pHs,
        **kwargs)`` for several, returning an array. See ``estimator_profiles``
        and ``transition_profiles``; ``lambda H, p: histogram_moments(H).mean``
        gives the titration curve.
    n_boot : int
        Number of resamples. Every resample has its own seed derived from
        ``seed``, so the result does not depend on ``nprocs``.
    ci : float
        Coverage of the percentile interval returned.
    nprocs : int or None
        Worker processes (None for all cores). Parallel runs use forked
        workers, which inherit ``func`` even when it is a lambda; where fork
        is unavailable the bootstrap runs in this process.

    Returns
    -------
    nominal : the value of ``func`` on the full histograms
    boots : ndarray, shape (n_ok, ...), the values on the resamples that
        succeeded (a resample whose windows share no sampled protonation
        number, or on which ``func`` raises or returns NaN, is dropped; compare
        ``len(boots)`` with ``n_boot``)
    lo, hi : percentile bounds of the interval, elementwise
    """
    several = isinstance(h_blk, (list, tuple))
    blocks = [np.asarray(h) for h in h_blk] if several else [np.asarray(h_blk)]
    pHs = np.asarray(pHs, dtype=float)
    full = [h.sum(axis=1) for h in blocks]
    nominal = np.asarray(func(full if several else full[0], pHs, **kwargs), dtype=float)

    seeds = np.random.SeedSequence(seed).spawn(n_boot)
    _BOOT["ctx"] = (blocks, pHs, func, kwargs, several, aligned)
    try:
        if nprocs == 1 or "fork" not in multiprocessing.get_all_start_methods():
            if nprocs != 1:
                warnings.warn("fork is not available; the bootstrap runs in this process.", stacklevel=2)
            results = _boot_chunk(seeds)
        else:
            nprocs = multiprocessing.cpu_count() if nprocs is None else int(nprocs)
            chunks = [seeds[i::nprocs] for i in range(nprocs)]
            with multiprocessing.get_context("fork").Pool(nprocs) as pool:
                parts = pool.map(_boot_chunk, chunks)
            results = [None] * n_boot
            for i, part in enumerate(parts):                      # back into the order of the seeds
                results[i::nprocs] = part
    finally:
        _BOOT.pop("ctx", None)
    boots = np.array([v for v in results if v is not None])
    if len(boots) == 0:
        raise RuntimeError("no bootstrap resample succeeded.")
    alpha = 100.0 * (1.0 - ci) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    return nominal, boots, lo, hi


# ======================================================================
# 5. Overlap and connectivity diagnostics
# ======================================================================
def overlap_matrix(g, H, pHs, M=None):
    """Overlap between pH ensembles in protonation-number space.

        Theta_kl = sum_N H_tot(N) Pr(k|N) Pr(l|N)

    Theta is symmetric and non-negative, and at the stationary point its row
    sums equal M_k, so Theta_kl measures the effective number of sampled
    configurations shared between ensembles k and l. The Hessian of the
    negative log-likelihood is the weighted graph Laplacian of this matrix,
    H = diag(Theta 1) - Theta.
    """
    g = np.asarray(g, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    H = np.asarray(H, dtype=float)
    M = H.sum(axis=1) if M is None else np.asarray(M, dtype=float)

    N = np.arange(H.shape[1], dtype=float)
    Htot = H.sum(axis=0)
    logPr, _ = _prime_log_probs(g, LN10 * pHs, np.log(M), N)
    Pr = np.exp(logPr)
    return (Pr * Htot[None, :]) @ Pr.T


def connectivity_diagnostics(g, H, pHs, M=None, min_shared=1.0):
    """Identifiability diagnostics for the multi-pH ensemble.

    The PRIME solution is unique if and only if the overlap network is
    connected, i.e. every ensemble is linked to the reference through a chain
    of ensembles sharing sampled protonation numbers. Connectivity is assessed
    here with the overlap thresholded at `min_shared` effective configurations,
    since an overlap corresponding to less than one shared configuration
    carries no statistical information.

    Returns
    -------
    dict with keys
        n_components : number of connected components (1 if identifiable)
        labels : component index of each window
        algebraic_connectivity : second-smallest eigenvalue of the Laplacian;
            a quantitative measure of how well linked the network is
        fiedler_vector : eigenvector of that eigenvalue, which localizes the
            weakest cut and hence indicates where to add pH windows
        min_eig_reduced : smallest eigenvalue of the reduced Hessian
        theta : the overlap matrix itself
    """
    Theta = overlap_matrix(g, H, pHs, M)
    L = np.diag(Theta.sum(axis=1)) - Theta

    adjacency = (Theta > min_shared).astype(int)
    np.fill_diagonal(adjacency, 0)
    ncomp, labels = connected_components(csr_matrix(adjacency), directed=False)

    ev, evec = np.linalg.eigh(L)
    return dict(n_components=int(ncomp), labels=labels,
                algebraic_connectivity=float(ev[1]),
                fiedler_vector=evec[:, 1],
                min_eig_reduced=float(np.linalg.eigvalsh(L[1:, 1:]).min()),
                theta=Theta)


def support_connected(H):
    """Necessary condition for identifiability, from the histograms alone.

    True if the windows form one component when two windows are linked
    whenever they share a sampled protonation number. Cheap, needs no fit, and
    sufficient to decide that a fit is pointless; the fitted overlap network of
    ``connectivity_diagnostics`` is the stricter test.
    """
    S = (np.asarray(H) > 0).astype(int)
    adjacency = (S @ S.T) > 0
    return connected_components(csr_matrix(adjacency), directed=False)[0] == 1


# ======================================================================
# 6. pH-independent weights of the protonation numbers
# ======================================================================
def log_weights(g, H, pHs, M=None):
    """Logarithm of the pH-independent weights of the protonation numbers,

        ln c(N) = ln H_tot(N) - ln D(N) ,

    where D(N) is the Boltzmann denominator already evaluated by the solver
    (Eq. 22 of the paper). The weights are thus a by-product of the converged
    solution. They are the sum of states g(N) up to one constant, which cancels
    in every quantity derived from them.

    Returns
    -------
    lnc : ndarray, shape (Nmax + 1,); ln c(N), NaN where N was never sampled.
        The logarithm is returned rather than c(N) because for large numbers
        of titratable sites the weights exceed the range of double-precision
        arithmetic while their logarithms do not.
    """
    g = np.asarray(g, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    H = np.asarray(H, dtype=float)
    M = H.sum(axis=1) if M is None else np.asarray(M, dtype=float)

    N = np.arange(H.shape[1], dtype=float)
    Htot = H.sum(axis=0)
    _, log_denom = _prime_log_probs(g, LN10 * pHs, np.log(M), N)

    ok = Htot > 0
    lnc = np.full(H.shape[1], np.nan)
    lnc[ok] = np.log(Htot[ok]) - log_denom[ok]
    return lnc


def _weights_at(lnc, pH_grid):
    """Pr(N | pH) over the sampled N for every pH of the grid.

    Returns (Nn, lp, w): the sampled protonation numbers, the log weights of
    shape (P, n_ok) and the normalized weights.
    """
    lnc = np.asarray(lnc, dtype=float)
    ok = np.isfinite(lnc)
    Nn = np.flatnonzero(ok).astype(float)
    lp = lnc[ok][None, :] - np.outer(LN10 * np.atleast_1d(np.asarray(pH_grid, dtype=float)), Nn)
    return Nn, lp, np.exp(lp - logsumexp(lp, axis=1)[:, None])


def reweight(lnc, pH_grid, pH_ref=None):
    """Free energy, <N> and Var(N) at arbitrary pH from the weights.

    Evaluable at any pH, including values that were not simulated, provided
    the protonation numbers dominant there are populated in the aggregated
    histogram (``predict`` decides whether that is the case).

    Returns
    -------
    dUps : reduced semi-grand potential, relative to pH_ref if given
    Nav : average protonation number
    varN : protonation variance (the buffer capacity, up to -1/ln 10)
    """
    Nn, lp, w = _weights_at(lnc, pH_grid)
    bU = -logsumexp(lp, axis=1)
    if pH_ref is not None:
        bU = bU - (-logsumexp(np.asarray(lnc)[np.isfinite(lnc)] - Nn * LN10 * pH_ref))
    Nav = w @ Nn
    varN = (w * (Nn[None, :] - Nav[:, None]) ** 2).sum(axis=1)     # two-pass form
    return bU, Nav, varN


def distribution(lnc, pH):
    """Conditional protonation-number distribution Pr(N | pH) at one pH, over
    the sampled N. Returns (N, Pr)."""
    Nn, _, w = _weights_at(lnc, pH)
    return Nn, w[0]


def stepwise_pka_from_weights(lnc):
    """Stepwise pKa of the steps N-1 -> N from the weights (Eq. 25 of the paper),

        pK_N = [ln c(N) - ln c(N-1)] / ln 10 ,

    NaN where either protonation number is unsampled. Index j is the step
    j -> j + 1.
    """
    lnc = np.asarray(lnc, dtype=float)
    return (lnc[1:] - lnc[:-1]) / LN10


def edge_weights(lnc, pH_grid):
    """Weight that Pr(N | pH) places on the lowest and the highest sampled N.

    Returns (w_lo, w_hi, N_lo, N_hi): the weights at every pH of the grid and
    the two protonation numbers. A large weight on an edge that is not a
    physical bound of N means that the distribution at that pH extends beyond
    the sampled range, and the prediction there is undetermined.
    """
    Nn, _, w = _weights_at(lnc, pH_grid)
    return w[:, 0], w[:, -1], int(Nn[0]), int(Nn[-1])


# ======================================================================
# 7. Thermodynamic integration reference estimators
# ======================================================================
def fourth_cumulant(x):
    """Fourth cumulant of a sample, kappa_4 = mu_4 - 3 mu_2^2.

    Required to evaluate the leading discretization bias of the cubic-Hermite
    TI estimator, since in the semi-grand ensemble the third derivative of the
    titration curve is d^3<N>/d(pH)^3 = -ln(10)^3 kappa_4. ``histogram_moments``
    gives the same quantity from the histogram.
    """
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    return float((d ** 4).mean() - 3.0 * (d ** 2).mean() ** 2)


def _as_arrays(pHs, N_avg, N_var, M_samples):
    pHs = np.asarray(pHs, dtype=float)
    N_avg = np.asarray(N_avg, dtype=float)
    N_var = np.asarray(N_var, dtype=float)
    M_samples = (np.full_like(pHs, M_samples) if np.isscalar(M_samples)
                 else np.asarray(M_samples, dtype=float))
    if not np.allclose(np.diff(pHs), pHs[1] - pHs[0]):
        raise ValueError("pH grid must be uniformly spaced.")
    return pHs, N_avg, N_var, M_samples


def _stat_error(pHs, N_var, M_samples, kT):
    """Propagated statistical error of the cumulative trapezoidal integral.

    The trapezoidal weights for the partial integral from pH_0 to pH_i are
    1/2 at the two endpoints and 1 in between.
    """
    n = len(pHs)
    dpH = pHs[1] - pHs[0]
    se = np.zeros(n)
    for i in range(1, n):
        w = np.ones(i + 1)
        w[0] = 0.5
        w[-1] = 0.5
        se[i] = kT * LN10 * dpH * np.sqrt(
            np.sum(w ** 2 * N_var[:i + 1] / M_samples[:i + 1]))
    return se


def ti_bias_trapezoid(pHs, N_var, kT=1.0):
    """Signed leading Euler-Maclaurin bias of the trapezoidal rule, exact minus
    trapezoid, on a uniform grid: kT (ln10 dpH)^2 / 12 [Var(N)_k - Var(N)_0]."""
    pHs = np.asarray(pHs, dtype=float)
    N_var = np.asarray(N_var, dtype=float)
    return kT * (LN10 * (pHs[1] - pHs[0])) ** 2 / 12.0 * (N_var - N_var[0])


def ti_bias_hermite(pHs, N_kappa4, kT=1.0):
    """Signed leading bias of the cubic-Hermite rule, exact minus Hermite:
    -kT (ln10 dpH)^4 / 720 [kappa_4(pH_k) - kappa_4(pH_0)]."""
    pHs = np.asarray(pHs, dtype=float)
    N_kappa4 = np.asarray(N_kappa4, dtype=float)
    return -kT * (LN10 * (pHs[1] - pHs[0])) ** 4 / 720.0 * (N_kappa4 - N_kappa4[0])


def compute_TI_trapezoid(pHs, N_avg, N_var, M_samples, kT=1.0,
                         include_discretization=False):
    """Thermodynamic integration of <N> along pH by the trapezoidal rule.

    Uses only the first moment of the protonation distribution. The leading
    discretization error is O(dpH^2) and is a systematic bias rather than a
    random error; it is included in g_err only if include_discretization is
    True, in which case it is added linearly to the statistical standard
    error rather than in quadrature.

    Returns
    -------
    g : cumulative free-energy profile, g[0] = 0
    g_err : standard error profile
    """
    pHs, N_avg, N_var, M_samples = _as_arrays(pHs, N_avg, N_var, M_samples)

    g = kT * LN10 * cumulative_trapezoid(N_avg, pHs, initial=0.0)
    g -= g[0]

    g_err = _stat_error(pHs, N_var, M_samples, kT)
    if include_discretization:
        g_err = g_err + np.abs(ti_bias_trapezoid(pHs, N_var, kT))
    return g, g_err


def compute_TI_hermite(pHs, N_avg, N_var, M_samples, kT=1.0,
                       N_kappa4=None, include_discretization=False):
    """Thermodynamic integration with cubic-Hermite interpolation of <N>.

    Uses the first and second moments of the protonation distribution: the
    slope of the titration curve is known analytically from the protonation
    variance,

        d<N>/d(pH) = -ln(10) Var(N),

    so <N>(pH) is interpolated by a cubic Hermite spline and integrated
    exactly. On a uniform grid this is identical to the trapezoidal result
    plus the leading Euler-Maclaurin term, so the O(dpH^2) discretization bias
    is removed rather than reported.

    The residual bias is the next Euler-Maclaurin term, which in the
    semi-grand ensemble is governed by the fourth cumulant of the protonation
    number,

        bias = -kT (ln(10) dpH)^4 / 720 * [kappa_4(pH_k) - kappa_4(pH_0)],

    since d^3<N>/d(pH)^3 = -ln(10)^3 kappa_4. Supply N_kappa4 (see
    `fourth_cumulant` or `histogram_moments`) and set include_discretization
    to include it.

    Returns
    -------
    g : cumulative free-energy profile, g[0] = 0
    g_err : standard error profile
    """
    pHs, N_avg, N_var, M_samples = _as_arrays(pHs, N_avg, N_var, M_samples)

    ch = CubicHermiteSpline(pHs, N_avg, -LN10 * N_var)
    g = kT * LN10 * np.array([ch.integrate(pHs[0], x) for x in pHs])
    g -= g[0]

    # The statistical error is carried by <N>, whose weights are unchanged.
    # The contribution from the uncertainty in Var(N) enters at O(dpH^2) and
    # is neglected.
    g_err = _stat_error(pHs, N_var, M_samples, kT)

    if include_discretization:
        if N_kappa4 is None:
            raise ValueError(
                "N_kappa4 is required to evaluate the Hermite discretization "
                "bias; compute it with fourth_cumulant() for each window.")
        g_err = g_err + np.abs(ti_bias_hermite(pHs, N_kappa4, kT))
    return g, g_err


# ======================================================================
# 8. Consistency of the pH windows (stepwise pKa test)
# ======================================================================
def stepwise_pka(H, pHs, h_blk=None, n_boot=400, seed=0, min_count=200, max_se=0.1):
    """Stepwise pKa of every step N-1 -> N as measured in every window alone,

        pK_{N,k} = pH_k + log10 [H_k(N) / H_k(N-1)] ,

    with its block-bootstrap standard error. A window measures a step only if
    both counts reach ``min_count`` and the error stays below ``max_se`` pKa
    units; the other entries are NaN.

    Without block histograms the error is the Poisson counting error, which
    ignores the correlation of the records and is a lower bound.

    Returns
    -------
    pK, se : ndarray, shape (K, Nmax); column j is the step j -> j + 1
    usable : bool ndarray, shape (K, Nmax)
    """
    H = np.asarray(H, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    usable = (H[:, 1:] >= min_count) & (H[:, :-1] >= min_count)
    with np.errstate(divide="ignore", invalid="ignore"):
        pK = np.log10(H[:, 1:] / H[:, :-1]) + pHs[:, None]
        if h_blk is not None:
            rng = np.random.default_rng(seed)
            pKb = []
            for start in range(0, n_boot, 50):                        # in chunks, the resamples can be many and large
                Hb = resample_histograms(h_blk, rng, n=min(50, n_boot - start)).astype(float)
                pKb.append(np.log10(Hb[:, :, 1:] / Hb[:, :, :-1]) + pHs[None, :, None])
            pKb = np.concatenate(pKb)
            pKb = np.where(np.isfinite(pKb), pKb, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                se = np.nanstd(pKb, axis=0, ddof=1)
        else:
            se = np.sqrt(1.0 / H[:, 1:] + 1.0 / H[:, :-1]) / LN10
    usable &= np.isfinite(pK) & np.isfinite(se) & (se > 0) & (se <= max_se)
    pK = np.where(usable, pK, np.nan)
    se = np.where(usable, se, np.nan)
    return pK, se, usable


def consistency_test(H, pHs, h_blk=None, n_boot=400, seed=0, min_count=200,
                     max_se=0.1, tau_flag=0.01, **solver):
    """Do the pH windows sample the same ensemble? (Section IV F of the paper.)

    Two windows that both sample the protonation numbers N-1 and N must agree
    on the stepwise pKa of that step, since it is a property of the molecule
    and not of the window. For every step the per-window values are combined
    into a weighted mean, with weights 1/se^2, and their scatter beyond the statistical
    errors is summarized by Cochran's Q and the excess standard deviation tau
    of DerSimonian and Laird,

        tau^2 = max[0, (Q - dof) / C],   C = sum w - sum w^2 / sum w .

    A step with tau above ``tau_flag`` (pKa units) is flagged: its windows
    disagree beyond their errors, which points to incomplete equilibration or
    to a slow mode in those windows. The threshold is absolute, so choose it
    against the per-window errors (the paper uses 0.01 with errors of about
    0.005); the p-value of Q is reported as well. The test uses only the
    sampled histograms and is independent of the free-energy estimate.

    Returns
    -------
    dict with keys
        steps : ndarray (Nmax,), the step index N of the steps N-1 -> N
        pK, se, usable : per-window values (see ``stepwise_pka``)
        n_windows : number of windows measuring each step
        pK_mean, pK_mean_se : weighted mean of the windows and its error
        chi2_red : Q / dof
        p_value : probability of a larger Q under consistent windows
        tau : excess scatter in pKa units
        pK_prime : the stepwise pKa implied by the full PRIME fit
        flagged : bool ndarray, tau > tau_flag
    """
    H = np.asarray(H, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    pK, se, usable = stepwise_pka(H, pHs, h_blk, n_boot, seed, min_count, max_se)
    n_steps = H.shape[1] - 1
    n_win = usable.sum(axis=0)
    mean = np.full(n_steps, np.nan)
    mean_se = np.full(n_steps, np.nan)
    chi2_red = np.full(n_steps, np.nan)
    p_value = np.full(n_steps, np.nan)
    tau = np.full(n_steps, np.nan)
    for j in range(n_steps):
        m = usable[:, j]
        if m.sum() < 2:
            continue
        r, w = pK[m, j], 1.0 / se[m, j] ** 2
        mu = (w * r).sum() / w.sum()
        Q = (w * (r - mu) ** 2).sum()
        dof = m.sum() - 1
        C = w.sum() - (w ** 2).sum() / w.sum()
        mean[j], mean_se[j] = mu, 1.0 / np.sqrt(w.sum())
        chi2_red[j] = Q / dof
        p_value[j] = chi2.sf(Q, dof)
        tau[j] = np.sqrt(max(0.0, (Q - dof) / C))
    lnc = log_weights(solve_prime(H, pHs, **solver), H, pHs)
    with np.errstate(invalid="ignore"):
        flagged = tau > tau_flag
    return dict(steps=np.arange(1, n_steps + 1), pK=pK, se=se, usable=usable, n_windows=n_win,
                pK_mean=mean, pK_mean_se=mean_se, chi2_red=chi2_red, p_value=p_value, tau=tau,
                pK_prime=stepwise_pka_from_weights(lnc), flagged=flagged)


def format_consistency(result, tau_flag=0.01):
    """The table of ``consistency_test`` as text, one line per step."""
    lines = [f"{'step':>7} {'windows':>8} {'pKa mean':>18} {'PRIME':>7} {'chi2/dof':>9} {'p(Q)':>6} {'tau':>7}"]
    for j, N in enumerate(result["steps"]):
        if result["n_windows"][j] < 2:
            continue
        lines.append(f"{N - 1:>3d}->{N:<3d} {result['n_windows'][j]:>8d} "
                     f"{result['pK_mean'][j]:>11.3f} +- {result['pK_mean_se'][j]:.3f} "
                     f"{result['pK_prime'][j]:>7.3f} {result['chi2_red'][j]:>9.2f} {result['p_value'][j]:>6.2f} "
                     f"{result['tau'][j]:>7.3f}"
                     + ("   <-- inconsistent" if result["tau"][j] > tau_flag else ""))
    return "\n".join(lines)


# ======================================================================
# 9. Decomposition of the PRIME - TI residual
# ======================================================================
def residual_decomposition(H, pHs, **solver):
    """Split the residual between PRIME and TI into quadrature and consistency.

    TI integrates the sampled window means <N>_k. PRIME reconciles all
    windows into one set of weights, which implies its own means <N>_g(pH_k)
    at the grid points. Evaluating TI on those implied means and variances
    instead of the sampled ones splits the residual exactly:

        PRIME - TI(sampled) = [PRIME - TI(g)] + [TI(g) - TI(sampled)]
                               quadrature part   consistency part

    The first term is the discretization error of the quadrature applied to
    the smooth curve PRIME implies, and should match the Euler-Maclaurin bias.
    The second is ln10 times the integral of [<N>_g - <N>_sampled]: zero if
    every window is consistent with one set of weights, and otherwise a measure
    of how much the windows disagree with each other.

    Returns
    -------
    dict with the profiles (relative to the first window) prime, trapezoid,
    hermite (on the sampled moments), trapezoid_implied, hermite_implied (on the
    implied moments), the split quadrature_trapezoid, quadrature_hermite,
    consistency, the per-window difference dN = <N>_g - <N>_sampled and the
    predicted biases bias_trapezoid, bias_hermite.
    """
    H = np.asarray(H, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    mom = histogram_moments(H)
    g = solve_prime(H, pHs, mom.M, **solver)
    g = g - g[0]
    _, Nav_g, Var_g = reweight(log_weights(g, H, pHs, mom.M), pHs)
    gt_s, _ = compute_TI_trapezoid(pHs, mom.mean, mom.var, mom.M)
    gh_s, _ = compute_TI_hermite(pHs, mom.mean, mom.var, mom.M)
    gt_g, _ = compute_TI_trapezoid(pHs, Nav_g, Var_g, mom.M)
    gh_g, _ = compute_TI_hermite(pHs, Nav_g, Var_g, mom.M)
    return dict(prime=g, trapezoid=gt_s, hermite=gh_s, trapezoid_implied=gt_g, hermite_implied=gh_g,
                quadrature_trapezoid=g - gt_g, quadrature_hermite=g - gh_g,
                consistency=gt_g - gt_s, dN=Nav_g - mom.mean,
                bias_trapezoid=ti_bias_trapezoid(pHs, mom.var),
                bias_hermite=ti_bias_hermite(pHs, mom.kappa4))


# ======================================================================
# 10. Coarsening of the pH grid
# ======================================================================
def coarsening_rms(H, pHs, max_step=14, min_pts=3, weights=None, **solver):
    """RMS deviation of each estimator from the finest-grid PRIME profile, as a
    function of the pH-grid spacing (Section IV D of the paper).

    For every stride s the retained windows are pHs[off::s]; the s inequivalent
    alignments are all evaluated and the RMS over the retained windows is
    averaged over them, so that the value characterizes the spacing and not
    one alignment. Several systems may be combined linearly, for instance
    ``H=[H_native, H_fibril]`` with ``weights=(-1, 1/2)`` for the free energy
    of the transition per molecule; both systems are then coarsened onto the
    same windows, preserving the cancellation of errors between them.

    Returns
    -------
    spacings : ndarray (max_step,), s times the grid spacing
    rms : ndarray (3, max_step): PRIME, trapezoidal TI, Hermite TI; NaN where
        fewer than ``min_pts`` windows remain
    """
    systems = [np.asarray(h, dtype=float) for h in (H if isinstance(H, (list, tuple)) else [H])]
    weights = np.ones(len(systems)) if weights is None else np.asarray(weights, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    n = len(pHs)

    def profiles(idx):
        total = np.zeros((3, len(idx)))
        for w, Hs in zip(weights, systems):
            total += w * estimator_profiles(Hs[idx], pHs[idx], **solver)
        return total

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PRIMEWarning)
        ref = profiles(np.arange(n))[0]
        rms = np.full((3, max_step), np.nan)
        for s in range(1, max_step + 1):
            acc = []
            for off in range(s):
                idx = np.arange(off, n, s)
                if len(idx) < min_pts:
                    continue
                r = ref[idx] - ref[idx][0]
                acc.append(np.sqrt(np.mean((profiles(idx) - r[None, :]) ** 2, axis=1)))
            if acc:
                rms[:, s - 1] = np.mean(acc, axis=0)
    return np.arange(1, max_step + 1) * (pHs[1] - pHs[0]), rms


# ======================================================================
# 11. Prediction at pH values that were not simulated
# ======================================================================
def _fit_weights(H, pHs, max_iter, diagnostics):
    """ln g(N) from PRIME on the retained windows; None when they are not connected."""
    M = H.sum(axis=1)
    if len(pHs) == 1:
        g = np.zeros(1)
    else:
        if not support_connected(H):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PRIMEWarning)
            g = solve_prime(H, pHs, M, max_iter=max_iter)
        if diagnostics and connectivity_diagnostics(g, H, pHs, M)["n_components"] != 1:
            return None
    return log_weights(g, H, pHs, M)


def predict(H, pHs, pH_eval, n_sites=None, h_blk=None, tolerance=1e-3, n_boot=100,
            min_boot=0.9, seed=0, max_iter=1000):
    """Predict <N>, Var(N) and the free energy at any pH from a subset of
    windows, and say whether the prediction is determined (Section IV E).

    PRIME on the retained windows gives the weights over the sampled
    protonation numbers; reweighting it to a pH that was not simulated is a
    prediction. The prediction at a pH is reported as determined when three
    conditions hold, all read from the retained histograms alone:

    1. the sampled protonation numbers form one contiguous range with no hole;
    2. the weight that Pr(N | pH) places on an edge of that range which is not
       a physical bound of N (0 or n_sites) is below the prediction tolerance
       (a missing tail of weight f shifts <N> by about f protons);
    3. the connection between the windows survives the block bootstrap: when
       fewer than ``min_boot`` of the resamples stay connected the set of
       windows is 'fragile' and nothing is determined.

    The standard error ``sigma`` of the predicted <N> is PRIME's own estimate
    of the reconstruction error, from the resampled retained windows only.

    Parameters
    ----------
    H : ndarray (K, Nmax + 1), histograms of the retained windows
    pHs : array_like (K,), their pH
    pH_eval : array_like, pH values at which to predict
    n_sites : int, number of titratable sites (default Nmax of H)
    h_blk : ndarray (K, n_blocks, Nmax + 1), optional block histograms of the
        same windows, for condition 3 and sigma; without them neither is
        evaluated (sigma is inf and fragile is False)
    tolerance : the prediction tolerance epsilon
    n_boot, min_boot, seed : block-bootstrap settings

    Returns
    -------
    dict with keys mean, var, free_energy (reduced, relative to the first
    retained window), sigma, determined (per pH of pH_eval), connected,
    fragile, n_boot_ok, and lnc, the weights of the nominal fit
    """
    H = np.asarray(H, dtype=float)
    pHs = np.asarray(pHs, dtype=float)
    pH_eval = np.atleast_1d(np.asarray(pH_eval, dtype=float))
    n_eval = len(pH_eval)
    N_phys = H.shape[1] - 1 if n_sites is None else int(n_sites)
    out = dict(mean=np.full(n_eval, np.nan), var=np.full(n_eval, np.nan),
               free_energy=np.full(n_eval, np.nan),
               sigma=np.full(n_eval, np.inf), determined=np.zeros(n_eval, dtype=bool),
               connected=False, fragile=False, n_boot_ok=0, lnc=None)

    lnc = _fit_weights(H, pHs, max_iter, diagnostics=True)
    if lnc is None:
        return out
    dY, Nav, Var = reweight(lnc, pH_eval, pH_ref=pHs[0])
    w_lo, w_hi, N_lo, N_hi = edge_weights(lnc, pH_eval)
    idx = np.flatnonzero(np.isfinite(lnc))
    contiguous = bool(np.isfinite(lnc[idx.min():idx.max() + 1]).all())
    open_edge = ((w_lo > tolerance) & (N_lo > 0)) | ((w_hi > tolerance) & (N_hi < N_phys))
    det = contiguous & ~open_edge
    out.update(mean=Nav, var=Var, free_energy=dY, determined=det, connected=True, lnc=lnc)

    if h_blk is None or n_boot == 0 or not det.any():
        return out
    rng = np.random.default_rng(seed)
    boots = []
    for Hb in resample_histograms(h_blk, rng, n=n_boot):
        try:
            lnc_b = _fit_weights(Hb.astype(float), pHs, max_iter, diagnostics=False)
        except (np.linalg.LinAlgError, ValueError):
            lnc_b = None
        if lnc_b is not None:
            boots.append(reweight(lnc_b, pH_eval)[1])
    out["n_boot_ok"] = len(boots)
    if len(boots) < min_boot * n_boot:
        out.update(fragile=True, determined=np.zeros(n_eval, dtype=bool))
        return out
    sigma = np.std(boots, axis=0, ddof=1)
    sigma[~det] = np.inf
    out["sigma"] = sigma
    return out


def variance_maxima(H, n=2, prominence=0.05):
    """Indices of the ``n`` windows at the highest maxima of Var(N) over pH.

    The variance of N is largest where the protein takes up or releases
    protons most readily, which is where a window samples the most
    protonation numbers. These windows are the natural choice when few are to
    be retained (the a priori pair of Section IV E).
    """
    from scipy.signal import find_peaks

    var = histogram_moments(H).var
    peaks, _ = find_peaks(var, prominence=prominence * var.max())
    if len(peaks) < n:
        peaks = np.argsort(var)[-n:]
    return np.sort(peaks[np.argsort(var[peaks])[-n:]])


# ======================================================================
# 12. Object interface
# ======================================================================
class PRIME:
    """One fit, evaluated anywhere.

    >>> fit = PRIME(H, pHs, n_sites=11)
    >>> fit.free_energy(7.0, pH_ref=2.0)      # beta * [Upsilon(7) - Upsilon(2)]
    >>> fit.mean(7.0), fit.variance(7.0)       # <N> and Var(N) at pH 7
    >>> fit.distribution(7.0)                  # (N, Pr(N | pH 7))
    >>> fit.errors()                           # asymptotic sigma of g
    >>> fit.predict(np.linspace(2, 12, 101), h_blk=h_blk)
    >>> fit.consistency(h_blk=h_blk)["tau"]

    Attributes: H, pHs, M, n_sites, g (relative potentials, g[0] = 0) and lnc
    (the weights ln c(N), NaN where N was never sampled).
    """

    def __init__(self, H, pHs, n_sites=None, M=None, **solver):
        self.H = np.asarray(H, dtype=float)
        self.pHs = np.asarray(pHs, dtype=float)
        self.M = self.H.sum(axis=1) if M is None else np.asarray(M, dtype=float)
        self.n_sites = self.H.shape[1] - 1 if n_sites is None else int(n_sites)
        self.g = solve_prime(self.H, self.pHs, self.M, **solver)
        self.lnc = log_weights(self.g, self.H, self.pHs, self.M)
        self._solver = solver

    def free_energy(self, pH, pH_ref=None):
        """Reduced semi-grand potential beta*Upsilon at pH, relative to pH_ref
        (the first window by default)."""
        pH_ref = self.pHs[0] if pH_ref is None else pH_ref
        return reweight(self.lnc, pH, pH_ref)[0]

    def mean(self, pH):
        return reweight(self.lnc, pH)[1]

    def variance(self, pH):
        return reweight(self.lnc, pH)[2]

    def distribution(self, pH):
        return distribution(self.lnc, pH)

    def stepwise_pka(self):
        return stepwise_pka_from_weights(self.lnc)

    def errors(self):
        return compute_prime_errors(self.g, self.H, self.pHs, self.M)

    def overlap(self):
        return overlap_matrix(self.g, self.H, self.pHs, self.M)

    def connectivity(self, min_shared=1.0):
        return connectivity_diagnostics(self.g, self.H, self.pHs, self.M, min_shared)

    def predict(self, pH_eval, h_blk=None, **kwargs):
        return predict(self.H, self.pHs, pH_eval, self.n_sites, h_blk, **kwargs)

    def consistency(self, h_blk=None, **kwargs):
        return consistency_test(self.H, self.pHs, h_blk, **kwargs)

    def residuals(self):
        return residual_decomposition(self.H, self.pHs, **self._solver)
