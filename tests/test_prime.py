"""Tests of prime against an exactly solvable model and against its own invariants.

The model is a set of independent titratable sites with known pKa values, for which
the weights of the protonation numbers, the free-energy profile and the stepwise pKa are known in closed
form. Records are drawn either independently (fast, exact reference) or from two-state
Markov chains (autocorrelated, for the block machinery).

Run with ``pytest`` from the repository root (about a minute).
"""
import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

import prime

LN10 = np.log(10.0)


# =====================================================================================
# fixtures
# =====================================================================================
@pytest.fixture(scope="session")
def model():
    """Independent sites, records drawn independently at 48 windows: the exact reference."""
    rng = np.random.default_rng(0)
    pK = np.array([6.8, 6.6, 6.4, 6.2, 4.8, 4.6, 4.4, 4.2])
    Nmax = len(pK)
    Nax = np.arange(Nmax + 1, dtype=float)
    lnC = gammaln(Nmax + 1) - gammaln(Nax + 1) - gammaln(Nmax - Nax + 1)
    lnc_true = lnC + LN10 * np.concatenate([[0.0], np.cumsum(pK)])
    pHs = np.round(np.arange(0.8, 10.21, 0.2), 1)
    M = 50000
    records = []
    for p in pHs:
        lp = lnc_true - Nax * LN10 * p
        records.append(rng.choice(Nmax + 1, size=M, p=np.exp(lp - logsumexp(lp))))
    truth = np.array([-logsumexp(lnc_true - Nax * LN10 * p) for p in pHs])
    return dict(pK=pK, Nmax=Nmax, Nax=Nax, lnc_true=lnc_true, pHs=pHs, M=M, records=records,
                truth=truth - truth[0], pK_step=np.diff(lnc_true) / LN10)


@pytest.fixture(scope="session")
def fitted(model):
    H = prime.build_histograms(model["records"], model["pHs"], n_sites=model["Nmax"])
    h_blk = prime.block_histograms(model["records"], model["pHs"], n_blocks=32, n_sites=model["Nmax"])
    g = prime.solve_prime(H, model["pHs"])
    return dict(H=H, h_blk=h_blk, g=g, lnc=prime.log_weights(g, H, model["pHs"]), mom=prime.histogram_moments(H))


def markov_records(pHs, pKa, n_frames, tau, seed):
    """N(t) from independent two-state Markov chains with relaxation time tau (frames)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + 10.0 ** (pHs[:, None] - pKa[None, :]))
    a, b = p / tau, (1.0 - p) / tau
    state = rng.random(p.shape) < p
    N = np.empty((len(pHs), n_frames), dtype=np.int64)
    for t in range(n_frames):
        u = rng.random(p.shape)
        state = np.where(state, u >= b, u < a)
        N[:, t] = state.sum(axis=1)
    return list(N)


@pytest.fixture(scope="session")
def chain():
    """Autocorrelated records of 11 sites at 31 windows, relaxation time 10 frames."""
    pKa = np.array([4.0, 4.4, 4.6, 4.9, 5.9, 6.5, 9.9, 10.4, 11.2, 13.6, 14.5])
    pHs = np.round(np.arange(1.8, 14.01, 0.4), 1)
    records = markov_records(pHs, pKa, 20000, 10.0, seed=3)
    return dict(pKa=pKa, pHs=pHs, records=records, tau=10.0,
                H=prime.build_histograms(records, pHs, n_sites=len(pKa)),
                h_blk=prime.block_histograms(records, pHs, n_blocks=32, n_sites=len(pKa)))


# =====================================================================================
# 1. histograms, moments, blocks
# =====================================================================================
def test_histograms_accept_the_three_input_forms(model):
    pHs, rec = model["pHs"], model["records"]
    H1 = prime.build_histograms(rec, pHs, n_sites=model["Nmax"])
    H2 = prime.build_histograms({p: {"N": r} for p, r in zip(pHs, rec)}, pHs, n_sites=model["Nmax"])
    H3 = prime.build_histograms({p: r for p, r in zip(pHs, rec)}, n_sites=model["Nmax"])
    assert np.array_equal(H1, H2) and np.array_equal(H1, H3)
    assert H1.shape == (len(pHs), model["Nmax"] + 1) and H1.sum() == len(pHs) * model["M"]
    with pytest.raises(ValueError):
        prime.build_histograms(rec, pHs, n_sites=3)                 # sampled N above the stated bound


def test_moments_match_the_records(model, fitted):
    mom = fitted["mom"]
    rec = model["records"]
    assert np.allclose(mom.mean, [r.mean() for r in rec])
    assert np.allclose(mom.var, [r.var() for r in rec])
    assert np.allclose(mom.kappa4, [prime.fourth_cumulant(r) for r in rec])
    assert np.array_equal(mom.M, [len(r) for r in rec])


def test_blocks_sum_to_the_histograms(model, fitted):
    h_blk, H = fitted["h_blk"], fitted["H"]
    assert h_blk.shape == (len(model["pHs"]), 32, model["Nmax"] + 1)
    assert np.array_equal(h_blk.sum(axis=1), H)
    halves = prime.partial_histograms(h_blk, 2)
    assert np.array_equal(halves[0] + halves[1], H)
    assert np.allclose(prime.block_means(h_blk).mean(axis=1), fitted["mom"].mean, atol=1e-3)   # blocks differ by a frame
    hb = prime.block_histograms(model["records"], model["pHs"], block_length=1000, n_sites=model["Nmax"])
    assert hb.shape[1] == model["M"] // 1000
    with pytest.raises(ValueError):
        prime.block_histograms(model["records"], model["pHs"])     # neither n_blocks nor block_length


# =====================================================================================
# 2. correlation of the records
# =====================================================================================
def test_statistical_inefficiency_of_a_markov_chain(chain):
    g = np.array([prime.statistical_inefficiency(r) for r in chain["records"]])
    expected = 2 * chain["tau"] - 1                                   # for an exponential autocorrelation
    assert 0.6 * expected < np.median(g) < 1.5 * expected
    assert prime.statistical_inefficiency(np.ones(1000)) == 1.0
    L, g2 = prime.block_length(chain["records"], chain["pHs"])
    assert np.allclose(g, g2) and L == int(3 * np.ceil(g.max()))


def test_blocking_scan_reaches_the_correlated_standard_error(chain):
    x = chain["records"][10]
    lengths, se = prime.blocking_scan(x, lengths=[1, 4, 16, 64, 256, 1024])
    assert np.isfinite(se).all() and se[0] < se[-1]
    g = 2 * chain["tau"] - 1                                          # inefficiency of an exponential autocorrelation
    assert 0.8 < se[0] / np.sqrt(x.var() / len(x)) < 1.2             # single frames: the naive error
    assert 0.6 < se[4] / np.sqrt(g * x.var() / len(x)) < 1.5         # long blocks: the true error, sqrt(g) larger
    _, se_bm = prime.blocking_scan(prime.block_means(chain["h_blk"])[10], lengths=[1, 2, 4])
    assert se_bm.shape == (3,) and np.isfinite(se_bm).all()
    _, se_short = prime.blocking_scan(x, lengths=[len(x)])           # fewer than min_blocks blocks
    assert np.isnan(se_short).all()


# =====================================================================================
# 3. estimator, errors, overlap
# =====================================================================================
def test_prime_reproduces_the_analytic_free_energy(model, fitted):
    g, truth = fitted["g"], model["truth"]
    assert g[0] == 0.0
    assert np.sqrt(np.mean((g - truth) ** 2)) < 0.01
    sigma, Sigma = prime.compute_prime_errors(g, fitted["H"], model["pHs"])
    assert sigma[0] == 0.0 and Sigma.shape == (len(g) - 1, len(g) - 1)
    assert np.abs(g - truth)[1:].max() < 5 * sigma[1:].max()
    g_fp = prime.solve_prime(fitted["H"], model["pHs"], method="fixedpoint", max_iter=20000)
    assert np.abs(g_fp - g).max() < 1e-6


def test_overlap_matrix_rows_sum_to_the_frame_numbers(model, fitted):
    d = prime.connectivity_diagnostics(fitted["g"], fitted["H"], model["pHs"])
    assert d["n_components"] == 1 and d["algebraic_connectivity"] > 0
    assert np.allclose(d["theta"].sum(axis=1), fitted["mom"].M)
    assert prime.support_connected(fitted["H"])
    assert not prime.support_connected(fitted["H"][[0, -1]])        # pH 0.8 and 10.2 share no N


def test_disconnected_windows_raise_a_warning(model, fitted):
    H2 = fitted["H"][[0, -1]]
    with pytest.warns(prime.PRIMEWarning):
        prime.solve_prime(H2, model["pHs"][[0, -1]])


# =====================================================================================
# 4. weights of the protonation numbers
# =====================================================================================
def test_sum_of_states_reproduces_the_fit_and_the_model(model, fitted):
    lnc, g, pHs = fitted["lnc"], fitted["g"], model["pHs"]
    assert np.isfinite(lnc).all()                                     # every N sampled somewhere
    dU, Nav, Var = prime.reweight(lnc, pHs, pH_ref=pHs[0])
    assert np.abs(dU - g).max() < 1e-10
    assert np.abs(Nav - fitted["mom"].mean).max() < 0.02
    assert np.abs(prime.stepwise_pka_from_weights(lnc) - model["pK_step"]).max() < 0.05
    N, Pr = prime.distribution(lnc, 5.5)
    assert np.isclose(Pr.sum(), 1.0) and len(N) == model["Nmax"] + 1
    w_lo, w_hi, N_lo, N_hi = prime.edge_weights(lnc, [0.0, 12.0])
    assert (N_lo, N_hi) == (0, model["Nmax"]) and w_hi[0] > 0.99 and w_lo[1] > 0.99


# =====================================================================================
# 5. thermodynamic integration
# =====================================================================================
def test_ti_estimators_and_their_biases(model, fitted):
    pHs, mom, truth = model["pHs"], fitted["mom"], model["truth"]
    gt, et = prime.compute_TI_trapezoid(pHs, mom.mean, mom.var, mom.M)
    gh, eh = prime.compute_TI_hermite(pHs, mom.mean, mom.var, mom.M)
    assert np.sqrt(np.mean((gt - truth) ** 2)) < 0.05 and np.sqrt(np.mean((gh - truth) ** 2)) < 0.01
    # Hermite = trapezoid + leading Euler-Maclaurin term on a uniform grid
    assert np.abs(gh - (gt + prime.ti_bias_trapezoid(pHs, mom.var))).max() < 1e-9
    assert (et >= 0).all() and et[0] == 0.0
    _, et2 = prime.compute_TI_trapezoid(pHs, mom.mean, mom.var, mom.M, include_discretization=True)
    assert (et2 >= et).all()
    with pytest.raises(ValueError):
        prime.compute_TI_hermite(pHs, mom.mean, mom.var, mom.M, include_discretization=True)
    with pytest.raises(ValueError):
        prime.compute_TI_trapezoid(pHs[[0, 1, 3]], mom.mean[[0, 1, 3]], mom.var[[0, 1, 3]], mom.M[[0, 1, 3]])
    p3 = prime.estimator_profiles(fitted["H"], pHs)
    assert p3.shape == (3, len(pHs)) and np.allclose(p3[1], gt) and np.allclose(p3[2], gh)


# =====================================================================================
# 6. block bootstrap
# =====================================================================================
def test_bootstrap_is_calibrated_for_independent_records(model, fitted):
    """For independent records the block bootstrap of a window mean must reproduce sqrt(Var/M)."""
    nominal, boots, lo, hi = prime.bootstrap(fitted["h_blk"], model["pHs"],
                                             lambda H, p: prime.histogram_moments(H).mean, n_boot=200, seed=1)
    assert np.allclose(nominal, fitted["mom"].mean)
    ratio = boots.std(axis=0, ddof=1) / np.sqrt(fitted["mom"].var / fitted["mom"].M)
    assert 0.85 < np.median(ratio) < 1.15
    assert (lo <= nominal).all() and (nominal <= hi).all()


def test_bootstrap_is_independent_of_the_number_of_processes(model, fitted):
    a = prime.bootstrap(fitted["h_blk"][::6], model["pHs"][::6], prime.estimator_profiles, n_boot=24, seed=2, nprocs=1)
    b = prime.bootstrap(fitted["h_blk"][::6], model["pHs"][::6], prime.estimator_profiles, n_boot=24, seed=2, nprocs=2)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert a[1].shape[0] == 24 and a[1].shape[1:] == (3, len(model["pHs"][::6]))


def test_bootstrap_of_two_systems(model, fitted):
    nominal, boots, lo, hi = prime.bootstrap([fitted["h_blk"], fitted["h_blk"]], model["pHs"],
                                             prime.transition_profiles, n_boot=10, seed=0, nmol=(1, 2))
    assert nominal.shape == (3, len(model["pHs"]))
    assert np.allclose(nominal, -0.5 * prime.estimator_profiles(fitted["H"], model["pHs"]))


def test_resample_histograms_aligned_and_independent(fitted):
    rng = np.random.default_rng(0)
    Hs = prime.resample_histograms(fitted["h_blk"], rng, n=3)
    assert Hs.shape == (3,) + fitted["H"].shape
    assert np.abs(Hs.sum(axis=2) - fitted["mom"].M).max() <= 32      # blocks differ in length by one frame
    Ha = prime.resample_histograms(fitted["h_blk"], rng, n=2, aligned=True)
    assert Ha.shape == (2,) + fitted["H"].shape


# =====================================================================================
# 7. consistency test
# =====================================================================================
def test_consistency_test_finds_consistent_windows_consistent(model, fitted):
    cons = prime.consistency_test(fitted["H"], model["pHs"], fitted["h_blk"], n_boot=200, seed=0)
    ok = cons["n_windows"] >= 2
    assert ok.sum() >= model["Nmax"] - 1
    assert np.nanmax(np.abs(cons["pK_mean"] - model["pK_step"])[ok]) < 0.03
    assert np.nanmax(np.abs(cons["pK_prime"] - model["pK_step"])[ok]) < 0.03
    assert np.nanmax(cons["tau"]) < 0.03 and np.nanmean(cons["p_value"][ok]) > 0.1
    table = prime.format_consistency(cons)
    assert table.count("\n") == ok.sum()
    # the weighted mean and tau follow the DerSimonian-Laird formulas
    j = int(np.flatnonzero(ok)[0])
    m = cons["usable"][:, j]
    r, w = cons["pK"][m, j], 1.0 / cons["se"][m, j] ** 2
    mu = (w * r).sum() / w.sum()
    Q = (w * (r - mu) ** 2).sum()
    C = w.sum() - (w ** 2).sum() / w.sum()
    assert np.isclose(mu, cons["pK_mean"][j]) and np.isclose(Q / (m.sum() - 1), cons["chi2_red"][j])
    assert np.isclose(np.sqrt(max(0.0, (Q - (m.sum() - 1)) / C)), cons["tau"][j])


def test_consistency_test_flags_a_shifted_window(model, fitted):
    """A window whose records come from a different molecule is caught."""
    H = fitted["H"].copy()
    k = 20                                                            # pH 4.8, on the acidic step
    lp = model["lnc_true"] - model["Nax"] * LN10 * (model["pHs"][k] + 0.3)   # as if simulated 0.3 units away
    H[k] = np.random.default_rng(5).multinomial(model["M"], np.exp(lp - logsumexp(lp)))
    cons = prime.consistency_test(H, model["pHs"], None, min_count=200)     # Poisson errors suffice here
    assert cons["flagged"].any() and cons["tau"][np.nan_to_num(cons["tau"]).argmax()] > 0.05


def test_stepwise_pka_without_blocks_uses_poisson_errors(model, fitted):
    pK, se, usable = prime.stepwise_pka(fitted["H"], model["pHs"], None)
    H = fitted["H"].astype(float)
    i, j = np.argwhere(usable)[0]
    assert np.isclose(se[i, j], np.sqrt(1 / H[i, j + 1] + 1 / H[i, j]) / LN10)
    assert np.isnan(pK[~usable]).all()


# =====================================================================================
# 8. residual decomposition and coarsening
# =====================================================================================
def test_residual_decomposition_splits_exactly(model, fitted):
    r = prime.residual_decomposition(fitted["H"][::3], model["pHs"][::3])      # coarser grid: visible biases
    total = r["prime"] - r["trapezoid"]
    assert np.allclose(total, r["quadrature_trapezoid"] + r["consistency"])
    assert np.allclose(r["prime"] - r["hermite"], r["quadrature_hermite"] + (r["hermite_implied"] - r["hermite"]))
    # on consistent windows the residual is quadrature, and the quadrature part is the Euler-Maclaurin bias
    assert np.abs(r["consistency"]).max() < 0.3 * np.abs(r["quadrature_trapezoid"]).max()
    assert np.abs(r["quadrature_trapezoid"] - r["bias_trapezoid"]).max() < 0.25 * np.abs(r["bias_trapezoid"]).max()


def test_coarsening_rms(model, fitted):
    spacing, rms = prime.coarsening_rms(fitted["H"], model["pHs"], max_step=6)
    assert rms.shape == (3, 6) and np.allclose(spacing, 0.2 * np.arange(1, 7))
    assert np.allclose(rms[:, 0], 0.0, atol=1e-9) or rms[0, 0] < 1e-9        # the finest grid is the reference
    assert rms[0, -1] < rms[1, -1]                                            # PRIME beats the trapezoid when coarse
    assert np.all(np.diff(rms[1]) >= -1e-9)                                   # the trapezoid error grows with spacing
    _, rms2 = prime.coarsening_rms([fitted["H"], fitted["H"]], model["pHs"], weights=(-1.0, 1.0), max_step=3)
    assert np.allclose(rms2, 0.0, atol=1e-9)                                  # identical systems cancel


# =====================================================================================
# 9. prediction
# =====================================================================================
def test_prediction_from_two_windows(model, fitted):
    pHs, Nmax = model["pHs"], model["Nmax"]
    keep = [int(np.argmin(np.abs(pHs - 4.4))), int(np.argmin(np.abs(pHs - 6.6)))]
    r = prime.predict(fitted["H"][keep], pHs[keep], pHs, n_sites=Nmax, h_blk=fitted["h_blk"][keep], seed=2)
    assert r["connected"] and not r["fragile"] and r["n_boot_ok"] == 100
    det = r["determined"]
    assert det[np.abs(pHs - 5.4) < 0.05].all()                        # the middle of the ladder is covered
    assert np.abs(r["mean"] - fitted["mom"].mean)[det].max() < 0.1
    assert np.isfinite(r["sigma"][det]).all() and np.isinf(r["sigma"][~det]).all()
    assert np.isclose(r["free_energy"][keep[0]], 0.0)                 # relative to the first retained window
    one = prime.predict(fitted["H"][[keep[0]]], pHs[[keep[0]]], pHs, n_sites=Nmax)
    assert one["connected"] and one["determined"].sum() < det.sum()   # one window covers less than two
    assert np.isinf(one["sigma"]).all()                               # no blocks, no sigma


def test_prediction_of_a_disconnected_pair(model, fitted):
    pHs = model["pHs"]
    r = prime.predict(fitted["H"][[0, -1]], pHs[[0, -1]], pHs, n_sites=model["Nmax"], h_blk=fitted["h_blk"][[0, -1]])
    assert not r["connected"] and not r["determined"].any() and np.isnan(r["mean"]).all()


def test_prediction_through_a_single_record_is_fragile():
    """Two windows linked by one record in one block: the link is lost in about (31/32)^32 = 36% of
    the resamples, far more than the 10% allowed, so nothing is determined."""
    B, nb = 32, 9
    h = np.zeros((2, B, nb), dtype=np.int64)
    h[0, :, 0:4] = 300                                                # window A samples N = 0..3
    h[0, 5, 4] = 1                                                    # ... and N = 4 once, in block 5
    h[1, :, 4:9] = 300                                                # window B samples N = 4..8
    r = prime.predict(h.sum(axis=1), [3.0, 7.0], np.linspace(2, 8, 13), n_sites=8, h_blk=h, seed=0)
    assert r["connected"] and r["fragile"] and not r["determined"].any()
    assert 40 < r["n_boot_ok"] < 85
    h[0, :, 4] = 1                                                    # the link in every block: robust
    r2 = prime.predict(h.sum(axis=1), [3.0, 7.0], np.linspace(2, 8, 13), n_sites=8, h_blk=h, seed=0)
    assert r2["connected"] and not r2["fragile"] and r2["n_boot_ok"] == 100


def test_variance_maxima(model, fitted):
    idx = prime.variance_maxima(fitted["H"], n=2)
    assert len(idx) == 2 and idx[0] < idx[1]
    assert abs(model["pHs"][idx[0]] - 4.5) < 0.5 and abs(model["pHs"][idx[1]] - 6.5) < 0.5   # the two steps


# =====================================================================================
# 10. object interface
# =====================================================================================
def test_object_interface_matches_the_functions(model, fitted):
    fit = prime.PRIME(fitted["H"], model["pHs"], n_sites=model["Nmax"])
    assert np.allclose(fit.g, fitted["g"]) and np.allclose(fit.lnc, fitted["lnc"], equal_nan=True)
    grid = np.linspace(1.0, 10.0, 7)
    dU, Nav, Var = prime.reweight(fitted["lnc"], grid, pH_ref=model["pHs"][0])
    assert np.allclose(fit.free_energy(grid), dU) and np.allclose(fit.mean(grid), Nav) and np.allclose(fit.variance(grid), Var)
    assert np.isclose(fit.distribution(5.0)[1].sum(), 1.0)
    assert np.allclose(fit.stepwise_pka(), prime.stepwise_pka_from_weights(fitted["lnc"]), equal_nan=True)
    assert fit.connectivity()["n_components"] == 1 and fit.errors()[0].shape == fit.g.shape
    assert np.allclose(fit.residuals()["prime"], fitted["g"] - fitted["g"][0])
    assert fit.predict(grid)["connected"] and "tau" in fit.consistency(n_boot=0)


def test_profile_bootstrap_against_the_asymptotic_errors(model, fitted):
    """The observed-information errors treat the window sizes as random and are conservative
    (see compute_prime_errors), so the block bootstrap of the profile must come out smaller,
    but not by more than the factor the fixed window sizes account for."""
    sigma, _ = prime.compute_prime_errors(fitted["g"], fitted["H"], model["pHs"])
    _, boots, _, _ = prime.bootstrap(fitted["h_blk"], model["pHs"], lambda H, p: prime.estimator_profiles(H, p)[0],
                                     n_boot=100, seed=1)
    ratio = boots.std(axis=0, ddof=1)[1:] / sigma[1:]
    assert 0.4 < np.median(ratio) < 1.0
