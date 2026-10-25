import numpy as np
import pytest
from multi_gxe import nulltest as NT


def test_residual_variance_model_recovers_log_linear_pattern():
    rng = np.random.default_rng(0); n = 20000
    E = rng.standard_normal((n, 3)); true = np.exp(0.5 * E[:, 0] - 0.3 * E[:, 2]); true /= true.mean()
    y = rng.standard_normal(n) * np.sqrt(true)
    v, eta = NT.residual_variance_model(y, E)
    assert abs(v.mean() - 1) < 1e-9
    assert np.corrcoef(np.log(v), np.log(true))[0, 1] > 0.98
    assert eta[1] == pytest.approx(0.5, abs=0.05) and eta[3] == pytest.approx(-0.3, abs=0.05)


def test_null_noise_sd_deconvolves_the_total_variance():
    v = np.array([0.5, 1.0, 1.5]); sd = NT.null_noise_sd(v, h2=0.3)
    assert np.allclose(sd ** 2 + 0.3, v)                      # genetic + noise = fitted total, no floor hit here
    sd2 = NT.null_noise_sd(np.array([0.1, 1.9]), h2=0.3)      # first entry floored, then rescaled to mean 0.7
    assert (sd2 ** 2).mean() == pytest.approx(0.7) and sd2[0] > 0


def _toy_genotypes(n=400, m=300, seed=1):
    rng = np.random.default_rng(seed); X = rng.binomial(2, rng.uniform(0.1, 0.5, m), (n, m)).astype(float)
    return (X - X.mean(0)) / X.std(0)


def test_null_phenotypes_variance_reproducibility_and_fresh_causal_sets():
    X = _toy_genotypes(); n, m = X.shape; v = np.ones(n)
    read = lambda a, b: X[:, a:b]
    Y = NT.null_phenotypes(read, m, n, h2=0.4, v=v, reps=range(6), trait="toy", n_causal=50, snp_block=64)
    assert Y.shape == (n, 6) and Y.dtype == np.float32
    assert np.allclose(Y.var(0), 1.0, atol=0.25)
    # a replicate does not depend on the batch it is drawn in or on the SNP block size
    Y3 = NT.null_phenotypes(read, m, n, h2=0.4, v=v, reps=[3], trait="toy", n_causal=50, snp_block=17)
    assert np.allclose(Y3[:, 0], Y[:, 3], atol=1e-5)
    # the causal set is drawn afresh per replicate: the genetic parts are nearly uncorrelated
    g = [NT.null_phenotypes(read, m, n, h2=1.0 - 1e-9, v=v, reps=[r], trait="toy", n_causal=10)[:, 0] for r in (0, 1)]
    assert abs(np.corrcoef(g[0], g[1])[0, 1]) < 0.3


def test_mc_pvalue_and_extension_rule():
    pn = np.array([0.01, 0.2, 0.5, 0.9])
    assert NT.mc_pvalue(0.001, pn) == pytest.approx(1 / 5)
    assert NT.mc_pvalue(0.2, pn) == pytest.approx(3 / 5)        # ties count against the trait
    assert np.allclose(NT.mc_pvalue([0.001, 1.0], pn), [0.2, 1.0])
    assert NT.needs_more_nulls(0.001, np.linspace(0.01, 1, 100)) and not NT.needs_more_nulls(0.5, np.linspace(0.01, 1, 100))


def test_benjamini_hochberg_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.0033])
    q = NT.benjamini_hochberg(p); m = len(p)
    ref = np.array([min(1, min(p[j] * m / r for j, r in zip(np.argsort(p), range(1, m + 1)) if p[j] >= p[i])) for i in range(m)])
    assert np.allclose(q, ref)
    assert np.allclose(q, [0.025, 0.05, 0.05, 0.2, 0.0165])
