"""
Unit tests for KingPSFFitter in fitting.py.

Tests cover:
- Correct bin structure (shape, names)
- Uncorrelated auxiliary parameter → similar fit values across bins
- Correlated auxiliary parameter (log10_energy) → monotonic trend in alpha/beta
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kingmaker.fitting import KingPSFFitter
from kingmaker.pdf import KingPDF


RNG_SEED = 42

# One shared PDF instance to avoid re-building the 200×200 norm grid per test.
_KING_PDF = KingPDF()


def _make_events(n, alpha, beta, aux_field, aux_vals, rng):
    """
    Sample n King-distributed events and return a structured array.

    True positions are fixed at (ra=0, dec=0).  Reconstructed positions are
    displaced by dpsi sampled from the King distribution in a random direction
    phi, using exact spherical geometry so that
    angular_distance(reco_ra, reco_dec, 0, 0) == dpsi exactly.
    """
    dpsi = _KING_PDF.sample(n, alpha, beta, rng=rng)
    phi = rng.uniform(0, 2 * np.pi, n)

    # Exact spherical offset from (ra=0, dec=0) in direction phi.
    reco_dec = np.arcsin(np.clip(np.sin(dpsi) * np.cos(phi), -1.0, 1.0))
    reco_ra = np.arctan2(np.sin(dpsi) * np.sin(phi), np.cos(dpsi))

    dtype = [
        ("ra", np.float64),
        ("dec", np.float64),
        ("trueRa", np.float64),
        ("trueDec", np.float64),
        (aux_field, np.float64),
    ]
    events = np.zeros(n, dtype=dtype)
    events["ra"] = reco_ra
    events["dec"] = reco_dec
    events["trueRa"] = 0.0
    events["trueDec"] = 0.0
    events[aux_field] = aux_vals
    return events


# ---------------------------------------------------------------------------
# Bin structure tests
# ---------------------------------------------------------------------------


class TestKingPSFFitterStructure:
    """KingPSFFitter should report the correct bin structure after construction."""

    @pytest.fixture(scope="class")
    def fitter(self):
        rng = np.random.default_rng(RNG_SEED)
        aux_vals = rng.uniform(0, 1, 3000)
        events = _make_events(3000, np.radians(1.0), 2.5, "aux", aux_vals, rng)
        return KingPSFFitter(
            events,
            parametrization_bins={"aux": 3},
            dpsi_nbins=30,
            minimum_counts=100,
            weight_field=None,
        )

    def test_parametrization_shape(self, fitter):
        assert fitter.parametrization_shape == [3]

    def test_bin_names(self, fitter):
        assert fitter.bin_names == ["aux"]

    def test_fit_alpha_shape_before_fitting(self, fitter):
        # Shape should be (n_spectral_indices=1, n_bins=3).
        assert fitter.fit_alpha.shape == (1, 3)

    def test_fit_beta_shape_before_fitting(self, fitter):
        assert fitter.fit_beta.shape == (1, 3)

    def test_explicit_bin_edges_count(self):
        """Passing k explicit edges should produce k-1 bins."""
        rng = np.random.default_rng(RNG_SEED)
        aux_vals = rng.uniform(0, 4, 3000)
        events = _make_events(3000, np.radians(1.0), 2.5, "aux", aux_vals, rng)
        fitter = KingPSFFitter(
            events,
            parametrization_bins={"aux": [0.0, 1.0, 2.0, 3.0, 4.0]},
            dpsi_nbins=30,
            minimum_counts=100,
            weight_field=None,
        )
        assert fitter.parametrization_shape == [4]


# ---------------------------------------------------------------------------
# Uncorrelated auxiliary parameter
# ---------------------------------------------------------------------------


class TestKingPSFFitterUncorrelated:
    """
    When the auxiliary parameter is independent of the angular error, every bin
    should yield similar fitted alpha and beta values.
    """

    @pytest.fixture(scope="class")
    def result(self):
        rng = np.random.default_rng(RNG_SEED)
        alpha_true = np.radians(1.0)
        beta_true = 2.5
        n = 3000

        aux_vals = rng.uniform(0, 1, n)
        events = _make_events(n, alpha_true, beta_true, "aux", aux_vals, rng)

        fitter = KingPSFFitter(
            events,
            parametrization_bins={"aux": 3},
            dpsi_nbins=30,
            minimum_counts=100,
            weight_field=None,
        )
        fitter.fit_all_bins(verbose=False)
        return fitter, alpha_true, beta_true

    def test_all_bins_have_enough_events(self, result):
        fitter, _, _ = result
        assert np.all(fitter.event_counts[0] >= 100)

    def test_alpha_consistent_across_bins(self, result):
        """Coefficient of variation of fitted alpha should be small (<30%)."""
        fitter, _, _ = result
        alphas = fitter.fit_alpha[0]
        assert alphas.std() / alphas.mean() < 0.30

    def test_beta_consistent_across_bins(self, result):
        """Coefficient of variation of fitted beta should be small (<30%)."""
        fitter, _, _ = result
        betas = fitter.fit_beta[0]
        assert betas.std() / betas.mean() < 0.30

    def test_mean_alpha_roughly_accurate(self, result):
        """Mean fitted alpha should be within 30% of the true value."""
        fitter, alpha_true, _ = result
        assert_allclose(fitter.fit_alpha[0].mean(), alpha_true, rtol=0.30)

    def test_mean_beta_roughly_accurate(self, result):
        """Mean fitted beta should be within 40% of the true value."""
        fitter, _, beta_true = result
        assert_allclose(fitter.fit_beta[0].mean(), beta_true, rtol=0.40)


# ---------------------------------------------------------------------------
# Correlated auxiliary parameter (log10_energy)
# ---------------------------------------------------------------------------


class TestKingPSFFitterCorrelated:
    """
    Higher log10_energy events are drawn with smaller alpha and larger beta,
    mimicking an energy-dependent PSF.  The fitter should recover the trend.
    """

    # (alpha_true, beta_true, (log10_E_lo, log10_E_hi))
    _GROUP_PARAMS = [
        (np.radians(2.0), 2.0, (3.0, 4.0)),  # low energy:  broad PSF
        (np.radians(1.0), 3.0, (4.0, 5.0)),  # mid energy
        (np.radians(0.5), 4.0, (5.0, 6.0)),  # high energy: narrow PSF
    ]

    @pytest.fixture(scope="class")
    def result(self):
        rng = np.random.default_rng(RNG_SEED)
        n_per_group = 2000

        groups = []
        for alpha, beta, (e_lo, e_hi) in self._GROUP_PARAMS:
            log10_e = rng.uniform(e_lo, e_hi, n_per_group)
            groups.append(_make_events(n_per_group, alpha, beta, "log10_energy", log10_e, rng))
        events = np.concatenate(groups)

        fitter = KingPSFFitter(
            events,
            parametrization_bins={"log10_energy": [3.0, 4.0, 5.0, 6.0]},
            dpsi_nbins=30,
            minimum_counts=100,
            weight_field=None,
        )
        fitter.fit_all_bins(verbose=False)
        return fitter

    def test_parametrization_shape(self, result):
        assert result.parametrization_shape == [3]

    def test_alpha_decreases_with_energy(self, result):
        """Higher-energy bins should produce a smaller fitted alpha."""
        alphas = result.fit_alpha[0]
        assert alphas[0] > alphas[1] > alphas[2]

    def test_beta_increases_with_energy(self, result):
        """Higher-energy bins should produce a larger fitted beta."""
        betas = result.fit_beta[0]
        assert betas[0] < betas[1] < betas[2]

    def test_alpha_values_per_bin(self, result):
        """Fitted alpha per bin should be within 40% of the true value."""
        alphas = result.fit_alpha[0]
        for i, (alpha_true, _, _) in enumerate(self._GROUP_PARAMS):
            assert_allclose(alphas[i], alpha_true, rtol=0.40, err_msg=f"bin {i}: alpha mismatch")

    def test_beta_values_per_bin(self, result):
        """Fitted beta per bin should be within 50% of the true value."""
        betas = result.fit_beta[0]
        for i, (_, beta_true, _) in enumerate(self._GROUP_PARAMS):
            assert_allclose(betas[i], beta_true, rtol=0.50, err_msg=f"bin {i}: beta mismatch")
