"""
Characterization and regression tests for KingSpatialLikelihood in wrapper.py.

No existing fixture constructs KingSpatialLikelihood directly, so these tests
bypass KingPSFFitter entirely by writing a small .npz cache file matching the
schema __init__ reads when cache_parameters=True and the cache file already
exists. This gives full, deterministic control over the fitted alpha/beta
grid without waiting on a real fit.
"""

import numpy as np
import pytest

from kingmaker.pdf import KingPDF
from kingmaker.utils import angular_distance, _interp1d
from kingmaker.wrapper import KingSpatialLikelihood


SPECTRAL_INDICES = np.array([1.0, 2.0, 3.0])
BIN_EDGES = np.array([0.0, 1.0, 2.0, 3.0])  # bin centers: 0.5, 1.5, 2.5
ALPHA_VALUES = np.array(
    [
        [np.radians(0.5), np.radians(1.0), np.radians(1.5)],
        [np.radians(0.6), np.radians(1.1), np.radians(1.6)],
        [np.radians(0.7), np.radians(1.2), np.radians(1.7)],
    ]
)
BETA_VALUES = np.array(
    [
        [2.0, 2.5, 3.0],
        [2.1, 2.6, 3.1],
        [2.2, 2.7, 3.2],
    ]
)


def _make_likelihood(tmp_path, angular_cutoff=np.pi):
    cache_path = tmp_path / "king_cache.npz"
    np.savez(
        cache_path,
        parametrization_bins=np.array({"aux": BIN_EDGES}, dtype=object),
        alpha=ALPHA_VALUES,
        beta=BETA_VALUES,
    )
    return KingSpatialLikelihood(
        signal_events=np.empty(0),
        parametrization_bins={"aux": 3},
        spectral_indices=SPECTRAL_INDICES,
        cache_parameters=True,
        cache_name=str(cache_path),
        angular_cutoff=angular_cutoff,
    )


def _make_events(n_per_bin, rng, offset_scale=np.radians(2.0)):
    """n_per_bin events at each of the 3 known bin centers (0.5, 1.5, 2.5),
    at small random offsets from a source at (ra=0, dec=0)."""
    aux_centers = [0.5, 1.5, 2.5]
    n = n_per_bin * len(aux_centers)
    dtype = [("ra", float), ("dec", float), ("aux", float)]
    events = np.zeros(n, dtype=dtype)
    events["ra"] = rng.uniform(-offset_scale, offset_scale, n)
    events["dec"] = rng.uniform(-offset_scale, offset_scale, n)
    events["aux"] = np.repeat(aux_centers, n_per_bin)
    return events


def _bin_index(aux_value):
    """Mirror the nearest-bin-center lookup for centers [0.5, 1.5, 2.5]."""
    if aux_value < 1.0:
        return 0
    if aux_value < 2.0:
        return 1
    return 2


@pytest.fixture
def likelihood(tmp_path):
    return _make_likelihood(tmp_path)


class TestEvaluatePdfExactGamma:
    def test_matches_direct_pdf_computation(self, likelihood):
        rng = np.random.default_rng(0)
        events = _make_events(20, rng)
        src_ra, src_dec = 0.0, 0.0
        likelihood.set_events(
            events, source_ras=np.array([src_ra]), source_decs=np.array([src_dec])
        )

        bin_idx = np.array([_bin_index(a) for a in events["aux"]])
        dist = angular_distance(src_ra, src_dec, events["ra"], events["dec"])
        king_pdf = KingPDF(angular_cutoff=likelihood.king_pdf.angular_cutoff)

        for gamma_idx, gamma in enumerate(SPECTRAL_INDICES):
            result = likelihood.evaluate_pdf(events, gamma=gamma).copy()

            alpha = ALPHA_VALUES[gamma_idx][bin_idx]
            beta = BETA_VALUES[gamma_idx][bin_idx]
            expected = king_pdf.pdf(dist, alpha, beta)

            np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestEvaluatePdfInterpolatedGamma:
    def test_matches_manual_interp1d(self, likelihood):
        rng = np.random.default_rng(1)
        events = _make_events(15, rng)
        likelihood.set_events(events, source_ras=np.array([0.0]), source_decs=np.array([0.0]))

        gamma = 1.5  # between spectral_indices[0]=1.0 and [1]=2.0
        result = likelihood.evaluate_pdf(events, gamma=gamma).copy()
        low = likelihood.evaluate_pdf(events, gamma=1.0).copy()
        high = likelihood.evaluate_pdf(events, gamma=2.0).copy()

        expected = _interp1d(gamma, 1.0, 2.0, low, high)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestGetAlphaBetaGammaExact:
    def test_exact_gamma_matches_direct_lookup(self, likelihood):
        rng = np.random.default_rng(2)
        events = _make_events(10, rng)
        likelihood.set_events(events, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        all_alpha, all_beta = likelihood.get_alpha_beta(events, copy=False)

        for gamma_idx, gamma in enumerate(SPECTRAL_INDICES):
            alpha, beta = likelihood.get_alpha_beta_gamma(gamma, alpha=all_alpha, beta=all_beta)
            np.testing.assert_array_equal(alpha, all_alpha[gamma_idx])
            np.testing.assert_array_equal(beta, all_beta[gamma_idx])


class TestGetAlphaBetaGammaInterpolation:
    """Regression test locking in the get_alpha_beta_gamma interpolation fix.

    Before the fix, this raises TypeError (missing the `gamma` argument to
    _interp1d) -- this is dead code on the set_events hot path (which only
    ever calls with an exact-matching gamma) but is part of the public,
    documented API for direct callers.
    """

    def test_interpolated_gamma(self, likelihood):
        rng = np.random.default_rng(3)
        events = _make_events(10, rng)
        likelihood.set_events(events, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        all_alpha, all_beta = likelihood.get_alpha_beta(events, copy=False)

        gamma = 1.5  # between index 0 (gamma=1.0) and index 1 (gamma=2.0)
        alpha, beta = likelihood.get_alpha_beta_gamma(gamma, alpha=all_alpha, beta=all_beta)

        expected_alpha = _interp1d(gamma, 1.0, 2.0, all_alpha[0], all_alpha[1])
        expected_beta = _interp1d(gamma, 1.0, 2.0, all_beta[0], all_beta[1])
        np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-10)
        np.testing.assert_allclose(beta, expected_beta, rtol=1e-10)


class TestSetEventsNoOp:
    def test_repeated_call_is_noop(self, likelihood):
        rng = np.random.default_rng(4)
        events = _make_events(10, rng)
        likelihood.set_events(events, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        ids_before = {g: id(v) for g, v in likelihood.event_pvalue.items()}

        likelihood.set_events(events, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        ids_after = {g: id(v) for g, v in likelihood.event_pvalue.items()}

        assert ids_before == ids_after


class TestEvaluatePdfMismatch:
    def test_raises_on_mismatched_events(self, likelihood):
        rng = np.random.default_rng(5)
        events = _make_events(10, rng)
        other_events = _make_events(5, np.random.default_rng(6))  # different length
        likelihood.set_events(events, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        with pytest.raises(RuntimeError):
            likelihood.evaluate_pdf(other_events, gamma=2.0)


class TestBufferReuseAcrossTrials:
    """Same-length, different-mask trials must not leak stale values between
    calls into positions that should now read as zero (outside the cutoff)."""

    def test_no_stale_values_with_differing_mask(self, tmp_path):
        likelihood = _make_likelihood(tmp_path, angular_cutoff=np.radians(5.0))

        dtype = [("ra", float), ("dec", float), ("aux", float)]
        events_a = np.zeros(2, dtype=dtype)
        events_a["ra"] = [0.0, np.radians(20.0)]
        events_a["dec"] = [0.0, 0.0]
        events_a["aux"] = [0.5, 0.5]

        events_b = events_a.copy()
        events_b["ra"] = events_a["ra"] + np.radians(0.01)  # force a full recompute

        src_a = (0.0, 0.0)  # event 0 within cutoff, event 1 far outside
        src_b = (np.radians(20.0), 0.0)  # event 1 now within cutoff of src_b

        likelihood.set_events(
            events_a, source_ras=np.array([src_a[0]]), source_decs=np.array([src_a[1]])
        )
        result_a = likelihood.evaluate_pdf(events_a, gamma=2.0).copy()
        assert result_a[0] > 0.0
        assert result_a[1] == 0.0  # event 1 outside cutoff for source A

        likelihood.set_events(
            events_b, source_ras=np.array([src_b[0]]), source_decs=np.array([src_b[1]])
        )
        result_b = likelihood.evaluate_pdf(events_b, gamma=2.0).copy()
        assert result_b[0] == 0.0  # event 0 now outside cutoff for source B --
        # must NOT retain the nonzero value computed for source A.
        assert result_b[1] > 0.0


class TestPdfFromNorm:
    """Isolated equivalence check, independent of the wrapper integration
    tests: pdf_from_norm must exactly match pdf() for in-cutoff x."""

    def test_matches_pdf_for_in_cutoff_x(self):
        king_pdf = KingPDF(angular_cutoff=np.radians(10.0))
        rng = np.random.default_rng(9)
        x = rng.uniform(0, np.radians(10.0), 100)
        alpha = rng.uniform(np.radians(0.1), np.radians(2.0), 100)
        beta = rng.uniform(1.5, 4.0, 100)

        expected = king_pdf.pdf(x, alpha, beta)
        norm = king_pdf.norm(alpha, beta)
        result = king_pdf.pdf_from_norm(x, alpha, beta, norm)

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestBufferResize:
    def test_resizes_on_event_count_change(self, likelihood):
        rng = np.random.default_rng(8)
        events_5 = _make_events(5, rng)  # 15 events
        likelihood.set_events(events_5, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        assert len(likelihood._result_buffer) == len(events_5)

        events_10 = _make_events(10, rng)  # 30 events
        likelihood.set_events(events_10, source_ras=np.array([0.0]), source_decs=np.array([0.0]))
        assert len(likelihood._result_buffer) == len(events_10)

        result = likelihood.evaluate_pdf(events_10, gamma=2.0)
        assert len(result) == len(events_10)
