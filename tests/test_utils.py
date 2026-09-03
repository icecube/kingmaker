"""
Unit tests for kingmaker.utils.

Covers angular_distance, meshgrid2d, and sample_with_extension.
"""

import numpy as np
from numpy.testing import assert_allclose

from kingmaker.utils import angular_distance, meshgrid2d, sample_with_extension


# ---------------------------------------------------------------------------
# angular_distance
# ---------------------------------------------------------------------------


class TestAngularDistance:
    def test_self_distance_is_zero(self):
        """Distance from a point to itself is 0."""
        ra, dec = 1.2, 0.5
        assert_allclose(angular_distance(ra, dec, ra, dec), 0.0, atol=1e-12)

    def test_antipodal_distance_is_pi(self):
        """Distance between antipodal points is π."""
        ra, dec = 0.0, np.pi / 2
        assert_allclose(angular_distance(ra, dec, ra + np.pi, -dec), np.pi, rtol=1e-12)

    def test_equatorial_90deg(self):
        """Two points separated by 90° of RA on the equator."""
        assert_allclose(
            angular_distance(0.0, 0.0, np.pi / 2, 0.0),
            np.pi / 2,
            rtol=1e-12,
        )

    def test_symmetry(self):
        """d(A, B) == d(B, A)."""
        ra1, dec1 = 0.3, 0.7
        ra2, dec2 = 1.1, -0.4
        d1 = angular_distance(ra1, dec1, ra2, dec2)
        d2 = angular_distance(ra2, dec2, ra1, dec1)
        assert_allclose(d1, d2, rtol=1e-12)

    def test_pole_to_pole(self):
        """North pole to south pole is π."""
        assert_allclose(
            angular_distance(0.0, np.pi / 2, 0.0, -np.pi / 2),
            np.pi,
            rtol=1e-12,
        )

    def test_nonnegative(self):
        """All angular distances are non-negative."""
        rng = np.random.default_rng(0)
        ras = rng.uniform(0, 2 * np.pi, 50)
        decs = rng.uniform(-np.pi / 2, np.pi / 2, 50)
        dists = angular_distance(ras[0], decs[0], ras, decs)
        assert np.all(dists >= 0)

    def test_output_in_range(self):
        """Angular distance is always in [0, π]."""
        rng = np.random.default_rng(1)
        ras = rng.uniform(0, 2 * np.pi, 100)
        decs = rng.uniform(-np.pi / 2, np.pi / 2, 100)
        dists = angular_distance(ras[0], decs[0], ras, decs)
        assert np.all(dists <= np.pi + 1e-12)


# ---------------------------------------------------------------------------
# meshgrid2d
# ---------------------------------------------------------------------------


class TestMeshgrid2d:
    def test_output_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        ga, gb = meshgrid2d(a, b)
        assert ga.shape == (len(b), len(a))
        assert gb.shape == (len(b), len(a))

    def test_a_values_constant_along_rows(self):
        """Each row of ga should contain the same a value."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        ga, _ = meshgrid2d(a, b)
        for row in ga:
            assert_allclose(row, a)

    def test_b_values_constant_along_columns(self):
        """Each column of gb should contain the same b value."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        _, gb = meshgrid2d(a, b)
        for col in gb.T:
            assert_allclose(col, b)

    def test_matches_numpy_meshgrid(self):
        """Result should equal np.meshgrid(a, b) in 'xy' indexing."""
        a = np.linspace(0, 1, 4)
        b = np.linspace(0, 2, 5)
        ga, gb = meshgrid2d(a, b)
        np_a, np_b = np.meshgrid(a, b)
        assert_allclose(ga, np_a)
        assert_allclose(gb, np_b)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        ga, gb = meshgrid2d(a, b)
        assert ga.dtype == np.float32
        assert gb.dtype == np.float32


# ---------------------------------------------------------------------------
# sample_with_extension
# ---------------------------------------------------------------------------


class TestSampleWithExtension:
    def test_zero_extension_is_noop(self):
        rng = np.random.default_rng(0)
        ra, dec = sample_with_extension(1.0, 0.5, 0.0, rng)
        assert_allclose(ra, 1.0)
        assert_allclose(dec, 0.5)

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(1)
        true_ra = np.linspace(0, 2 * np.pi, 50)
        true_dec = np.linspace(-1.0, 1.0, 50)
        ra, dec = sample_with_extension(true_ra, true_dec, np.radians(2.0), rng)
        assert ra.shape == (50,)
        assert dec.shape == (50,)

    def test_finite_and_in_range(self):
        rng = np.random.default_rng(2)
        n = 5000
        true_ra = rng.uniform(0, 2 * np.pi, n)
        true_dec = np.arcsin(rng.uniform(-1, 1, n))
        ra, dec = sample_with_extension(true_ra, true_dec, np.radians(3.0), rng)
        assert np.all(np.isfinite(ra))
        assert np.all(np.isfinite(dec))
        assert np.all(dec >= -np.pi / 2 - 1e-9)
        assert np.all(dec <= np.pi / 2 + 1e-9)
        assert np.all(ra >= 0.0)
        assert np.all(ra < 2 * np.pi)

    def test_rayleigh_scale_recovered(self):
        """Offset magnitude follows Rayleigh(extension): mean ~ scale*sqrt(pi/2)."""
        rng = np.random.default_rng(3)
        n = 200_000
        extension = np.radians(2.0)
        true_ra = np.full(n, 1.0)
        true_dec = np.full(n, 0.3)
        ra, dec = sample_with_extension(true_ra, true_dec, extension, rng)
        offset = angular_distance(true_ra, true_dec, ra, dec)
        expected_mean = extension * np.sqrt(np.pi / 2)
        assert_allclose(offset.mean(), expected_mean, rtol=0.02)

    def test_no_pole_crash(self):
        rng = np.random.default_rng(4)
        for dec0 in [np.pi / 2, -np.pi / 2, np.radians(89.9), np.radians(-89.9)]:
            ra, dec = sample_with_extension(0.0, dec0, np.radians(3.0), rng)
            assert np.isfinite(ra)
            assert np.isfinite(dec)
            assert -np.pi / 2 - 1e-9 <= dec <= np.pi / 2 + 1e-9

    def test_seeded_rng_is_reproducible(self):
        true_ra, true_dec, extension = 0.5, 0.2, np.radians(1.5)
        ra1, dec1 = sample_with_extension(true_ra, true_dec, extension, np.random.default_rng(7))
        ra2, dec2 = sample_with_extension(true_ra, true_dec, extension, np.random.default_rng(7))
        assert_allclose(ra1, ra2)
        assert_allclose(dec1, dec2)

    def test_default_rng_when_none(self):
        ra, dec = sample_with_extension(0.0, 0.0, np.radians(1.0))
        assert np.isfinite(ra)
        assert np.isfinite(dec)
