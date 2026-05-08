"""
Unit tests for kingmaker.utils.

Covers angular_distance (known angles, symmetry, self-distance) and
meshgrid2d (shape, values, dtype preservation).
"""

import numpy as np
from numpy.testing import assert_allclose

from kingmaker.utils import angular_distance, meshgrid2d


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
