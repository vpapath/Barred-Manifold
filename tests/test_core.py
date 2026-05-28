"""
Core tests for The Barred Manifold computational package.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '../src')

from information_geometry.gaussian_manifold import (
    to_natural, from_natural, to_expectation, from_expectation,
    kl_divergence_gaussians, demonstrate_incommensurability
)
from information_geometry.hyperprior_singularity import (
    kl_gaussian_to_degenerate, fisher_geodesic_distance_gaussian
)
from ising_fep.ising_fep_bridge import (
    irt_to_ising, ising_hamiltonian, log_partition_function,
    variational_free_energy
)


class TestProposition1:
    """Proposition 1: KL divergence → ∞ as boundary is approached."""

    def test_kl_finite_interior(self):
        kl = kl_gaussian_to_degenerate(1.0, 0.0, 0.0, 1.0)
        assert np.isfinite(kl)
        assert kl >= 0

    def test_kl_diverges_at_boundary(self):
        kl = kl_gaussian_to_degenerate(1.0, 0.0, 0.0, 1e-8)
        assert kl > 1e10

    def test_kl_monotone_increasing(self):
        """KL should increase monotonically as σ* decreases."""
        sigmas = [1.0, 0.5, 0.1, 0.01, 0.001]
        kls = [kl_gaussian_to_degenerate(1.0, 0.0, 0.0, s) for s in sigmas]
        for i in range(len(kls) - 1):
            assert kls[i] < kls[i + 1]


class TestProposition1b:
    """Proposition 1b: Fisher geodesic distance → ∞ (Gaussian class)."""

    def test_distance_finite_interior(self):
        d = fisher_geodesic_distance_gaussian(0.0, 1.0, 2.0, 2.0)
        assert np.isfinite(d)
        assert d > 0

    def test_distance_diverges_at_boundary(self):
        d = fisher_geodesic_distance_gaussian(0.0, 1.0, 0.0, 1e-8)
        assert d > 100

    def test_distance_symmetric(self):
        d01 = fisher_geodesic_distance_gaussian(0.0, 1.0, 0.0, 2.0)
        d10 = fisher_geodesic_distance_gaussian(0.0, 2.0, 0.0, 1.0)
        assert np.isclose(d01, d10)


class TestIsingFEPBridge:
    """F*(σ) = H(σ) + log Z identity."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        N = 5
        a = rng.uniform(0.5, 2.0, N)
        b = rng.uniform(-1.0, 1.0, N)
        self.h, self.J = irt_to_ising(a, b)
        self.log_Z = log_partition_function(self.h, self.J)
        self.N = N

    def test_fep_ising_identity(self):
        """F*(σ) = H(σ) + log Z exactly."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            sigma = rng.integers(0, 2, self.N).astype(float)
            H = ising_hamiltonian(sigma, self.h, self.J)
            F_star = variational_free_energy(sigma, self.h, self.J, self.log_Z)
            assert np.isclose(F_star, H + self.log_Z), \
                f"F*={F_star:.6f} ≠ H+logZ={H + self.log_Z:.6f}"

    def test_probabilities_sum_to_one(self):
        """Boltzmann probabilities sum to 1."""
        from itertools import product
        total = sum(
            np.exp(-ising_hamiltonian(np.array(s, dtype=float), self.h, self.J))
            for s in product([0, 1], repeat=self.N)
        )
        Z = np.exp(self.log_Z)
        assert np.isclose(total / Z, 1.0, rtol=1e-5)


class TestIncommensurability:
    """Supplement 3: e-geodesics and m-geodesics are incommensurable."""

    def test_incommensurable_when_sigmas_differ(self):
        result = demonstrate_incommensurability(0.0, 1.0, 2.0, 2.0)
        assert result['incommensurable'], "Should be incommensurable when σ₀ ≠ σ₁"

    def test_commensurable_when_sigmas_equal(self):
        """e- and m-geodesics coincide when σ₀ = σ₁."""
        from information_geometry.gaussian_manifold import e_geodesic, m_geodesic
        mu0, sigma = 0.0, 1.5
        mu1 = 3.0
        e_mus, e_sigs = e_geodesic(mu0, sigma, mu1, sigma, n_pts=50)
        m_mus, m_sigs = m_geodesic(mu0, sigma, mu1, sigma, n_pts=50)
        # Midpoints should coincide
        mid = 25
        assert np.isclose(e_mus[mid], m_mus[mid], atol=1e-3), \
            f"μ midpoints differ: e={e_mus[mid]:.4f}, m={m_mus[mid]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
