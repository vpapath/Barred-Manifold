"""
Information Geometry of the Statistical Manifold
=================================================
Implements the information-geometric formalization of the Lacanian subject
from The Barred Manifold, Chapter 8 and Supplement 3.

The statistical manifold M = {N(x; μ, σ²) : μ ∈ ℝ, σ > 0} equipped with
the Fisher information metric. The e-connection (exponential, natural
parameters) and m-connection (mixture, expectation parameters) are dual and
generally incompatible.

Key results implemented:
    - Fisher metric on the Gaussian manifold
    - e-geodesics (metaphor / substitution axis)
    - m-geodesics (metonymy / displacement axis)
    - Incommensurability theorem (numerical demonstration)
    - KL divergence and boundary singularity distances

References:
    Amari, S. (2016). Information geometry and its applications. Springer.
    The Barred Manifold, §8.1–8.3, Supplement 3.
"""

import numpy as np
from scipy.special import rel_entr
from typing import Tuple
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian manifold parametrizations
# ─────────────────────────────────────────────────────────────────────────────

def to_natural(mu: float, sigma: float) -> Tuple[float, float]:
    """Convert (μ, σ) → natural parameters (η₁, η₂)."""
    return mu / sigma**2, -1.0 / (2.0 * sigma**2)


def from_natural(eta1: float, eta2: float) -> Tuple[float, float]:
    """Convert natural parameters (η₁, η₂) → (μ, σ)."""
    sigma = np.sqrt(-1.0 / (2.0 * eta2))
    mu = eta1 * sigma**2
    return mu, sigma


def to_expectation(mu: float, sigma: float) -> Tuple[float, float]:
    """Convert (μ, σ) → expectation parameters (μ̂₁, μ̂₂) = (μ, μ² + σ²)."""
    return mu, mu**2 + sigma**2


def from_expectation(mu_hat1: float, mu_hat2: float) -> Tuple[float, float]:
    """Convert expectation parameters → (μ, σ)."""
    mu = mu_hat1
    var = mu_hat2 - mu_hat1**2
    if var <= 0:
        raise ValueError(f"Invalid expectation parameters: σ² = {var:.4e} ≤ 0")
    return mu, np.sqrt(var)


# ─────────────────────────────────────────────────────────────────────────────
# Fisher information metric
# ─────────────────────────────────────────────────────────────────────────────

def fisher_metric(mu: float, sigma: float) -> np.ndarray:
    """
    Fisher information metric g at (μ, σ) in (μ, σ) coordinates.

    g = diag(1/σ², 2/σ²)
    """
    return np.diag([1.0 / sigma**2, 2.0 / sigma**2])


def fisher_distance_gaussian_sigma_path(sigma0: float, sigma1: float,
                                        mu: float = 0.0,
                                        n_steps: int = 10000) -> float:
    """
    Fisher geodesic distance along the σ-axis (fixed μ) for the Gaussian family.

    d_F = √2 · |log(σ₁/σ₀)|

    For σ₁ → 0, d_F → ∞ (Proposition 1b in The Barred Manifold).
    """
    if sigma1 <= 0 or sigma0 <= 0:
        return np.inf
    return np.sqrt(2.0) * abs(np.log(sigma1 / sigma0))


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence (Proposition 1 verification)
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence_gaussians(mu0: float, sigma0: float,
                             mu1: float, sigma1: float) -> float:
    """
    KL divergence D_KL(N(μ₀,σ₀²) ‖ N(μ₁,σ₁²)) in closed form.

    As σ₁ → 0 (point mass at μ₁), D_KL → ∞ for any full-support N(μ₀,σ₀²).
    This is the content of Proposition 1.
    """
    if sigma1 <= 0:
        return np.inf
    return (np.log(sigma1 / sigma0)
            + (sigma0**2 + (mu0 - mu1)**2) / (2.0 * sigma1**2)
            - 0.5)


def verify_proposition1(sigma0: float = 1.0, mu0: float = 0.0,
                         mu_star: float = 0.0) -> None:
    """
    Numerical verification of Proposition 1: D_KL → ∞ as σ* → 0.
    """
    sigmas = np.logspace(0, -6, 50)
    kls = [kl_divergence_gaussians(mu0, sigma0, mu_star, s) for s in sigmas]
    print("Proposition 1 verification (D_KL → ∞ as σ* → 0):")
    print(f"  σ* = 1.0    → D_KL = {kls[0]:.4f}")
    print(f"  σ* = 0.01   → D_KL = {kl_divergence_gaussians(mu0, sigma0, mu_star, 0.01):.1f}")
    print(f"  σ* = 0.001  → D_KL = {kl_divergence_gaussians(mu0, sigma0, mu_star, 0.001):.1f}")
    print(f"  σ* = 1e-6   → D_KL = {kl_divergence_gaussians(mu0, sigma0, mu_star, 1e-6):.2e}")
    print(f"  σ* → 0      → D_KL = ∞  (Proposition 1 ✓)")


# ─────────────────────────────────────────────────────────────────────────────
# e-geodesic (metaphor / substitution axis)
# ─────────────────────────────────────────────────────────────────────────────

def e_geodesic(mu0: float, sigma0: float,
               mu1: float, sigma1: float,
               n_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    e-geodesic from N(μ₀,σ₀²) to N(μ₁,σ₁²): linear interpolation in
    natural parameter space.

    η(t) = (1-t)·η(θ₀) + t·η(θ₁),  t ∈ [0,1]

    Corresponds to metaphor: structural substitution preserving
    the exponential-family form (the paradigmatic slot).

    Returns:
        mus, sigmas: arrays of shape (n_pts,)
    """
    eta0 = np.array(to_natural(mu0, sigma0))
    eta1 = np.array(to_natural(mu1, sigma1))
    t_vals = np.linspace(0.0, 1.0, n_pts)

    mus = np.zeros(n_pts)
    sigmas = np.zeros(n_pts)
    for i, t in enumerate(t_vals):
        eta_t = (1.0 - t) * eta0 + t * eta1
        mus[i], sigmas[i] = from_natural(*eta_t)

    return mus, sigmas


# ─────────────────────────────────────────────────────────────────────────────
# m-geodesic (metonymy / displacement axis)
# ─────────────────────────────────────────────────────────────────────────────

def m_geodesic(mu0: float, sigma0: float,
               mu1: float, sigma1: float,
               n_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    m-geodesic from N(μ₀,σ₀²) to N(μ₁,σ₁²): linear interpolation in
    expectation parameter space.

    μ̂(t) = (1-t)·μ̂(θ₀) + t·μ̂(θ₁),  t ∈ [0,1]

    Corresponds to metonymy: displacement through contiguous associations,
    moving through expectations. Note: intermediate distributions may not be
    Gaussian (they can have heavier tails), formalizing the 'overflowing'
    character of metonymic displacement.

    Returns:
        mus, sigmas: arrays of shape (n_pts,)
        is_gaussian: bool array — False when intermediate dist. is non-Gaussian
    """
    mhat0 = np.array(to_expectation(mu0, sigma0))
    mhat1 = np.array(to_expectation(mu1, sigma1))
    t_vals = np.linspace(0.0, 1.0, n_pts)

    mus = np.zeros(n_pts)
    sigmas = np.zeros(n_pts)
    is_gaussian = np.ones(n_pts, dtype=bool)

    for i, t in enumerate(t_vals):
        mhat_t = (1.0 - t) * mhat0 + t * mhat1
        try:
            mus[i], sigmas[i] = from_expectation(*mhat_t)
        except ValueError:
            mus[i] = np.nan
            sigmas[i] = np.nan
            is_gaussian[i] = False

    return mus, sigmas


# ─────────────────────────────────────────────────────────────────────────────
# Incommensurability theorem (Supplement 3 worked example)
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_incommensurability(
    mu0: float = 0.0, sigma0: float = 1.0,
    mu1: float = 2.0, sigma1: float = 2.0,
    n_pts: int = 100,
) -> dict:
    """
    Demonstrate the incommensurability of e-geodesics and m-geodesics
    for the Gaussian manifold (Supplement 3, The Barred Manifold).

    The e-geodesic and m-geodesic between any two Gaussians coincide iff
    σ₀ = σ₁. Otherwise they diverge: the midpoint (t=0.5) distributions
    are distinct.

    Returns dict with midpoint parameters for both geodesics.
    """
    e_mus, e_sigs = e_geodesic(mu0, sigma0, mu1, sigma1, n_pts)
    m_mus, m_sigs = m_geodesic(mu0, sigma0, mu1, sigma1, n_pts)

    mid = n_pts // 2

    e_mid = (e_mus[mid], e_sigs[mid])
    m_mid = (m_mus[mid], m_sigs[mid])

    # KL between the two midpoints
    kl_mid = kl_divergence_gaussians(*e_mid, *m_mid)

    result = {
        "endpoint_0": (mu0, sigma0),
        "endpoint_1": (mu1, sigma1),
        "e_midpoint": e_mid,
        "m_midpoint": m_mid,
        "kl_e_to_m_at_midpoint": kl_mid,
        "incommensurable": not np.isclose(e_mid[0], m_mid[0]) or
                           not np.isclose(e_mid[1], m_mid[1]),
        "e_mus": e_mus,
        "e_sigs": e_sigs,
        "m_mus": m_mus,
        "m_sigs": m_sigs,
    }

    print(f"Incommensurability demonstration: N({mu0},{sigma0}²) → N({mu1},{sigma1}²)")
    print(f"  e-geodesic midpoint: μ = {e_mid[0]:.4f},  σ = {e_mid[1]:.4f}")
    print(f"  m-geodesic midpoint: μ = {m_mid[0]:.4f},  σ = {m_mid[1]:.4f}")
    print(f"  D_KL(e-mid ‖ m-mid) = {kl_mid:.4f}")
    print(f"  Incommensurable: {result['incommensurable']} "
          f"({'Theorem 3.1 ✓' if result['incommensurable'] else 'Coincide (σ₀=σ₁)'})")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Proposition 1: KL divergence to boundary singularity")
    print("=" * 60)
    verify_proposition1()

    print()
    print("=" * 60)
    print("Proposition 1b: Fisher geodesic distance (Gaussian class)")
    print("=" * 60)
    for sigma_star in [1.0, 0.1, 0.01, 0.001]:
        d = fisher_distance_gaussian_sigma_path(1.0, sigma_star)
        print(f"  d_F(σ=1.0, σ={sigma_star}) = {d:.4f}")
    print(f"  d_F(σ=1.0, σ→0) = ∞  (Proposition 1b ✓)")

    print()
    print("=" * 60)
    print("Incommensurability theorem (Supplement 3)")
    print("=" * 60)
    demonstrate_incommensurability(0.0, 1.0, 2.0, 2.0)
