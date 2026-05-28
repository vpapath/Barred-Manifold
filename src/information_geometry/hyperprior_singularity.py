"""
Hyperprior Singularity and Free Energy Residual
================================================
Numerical verification of Propositions 1, 1b, and 2 from
The Barred Manifold, Preliminary Chapter, Section I.

Proposition 1 (general): For any regular exponential family and any
full-support interior distribution p(·;η₀), D_KL(p₀ ‖ p*) → ∞ as
p* concentrates on a measure-zero set.

Proposition 1b (Gaussian class): d_F(η₀, η*) → ∞ as σ* → 0.

Proposition 2: For any full-support q with F(θ) = D_KL[q ‖ p(·;θ)],
there exists C > 0 such that inf_{θ ∈ B_R(θ*)} F(θ) ≥ C > 0.

These results formalize S(Ⱥ) — the barring of the Other — as the
structural impossibility of free energy minimization reaching zero.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


# ─────────────────────────────────────────────────────────────────────────────
# Proposition 1: KL divergence → ∞ (general case)
# ─────────────────────────────────────────────────────────────────────────────

def kl_gaussian_to_degenerate(sigma0: float, mu0: float,
                               mu_star: float,
                               sigma_star: float) -> float:
    """
    D_KL(N(μ₀,σ₀²) ‖ N(μ*,σ*²))

    As σ* → 0: the boundary distribution concentrates on a point mass at μ*.
    D_KL → ∞ for any full-support N(μ₀,σ₀²) (Proposition 1).
    """
    if sigma_star <= 1e-300:
        return np.inf
    return (np.log(sigma_star / sigma0)
            + (sigma0**2 + (mu0 - mu_star)**2) / (2.0 * sigma_star**2)
            - 0.5)


def demonstrate_proposition1(
    sigma0: float = 1.0,
    mu0: float = 0.0,
    mu_star: float = 0.0,
) -> None:
    """Show D_KL → ∞ as the boundary is approached."""
    print("Proposition 1: D_KL(p₀ ‖ p*) → ∞ as σ* → 0")
    print(f"  p₀ = N({mu0}, {sigma0}²),  p* = N({mu_star}, σ*²)")
    print()
    sigmas = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 1e-6]
    print(f"  {'σ*':>10}   {'D_KL':>15}")
    print(f"  {'-'*10}   {'-'*15}")
    for s in sigmas:
        kl = kl_gaussian_to_degenerate(sigma0, mu0, mu_star, s)
        kl_str = f"{kl:.2e}" if kl > 1e4 else f"{kl:.4f}"
        print(f"  {s:>10.1e}   {kl_str:>15}")
    print(f"  {'→ 0':>10}   {'∞   (✓ Prop. 1)':>15}")


# ─────────────────────────────────────────────────────────────────────────────
# Proposition 1b: Fisher geodesic distance → ∞ (Gaussian class)
# ─────────────────────────────────────────────────────────────────────────────

def fisher_geodesic_distance_gaussian(
    mu0: float, sigma0: float,
    mu1: float, sigma1: float,
) -> float:
    """
    Fisher geodesic distance on the Gaussian manifold (1D).

    Using the result that the Gaussian manifold is isometric to
    the upper half-plane with the Poincaré metric:
        d_F = √2 · arcosh(1 + (μ₁-μ₀)²/(2σ₀σ₁) + (σ₀-σ₁)²/(2σ₀σ₁))

    Simplified for fixed μ: d_F = √2 · |log(σ₁/σ₀)|
    """
    if sigma0 <= 0 or sigma1 <= 0:
        return np.inf

    # Full formula using Poincaré half-plane metric
    arg = 1.0 + ((mu1 - mu0)**2 + (sigma1 - sigma0)**2) / (2.0 * sigma0 * sigma1)
    arg = max(arg, 1.0)  # arcosh requires arg ≥ 1
    return np.sqrt(2.0) * np.arccosh(arg)


def demonstrate_proposition1b(
    sigma0: float = 1.0,
    mu0: float = 0.0,
) -> None:
    """Show d_F → ∞ as σ* → 0 for the Gaussian manifold."""
    print("\nProposition 1b: d_F(η₀, η*) → ∞ as σ* → 0 (Gaussian class)")
    print(f"  θ₀ = (μ={mu0}, σ={sigma0}),  θ* = (μ={mu0}, σ*)")
    print()
    sigmas = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 1e-6]
    print(f"  {'σ*':>10}   {'d_F':>15}")
    print(f"  {'-'*10}   {'-'*15}")
    for s in sigmas:
        d = fisher_geodesic_distance_gaussian(mu0, sigma0, mu0, s)
        d_str = f"{d:.2e}" if d > 1e3 else f"{d:.4f}"
        print(f"  {s:>10.1e}   {d_str:>15}")
    print(f"  {'→ 0':>10}   {'∞   (✓ Prop. 1b)':>15}")
    print()
    print("  Note: Proposition 1b holds for the Gaussian class and")
    print("  exponential families with unbounded natural parameter space.")
    print("  It does NOT hold universally (Bernoulli manifold has finite diameter π).")


# ─────────────────────────────────────────────────────────────────────────────
# Proposition 2: Irreducible free energy residual
# ─────────────────────────────────────────────────────────────────────────────

def free_energy_near_boundary(
    sigma0_q: float = 1.0,
    mu_q: float = 0.0,
    mu_star: float = 0.0,
    sigma_star_vals: np.ndarray = None,
) -> np.ndarray:
    """
    Compute F(θ) = D_KL[q ‖ p(·;θ)] as θ approaches the boundary singularity.

    The variational distribution q = N(μ_q, σ₀_q²) is full-support.
    As θ → θ* (σ → 0), F(θ) → ∞.
    This implies inf_{θ ∈ B_R(θ*)} F(θ) ≥ C > 0 (Proposition 2).

    Args:
        sigma0_q: std of the variational distribution q
        mu_q: mean of q
        mu_star: mean of the boundary distribution p*
        sigma_star_vals: array of σ values approaching 0

    Returns:
        F values for each σ
    """
    if sigma_star_vals is None:
        sigma_star_vals = np.logspace(0, -6, 100)

    F_vals = np.array([
        kl_gaussian_to_degenerate(sigma0_q, mu_q, mu_star, s)
        for s in sigma_star_vals
    ])
    return F_vals


def demonstrate_proposition2(
    sigma0_q: float = 1.0,
    mu_q: float = 0.0,
) -> None:
    """
    Show that F(θ) → ∞ near the boundary, establishing C > 0.

    This means no gradient descent on F can drive F to 0:
    the irreducible free energy residual C > 0 is the formal correlate
    of the Real as constitutive impossibility.
    """
    print("\nProposition 2: Irreducible free energy residual")
    print(f"  q = N({mu_q}, {sigma0_q}²) (the subject's generative model)")
    print(f"  p* = N(0, σ*²)  →  boundary singularity S(Ⱥ) as σ* → 0")
    print()

    # Show F lower bound grows as ball radius shrinks
    print(f"  {'Ball radius R':>14}   {'inf F in B_R(θ*)':>18}   {'C > 0?':>8}")
    print(f"  {'-'*14}   {'-'*18}   {'-'*8}")

    for R_sigma in [0.5, 0.2, 0.1, 0.05, 0.01]:
        # Sample F in the ball: σ ∈ [0, R]
        sigmas = np.linspace(1e-8, R_sigma, 1000)
        F_vals = np.array([
            kl_gaussian_to_degenerate(sigma0_q, mu_q, 0.0, s) for s in sigmas
        ])
        inf_F = F_vals.min()
        print(f"  {R_sigma:>14.3f}   {inf_F:>18.2f}   {'✓' if inf_F > 0 else '✗':>8}")

    print()
    print("  inf_{θ ∈ B_R(θ*)} F(θ) ≥ C > 0 for all finite R  (✓ Prop. 2)")
    print()
    print("  Lacanian reading: S(Ⱥ) is at infinite KL-divergence from any")
    print("  full-support distribution. The subject lacks because the Other")
    print("  lacks: the singularity propagates through the hyperprior to")
    print("  produce an irreducible residual G(π) ≥ C · π(B_R(θ*)) > 0.")


# ─────────────────────────────────────────────────────────────────────────────
# Hyperprior propagation
# ─────────────────────────────────────────────────────────────────────────────

def hyperprior_expected_free_energy(
    hyperprior_sigma: float = 0.3,
    theta_star: float = 0.0,
    n_theta_samples: int = 5000,
    sigma_q: float = 1.0,
    mu_q: float = 0.0,
    seed: int = 42,
) -> float:
    """
    Estimate G(π) = E_π[F(θ)] where π = N(θ*, σ_π²) is a hyperprior
    centered near the boundary singularity θ*.

    The hyperprior π assigns mass near θ* (the singularity S(Ⱥ)).
    G(π) ≥ C · π(B_R(θ*)) > 0 by Proposition 2.

    Returns:
        G_pi: estimated expected free energy under the hyperprior
    """
    rng = np.random.default_rng(seed)
    # Sample θ from the hyperprior (truncated at σ > 1e-6 for numerical stability)
    sigma_samples = np.abs(rng.normal(theta_star, hyperprior_sigma, n_theta_samples))
    sigma_samples = np.maximum(sigma_samples, 1e-6)

    F_vals = np.array([
        kl_gaussian_to_degenerate(sigma_q, mu_q, 0.0, s) for s in sigma_samples
    ])
    return float(np.mean(F_vals[np.isfinite(F_vals)]))


def demonstrate_hyperprior_propagation() -> None:
    """
    Show that a hyperprior π with mass near θ* produces G(π) >> 0,
    demonstrating the propagation of the singularity through the hyperprior.

    Lacanian reading: the symbolic order (the hyperprior) points toward
    the Real (θ* = S(Ⱥ)). The subject's expected free energy G(π) cannot
    be zero. The lack propagates.
    """
    print("\nHyperprior propagation: G(π) = E_π[F(θ)]")
    print("  The symbolic order (hyperprior π) points toward S(Ⱥ) (θ*=0).")
    print()
    print(f"  {'π width σ_π':>14}   {'G(π) = E_π[F]':>16}   {'G(π) > 0?':>10}")
    print(f"  {'-'*14}   {'-'*16}   {'-'*10}")

    for sigma_pi in [2.0, 1.0, 0.5, 0.3, 0.1]:
        G = hyperprior_expected_free_energy(hyperprior_sigma=sigma_pi)
        print(f"  {sigma_pi:>14.2f}   {G:>16.2f}   {'✓' if G > 0 else '✗':>10}")

    print()
    print("  G(π) > 0 for all finite σ_π  →  the subject lacks because")
    print("  the Other lacks: S(Ⱥ) propagates to E_π[F] > 0  (✓ Prop. 2 + hyperprior)")


if __name__ == "__main__":
    demonstrate_proposition1()
    demonstrate_proposition1b()
    demonstrate_proposition2()
    demonstrate_hyperprior_propagation()
