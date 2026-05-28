"""
Helmholtz Decomposition of NESS Flow and Clinical Regime Characterization
=========================================================================
Implements the core dynamical analysis from The Barred Manifold, Chapter 8
and §10.3.

The Helmholtz decomposition of a NESS (Non-Equilibrium Steady-State) flow F(x)
into gradient (Q) and solenoidal (Γ) components:

    F(x) = -Q ∇Φ(x) + Γ(x)

where Q∇Φ is the free-energy gradient (can be minimized) and Γ is the
solenoidal component (irrotational, cannot be minimized).

The solenoidal fraction sol_frac = ‖Γ‖ / ‖F‖ formalizes jouissance as the
irreducible surplus that free energy minimization cannot eliminate.

Clinical structures correspond to distinct Lorenz attractor regimes:
    - Neurosis:   ρ = 28.0  →  strange attractor  (λ₁ > 0, sol_frac ≈ 0.820)
    - Melancholia: ρ = 4.0  →  fixed point        (λ₁ < 0, sol_frac ≈ 0.589)
    - Psychosis:  ρ = 0.5   →  near singularity   (λ₁ < 0, sol_frac ≈ 0.460)

References:
    Friston, K. J. (2019). A free energy principle for a particular physics.
        arXiv:1906.10184.
    The Barred Manifold, §8.3, §8.4, Supplement 6.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr
from dataclasses import dataclass
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Lorenz system
# ─────────────────────────────────────────────────────────────────────────────

def lorenz(t, state, rho: float, sigma: float = 10.0, beta: float = 8/3):
    """Lorenz system dx/dt = f(x) with parameters (σ, ρ, β)."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]


def lorenz_jacobian(state, rho: float, sigma: float = 10.0, beta: float = 8/3):
    """Jacobian ∂f/∂x of the Lorenz system."""
    x, y, z = state
    return np.array([
        [-sigma,  sigma,   0.0  ],
        [rho - z, -1.0,   -x   ],
        [y,        x,    -beta ],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Lyapunov spectrum
# ─────────────────────────────────────────────────────────────────────────────

def lyapunov_spectrum(
    rho: float,
    sigma: float = 10.0,
    beta: float = 8 / 3,
    t_transient: float = 200.0,
    t_lyapunov: float = 5000.0,
    dt: float = 0.01,
    x0: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Compute the Lyapunov spectrum of the Lorenz system by QR renormalization.

    Returns:
        exponents: array of shape (3,) with λ₁ ≥ λ₂ ≥ λ₃.
    """
    n = 3

    # Run transient
    sol = solve_ivp(lorenz, [0, t_transient], x0, args=(rho, sigma, beta),
                    method='RK45', rtol=1e-10, atol=1e-12)
    state = sol.y[:, -1].copy()

    # Initialize orthonormal perturbation matrix
    Q = np.eye(n)
    exponents = np.zeros(n)
    t = 0.0
    n_steps = int(t_lyapunov / dt)

    for _ in range(n_steps):
        # Advance state
        sol = solve_ivp(lorenz, [t, t + dt], state, args=(rho, sigma, beta),
                        method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
        state_new = sol.y[:, -1]

        # Advance tangent vectors: dQ/dt = J · Q
        J = lorenz_jacobian(state, rho, sigma, beta)
        Q_new = Q + dt * (J @ Q)

        # QR decomposition for renormalization
        Q_new, R = qr(Q_new)
        # Accumulate log of diagonal of R
        exponents += np.log(np.abs(np.diag(R)))

        state = state_new
        t += dt

    return exponents / t_lyapunov


# ─────────────────────────────────────────────────────────────────────────────
# Helmholtz decomposition and solenoidal fraction
# ─────────────────────────────────────────────────────────────────────────────

def helmholtz_decompose_lorenz(
    rho: float,
    sigma: float = 10.0,
    beta: float = 8 / 3,
    t_transient: float = 200.0,
    t_sample: float = 1000.0,
    dt: float = 0.01,
    x0: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the solenoidal fraction of the Lorenz flow at NESS.

    The Helmholtz decomposition F = -Q∇Φ + Γ is estimated numerically:
        - The gradient component Q∇Φ(x) is approximated as the time-averaged
          flow direction (pointing toward the attractor center of mass).
        - The solenoidal component Γ(x) = F(x) - Q∇Φ(x) is the residual.

    The solenoidal fraction sol_frac = E[‖Γ‖] / E[‖F‖] is the fraction of
    the flow that cannot be attributed to free energy gradient descent.

    Returns:
        sol_frac: scalar solenoidal fraction in [0, 1]
        trajectory: (n_pts, 3) array of trajectory states
        gradient_norms: (n_pts,) gradient flow norms
        solenoidal_norms: (n_pts,) solenoidal flow norms
    """
    # Run transient
    sol = solve_ivp(lorenz, [0, t_transient], x0, args=(rho, sigma, beta),
                    method='RK45', rtol=1e-10, atol=1e-12)
    state = sol.y[:, -1].copy()

    # Collect trajectory and flow vectors
    t_pts = np.arange(0, t_sample, dt)
    sol = solve_ivp(lorenz, [0, t_sample], state, args=(rho, sigma, beta),
                    method='RK45', rtol=1e-10, atol=1e-12,
                    t_eval=t_pts, dense_output=False)
    traj = sol.y.T  # (n_pts, 3)

    # Flow vectors at each point
    flows = np.array([lorenz(0, s, rho, sigma, beta) for s in traj])  # (n_pts, 3)

    # Estimate attractor centre of mass
    centre = traj.mean(axis=0)

    # Gradient component: projection of flow onto (x - centre)
    # This approximates the drift toward/away from the NESS density peak
    radial = traj - centre  # (n_pts, 3)
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    radial_unit = radial / radial_norm

    proj = np.sum(flows * radial_unit, axis=1, keepdims=True)  # scalar projection
    gradient_component = proj * radial_unit  # (n_pts, 3)
    solenoidal_component = flows - gradient_component  # (n_pts, 3)

    g_norms = np.linalg.norm(gradient_component, axis=1)
    s_norms = np.linalg.norm(solenoidal_component, axis=1)
    f_norms = np.linalg.norm(flows, axis=1)

    sol_frac = s_norms.mean() / (f_norms.mean() + 1e-12)

    return sol_frac, traj, g_norms, s_norms


# ─────────────────────────────────────────────────────────────────────────────
# Clinical regime characterization
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClinicalRegime:
    name: str
    rho: float
    lambda1: float          # largest Lyapunov exponent
    kaplan_yorke_dim: float
    sol_frac: float
    attractor_type: str


def characterize_clinical_regimes(verbose: bool = True) -> dict:
    """
    Characterize the three Lacanian clinical structures as Lorenz attractor
    regimes, reproducing the results in The Barred Manifold §8.4.

    Returns:
        dict mapping clinical structure name to ClinicalRegime dataclass.
    """
    REGIMES = [
        ("neurosis",   28.0,  "strange attractor"),
        ("melancholia", 4.0,  "fixed point"),
        ("psychosis",   0.5,  "near singularity"),
    ]

    results = {}

    for name, rho, atype in REGIMES:
        if verbose:
            print(f"\nComputing {name} (ρ = {rho})...")

        # Lyapunov spectrum
        exponents = lyapunov_spectrum(rho, t_lyapunov=2000.0)
        lambda1 = exponents[0]

        # Kaplan-Yorke dimension
        sorted_exp = np.sort(exponents)[::-1]
        cumsum = np.cumsum(sorted_exp)
        j = np.searchsorted(-cumsum, 0) - 1
        j = max(0, min(j, len(exponents) - 2))
        if abs(sorted_exp[j + 1]) > 1e-12:
            ky_dim = j + 1 + cumsum[j] / abs(sorted_exp[j + 1])
        else:
            ky_dim = float(j + 1)

        # Solenoidal fraction
        sol_frac, *_ = helmholtz_decompose_lorenz(rho, t_sample=500.0)

        regime = ClinicalRegime(
            name=name,
            rho=rho,
            lambda1=lambda1,
            kaplan_yorke_dim=max(0.0, ky_dim),
            sol_frac=sol_frac,
            attractor_type=atype,
        )
        results[name] = regime

        if verbose:
            print(f"  λ₁         = {lambda1:.4f}")
            print(f"  D_KY       = {regime.kaplan_yorke_dim:.4f}")
            print(f"  sol_frac   = {sol_frac:.4f}")
            print(f"  type       = {atype}")

    return results


if __name__ == "__main__":
    results = characterize_clinical_regimes(verbose=True)

    print("\n" + "=" * 60)
    print("Summary (Table 8.1 in The Barred Manifold)")
    print("=" * 60)
    print(f"{'Structure':<14} {'ρ':>6} {'λ₁':>8} {'D_KY':>8} {'sol_frac':>10} Type")
    print("-" * 60)
    for r in results.values():
        print(f"{r.name:<14} {r.rho:>6.1f} {r.lambda1:>8.4f} "
              f"{r.kaplan_yorke_dim:>8.4f} {r.sol_frac:>10.4f}  {r.attractor_type}")
