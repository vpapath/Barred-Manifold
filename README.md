# The Barred Manifold

**Lacanian Psychoanalysis, Information Geometry, and the Free Energy Principle**

Vassilis Papathanasiou — b.papath02@gmail.com  
Department of Psychology, National and Kapodistrian University of Athens  
Research Master's in Brain and Cognitive Sciences (Cognitive Neuroscience), University of Amsterdam, 2026–

---

## Overview

This repository contains the complete working documents for the research program developed in *The Barred Manifold*: a formal reconstruction of Lacanian psychoanalytic theory in information geometry and active inference, conducted as a bidirectional formal dialogue. The formalization direction maps Lacanian structural claims to their mathematical successors in the Fisher metric, the Helmholtz decomposition, and variational inference. The extension direction proposes formal objects — the structural hyperprior with singularity, the category of symbolic positions, the non-closing trajectory — that the FEP literature did not previously contain.

The project constitutes a progressive research programme in the Lakatosian sense: hard core (Borromean entanglement of RSI, constitutive incompleteness of the subject, irreducible solenoidal surplus), protective belt (specific formalizations), positive heuristic (three empirical predictions in §10.5 of the main thesis).

---

## Repository Structure

```
barred_manifold/
├── thesis/
│   └── the_barred_manifold_complete.docx     # Main thesis (733 paragraphs, ~40k words)
│
├── papers/
│   ├── paper_barred_other_hyperprior.docx    # Three problems in Lacanian active inference
│   ├── paper_sexuation_zfc_hott.docx         # Formulas of sexuation in ZFC and HoTT
│   ├── paper_borromean_triple_network.docx   # Borromean RSI and triple network
│   ├── paper_information_geometry_subject.docx  # Statistical manifold and Lacanian subject
│   ├── paper_transference_logical_time.docx  # Transference and logical time
│   ├── paper_symptom_networks_free_energy.docx  # Symptom networks as free-energy landscapes
│   └── loudovikos_fep_paper.docx             # Dialogical personhood and the barred Other
│
├── supplements/
│   └── epistemology_frameworks.docx          # Epistemological framework research report
│
└── README.md
```

---

## Core Formal Results

| Result | Location | Status |
|--------|----------|--------|
| Proposition 1: KL divergence to boundary (general) | Preliminary Ch. | ✓ Proved |
| Proposition 1b: Fisher geodesic distance (Gaussian class) | Preliminary Ch. | ✓ Proved |
| Proposition 2: Irreducible free energy residual | Preliminary Ch. | ✓ Proved |
| Ising-IRT free energy identity F*(σ) = H(σ) + log Z | §10.4 | ✓ Derived |
| Metaphor/metonymy as e/m-geodesics (Gaussian worked example) | §2.3 | ✓ Executed |
| Après-coup as Bayesian theorem | §2.2 | ✓ Proved |
| Naturality conditions: all five morphisms | §10.1 | ✓ Verified |
| General exponential family Fisher geodesic result | — | Open (dissertation direction) |

---

## Empirical Predictions (§10.5)

Three testable predictions with explicit datasets, pipelines, and disconfirmation conditions:

1. **Topological signatures**: neurotic symptom networks → high β₁ (persistent homology), psychotic → collapsed β₁. Dataset: NESDA, Borsboom lab networks. Pipeline: Gudhi.
2. **Borromean conditional independence**: triple network resting-state fMRI shows pairwise near-independence with non-zero three-way interaction. Dataset: HCP, COBRE.
3. **Attractor regime signatures**: solenoidal fraction differences across diagnostic groups in longitudinal ESM data. Dataset: Maastricht ESM, TRAILS.

---

## Six Standalone Papers

| Paper | Target journal | Status |
|-------|---------------|--------|
| Barred Other as Structural Hyperprior | Psychoanalysis, Culture & Society / Frontiers in Psychology | Ready |
| Formulas of Sexuation in ZFC and HoTT | JAPA / Psychoanalysis, Culture & Society | Ready |
| Borromean RSI and Triple Network | Neuropsychoanalysis | Ready |
| Statistical Manifold and Lacanian Subject | Neuropsychoanalysis / J. Theoretical Biology | Ready |
| Transference and Logical Time | Frontiers in Psychiatry / Psychoanalytic Dialogues | Ready |
| Symptom Networks as Free-Energy Landscapes | PsyArXiv (preprint) / Psychological Review | Ready — push first |

---

## Key References

- Amari, S. (2016). *Information geometry and its applications*. Springer.
- Friston, K. J. (2019). A free energy principle for a particular physics. *arXiv:1906.10184*.
- Hesp, C., Smith, R., Friston, K. J., & Ramstead, M. J. D. (2021). Deeply felt affect. *Neural Computation, 33*(2), 398–446.
- Lacan, J. (2002). *Écrits*. Norton.
- Ladyman, J., & Ross, D. (2007). *Every thing must go*. Oxford University Press.
- Marsman, M., et al. (2018). An introduction to network psychometrics. *Multivariate Behavioral Research, 53*(1), 15–35.

---

## Contact / Correspondence

Vassilis Papathanasiou · b.papath02@gmail.com  
Priority contacts: Sacha Marsman (UvA, Ising-IRT), Ariane Bazan (Université de Lorraine)
