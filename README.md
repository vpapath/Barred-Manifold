# The Barred Manifold: Formal Foundations

This repository contains the Agda formalizations and paper sources for a series of papers applying Homotopy Type Theory to Lacan's formal apparatus.

## Papers and Code

### 01 — Sexual Difference as Universe Stratification

**Paper:** `sexual_difference_universe_stratification.pdf`
**Code:** `Sexuation.agda`
**Status:** Machine-verified in Agda 2.6.3, `--without-K --safe`

Formalizes Lacan's four formulas of sexuation using HoTT's universe hierarchy. The masculine position is Type_i-bounded; the feminine position requires universe-polymorphic operations that lift to Type_{i+1}. Three theorems: structural asymmetry, universe grounding asymmetry, and the absence of the sexual relation (*il n'y a pas de rapport sexuel*) as a universe-level result — any relation between the structures lives at Type_{i+2}.

**Key result:** `MascFem-Contradiction`, `MascGrounding`, `SexualRelation-Type` — all verified.

---

### 02 — Logical Time as Universe Lifting

**Paper:** `logical_time_universe_stratification.pdf`
**Code:** `LogicalTime.agda`
**Status:** Part One machine-verified (`--without-K --safe`); Part Two conditional on shape modality axioms (not `--safe`)

Formalizes Lacan's three moments of logical time (1945) in the universe hierarchy. The instant of the glance is a Type_i observation; the time for comprehending is oscillation that cannot self-ground within Type_i; the moment of concluding is a universe-lifting Π-type at Type_{i+1}. The *après-coup* is a theorem: the Type_{i+1} commitment constitutes the meaning of the Type_i evidence retroactively.

**Key result:** `Concluding-is-MascUniversal` — proved by `refl`. The moment of concluding and the masculine universal are definitionally identical. Logical time and sexual difference share a single formal structure.

---

### 03 — Constitutive Directionality and the Limits of Simplicial Type Theory

**Paper:** `constitutive_directionality_directed_HoTT.pdf`
**Code:** `CTT.agda`
**Status:** Shallow embedding with postulated axioms. Core theorems verified given postulates. Irreversibility encoded in formation rules. Open conjecture (constitutive directed univalence) stated but unproved.

Proposes Constitutive Type Theory (CTT): a minimal extension of HoTT with a constitutive homomorphism type `ConstHom a b` for `a : Type_i`, `b : Type_{i+1}`. The type captures the structure absent from simplicial type theory (Riehl–Shulman 2017; Gratzer–Weinberger–Buchholtz 2024): constitutive directionality, where the later term retroactively constitutes the meaning of the earlier. The *après-coup* as an elimination rule. Irreversibility enforced by the formation rule's level constraints.

**Open conjecture:** Constitutive directed univalence — types with the same constituting term are identified. Resolving this would connect CTT to the directed univalence results of Gratzer–Weinberger–Buchholtz 2024.

---

## Epistemic Status Summary

| File | Mode | Status |
|------|------|--------|
| `Sexuation.agda` | `--without-K --safe` | ✓ Fully verified |
| `LogicalTime.agda` (Part 1) | `--without-K --safe` | ✓ Fully verified |
| `LogicalTime.agda` (Part 2) | postulates | ✓ Conditional |
| `CTT.agda` | postulates | ✓ Conditional |

---

## How to Verify

Requirements: Agda 2.6.3, agda-stdlib 1.7.3

```bash
# Verify Paper 01
agda -i /path/to/agda-stdlib -i 01-sexuation 01-sexuation/Sexuation.agda

# Verify Paper 02 (Part One only — Part Two requires removing --safe)
agda -i /path/to/agda-stdlib -i 02-logical-time 02-logical-time/LogicalTime.agda

# Verify Paper 03 (postulates only, no --safe)
agda -i /path/to/agda-stdlib -i 03-directed-hott 03-directed-hott/CTT.agda
```

---

## Disclosure

These papers were developed with assistance from Claude (Anthropic) for mathematical derivation, Agda development, and structural drafting. All theoretical arguments, formal specifications, interpretive claims, and editorial choices are the author's own. The AI collaboration is disclosed in the acknowledgments of each paper.

---

## License

All content © Vassilis Papathanasiou 2026. Code released under MIT License. Papers released under CC BY 4.0.

---

## Contact

Vassilis Papathanasiou  
University of Amsterdam  
b.papath02@gmail.com  
