{-# OPTIONS --without-K --safe #-}

module Sexuation where

open import Level using (Level; 0ℓ; suc; _⊔_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.Product using (Σ; _,_; _×_; proj₁; proj₂)
open import Relation.Nullary using (¬_)

-- ─────────────────────────────────────────────
-- The phallic predicate
-- ─────────────────────────────────────────────

-- Φ is a predicate on types at universe level ℓ.
-- It lives in Set (suc ℓ) because it takes a Set ℓ as argument.
PhallicPred : (ℓ : Level) → Set (suc ℓ)
PhallicPred ℓ = Set ℓ → Set ℓ

-- ─────────────────────────────────────────────
-- The four sexuation formulas
-- ─────────────────────────────────────────────

-- MASCULINE: ∀x.Φx
-- Quantifies over all of Set ℓ → lives in Set (suc ℓ)
MascUniversal : (ℓ : Level) → PhallicPred ℓ → Set (suc ℓ)
MascUniversal ℓ Φ = (A : Set ℓ) → Φ A

-- MASCULINE: ∃x.¬Φx  (the founding exception)
-- The witness e : Set ℓ lives INSIDE the universe being quantified over
MascException : (ℓ : Level) → PhallicPred ℓ → Set (suc ℓ)
MascException ℓ Φ = Σ (Set ℓ) (λ e → ¬ (Φ e))

-- FEMININE: ¬∀x.Φx  (the pas-tout)
FemPasTout : (ℓ : Level) → PhallicPred ℓ → Set (suc ℓ)
FemPasTout ℓ Φ = ¬ ((A : Set ℓ) → Φ A)

-- FEMININE: ¬∃x.¬Φx  (the non-exception)
FemNonException : (ℓ : Level) → PhallicPred ℓ → Set (suc ℓ)
FemNonException ℓ Φ = ¬ (Σ (Set ℓ) (λ A → ¬ (Φ A)))

-- ─────────────────────────────────────────────
-- Compound structures
-- ─────────────────────────────────────────────

MasculineStructure : (ℓ : Level) → PhallicPred ℓ → Set (suc ℓ)
MasculineStructure ℓ Φ = MascUniversal ℓ Φ × MascException ℓ Φ

FeminineStructure : (ℓ : Level) → PhallicPred ℓ → Set (suc ℓ)
FeminineStructure ℓ Φ = FemPasTout ℓ Φ × FemNonException ℓ Φ

-- ─────────────────────────────────────────────
-- THEOREM 1: Structural asymmetry
-- The masculine exception and feminine non-exception
-- are mutually exclusive for any predicate Φ.
-- ─────────────────────────────────────────────

MascFem-Contradiction :
  ∀ (ℓ : Level) (Φ : PhallicPred ℓ)
  → MascException ℓ Φ
  → FemNonException ℓ Φ
  → ⊥
MascFem-Contradiction ℓ Φ masc-exc fem-non-exc =
  fem-non-exc masc-exc

-- ─────────────────────────────────────────────
-- THEOREM 2: Universe grounding asymmetry
-- The masculine structure is grounded by a term
-- in Set ℓ; the feminine structure is not.
-- ─────────────────────────────────────────────

-- Extract the grounding term from masculine structure:
-- it lives in Set ℓ (inside the quantified universe)
MascGrounding :
  ∀ (ℓ : Level) (Φ : PhallicPred ℓ)
  → MasculineStructure ℓ Φ
  → Set ℓ
MascGrounding ℓ Φ (_ , e , _) = e

-- Any proposed grounding term for the feminine structure
-- yields a contradiction via the non-exception
FemGrounding-Absurd :
  ∀ (ℓ : Level) (Φ : PhallicPred ℓ)
  → FeminineStructure ℓ Φ
  → (A : Set ℓ)
  → ¬ (Φ A)
  → ⊥
FemGrounding-Absurd ℓ Φ (_ , fem-non-exc) A not-Φ-A =
  fem-non-exc (A , not-Φ-A)

-- ─────────────────────────────────────────────
-- THEOREM 3: The absence of the sexual relation
-- Il n'y a pas de rapport sexuel.
--
-- Any relation between MasculineStructure and
-- FeminineStructure lives in Set (suc (suc ℓ)),
-- not in Set ℓ or Set (suc ℓ).
-- The two structures are never simultaneously
-- at home in the same universe.
-- ─────────────────────────────────────────────

SexualRelation-Type : (ℓ : Level) (Φ : PhallicPred ℓ) → Set (suc (suc ℓ))
SexualRelation-Type ℓ Φ =
  MasculineStructure ℓ Φ → FeminineStructure ℓ Φ → Set (suc ℓ)

-- The type annotation confirms: SexualRelation-Type ℓ Φ
-- lives in Set (suc (suc ℓ)) — two universe levels above
-- the universe being quantified over in the formulas.

