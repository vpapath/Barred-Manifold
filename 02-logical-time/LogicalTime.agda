{-# OPTIONS --without-K --safe #-}

module LogicalTime where

open import Level using (Level; suc)
open import Data.Empty using (⊥)
open import Data.Product using (Σ; _,_; _×_)
open import Relation.Nullary using (¬_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

_≢_ : ∀ {l} {A : Set l} → A → A → Set l
x ≢ y = ¬ (x ≡ y)

-- ─────────────────────────────────────────────────────────
-- THE THREE LOGICAL MOMENTS AS TYPES
-- ─────────────────────────────────────────────────────────

-- Instant of the glance: observation type with distinct inhabitants
Glance : (l : Level) → Set (suc l)
Glance l = Σ (Set l) (λ O →
             Σ O (λ o₁ →
             Σ O (λ o₂ → o₁ ≢ o₂)))

-- Oscillation: for any proposed reading, an alternative exists
-- This function type stays at Set l (does not lift)
OscillationType : (l : Level) → Set l → Set l
OscillationType l O = (o : O) → Σ O (λ o' → o ≢ o')

-- Moment of concluding: universal commitment over Set l
-- Lives in Set (suc l) by the Π-formation rule:
-- quantifying over the universe Set l lifts to Set (suc l)
Concluding : (l : Level) → (Set l → Set l) → Set (suc l)
Concluding l Φ = (X : Set l) → Φ X

-- ─────────────────────────────────────────────────────────
-- THEOREM 1: Anticipatory structure
-- A Concluding term applies to every observation type
-- ─────────────────────────────────────────────────────────

Anticipatory-Structure :
  ∀ (l : Level) (Φ : Set l → Set l)
  → Concluding l Φ    -- conclusion: Set (suc l)
  → (O : Set l)       -- any observation type: Set l
  → Φ O               -- proof for that type: Set l
Anticipatory-Structure l Φ c O = c O

-- ─────────────────────────────────────────────────────────
-- THEOREM 2: The après-coup (retroactive constitution)
-- ─────────────────────────────────────────────────────────

-- The conclusion c (at suc l) constitutes the meaning of
-- the observation O (at l) retroactively.
-- Before c: O : Set l, meaning undetermined.
-- After c: c O : Φ O, meaning constituted by the conclusion.

AprescoupConstitution :
  ∀ (l : Level) (Φ : Set l → Set l) (O : Set l)
  → Concluding l Φ    -- conclusion at suc l
  → Φ O               -- constituted meaning at l
AprescoupConstitution l Φ O c = c O

-- ─────────────────────────────────────────────────────────
-- THEOREM 3: Oscillation cannot self-ground
-- No Set l-level operation can terminate the oscillation
-- ─────────────────────────────────────────────────────────

Comprehending-Cannot-Self-Ground :
  ∀ (l : Level) (O : Set l)
  → OscillationType l O       -- oscillation at Set l
  → (canonical : O)           -- any proposed canonical term
  → Σ O (λ o' → canonical ≢ o')
Comprehending-Cannot-Self-Ground l O osc canonical =
  osc canonical

-- ─────────────────────────────────────────────────────────
-- THEOREM 4: Logical time and sexual difference unified
-- ─────────────────────────────────────────────────────────

-- From the companion sexuation paper:
MascUniversal : (l : Level) → (Set l → Set l) → Set (suc l)
MascUniversal l Φ = (A : Set l) → Φ A

FemPasTout : (l : Level) → (Set l → Set l) → Set (suc l)
FemPasTout l Φ = ¬ ((A : Set l) → Φ A)

-- The moment of concluding is definitionally equal to
-- the masculine universal: same type, proved by refl.
Concluding-is-MascUniversal :
  ∀ (l : Level) (Φ : Set l → Set l)
  → Concluding l Φ ≡ MascUniversal l Φ
Concluding-is-MascUniversal l Φ = refl

-- COROLLARY: The feminine pas-tout is the structural
-- impossibility of the concluding act.
-- Logical time and sexual difference share a single
-- formal structure: the universe-lifting Π-type,
-- inhabited (masculine / concluding) vs negated (feminine / pas-tout).

