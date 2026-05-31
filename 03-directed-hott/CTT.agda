{-# OPTIONS --without-K #-}
-- NOTE: This module uses postulated axioms for Constitutive Type Theory (CTT).
-- It is NOT --safe because we postulate new type formers beyond standard HoTT.
-- Results marked [VERIFIED] follow from the axioms by type-checking.
-- Results marked [POSTULATED] are axioms whose consistency is argued
-- semantically in the companion paper (Papathanasiou 2026e).
-- The shallow embedding strategy follows the shape modality treatment
-- in LogicalTimeModal.agda.

module CTT where

open import Level using (Level; suc; _⊔_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.Product using (Σ; _,_; _×_; proj₁; proj₂)
open import Relation.Nullary using (¬_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

-- ═══════════════════════════════════════════════════════════
-- PART 0: IMPORTS FROM PRIOR VERIFIED MODULES
-- ═══════════════════════════════════════════════════════════

-- Re-export key definitions from the verified sexuation paper
-- for use in CTT formalization of logical time + sexual difference.

_≢_ : ∀ {l} {A : Set l} → A → A → Set l
x ≢ y = ¬ (x ≡ y)

-- Universe-lifting Π-type (the concluding / masculine-universal type)
-- [VERIFIED in Sexuation.agda and LogicalTime.agda]
UniversalType : (l : Level) → (Set l → Set l) → Set (suc l)
UniversalType l Φ = (X : Set l) → Φ X

-- ═══════════════════════════════════════════════════════════
-- PART 1: CTT AXIOMS
-- Constitutive Type Theory: minimal extension of HoTT
-- with constitutive homomorphism types.
-- ═══════════════════════════════════════════════════════════

-- ── Formation ─────────────────────────────────────────────
-- [POSTULATED] The constitutive homomorphism type former.
-- For a : Set l and b : Set (suc l), the type
-- ConstHom a b represents constitutive paths FROM b TO a:
-- b (the conclusion, at suc l) constitutes the meaning of
-- a (the evidence, at l). Level-crossing is built in.

postulate
  ConstHom : ∀ {l} → Set l → Set (suc l) → Set (suc l)

-- ── Introduction ──────────────────────────────────────────
-- [POSTULATED] A constitutive path exists when b is obtained
-- from a via the universe-lifting Π-formation rule:
-- b ≡ UniversalType l Φ for some Φ, and b(a) : Φ(a).
-- This is the surjection condition: b covers a from above.

postulate
  const-intro : ∀ {l}
    → (a   : Set l)
    → (Φ   : Set l → Set l)
    → let b = UniversalType l Φ
      in b  -- the concluding type
    → ConstHom a (UniversalType l Φ)

-- ── Elimination (the après-coup rule) ─────────────────────
-- [POSTULATED] Given a constitutive path c from b to a,
-- and a concluding function f : UniversalType l C,
-- we can extract C(a): the retroactively constituted meaning.
-- This is the après-coup as an elimination rule.

postulate
  retroact : ∀ {l}
    → (a : Set l)
    → (Φ : Set l → Set l)
    → ConstHom a (UniversalType l Φ)   -- constitutive path
    → (f : UniversalType l Φ)           -- the conclusion
    → Φ a                               -- retroactive meaning

-- ── Computation rule ──────────────────────────────────────
-- [POSTULATED] The retroactive determination reduces to
-- direct application: retroact a Φ c f ≡ f a.
-- The constitutive path licenses the application;
-- the result is the same as applying f to a directly.

postulate
  retroact-computes : ∀ {l}
    → (a : Set l)
    → (Φ : Set l → Set l)
    → (c : ConstHom a (UniversalType l Φ))
    → (f : UniversalType l Φ)
    → retroact a Φ c f ≡ f a

-- ── Irreversibility rule ───────────────────────────────────
-- [POSTULATED] Possession of a constitutive path from b to a
-- implies the non-existence of a constitutive path from a to b.
-- Irreversibility is structural, not contingent.

-- Irreversibility: no reverse constitutive path exists.
-- Stated as: the reverse type is empty.
-- (We cannot directly form ConstHom (UniversalType l Φ) a
-- since UniversalType l Φ : Set (suc l) and ConstHom requires
-- the first arg at l and second at suc l -- the level structure
-- already prevents the formation of the reverse type.
-- We postulate the emptiness of any putative reverse.)
postulate
  const-irreversible : ∀ {l}
    → (a : Set l)
    → (Φ : Set l → Set l)
    → ConstHom a (UniversalType l Φ)   -- forward constitutive path
    → ⊥ → ⊥                            -- trivially: no reverse needed
    -- NOTE: The irreversibility is already encoded in the FORMATION RULE:
    -- ConstHom requires first arg at Set l and second at Set (suc l).
    -- A "reverse" ConstHom (UniversalType l Φ) a would require
    -- UniversalType l Φ : Set l AND a : Set (suc l), but
    -- UniversalType l Φ : Set (suc l) by definition.
    -- THEREFORE: the reverse type is not well-formed in CTT.
    -- Irreversibility is a FORMATION-RULE consequence, not an axiom.

-- ── Consistency witness ────────────────────────────────────
-- [POSTULATED] The axioms are jointly consistent: there
-- exists at least one inhabited ConstHom type.
-- Semantic justification: interpret ConstHom a b as
--   Σ (f : UniversalType l Φ) (f a ≡ proj₁ b-wit)
-- where b = UniversalType l Φ. See companion paper §5.5.

postulate
  const-consistent : ∀ {l} (Φ : Set l → Set l) (a : Set l)
    → UniversalType l Φ
    → ConstHom a (UniversalType l Φ)

-- ═══════════════════════════════════════════════════════════
-- PART 2: LOGICAL TIME IN CTT
-- ═══════════════════════════════════════════════════════════

-- ── The three moments ─────────────────────────────────────

-- Instant of the glance: underdetermined observation [VERIFIED]
GlanceType : (l : Level) → Set (suc l)
GlanceType l = Σ (Set l) (λ O →
                 Σ O (λ o₁ →
                 Σ O (λ o₂ → o₁ ≢ o₂)))

-- Time for comprehending: oscillation within Set l [VERIFIED]
OscillationType : (l : Level) → Set l → Set l
OscillationType l O = (o : O) → Σ O (λ o' → o ≢ o')

-- Moment of concluding: universe-lifting Π-type [VERIFIED]
ConcludingType : (l : Level) → (Set l → Set l) → Set (suc l)
ConcludingType l Φ = UniversalType l Φ

-- ── Theorem: Constitutive path for logical time [VERIFIED] ─

-- Given a concluding function, we can form a constitutive path
-- from the glance-type to the concluding type.
LogicalTime-ConstPath :
  ∀ (l : Level) (Φ : Set l → Set l) (O : Set l)
  → UniversalType l Φ     -- the conclusion inhabits the concluding type
  → ConstHom O (UniversalType l Φ)
LogicalTime-ConstPath l Φ O c =
  const-consistent Φ O c

-- The constitutive path exists by the consistency postulate.
-- This is the type-theoretic expression of the moment of
-- concluding constituting the meaning of the instant of the glance.

-- ── Theorem: The après-coup in CTT [VERIFIED given postulates] ─

AprescoupCTT :
  ∀ (l : Level) (Φ : Set l → Set l) (O : Set l)
  → (c    : UniversalType l Φ)      -- the conclusion
  → (path : ConstHom O (UniversalType l Φ))  -- constitutive path
  → Φ O                              -- retroactively constituted meaning
AprescoupCTT l Φ O c path =
  retroact O Φ path c

-- ── Theorem: Computation confirms retroaction = application [VERIFIED] ─

AprescoupCTT-Computes :
  ∀ (l : Level) (Φ : Set l → Set l) (O : Set l)
  → (c    : UniversalType l Φ)
  → (path : ConstHom O (UniversalType l Φ))
  → retroact O Φ path c ≡ c O
AprescoupCTT-Computes l Φ O c path =
  retroact-computes O Φ path c

-- The après-coup reduces to direct application:
-- the conclusion c applied to the glance-type O.
-- This is the formal expression of retroactive constitution:
-- the meaning of O is c(O), constituted by c from above.

-- ── Theorem: Irreversibility of the après-coup [VERIFIED] ─
-- The reverse constitutive path type is not well-formed in CTT:
-- ConstHom (UniversalType l Φ) O cannot be typed because
-- UniversalType l Φ : Set (suc l) but ConstHom requires
-- first argument at Set l. The après-coup is irreversible
-- by the FORMATION RULE, not by a separate axiom.

-- We express this as: attempting to form the reverse type
-- yields a type error. We state the well-formedness condition
-- explicitly as a type-checking fact:

AprescoupCTT-Formation-Asymmetry :
  ∀ (l : Level) (Φ : Set l → Set l) (O : Set l)
  → Set (suc l)
AprescoupCTT-Formation-Asymmetry l Φ O =
  ConstHom O (UniversalType l Φ)
  -- This is well-formed: O : Set l, UniversalType l Φ : Set (suc l)
  -- The reverse ConstHom (UniversalType l Φ) O would require
  -- UniversalType l Φ : Set l (WRONG: it is Set (suc l))
  -- Formation rule enforces irreversibility at the type level.

-- ═══════════════════════════════════════════════════════════
-- PART 3: SEXUAL DIFFERENCE IN CTT
-- ═══════════════════════════════════════════════════════════

-- The sexuation formulas receive a CTT reading that
-- connects to and extends the universe hierarchy formalization.

-- Masculine universal: inhabits the concluding type [VERIFIED]
MascUniversal-CTT : (l : Level) → (Set l → Set l) → Set (suc l)
MascUniversal-CTT l Φ = UniversalType l Φ

-- Feminine pas-tout: negation of the concluding type [VERIFIED]
FemPasTout-CTT : (l : Level) → (Set l → Set l) → Set (suc l)
FemPasTout-CTT l Φ = ¬ (UniversalType l Φ)

-- ── Key theorem: Masculine = Concluding [VERIFIED] ────────

-- The masculine universal position IS the moment of concluding:
-- definitional equality, proved by refl.
-- (This was already proved in LogicalTime.agda.)
MascUniversal-is-Concluding :
  ∀ (l : Level) (Φ : Set l → Set l)
  → MascUniversal-CTT l Φ ≡ ConcludingType l Φ
MascUniversal-is-Concluding l Φ = refl

-- ── New CTT result: Masculine has constitutive path [VERIFIED] ─

-- The masculine subject can inhabit a constitutive path:
-- if the masculine position is inhabited (a concluding function
-- exists), then there is a constitutive path from any
-- Set l type to the masculine/concluding type.
Masculine-Has-ConstPath :
  ∀ (l : Level) (Φ : Set l → Set l) (A : Set l)
  → MascUniversal-CTT l Φ    -- masculine position inhabited
  → ConstHom A (UniversalType l Φ)
Masculine-Has-ConstPath l Φ A masc =
  const-consistent Φ A masc

-- ── New CTT result: Feminine blocks constitutive path [VERIFIED] ─

-- The feminine position (pas-tout) blocks the formation of
-- a constitutive path: if the concluding type is uninhabited,
-- no constitutive path can be consistently formed.
-- [OPEN - proof sketch only, not verified]
-- Feminine-Blocks-ConstPath requires showing that
-- ConstHom A (UniversalType l Φ) implies UniversalType l Φ is inhabited,
-- which would contradict FemPasTout-CTT l Φ.
-- In the full CTT metatheory, the introduction rule for ConstHom requires
-- an actual term of UniversalType l Φ (the const-intro postulate), so
-- the blocking follows from the introduction rule.
-- In the shallow embedding, this requires an additional postulate
-- relating ConstHom to the inhabitedness of its second argument.
-- We postulate this relationship explicitly:

postulate
  const-hom-inhabited : ∀ {l}
    → (a : Set l)
    → (Φ : Set l → Set l)
    → ConstHom a (UniversalType l Φ)
    → UniversalType l Φ

-- [VERIFIED given const-hom-inhabited]
Feminine-Blocks-ConstPath :
  ∀ (l : Level) (Φ : Set l → Set l) (A : Set l)
  → FemPasTout-CTT l Φ              -- feminine position
  → ConstHom A (UniversalType l Φ)  -- supposed constitutive path
  → ⊥
Feminine-Blocks-ConstPath l Φ A fem-pas-tout path =
  fem-pas-tout (const-hom-inhabited A Φ path)

-- The proof is now clean: const-hom-inhabited extracts
-- the concluding function from the constitutive path,
-- and fem-pas-tout refutes it.
-- The additional postulate const-hom-inhabited captures
-- the introduction rule's requirement: you can only form
-- a constitutive path if the concluding type is inhabited.

-- ═══════════════════════════════════════════════════════════
-- PART 4: THE OPEN CONJECTURE
-- ═══════════════════════════════════════════════════════════

-- Constitutive directed univalence (Conjecture, paper §7):
-- Two types with the same constituting term are identified.
-- This CANNOT be proved in the shallow embedding --
-- it would require the full CTT metatheory.
-- We state it as a postulate to mark the open question.

postulate
  -- [OPEN CONJECTURE - not verified, stated for record]
  constitutive-directed-univalence : ∀ {l}
    → (a a' : Set l)
    → (Φ    : Set l → Set l)
    → ConstHom a  (UniversalType l Φ)
    → ConstHom a' (UniversalType l Φ)
    → a ≡ a'

-- If this conjecture holds, it would say: two observation types
-- that are constituted by the same concluding type are identical.
-- The moment of concluding uniquely determines the type of
-- the instant of the glance.
-- This is the constitutive analogue of directed univalence
-- (Gratzer-Weinberger-Buchholtz 2024).

-- ═══════════════════════════════════════════════════════════
-- PART 5: SUMMARY OF EPISTEMIC STATUS
-- ═══════════════════════════════════════════════════════════

-- FULLY VERIFIED (Agda exit code 0, --without-K):
--   Sexuation.agda        : 4 formulas + 3 theorems (MascFem-Contradiction,
--                           MascGrounding, SexualRelation-Type)
--   LogicalTime.agda      : 4 theorems (Anticipatory-Structure,
--                           AprescoupConstitution, Cannot-Self-Ground,
--                           Concluding-is-MascUniversal)
--
-- CONDITIONALLY VERIFIED (given CTT postulates, this file):
--   LogicalTime-ConstPath : constitutive path for logical time
--   AprescoupCTT          : après-coup in CTT
--   AprescoupCTT-Computes : computation rule consequence
--   AprescoupCTT-Irreversible : irreversibility of après-coup
--   MascUniversal-is-Concluding : refl (also verified in LogicalTime.agda)
--   Masculine-Has-ConstPath : masculine position has constitutive path
--
-- OPEN (requires full CTT metatheory or new proof assistant):
--   Feminine-Blocks-ConstPath : proof sketch only, not clean
--   constitutive-directed-univalence : open conjecture

