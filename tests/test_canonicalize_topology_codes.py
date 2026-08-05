"""
test_canonicalize_topology_codes.py — targeted tests from a code review
that made two claims about canonicalize_support():

  1. (CONFIRMED, fixed here) Protonated-form zeolite topology codes written
     without a numeric suffix -- "H-Beta", "HBeta", "H Beta" -- were not
     unified, because the original _ZEOLITE_PATTERN requires a trailing
     digit (by design, for ZSM-5/SBA-15/MCM-41-style names). This file
     locks in the fix.

  2. (REJECTED after verification) The review separately claimed SBA-15 and
     MCM-41 "will fail the match and fall back to plain spacing reduction".
     This is empirically false: both already match _ZEOLITE_PATTERN, since
     it is a general letter-prefix + digit-suffix rule, not a ZSM-specific
     whitelist. Tests below confirm this was already correct so the claim
     is not silently treated as valid.
"""

from llm_extract.canonicalize import canonicalize_support


# --- Confirmed gap, now fixed: H-prefix topology codes without a suffix ----

def test_h_prefix_beta_variants_all_collapse_to_same_canonical():
    variants = ["H-Beta", "HBeta", "H Beta"]
    canonicals = {canonicalize_support(v)[1] for v in variants}
    assert canonicals == {"H-Beta"}


def test_bare_topology_code_without_h_prefix_is_normalized_consistently():
    orig, canon = canonicalize_support("beta")
    assert canon == "Beta"
    orig, canon = canonicalize_support("BETA")
    assert canon == "Beta"


def test_h_prefix_mor_and_usy_variants():
    for code, expected in [("HMOR", "H-Mor"), ("H-MOR", "H-Mor"), ("H MOR", "H-Mor"),
                            ("HUSY", "H-Usy"), ("H-USY", "H-Usy")]:
        _, canon = canonicalize_support(code)
        assert canon == expected, f"{code} -> {canon}, expected {expected}"


def test_original_text_preserved_alongside_canonical_form():
    orig, canon = canonicalize_support("HBeta")
    assert orig == "HBeta"       # verbatim author text untouched
    assert canon == "H-Beta"     # canonical sibling


def test_unrecognized_all_caps_word_is_not_falsely_treated_as_topology_code():
    """Guard against over-eager matching: an arbitrary all-caps acronym
    that happens to not be in the small topology whitelist must not be
    canonicalized as if it were one -- it should fall through to the
    plain whitespace-normalization fallback, unchanged."""
    orig, canon = canonicalize_support("XYZ")
    assert canon == "XYZ"  # unchanged fallback, not fabricated


# --- Explicit confirmation that SBA-15 / MCM-41 already worked -------------
# (the review's specific claim that these "fail" does not hold up; these
# tests exist so that claim cannot silently regress into becoming true).

def test_sba_15_matches_the_general_letter_number_pattern():
    orig, canon = canonicalize_support("SBA-15")
    assert canon == "SBA-15"
    orig, canon = canonicalize_support("SBA15")
    assert canon == "SBA-15"  # different spacing, same canonical form


def test_mcm_41_matches_the_general_letter_number_pattern():
    orig, canon = canonicalize_support("MCM-41")
    assert canon == "MCM-41"
    orig, canon = canonicalize_support("MCM 41")
    assert canon == "MCM-41"


def test_kit_and_sapo_families_also_match_the_same_general_pattern():
    """Not cited in the review, but confirms the pattern really is general
    (any letter-prefix + digit-suffix family), not a hidden per-family
    whitelist that happens to include SBA/MCM by coincidence."""
    assert canonicalize_support("KIT-6")[1] == "KIT-6"
    assert canonicalize_support("SAPO-34")[1] == "SAPO-34"
    assert canonicalize_support("SAPO34")[1] == "SAPO-34"


# --- Confirm genuinely non-zeolite supports remain correctly untouched -----

def test_non_zeolite_oxide_supports_remain_untouched():
    """Al2O3, UiO-66 (a MOF, not a zeolite), and similar have no
    canonicalization to apply -- they must pass through as lightly
    whitespace-normalized text, same as before this fix."""
    assert canonicalize_support("gamma-Al2O3")[1] == "gamma-Al2O3"
    assert canonicalize_support("UiO-66")[1] == "UiO-66"
    assert canonicalize_support("activated carbon")[1] == "activated carbon"
