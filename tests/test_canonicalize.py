from llm_extract.canonicalize import canonicalize_metal, canonicalize_support, apply_canonicalization


def test_canonicalize_metal_variants_collapse_to_same_canonical():
    variants = ["NiMo", "Ni-Mo", "Ni/Mo", "Nickel Molybdenum", "Ni and Mo", "Ni,Mo"]
    canonicals = {canonicalize_metal(v)[1] for v in variants}
    assert canonicals == {"Ni-Mo"}


def test_canonicalize_metal_preserves_original_text():
    orig, canon = canonicalize_metal("Ni/Mo")
    assert orig == "Ni/Mo"          # untouched
    assert canon == "Ni-Mo"         # normalized sibling


def test_canonicalize_metal_single_element():
    assert canonicalize_metal("Pt")[1] == "Pt"
    assert canonicalize_metal("platinum")[1] == "Pt"


def test_canonicalize_metal_trimetallic():
    assert canonicalize_metal("NiCoMo")[1] == "Ni-Co-Mo"


def test_canonicalize_metal_none_and_empty():
    assert canonicalize_metal(None) == (None, None)
    assert canonicalize_metal("")[1] is None
    assert canonicalize_metal("   ")[1] is None


def test_canonicalize_metal_stoichiometric_subscripts():
    # Bimetallic loadings reported with inline fractional/decimal
    # subscripts (e.g. from a paper's formula notation) must not be
    # dropped just because digits break the naive length-coverage check.
    assert canonicalize_metal("Ni0.5Cu0.5")[1] == "Ni-Cu"
    assert canonicalize_metal("Pt1.2")[1] == "Pt"
    assert canonicalize_metal("Ni0.5Mo0.5W0.2")[1] == "Ni-Mo-W"


def test_canonicalize_metal_unrecognized_returns_none_not_guess():
    orig, canon = canonicalize_metal("XyzUnknownCompound123")
    assert orig == "XyzUnknownCompound123"
    assert canon is None  # must not fabricate a guess


def test_canonicalize_support_zeolite_variants_collapse():
    variants = ["HZSM-5", "ZSM-5", "ZSM5", "H-ZSM-5"]
    canonicals = {canonicalize_support(v)[1] for v in variants}
    assert canonicals == {"ZSM-5"}


def test_canonicalize_support_non_zeolite_passes_through():
    orig, canon = canonicalize_support("gamma-Al2O3")
    assert canon == "gamma-Al2O3"


def test_canonicalize_support_none():
    assert canonicalize_support(None) == (None, None)


def test_apply_canonicalization_adds_sibling_columns_without_mutating_originals():
    record = {"active_metal": "NiMo", "support_material": "HZSM-5", "conversion_pct": 95.0}
    out = apply_canonicalization(record)
    assert out["active_metal"] == "NiMo"            # original untouched
    assert out["active_metal_canonical"] == "Ni-Mo"  # canonical added
    assert out["support_material"] == "HZSM-5"
    assert out["support_material_canonical"] == "ZSM-5"
    assert out["conversion_pct"] == 95.0             # other fields untouched


def test_apply_canonicalization_noop_on_missing_fields():
    record = {"some_other_field": "x"}
    out = apply_canonicalization(record)
    assert out == {"some_other_field": "x"}  # no crash, no spurious keys
