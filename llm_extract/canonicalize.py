"""
canonicalize.py — maps the many surface forms a catalyst/support name can
take in the literature ("NiMo", "Ni-Mo", "Ni/Mo", "Nickel Molybdenum") to one
canonical identifier, while ALWAYS preserving the original extracted text
alongside it. This is applied as a post-extraction normalization pass (in
normalize.py), not baked into the LLM prompt -- canonicalization is a
deterministic string-matching problem, not something that benefits from a
model's variance, and keeping it as plain Python means it's instantly
auditable and trivially extensible by editing the tables below (no prompt
engineering, no re-running extraction).

Design: every canonicalize_* function returns (original_text, canonical) so
callers always keep both. A canonical value is never substituted in place of
the original -- it's added as a sibling column (e.g. active_metal_canonical)
so nothing the LLM actually extracted is lost or silently overwritten.
"""

from __future__ import annotations

import re
from typing import Optional

# --- Bimetallic / multi-metal catalyst name normalization -------------------
# Strategy: extract element symbols in the order they appear, normalize
# separators (hyphen, slash, "and", whitespace) and casing variants
# ("nickel" -> "Ni"), then rejoin with a single canonical separator. This
# handles "NiMo", "Ni-Mo", "Ni/Mo", "Ni and Mo", "Nickel Molybdenum",
# "Ni,Mo" all collapsing to "Ni-Mo".

_ELEMENT_NAME_TO_SYMBOL = {
    "nickel": "Ni", "molybdenum": "Mo", "cobalt": "Co", "platinum": "Pt",
    "palladium": "Pd", "ruthenium": "Ru", "rhodium": "Rh", "iron": "Fe",
    "copper": "Cu", "tungsten": "W", "zinc": "Zn", "chromium": "Cr",
    "vanadium": "V", "titanium": "Ti", "manganese": "Mn", "gold": "Au",
    "silver": "Ag", "iridium": "Ir", "niobium": "Nb", "tin": "Sn",
    "cerium": "Ce", "zirconium": "Zr", "aluminum": "Al", "aluminium": "Al",
}
_KNOWN_SYMBOLS = set(_ELEMENT_NAME_TO_SYMBOL.values())

_SPLIT_PATTERN = re.compile(r"[\s\-/,]+|(?:\s+and\s+)", re.IGNORECASE)


def canonicalize_metal(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """('NiMo', 'Ni-Mo'), ('Ni-Mo', 'Ni-Mo'), ('Nickel Molybdenum', 'Ni-Mo'),
    ('Pt', 'Pt'). Returns (original, canonical); canonical is None if raw is
    None/empty or no recognizable element token was found (left as None
    rather than guessed, consistent with the rest of this package's
    null-over-guess policy)."""
    if not raw or not raw.strip():
        return raw, None

    tokens = [t for t in _SPLIT_PATTERN.split(raw.strip()) if t]
    symbols: list[str] = []
    for tok in tokens:
        lower = tok.lower()
        if lower in _ELEMENT_NAME_TO_SYMBOL:
            symbols.append(_ELEMENT_NAME_TO_SYMBOL[lower])
            continue
        # Already-a-symbol case, e.g. "NiMo" with no separator at all, or
        # "Ni0.5Cu0.5" / "Pt1.2" with an inline stoichiometric subscript or
        # loading indicator: split on capital-letter boundaries into 1-2
        # char element symbols, after stripping digits/decimal points/signs
        # so a subscript doesn't count as "leftover" coverage. Require the
        # matched chunks to cover the WHOLE alphabetic residue (no leftover
        # letters) before accepting any of them -- otherwise a plain
        # English word that happens to contain an element substring (e.g.
        # "Compound" contains "Co") would be misread as a catalyst formula.
        # Stripping numerals only affects the coverage check, never the
        # symbols extracted, so a genuinely unrecognizable token (e.g.
        # "XyzUnknownCompound123") is still correctly rejected: its
        # alphabetic residue doesn't decompose into known symbols either.
        alpha_only = re.sub(r"[0-9.+\-:]", "", tok)
        sub_tokens = re.findall(r"[A-Z][a-z]?", tok)
        if sub_tokens and alpha_only and \
                sum(len(s) for s in sub_tokens) == len(alpha_only) and \
                all(s in _KNOWN_SYMBOLS for s in sub_tokens):
            symbols.extend(sub_tokens)

    if not symbols:
        return raw, None

    # De-duplicate while preserving first-seen order (e.g. "Ni-Ni-Mo" -> Ni-Mo)
    seen: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.append(s)
    return raw, "-".join(seen)


# --- Zeolite / support framework normalization -------------------------------
# "HZSM-5", "ZSM-5", "ZSM5", "H-ZSM-5" all refer to the same MFI-topology
# zeolite framework (the H- prefix denotes the protonated/acid form, which
# is preserved as a separate flag rather than discarded, since it's
# chemically meaningful -- a protonated vs. sodium-form zeolite are
# different catalysts).
#
# This pattern is a general letter-prefix + numeric-suffix rule -- it is NOT
# a hardcoded whitelist, so it already covers any family written the same
# way, e.g. SBA-15, MCM-41, KIT-6, SAPO-34, MFI-11, not just ZSM-n. Verified
# empirically (see test_canonicalize.py) rather than assumed.
_ZEOLITE_PATTERN = re.compile(
    r"^(H[-\s]?)?([A-Z]+)[-\s]?(\d+)$"
)

# Framework/topology codes that are conventionally written WITHOUT a
# trailing number (unlike ZSM-5, SBA-15, etc.) -- "Beta", "MOR", "USY", and
# similar. _ZEOLITE_PATTERN above requires a digit suffix and so never
# matches these; without this second table, "H-Beta" / "HBeta" / "H Beta"
# (three spacings for the same protonated Beta zeolite) would each produce
# a different, un-normalized string and fail to be recognised as the same
# support. This is intentionally a short, conservative whitelist (not a
# regex over any capitalized word) to avoid false-positive canonicalization
# of an ordinary catalyst-name token that happens to be all-letters -- add
# to this set as new topology codes are encountered in practice.
_ZEOLITE_TOPOLOGY_CODES_NO_SUFFIX = {
    "beta", "mor", "fau", "mfi", "cha", "fer", "mww", "lta", "emt",
    "usy", "bea", "mel", "mtw", "ton",
}


def canonicalize_support(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """('HZSM-5', 'ZSM-5'), ('ZSM5', 'ZSM-5'), ('H-ZSM-5', 'ZSM-5'),
    ('H-Beta', 'H-Beta'), ('HBeta', 'H-Beta'), ('H Beta', 'H-Beta'),
    ('gamma-Al2O3', 'gamma-Al2O3' -- unchanged, not a zeolite pattern).
    Returns (original, canonical). Falls back to a lightly-normalized
    version of the original (stripped + collapsed whitespace) if no known
    pattern matches, rather than None -- unlike metals, most support
    strings ARE already usable as-is; this function only rewrites the
    small set of zeolite-style names with known separator inconsistency.
    """
    if not raw or not raw.strip():
        return raw, None

    candidate = raw.strip()
    no_space = candidate.replace(" ", "")

    m = _ZEOLITE_PATTERN.match(no_space)
    if m:
        family, number = m.group(2), m.group(3)
        return raw, f"{family}-{number}"

    # Second pass: H-prefix + known no-suffix topology code, any of the
    # three common spacings ("H-Beta", "HBeta", "H Beta").
    no_hyphen_no_space = no_space.replace("-", "")
    lower = no_hyphen_no_space.lower()
    if lower.startswith("h") and lower[1:] in _ZEOLITE_TOPOLOGY_CODES_NO_SUFFIX:
        return raw, f"H-{lower[1:].capitalize()}"
    if lower in _ZEOLITE_TOPOLOGY_CODES_NO_SUFFIX:
        return raw, lower.capitalize()

    return raw, re.sub(r"\s+", " ", candidate)


def apply_canonicalization(record: dict, metal_field: str = "active_metal",
                            support_field: str = "support_material") -> dict:
    """Add `<metal_field>_canonical` and `<support_field>_canonical` keys to
    a flat dict (one experiment row, or one flat paper record) in place,
    without touching the original fields. Silently no-ops for any field
    name not present in the record, so it's safe to call on domains that
    don't have these exact field names."""
    out = dict(record)
    if metal_field in out:
        _, canonical = canonicalize_metal(out.get(metal_field))
        out[f"{metal_field}_canonical"] = canonical
    if support_field in out:
        _, canonical = canonicalize_support(out.get(support_field))
        out[f"{support_field}_canonical"] = canonical
    return out
