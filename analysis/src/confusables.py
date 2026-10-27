"""Fold visually confusable Unicode in token metadata to a stable ASCII skeleton.

Why this exists
---------------
Phishing campaigns are clustered by the registrable domain carried in a token contract's
`name()`/`symbol()` strings, and adversaries evade that clustering by substituting codepoints that
render identically to the ASCII ones. Measured on a 2,887-string sample of the live corpus: 218
strings are non-ASCII, **83 contain Cyrillic**, 163 are emoji-decorated, and 3 carry zero-width
injection (U+200B/U+200D/U+206F). Two DISTINCT Optimism contracts read "www.fomox.club" and
"www.fomox.сlub" -- the second spells `club` with Cyrillic U+0441. Base carries "UЅDС"
(Cyrillic Ѕ and С) and "U⁠SD‍C" (invisible U+2060 and U+200D injected mid-word).

If normalisation does not run BEFORE domain extraction, one campaign fragments into many and the
campaign counts are wrong. This module is therefore load-bearing for correctness, not cosmetic.

`skeleton()` is idempotent (Proposition 1 in the paper). That is what licenses the paper's claim that
skipping normalisation can only FRAGMENT a campaign and never MERGE two distinct ones: the map is a
function from strings to a fixed point, so re-normalising a key cannot move it.

Prior art -- the technique is not our contribution
--------------------------------------------------
Homoglyph folding of token metadata is established practice and we reimplement and MEASURE it rather
than claim it. Forta's open-source spam detector (kovart/forta-spam-detector, `normalizer.ts`) folds
Cyrillic homoglyphs and strips invisible characters before link extraction, and Ye et al. (WWW 2024,
"Interface Illusions") published a homograph mapping table for token metadata. Our contribution is
the measurement on rollup corpora and the idempotence argument above.

Inputs
------
Single Python `str` values (a contract `name()` or `symbol()` as decoded by `token_meta`). All tables
are vendored in this file: no network, no filesystem, no global state, no randomness.

Outputs
-------
Normalised `str` values. `skeleton()` returns the clustering key; the intermediate primitives are
exported so a caller can compose a different pipeline (e.g. keep emoji, fold only invisibles) and so
each stage is independently testable.

Failure modes
-------------
- A non-`str` input RAISES `TypeError`. It is never coerced. The realistic bug is a pandas missing
  value: `str(float('nan')) == 'nan'`, which would enter the corpus as a token literally named "nan"
  and cluster with every other unresolved contract. That silent path is exactly the fabrication class
  this project's guards exist to stop, so it fails loudly instead.
- `skeleton()` legitimately returns `""` for an empty string, for a string of only invisible
  characters, and for a string of only emoji. `""` means "no signal survived normalisation" and is NOT
  a cluster key: callers must test for it and exclude, or every such contract merges into one campaign.
- A malformed `CONFUSABLE_MAP` edit (multi-character key, ASCII key, non-ASCII value) raises
  `ValueError` at import time rather than silently breaking idempotence.
- Known and accepted false positive: the bare-word dot form folds " DOT " between two alphanumerics,
  so "Wrapped DOT token" becomes "wrapped.token" -- `DOT` is a real ticker. Accepted deliberately:
  downstream domain extraction validates the TLD, so `.token` is discarded, whereas refusing the
  uppercase form would silently lose real "WWW DOT X DOT COM" campaign members. Callers that must not
  pay this cost should compose the primitives themselves and skip `deobfuscate`.

Complexity: O(len(s)) time and space per `skeleton` call -- a constant number of linear passes (the
confusable fold runs twice and NFKC twice; see `skeleton`), one dict lookup and one
`unicodedata.category` call per codepoint, and regexes with no nested quantifiers. `deobfuscate`
iterates to a fixed point, which is O(len(s)) rounds in the worst case and one round on every real
string, since each round must strictly shorten the input. Tables are built once at import (measured
0.003 s): 175 confusable entries, 100 invisible codepoints, 99 abbreviation symbols derived by
scanning ~5,000 candidate codepoints.
"""
from __future__ import annotations

import re
import unicodedata

_ASCII_MAX = 0x7F

# Zero-width, joiner and bidi/format controls. U+206A-U+206F (deprecated format controls) are in the
# corpus, so the whole block is covered rather than just the observed U+206F. Variation selectors
# U+FE00-U+FE0F are here and not in fold_emoji because they are invisible on their own and must go
# even when the caller keeps emoji.
_FORMAT_CODEPOINTS = frozenset(
    {
        "­",  # SOFT HYPHEN
        "᠎",  # MONGOLIAN VOWEL SEPARATOR
        "​",  # ZERO WIDTH SPACE
        "‌",  # ZERO WIDTH NON-JOINER
        "‍",  # ZERO WIDTH JOINER          (observed on Base inside "USDC")
        "‎",  # LEFT-TO-RIGHT MARK
        "‏",  # RIGHT-TO-LEFT MARK
        "‪",  # LEFT-TO-RIGHT EMBEDDING
        "‫",  # RIGHT-TO-LEFT EMBEDDING
        "‬",  # POP DIRECTIONAL FORMATTING
        "‭",  # LEFT-TO-RIGHT OVERRIDE
        "‮",  # RIGHT-TO-LEFT OVERRIDE
        "⁠",  # WORD JOINER                (observed on Base inside "USDC")
        "⁡",  # FUNCTION APPLICATION
        "⁢",  # INVISIBLE TIMES
        "⁣",  # INVISIBLE SEPARATOR
        "⁤",  # INVISIBLE PLUS
        "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
    | {chr(cp) for cp in range(0x206A, 0x2070)}   # deprecated format controls, incl. U+206F
    | {chr(cp) for cp in range(0xFE00, 0xFE10)}   # VARIATION SELECTOR-1..16
)

# Unprintable C0/C1 controls. Deliberately included even though most are ASCII (the one place this
# module touches ASCII): a raw 0x00 or 0x01 inside a decoded name is never part of a brand or a
# domain, but it does fragment a cluster. \t \n \r \f \v are excluded -- they are whitespace and are
# collapsed, not deleted, so "a\nb" does not become the different word "ab".
_CONTROL_CODEPOINTS = frozenset(
    {chr(cp) for cp in range(0x00, 0x20)} - set("\t\n\r\f\v")
    | {chr(0x7F)}
    | {chr(cp) for cp in range(0x80, 0xA0)}
)

#: Codepoints deleted outright by :func:`strip_invisible`.
INVISIBLE_CODEPOINTS: frozenset[str] = _FORMAT_CODEPOINTS | _CONTROL_CODEPOINTS

# --------------------------------------------------------------------------------------------------
# Vendored confusable table. Keys are single NON-ASCII codepoints, values are ASCII.
#
# Two invariants make skeleton() idempotent, and _validate_confusable_map enforces them:
#   1. no key is ASCII        -> folding never rewrites the target alphabet, so a second pass is a
#                                no-op (a hypothetical 'l' -> '1' entry would also destroy real names)
#   2. every value is ASCII   -> a value can never itself be a key, and NFKC(value) == value, so the
#                                second pass's NFKC cannot reopen what this pass closed
#
# Case is preserved rather than lowercased here because skeleton() casefolds AFTER folding; keeping the
# case means fold_confusables is usable on its own without changing what the string looks like.
# Coverage is deliberately asymmetric where the scripts are: Greek capital zeta is indistinguishable
# from "Z" while lowercase zeta is not a "z" lookalike, so only the capital is mapped. Mapping
# unconvincing pairs would merge legitimately distinct brands, which is the one direction Proposition 1
# does not protect against.
# --------------------------------------------------------------------------------------------------
_CYRILLIC: dict[str, str] = {
    "а": "a", "А": "A",   # а А  CYRILLIC LETTER A
    "в": "b", "В": "B",   # в В  CYRILLIC LETTER VE
    "с": "c", "С": "C",   # с С  CYRILLIC LETTER ES      (fomox.сlub evasion, Optimism)
    "ԁ": "d", "Ԁ": "D",   # ԁ Ԁ  CYRILLIC LETTER KOMI DE
    "е": "e", "Е": "E",   # е Е  CYRILLIC LETTER IE      (aavе evasion)
    "г": "r",                  # г    CYRILLIC LETTER GHE -- renders as "r", not as "g"
    "һ": "h", "Һ": "H",   # һ Һ  CYRILLIC LETTER SHHA
    "н": "h", "Н": "H",   # н Н  CYRILLIC LETTER EN
    "і": "i", "І": "I",   # і І  CYRILLIC LETTER BYELORUSSIAN-UKRAINIAN I
    "ј": "j", "Ј": "J",   # ј Ј  CYRILLIC LETTER JE
    "к": "k", "К": "K",   # к К  CYRILLIC LETTER KA
    "ӏ": "l", "Ӏ": "I",   # ӏ Ӏ  CYRILLIC LETTER PALOCHKA (small reads "l", capital reads "I")
    "м": "m", "М": "M",   # м М  CYRILLIC LETTER EM
    "о": "o", "О": "O",   # о О  CYRILLIC LETTER O
    "р": "p", "Р": "P",   # р Р  CYRILLIC LETTER ER
    "ԛ": "q", "Ԛ": "Q",   # ԛ Ԛ  CYRILLIC LETTER QA
    "ѕ": "s", "Ѕ": "S",   # ѕ Ѕ  CYRILLIC LETTER DZE     (UЅDС evasion, Base)
    "т": "t", "Т": "T",   # т Т  CYRILLIC LETTER TE
    "ѵ": "v", "Ѵ": "V",   # ѵ Ѵ  CYRILLIC LETTER IZHITSA
    "ԝ": "w", "Ԝ": "W",   # ԝ Ԝ  CYRILLIC LETTER WE
    "ѡ": "w", "Ѡ": "W",   # ѡ Ѡ  CYRILLIC LETTER OMEGA
    "х": "x", "Х": "X",   # х Х  CYRILLIC LETTER HA
    "у": "y", "У": "Y",   # у У  CYRILLIC LETTER U
    "з": "3", "З": "3",   # з З  CYRILLIC LETTER ZE -- digit lookalike, both cases
}

_GREEK: dict[str, str] = {
    "α": "a", "Α": "A",   # α Α  ALPHA
    "β": "b", "Β": "B",   # β Β  BETA
    "ϲ": "c", "Ϲ": "C",   # ϲ Ϲ  LUNATE SIGMA
    "ε": "e", "Ε": "E",   # ε Ε  EPSILON
    "ϝ": "f", "Ϝ": "F",   # ϝ Ϝ  DIGAMMA
    "η": "n", "Η": "H",   # η Η  ETA -- small reads "n", capital reads "H"
    "ι": "i", "Ι": "I",   # ι Ι  IOTA
    "ϳ": "j", "Ϳ": "J",   # ϳ Ϳ  YOT
    "κ": "k", "Κ": "K",   # κ Κ  KAPPA
    "Μ": "M",                  #   Μ  MU (capital only; µ is not a convincing "u")
    "Ν": "N",                  #   Ν  NU (capital only; ν reads "v", see below)
    "ο": "o", "Ο": "O",   # ο Ο  OMICRON
    "ρ": "p", "Ρ": "P",   # ρ Ρ  RHO
    "τ": "t", "Τ": "T",   # τ Τ  TAU
    "υ": "u", "Υ": "Y",   # υ Υ  UPSILON -- small reads "u", capital reads "Y"
    "ν": "v",                  #   ν  NU (small) reads "v"
    "ω": "w",                  #   ω  OMEGA (small) reads "w"
    "χ": "x", "Χ": "X",   # χ Χ  CHI
    "γ": "y",                  #   γ  GAMMA (small) reads "y"
    "Ζ": "Z",                  #   Ζ  ZETA (capital only; ζ is not a "z" lookalike)
}

# Fullwidth forms U+FF01-U+FF5E map 1:1 onto printable ASCII U+0021-U+007E. Generated from that
# documented offset rather than typed out: 94 hand-written lines is 94 chances to transpose a
# codepoint. NFKC already performs this fold, so these entries are unreachable from skeleton() -- they
# exist so fold_confusables() is correct when called on its own, which the tests exercise.
_FULLWIDTH: dict[str, str] = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}

#: Vendored single-codepoint -> ASCII folding. See the block comment above for the invariants.
CONFUSABLE_MAP: dict[str, str] = {**_CYRILLIC, **_GREEK, **_FULLWIDTH}


def _validate_confusable_map(mapping: dict[str, str]) -> None:
    """Fail at import time on a map edit that would break idempotence.

    Inputs: candidate mapping. Outputs: None. Side effects: none.
    Failure modes: raises ValueError naming the offending entry. Runs at import so a bad edit cannot
    reach a result file -- a silently non-idempotent skeleton would invalidate Proposition 1 and
    every campaign count derived from it, with no visible symptom.
    """
    for key, value in mapping.items():
        if len(key) != 1:
            raise ValueError(
                f"CONFUSABLE_MAP key {key!r} is {len(key)} characters; keys must be single "
                f"codepoints because fold_confusables folds codepoint-wise"
            )
        if ord(key) <= _ASCII_MAX:
            raise ValueError(
                f"CONFUSABLE_MAP key {key!r} (U+{ord(key):04X}) is ASCII; an ASCII key would rewrite "
                f"the target alphabet and make skeleton() non-idempotent"
            )
        if not value:
            raise ValueError(f"CONFUSABLE_MAP value for {key!r} is empty; use fold_emoji to delete")
        if not value.isascii():
            raise ValueError(
                f"CONFUSABLE_MAP value {value!r} for key {key!r} is not ASCII; a non-ASCII value can "
                f"be reopened by the next pass's NFKC or be a key itself"
            )
        # skeleton() folds a SECOND time after casefolding, so a caseless key is still foldable at
        # that point and its value is emitted after the last casefold. An upper-case value would then
        # survive into the key and be lower-cased by the next pass -- i.e. non-idempotent. Caseless
        # keys must therefore carry caseless values. (Keys that DO have case are unaffected: they are
        # consumed by the first fold, before casefold runs.)
        if key.casefold() == key and value.casefold() != value:
            raise ValueError(
                f"CONFUSABLE_MAP maps caseless key {key!r} to case-bearing value {value!r}; that "
                f"value would escape skeleton()'s casefold and break idempotence"
            )


_validate_confusable_map(CONFUSABLE_MAP)

# Categories So (Other_Symbol) and Sk (Modifier_Symbol) cover emoji, pictographs, skin-tone modifiers,
# regional-indicator flag halves, ™, ©, °. Sm (Math_Symbol) and Sc (Currency_Symbol) are NOT folded:
# "$" and "×" carry meaning in a token name.
_SYMBOL_CATEGORIES = frozenset({"So", "Sk"})

# Explicit ranges for codepoints whose category is not So/Sk but which are plainly decoration.
# U+2400-U+24FF (enclosed alphanumerics) is deliberately ABSENT: those are letters, and NFKC turns
# them into ASCII letters. That is also why skeleton() runs NFKC before fold_emoji -- run the other
# way round, a name written "🅳🅾🆃" (all category So) would lose its letters instead of folding to "dot".
_PICTOGRAPHIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x20D0, 0x20FF),   # combining diacritics for symbols, incl. U+20E3 keycap enclosure
    (0x2190, 0x21FF),   # arrows
    (0x2300, 0x23FF),   # miscellaneous technical
    (0x2500, 0x27BF),   # box drawing, blocks, geometric shapes, misc symbols, dingbats
    (0x2B00, 0x2BFF),   # miscellaneous symbols and arrows
    (0xE000, 0xF8FF),   # BMP private use -- vendor glyphs, never part of a brand name
    (0x1F000, 0x1FAFF),  # emoji planes. NOT extended to 0x1FBFF: U+1FBF0-U+1FBF9 are DIGITs.
)


# Symbols that stand for a WORD rather than a letter: U+2122 "™" -> "TM", U+2116 "№" -> "No",
# U+1F14D "🅍" -> "SS". They are category So, so fold_emoji deletes them -- but NFKC runs first and
# turns them into ASCII letters, which defeats that. Measured consequence: "USDC™" became "usdctm",
# and "www.fomox.club™" became "www.fomox.clubtm", whose registrable domain no longer has a valid TLD,
# so the campaign was LOST. They are therefore deleted before NFKC, restoring the intended behaviour.
#
# The complementary case must NOT be deleted: a symbol whose expansion preserves a single letter or
# digit IS that letter in disguise and is a real evasion -- U+1F144 "🅄" -> "U", U+249C "⒜" -> "(a)",
# U+00B2 "²" -> "2". Hence the "two or more ASCII letters" predicate rather than "expands at all".
#
# Membership is derived from the interpreter's own UCD over these vendored ranges (~5,000 codepoints,
# once at import) rather than hand-listing 99 codepoints. Nothing is downloaded; unicodedata ships
# with Python, and this module's behaviour already depends on its Unicode version through
# category()/normalize(). The ranges bound the search: every member of the class lives in Letterlike
# Symbols, the CJK squared-abbreviation blocks, or Enclosed Alphanumeric Supplement.
_ABBREVIATION_SEARCH_RANGES: tuple[tuple[int, int], ...] = (
    (0x2100, 0x213F),    # Letterlike Symbols: № ℠ ℡ ™ ℻
    (0x3250, 0x33FF),    # CJK squared units and abbreviations: ㎏ ㏈ ㋏
    (0x1F100, 0x1F19F),  # Enclosed Alphanumeric Supplement: 🅍 🅪 🆐
)
_MULTI_LETTER_EXPANSION = re.compile(r"[A-Za-z]{2,}")


def _abbreviation_symbols() -> frozenset[str]:
    """So/Sk characters whose NFKC expansion is two or more ASCII letters. See the comment above."""
    found: set[str] = set()
    for low, high in _ABBREVIATION_SEARCH_RANGES:
        for codepoint in range(low, high + 1):
            ch = chr(codepoint)
            if unicodedata.category(ch) not in _SYMBOL_CATEGORIES:
                continue
            if _MULTI_LETTER_EXPANSION.fullmatch(unicodedata.normalize("NFKC", ch)):
                found.add(ch)
    return frozenset(found)


ABBREVIATION_SYMBOLS: frozenset[str] = _abbreviation_symbols()


def _require_str(value: object, argument: str = "s") -> str:
    """Reject a non-str input loudly instead of coercing it.

    Inputs: candidate value, argument name for the message. Outputs: the value, narrowed to str.
    Side effects: none. Failure modes: raises TypeError.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{argument} must be str, got {type(value).__name__}: {value!r}. Not coerced on purpose -- "
            f"str() on a pandas missing value yields 'nan', which would cluster as a real token name."
        )
    return value


def strip_invisible(s: str) -> str:
    """Delete zero-width, format and control characters (see INVISIBLE_CODEPOINTS).

    Inputs: any str. Outputs: the string with those codepoints removed; whitespace is left alone so
    word boundaries survive. Side effects: none. Failure modes: TypeError on a non-str input.
    """
    _require_str(s)
    if not s:
        return s
    return "".join(ch for ch in s if ch not in INVISIBLE_CODEPOINTS)


def fold_confusables(s: str) -> str:
    """Apply CONFUSABLE_MAP codepoint-wise (Cyrillic/Greek lookalikes, fullwidth forms).

    Inputs: any str. Outputs: the string with mapped codepoints replaced by their ASCII lookalike;
    case is preserved (skeleton casefolds later). Side effects: none.
    Failure modes: TypeError on a non-str input. An unmapped non-ASCII codepoint is passed through
    unchanged -- that fragments a cluster, which is the safe direction, and never merges two brands.
    """
    _require_str(s)
    return "".join(CONFUSABLE_MAP.get(ch, ch) for ch in s)


def fold_emoji(s: str) -> str:
    """Delete emoji, pictographs and decorative symbols.

    Inputs: any str. Outputs: the string with category So/Sk codepoints and the explicit pictographic
    ranges removed. ASCII is never touched: "^" and "`" are category Sk but are not pictographs, and
    deleting them would corrupt the very alphabet this module folds towards.
    Side effects: none. Failure modes: TypeError on a non-str input. An all-emoji string becomes "",
    which callers must not use as a cluster key (see the module docstring).
    """
    _require_str(s)
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if cp <= _ASCII_MAX:
            out.append(ch)
            continue
        if unicodedata.category(ch) in _SYMBOL_CATEGORIES:
            continue
        if any(lo <= cp <= hi for lo, hi in _PICTOGRAPHIC_RANGES):
            continue
        out.append(ch)
    return "".join(out)


# Bracket pairs must MATCH ("[.]", not "[.)"). Written as pairs rather than as two character classes
# because a class accepts mixed brackets, and a substitution can manufacture one: collapsing "{.}"
# inside "c[{.})b" leaves "c[.)b", which a mixed-bracket pattern then matched on the NEXT pass --
# non-idempotent, found by the seeded fuzz at 1 case in 20,000.
_BRACKET_PAIRS: tuple[tuple[str, str], ...] = ((r"\[", r"\]"), (r"\(", r"\)"), (r"\{", r"\}"))


def _bracketed(inner: str) -> str:
    """Regex alternation matching `inner` inside any one MATCHING bracket pair."""
    return "|".join(rf"{left}\s*{inner}\s*{right}" for left, right in _BRACKET_PAIRS)


# "[.]", "(.)", "{.}" with optional inner whitespace. Unambiguous: no legitimate name writes a dot
# inside brackets.
_BRACKETED_DOT = re.compile(_bracketed(r"\."))

# The word "dot" standing in for ".". Two shapes, both requiring an alphanumeric on each side so the
# fold only fires between what could be domain labels:
#   - bracketed and TIGHT ("www[dot]fomox"): no whitespace allowed outside the brackets. This is what
#     keeps a ticker in parentheses intact -- "Polkadot (DOT)" has a space before "(" and does not
#     match, whereas the adversarial form is always written closed up.
#   - bare, whitespace-delimited ("www dot fomox").
# Case-insensitive on purpose; the residual false positive is documented in the module docstring.
_WORD_DOT = re.compile(
    rf"(?<=[0-9A-Za-z])(?:{_bracketed('dot')}|\s+dot\s+)(?=[0-9A-Za-z])",
    re.IGNORECASE,
)

_WHITESPACE_RUN = re.compile(r"\s+")


def deobfuscate(s: str) -> str:
    """Collapse the dot-obfuscations that hide a domain from extraction, and squeeze whitespace.

    Handles "[.]", "(.)", "{.}", "[dot]", "(dot)" and " dot " (any case). Inputs: any str.
    Outputs: the string with those forms replaced by "." and runs of whitespace collapsed to one
    space; leading/trailing whitespace is left for skeleton() to strip.
    Nested forms are collapsed to a FIXED POINT ("a[{.}]b" -> "a[.]b" -> "a.b"), because one
    substitution can complete another: the replacement dot lands between the outer brackets. Iterating
    is what makes the function idempotent, and it terminates because every substitution strictly
    shortens the string -- so exceeding len(s) rounds is impossible and RAISES rather than looping.

    Side effects: none. Failure modes: TypeError on a non-str input; RuntimeError if the fixed point
    is somehow not reached (unreachable by the shortening argument, loud rather than an infinite loop);
    the " DOT " ticker false positive documented in the module docstring.
    """
    _require_str(s)
    for _ in range(len(s) + 1):
        collapsed = _WORD_DOT.sub(".", _BRACKETED_DOT.sub(".", s))
        if collapsed == s:
            break
        s = collapsed
    else:
        raise RuntimeError(
            f"deobfuscate did not reach a fixed point on {s!r}; every substitution should shorten "
            f"the string, so this means a substitution grew it -- fix the patterns, not this bound."
        )
    return _WHITESPACE_RUN.sub(" ", s)


def skeleton(s: str) -> str:
    """The composed normal form used as the campaign-clustering key.

    Order is load-bearing and fixed:
      0. delete ABBREVIATION_SYMBOLS -- must precede NFKC, which would otherwise turn "™" into the
                          letters "TM" before fold_emoji could drop it (see the comment on that set).
      1. NFKC          -- unifies compatibility variants first, so fullwidth "ＵＳＤＣ", bold
                          "𝐔𝐒𝐃𝐂" and enclosed "🅄🅂🄳🄲" become letters BEFORE fold_emoji could
                          delete them as symbols.
      2. strip_invisible -- remove injected zero-widths, so "U⁠SD‍C" is one word.
      3. fold_confusables -- Cyrillic/Greek lookalikes to ASCII, while case still distinguishes them.
      4. fold_emoji    -- drop decoration that survived NFKC.
      5. casefold      -- after steps 3-4, so they still see the original case.
      6. CLOSURE: fold_confusables again, then NFKC again -- see below; this is what makes the
                          function idempotent, and it is measured, not assumed.
      7. deobfuscate   -- restore dots so a domain is extractable. LAST, because it is the only
                          context-sensitive step: it fires on "[dot]" only between two alphanumerics,
                          so it must see every character's final identity. Run before step 6 (as
                          first written) it missed "z[dot]ᲀ", because U+1C80 only becomes the
                          alphanumeric "b" at the closure -- and pass two then found the match that
                          pass one had walked past. A seeded fuzz caught this on 94 of 2,000 inputs;
                          the single-codepoint sweep could not see it, since it takes two characters.
      8. collapse whitespace, strip.

    Why step 6 exists. Steps 3 and 5 each FEED an earlier step, so a pipeline that stops at 5 is not
    closed, and the un-closed version was not idempotent on 104 codepoints:

      - casefold (5) is a source of foldable characters (96 codepoints). U+0393 "Γ" is deliberately
        unmapped (capital gamma resembles no ASCII letter) but casefolds to "γ", which IS mapped, so
        pass two produced "y" where pass one produced "γ". Likewise U+03A9 "Ω"->"ω", U+0413 "Г"->"г",
        U+2126 OHM SIGN, the U+1C80 Cyrillic variant letters, and the polytonic Greek block, where one
        character casefolds to TWO ("ᾳ" -> "αι").
      - fold_confusables (3) is a source of NFKC-unstable text (8 codepoints). Replacing a base letter
        with ASCII leaves its combining marks attached: "ΐ" (iota + dialytika + tonos) becomes
        "i" + U+0308 + U+0301, which pass two's NFKC then composes to U+1E2F.

    The first is not fixable in the table: those 96 entries would need multi-character, non-ASCII
    values, breaking the invariant that keeps the fold closed. And dropping step 3 in favour of
    folding only after casefold would lose real coverage -- "ΖΕΤΑ" folds to "zeta" only because step 3
    still sees the capitals, lowercase zeta being an unmapped non-lookalike. So both applications earn
    their place. Order within step 6 is fold-then-NFKC: verified over every codepoint in every plane
    (1,114,112 x 5 templates, 0 violations) plus 60,000 seeded random mixtures over three seeds;
    NFKC-then-fold is also sound but does strictly more work.

    Inputs: a decoded name()/symbol() str. Outputs: the clustering key. Side effects: none.
    Failure modes: TypeError on a non-str input; returns "" when nothing survives (empty input,
    only-invisible, only-emoji) and "" must not be used as a cluster key.
    Idempotent: skeleton(skeleton(s)) == skeleton(s) for every input (Proposition 1); verified by
    exhaustive sweep over every assigned codepoint in the test suite, not merely on examples.
    """
    _require_str(s)
    folded = "".join(ch for ch in s if ch not in ABBREVIATION_SYMBOLS)
    folded = unicodedata.normalize("NFKC", folded)
    folded = strip_invisible(folded)
    folded = fold_confusables(folded)
    folded = fold_emoji(folded)
    folded = unicodedata.normalize("NFKC", fold_confusables(folded.casefold()))
    return _WHITESPACE_RUN.sub(" ", deobfuscate(folded)).strip()


def is_idempotent(s: str) -> bool:
    """True when re-normalising the key is a no-op: skeleton(skeleton(s)) == skeleton(s).

    Inputs: any str. Outputs: bool. Side effects: none. Failure modes: TypeError on a non-str input.
    Exposed so a pipeline can assert the property on real corpus strings instead of trusting it.
    """
    once = skeleton(s)
    return skeleton(once) == once
