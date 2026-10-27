"""Detect token contracts whose name()/symbol() strings ARE the advertisement.

Why this exists
---------------
A class of contracts exists so that a string appears in a stranger's wallet: "Claim OP Rewards at
https://optimisam.org", "PENDLE | t.me/s/claimpendle | *Claim within 7 days", "# usdc-airdrop.com /
Visit usdc-airdrop.com to claim $USDC". The token is not bait for a trade and not bait for a mistyped
address; it is a delivery medium for a URL. This module identifies that class and extracts the payload.

Two things this module deliberately is NOT
-----------------------------------------
1. **Not a brand-registry matcher.** It shares no code path with `heuristic.find_impersonators`, takes
   no list of legitimate names or symbols, and never compares against a brand. That is the point: on
   Base the registry rule finds no lure contract at all (its released fake-token set is 31 rows, all
   legitimate LP/lending tokens, 0 URL/verb hits) while this rule finds 315 in the volume head. A
   registry cannot see a lure whose name impersonates nothing.
2. **Not a novel technique.** Forta's open-source spam detector
   (kovart/forta-spam-detector, src/analyzer/modules/phishing-metadata.ts, public since April 2023) already
   extracts links from name()+symbol(), applies an imperative-verb lexicon, requires mass broadcast
   (ERC20_RECEIVERS_THRESHOLD = 499), deobfuscates "[.]"/"[dot]", and folds homoglyphs before link
   extraction. We reimplement it faithfully at census scale and MEASURE what it finds; the detector is
   cited prior art, and the contribution is the measurement, the campaign reconstruction and the
   evaluation. Ye et al. (WWW 2024) separately own the wallet-UI-omission framing and a homograph table.

When normalisation matters, and when it measurably does not
-----------------------------------------------------------
It matters when the substitution falls INSIDE the domain or handle. Two distinct Optimism contracts read
"www.fomox.club" and "www.fomox.сlub" (Cyrillic U+0441); without folding before extraction, that one
campaign fragments into two. `confusables` therefore runs first, always, and a test pins that pair.

But the effect must not be overstated, and on the only sample measurable so far it is NULL. Ablating
every stage (confusable fold, invisible strip, deobfuscation, and all normalisation) over the 1,199
cached Base contracts at volume ranks 3-1206 changes nothing: 132 flagged, 42 campaigns, 0
unattributable, identical in every arm. The reason is diagnostic rather than reassuring -- on that
sample the Cyrillic sits in the BRAND-NAME portion ("✅СIRCLE TOKEN DISTRIBUTION", 58 of the 132 flagged
contracts carry Cyrillic somewhere), while the campaign identities are Telegram handles and domains that
are already pure ASCII. Folding a brand name cannot change which domain is extracted.

So the defensible claim is conditional: normalisation is load-bearing WHERE the substitution lands in the
identity-bearing part of the string, and its effect on campaign counts is an empirical question per
chain, not a given. Any paper text asserting it must cite the ablation on a chain where it binds.

The precision problem this module must not have
-----------------------------------------------
A bare regex over TLDs is not a measurement. An early probe version matched "Curve.fi 2CRV" on the `.fi`
TLD, and the FTT "decorated" matcher in the previous paper misclassifies Aave receipt tokens, Axelar
wrapped USDC and SyncSwap LP tokens as counterfeits. Hence: an explicit benign-domain allowlist, a
`strength` that separates a bare domain mention from scheme-plus-imperative, and a requirement that any
published prevalence figure be backed by a two-rater labelled precision set. `classify` returns the
evidence it used so a human adjudicator can see WHY a contract was flagged.

Inputs
------
Token metadata strings (name, symbol) as fetched by `token_meta.fetch_metadata`. No network, no disk.

Outputs
-------
`LureVerdict` per contract. Nothing is written by this module.

Failure modes
-------------
- A None/empty name or symbol is tolerated (many lure contracts set only one of the two) but a verdict
  built from no strings at all is `strength="none"`, never a silent True.
- `registrable_domain` uses a vendored two-label rule plus a multi-part-suffix set; it is an
  APPROXIMATION of the Public Suffix List and its failure mode is documented on the function. A domain
  under an unlisted multi-part suffix collapses to the wrong registrable name, which would SPLIT a
  campaign rather than merge two -- the conservative direction, but it must be reported.

Complexity: O(len(name) + len(symbol)) per contract; pure string work, no allocation of note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .confusables import skeleton

# --------------------------------------------------------------------------- vendored tables

#: Imperative/lure lexicon. The first block reimplements Forta's published PHISHING_NAME_KEYWORDS
#: (visit, claim, activate, reward, giveaway, free, bonus, mint, gift, access); the rest were observed
#: in the corpus. "at" and "to" from Forta's list are deliberately EXCLUDED here: as bare English
#: prepositions they fire on legitimate names and Forta can afford them only because it additionally
#: requires a 499-recipient broadcast. Our recipient threshold is applied separately, so the lexicon
#: must stand on its own.
LURE_VERBS: frozenset[str] = frozenset({
    "claim", "claimable", "visit", "activate", "reward", "rewards", "giveaway", "free", "bonus",
    "gift", "access", "redeem", "unlock", "unlocked", "collect", "airdrop", "airdropped",
    "distribution", "eligible", "eligibility", "withdraw", "swap", "connect", "verify", "enter",
    "join", "open", "start", "get", "win", "winner", "voucher", "presale", "whitelist",
})

#: Valid top-level domains, from the pinned IANA authority list (1,438 entries, version 2026081200).
#:
#: This replaced a hand-curated set of ~70 TLDs, and the reason is worth recording because it is a
#: methodological point rather than a convenience. Human annotation of a 372-contract Base sample found
#: nine lures the detector missed, and six of those misses had a single mechanical cause: the campaign's
#: TLD was absent from the curated list (`moodeng.ink`, `degen.gifts`, `fli.so`, `kutt.it`, `USD.AC`).
#: All five are real IANA TLDs. Patching the curated list to cover them would have been fitting the
#: detector to the annotated sample, and any recall improvement measured afterwards on that same sample
#: would have been optimistically biased. Adopting the complete authority list instead removes the
#: author's choice from the measurement entirely: "is this a valid TLD" becomes a recorded, external,
#: reproducible fact, and the whole class of gap disappears rather than being patched case by case.
#:
#: The cost is that more strings now register a destination, so the `weak` band grows. Precision is
#: protected structurally rather than by the TLD list: `strong` requires a destination AND a call to
#: action, so a project whose brand happens to be a domain (`Tairon.ai`, `Sport.fun`, `RETAKE.TV`,
#: `FLock.io`, `Rai.Finance` -- all present in the sample, all correctly judged legitimate by the
#: annotator) lands in `weak`, never in `strong`.
def _load_iana_tlds() -> frozenset[str]:
    """Read the pinned IANA list. Raises if it is absent -- a silently empty TLD set would disable
    domain extraction entirely and report zero lures as a finding."""
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2]
            / "papers" / "lure-broadcast" / "data" / "tlds" / "tlds-alpha-by-domain.txt")
    if not path.exists():
        raise FileNotFoundError(
            f"pinned IANA TLD list missing at {path}; run papers/lure-broadcast/scripts/fetch_tlds.py. "
            f"Refusing to fall back to a curated list, which would silently change what counts as a "
            f"domain and therefore what counts as a lure."
        )
    tlds = {ln.strip().lower() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}
    if len(tlds) < 1000:
        raise ValueError(f"pinned TLD list has only {len(tlds)} entries; expected ~1,400. Refusing to "
                         f"run against a truncated authority file.")
    return frozenset(tlds)


LURE_TLDS: frozenset[str] = _load_iana_tlds()

#: Multi-part public suffixes needed so `registrable_domain` does not truncate to the wrong label.
#: An approximation of the Public Suffix List, kept deliberately small and vendored (no network at
#: analysis time). Extend when a campaign is observed under an unlisted suffix.
MULTI_SUFFIXES: frozenset[str] = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.br", "com.cn", "com.au", "co.jp", "co.kr",
    "com.mx", "com.tr", "com.tw", "co.in", "co.za", "com.ar", "com.sg", "com.hk", "co.nz",
    "github.io", "s3.amazonaws.com", "web.app", "firebaseapp.com", "netlify.app", "vercel.app",
    "pages.dev", "workers.dev", "r2.dev", "ipfs.io", "eth.link", "eth.limo",
})

#: URL shorteners and messaging platforms. These are DESTINATION INDIRECTIONS, not campaign
#: identities, and conflating the two is a measurement error with a known instance: the earlier claim
#: that brand-registry matching yields "0 of 116 shared cross-chain domains" failed to reproduce partly
#: because `t.ly` -- a shortener used by unrelated campaigns -- was counted as a shared domain. Keying on
#: the host would likewise collapse every Telegram campaign into a single "t.me", merging distinct
#: operators and under-counting campaigns. So these hosts still count as a destination for STRENGTH
#: (they are genuinely URLs), but the campaign key comes from the handle or path behind them.
AGGREGATOR_HOSTS: frozenset[str] = frozenset({
    "t.me", "telegram.me", "telegram.dog", "telesco.pe",
    "t.ly", "bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "rb.gy", "rebrand.ly", "shorturl.at",
    "short.io", "s.id", "v.gd", "ow.ly", "buff.ly", "trib.al", "lnkd.in", "goo.gl",
    "linktr.ee", "linkr.bio", "beacons.ai", "carrd.co",
    "discord.gg", "discord.com", "twitter.com", "x.com", "medium.com", "mirror.xyz",
    "github.io", "notion.site", "gitbook.io",
})

#: Domains that legitimately appear in token metadata. Without this, "Curve.fi 2CRV" is a false
#: positive on the `.fi` TLD -- an error actually made by an early probe. Any protocol whose own brand
#: IS a domain belongs here. This list is an input to a PRECISION claim, so it must be reported in the
#: paper and swept in the sensitivity analysis, not treated as a private constant.
BENIGN_DOMAINS: frozenset[str] = frozenset({
    "curve.fi", "aerodrome.finance", "velodrome.finance", "uniswap.org", "sushi.com", "aave.com",
    "syncswap.xyz", "zerolend.xyz", "pendle.finance", "balancer.fi", "gmx.io", "1inch.io",
    "lido.fi", "rocketpool.net", "frax.finance", "yearn.fi", "convexfinance.com", "stargate.finance",
    "across.to", "hop.exchange", "connext.network", "axelar.network", "layerzero.network",
    "ethena.fi", "usual.money", "ether.fi", "renzoprotocol.com", "kelpdao.xyz", "swell.network",
    "beefy.finance", "quickswap.exchange", "camelot.exchange", "ramses.exchange", "chronos.exchange",
})

# A scheme-bearing URL is much stronger evidence than a bare domain mention.
_SCHEME_RE = re.compile(r"\b(?:https?://|www\.)", re.I)

# One regex matching optional scheme + host + optional path, scanned with finditer so that each match
# CONSUMES its path. That consumption is the whole point: a path segment can look exactly like a
# hostname, and re-scanning it produces a false host. Observed live: "Claim on https://t.ly/optimism.io",
# where "optimism.io" is a shortener SLUG -- and is also Optimism's genuine domain, so reading it as a
# host would attribute a phishing campaign to the brand it impersonates. Because finditer does not
# overlap, matching host+path together means the slug is inside the consumed span and never re-matched.
# (An earlier attempt used a negative lookbehind on "/" instead; that also rejected the host in
# "https://host", because a scheme ends in "//". Consuming the path is the correct mechanism.)
_URL_RE = re.compile(
    r"(?:https?://|www\.)?"
    r"(?P<host>(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?P<tld>[a-z]{2,24}))"
    r"(?P<path>/[^\s\]\)\|,;'\"]*)?",
    re.I,
)
# Forta's price pattern; matches e.g. "Optimism (OP) (@$1.4537)".
_PRICE_RE = re.compile(r"\$\s*\d")
_TELEGRAM_RE = re.compile(r"\b(?:t\.me|telegram\.me|telegram\.dog)/(?:s/)?([a-z0-9_]{3,64})", re.I)


@dataclass(frozen=True)
class LureVerdict:
    """Why a contract was or was not flagged. Carries its evidence so a human can adjudicate."""
    is_lure: bool
    strength: str                    # "strong" | "weak" | "none"
    domains: tuple[str, ...] = ()     # registrable domains, normalised, benign ones removed
    verbs: tuple[str, ...] = ()
    has_scheme: bool = False
    has_price: bool = False
    telegram_handles: tuple[str, ...] = ()
    benign_domains: tuple[str, ...] = ()   # matched but allowlisted; kept for the precision audit
    aggregator_hosts: tuple[str, ...] = ()  # shorteners/messaging: a destination, never an identity
    normalised: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)


def registrable_domain(host: str) -> str:
    """Collapse a hostname to its registrable domain ("app.optimisam.ltd" -> "optimisam.ltd").

    Vendored approximation of the Public Suffix List: strip to the last two labels, unless the last
    two form a known multi-part suffix, in which case keep three. FAILURE MODE: a domain under an
    unlisted multi-part suffix (e.g. a country-code second level not in MULTI_SUFFIXES) collapses to
    the suffix itself, which SPLITS one campaign into several rather than merging two distinct ones --
    conservative for our claims, but it must be reported rather than assumed away. Extend
    MULTI_SUFFIXES when such a campaign is observed.
    """
    labels = [x for x in host.lower().strip(".").split(".") if x]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in MULTI_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def extract_domains(text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Registrable domains in `text`, split into (suspect, benign, aggregator).

    Three buckets, because they play three different roles and merging any two is a measurement error:
      suspect    -- a campaign identity; these are the campaign-clustering keys.
      benign     -- allowlisted protocol domains (curve.fi and friends). Kept, not discarded, so the
                    precision audit can report what the allowlist suppressed.
      aggregator -- shorteners and messaging hosts (t.me, t.ly, ...). Evidence of a destination, but
                    NOT an identity: keying on them merges unrelated campaigns.

    Runs on ALREADY-normalised text (see `classify`), because a Cyrillic homoglyph inside a hostname
    would otherwise produce a second, spurious campaign. Only hosts whose TLD is in LURE_TLDS are
    returned, so version strings and LP-pair names ("USDC/WETH 0.05") do not register as domains.
    Order is preserved and duplicates removed.
    """
    suspect: list[str] = []
    benign: list[str] = []
    aggregator: list[str] = []

    for m in _URL_RE.finditer(text):
        if m.group("tld").lower() not in LURE_TLDS:
            continue
        reg = registrable_domain(m.group("host"))
        if not reg or "." not in reg:
            continue
        if reg in AGGREGATOR_HOSTS:
            # Keep the path: it is what distinguishes two campaigns behind one shared host.
            slug = (m.group("path") or "").lower().lstrip("/").rstrip("/")
            if slug.startswith("s/"):          # t.me/s/<handle> is the channel-preview form
                slug = slug[2:]
            entry, bucket = (f"{reg}/{slug}" if slug else reg), aggregator
        elif reg in BENIGN_DOMAINS:
            entry, bucket = reg, benign
        else:
            entry, bucket = reg, suspect
        if entry not in bucket:
            bucket.append(entry)
    return tuple(suspect), tuple(benign), tuple(aggregator)


def extract_verbs(text: str) -> tuple[str, ...]:
    """Imperative lure words in already-normalised `text`, as whole words, deduplicated.

    A trailing "s" is stripped before matching so plurals hit the same lexicon entry. This was a real
    miss, not a hypothetical: the sample contains `Airdapp.net / View Airdrops at ...` whose only
    inducement is the plural "Airdrops", which failed to match a lexicon holding "airdrop" and
    "airdropped" but not "airdrops". Enumerating every inflection by hand invites the same gap again, so
    the match is done on the stem. Both forms are reported, so the evidence trail shows the word that
    actually appeared.
    """
    words = set(re.findall(r"[a-z]+", text))
    hits = words & LURE_VERBS
    for w in words:
        if w.endswith("s") and w[:-1] in LURE_VERBS:
            hits.add(w)
    return tuple(sorted(hits))


def classify(name: str | None, symbol: str | None, normalise: bool = True) -> LureVerdict:
    """Decide whether a contract's metadata is a delivery payload, and record the evidence.

    Normalisation runs FIRST (Proposition 1): NFKC, invisible-character stripping, confusable folding,
    emoji folding, dot-deobfuscation, casefold. `name` and `symbol` are concatenated because the payload
    is routinely split across them -- e.g. name "PENDLE" with symbol
    "PENDLE | t.me/s/claimpendle | *Claim within 7 days".

    Strength, deliberately three-valued so a prevalence figure can be reported at either bar:
      strong -- a suspect domain or an explicit scheme, TOGETHER WITH an imperative verb, a Telegram
                handle or a price pattern. This is the conjunction Forta's production rule uses and is
                the bar for the paper's headline counts.
      weak   -- one side only: a suspect domain with no call to action, or verbs with no destination.
                Reported separately; NEVER merged into the headline, because "free" or "claim" appears
                in legitimate names and a bare domain appears in legitimate branding.
      none   -- no evidence.

    `is_lure` is True only for `strong`. A caller wanting the looser population must ask for it
    explicitly by reading `strength`, so the stricter number cannot be inflated by accident.

    `normalise=False` skips the confusable/invisible/emoji fold and classifies the raw string. It
    exists solely to ABLATE Proposition 1, which predicts that omitting the fold can only split a
    campaign, never merge two. The default is True, so no existing caller changes behaviour.
    """
    raw = " ".join(x for x in (name, symbol) if x)
    if not raw.strip():
        return LureVerdict(False, "none", reasons=("no metadata strings",))

    # `skeleton` applies deobfuscation itself, as its final step and iterated to a fixed point, so
    # pre-deobfuscating here would be a no-op. Verified empirically over 4,011 strings (including 4,000
    # randomised mixtures of brackets, dots, Cyrillic, emoji and "dot" spellings): zero disagreements
    # between skeleton(x) and skeleton(deobfuscate(x)). An earlier version called both and took the
    # longer result, on the mistaken belief that a scheme written "hxxps[://]" needed a second pass;
    # deobfuscate only rewrites dot forms, so it never did.
    text = skeleton(raw) if normalise else raw.casefold()

    suspect, benign, aggregators = extract_domains(text)
    verbs = extract_verbs(text)
    has_scheme = bool(_SCHEME_RE.search(text))
    has_price = bool(_PRICE_RE.search(text))
    handles = tuple(dict.fromkeys(h.lower() for h in _TELEGRAM_RE.findall(text)))

    # An aggregator host is a real destination even though it is not a campaign identity.
    destination = bool(suspect) or bool(aggregators) or has_scheme
    call_to_action = bool(verbs) or bool(handles) or has_price

    reasons: list[str] = []
    if suspect:
        reasons.append(f"suspect domain(s): {', '.join(suspect)}")
    if aggregators:
        reasons.append(f"aggregator host(s): {', '.join(aggregators)}")
    if has_scheme:
        reasons.append("explicit URL scheme")
    if verbs:
        reasons.append(f"imperative verb(s): {', '.join(verbs)}")
    if handles:
        reasons.append(f"telegram handle(s): {', '.join(handles)}")
    if has_price:
        reasons.append("price pattern")
    if benign:
        reasons.append(f"allowlisted domain(s) ignored: {', '.join(benign)}")

    if destination and call_to_action:
        strength = "strong"
    elif destination or call_to_action:
        strength = "weak"
    else:
        strength = "none"

    return LureVerdict(
        is_lure=(strength == "strong"),
        strength=strength,
        domains=suspect,
        verbs=verbs,
        has_scheme=has_scheme,
        has_price=has_price,
        telegram_handles=handles,
        benign_domains=benign,
        aggregator_hosts=aggregators,
        normalised=text,
        reasons=tuple(reasons),
    )


def campaign_key(verdict: LureVerdict) -> str | None:
    """The clustering key that identifies a campaign, in strict precedence order.

    1. A suspect registrable domain -- the campaign's own infrastructure.
    2. A messaging handle, keyed as "t.me/<handle>" -- the handle IS the identity behind a shared host.
    3. An aggregator host alone, keyed as "<host>/?" and flagged as unresolved, because a shortener with
       no recoverable target cannot be attributed to a campaign; such contracts must be reported as an
       unattributed residual rather than lumped together as one campaign.
    4. None, when there is no destination at all -- which is what keeps `weak` verbs-only matches out of
       the campaign census.

    Returns the alphabetically first domain so the key does not depend on string order within the
    metadata. Never returns a bare aggregator host as though it were a campaign: doing so would merge
    every Telegram or bit.ly campaign into a single false mega-campaign, and the same conflation already
    broke one cross-chain claim (`t.ly` counted as a shared domain between unrelated campaigns).
    """
    if verdict.domains:
        return sorted(verdict.domains)[0]
    if verdict.telegram_handles:
        return f"t.me/{sorted(verdict.telegram_handles)[0]}"
    if verdict.aggregator_hosts:
        best = sorted(verdict.aggregator_hosts)[0]
        # A host with a path keeps its slug (distinct campaigns); a bare host cannot be attributed and
        # is marked unresolved so such contracts are reported as a residual, not as one mega-campaign.
        return best if "/" in best else f"{best}/?"
    return None
