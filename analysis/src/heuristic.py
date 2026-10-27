"""Heuristic candidate-generation for address-poisoning detection on rollups.

Role in the pipeline (docs/PAPER_PLAN.md): this is the HIGH-RECALL first stage. It
emits candidate poisoning transfers cheaply; a downstream ML arbiter refines the
ambiguous FP/FN band. Keep it recall-oriented and interpretable — precision is the
ML's job.

New, self-contained implementation (not ported from prior papers). Grounded in the
attack characteristics established in the group's prior work and in Guan & Li (2024)
and Tsuchiya et al. (2025):
  - Fake-type transfer (FTT): a counterfeit ERC-20 whose name/symbol impersonates a
    legitimate token (contract NOT in the ground-truth legit set).
  - Zero-type transfer (ZTT): a zero-value transfer of a LEGITIMATE token (the
    zero-value `transferFrom` exploit).
  - Lookalike: the transfer's counterparty address is prefix/suffix-similar to a real
    prior counterparty of the victim.
Dust-type transfers are intentionally out of scope (consistent with the prior IEEE
Access paper — low volume, high noise).

Everything here is pure and unit-tested on synthetic positives AND negatives; a
DataFrame/parquet adapter over real data (data/processed/) is a later step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

# ERC-20 Transfer(address,address,uint256) event signature hash.
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ADDRESS_HEX_LEN = 40  # 20 bytes, no 0x
ZERO_ADDRESS = "0x" + "0" * ADDRESS_HEX_LEN  # mint/burn counterparty — exclude from lookalike checks


# --------------------------------------------------------------------------- #
# Address similarity (also the seed for the learnable prefix/suffix weighting)
# --------------------------------------------------------------------------- #
def normalize_address(addr: str) -> str:
    """Lowercase, strip 0x. Raises on a malformed 20-byte address."""
    a = addr.lower()
    if a.startswith("0x"):
        a = a[2:]
    if len(a) != ADDRESS_HEX_LEN or any(c not in "0123456789abcdef" for c in a):
        raise ValueError(f"not a 20-byte hex address: {addr!r}")
    return a


@dataclass(frozen=True)
class SimilarityWeighting:
    """Weights on matched prefix/suffix nibbles. Symmetric by default; the ML/tuning
    stage learns the optimal weighting (the salvaged Topic-2 contribution)."""

    prefix_weight: float = 1.0
    suffix_weight: float = 1.0


@dataclass(frozen=True)
class Similarity:
    prefix_len: int  # matched leading hex nibbles
    suffix_len: int  # matched trailing hex nibbles

    def score(self, weighting: SimilarityWeighting = SimilarityWeighting()) -> float:
        return self.prefix_len * weighting.prefix_weight + self.suffix_len * weighting.suffix_weight


def address_similarity(a: str, b: str) -> Similarity:
    """Matched prefix/suffix nibble counts between two addresses.

    Prefix and suffix are counted independently but never allowed to overlap past the
    address length (identical addresses report 40/0, not 40/40 double-counting)."""
    na, nb = normalize_address(a), normalize_address(b)
    prefix = 0
    for ca, cb in zip(na, nb):
        if ca != cb:
            break
        prefix += 1
    if prefix == ADDRESS_HEX_LEN:  # identical
        return Similarity(prefix_len=ADDRESS_HEX_LEN, suffix_len=0)
    suffix = 0
    # cap suffix so prefix+suffix <= length (non-overlapping)
    max_suffix = ADDRESS_HEX_LEN - prefix
    for ca, cb in zip(reversed(na), reversed(nb)):
        if ca != cb or suffix >= max_suffix:
            break
        suffix += 1
    return Similarity(prefix_len=prefix, suffix_len=suffix)


@dataclass(frozen=True)
class LookalikeThresholds:
    """Minimum matched nibbles for a genuine lookalike. Default 4/4 = the labelling
    seed from docs/LABELLING_PROTOCOL.md (prior work used an asymmetric 3/4)."""

    min_prefix: int = 4
    min_suffix: int = 4


def is_lookalike(a: str, b: str, thresholds: LookalikeThresholds = LookalikeThresholds()) -> bool:
    """True if b resembles a but is a DIFFERENT address (a real lookalike, not a: itself)."""
    if normalize_address(a) == normalize_address(b):
        return False
    sim = address_similarity(a, b)
    return sim.prefix_len >= thresholds.min_prefix and sim.suffix_len >= thresholds.min_suffix


def best_lookalike(
    target: str, candidates: Iterable[str], thresholds: LookalikeThresholds = LookalikeThresholds()
) -> tuple[str, Similarity] | None:
    """Among candidates, the one most resembling target above thresholds (by score), or None."""
    best: tuple[str, Similarity] | None = None
    for c in candidates:
        if is_lookalike(target, c, thresholds):
            sim = address_similarity(target, c)
            if best is None or sim.score() > best[1].score():
                best = (c, sim)
    return best


# --------------------------------------------------------------------------- #
# Transfer records, token registry, classification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Transfer:
    chain: str
    tx_hash: str
    log_index: int
    token: str      # token contract address
    sender: str
    receiver: str
    amount: int     # raw ERC-20 units
    timestamp: int  # unix seconds
    block: int


@dataclass(frozen=True)
class TokenInfo:
    address: str
    name: str
    symbol: str


@dataclass(frozen=True)
class TokenRegistry:
    """Ground truth for one chain. legit_addresses = verified canonical contracts;
    legit_names/legit_symbols = their names/symbols (case-folded) for impersonation
    checks; token_info = optional metadata for arbitrary (incl. fake) contracts."""

    legit_addresses: frozenset[str]
    legit_names: frozenset[str] = frozenset()
    legit_symbols: frozenset[str] = frozenset()
    token_info: Mapping[str, TokenInfo] = field(default_factory=dict)

    def is_legit(self, token: str) -> bool:
        return normalize_address(token) in self.legit_addresses


class TransferType(str, Enum):
    FAKE = "fake"  # FTT
    ZERO = "zero"  # ZTT


def classify_transfer(t: Transfer, registry: TokenRegistry) -> TransferType | None:
    """Suspicious-type classification (recall-oriented). None = not a candidate type."""
    if registry.is_legit(t.token):
        # Legitimate token: only a zero-value transfer is suspicious (ZTT).
        return TransferType.ZERO if t.amount == 0 else None
    # Non-legit token: suspicious if it impersonates a legit token's name/symbol (FTT).
    info = registry.token_info.get(normalize_address(t.token))
    if info is None:
        return None
    if info.name.casefold() in registry.legit_names or info.symbol.casefold() in registry.legit_symbols:
        return TransferType.FAKE
    return None


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    transfer: Transfer
    transfer_type: TransferType
    victim: str            # party whose history is poisoned (sender, zero/fake orientation)
    lookalike: str         # crafted counterparty (receiver)
    matched_counterparty: str  # the real prior counterparty the lookalike imitates
    similarity: Similarity


def generate_candidates(
    transfers: Iterable[Transfer],
    registry: TokenRegistry,
    victim_history: Mapping[str, Iterable[str]],
    thresholds: LookalikeThresholds = LookalikeThresholds(),
) -> list[Candidate]:
    """Emit candidate poisoning transfers.

    For each suspicious-type transfer (ZTT/FTT) we take the zero/fake orientation
    (victim = sender, lookalike = receiver, per prior work) and check whether the
    receiver is a lookalike of any *real* prior counterparty of the sender
    (victim_history[sender]). High recall by design; the ML arbiter prunes FPs.

    victim_history maps an address -> its known legitimate counterparties (built from
    the transfer stream in a separate pass; injected here to keep this pure/testable).
    """
    candidates: list[Candidate] = []
    for t in transfers:
        ttype = classify_transfer(t, registry)
        if ttype is None:
            continue
        history = victim_history.get(normalize_address(t.sender)) or victim_history.get(t.sender) or ()
        match = best_lookalike(t.receiver, history, thresholds)
        if match is None:
            continue
        matched_addr, sim = match
        candidates.append(
            Candidate(
                transfer=t,
                transfer_type=ttype,
                victim=t.sender,
                lookalike=t.receiver,
                matched_counterparty=matched_addr,
                similarity=sim,
            )
        )
    return candidates


# --------------------------------------------------------------------------- #
# Log decoding helpers (bridge HyperSync/3xpl parquet rows -> Transfer)
# --------------------------------------------------------------------------- #
def address_from_topic(topic_hex: str) -> str:
    """An address stored in an indexed topic is a 32-byte left-padded value: take the
    last 40 hex nibbles and re-prefix."""
    h = topic_hex.lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) != 64:
        raise ValueError(f"topic is not 32 bytes: {topic_hex!r}")
    return "0x" + h[-ADDRESS_HEX_LEN:]


def amount_from_data(data_hex: str) -> int:
    """ERC-20 Transfer non-indexed `value` (uint256) from the log data field."""
    h = data_hex.lower()
    if h.startswith("0x"):
        h = h[2:]
    return int(h, 16) if h else 0
