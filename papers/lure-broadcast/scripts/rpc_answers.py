"""Tell a contract's answer from an endpoint's refusal, one JSON-RPC batch element at a time.

Why this exists
---------------
A batched `eth_call` returns one element per call, each carrying `result` or `error`. An `error` is
not one thing. An execution error is the contract's answer: the call ran and failed, which is the
positive result of the authorisation test and an informative "no ERC-165" in the interface probe. A
rate-limit or server error is the endpoint declining to run the call, which says nothing about the
contract. Public Base endpoints return the second kind INSIDE a batch whose other elements are
genuine: in the 2026-10-28 re-run, mainnet.base.org answered 56 of 423 calls with -32007 "25/second
request limit reached" next to 114 genuine code-3 "execution reverted" (results/verification_rerun.json).
Reading every error as a revert turned throttling into evidence of an unimplemented approval path,
and a partly throttled interface batch into manufactured denials.

The rule is an allowlist for the contract's answer: code 3, or a message naming an EVM execution
failure. Anything else -- including an error this module has never seen -- is a transport failure,
so an unrecognised error can only withhold evidence, never supply it.

Inputs:  one parsed batch element (or None when the endpoint returned no element for that id)
Outputs: RESULT | EXECUTION_ERROR | TRANSPORT
Complexity: O(1).
"""
from __future__ import annotations

RESULT = "result"
EXECUTION_ERROR = "execution_error"
TRANSPORT = "transport"

#: geth/reth report a revert that carries revert data with code 3.
EXECUTION_ERROR_CODE = 3
#: Message fragments of EVM execution failures (geth, erigon, nethermind wording). A revert without
#: data arrives as -32000 "execution reverted", so the message, not the code, identifies it.
EXECUTION_MESSAGES = ("execution reverted", "invalid opcode", "invalid jump destination",
                      "stack underflow", "out of gas", "vm execution error")


def element_kind(entry: dict | None) -> str:
    """Classify one batch element. Pure.

    RESULT          the element carries a `result` (which may still be "0x": that is the caller's
                    reading, not a transport question)
    EXECUTION_ERROR an `error` the contract produced -- code 3, or an EVM execution message
    TRANSPORT       no element, a malformed one, or any other `error` (rate limit, server error,
                    unknown code): absence of evidence
    """
    if not isinstance(entry, dict):
        return TRANSPORT
    if "error" in entry:
        err = entry["error"]
        if not isinstance(err, dict):
            return TRANSPORT
        if err.get("code") == EXECUTION_ERROR_CODE:
            return EXECUTION_ERROR
        msg = str(err.get("message") or "").lower()
        return EXECUTION_ERROR if any(m in msg for m in EXECUTION_MESSAGES) else TRANSPORT
    return RESULT if "result" in entry else TRANSPORT
