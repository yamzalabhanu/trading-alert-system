# Trading Alert System Improvement Recommendations

## 1) Fix optional-import fallback bug in `routes.py` (high impact)
`routes.py` currently imports optional helpers inside `with suppress(Exception):` blocks and then unconditionally assigns them to `None` in a second suppressed block. This means imported functions are always overwritten and never used.

**Why this matters**
- Background scanner and daily reporter are effectively disabled.
- Shared window helper (`window_ok_now`) is never used even when available.

**Recommendation**
- Replace the two-step suppressed blocks with `try/except ImportError` patterns (or define defaults first, then conditionally overwrite on successful import).

## 2) Add webhook authentication and replay protection (high impact)
The `/webhook/tradingview` endpoint accepts requests without signature verification or a shared secret.

**Why this matters**
- Anyone who can reach the endpoint can enqueue jobs.
- Increases risk of spam alerts or malicious payloads.

**Recommendation**
- Require an HMAC header or pre-shared token in headers.
- Add timestamp + nonce checks to prevent replay.
- Return `401/403` before parsing payload when auth fails.

## 3) Move in-memory queue/LLM quota state to durable shared storage (medium-high)
Queue and quota tracking are currently process-local (asyncio queue + in-memory quota map).

**Why this matters**
- State resets on restart/deploy.
- Multi-instance deployments can process unevenly and bypass quota consistency.

**Recommendation**
- Use Redis for queueing and quota counters with TTL.
- Add backpressure metrics and dead-letter handling.

## 4) Improve dependency hygiene and reproducibility (medium)
`requirements.txt` has unpinned packages and entries that appear accidental or unnecessary (`micropip`, `router`).

**Why this matters**
- Build reproducibility is weak.
- Increased supply-chain and compatibility risk.

**Recommendation**
- Pin versions via `pip-tools`/`uv lock`.
- Remove unused dependencies.
- Add CI check for `pip-audit` and import usage.

## 5) Remove committed secrets from repository (high)
A credentials file (`gspread_creds.json`) is stored in the repo.

**Why this matters**
- High security risk if this repo is shared/backed up externally.

**Recommendation**
- Rotate exposed credentials immediately.
- Move secrets to environment/secret manager.
- Add `.gitignore` + pre-commit secret scanning (e.g., `gitleaks`, `detect-secrets`).

## 6) Add tests for parsing and webhook contracts (medium)
Critical paths (payload parsing, fallback behaviors, window enforcement) are not covered by tests in this repo.

**Why this matters**
- Easy to regress behavior in parsing and worker handoffs.

**Recommendation**
- Add unit tests for parser normalization and expiry derivation.
- Add API tests for `/health` and `/webhook/tradingview` success/failure paths.
- Add concurrency test for queue-full response behavior.

## 7) Standardize logging and observability (medium)
Current logs are useful but mostly unstructured and missing request correlation.

**Recommendation**
- Emit JSON logs with request/job IDs.
- Add counters for accepted/blocked webhooks, queue depth, processing latency, and error classes.
- Expose Prometheus metrics endpoint.

## 8) Configuration validation on startup (medium)
Environment variables are read directly without strict validation.

**Recommendation**
- Introduce a validated settings model (Pydantic settings/dataclass validation).
- Fail fast on invalid ranges (e.g., `MIN_DTE > MAX_DTE`, invalid time windows).
