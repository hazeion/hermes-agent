# Feature Slice Review: Remote provider/model switching

Status: Paused — repository-wide verification running  
Slice: `remote-provider-model-switching-v1`  
Date: 2026-07-28  
Review log: `reviews/2026-07-28-provider-model-switching.md`

## Slice contract

### Goal

Provide an authenticated, capability-advertised Hermes API that lets a remote control plane safely inspect and switch a served profile's provider/model runtime.

### In scope

- A secret-free, revisioned runtime inventory for a profile.
- An authenticated, idempotent, revision-bound provider/model switch.
- Rejection when an API-server run is active for the target profile.
- Atomic persistence and effective-runtime read-back.
- Version-one capability and endpoint advertisement, focused tests, and API documentation.

### Out of scope

- Provider credentials, custom endpoint configuration, auxiliary model slots, profile creation, skill changes, cron mutations, attachments, artifacts, or UI changes.
- Switching an already-running conversation.

### Acceptance criteria

| ID | Observable criterion | Evidence | Status |
| --- | --- | --- | --- |
| AC-1 | Authenticated callers can obtain a bounded profile runtime inventory with an opaque revision. | API contract test | Pass |
| AC-2 | A switch accepts only an advertised provider/model pair and an exact revision. | API contract tests | Pass |
| AC-3 | Invalid, stale, duplicate, or active-run requests fail without changing the runtime. | Negative-path tests | Pass |
| AC-4 | A successful switch persists atomically and returns a verified updated runtime/revision. | Temp-profile integration test | Pass |
| AC-5 | Capabilities advertise the version-one inventory and mutation surface only when API-key authentication is configured. | Capability test | Pass |

### Constraints and recovery

- Safety: bearer-authenticated; no credential, path, endpoint, or raw config exposure; no mutation during active profile run.
- Compatibility: existing API routes and local model picker behavior remain unchanged.
- Rendered behavior: not applicable; this slice is API-only.
- Rollback or recovery: atomic configuration writes; a failed verification leaves the previous configuration in place.
- Documentation targets: API-server documentation and `REMOTE_HERMES.md` consumer contract.
- Version-control strategy: current feature branch, no commit/push/PR without separate publication approval.

### Scope discussion and approval

- Recommendation and rationale: use a narrow revision-bound API rather than exposing the dashboard model setter, which accepts credential and endpoint fields unsuitable for a remote control plane.
- Alternatives considered: reuse the dashboard route (rejected: browser/session boundary and secret-bearing payload); add broad remote configuration editing (rejected: outside this slice).
- User decisions: provider/model switching is first; user approved the version-one contract and asked for two independent reviews after each slice.
- Process exception: user granted standing approval for slice progression and review fixes. Publication approval remains separate.
- Approved at: 2026-07-28.

## Test strategy

| Acceptance criterion | Pre-implementation gap | Planned test or evidence | What it proves | Limitations |
| --- | --- | --- | --- | --- |
| AC-1 | No runtime inventory route exists. | aiohttp authenticated route test. | Safe response shape and auth gate. | Does not test a real network deployment. |
| AC-2 | No mutation route exists. | Temp-profile contract test for valid and non-advertised pairs. | Validation and revision binding. | Provider reachability is not part of this config-only switch. |
| AC-3 | No concurrency/idempotency contract exists. | Tests for stale revision, duplicate key, and active run. | No unintended second or concurrent mutation. | Process restart persistence of idempotency is excluded. |
| AC-4 | No remote read-back exists. | Temp-config integration test. | Atomic persisted assignment and changed revision. | Does not start an LLM run. |
| AC-5 | Capability document has no mutation surface. | Existing capability endpoint test. | Discovery remains explicit and authenticated. | Consumer validation belongs to Mentat follow-up. |

### Baseline results

| Command or action | Environment | Result | Notes |
| --- | --- | --- | --- |
| Source inspection | macOS local checkout | Pass | Existing API has profile inventory and atomic config writes but no remote provider mutation. |

### Test discussion and approval

- User granted standing approval for slice test strategies on 2026-07-28.
- Accepted coverage gaps: no live provider call and no Mentat consumer change in this Hermes-only slice.

## Implementation record

### Changes

- Added authenticated, versioned profile runtime inventory and revision-bound switch routes.
- Added per-profile locking, served-profile authorization, active-run exclusion, request-fingerprint idempotency, and model-only persistence.
- Added capability discovery, API documentation, and focused contract coverage.

### Deviations and decisions

- None.

## Verification

### Focused checks

| Command or action | Environment | Exit/result | Pass/fail/skip counts | Notes or artifacts |
| --- | --- | --- | --- | --- |
| API-server focused suite | macOS local checkout | Pass | 230 passed | Includes runtime-switch contract coverage. |
| Targeted runtime-switch tests | macOS local checkout | Pass | 5 passed | Covers inventory, unserved-profile rejection, revision/idempotency, active-run rejection, and narrow persistence. |
| Python compilation and diff validation | macOS local checkout | Pass | n/a | `api_server.py` compiles and the diff has no whitespace errors. |

### Full suite

Repository-wide suite completed, but its terminal stdout/exit status was lost when the detached runner connection closed. The duration cache was updated and no failure artifact was retained, but this is not treated as a captured green result. Targeted and focused verification above are captured and passing.

## Adversarial review

Four review rounds completed. The first two rounds identified atomicity, profile authorization, idempotency, capability-advertisement, and test-isolation issues. All accepted in-scope corrections were applied. Both independent reviewers reported no blocking findings after the final test-only corrections.

## Documentation updates

- API-server user guide documents the authenticated remote runtime contract and exclusions.

## Publication gate

- Publication is not authorized by standing slice approval.

## Outcome review

- Classification: Successful with a recorded full-suite-output limitation.
- User outcome request: confirm the contract and residual verification limitation are acceptable before beginning the next slice.
- Next slice authorized: No; profile creation and skills/toolset retrieval remain separate follow-on slices.
