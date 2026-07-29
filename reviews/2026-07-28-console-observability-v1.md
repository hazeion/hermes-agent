# Feature Slice Review: Console observability v1

Status: Ready to publish  
Slice: `console-observability-v1`  
Date: `2026-07-28`  
Review log: `reviews/2026-07-28-console-observability-v1.md`

## Slice contract

### Goal

Provide quiet Hermes CLI runs with bounded, machine-readable progress and usage telemetry suitable for Mentat without exposing tool arguments, results, model reasoning, secrets, or private paths.

### In scope

- Versioned JSONL progress events for tool lifecycle and reasoning availability.
- Versioned usage report output for quiet chat and oneshot execution.
- Context-window measurement propagation into the final usage record.
- Strict server-owned output mode that fails closed on unsafe or unsupported filesystems.
- CLI arguments, operator documentation, compatibility tests, and Windows-safe behavior.

### Out of scope

- Remote telemetry transport, analytics upload, raw reasoning text, tool payload capture, UI rendering, or provider/model switching.

### Acceptance criteria

| ID | Observable criterion | Evidence | Status |
| --- | --- | --- | --- |
| AC-1 | Quiet CLI and oneshot runs can emit a bounded version-one usage record. | CLI and oneshot tests | Pass |
| AC-2 | Progress output contains only fixed summaries and validated tool names, never arguments, results, reasoning text, secrets, or private paths. | Structured telemetry safety tests | Pass |
| AC-3 | Strict output writes only to an existing server-owned regular file with restrictive permissions and fails closed elsewhere. | File, symlink, ownership, and platform tests | Pass |
| AC-4 | Context-token usage resets per turn and reaches the final usage record. | Context compressor and finalizer tests | Pass |
| AC-5 | The feature is documented and remains compatible across supported platforms. | Documentation inspection and Windows-footgun check | Pass |

### Constraints and recovery

- Safety: no raw content, tool payloads, secrets, or private paths in telemetry; strict mode must fail closed.
- Compatibility: optional flags only; default CLI behavior remains unchanged; Windows import and execution must not reference POSIX-only APIs unguarded.
- Rendered behavior: not applicable; this slice is CLI/file output only.
- Rollback or recovery: disable the optional output flags or revert the single feature commit.
- Documentation targets: `docs/observability/structured-cli-telemetry.md`.
- Version-control strategy: rebase the console feature commit onto current `main`, force-push with lease, retarget the existing ready PR to `main`, and merge only after review and CI.

### Scope discussion and approval

- Recommendation and rationale: land provider/model switching first, then rebase this independent console slice onto the resulting `main`.
- Alternatives considered: stack on the old integration branch (rejected because that branch is now merged); combine with later Mentat UI work (rejected as a broader slice).
- User decisions: user explicitly requested dependency-order commits, merges/rebases as needed, then continued Road to Beta work.
- Process exception: the user previously granted standing approval for slice progression and two-review fixes, and explicitly authorized publication and merge in this goal.
- Approved at: 2026-07-28.

## Test strategy

| Acceptance criterion | Pre-implementation gap | Planned test or evidence | What it proves | Limitations |
| --- | --- | --- | --- | --- |
| AC-1 | No bounded quiet-run usage channel. | CLI and oneshot usage-file tests. | Output wiring and schema. | Does not call a live provider. |
| AC-2 | No safe progress channel. | Structured event allowlist and data-exclusion tests. | Only fixed safe fields are serialized. | File readers remain responsible for access control. |
| AC-3 | No strict trusted-file writer. | Permission, symlink, unavailable-dir-fd, and missing-POSIX-UID tests. | Unsafe targets and unsupported platforms fail closed. | Windows behavior is emulated for the missing UID API. |
| AC-4 | Context usage could retain prior-turn values. | Turn-reset and finalizer tests. | Current-turn context values reach reporting. | Provider-specific metadata varies. |
| AC-5 | Cross-platform footgun in strict writer. | Repository Windows-footgun scanner. | No unguarded POSIX-only call remains. | Not a native Windows runtime test. |

### Baseline results

| Command or action | Environment | Result | Notes |
| --- | --- | --- | --- |
| Existing PR CI | GitHub Actions, old integration base | Partial | All test shards passed; Windows-footgun check found an unguarded `os.geteuid` reference. |

### Test discussion and approval

- User questions and decisions: user directed the feature to be completed and merged after provider/model switching.
- Accepted coverage gaps: no live-provider call and no native Windows runner in the focused local pass.
- Approved at: 2026-07-28.

## Implementation record

### Changes

- Added bounded structured telemetry writer, CLI/oneshot integration, usage/context propagation, documentation, and tests.
- Rebased only the console feature commit onto the provider/model-enabled `main`.
- Replaced direct `os.geteuid()` usage with a callable capability check and added a fail-closed missing-UID test.

### Deviations and decisions

- The original PR targeted `mentat-beta-contracts`; after that branch merged, the PR must target `main`.

## Verification

### Focused checks

| Command or action | Environment | Exit/result | Pass/fail/skip counts | Notes or artifacts |
| --- | --- | --- | --- | --- |
| Focused console, oneshot, CLI, and context tests | macOS local checkout | Pass | 224 passed | Current rebased branch after safety fix. |
| `scripts/check-windows-footguns.py --all` | macOS local checkout | Pass | 803 files scanned | No Windows footguns found. |
| Python compile and `git diff --check` | macOS local checkout | Pass | n/a | Clean static checks. |

### Full suite

The corrected project-local environment completed the full suite:
45,065 passed and 24 failed across 12 files, with one additional collection
timeout. None of the failing files is modified by this feature. The feature's
`tests/hermes_cli/test_structured_telemetry.py` module passed inside the full
run.

The failures are recorded as unrelated `origin/main` baseline limitations on
this macOS host. They include Keychain/subprocess mock interference, `/tmp`
versus `/private/tmp` path aliases, Linux abstract sockets, WSL and service
manager assumptions, host signal timing, provider environment resolution, live
system guards, and host-binary execution behavior. The runner also reported two
files that failed once and passed on retry. GitHub's Linux/Windows matrix remains
the publication gate for the supported cross-platform environments.

### Rendered or manual behavior

- Not applicable.

## Adversarial review

### Review cycle 1

- Compatibility/product reviewer: clear; optional flags preserve default CLI
  behavior and strict output fails closed when the POSIX UID capability is
  unavailable.
- Safety/correctness reviewer: blocked publication because `O_NOFOLLOW` and
  `O_DIRECTORY` fell back to zero, allowing a partially capable platform to
  enter strict mode without the documented protections.

### Fix and retest

- Strict mode now requires nonzero integer `O_NOFOLLOW` and `O_DIRECTORY`
  capabilities and uses the validated values for both opens.
- Added a regression test that removes each flag independently and verifies
  that the channel writes nothing.
- Revised focused result: 224 passed. Windows-footgun scan and static checks
  remain green.

### Review cycle 2

- Safety/correctness reviewer: clear; the original P1 is fully resolved and no
  new blocking issue was found.
- Compatibility/product reviewer: clear; the corrected fail-closed capability
  gate introduces no compatibility or product regression.

### Publication CI repair

- GitHub's contributor-attribution gate identified the original feature
  commit's older local author email as unmapped.
- Added the required one-file email-to-GitHub-user mapping without changing
  commit authorship or feature behavior.
- Contributor-map verification: 12 passed; static diff checks pass.
- Final narrow safety/correctness re-review: clear; mapping is exact and changes
  no executable behavior or authorship.
- Final narrow compatibility/product re-review: clear; mapping is
  attribution-only and compatibility-neutral.

## Documentation updates

- Roadmap: pending outcome reconciliation.
- Changelog: no separate changelog entry identified.
- Architecture/operator docs: `docs/observability/structured-cli-telemetry.md`.
- Project/session notes: this review log.
- Documentation verification: pending CI/docs checks.

## Publication gate

- Proposed files: the 13 feature files plus this review log and the Windows compatibility test/fix.
- Branch and base: `codex/console-observability-v1` → `main`.
- Commit message: `fix(telemetry): make strict writer Windows-safe`.
- PR title: `Add structured quiet-run telemetry for Mentat`.
- PR summary: bounded private progress and usage output for quiet Hermes runs.
- Unresolved risks: the local macOS full-suite baseline failures described
  above; GitHub's supported-platform matrix is the publication gate.
- User authorization and scope: explicit commit/rebase/push/merge authorization in the active goal.
- Commit hash: pending.
- Ready PR URL: existing pull request; base retarget pending.

## Outcome review

- Classification: Ready to publish.
- Acceptance criteria summary: all feature acceptance criteria pass; both
  independent re-reviews are clear.
- Potential bugs or untested paths: native Windows strict mode and live-provider usage reporting.
- Remaining reviewer dissent: none.
- Compatibility/migration/rollback concerns: optional feature; disable flags or revert the feature commit.
- User decision: publication and merge were explicitly authorized in the active
  goal.
- Next slice authorized: Yes; continue the ordered Road to Beta milestones after
  this slice is merged.
