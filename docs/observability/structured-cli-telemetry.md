# Structured quiet-run telemetry

Trusted local supervisors can request bounded machine-readable telemetry from
the quiet single-query CLI. For the strict channel, the supervisor must first
create both files as owner-only regular files, then pass their paths through
the Mentat environment contract:

```bash
MENTAT_HERMES_USAGE_FILE=/private/run/usage.json \
MENTAT_HERMES_PROGRESS_FILE=/private/run/progress.jsonl \
hermes chat -q "Inspect this project" -Q
```

Explicit `--usage-file` and `--progress-file` arguments remain convenience
outputs for ordinary CLI use and create their destinations when needed; they
are not the strict supervisor channel. Explicit arguments take precedence.
Older Hermes builds ignore the Mentat environment variables and continue
running normally. New Hermes consumes and removes them before agent tools can
inherit their private paths.

Strict supervisor writes require the platform's directory-descriptor and
no-follow support. Where those primitives are unavailable (including current
Windows Python builds), Hermes fails the optional channel closed and the run
continues without detailed local telemetry. Explicit `--usage-file` keeps its
existing cross-platform file-creation behavior.

The usage document has `schema_version: 1`. In addition to billing-oriented
token totals it may contain:

- `context_tokens`: the token count of the last actual model prompt;
- `context_length`: the resolved context window for the active model.

Either value can be `null` when Hermes cannot measure it reliably. Consumers
must not substitute cumulative `total_tokens` for active context usage.

The progress file is append-only JSON Lines with monotonic per-process sequence
numbers. Version 1 emits only:

- `tool.started`, with the exact validated tool identifier;
- `tool.completed`, with the exact tool identifier and optional duration;
- `reasoning.available`, with a fixed non-reconstructive summary, only when
  the provider supplied an explicit reasoning field.

This channel deliberately excludes raw reasoning, tool arguments, tool results,
assistant response text, credentials, and filesystem paths. Hermes does not
label ordinary assistant text as reasoning. Strict writes are best-effort,
bounded, refuse symlink destinations, and never change the agent run result.
Callers should place both files in private, run-scoped storage, validate their
schemas and bounds, and delete them after ingestion.
