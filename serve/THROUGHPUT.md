# Concurrent throughput metrics

Set `STRATA_THROUGHPUT=1` for the native process to enable exact prefill progress.
Without it, native progress output is byte-compatible and aggregate prefill stays
unknown. The frontend can still count committed decode tokens.

`GET /metrics` adds `throughput` alongside the unchanged legacy `live` object:

```json
{"window_s":5.0,"decode_tok_s":42.0,"prefill_tok_s":1200.0}
```

These are sums of native committed T/BT tokens and exact fresh prompt progress
in a five-second wall-clock window, with 100 ms bookkeeping buckets. They are
independent of the newest request's reading/generating label. No request text,
IDs or token values are retained by the counters. No GPU polling is added.

The native PP line appends cumulative fresh tokens for the current read leg as
its sixth field. The previous five fields retain their meaning. The native
reader advertises `INFO throughput_progress=1` before READY and again before RESUME. This lets unused replicas contribute a known zero instead of making the aggregate unavailable. Unlike the prompt
position, this count excludes reused prefixes and includes actual re-reading.
An INFO fresh_prompt_tokens counter reports committed tails at the prompt boundary
and the final prompt input at the first committed output. DONE's legacy resume-based
total is deliberately ignored, so re-reading and completion delays cannot distort it.
Cancellation drain lines are observed by the pump too. BDONE and speculative
proposals do not add tokens; BT and T represent native committed tokens.

Progress arrives at chunk boundaries; throughput reflects observed committed
work, not continuously sampled GPU execution. The common denominator stays five
seconds even immediately after startup. At idle it decays to zero as samples
expire. A native restart creates fresh counters and an old pump never writes a
successor's counters. Before capability is advertised, prefill is null; an old
PP format or inconsistent cumulative count also disables it. Replica sums are
unknown if any constituent's metric is unknown.

CPU fixture coverage: `python -m unittest serve.test_throughput`, plus the normal
server and replica suites. Real-engine execution and installation are separate
qualification steps; this source change does not install a service or model.

Idle services use a two-second window; active services retain five seconds. Reading idle metrics does not mutate the active five-second observation window. Dashboard display smoothing resets on resumption. Missing prefill support remains null.

With STRATA_THROUGHPUT=1, metrics also includes request_activity: a content-free
boot identity and up to 128 opaque request IDs with queued/processing/completed/failed/cancelled states.
Active records take priority; truncation is explicit. This sampled display feed
never changes admission limits and does not expose prompt, output or error text.
Queued includes admitted preparation; processing follows the native execution flag.
Terminal outcomes follow handler completion, cancellation and recorded errors.

## Native retained-prompt admission

The frontend selects native admission only after this incarnation reports
`INFO retained_prompt=1 retained_prompt_version=1 native_admission=1 native_admission_version=1`
and EPLAN resolves `admission.authority` to `native` with `support: "on"`.
Legacy and packed-only engines keep their existing admission paths.

The total unfinished HTTP generation allowance remains effective slots plus
`request_queue_limit`, including a zero queue depth. Accepted requests register
their logical tags before rendering/tokenization; a lazy engine registers its
already accepted requests in original ingress order when its capability arrives.
The same rule applies after unload: stale startup capability cannot register
against a dead process. Data-parallel replicas with native admission are refused,
not routed through a replica-zero registration.
The frontend retains bounded preparation permits, but neither those permits nor
the speculative store-offer frontier select native seats. Complete validated
input is published once with slot `-1`; the same tagged output owner awaits BACK
before any generation SSE headers or dialect preamble are emitted. A queued
timeout returns HTTP 408 without Retry-After. Validation failures, shortcuts,
disconnects and shutdown also cancel/drain early registrations.
An admission failure after headers (including MCP continuation preparation)
uses that dialect's SSE error/termination, never another HTTP status line.
Cancelled/expired ingress-preparation waits remain part of final `queue_s`.

Lifecycle records and response-attribution logs expose `ingress_wait_s` separately
from actual frontend preparation. `native_admission_wait_s` comes only from BACK
or BQDONE. `native_service_wait_s` is published as an explicit initial zero on
BQACCEPT (before input or seating), confirmed on BACK, updated cumulatively on
BQPP, and finalized on BQDONE. Missing service receipts stay missing/flagged;
zero is not synthesized from admission.
Neither wait counter is inferred from seat-to-first-token time, which includes real restore/prefill work. The
native admission counter includes the time after early registration, so it can
overlap frontend preparation. `queue_s` and its compatibility alias
`admission_wait_s` both mean frontend ingress-resource wait plus the observed
input-publication-to-seat-receipt interval; they exclude preparation and native
service wait. The second interval includes command dispatch and receipt delivery:
it can be positive even when `native_admission_wait_s` is zero. Do not subtract
the native counters from this frontend-clock aggregate or relabel it as native
wait. The frontend `native_active` phase is a receipt-fenced seated interval,
not a partition of native compute and service wait.

Retained prompt scheduling rechecks decode readiness once at the fully drained
boundary, after any final-token verification and mandatory MTP catch-up have
activated new decoders. Cancellation is consumed before the existing single
decode round. Prompt rotation, priority fairness and budget charging are unchanged;
this does not add a suspension point or an independent reader.

`retained_events` contains the actual received BQACCEPT/BACK/BQPP/BQSUSPEND/
BQRESUMED/BQDONE/BQREJECT frames and sent BQCANCEL controls, with monotonic receipt
times, raw wire text, direction and tagged slot/request/incarnation identities.
The history is capped at 4096 events; `retained_events_dropped` and
`retained_events_complete` make truncation or missing terminal/acceptance receipts
explicit. `native_wait_receipts_missing` separately counts missing native wait
counters; unavailable counters remain absent. These observables work without
optional phase diagnostics. `metrics.engine.protocol_startup` preserves only the
raw INFO/EPLAN/READY frames actually received for this incarnation. Each owned
engine process emits exactly one `engine_exit` log after reaping, with its actual
PID and exit code; a forced ending is never replaced by an invented clean exit.
