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
