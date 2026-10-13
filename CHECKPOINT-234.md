# Ticket #234 checkpoint

## Disposition

**Keep the restoration mechanism and its scoped proof; numerical qualification and released reuse remain blocked.** No deployment or daily-profile/manifest/model/routing change occurred.

Implemented on `judge/234-prefix-branching`, frozen base `32e14d4f`, unified prerequisite integration `011e8cb1`. Implementation: `1b38d40d`; measured source: `8ebc2398a8bb6da1a13373ba1299d2a9857a7029` with empty dirty diff. Candidate binary: `/home/joshkirklin/strata-workspace/scratch/judge-234-build/strata`, SHA-256 `1aad797984efaecd58994b9237aedd343b9b4c6b78d6a4f00b8846dbe9c7762f`. The adopted control remains `5862438577303ac56ce4f385499100732f0e65ccd42540021991c3d4a7868b68`.

## Exercised evidence

- Exact CUDA 12.9.86 / GNU 14.4.0 make build, architectures `70;86`: `evidence/issue-234/build-1/result.json`. Two transient launcher refusals were retained; the granted build passed.
- Three compiled CPU seam tests passed. Python CPU suite initially passed 19/20; the test-fixture setup failure is retained, corrected, and the failed test alone passed. These are CPU software checks, not model qualification.
- All 56 GPU format/mode/prefix restoration cases passed, including deliberate retention of suffix PLE while every other field restores; the production verification rejected the omitted mutable history. Full snapshot suite: 6,360 assertions. `fixture-1/result.json`.
- Serial **GEN** only: one owned synthetic context, two distinct 1,093-token plans, one 1,056-token prefix prefill, restore before A/B/A. Six readouts retained. A replay had identical selected FP32 logits and committed GDN/PLE/KV fingerprints on all three stages. Fresh/fresh A was identical. Fresh/reused maximum probability differences: A `0.0045137332167269895`, B `0.0032865587945648533`; TV: A `0.004513733216727052`, B `0.003445352966357854`; no argmax changes. `model-1/result.json`.
- Checkpoint `291.034301 ms`; restores `241.854001`, `287.377512`, `242.715892 ms`; tokenisation `0.3554059669841081 s`; queue `0.22748470306396484 s`; reused whole-process call `52.0735808419995 s`. Model startup and observers are included: **no resident-lane performance claim**.
- Private owned MPS declarations exactly matched `[33285996544,33285996544,22548578304]`; clean `quit` exit 0 and no stderr. No clocks/power/thermal/existing-daemon changes or reset. Safety telemetry and model hash closure retained.

Plans were fixed before GPU evaluation: numerical criteria `2026-10-13T13:49:06.624615+00:00`; restoration plan `2026-10-13T13:50:08Z`, SHA-256 `186739a8c6cf9ccc1bcd70a9160e835fb7494ccb23cf9a454410cddc9cd0dbc1`.

Complete dispositions, scope, identities, commands and evidence index: `evidence/issue-234/acceptance.json`. Native C++/readout source is unchanged after the measured revision; later edits retain evidence, documentation and replay preparation only.

## Exact missing prerequisites

1. No frozen independently human-labelled source-state corpus exists (#224).
2. Contract defect: #224 specifies **200 development + 400 calibration + 600 sealed = 1,200 groups total**; #234 requires **3,000 independent groups per workload and intended mode**. Even a completed #224 corpus cannot satisfy this. The operator must reconcile the denominator/source pool; variants and siblings cannot manufacture independence.
3. Maximum-load logit dump wiring is absent. Batch-slot head rows exist: this is a capability gap, not an architectural limit. Adding the seam is outside this ticket. No parallel claim belongs here.

There are **zero qualifying campaign rows**. The synthetic restoration probes are not a shortened or relabelled paired qualification. No exact rate bound or full-campaign global drift pass is claimed. HIP/SYCL and ordinary-generation GPU parity/performance were not exercised.

## Next reproducible command

No qualification command is runnable until the source pool, denominator contract and contended seam are supplied. To repeat the completed restoration evidence from the current committed source without overwriting any retained plan, use a new output identity:

```sh
P=/home/joshkirklin/strata-workspace/toolchains/mm-het002/bin/python
W=/home/joshkirklin/strata-workspace/worktrees/judge-234
"$P" "$W/evidence/issue-234/run_restoration.py" prepare \
  --profile "$W/evidence/issue-234/serial-profile.json" --out "$W/evidence/issue-234/replay-2" &&
"$P" "$W/evidence/issue-234/measure.py" fixture \
  --plan-root "$W/evidence/issue-234/replay-2" --out "$W/evidence/issue-234/replay-2-fixture" &&
"$P" "$W/evidence/issue-234/measure.py" model \
  --plan-root "$W/evidence/issue-234/replay-2" --out "$W/evidence/issue-234/replay-2-model"
```

The helper uses `window.py run --class measure --ticket 234 --max-s 1800`, the launcher's unchanged finite unit properties, and a private MPS lifecycle. Transient exit 75 backs off 180 seconds; an operator machine hold stops. Do not edit tracked files between preparation and evaluation. Rebuild only through the exact retained `evidence/issue-234/build.py` recipe when engine source changes; choose a fresh build-log output directory.

Rollback: omit both `--score-branches` and `--score-prefix`, use the unchanged fresh #231 serial GEN reference, and retain the adopted control binary. No serving reuse mode is promoted.
