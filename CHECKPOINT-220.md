# #220 checkpoint — built; awaiting an admitted numerical window

Source implementation/study commit: `c7bccc9a` on `1cat/220-iq-decode`, based on
`32e14d4febdd026b5f50901c8c76b5be0b128ac5`. The report is
`docs/benchmarks/1cat-220-iq-decode.json`; the complete source comparison is there.
No daily profile, weight representation or default dispatch was changed.

Candidate: original-byte IQ2_S planar shared codebook, opt-in
`STRATA_IQ2_PLANAR_BOOK=1`; arithmetic-preserving, zero persistent expert bytes,
same 8 KiB shared table. Only SM70/default grouped 2560×640 gate/up is affected.
SM86, other formats/shapes, mode8 and disabled staging retain established paths.
No MTP state, sampler, routes or buffers are modified. Off/unset is rollback.

## Produced identities and evidence

- Engine `/home/joshkirklin/strata-workspace/scratch/1cat-220-build/strata`, SHA256
  `1bedc578776d2fb8523c7998e8454848697bb7b7ea06e617a6cd155f42401bed`.
- Official-weight oracle `native_expert_parity`, SHA256
  `91f3c235e5db3cde1389307c232f8058daf2627d042af80672e097aad873457a`.
- Exact grouped/graph fixture `iq_multi_parity`, SHA256
  `d81b78eee90c86abaad01f6b0c268576d4b049ab36621006232b6a953a63d94c`.
- Successful queued build `d818e45b6a6d`, exit0,309.56s. Full output retained at
  `/home/joshkirklin/strata-workspace/scratch/1cat-220/build-output.txt`.
- CUDA12.9.86, conda GNU14.4.0 host/CXX, Release, SM70+SM86, experimental SM60,
  native experts/tests ON, external ggml3cf03257. CMakeCache/configure logs retained
  in the build directory. Preparatory failures (missing CUDA enable, missing CUDA
  host compiler, one active-launcher refusal) are retained, not passing results.
- Numerical-window attempts: initial exit75 `machine busy: build, build, build`;
  first retry after120s exit75 `machine busy: build`; second retry after120s exit75
  `machine busy: measure`. No GPU fixture ran. Two retries exhausted the worker's
  declared same-failure-mode retry limit. This is a queue-window/retry-policy
  checkpoint, **not evidence that the candidate is incompatible or slower**.

## Next reproducible command

Run the official-weight/dequantized full-FFN and exact graph tests first, before
any timing comparison. This command is the denied command, not a passing check:

```sh
/home/joshkirklin/ai-optimise/tools/host-inference/window.py run \
  --class measure --ticket 220 \
  --purpose 'Official-weight FFN and duplicate/private graph exactness screen' \
  --max-s 1800 -- bash -c '
export CUDA_VISIBLE_DEVICES=GPU-f65777f5-e0a8-44d8-7a60-e0a06e0c7d21
export LD_LIBRARY_PATH=/home/joshkirklin/strata-workspace/toolchains/cuda129/lib
export STRATA_IQ2_PLANAR_BOOK=1 STRATA_IQ_MT_MIN=1
nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=uuid,name,temperature.gpu,power.draw,power.limit,memory.free --format=csv &&
/home/joshkirklin/strata-workspace/scratch/1cat-220-build/native_expert_parity /srv/development/models/swift-1.5-flash-next-iq3-s/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf 0 1 2 17 21 47 &&
/home/joshkirklin/strata-workspace/scratch/1cat-220-build/iq_multi_parity &&
nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=uuid,name,temperature.gpu,power.draw,power.limit,memory.free --format=csv'
```

Repeat with the opt-in off; also run applicable SM86 fallback fixtures selecting
`GPU-91dcf482-34bf-83fb-ff50-8a523e2a5b6b` through the same queue. Stop on any
numerical, safety or telemetry failure. Existing float-reference activation error
is not permission for an off/on mismatch: required exact difference is zero.

## Prepared runtime arms

Scratch root `/home/joshkirklin/strata-workspace/scratch/1cat-220` contains:

- `profile-0.json`, `profile-1.json`: same candidate binary, differ only in opt-in.
  Versus daily: candidate binary,900s lifetime, prompt-cache0, no-prefill-borrow,
  scratch artifact_root, opt-in0/1. Hardware ceilings/reserve/model/frontends and
  batch/MTP configuration unchanged. No diagnostic environment in timing arms.
- `fixture-1024.json`: frozen public request, exact1024 prompt tokens, binary and
  tokenizer/frontend identities; request hash
  `db8c3fda9a49a8a29923e86acc54b46d5d199f663616c944824b0564ccee039b`.
- `profile-diag-0.json`, `profile-diag-1.json`: separate **serial** diagnostics
  remove batch/batch-groups/batch-MTP/pipeline flags because complete-committed
  logits refuse batch slots. Both enable VERIFY_NODES, routing dump and complete
  committed logits. Diagnostic times must never be used as speed evidence.
- `cohort.py`: thin two-request adapter reusing `measure.py.run_arm` for owned
  launch/telemetry/stop, retaining response IDs, usage, own timings and monotonic
  submit/receive. It refuses missing response accounting. Source-prepared only;
  not exercised. Cohort basis is committed outputs / response window, not
  count-matching batched engine log lines.

After numerical gates pass, run each command separately through `window.py run
--class measure --ticket220 --max-s1800`. Distinct create-only run/result names:

```sh
/home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/ai-optimise/tools/host-inference/measure.py --profile /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-0.json --run-id iq220-c1-off --out /home/joshkirklin/strata-workspace/scratch/1cat-220/c1-off.json --cases placement
/home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/ai-optimise/tools/host-inference/measure.py --profile /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-1.json --run-id iq220-c1-on --out /home/joshkirklin/strata-workspace/scratch/1cat-220/c1-on.json --cases placement
/home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/ai-optimise/tools/host-inference/measure.py --profile /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-diag-0.json --run-id iq220-diag-off --out /home/joshkirklin/strata-workspace/scratch/1cat-220/diag-off.json --request-json /home/joshkirklin/strata-workspace/scratch/1cat-220/fixture-1024.json
/home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/ai-optimise/tools/host-inference/measure.py --profile /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-diag-1.json --run-id iq220-diag-on --out /home/joshkirklin/strata-workspace/scratch/1cat-220/diag-on.json --request-json /home/joshkirklin/strata-workspace/scratch/1cat-220/fixture-1024.json
/home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/strata-workspace/scratch/1cat-220/cohort.py /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-0.json iq220-cohort-off /home/joshkirklin/strata-workspace/scratch/1cat-220/cohort-off.json
/home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/strata-workspace/scratch/1cat-220/cohort.py /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-1.json iq220-cohort-on /home/joshkirklin/strata-workspace/scratch/1cat-220/cohort-on.json
```

Require exact same-prefix finite committed logits and tokens, complete replay
candidate exposure, actual per-expert routed row counts (consecutive layer
routing records expose window multiplicities),24576 resident experts, unchanged
hardware envelopes and clean stop. Predeclared useful threshold:>=3% C1 committed
decode, <=2% fresh-prefill regression. A promising result additionally needs a
compact held-out confirmation. No timing, model-quality or concurrency evidence
has been exercised at this checkpoint; missing gates are recorded as unmet,
never as a successful rejection.
