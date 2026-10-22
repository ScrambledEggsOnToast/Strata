# The interleaved small-batch matvec (STRATA_MMVQ_IL): how to measure a table for your card

The decode and verify windows run their dense projections (2 to 4 tokens at a time) through `native_mmvq_il`, which reads an
interleaved copy of the q8_1 activations. Every choice it makes gives the same bits as the plain kernels
(`STRATA_MMVQ_IL=0` turns the whole path off); only the speed differs. How many matrix rows one warp takes
(1, 2 or 4, or 0 = use the plain kernel) is a table in `src/kernels/cuda/native_mmvq.cu`, indexed by the
quantisation type (IQ4_XS, Q4_K, Q5_K, Q6_K), the token count (2-4) and the matrix's row count (five classes).

Which table a card gets:

| card | table |
|---|---|
| sm_86 (Ampere consumer) | `kIlRows_86`, the dedicated dispatch table, including shared-table entries and separately measured Q3_K cells |
| sm_120 (Blackwell consumer) | `kIlRowsShared`, measured on an RTX 3060 and an RTX 5070 together (a cell is non-zero only if the worse of the two cards still gains 3%) |
| sm_89 (Ada) and every other sm_80+ card | `kIlRowsShared` too. Nobody has measured sm_89 yet (we have no such card). |
| sm_70 (Volta) with `STRATA_SM70_TABLE=1` | the already-shipped `kIlRows_70`, measured on a V100-SXM2-32GB and registered as cc 70 in `kIlArch` |
| sm_70 without that flag, and other cards below sm_80 | the path is off; the plain kernels run |
| a card listed in `kIlArch` | its own table |

The Volta table is compiled into the engine; enabling it does not generate, download or load a table file.
Its rows are `23/12/13/14` (IQ4_XS/Q4_K/Q5_K/Q6_K), with three token counts and five output-row classes:
below 2048, 4096, 8192, 12288, and 12288 or more. All its below-2048 cells are zero, retaining the plain
key/value projections. A new local timing table is not required to try this existing route. Do not set
`STRATA_MMVQ_IL_ROWS` for that comparison: its override applies to every card, including sm_86.

`STRATA_SM70_TABLE` also selects resident expert mode 8 and the latency-hidden hyperconnection norm/up
on sm_70. It does not change sm_86's matvec table, default expert mode 0 or ordinary norm/up.
Restart with `STRATA_SM70_TABLE=0` for the established Volta path; unset the independent
`STRATA_EXP_MODE`, `STRATA_GR_FAST` and table overrides for a clean bundle comparison.

## Opt-in wide verifier-head splitting

`STRATA_VERIFY_HEAD_SPLIT_IL=1` tries the measured Q6_K output head on SM86 at
2560 input features and 248320 vocabulary rows. Six verifier rows split into
3+3 columns, seven into 4+3, and eight into 4+4. This reuses the existing Q8_1
activation image and four-column interleaved scratch, without changing the
per-column arithmetic or adding workspace. The default is off.

The option respects `STRATA_MMVQ_IL=0` and requires both sub-widths to be eligible
in the actual device table. Other types, shapes and devices keep their previous
dispatch. In particular, this does not globally force a table entry: the SM86
Q6_K two-column vocabulary cell remains plain, while three and four columns use IL.

A bounded `VERIFY_HEAD_SPLIT_IL completed=1` receipt identifies the first matching
batch graph per verifier that actually completed its existing drains and checks.
Eager non-batch windows can use the optimization without emitting that batch
receipt. Compare real service off/on before retaining a profile change; a faster
isolated projection does not establish a whole-model speedup.


## Checking enablement before timing

On an otherwise idle CUDA device, run `STRATA_SM70_TABLE=1 build/mmvq_il_parity` and then the same
command with `=0`. The fixture checks the actual architecture gate and reports interleaved/fallback
launch counts, so equal fallback bits are not mistaken for exercised Volta kernels. It covers T 2-4,
all rows choices, odd output-row tails in each supported format, table-zero shapes, unsupported
T 1/5/8 and a Q8_0 fallback. Repeat both processes on sm_86; its dispatch must be identical.

The bitwise contract is exact-layout, same-quantized-activation parity, not independent model accuracy.
Use `native_expert_bench ... 0 8` to compare both the q8_1 handoff bytes and finite down outputs for actual
expert formats, and `fused_gr_bench 1 1 8` for norm/up and its intermediates. Then compare all committed
full-vocabulary rows and emitted token IDs at the same model prefixes. Source review does not substitute
for those GPU/operator and real-model checks.

For in-engine attribution, `STRATA_VERIFY_NODES=1` records full captured kernel names, grids, blocks and
shared bytes with the verifier/device/layer-range/T/doorbell key. Match that inventory to a
`VERIFY_REPLAY ... completed=1` record from the existing serial or split-window completion join.
An inventory alone is not proof that a graph ran. These diagnostic records add no GPU events or
synchronization and are not an external kernel timeline; keep them out of normal speed comparisons.


## Measuring sm_89 (or any other card)

1. Build with tests: `cmake -S . -B build -DSTRATA_BUILD_TESTS=ON` and `cmake --build build --target mmvq_il_parity`.
2. Close everything else on the GPU, then run `build/mmvq_il_parity --bench --emit-table` (a few minutes).
   It first checks bit-for-bit parity of every shape and every rows choice (it must end with `mmvq_il_parity: OK`),
   prints a `BENCH` line per shape and token count, and last a ready-to-paste block, one line per type.
3. Post the whole output on the issue. A maintainer pastes the block as `kIlRows_89[]` and adds
   `{89, kIlRows_89, sizeof(kIlRows_89) / sizeof(IlRows)}` to `kIlArch`.
4. To try a table without rebuilding, set `STRATA_MMVQ_IL_ROWS="type:ncols:r0,r1,r2,r3,r4;..."`, for example
   `STRATA_MMVQ_IL_ROWS="23:3:0,0,2,4,4;12:2:0,1,1,1,1"` (the five numbers are the row classes below 2048, 4096, 8192,
   12288 and above; each is 0, 1, 2 or 4). Then A/B the decode: same flags, `STRATA_MMVQ_IL_ROWS` set or unset, 10
   interleaved pairs, medians.

The emit rule: per cell, the fastest rows count if it takes at least 3% off the plain kernel's time, else 0; shapes of one
class that disagree give 0; a class with no measured shape copies its neighbour below.
