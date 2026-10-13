#!/usr/bin/env python3
"""Offline #232 reconciliation; never launches an engine or changes GPU state.

Inputs are the scoped measure.py arm results. Output keeps derived payloads,
finite allowance declarations, explicit allocation high-water and residual-budget
observations distinct. Missing evidence is blocked, never a zero measurement.
"""
import argparse
import json
import re
from pathlib import Path


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def rows(lines, prefix):
    return [fields(line) for line in lines if line.startswith(prefix + ' ')]


def load_arm(path):
    result = json.loads(path.read_text())
    root = Path(result['run_root'])
    log = root / 'engine.log'
    return result, log.read_text(errors='replace').splitlines(), str(log)


def integer(row, key):
    return int(row[key])


def reconcile(control_path, candidate_path):
    control, _control_lines, control_log = load_arm(control_path)
    candidate, lines, candidate_log = load_arm(candidate_path)
    static = rows(lines, 'JUDGE_STATIC')
    allocations = rows(lines, 'JUDGE_ALLOC')
    memories = rows(lines, 'JUDGE_MEMORY')
    hosts = rows(lines, 'JUDGE_HOST')
    probes = rows(lines, 'JUDGE_PROBE')
    guards = rows(lines, 'JUDGE_REBIND_GUARDS')
    capacity = rows(lines, 'JUDGE_CAPACITY')
    plans = rows(lines, 'JUDGE_ADMISSION')
    device_plans = rows(lines, 'JUDGE_DEVICE_ADMISSION')
    workspaces = rows(lines, 'JUDGE_FIRST_USE_WORKSPACE')
    report = {
        'scope': '4096-cell serial ownership reconciliation under copied measurement configuration; not deployment or throughput qualification',
        'evidence': {'control_result': str(control_path), 'candidate_result': str(candidate_path),
                     'control_engine_log': control_log, 'candidate_engine_log': candidate_log},
        'derived_static_stage_ledger': static,
        'complete_generation_and_judge_allocation_classes': [line for line in lines if line.startswith('strata admission class:')],
        'measured_shared_gemm_conversion_storage': workspaces,
        'measured_explicit_allocation_high_water': allocations,
        'measured_host_heap_rss_high_water': hosts,
        'measured_mps_residual_observations': memories,
        'admission': {'summary': plans, 'devices': device_plans},
        'operation_canary_and_lease_records': [line for line in lines if line.startswith(('STATE_SCRATCH resource=judge ', 'JUDGE_RESTORE ', 'JUDGE_REBIND_GUARDS '))],
        'charges': {
            'derived': ['independent SessionOwner device/pinned/heap payload', '280 B device/108 B host per-stage canary',
                        'one head output/readback row', 'retained GDN/PLE/indexer/ID branch checkpoint plus 12.5% capacity margin',
                        'stage/module metadata', 'per-allocation 2 MiB granule padding', 'bounded prompt input', 'serial RoPE/page-table construction transient'],
            'measured': ['link-wrapped explicit engine CUDA allocation live/peak payload', 'link-wrapped pinned/registration live/peak payload',
                         'existing Gemm BF16 conversion storage', 'glibc heap live bytes', 'process RSS and kernel VmHWM',
                         'MPS remaining-budget deltas', 'complete launcher safety telemetry'],
            'allowances_not_measured_costs': ['additional judge_library_first_use_driver (includes deferred conversion/MMQ workspaces)',
                                           'additional judge_opaque_runtime_host',
                                           'retained generation graph/prefill-library/host-runtime allowances'],
            'shared_no_incremental_owner': ['same actual initialized Prefill/Gemm/cuBLAS scratch/workspace (not zero deferred first-use storage)',
                                            'same actual NativeHead weights/Q8_1 scratch'],
            'limitations': ['link wrapping excludes shared-library/driver internal allocations',
                            'MPS residual delta does not isolate cuBLAS from module loading or other opaque driver commitments',
                            'VmHWM covers the whole process, not one individually attributed host transient'],
        },
        'criteria': {}, 'stage_reconciliation': [],
    }
    unknown = [line for line in lines if line.startswith('strata admission class:') and ' unknown=1 ' in line]
    report['unknown_classes'] = unknown
    admitted = bool(plans and plans[-1].get('admit') == plans[-1].get('verified') == '1' and
                    plans[-1].get('unknown_classes') == '0' and not unknown and
                    len(device_plans) == len(static) and all(row.get('fits') == '1' for row in device_plans))
    report['criteria']['verified_complete_admission'] = 'keep' if admitted else 'reject' if plans else 'blocked'
    all_stages = bool(static) and len(static) == len(guards) == len(rows(lines, 'JUDGE_OWNER'))
    exact = all_stages
    priced_peak = all_stages
    for price in static:
        device = price['device']
        before = next((row for row in allocations if row.get('phase') == 'before-state-init' and row.get('device') == device), None)
        ready = [row for row in allocations if row.get('phase') == 'lane-ready' and row.get('device') == device]
        used = [row for row in allocations if row.get('phase') == 'first-use-complete' and row.get('device') == device]
        before_driver = next((row for row in memories if row.get('phase') == 'before-state-init' and row.get('device') == device), None)
        after_driver = [row for row in memories if row.get('phase') == 'first-use-complete' and row.get('device') == device]
        expected = sum(integer(price, key) for key in ('session_device', 'canary_device', 'head_output_device'))
        entry = {'device': int(device), 'derived_incremental_explicit_device_bytes': expected,
                 'derived_device_granule_padding_bytes': integer(price, 'allocator_padding_device'),
                 'declared_incremental_driver_library_allowance_bytes': integer(price, 'library_allowance')}
        stage_workspaces = [row for row in workspaces if row.get('device') == device]
        if stage_workspaces:
            entry['measured_retained_gemm_bf16_conversion_bytes'] = max(integer(row, 'bf16_conversion_bytes') for row in stage_workspaces)
        if before and ready and used and before_driver and after_driver:
            increment = integer(ready[-1], 'explicit_live_bytes') - integer(before, 'explicit_live_bytes')
            first_use_explicit = integer(used[-1], 'explicit_live_bytes') - integer(ready[-1], 'explicit_live_bytes')
            driver_increment = max(integer(row, 'driver_used_bytes') for row in after_driver) - integer(before_driver, 'driver_used_bytes')
            allowed = expected + entry['derived_device_granule_padding_bytes'] + entry['declared_incremental_driver_library_allowance_bytes']
            entry.update(measured_incremental_explicit_device_bytes=increment, measured_first_use_explicit_extra_bytes=first_use_explicit,
                         measured_max_residual_budget_increment_bytes=driver_increment,
                         declared_incremental_device_envelope_bytes=allowed,
                         matched_incremental_explicit_payload=increment == expected, within_declared_incremental_envelope=driver_increment <= allowed)
            exact = exact and increment == expected
            priced_peak = priced_peak and driver_increment <= allowed
            final_allocations = [row for row in allocations if row.get('phase') == 'generation-resumed-final' and row.get('device') == device]
            final_driver = [row for row in memories if row.get('phase') == 'generation-resumed-final' and row.get('device') == device]
            caps = candidate.get('profile_snapshot', {}).get('mps_caps_bytes', [])
            stage_plan = next((row for row in device_plans if row.get('device') == device), None)
            if final_allocations and final_driver and stage_plan and int(device) < len(caps):
                budget = integer(stage_plan, 'demand_bytes')
                ceiling = integer(stage_plan, 'ceiling_bytes')
                peak = integer(final_allocations[-1], 'explicit_high_water_bytes')
                cap = int(caps[int(device)])
                client_spent = cap - integer(final_driver[-1], 'driver_free_bytes')
                context_spent = cap - ceiling
                model_spent = client_spent - context_spent
                entry.update(full_run_explicit_device_high_water_bytes=peak,
                             measured_model_and_first_use_commitment_bytes=model_spent,
                             combined_generation_judge_admission_bytes=budget,
                             full_run_within_joint_admission=peak <= budget and model_spent <= budget)
                priced_peak = priced_peak and peak <= budget and model_spent <= budget
            else:
                entry['missing_complete_generation_high_water'] = True
                priced_peak = False
        else:
            entry['missing_allocation_or_residual_observation'] = True
            exact = priced_peak = False
        report['stage_reconciliation'].append(entry)
    before_host = next((row for row in hosts if row.get('phase') == 'before-state-init'), None)
    ready_hosts = [row for row in hosts if row.get('phase') == 'lane-ready']
    judge_hosts = [row for row in hosts if row.get('phase') == 'first-use-complete']
    generation_hosts = [row for row in hosts if row.get('phase') == 'generation-resumed-final']
    used_hosts = judge_hosts + generation_hosts
    host_exact = host_priced = False
    if before_host and ready_hosts and judge_hosts and generation_hosts and static and plans:
        derived_host = sum(integer(price, key) for price in static for key in
                           ('session_pinned_host', 'session_heap_host', 'canary_host', 'checkpoint_host', 'readback_host', 'metadata_host'))
        transient = max(integer(price, 'init_host_transient') for price in static)
        input_bytes = max(integer(price, 'cells') for price in static) * 8
        runtime_bound = integer(plans[-1], 'runtime_allowance')
        envelope = derived_host + transient + input_bytes + runtime_bound
        # The incremental envelope is for the isolated judge phase. Generation's later
        # first-use/checkpoint allocations retain their own charges in the joint ledger.
        heap_increment = max(integer(row, 'heap_live_bytes') for row in judge_hosts) - integer(before_host, 'heap_live_bytes')
        rss_increment = max(integer(row, 'rss_high_water_bytes') for row in judge_hosts) - integer(before_host, 'rss_high_water_bytes')
        expected_pinned = sum(integer(price, key) for price in static for key in ('session_pinned_host', 'canary_host'))
        pinned_increment = integer(ready_hosts[-1], 'pinned_live_bytes') - integer(before_host, 'pinned_live_bytes')
        joint_host = integer(plans[-1], 'guest_demand_bytes')
        joint_fits = (max(integer(row, 'heap_live_bytes') for row in used_hosts) <= joint_host and
                      max(integer(row, 'rss_high_water_bytes') for row in used_hosts) <= joint_host)
        host_exact = pinned_increment == expected_pinned
        host_priced = heap_increment <= envelope and rss_increment <= envelope and joint_fits
        report['host_reconciliation'] = {
            'derived_retained_incremental_host_bytes': derived_host, 'derived_serial_initialization_transient_bytes': transient,
            'derived_request_input_bytes': input_bytes, 'declared_additional_runtime_allowance_bytes': runtime_bound,
            'declared_incremental_host_envelope_bytes': envelope, 'measured_max_observed_heap_increment_bytes': heap_increment,
            'measured_rss_high_water_increment_bytes': rss_increment, 'derived_incremental_pinned_bytes': expected_pinned,
            'measured_incremental_pinned_bytes': pinned_increment, 'matched_pinned_payload': host_exact,
            'combined_generation_judge_host_admission_bytes': joint_host, 'within_incremental_and_joint_envelopes': host_priced,
            'incremental_measurement_scope': 'before-state-init through first-use-complete; ordinary generation is checked against the complete joint ledger',
            'measured_full_run_host_heap_live_max_bytes': max(integer(row, 'heap_live_bytes') for row in used_hosts),
            'measured_full_run_host_rss_high_water_bytes': max(integer(row, 'rss_high_water_bytes') for row in used_hosts),
            'measured_full_run_host_heap_increment_bytes': max(integer(row, 'heap_live_bytes') for row in used_hosts) - integer(before_host, 'heap_live_bytes'),
            'unattributed_generation_resume_heap_growth_bytes': max(integer(row, 'heap_live_bytes') for row in generation_hosts) - max(integer(row, 'heap_live_bytes') for row in judge_hosts),
            'full_run_within_joint_admission': joint_fits,
            'sampling_limit': 'heap live is phase-sampled, not an allocator-wide transient peak; VmHWM is process-wide physical RSS high-water',
        }
    else:
        report['host_reconciliation'] = {'missing_host_pricing_or_high_water': True}
    exact = exact and host_exact
    priced_peak = priced_peak and host_priced
    report['criteria']['static_ownership_ledger_reconciled'] = 'keep' if exact and priced_peak else 'reject' if static and allocations else 'blocked'
    restore = rows(lines, 'JUDGE_RESTORE')
    canaries = [line for line in lines if line.startswith('STATE_SCRATCH resource=judge ') and ' checked=1 ' in line]
    restored = len(probes) == 2 and len(restore) == 2 and len(canaries) == 2 * len(static) and all(row.get('restored') == '1' for row in probes)
    identical = restored and probes[0].get('head_fnv64') == probes[1].get('head_fnv64') and probes[0].get('tokens') == '4096'
    report['criteria']['same_actual_execution_owners_restored'] = 'keep' if identical and all_stages else 'blocked'
    resident = capacity and capacity[-1].get('resident_pairs') == '24576' and any('token graph hit path: 24576 resident experts' in line for line in lines)
    report['criteria']['all_24576_expert_pairs_resident'] = 'keep' if resident else 'reject' if plans else 'blocked'
    old = control.get('requests', [])
    new = candidate.get('requests', [])
    fixtures_match = control.get('input', {}).get('token_ids_sha256') == candidate.get('input', {}).get('token_ids_sha256') and bool(control.get('input'))
    generations_match = bool(old and new and old[0].get('status') == new[0].get('status') == 200 and
                             old[0].get('text_sha256') is not None and old[0].get('reasoning_sha256') is not None and
                             old[0].get('text_sha256') == new[0].get('text_sha256') and
                             old[0].get('reasoning_sha256') == new[0].get('reasoning_sha256') and
                             old[0].get('usage') == new[0].get('usage'))
    geometry_kept = bool(capacity and capacity[-1].get('context') == '262144' and capacity[-1].get('batch_slots') == '4' and capacity[-1].get('prefill_chunk') == '2048')
    report['generation_comparison'] = {'matched_frozen_input': fixtures_match, 'matched_text_reasoning_digests_and_usage': generations_match,
                                     'configured_capacity_unchanged': geometry_kept,
                                     'exercised_prompt_tokens': 1024 if fixtures_match else None,
                                     'full_context_boundary_exercised': False, 'four_simultaneous_slots_exercised': False}
    report['criteria']['generation_capacity_non_regression'] = 'keep' if fixtures_match and generations_match and geometry_kept and resident else 'blocked'
    clean = control.get('ok') and candidate.get('ok') and control.get('end_status') == candidate.get('end_status') == 'stopped'
    report['criteria']['clean_authorised_lifecycle'] = 'keep' if clean else 'reject'
    report['safety_peaks'] = {'control': control.get('peaks'), 'candidate': candidate.get('peaks')}
    report['failures'] = {'control': control.get('failures'), 'candidate': candidate.get('failures')}
    report['source_and_binary_identities'] = {'control_fixture_files': control.get('input', {}).get('files'),
                                             'candidate_fixture_files': candidate.get('input', {}).get('files')}
    report['disposition'] = 'keep' if all(value == 'keep' for value in report['criteria'].values()) else 'reject' if 'reject' in report['criteria'].values() else 'blocked'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to overwrite an earlier reconciliation result')
    result = reconcile(args.control, args.candidate)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'disposition': result['disposition'], 'criteria': result['criteria'], 'out': str(args.out)}, indent=2))
    return 0 if result['disposition'] == 'keep' else 1


if __name__ == '__main__':
    raise SystemExit(main())
