"""CPU-only tests of evidence refusal, not synthetic GPU qualification."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.judge_ownership_report import reconcile


def evidence():
    arm = {'ok': True, 'end_status': 'stopped', 'input': {'token_ids_sha256': 'frozen'},
           'requests': [{'status': 200, 'text_sha256': 'empty', 'reasoning_sha256': 'reasoning',
                         'usage': {'prompt_tokens': 1024, 'completion_tokens': 8}}],
           'profile_snapshot': {'mps_caps_bytes': [20000000]}}
    lines = [
        'JUDGE_STATIC device=0 cells=4096 session_device=1024 canary_device=280 head_output_device=128 allocator_padding_device=100 session_pinned_host=1 session_heap_host=12 canary_host=108 checkpoint_host=20 readback_host=128 metadata_host=1 init_host_transient=1 library_allowance=5000',
        'JUDGE_ADMISSION admit=1 verified=1 unknown_classes=0 guest_demand_bytes=1000000 runtime_allowance=512',
        'JUDGE_DEVICE_ADMISSION device=0 demand_bytes=10000000 ceiling_bytes=19000000 fits=1',
        'JUDGE_OWNER device=0', 'JUDGE_REBIND_GUARDS device=0',
        'JUDGE_CAPACITY context=262144 batch_slots=4 prefill_chunk=2048 resident_pairs=24576',
        'strata generate: token graph hit path: 24576 resident experts',
    ]
    for phase, live in [('before-state-init', 1000000), ('lane-ready', 1001432),
                        ('first-use-complete', 1005000), ('generation-resumed-final', 1008000)]:
        lines.append(f'JUDGE_ALLOC phase={phase} device=0 explicit_live_bytes={live} explicit_high_water_bytes={live}')
    for phase, spent, free in [('before-state-init', 100000, 19900000),
                              ('first-use-complete', 105000, 19895000),
                              ('generation-resumed-final', 2000000, 18000000)]:
        lines.append(f'JUDGE_MEMORY phase={phase} device=0 driver_used_bytes={spent} driver_free_bytes={free}')
    for phase, heap, pinned in [('before-state-init', 1000, 0), ('lane-ready', 1200, 109),
                               ('first-use-complete', 1500, 109), ('generation-resumed-final', 1500, 109)]:
        lines.append(f'JUDGE_HOST phase={phase} heap_live_bytes={heap} pinned_live_bytes={pinned} rss_high_water_bytes={heap}')
    for _ in range(2):
        lines.extend(['JUDGE_PROBE restored=1 head_fnv64=same tokens=4096', 'JUDGE_RESTORE checked=1',
                      'STATE_SCRATCH resource=judge checked=1 pass=1'])
    return arm, copy.deepcopy(arm), lines


class Reconciliation(unittest.TestCase):
    def report(self, control, candidate, lines):
        with patch('tools.judge_ownership_report.load_arm',
                   side_effect=[(control, [], 'control.log'), (candidate, lines, 'candidate.log')]):
            return reconcile(Path('control.json'), Path('candidate.json'))

    def test_complete_priced_evidence(self):
        report = self.report(*evidence())
        self.assertEqual(report['disposition'], 'keep')
        self.assertTrue(report['host_reconciliation']['matched_pinned_payload'])

    def test_unknown_or_unverified_admission_refuses(self):
        for change in ('unknown_classes=1', 'verified=0'):
            control, candidate, lines = evidence()
            lines[1] = lines[1].replace('unknown_classes=0' if change.startswith('unknown') else 'verified=1', change)
            report = self.report(control, candidate, lines)
            self.assertEqual(report['criteria']['verified_complete_admission'], 'reject')

    def test_missing_generation_high_water_is_not_complete(self):
        control, candidate, lines = evidence()
        lines = [line for line in lines if 'generation-resumed-final' not in line]
        self.assertEqual(self.report(control, candidate, lines)['criteria']['static_ownership_ledger_reconciled'], 'reject')

    def test_host_first_use_is_not_free(self):
        control, candidate, lines = evidence()
        lines = [line.replace('heap_live_bytes=1500', 'heap_live_bytes=35000')
                 .replace('rss_high_water_bytes=1500', 'rss_high_water_bytes=35000') for line in lines]
        report = self.report(control, candidate, lines)
        self.assertFalse(report['host_reconciliation']['within_incremental_and_joint_envelopes'])
        self.assertEqual(report['criteria']['static_ownership_ledger_reconciled'], 'reject')

    def test_empty_text_does_not_hide_reasoning_regression(self):
        control, candidate, lines = evidence()
        candidate['requests'][0]['reasoning_sha256'] = 'changed'
        self.assertEqual(self.report(control, candidate, lines)['criteria']['generation_capacity_non_regression'], 'blocked')


if __name__ == '__main__':
    unittest.main()
