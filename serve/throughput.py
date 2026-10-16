"""Content-free native work counts in a common five-second wall-clock window."""
import collections
import math
import threading
import time


class Throughput:
    window_s = 5.0

    def __init__(self, prefill_supported=False, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.buckets = collections.deque()
        self.prefill_supported = prefill_supported
        self.fresh = None

    def wire(self, line):
        fields = line.split()
        if not fields:
            return
        decode = prefill = 0
        with self.lock:
            try:
                if fields[0] == 'INFO' and fields[1:] == ['throughput_progress=1']:
                    self.prefill_supported = True
                elif fields[0] == 'RESUME':
                    self.fresh = 0 if len(fields) == 2 and int(fields[1]) >= 0 else None
                elif fields[0] == 'PP':
                    # Sixth field is fresh work, unlike position which includes cache hits.
                    if len(fields) < 6:
                        self.prefill_supported = False
                        self.fresh = None
                        return
                    fresh = int(fields[5])
                    if fresh < 0 or self.fresh is None or fresh < self.fresh:
                        self.prefill_supported = False
                        self.fresh = None
                        return
                    prefill = fresh - self.fresh
                    self.fresh = fresh
                elif fields[0] == 'INFO' and len(fields) == 2 and fields[1].startswith('fresh_prompt_tokens='):
                    fresh = int(fields[1].split('=', 1)[1])
                    if self.fresh is None or fresh < self.fresh:
                        self.prefill_supported = False
                        self.fresh = None
                        return
                    prefill = fresh - self.fresh
                    self.fresh = fresh
                elif fields[0] == 'DONE':
                    # Legacy DONE totals use cache-resume accounting, not actual re-read work.
                    self.fresh = None
                elif fields[0] == 'T' and len(fields) == 2:
                    if int(fields[1]) >= 0:
                        decode = 1
                elif fields[0] == 'BT' and len(fields) == 3:
                    if int(fields[1]) >= 0 and int(fields[2]) >= 0:
                        decode = 1
                else:
                    return
            except (ValueError, IndexError):
                if fields[0] in ('RESUME', 'PP') or (fields[0] == 'INFO' and 'fresh_prompt_tokens=' in line):
                    self.prefill_supported = False
                    self.fresh = None
                return
            now = self.clock()
            self._expire(now)
            if decode or prefill:
                # At most 51 buckets, independent of token rate or request count.
                bucket = math.floor(now * 10) / 10
                if self.buckets and self.buckets[-1][0] == bucket:
                    self.buckets[-1][1] += decode
                    self.buckets[-1][2] += prefill
                else:
                    self.buckets.append([bucket, decode, prefill])

    def _expire(self, now):
        while self.buckets and self.buckets[0][0] <= now - self.window_s:
            self.buckets.popleft()

    def snapshot(self, idle=False):
        with self.lock:
            now = self.clock()
            window = 2.0 if idle else self.window_s
            self._expire(now)
            selected = [p for p in self.buckets if p[0] > now - window]
            return {'window_s': window,
                    'decode_tok_s': sum(p[1] for p in selected) / window,
                    'prefill_tok_s': sum(p[2] for p in selected) / window if self.prefill_supported else None}
