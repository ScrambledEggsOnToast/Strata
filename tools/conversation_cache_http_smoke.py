"""Destructive-to-cache HTTP smoke test for an EXCLUSIVE, idle Strata test server.

Sends synthetic conversations, deliberately evicts cached conversations, and
disconnects a streaming response. Never run while other clients are using it.
Dry-run by default; --run explicitly enables network traffic. Does not restart
or reconfigure the server. Use the private-engine parity gate for state hashes.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Client:
    def __init__(self, url):
        self.url = url.rstrip('/')

    def metrics(self):
        with urllib.request.urlopen(self.url + '/metrics', timeout=10) as response:
            return json.load(response)

    def request(self, model, content, stream=False, count=8, conversation=None, messages=None):
        body = {'model': model, 'messages': messages or [{'role': 'user', 'content': content}],
                'max_tokens': count, 'temperature': 0, 'stream': stream,
                'strata_conversation_id': conversation,
                'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(self.url + '/v1/chat/completions',
                                        data=json.dumps(body).encode(),
                                        headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=120)


def run(client, scenario='full'):
    initial = client.metrics()
    require(initial['live']['state'] == 'idle' and initial['live']['queued'] == 0, 'server is busy')
    info = initial['engine']
    slots = info.get('conversation_cache_slots', 0)
    require(info.get('conversation_cache_mib', 0) > 0 and 1 <= slots <= 16,
            'requires an enabled cache with 1–16 slots')
    require(not info.get('api_key'), 'use an unauthenticated private test endpoint')
    marker = 'cache-smoke-' + uuid.uuid4().hex
    records = []

    def before():
        metrics = client.metrics()
        require(metrics['live']['state'] == 'idle' and metrics['live']['queued'] == 0,
                'concurrent client detected')
        return max((r['time'] for r in metrics['requests']), default=0)

    def after(stamp, label):
        for _ in range(50):
            metrics = client.metrics()
            if metrics['live']['state'] == 'idle':
                break
            time.sleep(.2)
        require(metrics['live']['state'] == 'idle', 'request did not drain')
        fresh = [r for r in metrics['requests'] if r['time'] > stamp]
        require(len(fresh) == 1 and metrics['live']['queued'] == 0,
                'missing request evidence or concurrent traffic')
        record = {'name': label, **fresh[0]}
        records.append(record)
        return record

    def prompt(label):
        return marker + ' ' + label + '\n' + '\n'.join(
            f'Record {i}: blue square, green triangle, red circle.' for i in range(128)
        ) + '\nName one color. Answer with one word.'

    def ask(label, content, conversation, messages=None):
        stamp = before()
        started = time.monotonic()
        with client.request(info['model'], content, conversation=marker + '-' + conversation, messages=messages) as response:
            result = json.load(response)
        require(bool(result.get('choices')), 'missing completion')
        record = after(stamp, label)
        record['http_wall_ms'] = 1000 * (time.monotonic() - started)
        record['committed_message'] = result['choices'][0]['message']
        require(record['finish'] in ('length', 'stop'), 'request did not finish normally')
        return result['choices'][0]['message'], record

    if scenario == 'turns':
        content = prompt('A')
        first, first_record = ask('A1', content, 'A')
        require(first_record['reused'] == 0, 'turn one unexpectedly reused state')
        second_prompt = [{'role': 'user', 'content': content}, first,
                         {'role': 'user', 'content': 'Name another color from that list. One word.'}]
        second, second_record = ask('A2', content, 'A', second_prompt)
        require(second_record['reused'] > 0, 'turn two re-prefilled its whole conversation')
        ask('B1', prompt('B'), 'B')
        third_prompt = second_prompt + [second, {'role': 'user', 'content': 'Name a shape from that list. One word.'}]
        third, third_record = ask('A3-parked', content, 'A', third_prompt)
        require(third_record['reused'] > 0, 'parked conversation did not resume')
        cold, cold_record = ask('A3-cold-control', content, 'cold-control', third_prompt)
        require(cold_record['reused'] == 0 and cold == third,
                'distinct-conversation canary reused state or committed output differs from cold control')
        cache = client.metrics().get('conversation_cache', {})
        require(cache.get('restores', 0) > 0, 'no RAM-parking restoration was observed')
        require(0 <= cache.get('parked', -1) <= slots and
                0 <= cache.get('bytes', -1) <= info['conversation_cache_mib'] * 1024 * 1024,
                'parked count/bytes missing or outside configured ceilings')
        return {'engine_info': info, 'records': records, 'cache': cache}

    a, _ = ask('A', prompt('A'), 'A')
    ask('B', prompt('B'), 'B')
    restored, record = ask('A-restored', prompt('A'), 'A')
    require(record['reused'] > 0 and a == restored, 'A/B/A reuse or output mismatch')
    for i in range(slots + 1):
        ask(f'evict-{i}', prompt(f'evict-{i}'), f'evict-{i}')
    _, record = ask('A-evicted', prompt('A'), 'A')
    require(record['reused'] == 0, 'old A survived slot eviction')

    text = marker + ' cancel: ' + 'blue green red. ' * 256
    text += '\nCount from 1 to 1000, one number per line. Do not summarize.'
    stamp = before()
    with client.request(info['model'], text, stream=True, count=4096, conversation=marker + '-cancel') as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            choices = json.loads(line[6:]).get('choices', [])
            if choices and choices[0].get('delta', {}).get('content'):
                break
        else:
            raise RuntimeError('no streamed content to cancel')
    record = after(stamp, 'cancelled')
    require(record['finish'] in ('disconnect', 'cancel') and 0 < record['output_tokens'] < 4096,
            'stream cancellation not observed')
    ask('after-cancel-B', marker + ' Unrelated B: say hello.', 'B')
    _, record = ask('cancelled-A-restored', text, 'cancel')
    require(record['reused'] > 0, 'cancelled conversation was not restored')
    return {'engine_info': info, 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:11434')
    parser.add_argument('--output', type=Path, required=True, help='new results file; existing paths refused')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--scenario', choices=('full', 'turns'), default='full',
                        help='turns: five requests for turn-two reuse, A/B/A parking and an identical cold control')
    args = parser.parse_args()
    if not args.run:
        print('Dry run: HTTP A/B/A, slot eviction, cancellation and recovery. No network traffic.')
        return
    # Refuse accidental overwrites before generating traffic.
    with args.output.open('x', encoding='utf-8') as output:
        try:
            result = run(Client(args.url), args.scenario)
        except Exception as error:
            json.dump({'passed': False, 'error': str(error)}, output, indent=2)
            output.write('\n')
            raise
        json.dump({'passed': True, **result}, output, indent=2)
        output.write('\n')
    print(f'PASS: HTTP reuse, slot eviction, cancellation/recovery. Results: {args.output}')


if __name__ == '__main__':
    main()
