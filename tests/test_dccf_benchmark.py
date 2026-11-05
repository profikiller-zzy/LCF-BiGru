"""Offline safeguards for the latency panel and synchronized wall-clock timing."""
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import benchmark_dccf as benchmark


class NoLabels(dict):
    def __getitem__(self, key):
        if key in {'kind', 'label', 'attack_family'}:
            raise AssertionError('Latency sampling must not consult labels')
        return super().__getitem__(key)


class FittingSplitsOnly(dict):
    def __getitem__(self, key):
        if key not in {'train', 'select'}:
            raise AssertionError('Latency sampling read a non-fitting split')
        return super().__getitem__(key)


class LengthTokenizer:
    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs.append(kwargs)
        return messages[0]['content']

    def __call__(self, rendered, **kwargs):
        assert kwargs == {'padding': True, 'truncation': False,
                          'add_special_tokens': False, 'return_attention_mask': True}
        length = int(rendered[0])
        return {'input_ids': [list(range(length))], 'attention_mask': [[1] * length]}


class BenchmarkTests(unittest.TestCase):
    def test_panel_uses_only_train_select_lengths_and_frozen_template_mode(self):
        rows = [NoLabels(id=f'id-{i:03d}', text=str(i + 1),
                         split='train' if i % 2 else 'select') for i in range(80)]
        data = SimpleNamespace(splits=FittingSplitsOnly({
            split: [row for row in reversed(rows) if row['split'] == split]
            for split in ('train', 'select')}))
        tokenizer = LengthTokenizer()
        panel, pool = benchmark.select_panel(data, tokenizer, {'enable_thinking': False})
        self.assertEqual(pool, 80)
        self.assertEqual(len(panel), 32)
        self.assertEqual(len({item['id'] for item in panel}), 32)
        self.assertEqual((panel[0]['tokens'], panel[-1]['tokens']), (1, 80))
        self.assertEqual((panel[0]['length_quantile'], panel[-1]['length_quantile']), (0., 1.))
        self.assertTrue(all(kwargs == {'tokenize': False, 'add_generation_prompt': True,
                                      'enable_thinking': False} for kwargs in tokenizer.template_kwargs))
        repeated, _ = benchmark.select_panel(data, LengthTokenizer(), {'enable_thinking': False})
        self.assertEqual([p['id'] for p in panel], [p['id'] for p in repeated])

    def test_timing_brackets_work_with_synchronization(self):
        events = []
        times = iter([10., 10.25])
        def clock():
            events.append('clock')
            return next(times)
        def work():
            events.append('work')
            return 'result'
        with mock.patch.object(benchmark.time, 'perf_counter', side_effect=clock):
            result, seconds = benchmark.timed_call(work, lambda: events.append('synchronize'))
        self.assertEqual(result, 'result')
        self.assertEqual(seconds, .25)
        self.assertEqual(events, ['synchronize', 'clock', 'work', 'synchronize', 'clock'])

    def panel_fixture(self):
        rows = [NoLabels(id=f'reference-{i}', text=str(i + 1), split='train' if i % 2 else 'select')
                for i in range(4)]
        data = SimpleNamespace(splits=FittingSplitsOnly({
            split: [row for row in rows if row['split'] == split] for split in ('train', 'select')}))
        panel, count = benchmark.select_panel(data, LengthTokenizer(), {}, 4)
        specification = benchmark.panel_specification(list(reversed(panel)), 'same-manifest', count)
        return data, specification

    def test_reference_directory_or_file_preserves_ids_and_recomputes_actual_lengths(self):
        data, specification = self.panel_fixture()
        class OtherTokenizer(LengthTokenizer):
            def __call__(self, rendered, **kwargs):
                return super().__call__([str(int(rendered[0]) * 2)], **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'panel.json'
            path.write_text(json.dumps(specification))
            for source in (root, path):
                panel, count, resolved = benchmark.reuse_panel(data, OtherTokenizer(), {}, source, 'same-manifest')
                self.assertEqual(count, 4)
                self.assertEqual(resolved, path)
                self.assertEqual([p['id'] for p in panel], [p['id'] for p in specification['samples']])
                self.assertEqual([p['tokens'] for p in panel], [p['tokens'] * 2 for p in specification['samples']])
                self.assertEqual([p['reference_tokens'] for p in panel], [p['tokens'] for p in specification['samples']])

    def test_reference_panel_rejects_diagnostic_or_disguised_nonfitting_ids(self):
        data, specification = self.panel_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'panel.json'
            for sample in ({'id': 'diagnose-only', 'split': 'diagnose', 'text_sha256': 'unused'},
                           {'id': 'diagnose-only', 'split': 'train', 'text_sha256': 'unused'}):
                with self.subTest(sample=sample):
                    path.write_text(json.dumps({**specification, 'samples': [sample]}))
                    with self.assertRaisesRegex(ValueError, 'train/select'):
                        benchmark.reuse_panel(data, LengthTokenizer(), {}, path, 'same-manifest')

    def test_reference_panel_rejects_changed_source_hash_text_and_duplicate_ids(self):
        data, specification = self.panel_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'panel.json'
            path.write_text(json.dumps(specification))
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                benchmark.reuse_panel(data, LengthTokenizer(), {}, path, 'different-manifest')
            first = specification['samples'][0]
            for samples, message in (([{**first, 'text_sha256': 'changed'}], 'text hash'),
                                     ([first, first], 'duplicate')):
                with self.subTest(message=message):
                    path.write_text(json.dumps({**specification, 'samples': samples}))
                    with self.assertRaisesRegex(ValueError, message):
                        benchmark.reuse_panel(data, LengthTokenizer(), {}, path, 'same-manifest')

    def arguments(self, root):
        return [part for name in ('model', 'data', 'cache', 'config', 'output')
                for part in ('--' + name, str(root / name))]

    def test_failed_load_keeps_honest_failure_evidence_and_never_emits_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(benchmark.Detector, 'load', side_effect=ValueError('invalid frozen model')):
                self.assertEqual(benchmark.main(self.arguments(root)), 1)
            report = json.loads((root / 'output/benchmark.json').read_text())
            failure = json.loads((root / 'output/failure.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(failure['error'], 'invalid frozen model')
            self.assertNotIn('phases', report)
            self.assertFalse((root / 'output/timings.jsonl').exists())

    def test_protocol_mismatch_stops_before_sampling_or_nonempty_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('model', 'data', 'cache'):
                (root / name).mkdir()
            (root / 'cache/rows').mkdir()
            expected = {'candidate_layers': [1, 2], 'hidden_size': 2}
            (root / 'cache/protocol.json').write_text(json.dumps(expected))
            (root / 'config').write_text(json.dumps({'backbone': {'candidate_layers': [1, 2]}}))
            for path in ('model/model.json', 'model/parameters.npz', 'data/manifest.json'):
                (root / path).write_text('{}')
            model = SimpleNamespace(metadata={'extraction_protocol': expected},
                                    bank=SimpleNamespace(layer_ids=[1, 2]))
            fake = SimpleNamespace(_torch=None,
                                   _decoder=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cpu')])),
                                   extract=mock.Mock(), get_protocol=lambda: {'different': True})
            with mock.patch.object(benchmark.Detector, 'load', return_value=model), \
                 mock.patch.object(benchmark, 'FrozenBackbone', return_value=fake), \
                 mock.patch.object(benchmark, 'assert_compatible_protocol', side_effect=ValueError('template mode changed')), \
                 mock.patch.object(benchmark, 'load_dataset', side_effect=AssertionError('No sampling on mismatched protocol')):
                self.assertEqual(benchmark.main(self.arguments(root)), 1)
            fake.extract.assert_called_once_with([])
            report = json.loads((root / 'output/benchmark.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertNotIn('panel', report)
            self.assertIn('template mode changed', report['failure']['error'])

    def test_existing_output_is_preserved_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            evidence = root / 'output/benchmark.json'
            evidence.write_text('prior evidence')
            with mock.patch.object(benchmark.Detector, 'load', side_effect=AssertionError('Must preserve existing output')):
                self.assertEqual(benchmark.main(self.arguments(root)), 2)
            self.assertEqual(evidence.read_text(), 'prior evidence')

    def test_three_timing_scopes_complete_on_one_fixed_offline_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('model', 'data', 'cache'):
                (root / name).mkdir()
            protocol = {'candidate_layers': [1, 2], 'hidden_size': 2}
            (root / 'cache/protocol.json').write_text(json.dumps(protocol))
            (root / 'config').write_text(json.dumps({'backbone': {'candidate_layers': [1, 2], 'max_length': 20}}))
            for path in ('model/model.json', 'model/parameters.npz', 'data/manifest.json'):
                (root / path).write_text('{}')
            rows = [NoLabels(id=f'row-{i}', text=str(i + 1), split='train' if i % 2 else 'select') for i in range(4)]
            data = SimpleNamespace(splits=FittingSplitsOnly({
                split: [row for row in rows if row['split'] == split] for split in ('train', 'select')}))
            cache = benchmark.StateCache(root / 'cache/rows')
            for row in rows:
                cache.save(row['id'], row['text'], protocol, np.zeros((2, 2), dtype=np.float32))
            model = SimpleNamespace(metadata={'extraction_protocol': protocol}, threshold=0.,
                                    bank=SimpleNamespace(layer_ids=[1, 2]),
                                    decision_function=lambda states: states.sum(axis=(1, 2)))
            forwards = []
            def extract(texts):
                forwards.append(texts)
                return np.zeros((len(texts), 2, 2), dtype=np.float32)
            fake = SimpleNamespace(_torch=None, _tokenizer=LengthTokenizer(),
                                   _decoder=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cpu')])),
                                   extract=extract, get_protocol=lambda: protocol)
            with mock.patch.object(benchmark.Detector, 'load', return_value=model), \
                 mock.patch.object(benchmark, 'FrozenBackbone', return_value=fake), \
                 mock.patch.object(benchmark, 'assert_compatible_protocol'), \
                 mock.patch.object(benchmark, 'load_dataset', return_value=data):
                self.assertEqual(benchmark.main(self.arguments(root)), 0)
            report = json.loads((root / 'output/benchmark.json').read_text())
            plan = json.loads((root / 'output/plan.json').read_text())
            timings = [json.loads(line) for line in (root / 'output/timings.jsonl').read_text().splitlines()]
            self.assertEqual(report['status'], 'succeeded')
            self.assertEqual(set(report['phases']), {'backbone_only', 'detector_head_only', 'end_to_end'})
            self.assertTrue(all(phase['requests'] == 12 for phase in report['phases'].values()))
            self.assertEqual(len(timings), 36)
            self.assertEqual(forwards.count([]), 1)
            self.assertEqual(len([call for call in forwards if call]), 32)  # 8 warmups + 12 + 12 measured forwards.
            self.assertEqual(plan['source_splits'], ['train', 'select'])
            self.assertTrue(all('text' not in item for item in plan['samples']))
            self.assertEqual(plan['panel_sha256'], hashlib.sha256((root / 'output/panel.json').read_bytes()).hexdigest())
            self.assertTrue(all(len(call) == 1 for call in forwards if call))


if __name__ == '__main__':
    unittest.main()
