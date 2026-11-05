#!/usr/bin/env python3
"""Measure frozen DCCF inference on a predeclared train/select length panel.

No examples are generated and no detector is fitted. The timings describe the
current complete-decoder implementation, including its requested hidden states;
layer selection and group sparsity do not provide early exit here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dual_concept.backbone import (FrozenBackbone, FrozenBackboneConfig,
                                   StateCache, assert_compatible_protocol, protocol_hash)
from dual_concept.data import load_dataset
from dual_concept.pipeline import Detector


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def formatted_length(tokenizer, text: str, template_kwargs: dict) -> int:
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=False,
        add_generation_prompt=True, **template_kwargs)
    encoded = tokenizer([rendered], padding=True, truncation=False,
                        add_special_tokens=False, return_attention_mask=True)
    count = len(encoded['input_ids'][0])
    if count < 1 or sum(encoded['attention_mask'][0]) != count:
        raise ValueError('Unexpected padding or empty single-request template')
    return count


def select_panel(dataset, tokenizer, template_kwargs: dict, maximum: int = 32):
    """Deterministic nearest-rank length quantiles; labels are never consulted."""
    if type(maximum) is not int or not 1 <= maximum <= 32:
        raise ValueError('max_samples must be in 1..32')
    candidates = []
    for split in ('train', 'select'):
        for row in dataset.splits[split]:
            if row['split'] != split:
                raise ValueError('Benchmark source row has an inconsistent split')
            candidates.append((formatted_length(tokenizer, row['text'], template_kwargs),
                               row['id'], row))
    if not candidates:
        raise ValueError('No train/select requests are available for benchmarking')
    candidates.sort(key=lambda item: (item[0], item[1]))
    ranks = np.rint(np.linspace(0, len(candidates) - 1,
                               min(maximum, len(candidates)))).astype(int)
    if len(set(ranks.tolist())) != len(ranks):
        raise RuntimeError('Length-panel ranks unexpectedly overlap')
    panel = []
    for rank in ranks:
        count, identifier, row = candidates[int(rank)]
        panel.append({'row': row, 'id': identifier, 'split': row['split'], 'tokens': count,
                      'sorted_rank': int(rank),
                      'length_quantile': float(rank / max(1, len(candidates) - 1))})
    return panel, len(candidates)


def panel_specification(panel, manifest_sha256: str, source_count: int) -> dict:
    """Portable fixed text identity panel; token counts refer to its creator."""
    return {'schema_version': 1, 'source_splits': ['train', 'select'],
            'data_manifest_sha256': manifest_sha256, 'source_count': source_count,
            'selection': 'reference-model token-length quantiles, deterministic ID tie-break; no labels',
            'token_count_scope': 'reference model only; each model recomputes its actual lengths in plan.json',
            'samples': [{**{key: value for key, value in item.items() if key != 'row'},
                         'text_sha256': hashlib.sha256(item['row']['text'].encode()).hexdigest()}
                        for item in panel]}


def reuse_panel(dataset, tokenizer, template_kwargs: dict, reference: Path,
                manifest_sha256: str, maximum: int = 32):
    """Preserve a reference panel's ID order, never consult other dataset roles."""
    source = reference / 'panel.json' if reference.is_dir() else reference
    specification = json.loads(source.read_text(encoding='utf-8'))
    if specification.get('schema_version') != 1 or specification.get('source_splits') != ['train', 'select']:
        raise ValueError('Reference panel must explicitly use train/select only')
    if specification.get('data_manifest_sha256') != manifest_sha256:
        raise ValueError('Reference panel data manifest hash differs')
    samples = specification.get('samples')
    if not isinstance(samples, list) or not 1 <= len(samples) <= min(32, maximum):
        raise ValueError('Reference panel sample count exceeds the fixed budget or is empty')
    allowed = {}
    for split in ('train', 'select'):
        for row in dataset.splits[split]:
            if row['split'] != split or row['id'] in allowed:
                raise ValueError('Benchmark source identity/split is inconsistent')
            allowed[row['id']] = row
    if specification.get('source_count') != len(allowed):
        raise ValueError('Reference panel train/select source count differs')
    panel, seen = [], set()
    for item in samples:
        if not isinstance(item, dict) or item.get('split') not in ('train', 'select'):
            raise ValueError('Reference panel sample must remain in train/select')
        identifier = item.get('id')
        if not isinstance(identifier, str) or identifier in seen or identifier not in allowed:
            raise ValueError('Reference panel contains duplicate or unknown train/select ID')
        row = allowed[identifier]
        if row['split'] != item['split']:
            raise ValueError('Reference panel sample split differs from source data')
        if hashlib.sha256(row['text'].encode()).hexdigest() != item.get('text_sha256'):
            raise ValueError('Reference panel sample text hash differs')
        seen.add(identifier)
        panel.append({'row': row, 'id': identifier, 'split': item['split'],
                      'tokens': formatted_length(tokenizer, row['text'], template_kwargs),
                      'reference_tokens': item.get('tokens'),
                      'reference_sorted_rank': item.get('sorted_rank'),
                      'reference_length_quantile': item.get('length_quantile')})
    return panel, len(allowed), source


def timed_call(function, synchronize=lambda: None):
    synchronize()
    started = time.perf_counter()
    result = function()
    synchronize()
    return result, time.perf_counter() - started


def timing_summary(seconds: list[float]) -> dict:
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Latency observations must be nonempty, finite and nonnegative')
    return {'requests': len(values), 'mean_ms': float(values.mean() * 1000),
            'p50_ms': float(np.percentile(values, 50) * 1000),
            'p95_ms': float(np.percentile(values, 95) * 1000),
            'min_ms': float(values.min() * 1000), 'max_ms': float(values.max() * 1000)}


def benchmark(args, report: dict) -> None:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    report['stage'] = 'load_frozen_detector_and_cache_protocol'
    model, detector_seconds = timed_call(lambda: Detector.load(args.model))
    expected = model.metadata.get('extraction_protocol')
    if not expected:
        raise ValueError('The frozen detector has no extraction protocol')
    cache_protocol_path = args.cache / 'protocol.json'
    cache_protocol = json.loads(cache_protocol_path.read_text())
    if protocol_hash(expected) != protocol_hash(cache_protocol):
        raise ValueError('The complete cache belongs to another extraction protocol')
    if not (args.cache / 'rows').is_dir():
        raise FileNotFoundError('The existing row cache is absent')
    config_payload = json.loads(args.config.read_text())
    if set(config_payload) - {'backbone', 'method'}:
        raise ValueError('Configuration accepts only backbone and method sections')
    config = FrozenBackboneConfig(**config_payload.get('backbone', {}))
    # Runtime changes are checked again against the actual loaded protocol.
    config = replace(config, batch_size=1, local_files_only=True,
                     candidate_layers=tuple(int(layer) for layer in model.bank.layer_ids))
    report['input_sha256'] = {str(path): sha256(path) for path in (
        args.config, args.model / 'model.json', args.model / 'parameters.npz',
        args.data / 'manifest.json', cache_protocol_path, Path(__file__).resolve())}
    report['stage'] = 'cold_backbone_load'
    started = time.perf_counter()
    backbone = FrozenBackbone(config)
    backbone.extract([])
    torch = backbone._torch
    device = next(backbone._decoder.parameters()).device
    is_cuda = str(device).startswith('cuda')

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)
    synchronize()
    load_seconds = time.perf_counter() - started
    actual = backbone.get_protocol()
    assert_compatible_protocol(expected, actual)
    report['protocol'] = actual
    report['cold_load'] = {'detector_ms': detector_seconds * 1000,
                           'backbone_ms': load_seconds * 1000,
                           'combined_ms': (detector_seconds + load_seconds) * 1000,
                           'scope': 'process model loading, including lazy library imports; operating-system file cache is not flushed'}
    report['device'] = str(device)
    if is_cuda:
        report['cuda_device'] = {'name': torch.cuda.get_device_name(device),
                                 'total_memory_bytes': int(torch.cuda.get_device_properties(device).total_memory)}
    write_json(args.output / 'benchmark.json', report)
    report['stage'] = 'select_train_select_length_panel'
    dataset = load_dataset(args.data)
    manifest_sha = sha256(args.data / 'manifest.json')
    if args.panel_from:
        panel, pool_count, reference = reuse_panel(
            dataset, backbone._tokenizer, dict(config.chat_template_kwargs),
            args.panel_from, manifest_sha, args.max_samples)
        # Identical panel bytes/hash document identical raw-text IDs across models.
        (args.output / 'panel.json').write_bytes(reference.read_bytes())
        reference_source = {'path': str(reference.resolve()), 'sha256': sha256(reference)}
    else:
        panel, pool_count = select_panel(dataset, backbone._tokenizer,
                                         dict(config.chat_template_kwargs), args.max_samples)
        write_json(args.output / 'panel.json', panel_specification(panel, manifest_sha, pool_count))
        reference_source = None
    if max(item['tokens'] for item in panel) > config.max_length:
        raise ValueError('A selected complete template exceeds the frozen max_length; no truncation is permitted')
    plan = {'source_splits': ['train', 'select'], 'source_count': pool_count,
            'selection': ('fixed reference raw-text ID order; recompute current-model lengths; no labels'
                          if args.panel_from else 'nearest-rank token-length quantiles, deterministic ID tie-break; no labels'),
            'samples': [{key: value for key, value in item.items() if key != 'row'} for item in panel],
            'batch_size': 1, 'warmup_rounds': args.warmup, 'repeats_per_sample': args.repeats,
            'protocol_sha256': protocol_hash(actual), 'config_sha256': sha256(args.config),
            'panel_sha256': sha256(args.output / 'panel.json'),
            'reference_panel_source': reference_source,
            'distribution_note': 'reference length-quantile text panel; current-model tokens are reported; latency is not production-request weighted'}
    write_json(args.output / 'plan.json', plan)
    report['panel'] = plan
    write_json(args.output / 'benchmark.json', report)
    cache = StateCache(args.cache / 'rows')
    positions = [cache_protocol['candidate_layers'].index(int(layer)) for layer in model.bank.layer_ids]
    cached_states = []
    for item in panel:
        row = item['row']
        state = cache.load(row['id'], row['text'], cache_protocol)
        if state is None:
            raise FileNotFoundError('Complete benchmark cache is missing row ' + row['id'])
        cached_states.append(state[positions][None, ...])

    def head(state):
        score = np.asarray(model.decision_function(state), dtype=np.float64)
        if score.shape != (1,) or not np.isfinite(score).all():
            raise ValueError('Detector returned a malformed or nonfinite logit')
        return float(score[0]), bool(score[0] >= model.threshold)

    def end_to_end(item):
        return head(backbone.extract([item['row']['text']]))

    report['stage'] = 'warmup'
    for _ in range(args.warmup):
        for item in panel:
            end_to_end(item)
    synchronize()
    report['phases'] = {}

    def peak():
        if not is_cuda:
            return None
        return {'allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
                'reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
                'total_memory_bytes': int(torch.cuda.get_device_properties(device).total_memory)}

    with (args.output / 'timings.jsonl').open('x', encoding='utf-8') as stream:
        for phase in ('backbone_only', 'detector_head_only', 'end_to_end'):
            report['stage'] = phase
            write_json(args.output / 'benchmark.json', report)
            if is_cuda:
                synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = []
            for repeat in range(args.repeats):
                for index, item in enumerate(panel):
                    if phase == 'backbone_only':
                        function = lambda item=item: backbone.extract([item['row']['text']])
                    elif phase == 'detector_head_only':
                        function = lambda index=index: head(cached_states[index])
                    else:
                        function = lambda item=item: end_to_end(item)
                    # The CPU NumPy head has no pending device work. GPU phases
                    # use synchronized wall time, including host tokenization.
                    result, seconds = timed_call(function, synchronize if phase != 'detector_head_only' else lambda: None)
                    if phase == 'backbone_only' and not np.isfinite(result).all():
                        raise ValueError('Backbone returned nonfinite states')
                    del result
                    elapsed.append(seconds)
                    stream.write(json.dumps({'phase': phase, 'repeat': repeat, 'id': item['id'],
                                             'split': item['split'], 'tokens': item['tokens'],
                                             'seconds': seconds}, allow_nan=False) + '\n')
                    stream.flush()
            report['phases'][phase] = {**timing_summary(elapsed),
                                       'cuda_peak': peak() if phase != 'detector_head_only' else None,
                                       'scope': {'backbone_only': 'text template/tokenization, complete decoder, state extraction and device-to-host copy',
                                                 'detector_head_only': 'CPU NumPy concept readout, standardization, fusion and >= decision on in-memory cached states; cache I/O excluded',
                                                 'end_to_end': 'raw text through complete backbone, state transfer and frozen detector decision; no model loading or cache I/O'}[phase]}
            write_json(args.output / 'benchmark.json', report)
    report['stage'] = 'complete'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'cache', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=32)
    parser.add_argument('--panel-from', type=Path,
                        help='Reference benchmark directory or panel.json; preserve identical text ID order')
    parser.add_argument('--warmup', type=int, default=2, help='Complete warmup rounds over the selected panel; minimum 2')
    parser.add_argument('--repeats', type=int, default=3, help='Measured repeats per sample; minimum 3')
    args = parser.parse_args(argv)
    if not 1 <= args.max_samples <= 32 or args.warmup < 2 or args.repeats < 3:
        parser.error('max-samples must be 1..32, warmup >=2 and repeats >=3')
    for name in ('model', 'data', 'cache', 'config', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.panel_from:
        args.panel_from = args.panel_from.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        print('Benchmark output must be new or empty; existing evidence was preserved', file=sys.stderr)
        return 2
    report = {'schema_version': 1, 'status': 'running', 'started_at': now(),
              'python': sys.version, 'platform': platform.platform(),
              'complete_backbone_forward': True, 'early_exit': False,
              'generates_answers': False, 'fits_detector': False,
              'source_splits': ['train', 'select'], 'script_sha256': sha256(Path(__file__).resolve())}
    write_json(args.output / 'benchmark.json', report)
    try:
        benchmark(args, report)
        report['status'] = 'succeeded'
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        failure = {'stage': report.get('stage'), 'type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc()}
        write_json(args.output / 'failure.json', failure)
        report.update(status='failed', failure=failure)
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    report['finished_at'] = now()
    write_json(args.output / 'benchmark.json', report)
    print(json.dumps({'status': report['status'], 'stage': report.get('stage'),
                      'report': str(args.output / 'benchmark.json')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
