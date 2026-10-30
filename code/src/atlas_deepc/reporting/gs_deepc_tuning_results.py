"""Audit saved GS-DeePC weight-only pilots; never launch simulations."""
from dataclasses import asdict
import json

import numpy as np

from atlas_deepc.benchmarks import gain_scheduled_deepc_reacher as gs
from atlas_deepc.benchmarks.tune_gain_scheduled_deepc import ROOT


def export():
    weights = ('q_xy', 'r', 'lambda_g', 'lambda_ini')
    reference = None
    candidates, records = [], []
    for path in sorted(ROOT.glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        directory = path.parent
        assert manifest['controller_sha256'] == gs.utils.digest(gs.__file__)
        assert manifest['data_sha256'] == gs.utils.digest(gs.DATA)
        assert manifest['utilities_sha256'] == gs.utils.digest(gs.utils.__file__)
        for key, value in asdict(gs.Config()).items():
            if key not in weights:
                assert manifest['config'][key] == value
        with np.load(directory/'offline_library.npz') as bank:
            arrays = {key: bank[key].copy() for key in bank.files}
        if reference is None:
            reference = arrays
        else:
            for key, value in reference.items():
                np.testing.assert_array_equal(arrays[key], value)
        rows = []
        for seed in manifest['seeds']:
            trace_path = directory/f'seed{seed}_control_results.npz'
            if trace_path.exists():
                with np.load(trace_path) as a:
                    row = json.loads(a['summary'].item())
                rows.append(row)
                records.append({'candidate': directory.name, **row})
        if not rows:
            continue
        candidates.append({'candidate': directory.name,
            **{key: manifest['config'][key] for key in weights},
            'seeds': manifest['seeds'], 'steps': manifest['steps'],
            'completed': len(rows), 'requested': len(manifest['seeds']),
            'successes': sum(row['strict_success'] for row in rows),
            'mean_final100_mm': float(np.mean([1000*row['mean_last_100'] for row in rows])),
            'fallback_steps': sum(row['fallback_steps'] for row in rows)})
    groups = {}
    by_name = {row['candidate']: row for row in candidates}
    for row in records:
        if row['steps'] == 320:
            config = by_name[row['candidate']]
            key = tuple(config[k] for k in weights)
            groups.setdefault(key, {})[row['seed']] = row
    def matched_panel(panel_seeds):
        matched = []
        for key, rows in groups.items():
            if all(seed in rows for seed in panel_seeds):
                errors = [1000*rows[seed]['mean_last_100'] for seed in panel_seeds]
                matched.append({**dict(zip(weights,key)), 'seeds': panel_seeds,
                    'successes': sum(rows[seed]['strict_success'] for seed in panel_seeds),
                    'mean_final100_mm': float(np.mean(errors)), 'per_seed_final100_mm': errors})
        return sorted(matched, key=lambda row: (-row['successes'], row['mean_final100_mm']))
    result = {'unique_weight_sets': len({tuple(row[k] for k in weights) for row in candidates}),
        'completed_rollouts': len(records), 'successful_rollouts': sum(row['strict_success'] for row in records),
        'all_offline_banks_identical': True, 'selected_candidate': None,
        'weights_status': 'No candidate automatically promoted. Different seed panels and rollout lengths must not be ranked as one comparable experiment.',
        'scope': 'Only Q_xy, R, lambda_g and lambda_ini changed. T_ini=2, T_f=10, raw columns, regional switching, terminal equality and solver fixed.',
        'matched_two_seed_320': matched_panel([211,222]),
        'matched_four_seed_320': matched_panel([211,222,258,281]),
        'matched_eight_seed_320': matched_panel([211,219,222,237,258,281,294,305]),
        'candidates': candidates, 'records': records,
        'interpretation': 'Exploratory development tuning of this Reacher adaptation, not a 100-seed success estimate or evidence against the original paper.'}
    gs.utils.write_json(ROOT/'tuning_report.json', result)
    return result


if __name__ == '__main__':
    report = export()
    print(json.dumps({key: value for key,value in report.items() if key not in ('records','candidates')},indent=2))
