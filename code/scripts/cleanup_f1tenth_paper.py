"""Remove superseded F1TENTH raw results after verifying the retained table."""
import json
from pathlib import Path
import shutil

from atlas_deepc.f1tenth import task
from atlas_deepc.f1tenth.paths import RESULTS_DIR, DATA_DIR, noise_tag
from atlas_deepc.f1tenth.config import NOISE_SIGMAS
from atlas_deepc.f1tenth.runner import verify_inventory

PAPER=RESULTS_DIR/'paper_nsr_20261027'
SUPERSEDED=(
    'contextual_comparison_20261027', 'partial_projection_20261027',
    'noise_ratio_sweep_20261027', 'noise025_kr_search_20261027',
    'noise025_kr_fine_20261027',
)


def read(p):
    return json.loads(p.read_text())


def main():
    if (PAPER/'cleanup_status.json').exists() and read(PAPER/'cleanup_status.json')['complete']:
        print(json.dumps(read(PAPER/'cleanup_status.json'),indent=2))
        return
    status=read(PAPER/'status.json')
    assert status['status']=='complete' and status['rollouts']==560
    for sigma in NOISE_SIGMAS:
        folder=PAPER/f'sigma_{noise_tag(sigma)}'
        protocol=read(folder/'protocol.json');audit=read(folder/'validation.json')
        assert protocol['seeds']==list(range(531,551)) and len(protocol['methods'])==4
        assert audit['complete'] and audit['rollouts']==80
        assert audit['offline_and_online_noise_verified'] and audit['clustering_uses_noisy_speed_verified']
        assert audit['maximum_replay_state_mismatch']==0
        verify_inventory(folder)
        # Keep aggregate search provenance, not discarded candidate trajectories.
        ledger=PAPER/'development'/f'sigma_{noise_tag(sigma)}'/'search_ledger.json'
        if ledger.exists():
            target=PAPER/'selection_audit'/ledger.parent.name/'search_ledger.json'
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ledger,target)
    targets=[RESULTS_DIR/name for name in SUPERSEDED]+[PAPER/'development',
             DATA_DIR/'offline_300_seconds_noisy_seed3408.npz',
             DATA_DIR/'offline_300_seconds_noisy_seed3408_manifest.json']
    inventory=[]
    for target in targets:
        if not target.exists():
            continue
        files=[p for p in target.rglob('*') if p.is_file()] if target.is_dir() else [target]
        inventory.append(dict(path=str(target),files=len(files),bytes=sum(p.stat().st_size for p in files),
                              file_manifest=[dict(path=str(p.relative_to(target)) if target.is_dir() else p.name,
                                                  bytes=p.stat().st_size,sha256=task.digest(p)) for p in files]))
    plan=dict(retained=str(PAPER),retained_rollouts=560,removed=inventory,
              reason='User requested only paper-presented F1TENTH data; all selected rows and all their failures are retained. Compact tuning provenance is retained for transparency.',
              retained_dependencies='Clean training source, seven noisy datasets, reference, map, simulator and backend methods are necessary to reproduce the table.')
    task.write(PAPER/'cleanup_manifest.json',plan)
    for item in inventory:
        target=Path(item['path'])
        assert (target.parent==RESULTS_DIR or target==PAPER/'development' or target.parent==DATA_DIR)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    task.write(PAPER/'cleanup_status.json',dict(complete=True,removed_files=sum(x['files'] for x in inventory),
               removed_bytes=sum(x['bytes'] for x in inventory),retained_rollouts=560))
    print(json.dumps(read(PAPER/'cleanup_status.json'),indent=2))


if __name__=='__main__':
    main()
