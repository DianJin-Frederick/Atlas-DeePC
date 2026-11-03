"""Reproduce the frozen NSR comparison without repeating parameter searches."""
import argparse
import json
from pathlib import Path
import shutil

import refine_f1tenth_paper as study
from atlas_deepc.f1tenth import task
from atlas_deepc.f1tenth.config import NOISE_SIGMAS
from atlas_deepc.f1tenth.paths import RESULTS_DIR, noise_tag


def run_paper(output=RESULTS_DIR/'paper_nsr_rerun',
              selection=RESULTS_DIR/'paper_nsr_20261027'/'selection', seeds=study.TEST):
    output=Path(output);selection=Path(selection);seeds=tuple(seeds)
    if seeds!=study.TEST:
        raise ValueError('The frozen paper table uses the paired seeds 531 through 550.')
    choices={s:study.read(selection/f'sigma_{noise_tag(s)}.json') for s in NOISE_SIGMAS}
    if any(c['sigma']!=s or not c['frozen_before_paper_seeds'] for s,c in choices.items()):
        raise ValueError('Invalid frozen configuration selection.')
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(sigmas=list(NOISE_SIGMAS),paper_seeds=list(seeds),
              selected_methods={noise_tag(s):c['method'] for s,c in choices.items()},
              contextual_columns=[30,60,90],selection_source=str(selection.resolve()),
              study='Reproduction of frozen configurations; no parameter tuning.')
    file=output/'reproduction_protocol.json'
    if file.exists() and study.read(file)!=plan:
        raise ValueError('Output already contains a different reproduction protocol.')
    task.write(file,plan)
    for sigma,choice in choices.items():
        task.write(output/'selection'/f'sigma_{noise_tag(sigma)}.json',choice)
    for name in ('run_f1tenth_paper.py','refine_f1tenth_paper.py'):
        shutil.copyfile(Path(__file__).with_name(name),output/name)
    study.ROOT=output
    for sigma,choice in choices.items():
        study.stage(output/f'sigma_{noise_tag(sigma)}',sigma,
                    [choice['method'],*study.CONTEXTUAL],seeds)
        study.aggregate()
    task.write(output/'status.json',dict(status='complete',levels=len(NOISE_SIGMAS),
                                       rollouts=len(seeds)*4*len(NOISE_SIGMAS)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RESULTS_DIR/'paper_nsr_rerun')
    parser.add_argument('--selection',type=Path,default=RESULTS_DIR/'paper_nsr_20261027'/'selection')
    args=parser.parse_args()
    run_paper(args.output,args.selection)
