"""Recreate the retained 300-second random-excitation dataset at a new path."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas_deepc.f1tenth.task import collect
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Choose a new output file; existing data are never overwritten.')
    record=collect(args.output)
    print(f"Saved {record['Y'].shape[0]*record['Y'].shape[1]} samples to {args.output}")
