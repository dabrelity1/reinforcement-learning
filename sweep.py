from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, help='Base config JSON to start from', required=False)
    p.add_argument('--grid', type=str, help='JSON mapping of param -> list of values', required=True)
    p.add_argument('--runs-root', type=str, default='runs')
    p.add_argument('--python', type=str, default=sys.executable)
    args = p.parse_args()

    base = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            base = json.load(f)

    grid = json.loads(args.grid)
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    for combo in itertools.product(*values):
        cfg = dict(base)
        name_suffix = []
        for k, v in zip(keys, combo):
            cfg[k] = v
            name_suffix.append(f"{k}-{v}")
        cfg['run_name'] = f"sweep-{'_'.join(name_suffix)}"

        tmp = Path('presets') / '._sweep_tmp.json'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(cfg, f)

        cmd = [args.python, 'train.py', '--config', str(tmp)]
        print('Launching:', ' '.join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print('Run failed with code', proc.returncode)

    print('Sweep complete')


if __name__ == '__main__':
    main()
