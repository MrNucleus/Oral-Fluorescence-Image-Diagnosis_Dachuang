"""Run the four prepared validation-only repeats sequentially from PyCharm."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / 'data/raw/oral_binary_2026-10-23/口腔整图分类_完整交接包_2026-10-21/01_图片/dataset.json'
    cache = ROOT / 'data/cache/oral_binary_640_v1'
    if not data.is_file() or not (cache / 'manifest.json').is_file():
        raise FileNotFoundError('Frozen data or cache manifest missing')
    for seed, variants in ((43, ('0', '3')), (44, ('3', '0'))):
        print(f'Starting repeat block seed={seed}, variants={variants}', flush=True)
        subprocess.run([sys.executable, '-u', '-B', str(ROOT / 'scripts/run_architecture_ablation.py'),
                        '--data', str(data), '--cache-dir', str(cache), '--workers', '0',
                        '--variants', *variants, '--seed', str(seed)], cwd=ROOT, check=True)
    print('All four validation-only optimization repeats completed.', flush=True)


if __name__ == '__main__':
    main()
