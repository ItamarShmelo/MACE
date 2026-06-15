"""
Launcher – run all 4 mega-validation parts in parallel.

Each part runs as a separate process.  stdout/stderr for each is
captured to a log file under ``reports/generated/logs/``.

Usage:
    python3 reports/mega_validation/run_all.py [--plot-tier standard] [--no-cache]

    Extra CLI flags are forwarded to every part script.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(HERE, '..', 'generated')

PARTS = [
    'part1_sigma_comparison.py',
    'part2_derivative_comparison.py',
    'part3_det_convergence.py',
    'part4_mc_convergence.py',
]


def main():
    extra_args = sys.argv[1:]

    log_dir = os.path.join(GEN_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    t0 = time.time()
    procs = []

    for p in PARTS:
        script = os.path.join(HERE, p)
        log_path = os.path.join(log_dir, f'{p}.log')
        log_fh = open(log_path, 'w')
        proc = subprocess.Popen(
            [sys.executable, script] + extra_args,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=HERE,
        )
        procs.append((p, proc, log_path, log_fh))
        print(f'Launched {p}  (PID {proc.pid})  ->  {log_path}')

    print()
    print('All 4 parts launched.')
    print('NOTE: timing data in individual reports reflects concurrent-load')
    print('conditions (4 processes sharing CPU/memory bandwidth).')
    print('For isolated timing measurements, run parts individually.')
    print()

    for name, proc, log_path, log_fh in procs:
        proc.wait()
        log_fh.close()
        status = 'OK' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
        print(f'  {name}: {status}  ({log_path})')

    elapsed = time.time() - t0
    print(f'\nTotal wall-clock: {elapsed/3600:.1f} hours ({elapsed:.0f}s)')

    any_failed = any(proc.returncode != 0 for _, proc, _, _ in procs)
    sys.exit(1 if any_failed else 0)


if __name__ == '__main__':
    main()
