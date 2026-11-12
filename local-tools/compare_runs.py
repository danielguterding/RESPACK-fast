#!/usr/bin/env python3
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
"""Compare two RESPACK run directories numerically (text outputs token-wise; binary files only checked for byte identity).
usage: compare_runs.py <run-a> <run-b>
"""
import os
import re
import sys

import numpy as np

NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?')
SUBDIRS = ['dir-wan', 'dir-intW', 'dir-intJ', 'dir-model', 'dir-eps']


def numbers(path):
    with open(path, 'r', errors='replace') as f:
        txt = f.read().replace('D', 'E').replace('d', 'e')
    return np.array([float(x) for x in NUM.findall(txt)])


def is_text(path):
    with open(path, 'rb') as f:
        chunk = f.read(4096)
    return b'\0' not in chunk


def main(a, b):
    worst = []
    nfiles = 0
    nbin = [0, 0]  # binary files: total, not byte-identical
    for sub in SUBDIRS:
        for root, _, files in os.walk(os.path.join(a, sub)):
            for fn in files:
                pa = os.path.join(root, fn)
                pb = os.path.join(b, os.path.relpath(pa, a))
                if not os.path.exists(pb):
                    print('MISSING in b:', os.path.relpath(pa, a))
                    continue
                nfiles += 1
                if not is_text(pa):
                    nbin[0] += 1
                    if open(pa, 'rb').read() != open(pb, 'rb').read():
                        nbin[1] += 1
                    continue
                xa, xb = numbers(pa), numbers(pb)
                if xa.shape != xb.shape:
                    print('SHAPE MISMATCH', os.path.relpath(pa, a), xa.shape, xb.shape)
                    continue
                if xa.size == 0:
                    continue
                d = np.abs(xa - xb)
                scale = max(np.abs(xa).max(), 1e-300)
                worst.append((d.max(), d.max() / scale, os.path.relpath(pa, a)))
    worst.sort(reverse=True)
    print(f'compared {nfiles - nbin[0]} text files numerically; binary files: {nbin[0]} ({nbin[1]} not byte-identical)')
    print('largest absolute differences (abs, abs/max|value|, file):')
    for w in worst[:12]:
        print(f'  {w[0]:.3e}  {w[1]:.3e}  {w[2]}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
