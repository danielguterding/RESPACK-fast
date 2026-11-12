#!/usr/bin/env python3
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
"""Compare RESPACK binary dir-eps/qNNN/dat.epsqw.600 files (one Fortran record, complex128 NG*NG*nen).
usage: cmp_epsqw.py <dir-eps-a> <dir-eps-b>   (compares all common qNNN subdirectories)"""
import os, sys
import numpy as np

def read_rec(path):
    with open(path, 'rb') as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.complex128, count=n // 16)
        n2 = np.fromfile(f, dtype=np.int32, count=1)[0]
    assert n == n2, 'bad record markers'
    return data

a, b = sys.argv[1], sys.argv[2]
worst = 0.0
for q in sorted(d for d in os.listdir(a) if d.startswith('q')):
    pa, pb = os.path.join(a, q, 'dat.epsqw.600'), os.path.join(b, q, 'dat.epsqw.600')
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    xa, xb = read_rec(pa), read_rec(pb)
    d = np.abs(xa - xb)
    rel = d.max() / np.abs(xa).max()
    worst = max(worst, rel)
    print(f'{q}: n={xa.size:9d}  max|d|={d.max():.3e}  max|d|/max|x|={rel:.3e}  rms(d)/rms(x)={np.sqrt((d**2).mean()/(np.abs(xa)**2).mean()):.3e}')
print(f'worst relative deviation: {worst:.3e}')
