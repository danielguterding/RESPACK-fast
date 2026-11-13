# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
# Patch the Tensile FP16 kernel (clean source) so that every accumulating WMMA computes a fresh 16-term partial
# tile (C = 0) into a temporary tile and the accumulator is updated with v_dual_add_f32 (round to nearest).
#   python3 patch_rn.py clean_symbolic.s patched.s
# The adds of one WMMA are emitted after the next WMMA (two temporary tiles alternate), and all pending adds are
# flushed before labels, branches and any other instruction touching the accumulators v0..v127.
import re
import sys

src, outp = sys.argv[1], sys.argv[2]
TMP = [232, 240]                 # v[232:239], v[240:247]: unused from the first to the last WMMA (checked)
ACC_MAX = 127
wmma = re.compile(r"^\tv_wmma_f32_16x16x16_f16 v\[(\d+):(\d+)\], (v\[\d+:\d+\]), (v\[\d+:\d+\]), (\S+)$")
branch = re.compile(r"^\t(s_branch|s_cbranch\w*|s_setpc_b64|s_swappc_b64|s_endpgm|s_getpc_b64)\b")


def regs(t):
    s = set()
    for m in re.finditer(r"\bv\[(\d+):(\d+)\]", t):
        s.update(range(int(m.group(1)), int(m.group(2)) + 1))
    for m in re.finditer(r"\bv(\d+)\b", t):
        s.add(int(m.group(1)))
    return s


out, pending, k, n_patched, n_flush = [], None, 0, 0, 0


def flush():
    global pending, n_flush
    if pending is None:
        return
    a, t = pending
    for j in range(0, 8, 2):
        out.append("\tv_dual_add_f32 v%d, v%d, v%d :: v_dual_add_f32 v%d, v%d, v%d"
                   % (a + j, a + j, t + j, a + j + 1, a + j + 1, t + j + 1))
    pending = None
    n_flush += 1


for line in open(src).read().split("\n"):
    m = wmma.match(line)
    if m:
        a, b, A, B, C = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), m.group(5)
        if C == "v[%d:%d]" % (a, b) and b <= ACC_MAX:
            t = TMP[k % 2]
            k += 1
            out.append("\tv_wmma_f32_16x16x16_f16 v[%d:%d], %s, %s, 0" % (t, t + 7, A, B))
            flush()                      # adds of the previous WMMA, after this one is issued
            pending = (a, t)
            n_patched += 1
            continue
        flush()                          # a non-accumulating WMMA: keep it, after the pending adds
        out.append(line)
        continue
    if line.endswith(":") or branch.match(line) or (line.startswith("\t") and any(r <= ACC_MAX for r in regs(line))):
        flush()
    out.append(line)
flush()
open(outp, "w").write("\n".join(out) + "\n")
print("patched WMMAs:", n_patched, " add groups:", n_flush)
