# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
# Clean assembly source of one kernel from its disassembly: all labels from the symbol table (llvm-objdump prints
# only one per address), no addresses. With --symbolic the 8-byte-relative offsets of the s_getpc_b64 jumps
# (s_getpc_b64; s_add_co_i32 sX, LIT, 4; s_add_co_u32 ...; target = getpc address + 8 + LIT) are written as label
# differences, so that code can be inserted.      python3 gen_clean.py k_big.s allsyms.txt out.s [--symbolic]
import collections
import re
import sys

src, symf, outp = sys.argv[1], sys.argv[2], sys.argv[3]
symbolic = "--symbolic" in sys.argv
K = open("kernel_name.txt").read().strip()
ins = []
for line in open(src):
    m = re.match(r"^\t(\S.*?)\s*//\s*([0-9A-F]+):", line)
    if m:
        ins.append([int(m.group(2), 16), m.group(1)])
S, E = ins[0][0], ins[-1][0] + 16
labels = collections.defaultdict(list)
for line in open(symf):
    p = line.split()
    if len(p) >= 8 and re.match(r"^[0-9a-f]{16}$", p[1]) and p[3] in ("NOTYPE", "FUNC"):
        a = int(p[1], 16)
        if S <= a <= E and p[7] != K and p[7] not in labels[a]:
            labels[a].append(p[7])
extra = collections.defaultdict(list)
if symbolic:
    n = 0
    addrs = {a for a, _ in ins}
    for i, (a, t) in enumerate(ins):
        if not t.startswith("s_getpc_b64") or i + 1 >= len(ins):
            continue
        m = re.match(r"s_add_co_i32 (s\d+), (0x[0-9a-f]+|-?\d+), 4$", ins[i + 1][1])
        if not m:
            continue
        lit = int(m.group(2), 0)
        if lit >= 2 ** 31:
            lit -= 2 ** 32
        tgt = a + 8 + lit
        assert tgt in addrs or tgt in labels, hex(tgt)
        if labels.get(tgt):
            tl = labels[tgt][0]
        else:
            tl = "Lgpc_tgt_%d" % n
            extra[tgt].append(tl)
        pl = "Lgpc_after_%d" % n
        extra[a + 4].append(pl)
        ins[i + 1][1] = "s_add_co_i32 %s, %s-%s-4, 4" % (m.group(1), tl, pl)
        n += 1
    print("symbolic getpc offsets:", n)
out = []
for a, t in ins:
    for l in labels.get(a, []) + extra.get(a, []):
        out.append(l + ":")
    out.append("\t" + t)
for a in sorted(set(labels) | set(extra)):
    if a > ins[-1][0]:
        for l in labels.get(a, []) + extra.get(a, []):
            out.append(l + ":")
open(outp, "w").write("\n".join(out) + "\n")
print(outp, len(ins), "instructions,", sum(len(v) for v in labels.values()), "labels")
