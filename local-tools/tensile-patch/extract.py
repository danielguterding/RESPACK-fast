# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
# Extract one kernel of a code object into a working directory: disassembly of its address range, kernel
# descriptor, metadata, clean symbolic source, round-trip and RN-patched code objects, and a register report.
#   python3 extract.py <code object> <allsyms.txt> <notes.txt> <kernel name> <workdir>
import collections
import os
import re
import subprocess
import sys

co, symf, notesf, K, wd = sys.argv[1:6]
L = "/opt/rocm/llvm/bin/"
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(wd, exist_ok=True)
co, symf, notesf = os.path.abspath(co), os.path.abspath(symf), os.path.abspath(notesf)

# address range: kernel start .. next function
starts = sorted({int(p[1], 16) for p in (l.split() for l in open(symf)) if len(p) >= 8 and p[3] == "FUNC"
                 and re.match(r"^[0-9a-f]{16}$", p[1])})
addr = [int(p[1], 16) for p in (l.split() for l in open(symf)) if len(p) >= 8 and p[3] == "FUNC" and p[7] == K][0]
end = [a for a in starts if a > addr][0]
os.chdir(wd)
open("kernel_name.txt", "w").write(K + "\n")
subprocess.run("%sllvm-objdump -d --mcpu=gfx1200 --start-address=0x%x --stop-address=0x%x %s > k.s"
               % (L, addr, end, co), shell=True, check=True)
subprocess.run('%sllvm-objdump -D --disassemble-symbols="%s.kd" --mcpu=gfx1200 %s > kd_raw.txt' % (L, K, co),
               shell=True, check=True)
open("kd.txt", "w").write("\n".join(l.strip() for l in open("kd_raw.txt") if l.strip().startswith(".amdhsa")) + "\n")
t = open(notesf).read()
i = t.find(".name:           " + K + "\n")
s = t.rfind("\n  - ", 0, i)
e = t.find("\n  - ", i)
blk = []
for l in t[s + 1:e].split("\n"):
    if l and not l.startswith(" "):
        break
    blk.append(l)
open("kernel_meta.txt", "w").write("\n".join(blk) + "\n")
vg = re.search(r"\.vgpr_count:\s+(\d+)", "\n".join(blk)).group(1)

py = sys.executable
subprocess.run([py, os.path.join(here, "gen_clean.py"), "k.s", symf, "clean.s", "--symbolic"], check=True,
               stdout=subprocess.DEVNULL)
lines = open("clean.s").read().split("\n")
wm = [i for i, l in enumerate(lines) if "v_wmma" in l]


def regs(op):
    r = set()
    for m in re.finditer(r"\bv\[(\d+):(\d+)\]", op):
        r.update(range(int(m.group(1)), int(m.group(2)) + 1))
    for m in re.finditer(r"\bv(\d+)\b", op):
        r.add(int(m.group(1)))
    return r


acc = set()
for i in wm:
    m = re.match(r"\tv_wmma\S* v\[(\d+):(\d+)\]", lines[i])
    acc.update(range(int(m.group(1)), int(m.group(2)) + 1))
free = []
for b in range(0, 256, 8):
    R = set(range(b, b + 8))
    ok, first_after = True, None
    for i, l in enumerate(lines):
        tt = l.strip()
        if not tt or tt.endswith(":"):
            continue
        p = tt.split(None, 1)
        if len(p) < 2:
            continue
        ops = [o.strip() for o in re.split(r",(?![^\[]*\])", p[1])]
        st = p[0].startswith(("global_store", "buffer_store", "ds_store", "s_"))
        dst = set() if st else regs(ops[0])
        src = set().union(*[regs(o) for o in ops[1:]]) if len(ops) > 1 else set()
        if st:
            src |= regs(p[1])
        if not ((dst | src) & R):
            continue
        if i <= wm[-1]:
            ok = False
            break
        if first_after is None:
            first_after = "W" if (dst & R and not src & R) else "R"
            if first_after == "R":
                ok = False
            break
    if ok:
        free.append(b)
print("%s: vgpr_count %s, %d WMMA, accumulators v[%d:%d], free groups of 8: %s"
      % (wd, vg, len(wm), min(acc), max(acc), free))
open("free_groups.txt", "w").write(",".join(str(b) for b in free) + "\n")
