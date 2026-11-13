# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
# Rebuild an assembly source of one Tensile kernel from its disassembly, kernel descriptor (kd.txt) and
# metadata (kernel_meta.txt), for llvm-mc:   python3 mk_asm.py <code.s> <out.s> [kernel-name-suffix]
import re
import sys

src, outp = sys.argv[1], sys.argv[2]
suffix = sys.argv[3] if len(sys.argv) > 3 else ""
K = open("kernel_name.txt").read().strip()
KN = K + suffix
code = []
for line in open(src).read().split("\n"):
    m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
    if m:
        if m.group(1) != K:
            code.append(m.group(1) + ":")
        continue
    m = re.match(r"^\t(\S.*?)\s*//\s*[0-9A-F]+:", line)      # disassembly line with address comment
    if m:
        code.append("\t" + m.group(1))
        continue
    m = re.match(r"^(\S+):\s*$", line)                         # plain label (patched source)
    if m:
        code.append(m.group(1) + ":")
        continue
    m = re.match(r"^\t(\S.*?)\s*$", line)                      # plain instruction (patched source)
    if m and not m.group(1).startswith("..."):
        code.append("\t" + m.group(1))

kd = [l.strip() for l in open("kd.txt")
      if l.strip().startswith(".amdhsa_") and not l.strip().startswith(".amdhsa_kernel")]
meta = [l for l in open("kernel_meta.txt").read().split("\n") if l.strip()]
meta = [l.replace(K, KN) for l in meta]

o = [".amdhsa_code_object_version 4",
     '.amdgcn_target "amdgcn-amd-amdhsa--gfx1200"',
     ".text", ".protected " + KN, ".globl " + KN, ".p2align 8", ".type " + KN + ",@function", KN + ":"]
o += code
o += [".Lfunc_end0:", ".size " + KN + ", .Lfunc_end0-" + KN,
      ".rodata", ".p2align 6", ".amdhsa_kernel " + KN]
o += ["  " + l for l in kd]
o += [".end_amdhsa_kernel", ".amdgpu_metadata", "---", "amdhsa.kernels:"]
o += meta
o += ["amdhsa.target:   amdgcn-amd-amdhsa--gfx1200", "amdhsa.version:", "  - 1", "  - 1", "...",
      ".end_amdgpu_metadata"]
open(outp, "w").write("\n".join(o) + "\n")
print(outp, len(code), "code lines")
