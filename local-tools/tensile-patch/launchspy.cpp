// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// LD_PRELOAD shim: print launch configuration and kernel-argument bytes of hipExtModuleLaunchKernel
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly,
                                               uint32_t lz, size_t shmem, hipStream_t st, void **kp, void **extra,
                                               hipEvent_t e0, hipEvent_t e1, uint32_t flags) {
  using fn_t = hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t,
                              void **, void **, hipEvent_t, hipEvent_t, uint32_t);
  static fn_t real = (fn_t)dlsym(RTLD_NEXT, "hipExtModuleLaunchKernel");
  const char *name = hipKernelNameRef(f);
  if (name && strstr(name, "Cijk_")) {
    fprintf(stderr, "LAUNCH %s\n  global %u %u %u  local %u %u %u  shmem %zu  flags %u\n", name, gx, gy, gz, lx, ly, lz, shmem, flags);
    if (extra) {
      void *buf = nullptr; size_t sz = 0;
      for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END; i += 2) {
        if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) buf = extra[i + 1];
        if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) sz = *(size_t *)extra[i + 1];
      }
      fprintf(stderr, "  kernarg %zu bytes:", sz);
      for (size_t i = 0; i < sz && buf; i += 4) fprintf(stderr, "%s%08x", i % 32 ? " " : "\n   ", *(uint32_t *)((char *)buf + i));
      fprintf(stderr, "\n");
    }
  }
  return real(f, gx, gy, gz, lx, ly, lz, shmem, st, kp, extra, e0, e1, flags);
}
