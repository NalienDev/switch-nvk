#!/usr/bin/env bash
# M-DV-0 — package our NVK as a Dawn-consumable install tree:  nvk-switch/{lib/libvulkan.a, include/}
# (see PLAN_DAWN_VULKAN.md). Mirrors winsys/build-exe-linktest.sh's authoritative archive set + shim
# compile flags, then MRI-merges everything into ONE fat libvulkan.a so Dawn's Switch Vulkan backend
# (which expects a single static libvulkan.a + vk_icdGetInstanceProcAddr) can link it.
#   docker run --rm -v "D:\switch-nvk:/work" -w /work switch-nvk-build bash package-nvk.sh
set -e

DKP=/opt/devkitpro
GCC=$DKP/devkitA64/bin/aarch64-none-elf-gcc
AR=$DKP/devkitA64/bin/aarch64-none-elf-ar
NM=$DKP/devkitA64/bin/aarch64-none-elf-nm
MB=/work/mb
OUT=/work/nvk-switch
OBJ=/tmp/nvk-pkg
mkdir -p "$OBJ" "$OUT/lib" "$OUT/include/vulkan"

ARCH="-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE"
INC="-I$DKP/libnx/include -I/opt/switch-cross-include -Imesa-25/include -Imesa-25/src/nouveau/drm -Iwinsys -Icompat"
DEFS="-D__SWITCH__ -D_GNU_SOURCE -D_DEFAULT_SOURCE -include /work/compat/switch_compat.h"

# Set DRM_SHIM_DEBUG=1 in the environment to compile the drm_shim with its
# per-op trace enabled (writes sdmc:/nvk_drmshim.log). Off by default (release).
SHIM_DEBUG_DEFS="${DRM_SHIM_DEBUG:+-DDRM_SHIM_DEBUG}"

echo "=== [1/4] compiling winsys shims (drm_shim + libc shim + compat) ==="
$GCC -c winsys/drm_shim.c           -o "$OBJ/drm_shim.o"           $ARCH $DEFS $SHIM_DEBUG_DEFS $INC -O2
$GCC -c winsys/switch_libc_shim.c   -o "$OBJ/switch_libc_shim.o"   $ARCH $DEFS $INC -O2
$GCC -c compat/compat.c             -o "$OBJ/compat.o"            $ARCH $DEFS $INC -O2
# loaderless-ICD glue: provides __wrap_vk_icdGetInstanceProcAddr (needs -Wl,--wrap at the consumer
# link) returning the loader-managed global procs Mesa NVK doesn't expose. Mirrors Dan's nvk_loaderless_vk.c.
$GCC -c winsys/nvk_loaderless_shim.c -o "$OBJ/nvk_loaderless_shim.o" $ARCH $DEFS $INC -O2

# The driver: libnvk.a (the ICD, holds vk_icdGetInstanceProcAddr) + the 18 support archives.
# Set + order verbatim from winsys/build-exe-linktest.sh (the proven symbol-complete link).
ARCHIVES="
  src/nouveau/vulkan/libnvk.a
  src/nouveau/codegen/libnouveau_codegen.a
  src/util/libmesa_util.a src/util/libmesa_util_sse41.a src/util/blake3/libblake3.a
  src/c11/impl/libmesa_util_c11.a
  src/nouveau/compiler/libnak.a src/nouveau/compiler/libnak_rs.a
  src/compiler/rust/libcompiler_c_helpers.a
  src/nouveau/headers/libnvidia_headers_c.a
  src/nouveau/nil/liblibnil.a src/nouveau/nil/liblibnil_format_table.a
  src/compiler/nir/libnir.a src/compiler/libcompiler.a
  src/nouveau/mme/libnouveau_mme.a src/nouveau/winsys/libnouveau_ws.a
  src/vulkan/util/libvulkan_util.a src/compiler/spirv/libvtn.a
  src/util/libxmlconfig.a"

echo "=== [2/4] MRI-merging driver archives + shims -> $OUT/lib/libvulkan.a ==="
rm -f "$OUT/lib/libvulkan.a"
{
  echo "CREATE $OUT/lib/libvulkan.a"
  for a in $ARCHIVES; do
    [ -f "$MB/$a" ] || { echo "MISSING ARCHIVE: $MB/$a" >&2; exit 1; }
    echo "ADDLIB $MB/$a"
  done
  echo "ADDMOD $OBJ/drm_shim.o"
  echo "ADDMOD $OBJ/switch_libc_shim.o"
  echo "ADDMOD $OBJ/compat.o"
  echo "ADDMOD $OBJ/nvk_loaderless_shim.o"
  echo "SAVE"
  echo "END"
} | $AR -M
$AR s "$OUT/lib/libvulkan.a"   # regenerate the symbol index over the merged members

echo "=== [3/4] staging Vulkan headers -> $OUT/include/vulkan ==="
# Our driver was built against mesa-25's bundled Vulkan headers; ship those so a consumer that
# includes <vulkan/...> against our ICD sees the same ABI. (Dawn vendors its own headers too.)
for h in vulkan.h vulkan_core.h vk_icd.h vk_platform.h vulkan_vi.h; do
  if [ -f "mesa-25/include/vulkan/$h" ]; then
    cp "mesa-25/include/vulkan/$h" "$OUT/include/vulkan/$h"
  else
    echo "  (note: mesa-25/include/vulkan/$h not found — consumer's own headers will be used)"
  fi
done
# vk_video/ lives at include/vk_video/ (sibling of vulkan/); vulkan_core.h includes it via -Iinclude.
if [ -d mesa-25/include/vk_video ]; then
  rm -rf "$OUT/include/vk_video"; cp -r mesa-25/include/vk_video "$OUT/include/vk_video"
fi

echo "=== [4/4] VERIFY symbols (nm) ==="
# The loaderless-ICD contract: the ONLY strong exported entry is vk_icdGetInstanceProcAddr; every
# vk* API (vkCreateInstance, vkCreateViSurfaceNN, ...) is reached via DISPATCH, not a strong T symbol.
# So we check (a) the ICD entry + our wrap shims are DEFINED, (b) the ViSurfaceNN ENTRYPOINT object is
# present (so dispatch can return it), (c) a whole-archive link is self-consistent (the real proof).
echo "--- exported ICD entry + wrap shims (must be T): ---"
$NM "$OUT/lib/libvulkan.a" 2>/dev/null | grep -E ' T (vk_icdGetInstanceProcAddr|__wrap_open|__wrap_close|__wrap_stat|__wrap_lstat)$' | sort -u || true
echo "--- VI-surface entrypoint present (reached via dispatch, not a public T): ---"
$NM "$OUT/lib/libvulkan.a" 2>/dev/null | grep -iE 'CreateViSurfaceNN|wsi_switch_init_wsi' | sort -u | head || true
echo -n "vk_icdGetInstanceProcAddr defined? : "; $NM "$OUT/lib/libvulkan.a" 2>/dev/null | grep -cE ' T vk_icdGetInstanceProcAddr$' || true
echo -n "total defined-text symbols         : "; $NM "$OUT/lib/libvulkan.a" 2>/dev/null | grep -cE ' [Tt] ' || true
echo -n "archive size                       : "; du -h "$OUT/lib/libvulkan.a" | cut -f1
echo "=== PACKAGE DONE -> $OUT ==="
ls -la "$OUT/lib" "$OUT/include/vulkan"
echo ""
echo "NOTE: validate self-consistency with a whole-archive link test:  bash _linktest_merged.sh"
