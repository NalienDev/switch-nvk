# BUILD_AND_RUN.md — everything needed to build & run NVK Vulkan on the Switch

This is the complete, self-contained recipe to reproduce the **M2 smoke test that PASSED on real
Tegra** (our own Mesa NVK Vulkan driver executing `vkCmdFillBuffer(0xCAFEBABE)` on the GM20B and reading
it back). Standalone effort — does NOT involve Dusklight. Live status: `RESUME_NVK.md`.

---

## 0. What you need (host + hardware)

**Host machine:**
- **Docker** (the ONLY host build dependency — the toolchain image carries everything else).
- ~15 GB disk for the image + Mesa tree + build dir.
- Windows/Linux/macOS (commands below shown for Windows PowerShell; on Linux swap the `-v` path).

**To RUN it (test on hardware):**
- A **Nintendo Switch with CFW** (Atmosphère) — Erista or Mariko, any that runs homebrew.
- **Sphaira** (homebrew launcher) with its **FTP server** (port 5000) — used to upload the `.nro`
  and fetch the log. ("boot as application" launch mode is required; the album-applet has too little
  memory for NVK channel init.)
- The Switch and PC on the **same network**.

**You do NOT need:** devkitPro installed on the host, an NVIDIA GPU, Dan's private NVK image, deko3d, or
the NVIDIA L4T blob. Everything is built from open Mesa + libnx via the Docker image.

---

## 1. The toolchain Docker image (`Dockerfile`)

`docker build -t switch-nvk-build -f Dockerfile .`

Base = `devkitpro/devkita64` (devkitA64 `aarch64-none-elf` GCC + libnx + the meson cross helper).
On top it installs (see `Dockerfile` for exact versions):

| Dependency | Why |
|---|---|
| **meson ≥1.4 + mako + pyyaml** | Mesa's build system. |
| **Rust nightly + rust-src** | NVK's shader compiler **NAK** is Rust; the Switch is a tier-3 target (`aarch64-nintendo-switch-freestanding`) needing `-Zbuild-std`. |
| **bindgen-cli** | NAK's C↔Rust FFI. |
| **cbindgen** | **NIL** (nouveau image-layout lib) generates C headers from Rust. |
| **rustfmt** | NAK formats its generated bindings; meson errors without it. |
| **LLVM-15 + clang-15 + libclang-15** | `mesa_clc` (host tool that compiles NVK's OpenCL kernels) needs LLVM≥15. |
| **libclc-15 + libclc-15-dev** | CLC headers + the runtime `spirv-mesa3d.spv` (in the RUNTIME pkg, not -dev — a known trap). |
| **llvm-spirv-15 + libllvmspirvlib-15-dev** | LLVM-IR → SPIR-V translator for `mesa_clc`. |
| **spirv-tools, libdrm-dev** | SPIR-V tooling; libdrm headers (`xf86drm.h` + drm-uapi) for the winsys. |
| **bison/flex/pkg-config/cmake/build-essential** | Mesa codegen + native tools. |

The image also **bakes** the CLC/SPIR-V `.pc` files into the Switch **cross** pkg-config dir (so cross-
configure is self-contained), copies the libdrm headers to `/opt/switch-cross-include` (NOT `/usr/include`,
which would shadow newlib for the cross compiler), and creates an empty `libdl.a` (the Switch is static —
no `dlopen`). Re-run `docker build` only when the `Dockerfile` changes (layers cache).

---

## 2. Get the Mesa source (gitignored — must be fetched)

`mesa-25/` is **NOT in this repo** (it's a large, reconstructable upstream tree — see `.gitignore`).
Obtain Mesa **25.0.7** and apply our patches:

```powershell
# Download + extract Mesa 25.0.7 into mesa-25/ (e.g. from https://archive.mesa3d.org/mesa-25.0.7.tar.xz)
#   tar xf mesa-25.0.7.tar.xz ; rename the extracted dir to  mesa-25
# Then apply the 9 Switch source patches (idempotent):
docker run --rm -v "D:\switch-nvk:/work" -w /work switch-nvk-build bash /work/apply-patches.sh
# Then install the VI/nwindow WSI backend + zero-copy present + the mesa source overrides
# (idempotent). REQUIRED — without it there is no swapchain/present and no zero-copy:
docker run --rm -v "D:\switch-nvk:/work" -w /work switch-nvk-build bash /work/winsys/wsi/apply-wsi-switch.sh
```

The authoritative patches live in `patches/switch-nvk-mesa-25.0.7.patch` (9 files, including the **FECS
no-op** in `nvk_cmd_draw.c` that is essential to the PASS). `pristine-25.0.7/` holds the unmodified
upstreams for re-diffing. `apply-patches.sh` is idempotent (skips if `DETECT_OS_HORIZON` is already
present). The 9 patched files and why are listed in `RESUME_NVK.md` → "The source patches".

**`apply-wsi-switch.sh`** installs the `VK_NN_vi_surface`/`nwindow` WSI backend
(`winsys/wsi/wsi_common_switch.c` — incl. the **zero-copy block-linear present**, `kind=0xfe`) plus the
verbatim source overrides under **`winsys/mesa-edits/`** (`nvk_image.c` — the load-bearing
`nvk_switch_image_layout` NIL-layout helper; `nvk_queue.c`/`wsi_common.c` — the `NVK_TRACE`-gated debug
logs). These are committed VERBATIM (not patched) to dodge CRLF/patch fuzz; the script `cp`s them over
the extracted tree and is idempotent. Zero-copy details: `RESUME_NVK.md` → "Zero-copy WSI".

---

## 3. Build (2-phase Mesa cross-build → static libs)

```powershell
docker run --rm -v "D:\switch-nvk:/work" -w /work/mesa-25 switch-nvk-build bash -lc '
  export NATIVE_PREFIX=/work/native-prefix
  export PATH=/work/native-prefix/bin:$PATH
  export BUILD=/work/mb
  bash /work/build-native-tools.sh   # PHASE 1: native (x86) mesa_clc + precomp compiler (cached/idempotent)
  bash /work/configure-mesa.sh       # PHASE 2: cross-configure NVK for the Switch (meson)
  ninja -C /work/mb                  # compile -> static libs in /work/mb/src/**/lib*.a
'
```

**KEY cross-build insight:** in a cross build, `-Dmesa-clc=enabled` wrongly compiles LLVM/CLC FOR the
target. The 2-phase recipe avoids it: build `mesa_clc`+precomp **natively** (phase 1, `build-native-tools.sh`),
then cross-configure with `-Dmesa-clc=system -Dprecomp-compiler=system -Dllvm=disabled` (phase 2,
`configure-mesa.sh`). Output: `libnvk.a` + ~20 static libs (NAK/NIL Rust libs, vulkan runtime/util/wsi, etc.).

⚠️ **Docker bind-mount mtimes can make ninja SKIP an edited Mesa file** → if you change a Mesa source,
delete its `.o` to force recompile, and verify the `.a` relinked.

---

## 4. Build the runnable `.nro` (`winsys/build-nro.sh`)

```powershell
# DRM_SHIM_DEBUG=1 enables the drm_shim trace in the log — ALWAYS use it for diagnosis.
docker run --rm -e DRM_SHIM_DEBUG=1 -v "D:\switch-nvk:/work" -w /work switch-nvk-build bash winsys/build-nro.sh
# => D:\switch-nvk\nvk_smoke.nro   (~12.6 MB, full Application)
```

This recompiles the winsys shim + the smoke app fresh and links them with the whole-archived `libnvk.a`
+ the static libs into a Switch `.nro`. It does NOT rebuild Mesa (step 3 already did). The pieces:
- `winsys/drm_shim.c` — the winsys: every `drm*`/nouveau ioctl over libnx `nv` (GEM/VM_BIND/EXEC/CHANNEL/
  fence). **This is the heart of the port.**
- `winsys/switch_libc_shim.c` — the `open`/`mmap` wrap + libc/std gaps.
- `winsys/smoke/nvk_smoke.c` — the headless smoke test (instance→device→fill→submit→readback).

**Verify which binary you built** (the build can silently omit `DRM_SHIM_DEBUG`): the `.nro` should
contain the strings `EXEC drain` / the BUILD tag. The smoke logs a `[BUILD vNN ...]` header line and the
nacp version (shown in Sphaira) so you can confirm the running binary.

---

## 5. Run on the Switch + read the log

```powershell
# 1. Upload into the favorited launch path (NOT sdmc root):
curl.exe -T "D:\switch-nvk\nvk_smoke.nro" "ftp://<SWITCH-IP>:5000/sdmc:/switch/nvk_smoke.nro"
# 2. On the Switch: open Sphaira -> launch nvk_smoke -> "boot as application".
# 3. It runs headless and exits; fetch the log:
curl.exe "ftp://<SWITCH-IP>:5000/sdmc:/nvk_smoke.log" -o nvk_smoke.log
```

**A PASS looks like:**
```
I VERIFY OK: all 1024 words == 0xcafebabe
=== SMOKE TEST PASSED — NVK rendered to memory on Tegra ===
```

**Workflow notes:**
- `nxlink -s` netloader is **flaky** (killing it wedges the Switch netloader) — use the FTP path above.
- **Always read the WHOLE log, every run** (standing rule — see `docs/knowledge/feedback_nvk_read_full_log.md`).
- If the app hangs, HOME → close it → reopen Sphaira to flush the log.

---

## 6. File map (what's in this repo)

| Path | Role |
|---|---|
| `Dockerfile` | The toolchain image `switch-nvk-build` (§1). |
| `apply-patches.sh` | Re-apply the 9 Mesa source patches to a fresh `mesa-25/` (§2). |
| `patches/switch-nvk-mesa-25.0.7.patch` | The authoritative source patches (incl. the FECS no-op). |
| `pristine-25.0.7/` | Unmodified upstream copies of the patched files (for re-diffing). |
| `build-native-tools.sh` | Phase 1: native `mesa_clc` + precomp (§3). |
| `configure-mesa.sh` | Phase 2: cross-configure NVK for the Switch (§3). |
| `winsys/drm_shim.{c,h}` | **The winsys** — all `drm*`/nouveau ioctls over libnx `nv`. |
| `winsys/switch_libc_shim.c` | `open`/`mmap` wrap + libc/std gaps. |
| `winsys/smoke/nvk_smoke.c` | The headless smoke test. |
| `winsys/build-nro.sh` | Build the `.nro` (§4). |
| `winsys/build-exe-linktest.sh` | Strict EXE link harness (symbol-completeness check). |
| `compat/` | Force-included shim header + stub headers + `compat.c` for newlib gaps. |
| `aarch64-switch-horizon.json`, `rustc-switch.sh`, `build-std-sysroot.sh` | The Rust-for-Switch toolchain. |
| `RESUME_NVK.md` | **Live source of truth** — status, the full symbol spec, next steps. |
| `PLAN_NVK.md` | Original phased plan (M0→M3). |
| `docs/knowledge/` | The accumulated research/debugging knowledge (memory + patterns + skill). |

---

## 7. Status & what's next

- **DONE:** M0 (Rust std), M1 (NVK cross-builds + links into a Switch NRO), **M2 (memory + submit +
  execute + coherency) PROVEN on real Tegra** — the smoke test PASSES.
- **NEXT — M3 (the visual part):** a real `vkCmdDraw` triangle into a VkImage + **WSI/present** over
  libnx `nwindow`/`vi` (ref `dantiicu/vulkan-triangle-test-switch`) → a triangle on the TV from a
  standalone `.nro`. The hard winsys foundation is proven; M3 is the rendering + display layer.

See `docs/knowledge/` for the full debugging-pattern and research trail, and `RESUME_NVK.md` for the
moment-to-moment state.

## Rebuilding after a WSI change (READ THIS FIRST)

`package-nvk.sh` does NOT build mesa. It compiles the three winsys shims and
MRI-merges *prebuilt* archives out of `mb/`. `wsi_common_switch.c` is copied into
the mesa tree by `apply-wsi-switch.sh` and compiled as part of mesa, and it lands
in `src/nouveau/vulkan/libnvk.a` (not in the un-merged `libvulkan_wsi.a`).

So editing the WSI and running `package-nvk.sh` ships the OLD object, and looks
exactly like a fix that did not work.

Worse: **ninja does not reliably notice the change**. The repo is a Windows bind
mount, and the container clock is offset from the host, so mtime comparisons say
"up to date" when they are not. Delete the object to force it:

```sh
bash winsys/wsi/apply-wsi-switch.sh /work/mesa-25
rm -f mb/src/vulkan/wsi/libvulkan_wsi.a.p/wsi_common_switch.c.o \
      mb/src/vulkan/wsi/libvulkan_wsi.a \
      mb/src/nouveau/vulkan/libnvk.a
ninja -C mb src/vulkan/wsi/libvulkan_wsi.a src/nouveau/vulkan/libnvk.a \
            src/vulkan/util/libvulkan_util.a
bash package-nvk.sh
```

(A bare `ninja -C mb` fails on an unrelated test executable that links crt0 as a
shared object; name the archive targets.)

**Always verify the change actually landed** before testing on hardware — three
separate "fixes" were tested on device without ever being compiled in:

```sh
aarch64-none-elf-objdump -d nvk-switch/lib/libvulkan.a | grep -c <a constant you added>
```
