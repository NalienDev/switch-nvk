/*
 * drm_shim.h — Switch (libnx nv) winsys shim for Mesa NVK.
 *
 * NVK (Mesa 25) talks to the kernel through libdrm *core* entry points
 * (drmGetVersion / drmCommandWrite[Read] / drmSyncobj* / drmPrime*), passing
 * nouveau uAPI structs (drm-uapi/nouveau_drm.h + the nvif headers). On a Linux box those
 * land in the nouveau KMD; on the Switch there is no DRM. This shim implements
 * those libdrm symbols on top of libnx `nv` (nvMap / nvGpu / nvAddressSpace /
 * nvFence), reporting a synthetic GM20B (Tegra X1, Maxwell-2).
 *
 * This header is the *non-libdrm* surface: the lifecycle + the open()/mmap()
 * hooks that the libc wrap (M2, switch_libc_shim.c) routes through. The libdrm
 * functions themselves match <xf86drm.h> exactly and are not redeclared here.
 *
 * Scope (M1): resolve every drm* symbol, and make device probe / device-info /
 * class query / syncobj real so vkEnumeratePhysicalDevices can succeed.
 * VM_BIND + EXEC (real GPU VA + submit) are stubbed and belong to M2.
 */
#ifndef SWITCH_NVK_DRM_SHIM_H
#define SWITCH_NVK_DRM_SHIM_H

#ifdef __SWITCH__

#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The single synthetic DRM render node the shim advertises. The libc open()
 * wrap (M2) compares the requested path against this and, on a match, returns a
 * shim fd via drm_shim_open() instead of hitting newlib. */
#define DRM_SHIM_RENDER_NODE "/dev/dri/renderD128"

/* True for the path(s) the shim owns. Lets the open() wrap decide quickly. */
bool drm_shim_is_render_node(const char *path);

/* Open/close the shim device. Returns a synthetic fd (always >= 0x6E760000 so
 * it can never collide with a real newlib descriptor) or -1 + errno on failure.
 * First open lazily brings up libnx nv (nvInitialize/Fence/Map/Gpu + an address
 * space), mirroring libdrm_nouveau's nouveau_device_new. */
int  drm_shim_open(const char *path, int flags);
int  drm_shim_close(int fd);

/* True if `fd` is one the shim handed out. The libc ioctl()/mmap()/close()
 * wraps use this to know whether to dispatch here or fall through to __real_*. */
bool drm_shim_owns_fd(int fd);

/* Base/size of the reserved small-page GPU VA arena. NVK's VA allocator must be
 * pointed here (nvkmd_nouveau_create_dev patch) so its fixed BO binds map inside
 * a real nvgpu reservation. Returns 0 if the arena couldn't be reserved. */
uint64_t drm_shim_va_base(uint64_t *size_out);

/* BO CPU mapping. nouveau_ws_bo_map() does mmap(NULL, size, prot, MAP_SHARED,
 * fd, bo->map_handle); the mmap() wrap forwards (fd, map_handle) here and gets
 * back the BO's CPU pointer (the memalign'd backing store). Returns NULL on a
 * bad cookie. drm_shim_munmap is a no-op (the page lives until GEM close). */
void *drm_shim_mmap(int fd, off_t map_handle, size_t length);
int   drm_shim_munmap(void *addr, size_t length);

/* True if `addr` is the CPU backing of a live shim BO (so a host-provided
 * munmap knows not to free memory the shim still owns). */
bool drm_shim_owns_ptr(const void *addr);

/* One-line trace into the shim's debug log (no-op unless DRM_SHIM_DEBUG). */
void drm_shim_dbg(const char *msg);

#ifdef __cplusplus
}
#endif

#endif /* __SWITCH__ */
#endif /* SWITCH_NVK_DRM_SHIM_H */
