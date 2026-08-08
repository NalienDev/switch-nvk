/*
 * drm_shim.c — Switch (libnx nv) winsys shim for Mesa NVK.  See drm_shim.h.
 *
 * NVK's winsys (src/nouveau/winsys + nvkmd/nouveau) drives the GPU entirely
 * through libdrm core calls carrying nouveau uAPI structs:
 *
 *     drmGetVersion(fd)                       -> "nouveau" >= 1.0.3 probe
 *     drmCommandWrite[Read](fd, DRM_NOUVEAU_*, &struct, size)
 *         GETPARAM / NVIF (device new+info, sclass) / VM_INIT / VM_BIND /
 *         EXEC / GEM_NEW / GEM_INFO / GEM_CPU_PREP / CHANNEL_ALLOC|FREE
 *     drmSyncobj{Create,Destroy,Wait,...}     -> CPU/GPU sync
 *     drmPrime{HandleToFD,FDToHandle}         -> dma-buf (unused on Switch)
 *     drmGetDevices2 / drmGetDeviceFromDevId  -> enumeration
 *
 * On Linux those land in the nouveau KMD. Here we satisfy them with libnx `nv`
 * (nvMap / nvGpu / nvAddressSpace / nvFence) — the same primitives devkitPro's
 * libdrm_nouveau uses — and report a synthetic GM20B (Tegra X1, Maxwell-2).
 *
 * M1 (this file): every drm* symbol resolves; device probe / device-info /
 * class query / GEM alloc / syncobj are real, so vkEnumeratePhysicalDevices and
 * device creation can run. VM_BIND + EXEC (real GPU VA + GPFIFO submit) are
 * deliberately no-op-success stubs — that is M2.
 */
#ifdef __SWITCH__

#include "drm_shim.h"

#include <errno.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <switch.h>
#include <xf86drm.h>

#include "drm-uapi/nouveau_drm.h"
#include "nvif/cl0080.h"
#include "nvif/class.h"
#include "nvif/ioctl.h"

/* ------------------------------------------------------------------------- */
/* Logging — off by default; svcOutputDebugString is always available.        */
/* Build with -DDRM_SHIM_DEBUG to trace dispatch.                             */
/* ------------------------------------------------------------------------- */
/* Optional external log sink (set by the app, e.g. the smoke test) so shim
 * traces land in the app's own log file. Always defined so the app can extern
 * it regardless of DRM_SHIM_DEBUG. */
void (*g_drm_shim_log_sink)(const char *) = NULL;

#ifdef DRM_SHIM_DEBUG
#include <stdio.h>
static FILE *g_shimlog;
static void shim_log(const char *fmt, ...)
{
   char buf[256];
   va_list ap;
   va_start(ap, fmt);
   int n = vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   if (n > 0)
      svcOutputDebugString(buf, (n < (int)sizeof(buf)) ? (u64)n : sizeof(buf) - 1);
   if (g_drm_shim_log_sink) { g_drm_shim_log_sink(buf); return; }
   if (!g_shimlog)
      g_shimlog = fopen("sdmc:/nvk_drmshim.log", "w");
   if (g_shimlog) { fputs(buf, g_shimlog); fflush(g_shimlog); }
}
#define SHIM_LOG(...) shim_log("drm_shim: " __VA_ARGS__)
#else
#define SHIM_LOG(...) ((void)0)
#endif

/* Error reporting that survives a RELEASE build.
 *
 * SHIM_LOG is compiled out without DRM_SHIM_DEBUG, so a release shim detects GPU
 * faults and then says nothing at all. Those are exactly the events a consumer
 * has to know about, so route them through a sink hook that is always present.
 * Kept tiny and allocation-free: it runs on the submit path. */
void (*g_drm_shim_err_sink)(const char *) = NULL;

/* Block on the GPU after every submit. Bring-up crutch, OFF by default: leaving
 * it on serialises the CPU against the GPU completely (see nouveau_exec). Flip
 * to 1 to make a GPU execution fault surface at the submit that caused it. */
/* 0 = off, 1 = drain after every kickoff, N = drain after every Nth.
 *
 * Now a PERIOD rather than a flag, because of what the sub-native rendering
 * runs showed. 0xd5c fires when frames are cheap and does not when they are
 * expensive, at an IDENTICAL submit rate -- so the variable is not how often we
 * submit, it is whether the CPU ever BLOCKS. An expensive frame blocks 15-200 ms
 * per frame in nvFenceWait; a cheap one blocks 0.4 ms. If nvgpu retires finished
 * jobs on the back of blocking syncpoint waits, then a GPU that keeps up means
 * nothing ever reaps, and the per-submit allocation eventually fails -- which is
 * exactly the failure, and exactly why the 40 s retry budget never helped
 * (kickoff_retry waits on chan->fence, already reached, so it returns at once
 * and never really waits).
 *
 * A period lets that be traded off instead of being all-or-nothing: draining
 * every submit serialises CPU and GPU completely, but draining every 4th or 8th
 * reintroduces blocking waits at a fraction of the cost. */
int g_drm_shim_sync_submit = 0;

/* Submit-path counters, read by the consumer's per-guest-frame profiler.
 *
 * The guest renders at ~8 fps while the host presents ~3x that, so present-side
 * timing says nothing about where a guest frame goes. These answer the question
 * that matters: how many GPU submits does one guest frame cost, and how much of
 * that frame is spent blocked in the in-flight throttle waiting for the GPU. */
uint64_t g_drm_shim_exec_count = 0;    /* EXEC ioctls accepted                  */
uint64_t g_drm_shim_exec_wait_ns = 0;  /* time blocked in the in-flight throttle */
uint64_t g_drm_shim_kickoff_count = 0; /* actual nvGpuChannelKickoff calls       */

/* How many EXECs may be coalesced into ONE kickoff. See channel_flush_locked.
 *
 * 0xd5c tracks how OFTEN we submit, not how much work or memory is involved --
 * measured two ways on hardware. A probe allocates and maps a fresh 4 MiB nvmap
 * object at the exact instant kickoff returns 0xd5c and both calls succeed, so
 * there is no byte ceiling; and making frames CHEAPER (shadows/haze off) made
 * 0xd5c MORE frequent, not less, while the BO bytes held at the failure fell.
 * That is nvgpu's per-submit job pool being reaped lazily while we submit faster
 * than it frees, and the only thing that helps is submitting less often.
 *
 * Set to 1 to get exactly the old one-kickoff-per-EXEC behaviour back -- the A/B
 * control for this change, since every previous submit-path theory here was
 * settled by a hardware A/B rather than by reasoning. */
/* DEFAULT 1 = OFF. Batching is implemented and correct, but on hardware it
 * measurably destabilises this port: with it live the run takes 8 kickoff
 * failures and never reaches the main menu, while the identical build with
 * this at 1 takes zero and plays. Raise it only alongside a hardware A/B. */
int g_drm_shim_batch_max = 1;
static void shim_err(const char *fmt, ...)
{
   char buf[192];
   va_list ap;
   va_start(ap, fmt);
   int n = vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   if (n <= 0)
      return;
   if (g_drm_shim_err_sink) { g_drm_shim_err_sink(buf); return; }
   if (g_drm_shim_log_sink) { g_drm_shim_log_sink(buf); return; }
}

/* Report at most `n` times per call site.
 *
 * The consumer's sink does SD-card I/O (a few ms per line), and this port has
 * already been bitten once by log volume that was enough to move a timing-
 * sensitive failure earlier in the run. A condition that repeats every frame —
 * an exhausted BO table, a channel that fails every kickoff — must not turn into
 * a write every frame. The counter is deliberately unsynchronised: a race can
 * only cost a couple of extra lines. */
#define SHIM_ERR_FIRST(n, ...)                              \
   do {                                                     \
      static unsigned _shim_err_seen = 0;                   \
      if (_shim_err_seen < (unsigned)(n)) {                 \
         _shim_err_seen++;                                  \
         shim_err(__VA_ARGS__);                             \
      }                                                     \
   } while (0)

/* Exported one-line trace so other TUs (switch_libc_shim.c) can log into the
 * same sink. No-op unless built with DRM_SHIM_DEBUG. */
void drm_shim_dbg(const char *msg)
{
#ifdef DRM_SHIM_DEBUG
   shim_log("%s", msg);
#else
   (void)msg;
#endif
}

/* ------------------------------------------------------------------------- */
/* Synthetic device model. The Switch has exactly one GPU, so one global       */
/* device guarded by a mutex; the shim fd is a sentinel that can never alias   */
/* a real newlib descriptor.                                                   */
/* ------------------------------------------------------------------------- */
#define DRM_SHIM_FD       0x6E760000  /* 'nv' << 16; see drm_shim_owns_fd()    */
#define SHIM_MAX_BOS      4096
#define SHIM_MAX_SYNCOBJS 4096
#define SHIM_MAX_CHANNELS 64
/* Submits allowed in flight per channel. Restored to the exact shape of the last
 * build measured STABLE on hardware (3.7 fps, no 0xd5c). Waiting on raw
 * chan->fence instead -- which looked equivalent -- crashed before anything
 * rendered, so the difference is real even if the mechanism is not understood:
 * this waits on a fence captured AFTER a successful kickoff, and only when the
 * slot actually holds one. */
#define SHIM_MAX_INFLIGHT 1

/* Ceiling on EXECs coalesced into one kickoff (g_drm_shim_batch_max is clamped
 * to this), and on the syncobjs one batch may promise to signal. */
#define SHIM_MAX_BATCH_EXECS 32
#define SHIM_MAX_BATCH_SIGS  128

/* Stop appending and kick off once the GPFIFO ring reaches this many entries.
 * NvGpuChannel::entries is a FIXED GPFIFO_QUEUE_SIZE (2048) array and
 * nvGpuChannelAppendEntry FAILS when it is full -- it does not kick off for you.
 * Batching is the first thing here that ever lets entries accumulate across
 * EXECs, so it is also the first thing that could hit that wall. Half the ring
 * is a wide margin: a failing submit has read entries=2/2048 every single time,
 * i.e. ~2 entries per EXEC. */
#define SHIM_BATCH_ENTRY_LIMIT (GPFIFO_QUEUE_SIZE / 2)

/* GM20B (Tegra X1) — Maxwell 2nd gen. chipset 0x12b => NVK sm_for_chipset()=53.
 * The 3D engine class is what NVK actually gates on (>= KEPLER_A); we report
 * the real Maxwell-B class set from the synthetic NVIF SCLASS query below. */
#define SHIM_CHIPSET      0x12b
/* GRAPH_UNITS: NVK reads gpc=value&0xff, tpc=(value>>8)&0xffff. X1 = 1 GPC,
 * 2 TPC (2 SMM, 256 CUDA cores). */
#define SHIM_GRAPH_UNITS  ((2u << 8) | 1u)
/* Unified memory; report a conservative slice as "VRAM" for heap sizing. */
#define SHIM_RAM_USER     (2ull << 30)

struct shim_bo {
   bool     used;
   void    *cpu;          /* memalign'd backing store (== mmap target)         */
   uint64_t size;
   NvMap    map;          /* libnx nvmap object                                */
   bool     gpu_mapped;   /* internal BOs (cmdbuf/zcull) get an auto GPU VA    */
   uint64_t gpu_va;       /* GPU VA: auto (internal) or VM_BIND addr (NVK BOs) */
   uint64_t gpu_bo_offset;/* VM_BIND bo_offset, for VA->CPU back-translation   */
   NvKind   kind;         /* the BO's PTE kind (from GEM_NEW tile_flags)       */
};

struct shim_syncobj {
   bool     used;
   bool     signaled;
   uint64_t value;        /* timeline point (binary syncobjs ignore)           */
   bool     has_fence;    /* M2: a real GPU fence was attached by EXEC         */
   NvFence  fence;        /* libnx syncpoint fence to wait on                  */
   /* Signalled by an EXEC whose kickoff is still DEFERRED in a batch. Neither
    * signalled nor holding a fence: the work is not on the GPU yet, so there is
    * nothing to wait on and reporting it complete would hand NVK a buffer the
    * GPU has not been given. Every path that observes completion flushes first,
    * which turns this into a real fence. */
   bool     pending;
   uint32_t pending_chan;  /* channel id owing the fence while pending        */
};

/* A real libnx GPFIFO channel + the builtin cmdbuf carrying the Maxwell
 * fence-increment and cache-flush command lists (mirrors libdrm_nouveau). */
struct shim_channel {
   bool         used;
   NvGpuChannel chan;
   int          cmdbuf_bo;       /* bos[] slot for the fence+flush cmdlists    */
   uint64_t     cmdbuf_va;       /* GPU VA of cmdbuf_bo                         */
   uint32_t     fence_num_cmds;  /* dwords of the fence-incr cmdlist           */
   uint32_t     flush_num_cmds;  /* dwords of the cache-flush cmdlist          */
   uint32_t     setobj_num_cmds; /* dwords of the per-subchannel SetObject bind */
   bool         engines_bound;   /* SetObject bind emitted before 1st EXEC      */
   int          zcull_bo;        /* bos[] slot for the Zcull context           */

   /* In-flight submit throttle; see SHIM_MAX_INFLIGHT. */
   NvFence      inflight[SHIM_MAX_INFLIGHT];
   bool         inflight_valid[SHIM_MAX_INFLIGHT];
   uint32_t     inflight_head;

   /* Deferred-kickoff batch: EXECs appended to the ring but not yet submitted.
    * batch_sig holds every syncobj those EXECs promised to signal, because the
    * completion fence for all of them only exists once the batch is kicked off
    * -- and if the kickoff FAILS they all have to be released together, since
    * the reset discards the whole ring, not just the last EXEC's entries. */
   uint32_t     batch_execs;
   uint32_t     batch_sig[SHIM_MAX_BATCH_SIGS];
   uint32_t     batch_sig_count;
};

static struct shim_device {
   Mutex          lock;
   int            refcnt;       /* open() count                                */
   bool           nv_up;        /* libnx nv brought up                         */
   NvAddressSpace addr_space;
   uint32_t       big_page_size;
   uint64_t       va_base;      /* base of the reserved small-page VA arena     */
   uint64_t       va_size;      /* size of that arena (NVK's heap lives here)    */
   uint32_t       channel_ctr;  /* hands out channel ids                       */
   struct shim_bo       bos[SHIM_MAX_BOS];           /* handle = index + 1     */
   struct shim_syncobj  syncobjs[SHIM_MAX_SYNCOBJS]; /* handle = index + 1     */
   struct shim_channel  channels[SHIM_MAX_CHANNELS]; /* id     = index + 1     */
} g_dev;

static inline uint64_t align_up(uint64_t v, uint64_t a)
{
   return (v + a - 1) & ~(a - 1);
}

#define ARRAY_SIZE_LOCAL(a) (sizeof(a) / sizeof((a)[0]))

/* Defined in the syncobj section; used by EXEC/VM_BIND to attach GPU fences. */
struct shim_syncobj;
static struct shim_syncobj *syncobj_lookup(uint32_t handle);
/* Defined in the channel section; used by NVIF NEW to alloc engine obj-ctx. */
struct shim_channel;
static struct shim_channel *channel_lookup(uint32_t id);
/* Defined with EXEC; channel teardown must submit a deferred batch first. */
static int channel_flush_locked(struct shim_channel *ch);

/* ------------------------------------------------------------------------- */
/* libnx nv bring-up — mirrors libdrm_nouveau's nouveau_device_new ordering.   */
/* ------------------------------------------------------------------------- */
static int shim_nv_up(void)
{
   Result rc = nvInitialize();
   if (R_FAILED(rc)) goto fail_init;
   rc = nvFenceInit();
   if (R_FAILED(rc)) goto fail_fence;
   rc = nvMapInit();
   if (R_FAILED(rc)) goto fail_map;
   rc = nvGpuInit();
   if (R_FAILED(rc)) goto fail_gpu;

   const nvioctl_gpu_characteristics *chars = nvGpuGetCharacteristics();
   g_dev.big_page_size = chars ? chars->big_page_size : 0x10000;

   rc = nvAddressSpaceCreate(&g_dev.addr_space, g_dev.big_page_size);
   if (R_FAILED(rc)) goto fail_as;

   /* Reserve one big SMALL-PAGE (4 KB) VA arena, NON-fixed (kernel picks the
    * base in the lower/small-page half). NVK's VA heap is pointed at this arena
    * (see nvkmd_nouveau_create_dev patch), so its BO binds become FIXED maps
    * *inside* a real reservation — which is the only fixed mapping Tegra's nvgpu
    * accepts (bare fixed-VA reservation returns -EINVAL). */
   g_dev.va_size = 0x200000000ull;  /* 8 GiB, within the ~16 GiB small-page half */
   uint64_t abase = 0;
   Result arc = nvioctlNvhostAsGpu_AllocSpace(g_dev.addr_space.fd,
                   (u32)(g_dev.va_size / 0x1000), 0x1000, 0 /*non-fixed*/,
                   0x10000 /*base align*/, &abase);
   if (R_FAILED(arc)) {
      SHIM_LOG("VA arena reserve FAILED rc=0x%x\n", arc);
      g_dev.va_base = 0;
      g_dev.va_size = 0;
   } else {
      g_dev.va_base = abase;
      SHIM_LOG("VA arena base=0x%llx size=0x%llx\n",
               (unsigned long long)abase, (unsigned long long)g_dev.va_size);
   }

   g_dev.channel_ctr = 0;
   g_dev.nv_up = true;
   SHIM_LOG("nv up (arch=0x%x big_page=0x%x)\n",
            chars ? chars->arch : 0, g_dev.big_page_size);
   return 0;

fail_as:    nvGpuExit();
fail_gpu:   nvMapExit();
fail_map:   nvFenceExit();
fail_fence: nvExit();
fail_init:
   SHIM_LOG("nv bring-up failed rc=0x%x\n", rc);
   errno = ENODEV;
   return -1;
}

static void shim_nv_down(void)
{
   /* Close any channels still open (frees their builtin BOs too). */
   for (int i = 0; i < SHIM_MAX_CHANNELS; i++) {
      if (g_dev.channels[i].used) {
         nvGpuChannelClose(&g_dev.channels[i].chan);
         g_dev.channels[i].used = false;
      }
   }
   /* Release any BOs the client (or our internal allocs) leaked. */
   for (int i = 0; i < SHIM_MAX_BOS; i++) {
      if (g_dev.bos[i].used) {
         if (g_dev.bos[i].gpu_mapped)
            nvAddressSpaceUnmap(&g_dev.addr_space, g_dev.bos[i].gpu_va);
         nvMapClose(&g_dev.bos[i].map);
         free(g_dev.bos[i].cpu);
         g_dev.bos[i].used = false;
      }
   }
   if (g_dev.va_base)
      nvAddressSpaceFree(&g_dev.addr_space, g_dev.va_base, g_dev.va_size);
   g_dev.va_base = 0;
   g_dev.va_size = 0;
   nvAddressSpaceClose(&g_dev.addr_space);
   nvGpuExit();
   nvMapExit();
   nvFenceExit();
   nvExit();
   g_dev.nv_up = false;
}

/* ------------------------------------------------------------------------- */
/* Public lifecycle (consumed by the M2 open()/mmap() libc wrap).              */
/* ------------------------------------------------------------------------- */
bool drm_shim_is_render_node(const char *path)
{
   return path && strcmp(path, DRM_SHIM_RENDER_NODE) == 0;
}

bool drm_shim_owns_fd(int fd)
{
   return fd == DRM_SHIM_FD;
}

/* The reserved small-page VA arena that NVK's VA heap must live inside (so its
 * fixed BO binds land in a real nvgpu reservation). Returns 0 if unavailable. */
uint64_t drm_shim_va_base(uint64_t *size_out)
{
   if (size_out)
      *size_out = g_dev.va_size;
   return g_dev.va_base;
}

int drm_shim_open(const char *path, int flags)
{
   (void)flags;
   if (!drm_shim_is_render_node(path)) {
      errno = ENOENT;
      return -1;
   }

   mutexLock(&g_dev.lock);
   if (g_dev.refcnt == 0) {
      if (shim_nv_up() != 0) {
         mutexUnlock(&g_dev.lock);
         return -1;
      }
   }
   g_dev.refcnt++;
   mutexUnlock(&g_dev.lock);
   SHIM_LOG("open(%s) -> fd 0x%x\n", path, DRM_SHIM_FD);
   return DRM_SHIM_FD;
}

int drm_shim_close(int fd)
{
   if (!drm_shim_owns_fd(fd)) {
      errno = EBADF;
      return -1;
   }
   mutexLock(&g_dev.lock);
   if (g_dev.refcnt > 0 && --g_dev.refcnt == 0)
      shim_nv_down();
   mutexUnlock(&g_dev.lock);
   return 0;
}

/* ------------------------------------------------------------------------- */
/* BO table helpers.                                                           */
/* ------------------------------------------------------------------------- */
static struct shim_bo *bo_lookup(uint32_t handle)
{
   if (handle == 0 || handle > SHIM_MAX_BOS)
      return NULL;
   struct shim_bo *bo = &g_dev.bos[handle - 1];
   return bo->used ? bo : NULL;
}

void *drm_shim_mmap(int fd, off_t map_handle, size_t length)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return NULL; }
   mutexLock(&g_dev.lock);
   struct shim_bo *bo = bo_lookup((uint32_t)map_handle);
   unsigned long long bosz = bo ? (unsigned long long)bo->size : 0ull;
   void *cpu = (bo && length <= bo->size) ? bo->cpu : NULL;
   mutexUnlock(&g_dev.lock);
   SHIM_LOG("mmap handle=%llu length=0x%llx bo=%s bo_size=0x%llx -> cpu=%p\n",
            (unsigned long long)map_handle, (unsigned long long)length,
            bo ? "found" : "NULL", bosz, cpu);
   if (!cpu) errno = EINVAL;
   return cpu;
}

/* True if `addr` is the CPU backing of a live shim BO. A host process that
 * provides its own mmap/munmap (e.g. skate3's rexglue mman shim) uses this to
 * avoid free()ing BO memory the shim still owns (it is released on GEM close). */
bool drm_shim_owns_ptr(const void *addr)
{
   if (!addr) return false;
   bool owned = false;
   mutexLock(&g_dev.lock);
   for (int i = 0; i < SHIM_MAX_BOS; i++) {
      if (g_dev.bos[i].used && g_dev.bos[i].cpu == addr) { owned = true; break; }
   }
   mutexUnlock(&g_dev.lock);
   return owned;
}

int drm_shim_munmap(void *addr, size_t length)
{
   /* BO backing lives until drmCloseBufferHandle; nothing to unmap. */
   (void)addr; (void)length;
   return 0;
}

/* ------------------------------------------------------------------------- */
/* GETPARAM.                                                                   */
/* ------------------------------------------------------------------------- */
static int nouveau_getparam(struct drm_nouveau_getparam *gp)
{
   switch (gp->param) {
   case NOUVEAU_GETPARAM_PCI_VENDOR:   gp->value = 0x10de;            return 0;
   case NOUVEAU_GETPARAM_PCI_DEVICE:   gp->value = 0;                 return 0; /* SOC */
   case NOUVEAU_GETPARAM_GRAPH_UNITS:  gp->value = SHIM_GRAPH_UNITS;  return 0;
   case NOUVEAU_GETPARAM_PTIMER_TIME:
      /* GPU timestamp; armGetSystemTick is the closest free-running counter. */
      gp->value = armTicksToNs(armGetSystemTick());
      return 0;
   case NOUVEAU_GETPARAM_EXEC_PUSH_MAX: gp->value = NOUVEAU_GEM_MAX_PUSH; return 0;
   case NOUVEAU_GETPARAM_VRAM_USED:     gp->value = 0;  return 0; /* => has_get_vram_used=false */
   case NOUVEAU_GETPARAM_HAS_VMA_TILEMODE: gp->value = 0; return 0; /* tiled-BO off for M1 */
   default:
      SHIM_LOG("GETPARAM unknown %llu\n", (unsigned long long)gp->param);
      return -EINVAL;
   }
}

/* ------------------------------------------------------------------------- */
/* NVIF — device object new+info, channel sclass query, object del.            */
/* ------------------------------------------------------------------------- */

/* GM20B engine classes, returned from the synthetic SCLASS query. NVK matches
 * by low byte: copy 0xb5, 2d 0x2d, 3d 0x97, compute 0xc0, m2mf 0x40. */
static const int32_t k_gm20b_classes[] = {
   MAXWELL_DMA_COPY_A,          /* 0xb0b5 */
   FERMI_TWOD_A,                /* 0x902d */
   MAXWELL_B,                   /* 0xb197 */
   MAXWELL_COMPUTE_B,           /* 0xb1c0 */
   KEPLER_INLINE_TO_MEMORY_B,   /* 0xa140 */
};

static int nouveau_nvif(void *data, unsigned long size)
{
   if (size < sizeof(struct nvif_ioctl_v0))
      return -EINVAL;
   struct nvif_ioctl_v0 *io = data;

   switch (io->type) {
   case NVIF_IOCTL_V0_NEW: {
      struct nvif_ioctl_new_v0 *nw = (void *)io->data;
      /* libnx's nvGpuChannelCreate auto-allocates the single GR(3D) obj-ctx the
       * channel is allowed; compute/copy/2d/m2mf are bound in-stream via the
       * Maxwell SetObject method (NVK emits these). A separate AllocObjCtx for
       * them is rejected by nvgpu (-EINVAL, one GR ctx per channel), so there's
       * nothing to do kernel-side here. */
      SHIM_LOG("NVIF NEW oclass=0x%x\n", (unsigned)nw->oclass);
      return 0;
   }

   case NVIF_IOCTL_V0_MTHD: {
      struct nvif_ioctl_mthd_v0 *m = (void *)io->data;
      if (m->method == NV_DEVICE_V0_INFO) {
         struct nv_device_info_v0 *inf = (void *)m->data;
         memset(inf, 0, sizeof(*inf));
         inf->version  = 0;
         inf->platform = NV_DEVICE_INFO_V0_SOC;   /* Tegra => NV_DEVICE_TYPE_SOC */
         inf->chipset  = SHIM_CHIPSET;
         inf->revision = 0xa1;                    /* GM20B rev A1                */
         inf->family   = NV_DEVICE_INFO_V0_MAXWELL;
         inf->ram_size = SHIM_RAM_USER;
         inf->ram_user = SHIM_RAM_USER;
         strncpy(inf->chip, "GM20B", sizeof(inf->chip) - 1);
         strncpy(inf->name, "NVIDIA Tegra X1", sizeof(inf->name) - 1);
         SHIM_LOG("NVIF device INFO -> GM20B\n");
         return 0;
      }
      SHIM_LOG("NVIF MTHD method=0x%x (ignored)\n", m->method);
      return 0;
   }

   case NVIF_IOCTL_V0_SCLASS: {
      struct nvif_ioctl_sclass_v0 *sc = (void *)io->data;
      unsigned cap = sc->count;   /* caller-provided capacity                  */
      unsigned n   = 0;
      for (unsigned i = 0; i < cap; i++) {
         if (i < ARRAY_SIZE_LOCAL(k_gm20b_classes)) {
            sc->oclass[i].oclass = k_gm20b_classes[i];
            sc->oclass[i].minver = 0;
            sc->oclass[i].maxver = 0;
            n++;
         } else {
            /* Zero the tail: NVK reads all `cap` entries unconditionally. */
            sc->oclass[i].oclass = 0;
            sc->oclass[i].minver = 0;
            sc->oclass[i].maxver = 0;
         }
      }
      sc->count = (uint8_t)n;
      SHIM_LOG("NVIF SCLASS -> %u classes\n", n);
      return 0;
   }

   case NVIF_IOCTL_V0_DEL:
      return 0;

   default:
      SHIM_LOG("NVIF type=0x%x (ignored)\n", io->type);
      return 0;
   }
}

/* ------------------------------------------------------------------------- */
/* GEM — BO alloc / info / cpu-prep / close.  Backed by libnx nvmap, mirroring */
/* libdrm_nouveau's nouveau_bo_new.  GPU VA mapping happens later in VM_BIND   */
/* (NVK BOs) or immediately for internal BOs.                                  */
/* ------------------------------------------------------------------------- */

/* Allocate a BO slot + memalign backing + nvmap. Returns slot index or -1.
 * Caller holds g_dev.lock. */
static int shim_bo_alloc_locked(uint64_t size, uint32_t align,
                                NvKind kind, bool cacheable)
{
   if (align < 0x1000) align = 0x1000;
   size = align_up(size, align);

   int slot = -1;
   for (int i = 0; i < SHIM_MAX_BOS; i++) {
      if (!g_dev.bos[i].used) { slot = i; break; }
   }
   /* These three failures used to be indistinguishable -1s visible only under
    * DRM_SHIM_DEBUG, so in a release build BO exhaustion looked exactly like a
    * mysterious VK_ERROR_OUT_OF_DEVICE_MEMORY with no explanation. A scene
    * transition is precisely when the table or the heap would run out. */
   if (slot < 0) {
      SHIM_ERR_FIRST(8, "drm_shim: BO TABLE FULL (%d BOs) size=0x%llx\n",
               SHIM_MAX_BOS, (unsigned long long)size);
      return -1;
   }

   void *mem = memalign(0x1000, size);
   if (!mem) {
      SHIM_ERR_FIRST(8, "drm_shim: BO memalign FAILED size=0x%llx align=0x%x\n",
               (unsigned long long)size, align);
      return -1;
   }
   memset(mem, 0, size);

   Result rc = nvMapCreate(&g_dev.bos[slot].map, mem, size, align, kind, cacheable);
   if (R_FAILED(rc)) {
      free(mem);
      SHIM_ERR_FIRST(8, "drm_shim: nvMapCreate FAILED rc=0x%x size=0x%llx kind=0x%x\n",
               rc, (unsigned long long)size, (unsigned)kind);
      return -1;
   }

   g_dev.bos[slot].used       = true;
   g_dev.bos[slot].cpu        = mem;
   g_dev.bos[slot].size       = size;
   g_dev.bos[slot].gpu_mapped = false;
   return slot;
}

/* Allocate an internal BO and give it an auto-assigned GPU VA (for the channel
 * cmdbuf / zcull ctx, which we own — NVK BOs get their VA from VM_BIND). */
static int shim_internal_bo_locked(uint64_t size, uint32_t align,
                                   uint64_t *gpu_va_out)
{
   int slot = shim_bo_alloc_locked(size, align, NvKind_Pitch, false);
   if (slot < 0)
      return -1;

   uint64_t va = 0;
   Result rc = nvAddressSpaceMap(&g_dev.addr_space,
                                 nvMapGetHandle(&g_dev.bos[slot].map),
                                 true /* gpu cacheable */, NvKind_Pitch, &va);
   if (R_FAILED(rc)) {
      nvMapClose(&g_dev.bos[slot].map);
      free(g_dev.bos[slot].cpu);
      memset(&g_dev.bos[slot], 0, sizeof(g_dev.bos[slot]));
      SHIM_LOG("internal BO GPU map failed rc=0x%x\n", rc);
      return -1;
   }
   g_dev.bos[slot].gpu_mapped = true;
   g_dev.bos[slot].gpu_va     = va;
   if (gpu_va_out)
      *gpu_va_out = va;
   return slot;
}

static int nouveau_gem_new(struct drm_nouveau_gem_new *req)
{
   uint32_t align = req->align ? req->align : 0x1000;
   NvKind kind = (NvKind)(req->info.tile_flags >> 8);  /* 0 => NvKind_Pitch    */

   int slot = shim_bo_alloc_locked(req->info.size, align, kind, false);
   if (slot < 0)
      return -ENOMEM;

   g_dev.bos[slot].kind = kind;     /* remember the BO's real PTE kind for VM_BIND */

   uint32_t handle = (uint32_t)slot + 1;
   req->info.handle     = handle;
   req->info.size       = g_dev.bos[slot].size;
   req->info.map_handle = handle;   /* mmap() cookie -> drm_shim_mmap()        */
   /* domain/offset/tile fields left as the caller set them. */
   SHIM_LOG("GEM_NEW handle=%u size=%llu tile_flags=0x%x kind=0x%x\n",
            handle, (unsigned long long)g_dev.bos[slot].size,
            req->info.tile_flags, (unsigned)kind);
   return 0;
}

static int nouveau_gem_info(struct drm_nouveau_gem_info *info)
{
   struct shim_bo *bo = bo_lookup(info->handle);
   if (!bo)
      return -ENOENT;
   info->size       = bo->size;
   info->map_handle = info->handle;
   return 0;
}

/* ---- switch-nvk WSI zero-copy (A1) -----------------------------------------
 * Given a GPU VA that falls inside a bound BO, expose that BO's libnx nvmap
 * *id* (the cross-process handle the VI compositor needs), its size and PTE
 * kind. The WSI backend (wsi_common_switch.c) uses this to wrap a swapchain
 * image's device memory in an NvGraphicBuffer (nwindowConfigureBuffer) so the
 * VI compositor scans out the rendered image directly — no per-frame CPU copy
 * (kills the ~9.4ms memcpy the present profiling found). Returns false if no
 * bound BO covers the VA. Thread-safe (takes the device lock).               */
bool drm_shim_bo_nvmap_by_va(uint64_t gpu_va, uint32_t *out_nvmap_id,
                             uint64_t *out_size, uint32_t *out_kind)
{
   bool found = false;
   mutexLock(&g_dev.lock);
   for (int b = 0; b < SHIM_MAX_BOS; b++) {
      struct shim_bo *bo = &g_dev.bos[b];
      if (bo->used && bo->gpu_va && gpu_va >= bo->gpu_va &&
          gpu_va < bo->gpu_va + bo->size) {
         if (out_nvmap_id) *out_nvmap_id = nvMapGetId(&bo->map);
         if (out_size)     *out_size     = bo->size;
         if (out_kind)     *out_kind     = (uint32_t)bo->kind;
         found = true;
         break;
      }
   }
   mutexUnlock(&g_dev.lock);
   return found;
}

static int nouveau_gem_cpu_prep(struct drm_nouveau_gem_cpu_prep *req)
{
   /* CPU-side wait for BO idle. We don't track per-BO fences yet; submissions
    * are serialized through the channel syncpoint, so this is a no-op for now.
    * (A per-BO fence map would make this precise — M2 follow-up.) */
   return bo_lookup(req->handle) ? 0 : -ENOENT;
}

int drmCloseBufferHandle(int fd, uint32_t handle)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   mutexLock(&g_dev.lock);
   struct shim_bo *bo = bo_lookup(handle);
   if (bo) {
      /* Release the GPU-VA mapping before closing. NVK closes + reuses GEM
       * handles many times during device bring-up; without this UnmapBuffer the
       * arena VA mapping leaks in the nv kernel each time, accumulating until
       * its per-submit kzalloc fails (NvError_InsufficientMemory 0xd5c) on NVK's
       * first large EXEC. (Internal BOs use nvAddressSpaceMap; skip those.) */
      if (bo->gpu_va && !bo->gpu_mapped)
         nvioctlNvhostAsGpu_UnmapBuffer(g_dev.addr_space.fd, bo->gpu_va);
      nvMapClose(&bo->map);
      free(bo->cpu);
      memset(bo, 0, sizeof(*bo));
   }
   mutexUnlock(&g_dev.lock);
   return 0;
}

/* ------------------------------------------------------------------------- *
 * Channel + VM_BIND + EXEC (M2).
 *
 * NVK's new-uAPI model: it owns the GPU VA space (its own allocator), VM_BIND
 * maps a BO sub-range at an NVK-chosen VA, and EXEC pushes pre-built IBs (by VA)
 * to a channel + signals syncobjs. We bridge that onto libnx nv:
 *   VM_BIND MAP  -> nvioctlNvhostAsGpu_{AllocSpace,MapBufferEx} at the fixed VA
 *                   (MapBufferEx carries buffer_offset/mapping_size, so partial
 *                    binds map exactly BO[bo_offset:+range] — what libnx's
 *                    nvAddressSpaceMapFixed wrapper can't do).
 *   EXEC         -> nvGpuChannelAppendEntry per push + a syncpoint-increment
 *                   cmdlist + nvGpuChannelKickoff, then stash the channel fence
 *                   into the signalled syncobjs (mirrors libdrm_nouveau pushbuf).
 * ------------------------------------------------------------------------- */

static struct shim_channel *channel_lookup(uint32_t id)
{
   if (id == 0 || id > SHIM_MAX_CHANNELS)
      return NULL;
   struct shim_channel *c = &g_dev.channels[id - 1];
   return c->used ? c : NULL;
}

/* Maxwell command lists, verbatim from devkitPro's libdrm_nouveau pushbuf.c:
 * one increments the channel syncpoint (so the fence fires on completion), the
 * other flushes GPU caches between submits. */
static uint32_t gen_fence_cmdlist(uint32_t *buf, uint32_t syncpt_id)
{
   uint32_t *cmd = buf;
   *cmd++ = 0x451 | (0 << 13) | (0 << 16) | (4 << 29);
   *cmd++ = 0x0B2 | (0 << 13) | (1 << 16) | (1 << 29);
   *cmd++ = syncpt_id | (1 << 20) | (1 << 16); /* syncpt incr + gpu cache flush */
   return cmd - buf;
}

static uint32_t gen_flush_cmdlist(uint32_t *buf)
{
   uint32_t *cmd = buf;
   *cmd++ = 0x00B | (6 << 13) | (1 << 16) | (1 << 29);
   *cmd++ = 0x80000000;
   *cmd++ = 0x00B | (6 << 13) | (1 << 16) | (1 << 29);
   *cmd++ = 0x70000000;
   *cmd++ = 0x4A2 | (0 << 13) | (0 << 16) | (4 << 29);      /* InvalidateTextureDataNoWfi */
   *cmd++ = 0x369 | (0 << 13) | (0x1011 << 16) | (4 << 29); /* unknown flush */
   *cmd++ = 0x50A | (0 << 13) | (0 << 16) | (4 << 29);      /* flush TICs */
   *cmd++ = 0x509 | (0 << 13) | (0 << 16) | (4 << 29);      /* flush TSCs */
   *cmd   = 0;                                              /* trailing NOP */
   return cmd - buf;
}

/* Bind each GPU engine class to the subchannel NVK addresses it by, via SET_OBJECT
 * (method 0x0, count 1, increasing). NVK references engines by subchannel NUMBER
 * and relies on the kernel having bound them via NVIF NEW — which our shim must
 * replicate, or methods hit an UNBOUND subchannel -> GPU MMU fault
 * (NvNotificationType 31) -> channel reset -> every later submit Timeout 0xd5c.
 * Subchannel map (Maxwell, from NVK src/nouveau/winsys nv_push.h):
 *   3D=0 (0xb197), compute=1 (0xb1c0), m2mf/i2m=2 (0xa140), 2D=3 (0x902d), copy=4 (0xb0b5). */
static uint32_t gen_setobj_cmdlist(uint32_t *buf)
{
   static const struct { uint32_t subch, cls; } eng[] = {
      { 0, 0xb197 }, { 1, 0xb1c0 }, { 2, 0xa140 }, { 3, 0x902d }, { 4, 0xb0b5 },
   };
   uint32_t *cmd = buf;
   for (unsigned i = 0; i < sizeof(eng) / sizeof(eng[0]); i++) {
      *cmd++ = 0x20010000u | (eng[i].subch << 13);  /* incr, count 1, method 0 = SET_OBJECT */
      *cmd++ = eng[i].cls;
   }
   return cmd - buf;
}

/* KickoffPb can transiently return NvError_InsufficientMemory (0xd5c) when the
 * channel's GPFIFO ring / submit pool is momentarily full (the GPU hasn't
 * drained prior entries yet) — observed as a FLAKY failure of NVK's first large
 * init submit (same code, succeeds or fails between runs). Retry with a short
 * backoff so submission is deterministic. (TODO: the g_dev.lock is held across
 * the sleep; fine for the single-threaded smoke test, revisit for concurrency.) */
/* Attempts per kickoff. DO NOT RAISE THIS.
 *
 * It was 400, assuming 0xd5c is a transient "ring full" that clears if you wait.
 * Hardware says otherwise: across consecutive failures the GPU syncpoint advanced
 * by EXACTLY the retry count per failed submit --
 *   hw delta +1 -> +401 -> +801 -> +1201   with 400 retries
 *   hw delta +1 -> +3   -> +5   -> +7      with 2
 * So nvGpuChannelKickoff is NOT failing to submit: the GPU executes the queued
 * work and advances the syncpoint, and the ioctl returns 0xd5c anyway. Every
 * retry RE-EXECUTES the same command buffers. Retrying recovers nothing here; it
 * only multiplies duplicated GPU work. */
#define SHIM_KICKOFF_ATTEMPTS 2

static Result kickoff_retry(NvGpuChannel *chan)
{
   Result rc = 0;
   for (int i = 0; i < SHIM_KICKOFF_ATTEMPTS; i++) {
      rc = nvGpuChannelKickoff(chan);
      if (R_SUCCEEDED(rc) || rc != 0xd5c)
         return rc;

      /* Ring / submit pool is full: the GPU has not drained prior entries yet.
       * Wait for work the GPU has ALREADY been given, then retry.
       *
       * Deliberately NOT nvGpuChannelGetFence(): that returns
       * `fence.value + fence_incr`, i.e. the value THIS pending submit will reach
       * once kicked off. The kickoff has just failed, so that increment never
       * happens and the wait can never be satisfied -- it times out on every
       * iteration and the loop is no better than the flat sleep it replaced
       * (measured: the failure was unchanged). `c->fence` alone is the target
       * previously ACCEPTED kickoffs asked the GPU to reach, so it is reachable. */
      NvFence submitted = chan->fence;
      const u64 t_wait0 = armGetSystemTick();
      nvFenceWait(&submitted, 100000);
      g_drm_shim_exec_wait_ns += armTicksToNs(armGetSystemTick() - t_wait0);
      if (i == 0) {
         /* First retry only. The ring is NOT the problem (entries has read 2/2048
          * every time), and retrying cannot help either: this loop already waits
          * up to 400 x 100 ms = 40 s and still fails, so the resource is exhausted
          * PERMANENTLY, not momentarily.
          *
          * 0xd5c is LibnxNvidiaError_InsufficientMemory and the crash lands
          * exactly at the gameplay transition, when the game loads the world. So
          * the suspect is GPU memory: every BO here is a heap allocation plus an
          * nvmap object, and the kernel needs to allocate for the submit too.
          * Count what we are holding -- that is the number that decides whether
          * this is BO/nvmap exhaustion or something else entirely. */
         uint32_t bo_used = 0;
         uint64_t bo_bytes = 0, bo_mapped_bytes = 0;
         for (int b = 0; b < SHIM_MAX_BOS; b++) {
            if (!g_dev.bos[b].used) continue;
            bo_used++;
            bo_bytes += g_dev.bos[b].size;
            if (g_dev.bos[b].gpu_va) bo_mapped_bytes += g_dev.bos[b].size;
         }
         uint32_t chan_used = 0;
         for (int c = 0; c < SHIM_MAX_CHANNELS; c++)
            if (g_dev.channels[c].used) chan_used++;
         /* Real hardware syncpoint value vs what libnx thinks it asked for.
          *
          * nouveau_exec requests an increment TWICE per submit: once via
          * nvGpuChannelIncrFence (KickoffPb BIT(8), the kernel inserts its own
          * increment) and once via our appended gen_fence_cmdlist. The comment
          * above IncrFence records that this exact pairing is what produced 0xd5c
          * historically ("a DOUBLE increment ... the cure is to use exactly ONE"),
          * and v32 reinstated it anyway.
          *
          * If hw > expected and the gap grows ~1 per submit, the double increment
          * is real: the syncpoint runs ahead of libnx's accounting, every
          * nvFenceWait returns instantly because the target is already passed, and
          * the kernel's per-job cleanup never sees the completion it is waiting
          * for -- which is how job state accumulates until a submit cannot be
          * allocated. hw == expected kills that theory outright. */
         u32 hw_syncpt = 0;
         Result srr = nvioctlNvhostCtrl_SyncptRead(nvFenceGetFd(), chan->fence.id, &hw_syncpt);
         SHIM_ERR_FIRST(4,
                        "drm_shim: kickoff 0xd5c: entries=%u/%u fence=%u/%u incr=%u | "
                        "syncpt hw=%u expected=%u delta=%d (readrc=0x%x) | "
                        "BOs=%u/%u %lluKiB (mapped %lluKiB) | channels=%u/%u\n",
                        (unsigned)chan->num_entries, (unsigned)GPFIFO_QUEUE_SIZE,
                        chan->fence.id, chan->fence.value, (unsigned)chan->fence_incr,
                        (unsigned)hw_syncpt, chan->fence.value,
                        (int)((int64_t)hw_syncpt - (int64_t)chan->fence.value), srr,
                        bo_used, (unsigned)SHIM_MAX_BOS,
                        (unsigned long long)(bo_bytes / 1024),
                        (unsigned long long)(bo_mapped_bytes / 1024),
                        chan_used, (unsigned)SHIM_MAX_CHANNELS);

         /* Is GPU memory ACTUALLY exhausted right now, or is 0xd5c about the
          * kernel's per-submit job state?
          *
          * The earlier "BO/nvmap exhaustion is ruled out" conclusion rested on
          * the COUNT (91 of 4096) -- but 4096 is this shim's own table size, not
          * a kernel limit, so it says nothing about BYTES. At the failure we hold
          * ~303 MiB of nvmap objects, and the failure state has now reproduced
          * byte-identically across different builds and sessions
          * (BOs=91/4096 310404KiB), which a merely load-dependent fault should
          * not do.
          *
          * So: try a fresh, modest allocation at the exact moment of failure.
          *   both succeed  -> nvmap and GPU VA are still available, the ceiling
          *                    is NOT total GPU memory, and the per-submit job
          *                    pool lead stands.
          *   nvMapCreate fails -> a byte ceiling is real and the count-based
          *                    exoneration does not cover it.
          *   memalign fails -> host newlib heap, a different problem entirely
          *                    (and one this port has hit before).
          *
          * Deliberately small and fully released, so the probe cannot itself be
          * what pushes anything over an edge. */
         {
            const uint64_t probe_size = 4 * 1024 * 1024;
            void *probe_mem = memalign(0x1000, probe_size);
            if (!probe_mem) {
               SHIM_ERR_FIRST(2, "drm_shim: 0xd5c probe: host memalign FAILED "
                                 "(%lluKiB) -> newlib heap exhausted\n",
                              (unsigned long long)(probe_size / 1024));
            } else {
               NvMap probe_map;
               iova_t probe_va = 0;
               Result prc = nvMapCreate(&probe_map, probe_mem, probe_size,
                                        0x1000, NvKind_Pitch, false);
               Result mrc = 0xFFFFFFFF;   /* "not attempted" */
               if (R_SUCCEEDED(prc)) {
                  mrc = nvAddressSpaceMap(&g_dev.addr_space,
                                          nvMapGetHandle(&probe_map),
                                          true, NvKind_Pitch, &probe_va);
                  if (R_SUCCEEDED(mrc))
                     nvAddressSpaceUnmap(&g_dev.addr_space, probe_va);
                  nvMapClose(&probe_map);
               }
               free(probe_mem);
               SHIM_ERR_FIRST(2, "drm_shim: 0xd5c probe: %lluKiB nvMapCreate=0x%x "
                                 "nvAddressSpaceMap=0x%x\n",
                              (unsigned long long)(probe_size / 1024), prc, mrc);
            }
         }
      }
   }
   return rc;
}

static int nouveau_channel_alloc(struct drm_nouveau_channel_alloc *req)
{
   int slot = -1;
   for (int i = 0; i < SHIM_MAX_CHANNELS; i++) {
      if (!g_dev.channels[i].used) { slot = i; break; }
   }
   if (slot < 0)
      return -ENOSPC;
   struct shim_channel *ch = &g_dev.channels[slot];

   Result rc = nvGpuChannelCreate(&ch->chan, &g_dev.addr_space,
                                  NvChannelPriority_Medium);
   if (R_FAILED(rc)) {
      SHIM_LOG("nvGpuChannelCreate failed rc=0x%x\n", rc);
      return -EIO;
   }

   /* Zcull context. */
   uint64_t zcull_va = 0;
   ch->zcull_bo = shim_internal_bo_locked(nvGpuGetZcullCtxSize(), 0x20000, &zcull_va);
   if (ch->zcull_bo < 0)
      goto fail_chan;
   rc = nvGpuChannelZcullBind(&ch->chan, zcull_va);
   if (R_FAILED(rc)) {
      SHIM_LOG("nvGpuChannelZcullBind failed rc=0x%x\n", rc);
      goto fail_bos;
   }

   /* Builtin cmdbuf holding the fence + flush command lists. */
   ch->cmdbuf_bo = shim_internal_bo_locked(0x1000, 0x20000, &ch->cmdbuf_va);
   if (ch->cmdbuf_bo < 0)
      goto fail_bos;

   uint32_t *cmds = (uint32_t *)g_dev.bos[ch->cmdbuf_bo].cpu;
   ch->fence_num_cmds = gen_fence_cmdlist(cmds, nvGpuChannelGetSyncpointId(&ch->chan));
   ch->flush_num_cmds = gen_flush_cmdlist(cmds + ch->fence_num_cmds);
   ch->setobj_num_cmds = gen_setobj_cmdlist(cmds + ch->fence_num_cmds + ch->flush_num_cmds);

   /* NOTE: do NOT AllocObjCtx here — libnx's nvGpuChannelCreate already did
    * AllocObjCtx(0xB197) at channel creation (the GR/graphics context is built
    * eagerly there). Calling it again returns -EINVAL "one obj ctx per channel". */

   ch->used = true;
   req->channel = (int32_t)(slot + 1);
   SHIM_LOG("CHANNEL_ALLOC -> %d (syncpt %u)\n",
            req->channel, nvGpuChannelGetSyncpointId(&ch->chan));
   return 0;

fail_bos:
   if (ch->zcull_bo >= 0) {
      nvAddressSpaceUnmap(&g_dev.addr_space, g_dev.bos[ch->zcull_bo].gpu_va);
      nvMapClose(&g_dev.bos[ch->zcull_bo].map);
      free(g_dev.bos[ch->zcull_bo].cpu);
      memset(&g_dev.bos[ch->zcull_bo], 0, sizeof(g_dev.bos[ch->zcull_bo]));
   }
fail_chan:
   nvGpuChannelClose(&ch->chan);
   memset(ch, 0, sizeof(*ch));
   return -ENOMEM;
}

static int nouveau_channel_free(struct drm_nouveau_channel_free *req)
{
   struct shim_channel *ch = channel_lookup((uint32_t)req->channel);
   if (!ch)
      return 0;
   /* Submit anything still batched before the channel goes away: closing over a
    * deferred batch would drop that work on the floor AND strand every syncobj
    * it promised to signal in the `pending` state, with no channel left to
    * resolve them. */
   channel_flush_locked(ch);
   nvGpuChannelClose(&ch->chan);
   for (int bo = ch->cmdbuf_bo; bo == ch->cmdbuf_bo; ) { /* free cmdbuf */
      if (bo >= 0 && g_dev.bos[bo].used) {
         nvAddressSpaceUnmap(&g_dev.addr_space, g_dev.bos[bo].gpu_va);
         nvMapClose(&g_dev.bos[bo].map);
         free(g_dev.bos[bo].cpu);
         memset(&g_dev.bos[bo], 0, sizeof(g_dev.bos[bo]));
      }
      break;
   }
   if (ch->zcull_bo >= 0 && g_dev.bos[ch->zcull_bo].used) {
      nvAddressSpaceUnmap(&g_dev.addr_space, g_dev.bos[ch->zcull_bo].gpu_va);
      nvMapClose(&g_dev.bos[ch->zcull_bo].map);
      free(g_dev.bos[ch->zcull_bo].cpu);
      memset(&g_dev.bos[ch->zcull_bo], 0, sizeof(g_dev.bos[ch->zcull_bo]));
   }
   memset(ch, 0, sizeof(*ch));
   return 0;
}

/* Map / unmap a single VM_BIND op onto the libnx address space.
 *
 * Tegra nvgpu rejects bare fixed-VA reservation (-EINVAL): a FIXED map is only
 * valid *inside* a region previously reserved non-fixed (validate_fixed_buffer).
 * We reserve one big small-page arena at nv-up and point NVK's VA heap at it
 * (nvkmd_nouveau_create_dev patch), so op->addr is always inside that arena and
 * we just MAP_BUFFER_EX FIXED with the small-page (4 KB) size — no per-bind
 * AllocSpace. The page size is forced by the VA's half anyway (low = 4 KB). */
static int vm_bind_op(const struct drm_nouveau_vm_bind_op *op)
{
   const bool     sparse = (op->flags & DRM_NOUVEAU_VM_BIND_SPARSE) != 0;
   const uint32_t fd     = g_dev.addr_space.fd;

   /* NVK issues an unbind as OP_UNMAP, and also (quirk) as OP_MAP with no
    * handle and no SPARSE flag. Treat both as unmap. */
   const bool is_unmap = (op->op == DRM_NOUVEAU_VM_BIND_OP_UNMAP) ||
                         (op->op == DRM_NOUVEAU_VM_BIND_OP_MAP &&
                          op->handle == 0 && !sparse);

   if (is_unmap) {
      nvioctlNvhostAsGpu_UnmapBuffer(fd, op->addr);  /* leaves the arena reserved */
      return 0;
   }

   /* Sparse reservation only: the whole arena is already reserved — nothing to
    * back, the VA is valid. */
   if (sparse && op->handle == 0)
      return 0;

   struct shim_bo *bo = bo_lookup(op->handle);
   if (!bo)
      return -ENOENT;

   /* PTE kind for the mapping. NVK's new-uAPI VM_BIND carries the kind in the
    * LOW BYTE of op->flags (nvkmd_nouveau_va.c: bind.flags = va->pte_kind; bit 8
    * = SPARSE). An OPTIMAL image gets tile_flags=0 at GEM_NEW (so bo->kind=0),
    * but its real block-linear kind (e.g. ZF32=0x7b for a D32 depth/ZETA surface,
    * or a colour kind) arrives HERE in op->flags. Honor op->flags; fall back to
    * bo->kind only for dedicated/modifier images that set it at GEM_NEW.
    * (Was bo->kind alone, which mapped the block-linear ZETA depth surface as
    * Pitch -> GM20B GR fault on CLEAR_SURFACE(Z). Confirmed via NVK source.) */
   NvKind map_kind = (NvKind)(op->flags & 0xff);
   if (map_kind == 0) map_kind = bo->kind;
   Result rc = nvioctlNvhostAsGpu_MapBufferEx(
      fd,
      NvMapBufferFlags_FixedOffset | NvMapBufferFlags_IsCacheable,
      map_kind, nvMapGetHandle(&bo->map), 0x1000 /* small page */,
      op->bo_offset, op->range, op->addr, NULL);
   SHIM_LOG("VM_BIND MAP h=%u off=0x%llx range=0x%llx @0x%llx bo_kind=0x%x opflags=0x%x rc=0x%x\n",
            op->handle, (unsigned long long)op->bo_offset,
            (unsigned long long)op->range, (unsigned long long)op->addr,
            (unsigned)bo->kind, op->flags, rc);
   if (R_SUCCEEDED(rc)) {
      bo->gpu_va = op->addr;
      bo->gpu_bo_offset = op->bo_offset;
   } else {
      SHIM_ERR_FIRST(8, "drm_shim: VM_BIND MAP FAILED rc=0x%x h=%u range=0x%llx @0x%llx\n",
               rc, op->handle, (unsigned long long)op->range,
               (unsigned long long)op->addr);
      return -EINVAL;
   }
   return 0;
}

static int nouveau_vm_bind(struct drm_nouveau_vm_bind *req)
{
   const struct drm_nouveau_vm_bind_op *ops =
      (const struct drm_nouveau_vm_bind_op *)(uintptr_t)req->op_ptr;
   for (uint32_t i = 0; i < req->op_count; i++) {
      int ret = vm_bind_op(&ops[i]);
      if (ret)
         return ret;
   }
   /* wait/sig syncs on VM_BIND are honored only for the async path; our binds
    * complete synchronously, so signalled syncobjs are immediately ready. */
   const struct drm_nouveau_sync *sigs =
      (const struct drm_nouveau_sync *)(uintptr_t)req->sig_ptr;
   for (uint32_t i = 0; i < req->sig_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(sigs[i].handle);
      /* empty EXEC: signal immediately (no GPU work) -> the timeline point is reached now. */
      if (s) { s->signaled = true; s->has_fence = false; s->pending = false; s->pending_chan = 0; s->value = sigs[i].timeline_value; }
   }
   return 0;
}

/* Clamp of g_drm_shim_batch_max; 1 == the old kickoff-per-EXEC behaviour. */
static inline uint32_t shim_batch_max(void)
{
   int v = g_drm_shim_batch_max;
   if (v < 1) v = 1;
   if (v > SHIM_MAX_BATCH_EXECS) v = SHIM_MAX_BATCH_EXECS;
   return (uint32_t)v;
}

/* Submit everything appended to `ch` since the last flush, as ONE kickoff.
 *
 * This is now the only place nvGpuChannelKickoff is reached from the EXEC path,
 * and it carries the complete submit protocol that nouveau_exec used to run
 * inline: one syncpoint increment, the fence cmdlist that performs it on the
 * GPU, the depth-1 in-flight throttle, the bounded retry, and the channel reset
 * on failure. None of that is changed. Four throttle shapes were measured on
 * hardware and exactly one is stable; batching deliberately does not touch the
 * shape, it only lets several EXECs' pushbuf entries ride in the same kickoff.
 *
 * ONE increment per KICKOFF, not per EXEC. Every syncobj signalled by any EXEC
 * in the batch gets the batch's single completion fence. That makes a fence
 * report completion no EARLIER than its own EXEC finished, and possibly later --
 * conservative in the only direction that matters, because early is corruption
 * and late is just latency. Predicting a separate fence value per EXEC would
 * instead mean relying on libnx kicking off with fence_incr > 1 and the kernel
 * applying every one of those increments, which has never been validated here;
 * this keeps the accounting at the single-increment case that is known to work.
 *
 * Caller holds g_dev.lock. */
static int channel_flush_locked(struct shim_channel *ch)
{
   if (ch->batch_execs == 0)
      return 0;                  /* nothing appended since the last kickoff */

   const unsigned chan_id = (unsigned)(ch - g_dev.channels) + 1;
   const uint32_t batched = ch->batch_execs;

   /* STANDARD libnx fence path (matches deko3d + libdrm_nouveau, which use this
    * and work): request ONE syncpoint increment via IncrFence (sets KickoffPb
    * BIT(8); the kernel adds the increment), plus the BUILTIN FENCE CMDLIST --
    * the actual Maxwell command that INCREMENTS the syncpt on the GPU
    * (gen_fence_cmdlist: method 0x0B2 with syncpt_id|incr-bit). IncrFence alone
    * only bumps libnx's expected counter; WITHOUT this entry the GPU never
    * increments the syncpt, so the completion fence is never reached. Exactly
    * what libdrm_nouveau/pushbuf.c:226-228 does. */
   nvGpuChannelIncrFence(&ch->chan);
   nvGpuChannelAppendEntry(&ch->chan, ch->cmdbuf_va, ch->fence_num_cmds,
                           GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);

   /* Bound outstanding work before kicking off.
    *
    * MEASURED TRADE-OFF, both on hardware:
    *   with this wait: 3.7 fps, STABLE,  submit-wait 206 ms/frame (78% of frame)
    *   without it:     3.0 fps, CRASHES 0xd5c, submit-wait 0 ms/frame
    * Removing it does eliminate the wait, but the kernel then refuses a submit
    * outright and kickoff_retry's on-demand back-off cannot recover.
    *
    * The wait is now paid once per BATCH rather than once per EXEC, which is the
    * whole point: a single nvFenceWait costs ~3-20 ms depending on how much real
    * GPU work it covers, and at 7-11 EXECs per guest frame that was being paid
    * 7-11 times a frame. Same throttle, same depth, fewer payments. */
   {
      const uint32_t slot = ch->inflight_head;
      if (ch->inflight_valid[slot]) {
         NvFence oldest = ch->inflight[slot];
         const u64 t_wait0 = armGetSystemTick();
         Result tr = nvFenceWait(&oldest, 2000000LL);   /* us => 2 s */
         g_drm_shim_exec_wait_ns += armTicksToNs(armGetSystemTick() - t_wait0);
         if (R_FAILED(tr)) {
            SHIM_ERR_FIRST(4, "drm_shim: inflight throttle TIMEOUT chan=%u fence=%u/%u rc=0x%x\n",
                           chan_id, oldest.id, oldest.value, tr);
         }
         ch->inflight_valid[slot] = false;
      }
   }

   g_drm_shim_kickoff_count++;
   Result rc = kickoff_retry(&ch->chan);
   SHIM_LOG("EXEC kickoff returned rc=0x%x (chan=%u batched=%u entries=%u)\n",
            rc, chan_id, batched, (unsigned)ch->chan.num_entries);
   if (R_FAILED(rc)) {
      /* This was the one submit-path failure with NO release-build trace at all:
       * SHIM_LOG compiles out, so a kickoff that failed its retries returned
       * -EIO in complete silence. NVK turns that into a failed vkQueueSubmit,
       * and the consumer responds by recreating the swapchain. Any hunt for what
       * starts that chain has to be able to see this. */
      SHIM_ERR_FIRST(8, "drm_shim: EXEC KICKOFF FAILED rc=0x%x chan=%u batched=%u\n",
                     rc, chan_id, batched);

      /* RESET THE CHANNEL, THEN DROP THE SUBMIT.
       *
       * Returning -EIO makes NVK fail the vkQueueSubmit, which the consumer turns
       * into abort() -- so ONE refused submit ends the run. 0xd5c is intermittent,
       * which is the whole reason "how far it gets" varies between identical
       * builds. Measured: that path aborts on failure #1, whereas resetting and
       * continuing survived 8 failures in a run.
       *
       * A failed kickoff does NOT clear what was appended, and that turns one
       * refusal into a permanently poisoned channel:
       *   entries=2 incr=1 -> 4/2 -> 6/3 -> 8/4   (accumulating)
       * `entries` climbs because libnx leaves num_entries alone on ioctl failure,
       * so the next append lands on top of the dead submit; fence_incr stacks the
       * same way and chan->fence.value freezes. Dropping work is only safe once
       * that state is cleared -- which is why these belong together and were
       * measured together.
       *
       * engines_bound is re-armed because the SET_OBJECT binds ride on the first
       * submit; discarding it without re-emitting them would leave NVK methods
       * hitting an unbound subchannel -> MMU fault -> dead channel.
       *
       * The signalled syncobjs MUST be complete-with-NO-fence: attaching the
       * channel fence would point them at a syncpoint value the GPU never reaches
       * (this work was never submitted) and anything waiting would hang forever --
       * worse than the abort being replaced. This releases the syncobjs of EVERY
       * EXEC in the batch, because the reset discards all of their entries, not
       * just the last one's. */
      ch->chan.num_entries = 0;
      ch->chan.fence_incr = 0;
      ch->engines_bound = false;
      for (uint32_t i = 0; i < ch->batch_sig_count; i++) {
         struct shim_syncobj *s = syncobj_lookup(ch->batch_sig[i]);
         if (s) { s->has_fence = false; s->signaled = true; s->pending = false; s->pending_chan = 0; }
      }
      ch->batch_sig_count = 0;
      ch->batch_execs = 0;

      /* KEEP THE THROTTLE ARMED ACROSS A FAILURE.
       *
       * This path used to return without recording an in-flight fence, so after
       * the very first 0xd5c the depth-1 throttle had nothing left to wait on
       * and never engaged again. Measured on hardware: submit-wait collapses to
       * 0.33 ms/frame (0% of frame) from 15-18 ms (36-55%), and failures
       * multiply to 8 in a run. Dropping back-pressure at exactly the moment
       * the kernel is refusing submits is backwards -- it guarantees the next
       * submit arrives sooner than the last one that was already refused.
       *
       * chan.fence, NOT nvGpuChannelGetFence(): that returns value + fence_incr,
       * the target THIS failed submit would have reached, and the GPU never
       * will because the work was never kicked off -- waiting on it can only
       * time out. chan.fence is what previously ACCEPTED kickoffs asked the GPU
       * to reach, so it is reachable and waiting on it is real back-pressure. */
      ch->inflight[ch->inflight_head] = ch->chan.fence;
      ch->inflight_valid[ch->inflight_head] = true;
      ch->inflight_head = (ch->inflight_head + 1) % SHIM_MAX_INFLIGHT;
      return 0;
   }

   /* The real completion fence: kickoff advanced c->fence by the increment. */
   NvFence fence;
   nvGpuChannelGetFence(&ch->chan, &fence);

   ch->inflight[ch->inflight_head] = fence;
   ch->inflight_valid[ch->inflight_head] = true;
   ch->inflight_head = (ch->inflight_head + 1) % SHIM_MAX_INFLIGHT;

   for (uint32_t i = 0; i < ch->batch_sig_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(ch->batch_sig[i]);
      if (s) { s->fence = fence; s->has_fence = true; s->signaled = true; s->pending = false; s->pending_chan = 0; }
   }
   ch->batch_sig_count = 0;
   ch->batch_execs = 0;

   /* Optional synchronous drain -- OFF by default.
    *
    * This used to block on EVERY submit until the GPU had fully finished it,
    * which means the CPU and GPU never overlapped: no pipelining, no queueing,
    * one submit at a time. It was a bring-up crutch. The correctness it provided
    * is not lost: NVK still gets a real fence on every signalled syncobj
    * (attached above), so anything that actually needs completion waits on it.
    *
    * Set to 1 to restore the old lock-step behaviour when debugging a suspected
    * GPU execution fault. */
   const int sync_period = g_drm_shim_sync_submit;
   static uint64_t s_kickoffs_since_drain = 0;
   if (sync_period > 0 && (++s_kickoffs_since_drain % (uint64_t)sync_period) == 0) {
      /* nvFenceWait timeout is in MICROSECONDS (not ns). 2e6 us = 2 s. */
      Result wr = nvFenceWait(&fence, 2000000LL);
      SHIM_LOG("EXEC drain chan=%u fence=%u/%u rc=0x%x\n",
               chan_id, fence.id, fence.value, wr);
      if (R_FAILED(wr)) {
         /* The GPU did not finish this submit. Reporting success here told NVK to
          * keep queueing work onto a channel that is stuck or has already been
          * reset by the kernel, and the display stack went down with it. Failing
          * the submit turns that into VK_ERROR_DEVICE_LOST instead. */
         shim_err("drm_shim: EXEC drain TIMEOUT chan=%u fence=%u/%u rc=0x%x\n",
                  chan_id, fence.id, fence.value, wr);
         return -ETIMEDOUT;
      }
      {
         /* An extra ioctl per submit; only worth paying while draining. */
         u32 cur = 0;
         Result srr = nvioctlNvhostCtrl_SyncptRead(nvFenceGetFd(), fence.id, &cur);
         SHIM_LOG("EXEC drain syncpt[%u] current=%u target=%u reached=%d readrc=0x%x\n",
                  fence.id, (unsigned)cur, fence.value,
                  (int)(cur >= fence.value), srr);
      }
   }

   /* Read the channel error notifier: the kickoff is accepted (rc=0) but the
    * GPU executes ASYNC and may FAULT (MMU fault / illegal method / PBDMA err),
    * which resets the channel and makes every later submit return 0xd5c.
    * info32 = NvNotificationType: 31=MMU fault, 25=GR illegal-method,
    * 32=PBDMA error, 8=idle timeout, 13=GR SW notify, 0=no fault. */
   NvNotification notif = {0};
   if (R_SUCCEEDED(nvGpuChannelGetErrorNotification(&ch->chan, &notif)) &&
       (notif.info32 || notif.info16)) {
      SHIM_LOG("EXEC ERRNOTIF chan=%u type=%u info16=%u (31=MMU 25=illegal 32=PBDMA 8=idle)\n",
               chan_id, notif.info32, notif.info16);
      /* Dump the detailed error info — for an MMU fault (type 31) this carries
       * the FAULTING GPU VA (+ engine/client/reason), which tells us EXACTLY
       * which buffer the 3D init dereferenced that we didn't map. */
      NvError err;
      memset(&err, 0, sizeof(err));
      if (R_SUCCEEDED(nvGpuChannelGetErrorInfo(&ch->chan, &err)))
         SHIM_LOG("EXEC ERRINFO type=%u info: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                  err.type, err.info[0], err.info[1], err.info[2], err.info[3],
                  err.info[4], err.info[5], err.info[6], err.info[7],
                  err.info[8], err.info[9], err.info[10], err.info[11]);

      /* The GPU faulted. The kernel has reset this channel, so every later submit
       * on it is meaningless; reporting success just kept NVK feeding a dead
       * channel until the display stack died with it. Surface it instead. */
      shim_err("drm_shim: GPU FAULT chan=%u type=%u info16=%u "
               "(31=MMU 25=illegal-method 32=PBDMA 8=idle)\n",
               chan_id, notif.info32, notif.info16);
      return -EIO;
   }

   return 0;
}

/* Submit every channel's pending batch.
 *
 * Called from each path that OBSERVES GPU completion. A deferred batch hands out
 * no fences, so a waiter that did not flush first would be waiting on a syncobj
 * whose work is still sitting in our ring -- the wait could only ever time out.
 * Caller holds g_dev.lock. */
static void shim_flush_all_locked(void)
{
   for (int i = 0; i < SHIM_MAX_CHANNELS; i++) {
      struct shim_channel *c = &g_dev.channels[i];
      if (c->used && c->batch_execs)
         channel_flush_locked(c);
   }
}

static int nouveau_exec(struct drm_nouveau_exec *req)
{
   struct shim_channel *ch = channel_lookup((uint32_t)req->channel);
   if (!ch) {
      SHIM_ERR_FIRST(4, "drm_shim: EXEC on UNKNOWN channel %u\n", (unsigned)req->channel);
      return -ENODEV;
   }

   SHIM_LOG("EXEC enter chan=%u push_count=%u wait=%u sig=%u\n",
            (unsigned)req->channel, req->push_count, req->wait_count, req->sig_count);

   const struct drm_nouveau_exec_push *pushes =
      (const struct drm_nouveau_exec_push *)(uintptr_t)req->push_ptr;
   const struct drm_nouveau_sync *waits =
      (const struct drm_nouveau_sync *)(uintptr_t)req->wait_ptr;
   const struct drm_nouveau_sync *sigs =
      (const struct drm_nouveau_sync *)(uintptr_t)req->sig_ptr;

   /* CPU-wait the dependencies before submitting. Correct (if coarse): GPU-side
    * semaphore waits would be the optimization.
    *
    * A dependency signalled by an EXEC still sitting in a deferred batch has no
    * fence yet, and must not be silently skipped -- that would submit work
    * reading buffers the GPU has not written. But how it is resolved decides
    * whether batching happens at all:
    *
    *   SAME channel  -> nothing to do. The GPFIFO executes in order, so work
    *                    appended earlier in this batch is guaranteed to run
    *                    before what we are appending now. No flush, no wait.
    *   OTHER channel -> flush THAT channel so the dependency acquires a real
    *                    fence, then wait on it below.
    *
    * The first version flushed everything whenever wait_count was non-zero.
    * NVK attaches a dependency syncobj to essentially every submit, so that
    * flushed the batch on every single EXEC and batching did nothing at all:
    * hardware measured kickoffs/frame EXACTLY equal to execs/frame (6.4/6.4,
    * 9.4/9.4). Same-channel dependencies are the common case and they are free. */
   const uint32_t this_chan = (uint32_t)req->channel;
   for (uint32_t i = 0; i < req->wait_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(waits[i].handle);
      if (s && s->pending && s->pending_chan != this_chan) {
         struct shim_channel *dep = channel_lookup(s->pending_chan);
         if (dep)
            channel_flush_locked(dep);
      }
   }
   for (uint32_t i = 0; i < req->wait_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(waits[i].handle);
      if (s && s->has_fence)
         nvFenceWait(&s->fence, 2000000LL); /* us! 2s cap (was -1=infinite) */
   }

   /* Empty EXEC (no pushbuf to run): do NOT submit a spurious fence cmdlist +
    * kickoff — that was failing KickoffPb 0xd5c right after the real init submit
    * succeeded. There is nothing to execute; just signal any sig syncs with the
    * channel's current fence (the last submitted work) and return. */
   if (req->push_count == 0) {
      /* An empty EXEC means "everything submitted on this channel so far is
       * complete". With a batch pending that is EXACTLY what the batch's own
       * completion fence will mean, so these syncobjs can just join the batch
       * and collect that fence when it is kicked off -- no flush needed.
       *
       * This used to flush, and it was a major cause of shallow batches: NVK
       * issues fence-only submits regularly, and every one of them ended the
       * batch. Hardware measured batched=1..2 against a cap of 8, which left
       * the submit rate high enough that turning on sub-native rendering (43%
       * of the pixels, so many more frames per second) brought 0xd5c straight
       * back -- 8 failures a run, no main menu.
       *
       * With no batch pending there is nothing unsubmitted, so the channel's
       * current fence already covers everything and the old flush call was a
       * no-op on that path anyway. */
      if (ch->batch_execs != 0 &&
          ch->batch_sig_count + req->sig_count <= SHIM_MAX_BATCH_SIGS) {
         for (uint32_t i = 0; i < req->sig_count; i++) {
            struct shim_syncobj *s = syncobj_lookup(sigs[i].handle);
            if (!s)
               continue;
            s->value        = sigs[i].timeline_value;
            s->has_fence    = false;
            s->signaled     = false;
            s->pending      = true;
            s->pending_chan = (uint32_t)req->channel;
            ch->batch_sig[ch->batch_sig_count++] = sigs[i].handle;
         }
         return 0;
      }
      NvFence f;
      nvGpuChannelGetFence(&ch->chan, &f);
      for (uint32_t i = 0; i < req->sig_count; i++) {
         struct shim_syncobj *s = syncobj_lookup(sigs[i].handle);
         /* Attach the GPU fence + record the TIMELINE point this submit signals. The point
          * counts as "reached" only when this fence completes (drmSyncobjQuery polls it). */
         if (s) { s->fence = f; s->has_fence = true; s->signaled = true; s->pending = false; s->pending_chan = 0; s->value = sigs[i].timeline_value; }
      }
      return 0;
   }

   /* Make room before appending. Either of these overflowing would lose work
    * silently: nvGpuChannelAppendEntry fails once the fixed 2048-entry ring is
    * full, and a syncobj that does not fit in batch_sig would never receive the
    * batch's fence (nor be released if the kickoff failed). Both are reachable
    * only because batching lets state accumulate across EXECs, so both are
    * resolved the same way -- submit what is already queued and start fresh. */
   if (ch->batch_execs &&
       ((uint32_t)ch->chan.num_entries + req->push_count + 2u > SHIM_BATCH_ENTRY_LIMIT ||
        ch->batch_sig_count + req->sig_count > SHIM_MAX_BATCH_SIGS)) {
      channel_flush_locked(ch);
   }

   /* First real submit on this channel: bind the engine classes to their
    * subchannels (SET_OBJECT) FIRST, so NVK's methods route to a bound engine
    * instead of MMU-faulting on an unbound subchannel. Prepended in the SAME
    * kickoff, before NVK's pushbuf, so the GPU executes the binds first. */
   if (!ch->engines_bound) {
      ch->engines_bound = true;
      uint64_t setobj_va = ch->cmdbuf_va + 4 * (ch->fence_num_cmds + ch->flush_num_cmds);
      nvGpuChannelAppendEntry(&ch->chan, setobj_va, ch->setobj_num_cmds,
                              GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
      SHIM_LOG("EXEC bind engines (setobj %u dw) on chan=%u\n",
               ch->setobj_num_cmds, (unsigned)req->channel);
   }

   for (uint32_t i = 0; i < req->push_count; i++) {
      /* Force NOT_MAIN | NO_PREFETCH on every push, matching deko3d's working 3D
       * submissions. NVK passes flags=0 (MAIN + prefetch ENABLED); on Horizon a
       * prefetch-enabled large 3D IB triggers a staging/validation the host1x
       * path never hits -> KickoffPb NvError_InsufficientMemory (0xd5c). Every
       * shim submit that SUCCEEDS on HW used NO_PREFETCH. (Sourced: deko3d
       * dk_queue.cpp, envytools dma-pusher, switchbrew NV_services.) */
      uint32_t flags = GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH;
      bool in_arena = pushes[i].va >= g_dev.va_base &&
                      pushes[i].va + pushes[i].va_len <= g_dev.va_base + g_dev.va_size;
      SHIM_LOG("EXEC push[%u] va=0x%llx len=0x%x in_arena=%d flags=0x%x\n",
               i, (unsigned long long)pushes[i].va, pushes[i].va_len,
               in_arena, pushes[i].flags);
#ifdef DRM_SHIM_DEBUG
      /* Peek the pushbuf's leading method words (find the BO covering the VA and
       * read its CPU mapping).
       *
       * DEBUG ONLY. This is a linear scan of up to SHIM_MAX_BOS (4096) entries
       * PER PUSH, PER SUBMIT, and it exists purely to feed the SHIM_LOG below --
       * which compiles out in release, leaving the scan behind doing nothing at
       * full cost on the hottest path in the driver. */
      for (int b = 0; b < SHIM_MAX_BOS; b++) {
         struct shim_bo *pb = &g_dev.bos[b];
         if (pb->used && pb->gpu_va && pushes[i].va >= pb->gpu_va &&
             pushes[i].va < pb->gpu_va + pb->size) {
            const uint32_t *w = (const uint32_t *)((const char *)pb->cpu +
                                pb->gpu_bo_offset + (pushes[i].va - pb->gpu_va));
            SHIM_LOG("EXEC push content: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                     w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
            break;
         }
      }
#endif
      /* One GPFIFO entry for the whole pushbuf (a single entry holds up to ~2M
       * dwords — research confirmed chunking is unnecessary; the per-submit
       * 0xd5c is a kernel kzalloc under memory pressure, fixed by priming the
       * channel early in nouveau_channel_alloc). */
      nvGpuChannelAppendEntry(&ch->chan, pushes[i].va,
                              pushes[i].va_len / 4, flags, 0);
   }

   /* Record what this EXEC promised to signal, and DEFER the kickoff.
    *
    * The syncobjs are left explicitly NOT complete and WITHOUT a fence. The
    * immediate-kickoff version could mark them signalled here because by this
    * point it had already submitted; we have not, so doing the same would tell
    * NVK the GPU was finished with buffers it has not even been handed -- it
    * would recycle them under a still-pending draw. `pending` records that the
    * real fence is owed, and every path that observes completion flushes first,
    * which is what pays it. */
   for (uint32_t i = 0; i < req->sig_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(sigs[i].handle);
      if (!s)
         continue;
      s->value     = sigs[i].timeline_value;
      s->has_fence = false;
      s->signaled  = false;
      s->pending   = true;
      s->pending_chan = (uint32_t)req->channel;
      /* Capacity was made above, so this cannot drop a handle. */
      if (ch->batch_sig_count < SHIM_MAX_BATCH_SIGS)
         ch->batch_sig[ch->batch_sig_count++] = sigs[i].handle;
   }

   ch->batch_execs++;
   g_drm_shim_exec_count++;

   /* Submit once the batch is full. Everything else is flushed on demand by a
    * waiter, so work is never stranded -- but waiting for a waiter EVERY time
    * would leave the GPU idle while the CPU builds the rest of the frame, and
    * this port is GPU-bound. The cap keeps the GPU fed while still collapsing
    * the 7-11 EXECs of a guest frame into a couple of kickoffs. */
   if (ch->batch_execs >= shim_batch_max())
      return channel_flush_locked(ch);

   return 0;
}

/* ------------------------------------------------------------------------- */
/* drmCommandWrite / drmCommandWriteRead — the nouveau ioctl front door.       */
/* `cmd` is the DRM_NOUVEAU_* index (drmCommandWrite would normally encode it   */
/* into an ioctl request; we receive it directly and dispatch).                */
/* ------------------------------------------------------------------------- */
static int nouveau_dispatch(int fd, unsigned long cmd, void *data, unsigned long size)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   if (!g_dev.nv_up)          { errno = ENODEV; return -ENODEV; }

   int ret;
   mutexLock(&g_dev.lock);
   switch (cmd) {
   case DRM_NOUVEAU_GETPARAM:
      ret = nouveau_getparam((struct drm_nouveau_getparam *)data);
      break;
   case DRM_NOUVEAU_NVIF:
      ret = nouveau_nvif(data, size);
      break;
   case DRM_NOUVEAU_VM_INIT:
      ret = 0;   /* accept => NVK sets has_vm_bind (requires "kernel >= 6.6")  */
      break;
   case DRM_NOUVEAU_VM_BIND:
      ret = nouveau_vm_bind((struct drm_nouveau_vm_bind *)data);
      break;
   case DRM_NOUVEAU_EXEC:
      ret = nouveau_exec((struct drm_nouveau_exec *)data);
      break;
   case DRM_NOUVEAU_GEM_NEW:
      ret = nouveau_gem_new((struct drm_nouveau_gem_new *)data);
      break;
   case DRM_NOUVEAU_GEM_INFO:
      ret = nouveau_gem_info((struct drm_nouveau_gem_info *)data);
      break;
   case DRM_NOUVEAU_GEM_CPU_PREP:
      ret = nouveau_gem_cpu_prep((struct drm_nouveau_gem_cpu_prep *)data);
      break;
   case DRM_NOUVEAU_CHANNEL_ALLOC:
      ret = nouveau_channel_alloc((struct drm_nouveau_channel_alloc *)data);
      break;
   case DRM_NOUVEAU_CHANNEL_FREE:
      ret = nouveau_channel_free((struct drm_nouveau_channel_free *)data);
      break;
   default:
      SHIM_LOG("unhandled DRM_NOUVEAU cmd=0x%lx\n", cmd);
      ret = -EINVAL;
      break;
   }
   mutexUnlock(&g_dev.lock);

   SHIM_LOG("ioctl cmd=0x%lx -> %d\n", cmd, ret);
   if (ret < 0)
      errno = -ret;
   return ret;
}

int drmCommandWrite(int fd, unsigned long cmd, void *data, unsigned long size)
{
   return nouveau_dispatch(fd, cmd, data, size);
}

int drmCommandWriteRead(int fd, unsigned long cmd, void *data, unsigned long size)
{
   return nouveau_dispatch(fd, cmd, data, size);
}

/* ------------------------------------------------------------------------- */
/* Version / device enumeration.                                              */
/* ------------------------------------------------------------------------- */
drmVersionPtr drmGetVersion(int fd)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return NULL; }

   drmVersionPtr v = calloc(1, sizeof(*v));
   if (!v)
      return NULL;

   /* NVK requires version >= 0x01000301 (1.3.1); report the new-uAPI 1.4.0. */
   v->version_major      = 1;
   v->version_minor      = 4;
   v->version_patchlevel = 0;
   v->name = strdup("nouveau");
   v->date = strdup("20240101");
   v->desc = strdup("nVidia Riva/TNT, GeForce, Quadro, Tesla (switch-nvk shim)");
   if (!v->name || !v->date || !v->desc) {
      free(v->name); free(v->date); free(v->desc); free(v);
      return NULL;
   }
   v->name_len = (int)strlen(v->name);
   v->date_len = (int)strlen(v->date);
   v->desc_len = (int)strlen(v->desc);
   SHIM_LOG("drmGetVersion -> nouveau 1.4.0\n");
   return v;
}

void drmFreeVersion(drmVersionPtr v)
{
   if (!v)
      return;
   free(v->name);
   free(v->date);
   free(v->desc);
   free(v);
}

/* Build the one synthetic platform device NVK enumerates: a "nvidia,*" SOC node
 * exposing only a render node. NVK's nvkmd_nouveau_try_create_pdev accepts a
 * DRM_BUS_PLATFORM device whose first `compatible` string starts with "nvidia,". */
static drmDevicePtr shim_build_device(void)
{
   drmDevicePtr d = calloc(1, sizeof(*d));
   if (!d)
      return NULL;

   d->nodes = calloc(DRM_NODE_MAX, sizeof(char *));
   if (!d->nodes) { free(d); return NULL; }
   d->nodes[DRM_NODE_RENDER] = strdup(DRM_SHIM_RENDER_NODE);
   d->available_nodes = 1 << DRM_NODE_RENDER;
   d->bustype = DRM_BUS_PLATFORM;

   drmPlatformBusInfoPtr bus = calloc(1, sizeof(*bus));
   drmPlatformDeviceInfoPtr di = calloc(1, sizeof(*di));
   char **compat = calloc(2, sizeof(char *));
   if (!d->nodes[DRM_NODE_RENDER] || !bus || !di || !compat) {
      free(compat); free(di); free(bus);
      if (d->nodes) free(d->nodes[DRM_NODE_RENDER]);
      free(d->nodes); free(d);
      return NULL;
   }
   strncpy(bus->fullname, "platform-gm20b", sizeof(bus->fullname) - 1);
   compat[0] = strdup("nvidia,gm20b");
   compat[1] = NULL;
   di->compatible = compat;
   d->businfo.platform = bus;
   d->deviceinfo.platform = di;
   return d;
}

static void shim_free_device(drmDevicePtr d)
{
   if (!d)
      return;
   if (d->deviceinfo.platform) {
      char **compat = d->deviceinfo.platform->compatible;
      if (compat) {
         for (int i = 0; compat[i]; i++)
            free(compat[i]);
         free(compat);
      }
      free(d->deviceinfo.platform);
   }
   free(d->businfo.platform);
   if (d->nodes) {
      for (int i = 0; i < DRM_NODE_MAX; i++)
         free(d->nodes[i]);
      free(d->nodes);
   }
   free(d);
}

int drmGetDevices2(uint32_t flags, drmDevicePtr devices[], int max_devices)
{
   (void)flags;
   SHIM_LOG("drmGetDevices2 ENTER (max_devices=%d devices=%p)\n",
            max_devices, (void *)devices);
   /* devices==NULL is the "just count" probe libdrm supports. */
   if (devices == NULL || max_devices < 1)
      return 1;
   drmDevicePtr d = shim_build_device();
   if (!d) {
      SHIM_LOG("drmGetDevices2: shim_build_device FAILED\n");
      return -ENOMEM;
   }
   devices[0] = d;
   SHIM_LOG("drmGetDevices2 -> 1 (synthetic GM20B platform device)\n");
   return 1;
}

int drmGetDevices(drmDevicePtr devices[], int max_devices)
{
   return drmGetDevices2(0, devices, max_devices);
}

void drmFreeDevices(drmDevicePtr devices[], int count)
{
   if (!devices)
      return;
   for (int i = 0; i < count; i++) {
      shim_free_device(devices[i]);
      devices[i] = NULL;
   }
}

void drmFreeDevice(drmDevicePtr *device)
{
   if (!device)
      return;
   shim_free_device(*device);
   *device = NULL;
}

/* Look up a device by dev_t. We have exactly one GPU, so return the synthetic
 * GM20B regardless of the id (NVK calls this in create_dev with render_dev). */
int drmGetDeviceFromDevId(dev_t dev_id, uint32_t flags, drmDevicePtr *device)
{
   (void)dev_id; (void)flags;
   if (!device) { errno = EINVAL; return -EINVAL; }
   drmDevicePtr d = shim_build_device();
   if (!d) { errno = ENOMEM; return -ENOMEM; }
   *device = d;
   SHIM_LOG("drmGetDeviceFromDevId -> synthetic GM20B\n");
   return 0;
}

/* ------------------------------------------------------------------------- */
/* dma-buf / PRIME — unused on the Switch (no external memory for the          */
/* placeholder triangle). Fail cleanly.                                        */
/* ------------------------------------------------------------------------- */
int drmPrimeHandleToFD(int fd, uint32_t handle, uint32_t flags, int *prime_fd)
{
   (void)fd; (void)handle; (void)flags;
   if (prime_fd) *prime_fd = -1;
   errno = ENOSYS;
   return -ENOSYS;
}

int drmPrimeFDToHandle(int fd, int prime_fd, uint32_t *handle)
{
   (void)fd; (void)prime_fd;
   if (handle) *handle = 0;
   errno = ENOSYS;
   return -ENOSYS;
}

/* ------------------------------------------------------------------------- */
/* Syncobjs — a handle table; EXEC attaches a real libnx NvFence (M2), so       */
/* drmSyncobjWait blocks on actual GPU completion via nvFenceWait.              */
/* ------------------------------------------------------------------------- */

/* DRM syncobj waits take an ABSOLUTE CLOCK_MONOTONIC ns deadline; nvFenceWait
 * takes a RELATIVE microsecond timeout (-1 = infinite). Convert. */
static int32_t shim_abs_ns_to_rel_us(int64_t abs_ns)
{
   if (abs_ns < 0)
      return -1;
   uint64_t now = armTicksToNs(armGetSystemTick());
   if ((uint64_t)abs_ns <= now)
      return 0;
   uint64_t rel_us = ((uint64_t)abs_ns - now) / 1000;
   return (rel_us > (uint64_t)INT32_MAX) ? -1 : (int32_t)rel_us;
}

static struct shim_syncobj *syncobj_lookup(uint32_t handle)
{
   if (handle == 0 || handle > SHIM_MAX_SYNCOBJS)
      return NULL;
   struct shim_syncobj *s = &g_dev.syncobjs[handle - 1];
   return s->used ? s : NULL;
}

int drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   mutexLock(&g_dev.lock);
   int slot = -1;
   for (int i = 0; i < SHIM_MAX_SYNCOBJS; i++) {
      if (!g_dev.syncobjs[i].used) { slot = i; break; }
   }
   if (slot < 0) {
      mutexUnlock(&g_dev.lock);
      SHIM_ERR_FIRST(4, "drm_shim: SYNCOBJ TABLE FULL (%d syncobjs)\n", SHIM_MAX_SYNCOBJS);
      errno = ENOMEM;
      return -ENOMEM;
   }
   g_dev.syncobjs[slot].used     = true;
   g_dev.syncobjs[slot].signaled = (flags & DRM_SYNCOBJ_CREATE_SIGNALED) != 0;
   g_dev.syncobjs[slot].value    = 0;
   mutexUnlock(&g_dev.lock);
   if (handle)
      *handle = (uint32_t)slot + 1;
   return 0;
}

int drmSyncobjDestroy(int fd, uint32_t handle)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   mutexLock(&g_dev.lock);
   struct shim_syncobj *s = syncobj_lookup(handle);
   if (s)
      memset(s, 0, sizeof(*s));
   mutexUnlock(&g_dev.lock);
   return s ? 0 : -ENOENT;
}

int drmSyncobjReset(int fd, const uint32_t *handles, uint32_t handle_count)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   mutexLock(&g_dev.lock);
   for (uint32_t i = 0; i < handle_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(handles[i]);
      if (s) { s->signaled = false; s->value = 0; s->has_fence = false; s->pending = false; s->pending_chan = 0; }
   }
   mutexUnlock(&g_dev.lock);
   return 0;
}

int drmSyncobjSignal(int fd, const uint32_t *handles, uint32_t handle_count)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   mutexLock(&g_dev.lock);
   for (uint32_t i = 0; i < handle_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(handles[i]);
      if (s) s->signaled = true;
   }
   mutexUnlock(&g_dev.lock);
   return 0;
}

int drmSyncobjTimelineSignal(int fd, const uint32_t *handles,
                             uint64_t *points, uint32_t handle_count)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   mutexLock(&g_dev.lock);
   for (uint32_t i = 0; i < handle_count; i++) {
      struct shim_syncobj *s = syncobj_lookup(handles[i]);
      if (s) { s->signaled = true; if (points) s->value = points[i]; }
   }
   mutexUnlock(&g_dev.lock);
   return 0;
}

int drmSyncobjQuery(int fd, uint32_t *handles,
                    uint64_t *points, uint32_t handle_count)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   /* Return the timeline value (s->value) ONLY once its GPU fence has completed
    * -- otherwise report 0 (conservative). This keeps NVK's upload-queue recycle
    * from reusing a BO the GPU still reads (a too-high value = corruption); the
    * worst case of "0 while pending" is just an extra fresh BO, never unsafe.
    * Snapshot the fence under the lock, poll it (0 timeout) OUTSIDE the lock. */
   for (uint32_t i = 0; i < handle_count; i++) {
      mutexLock(&g_dev.lock);
      struct shim_syncobj *s = syncobj_lookup(handles[i]);
      /* Still pending => another thread deferred an EXEC signalling it after the
       * flush above. Resolve it HERE, under the lock, where the race cannot
       * reopen: "no fence" is read below as "already reached", which is true of
       * a never-submitted syncobj but catastrophically false for one whose work
       * is merely sitting in a batch. */
      if (s && s->pending) {
         shim_flush_all_locked();
         s = syncobj_lookup(handles[i]);
      }
      bool     has_fence = s && s->has_fence;
      uint64_t value     = s ? s->value : 0;
      NvFence  fence     = has_fence ? s->fence : (NvFence){0};
      mutexUnlock(&g_dev.lock);

      uint64_t reached = value;                 /* no fence => already reached */
      if (has_fence && R_FAILED(nvFenceWait(&fence, 0)))
         reached = 0;                           /* GPU still busy => conservative */
      if (points) points[i] = reached;
   }
   return 0;
}

/* Wait on the GPU fences attached by EXEC. WAIT_ALL (default for NVK's binary
 * waits) blocks on every handle; otherwise return when the first is ready. A
 * handle with no attached fence is already-signalled (CPU-signalled or never
 * submitted). */
int drmSyncobjWait(int fd, uint32_t *handles, unsigned num_handles,
                   int64_t timeout_nsec, unsigned flags,
                   uint32_t *first_signaled)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   int32_t timeout_us = shim_abs_ns_to_rel_us(timeout_nsec);
   /* DEBUG cap: never wait forever, so a non-signalling fence yields a logged
    * timeout instead of a frozen app. The fill should complete in << 3s. */
   if (timeout_us < 0 || timeout_us > 3000000)
      timeout_us = 3000000;
   const bool wait_all = (flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL) != 0;
   int ret = 0;

   for (unsigned i = 0; i < num_handles; i++) {
      /* Snapshot the fence under the lock, then block on the GPU WITHOUT the
       * lock held — otherwise a long wait would stall other threads' submits. */
      mutexLock(&g_dev.lock);
      struct shim_syncobj *s = syncobj_lookup(handles[i]);
      /* See drmSyncobjQuery: resolve a raced deferral under the lock, so that
       * "no fence" below cannot mean "still queued in a batch". */
      if (s && s->pending) {
         shim_flush_all_locked();
         s = syncobj_lookup(handles[i]);
      }
      bool    has_fence = s && s->has_fence;
      NvFence fence     = has_fence ? s->fence : (NvFence){0};
      mutexUnlock(&g_dev.lock);

      bool ready = !has_fence;   /* no fence => already done */
      if (!ready)
         ready = R_SUCCEEDED(nvFenceWait(&fence, timeout_us));

      if (ready) {
         if (first_signaled) *first_signaled = i;
         if (!wait_all)                  /* WAIT_ANY: first ready is enough */
            return 0;
      } else if (wait_all) {
         errno = ETIME;                  /* this one timed out; WAIT_ALL fails */
         return -ETIME;
      }
   }
   if (!wait_all && num_handles > 0) {
      errno = ETIME;                     /* WAIT_ANY: none became ready */
      ret = -ETIME;
   }
   return ret;
}

/* Timeline syncobjs aren't advertised (drmGetCap reports no timeline), but keep
 * a correct binary-style wait in case a caller reaches here. */
int drmSyncobjTimelineWait(int fd, uint32_t *handles, uint64_t *points,
                           unsigned num_handles, int64_t timeout_nsec,
                           unsigned flags, uint32_t *first_signaled)
{
   (void)points;
   return drmSyncobjWait(fd, handles, num_handles, timeout_nsec,
                         flags, first_signaled);
}

/* Cross-process / sync_file sharing is not supported on the Switch. */
int drmSyncobjHandleToFD(int fd, uint32_t handle, int *obj_fd)
{
   (void)fd; (void)handle;
   if (obj_fd) *obj_fd = -1;
   errno = ENOSYS;
   return -ENOSYS;
}

int drmSyncobjFDToHandle(int fd, int obj_fd, uint32_t *handle)
{
   (void)fd; (void)obj_fd;
   if (handle) *handle = 0;
   errno = ENOSYS;
   return -ENOSYS;
}

int drmSyncobjImportSyncFile(int fd, uint32_t handle, int sync_file_fd)
{
   (void)fd; (void)handle; (void)sync_file_fd;
   errno = ENOSYS;
   return -ENOSYS;
}

int drmSyncobjExportSyncFile(int fd, uint32_t handle, int *sync_file_fd)
{
   (void)fd; (void)handle;
   if (sync_file_fd) *sync_file_fd = -1;
   errno = ENOSYS;
   return -ENOSYS;
}

/* Capabilities. We DO advertise binary + TIMELINE syncobj support: NVK's upload
 * queue (nvk_upload_queue) creates a TIMELINE sync (sync_types[0] must have
 * VK_SYNC_FEATURE_TIMELINE) and calls vk_sync_get_value() on the recycle path
 * once the upload BO fills (many shaders -> the game; the placeholder triangle
 * never filled it). Without the timeline cap, NVK's sync type has a NULL
 * get_value -> nvk_upload_queue_reserve calls null -> Instruction Abort. Our
 * drmSyncobjQuery/TimelineSignal/TimelineWait back it (the timeline point lives
 * in shim_syncobj.value; submits drain, so the value reflects GPU completion). */
#ifndef DRM_CAP_SYNCOBJ
#define DRM_CAP_SYNCOBJ          0x13
#endif
#ifndef DRM_CAP_SYNCOBJ_TIMELINE
#define DRM_CAP_SYNCOBJ_TIMELINE 0x14
#endif
int drmGetCap(int fd, uint64_t capability, uint64_t *value)
{
   if (!drm_shim_owns_fd(fd)) { errno = EBADF; return -EBADF; }
   if (!value) return 0;
   switch (capability) {
   case DRM_CAP_SYNCOBJ:
   case DRM_CAP_SYNCOBJ_TIMELINE:
      *value = 1;   /* binary + timeline syncobjs supported (see note above) */
      break;
   default:
      *value = 0;
      break;
   }
   return 0;
}

/* Submit-method MATRIX: tries one way to submit a large (~NVK-sized) IB on a
 * FRESH channel and logs whether KickoffPb accepts it (0x0) or rejects it
 * (0xd5c). All variants use the PROVEN-valid fence cmdlist as filler, so none
 * can fault/wedge the GPU — a failure is just a return code. Each runs on its
 * own channel (freed after), and we drain ~80ms between calls so GPU work does
 * not pile up. The caller must hold g_dev.lock. */
static void selftest_method(const char *name, uint64_t ib_va, uint32_t *ib_cpu,
                            uint32_t total_dw, uint32_t chunk_dw, int n_prime)
{
   struct drm_nouveau_channel_alloc ca = {0};
   if (nouveau_channel_alloc(&ca) != 0) { SHIM_LOG("MATRIX %-14s chan-alloc FAIL\n", name); return; }
   struct shim_channel *c = channel_lookup((uint32_t)ca.channel);
   if (!c) { SHIM_LOG("MATRIX %-14s lookup FAIL\n", name); return; }

   uint32_t spid = nvGpuChannelGetSyncpointId(&c->chan);
   uint32_t fill = 0;
   while (fill + 3 <= total_dw) fill += gen_fence_cmdlist(ib_cpu + fill, spid);

   for (int p = 0; p < n_prime; p++) {   /* optional small "warmup" submits */
      nvGpuChannelAppendEntry(&c->chan, c->cmdbuf_va, c->fence_num_cmds,
                              GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
      nvGpuChannelKickoff(&c->chan);
   }

   Result rc = 0;
   if (chunk_dw == 0) {                  /* one big GPFIFO submit */
      nvGpuChannelAppendEntry(&c->chan, ib_va, fill,
                              GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
      rc = nvGpuChannelKickoff(&c->chan);
   } else {                              /* split into <=chunk_dw kickoffs */
      uint32_t rem = fill; uint64_t va = ib_va;
      while (rem > 0) {
         uint32_t k = rem > chunk_dw ? chunk_dw : rem;
         nvGpuChannelAppendEntry(&c->chan, va, k,
                                 GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
         rc = nvGpuChannelKickoff(&c->chan);
         if (R_FAILED(rc)) break;
         va += (uint64_t)k * 4; rem -= k;
      }
   }
   SHIM_LOG("MATRIX %-14s total=%u chunk=%u prime=%d -> kickoff=0x%x\n",
            name, fill, chunk_dw, n_prime, rc);
   svcSleepThread(80000000ULL);          /* 80 ms: drain GPU before next method */

   struct drm_nouveau_channel_free cf = {0};
   cf.channel = ca.channel;
   nouveau_channel_free(&cf);
}

/* Diagnostic: prove the shim is linked + reachable + its logging works, by
 * exercising the enumeration entrypoints directly from the app. If this leaves
 * a trace but NVK's own enumeration doesn't, the bug is in how NVK calls us. */
void drm_shim_selftest(void)
{
   SHIM_LOG("selftest: begin\n");
   drmDevicePtr devs[4] = {0};
   int n = drmGetDevices2(0, devs, 4);
   SHIM_LOG("selftest: drmGetDevices2 -> %d\n", n);

   int fd = drm_shim_open(DRM_SHIM_RENDER_NODE, 0);
   SHIM_LOG("selftest: open(%s) -> %d\n", DRM_SHIM_RENDER_NODE, fd);
   if (fd >= 0) {
      drmVersionPtr v = drmGetVersion(fd);
      SHIM_LOG("selftest: drmGetVersion name='%s' va_base=0x%llx\n",
               (v && v->name) ? v->name : "(null)",
               (unsigned long long)g_dev.va_base);
      if (v) drmFreeVersion(v);

      /* A/B isolation: can a bare channel submit just our builtin fence cmdlist
       * (a known-mapped command list), with NO NVK pushbufs? If this kickoff
       * succeeds, the channel/submit setup is sound and the EXEC failure is
       * about NVK's pushbuf VAs/binds; if it fails too, the channel itself
       * can't submit. */
      mutexLock(&g_dev.lock);
      struct drm_nouveau_channel_alloc ca = {0};
      int car = nouveau_channel_alloc(&ca);
      struct shim_channel *tc = (car == 0) ? channel_lookup((uint32_t)ca.channel) : NULL;
      SHIM_LOG("selftest: CHANNEL_ALLOC rc=%d ch=%d\n", car, ca.channel);
      if (tc) {
         /* Test 1: submit the channel's builtin cmdbuf (big-page, nvAddressSpaceMap). */
         nvGpuChannelAppendEntry(&tc->chan, tc->cmdbuf_va, tc->fence_num_cmds,
                                 GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
         Result kr = nvGpuChannelKickoff(&tc->chan);
         SHIM_LOG("selftest: T1 bigpage(cmdbuf) kickoff rc=0x%x\n", kr);

         /* Test 2: same fence cmdlist, but from a BO mapped FIXED into the
          * small-page arena (exactly how NVK's pushbufs are mapped). If this
          * fails while T1 succeeds, small-page IBs are the EXEC blocker. */
         int sl = shim_bo_alloc_locked(0x1000, 0x1000, NvKind_Pitch, false);
         if (sl >= 0 && g_dev.va_base) {
            uint64_t tva = g_dev.va_base + 0x100000;
            Result mr = nvioctlNvhostAsGpu_MapBufferEx(
               g_dev.addr_space.fd,
               NvMapBufferFlags_FixedOffset | NvMapBufferFlags_IsCacheable,
               NvKind_Pitch, nvMapGetHandle(&g_dev.bos[sl].map), 0x1000,
               0, 0x1000, tva, NULL);
            uint32_t n = gen_fence_cmdlist((uint32_t *)g_dev.bos[sl].cpu,
                                           nvGpuChannelGetSyncpointId(&tc->chan));
            nvGpuChannelAppendEntry(&tc->chan, tva, n,
                                    GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
            Result kr2 = nvGpuChannelKickoff(&tc->chan);
            SHIM_LOG("selftest: T2 smallpage(arena) map=0x%x kickoff=0x%x\n", mr, kr2);
         }

         /* T3: a LARGE IB (~1859 dwords, like NVK's) from the arena, to test if
          * the EXEC InsufficientMemory is about IB SIZE rather than content. */
         int sl3 = shim_bo_alloc_locked(0x4000, 0x1000, NvKind_Pitch, false);
         if (sl3 >= 0 && g_dev.va_base) {
            uint64_t tva3 = g_dev.va_base + 0x300000;
            Result mr3 = nvioctlNvhostAsGpu_MapBufferEx(
               g_dev.addr_space.fd,
               NvMapBufferFlags_FixedOffset | NvMapBufferFlags_IsCacheable,
               NvKind_Pitch, nvMapGetHandle(&g_dev.bos[sl3].map), 0x1000,
               0, 0x4000, tva3, NULL);
            uint32_t *w = (uint32_t *)g_dev.bos[sl3].cpu;
            uint32_t spid = nvGpuChannelGetSyncpointId(&tc->chan);
            uint32_t total = 0;
            while (total + 3 <= 1859) total += gen_fence_cmdlist(w + total, spid);
            nvGpuChannelAppendEntry(&tc->chan, tva3, total,
                                    GPFIFO_ENTRY_NOT_MAIN | GPFIFO_ENTRY_NO_PREFETCH, 0);
            Result kr3 = nvGpuChannelKickoff(&tc->chan);
            SHIM_LOG("selftest: T3 large IB (%u dwords) map=0x%x kickoff=0x%x\n",
                     total, mr3, kr3);
         }
      }

      mutexUnlock(&g_dev.lock);

      drm_shim_close(fd);
   }
   if (n > 0)
      drmFreeDevices(devs, n);
   SHIM_LOG("selftest: end\n");
}

#endif /* __SWITCH__ */
