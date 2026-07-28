/*
 * wsi_common_switch.c — VK_NN_vi_surface WSI backend for the Nintendo Switch,
 * presenting through libnx `nwindow` (the VI compositor). Mirrors the structure
 * of wsi_common_headless.c (same wsi_interface contract) and the design reverse-
 * engineered from Dan/Tiicu's wsi_common_switch.c (see D:\switch-nvk\dan-re\RE_NOTES.md).
 *
 * Model (validated against Dan's NROs):
 *   surface  = vkCreateViSurfaceNN(window = nwindowGetDefault())  -> VkIcdSurfaceVi
 *   caps     = RGBA8/BGRA8, FIFO, <=4 images, extent from nwindowGetDimensions
 *   swapchain= per image: a renderable VkImage + a libnx nvmap buffer registered
 *              with the nwindow (nwindowConfigureBuffer); slot i = image i.
 *   acquire  = nwindowDequeueBuffer -> slot -> image; cancel+OUT_OF_DATE if unknown
 *   present  = copy rendered image -> the nwindow buffer, nwindowQueueBuffer(slot)
 *
 * First milestone (correctness-first): CPU-copy present + LINEAR nwindow buffers
 * (no block-linear/GOB matching). Optimize to zero-copy (kind=0xfe block-linear,
 * native-fence present) once pixels are proven.
 *
 * SPDX-License-Identifier: MIT
 */

#include "util/macros.h"
#include "util/u_thread.h"
#include "vk_util.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "wsi_common_entrypoints.h"
#include "wsi_common_private.h"

#include <vulkan/vulkan_vi.h>   /* VkViSurfaceCreateInfoNN (needs VK_USE_PLATFORM_VI_NN) */
#include <switch.h>
#include <stdio.h>              /* snprintf for the present-path profiler */

/* VkIcdSurfaceVi { VkIcdSurfaceBase base; void *window; } is defined by vk_icd.h.
 * .window holds the libnx NWindow* (cast on use). */

/* Present-path profiler sink. Dusklight defines a strong extern "C" dusk_switch_log
 * (writes sdmc:/dusklight.log, pulled over FTP). The weak no-op fallback keeps the
 * standalone smoke NROs (which don't link Dusklight) building/linking unchanged. */
void dusk_switch_log(const char *msg);
__attribute__((weak)) void dusk_switch_log(const char *msg) { (void)msg; }

/* ---- A1 zero-copy present (block-linear, kind=0xfe) ------------------------
 * Helpers from the winsys (drm_shim.c) and the NVK driver (nvk_image.c). The
 * VI compositor scans out the rendered block-linear image directly: present =
 * nwindowQueueBuffer, no per-frame CPU copy/swizzle. Recipe = Dan-RE
 * FUN_71000f2c00 + dan-re/RE_NOTES.md §131-178. */
extern bool drm_shim_bo_nvmap_by_va(uint64_t gpu_va, uint32_t *out_nvmap_id,
                                    uint64_t *out_size, uint32_t *out_kind);
extern bool nvk_switch_image_layout(VkImage image, uint32_t *row_stride_B,
                                    uint32_t *block_height_log2, uint64_t *offset_B,
                                    uint64_t *size_B, uint8_t *pte_kind,
                                    uint64_t *gpu_va);

/* Fill `gb` so nwindowConfigureBuffer registers `image`'s block-linear memory as
 * a scanout buffer. Returns false if the NIL layout / nvmap can't be resolved
 * (caller falls back). memset 0 then set the documented fields; libnx overwrites
 * the NativeHandle header + trailing bookkeeping during marshalling. */
static bool
wsi_switch_build_graphic_buffer(NvGraphicBuffer *gb, VkImage image,
                                VkFormat vk_format, VkExtent2D extent)
{
   uint32_t row_stride_B = 0, block_height_log2 = 0, bo_kind = 0, nvmap_id = 0;
   uint64_t offset_B = 0, nil_size_B = 0, gpu_va = 0, bo_size = 0;
   uint8_t  pte_kind = 0;

   if (!nvk_switch_image_layout(image, &row_stride_B, &block_height_log2,
                                &offset_B, &nil_size_B, &pte_kind, &gpu_va))
      return false;
   if (!drm_shim_bo_nvmap_by_va(gpu_va, &nvmap_id, &bo_size, &bo_kind))
      return false;

   const bool bgra = (vk_format == VK_FORMAT_B8G8R8A8_UNORM);
   memset(gb, 0, sizeof(*gb));
   /* NativeHandle: num_ints = count of u32 words after the handle. WITHOUT this the
    * marshalled GraphicBuffer is empty -> bqSetPreallocatedBuffer "succeeds" but
    * bqDequeueBuffer returns WouldBlock forever (the dequeue-hang bug). Mirrors libnx
    * framebufferCreate. */
   gb->header.num_fds  = 0;
   gb->header.num_ints = (sizeof(NvGraphicBuffer) - sizeof(NativeHandle)) / 4;
   gb->unk0        = -1;
   gb->nvmap_id    = (s32)nvmap_id;
   gb->magic       = 0xDAFFCAFF;
   gb->pid         = 42;
   gb->usage       = 0xb00;                 /* GRALLOC scanout usage (Dan RE §5.2) */
   gb->format      = bgra ? PIXEL_FORMAT_BGRA_8888 : PIXEL_FORMAT_RGBA_8888;
   gb->ext_format  = gb->format;
   gb->stride      = row_stride_B / 4;       /* RGBA8 = 4 B/px; field is in PIXELS  */
   gb->total_size  = (u32)nil_size_B;
   gb->num_planes  = 1;
   gb->planes[0].width             = extent.width;
   gb->planes[0].height            = extent.height;
   gb->planes[0].color_format      = bgra ? NvColorFormat_A8R8G8B8 : NvColorFormat_A8B8G8R8;
   gb->planes[0].layout            = NvLayout_BlockLinear;    /* = 3                  */
   gb->planes[0].pitch             = row_stride_B;
   gb->planes[0].offset            = (u32)offset_B;
   gb->planes[0].kind              = NvKind_Generic_16BX2;    /* = 0xfe, display kind */
   gb->planes[0].block_height_log2 = block_height_log2;
   gb->planes[0].scan              = NvDisplayScanFormat_Progressive;
   gb->planes[0].size              = nil_size_B;
   printf("[wsi-zc] gb: nvmap=%u stride_px=%u pitch_B=%u bh_log2=%u size=%llu pte_kind=%u gpu_va=0x%llx\n",
          nvmap_id, gb->stride, row_stride_B, block_height_log2,
          (unsigned long long)nil_size_B, pte_kind, (unsigned long long)gpu_va);
   fflush(stdout);
   return true;
}

/* ---- surface ---- */

struct wsi_switch {
   struct wsi_interface base;
   struct wsi_device *wsi;
   const VkAllocationCallbacks *alloc;
   VkPhysicalDevice physical_device;
};

VKAPI_ATTR VkResult VKAPI_CALL
wsi_CreateViSurfaceNN(VkInstance _instance,
                      const VkViSurfaceCreateInfoNN *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   VkIcdSurfaceVi *surface;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_VI_SURFACE_CREATE_INFO_NN);

   surface = vk_alloc2(&instance->alloc, pAllocator, sizeof(*surface), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (surface == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   surface->base.platform = VK_ICD_WSI_PLATFORM_VI;
   surface->window = (void *)pCreateInfo->window;

   *pSurface = VkIcdSurfaceBase_to_handle(&surface->base);
   return VK_SUCCESS;
}

/* ---- query callbacks ---- */

static VkResult
wsi_switch_surface_get_support(VkIcdSurfaceBase *surface,
                               struct wsi_device *wsi_device,
                               uint32_t queueFamilyIndex,
                               VkBool32 *pSupported)
{
   *pSupported = true;
   return VK_SUCCESS;
}

static const VkPresentModeKHR present_modes[] = {
   VK_PRESENT_MODE_FIFO_KHR,
};

static VkResult
wsi_switch_surface_get_capabilities(VkIcdSurfaceBase *icd_surface,
                                    struct wsi_device *wsi_device,
                                    VkSurfaceCapabilitiesKHR *caps)
{
   VkIcdSurfaceVi *surface = (VkIcdSurfaceVi *)icd_surface;
   u32 w = 1280, h = 720;
   if (surface->window != NULL)
      nwindowGetDimensions((NWindow *)surface->window, &w, &h);

   caps->minImageCount = 2;
   caps->maxImageCount = 4;
   caps->currentExtent = (VkExtent2D){ w, h };
   caps->minImageExtent = (VkExtent2D){ 1, 1 };
   caps->maxImageExtent = (VkExtent2D){ w, h };
   caps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->maxImageArrayLayers = 1;
   caps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
   caps->supportedUsageFlags = wsi_caps_get_image_usage();
   return VK_SUCCESS;
}

static VkResult
wsi_switch_surface_get_capabilities2(VkIcdSurfaceBase *surface,
                                     struct wsi_device *wsi_device,
                                     const void *info_next,
                                     VkSurfaceCapabilities2KHR *caps)
{
   assert(caps->sType == VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR);
   VkResult result =
      wsi_switch_surface_get_capabilities(surface, wsi_device, &caps->surfaceCapabilities);

   vk_foreach_struct(ext, caps->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR: {
         VkSurfaceProtectedCapabilitiesKHR *protected = (void *)ext;
         protected->supportsProtected = VK_FALSE;
         break;
      }
      default:
         break;
      }
   }
   return result;
}

/* RGBA8 / BGRA8 UNORM, sRGB-nonlinear — the formats the VI compositor scans out. */
static const VkSurfaceFormatKHR switch_formats[] = {
   { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
   { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
};

static VkResult
wsi_switch_surface_get_formats(VkIcdSurfaceBase *icd_surface,
                               struct wsi_device *wsi_device,
                               uint32_t *pSurfaceFormatCount,
                               VkSurfaceFormatKHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (uint32_t i = 0; i < ARRAY_SIZE(switch_formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f) { *f = switch_formats[i]; }
   }
   return vk_outarray_status(&out);
}

static VkResult
wsi_switch_surface_get_formats2(VkIcdSurfaceBase *icd_surface,
                                struct wsi_device *wsi_device,
                                const void *info_next,
                                uint32_t *pSurfaceFormatCount,
                                VkSurfaceFormat2KHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (uint32_t i = 0; i < ARRAY_SIZE(switch_formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, f) {
         f->surfaceFormat = switch_formats[i];
      }
   }
   return vk_outarray_status(&out);
}

static VkResult
wsi_switch_surface_get_present_modes(VkIcdSurfaceBase *surface,
                                     struct wsi_device *wsi_device,
                                     uint32_t *pPresentModeCount,
                                     VkPresentModeKHR *pPresentModes)
{
   if (pPresentModes == NULL) {
      *pPresentModeCount = ARRAY_SIZE(present_modes);
      return VK_SUCCESS;
   }
   *pPresentModeCount = MIN2(*pPresentModeCount, ARRAY_SIZE(present_modes));
   typed_memcpy(pPresentModes, present_modes, *pPresentModeCount);
   return (*pPresentModeCount < ARRAY_SIZE(present_modes)) ? VK_INCOMPLETE : VK_SUCCESS;
}

static VkResult
wsi_switch_surface_get_present_rectangles(VkIcdSurfaceBase *icd_surface,
                                          struct wsi_device *wsi_device,
                                          uint32_t *pRectCount,
                                          VkRect2D *pRects)
{
   VkIcdSurfaceVi *surface = (VkIcdSurfaceVi *)icd_surface;
   u32 w = 1280, h = 720;
   if (surface->window != NULL)
      nwindowGetDimensions((NWindow *)surface->window, &w, &h);
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);
   vk_outarray_append_typed(VkRect2D, &out, rect) {
      *rect = (VkRect2D){ .offset = { 0, 0 }, .extent = { w, h } };
   }
   return vk_outarray_status(&out);
}

/* ---- swapchain ---- */

#define WSI_SWITCH_MAX_IMAGES 4

struct wsi_switch_image {
   struct wsi_image base;
   NvGraphicBuffer gb;        /* registered with the nwindow at slot == image index */
   NvMultiFence acquire_fence;/* producer fence from the last dequeue of this slot   */
   bool configured;           /* nwindowConfigureBuffer succeeded for this slot       */
   bool busy;
};

struct wsi_switch_swapchain {
   struct wsi_swapchain base;
   NWindow *window;
   bool zero_copy;          /* all images registered as nwindow scanout buffers       */
   Framebuffer fb;          /* CPU-copy fallback path (libnx framebuffer)             */
   bool fb_created;
   VkExtent2D extent;
   VkFormat vk_format;
   uint32_t next;           /* round-robin acquire index (fallback path)              */
   struct wsi_switch_image images[WSI_SWITCH_MAX_IMAGES];
};
VK_DEFINE_NONDISP_HANDLE_CASTS(wsi_switch_swapchain, base.base, VkSwapchainKHR,
                               VK_OBJECT_TYPE_SWAPCHAIN_KHR)

static struct wsi_image *
wsi_switch_swapchain_get_wsi_image(struct wsi_swapchain *wsi_chain, uint32_t image_index)
{
   struct wsi_switch_swapchain *chain = (struct wsi_switch_swapchain *)wsi_chain;
   return &chain->images[image_index].base;
}

static VkResult
wsi_switch_swapchain_acquire_next_image(struct wsi_swapchain *wsi_chain,
                                        const VkAcquireNextImageInfoKHR *info,
                                        uint32_t *image_index)
{
   struct wsi_switch_swapchain *chain = (struct wsi_switch_swapchain *)wsi_chain;

   /* Zero-copy: dequeue a free nwindow slot (slot == image index, identity). The
    * producer fence says when the compositor is done reading it — wait it on the
    * CPU so the app never renders over a buffer still on screen (no tearing). */
   if (chain->zero_copy) {
      static uint32_t dbg = 0;
      bool trace = (dbg++ < 4);
      s32 slot = -1;
      NvMultiFence mf;
      memset(&mf, 0, sizeof(mf));
      if (trace) { printf("[wsi-zc] acquire: dequeue...\n"); fflush(stdout); }

      /* Report honestly and return; do NOT retry in here.
       *
       * A retry loop lived here and had to sleep. Calling libnx from inside a
       * mesa-built object jumps through a GOT slot the NRO does not relocate:
       * svcSleepThread resolved to image base + 0, and the loop died on its first
       * sleep (LR pointed straight at this function). The consumer's own code
       * links normally, so the waiting belongs there.
       *
       * VK_TIMEOUT rather than VK_NOT_READY whenever the caller allowed any time:
       * both mean "no image right now", but NOT_READY is only correct for a zero
       * timeout -- and neither may be read as "the swapchain is stale", which is
       * what led to it being torn down under the compositor.
       */
      Result rc = nwindowDequeueBuffer(chain->window, &slot, &mf);
      if (R_FAILED(rc))
         return (info && info->timeout == 0) ? VK_NOT_READY : VK_TIMEOUT;
      if (trace) { printf("[wsi-zc] acquire: dequeue -> 0x%x slot=%d\n", (unsigned)rc, slot); fflush(stdout); }
      if (slot < 0 || (uint32_t)slot >= chain->base.image_count) {
         nwindowCancelBuffer(chain->window, slot, NULL);
         return VK_ERROR_OUT_OF_DATE_KHR;
      }
      nvMultiFenceWait(&mf, 1000000 /* 1s */);
      if (trace) { printf("[wsi-zc] acquire: fence waited, slot=%d ready\n", slot); fflush(stdout); }
      chain->images[slot].acquire_fence = mf;
      chain->images[slot].busy = true;
      *image_index = (uint32_t)slot;
      return VK_SUCCESS;
   }

   /* Fallback (non-zero-copy): round-robin a free render image. Same contract as
    * the zero-copy branch -- report "no image right now" and let the consumer
    * decide how long to wait. No sleeping in here; see the note above. */
   for (uint32_t n = 0; n < chain->base.image_count; n++) {
      uint32_t i = (chain->next + n) % chain->base.image_count;
      if (!chain->images[i].busy) {
         chain->images[i].busy = true;
         chain->next = (i + 1) % chain->base.image_count;
         *image_index = i;
         return VK_SUCCESS;
      }
   }
   return (info && info->timeout == 0) ? VK_NOT_READY : VK_TIMEOUT;
}

static VkResult
wsi_switch_swapchain_queue_present(struct wsi_swapchain *wsi_chain,
                                   uint32_t image_index,
                                   uint64_t present_id,
                                   const VkPresentRegionKHR *damage)
{
   struct wsi_switch_swapchain *chain = (struct wsi_switch_swapchain *)wsi_chain;
   struct wsi_switch_image *img = &chain->images[image_index];

   /* Profiling: tick stamps per present phase. armGetSystemTick is ~19.2MHz, cheap. */
   const u64 t_start = armGetSystemTick();
   u64 t_fence = t_start, t_flush = t_start, t_copy = t_start;

   /* Zero-copy present: the rendered block-linear image IS the nwindow buffer.
    * Wait the render fence (the common WSI present signalled fences[image_index]),
    * then hand the slot to the VI compositor. No CPU copy, no libnx swizzle — the
    * GPU L2 flush in our fence cmdlist makes the block-linear writes visible to
    * the display block directly. */
   if (chain->zero_copy && img->configured) {
      const struct wsi_device *wsi = chain->base.wsi;
      if (chain->base.fences[image_index] != VK_NULL_HANDLE)
         wsi->WaitForFences(chain->base.device, 1,
                            &chain->base.fences[image_index], VK_TRUE, ~0ull);
      t_fence = armGetSystemTick();             /* GPU render done                     */
      t_flush = t_fence;                        /* no dcache flush / no copy            */
      nwindowQueueBuffer(chain->window, (s32)image_index, NULL);
      t_copy = armGetSystemTick();              /* nwindowQueueBuffer (VI queue) done   */
   }
   /* CPU-copy fallback: the rendered (host-visible) image -> the next libnx
    * framebuffer buffer (dequeue), row by row (strides differ), then present
    * (queue). Active only when zero-copy registration failed. */
   else if (chain->fb_created && img->base.cpu_map != NULL) {
      /* The common WSI present submitted the image->cpu_map blit ASYNCHRONOUSLY
       * (signalling fences[image_index]); wait for it, then invalidate the CPU
       * cache — GM20B is NOT IO-coherent, so a HOST_COHERENT map reads stale lines
       * (this was the "constant pink" bug: we re-read frame 0). Then mirror into
       * the nwindow buffer and present. */
      const struct wsi_device *wsi = chain->base.wsi;
      u32 src_stride = img->base.row_pitches[0];
      u32 sz = src_stride * chain->extent.height;
      if (chain->base.fences[image_index] != VK_NULL_HANDLE)
         wsi->WaitForFences(chain->base.device, 1,
                            &chain->base.fences[image_index], VK_TRUE, ~0ull);
      t_fence = armGetSystemTick();            /* GPU render + WSI blit-to-cpu_map done */
      armDCacheFlush((void *)(uintptr_t)img->base.cpu_map, sz);
      t_flush = armGetSystemTick();            /* dcache flush of the whole image done */
      u32 dst_stride = 0;
      u8 *dst = (u8 *)framebufferBegin(&chain->fb, &dst_stride);
      const u8 *src = (const u8 *)img->base.cpu_map;
      u32 row_B = MIN2(dst_stride, src_stride);
      for (u32 y = 0; y < chain->extent.height; y++)
         memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * src_stride, row_B);
      framebufferEnd(&chain->fb);
      t_copy = armGetSystemTick();             /* memcpy + framebufferEnd (VI queue) done */
   }

   /* Accumulate per-phase ns; emit one averaged line every 60 presents (batched so
    * the sdmc log write does not throttle the frame rate — see heuristics #9/#21). */
   {
      static u64 acc_fence = 0, acc_flush = 0, acc_copy = 0, acc_total = 0;
      static u64 last_start = 0, acc_interval = 0;
      static u32 nframes = 0, nintervals = 0;
      acc_fence += armTicksToNs(t_fence - t_start);
      acc_flush += armTicksToNs(t_flush - t_fence);
      acc_copy  += armTicksToNs(t_copy  - t_flush);
      acc_total += armTicksToNs(t_copy  - t_start);
      if (last_start != 0) { acc_interval += armTicksToNs(t_start - last_start); nintervals++; }
      last_start = t_start;
      if (++nframes >= 60) {
         unsigned long long fps = (acc_interval != 0)
            ? (1000000000ull * (unsigned long long)nintervals) / (unsigned long long)acc_interval : 0;
         char buf[256];
         snprintf(buf, sizeof buf,
            "[wsi-prof] over %u presents (avg us): fence/GPU=%llu dcacheflush=%llu memcpy+queue=%llu "
            "present_total=%llu | frame_interval=%llu (~%llu fps)",
            nframes,
            (unsigned long long)(acc_fence / nframes / 1000),
            (unsigned long long)(acc_flush / nframes / 1000),
            (unsigned long long)(acc_copy  / nframes / 1000),
            (unsigned long long)(acc_total / nframes / 1000),
            nintervals ? (unsigned long long)(acc_interval / nintervals / 1000) : 0ull,
            fps);
         dusk_switch_log(buf);
         acc_fence = acc_flush = acc_copy = acc_total = acc_interval = 0;
         nframes = nintervals = 0;
      }
   }

   img->busy = false;
   return VK_SUCCESS;
}

static VkResult
wsi_switch_swapchain_release_images(struct wsi_swapchain *wsi_chain,
                                    uint32_t count, const uint32_t *indices)
{
   struct wsi_switch_swapchain *chain = (struct wsi_switch_swapchain *)wsi_chain;
   for (uint32_t i = 0; i < count; i++)
      chain->images[indices[i]].busy = false;
   return VK_SUCCESS;
}

static VkResult
wsi_switch_swapchain_destroy(struct wsi_swapchain *wsi_chain,
                             const VkAllocationCallbacks *pAllocator)
{
   struct wsi_switch_swapchain *chain = (struct wsi_switch_swapchain *)wsi_chain;
   if (chain->zero_copy)
      nwindowReleaseBuffers(chain->window);
   if (chain->fb_created)
      framebufferClose(&chain->fb);
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->images[i].base.image != VK_NULL_HANDLE)
         wsi_destroy_image(&chain->base, &chain->images[i].base);
   }
   wsi_swapchain_finish(&chain->base);
   vk_free(pAllocator, chain);
   return VK_SUCCESS;
}

static VkResult
wsi_switch_surface_create_swapchain(VkIcdSurfaceBase *icd_surface,
                                    VkDevice device,
                                    struct wsi_device *wsi_device,
                                    const VkSwapchainCreateInfoKHR *pCreateInfo,
                                    const VkAllocationCallbacks *pAllocator,
                                    struct wsi_swapchain **swapchain_out)
{
   VkIcdSurfaceVi *surface = (VkIcdSurfaceVi *)icd_surface;
   struct wsi_switch_swapchain *chain;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);

   uint32_t num_images = pCreateInfo->minImageCount;
   if (num_images > WSI_SWITCH_MAX_IMAGES)
      num_images = WSI_SWITCH_MAX_IMAGES;

   size_t size = sizeof(*chain);
   chain = vk_zalloc(pAllocator, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (chain == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* Renderable images via the common WSI buffer-blit path (host-visible blit
    * buffer per image => img->base.cpu_map is the rendered pixels we mirror). */
   struct wsi_cpu_image_params image_params = {
      .base.image_type = WSI_IMAGE_TYPE_CPU,
   };
   result = wsi_swapchain_init(wsi_device, &chain->base, device,
                               pCreateInfo, &image_params.base, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, chain);
      return result;
   }

   chain->base.destroy = wsi_switch_swapchain_destroy;
   chain->base.get_wsi_image = wsi_switch_swapchain_get_wsi_image;
   chain->base.acquire_next_image = wsi_switch_swapchain_acquire_next_image;
   chain->base.queue_present = wsi_switch_swapchain_queue_present;
   chain->base.release_images = wsi_switch_swapchain_release_images;
   chain->base.present_mode = VK_PRESENT_MODE_FIFO_KHR;
   chain->base.image_count = num_images;
   chain->window = (NWindow *)surface->window;
   chain->extent = pCreateInfo->imageExtent;
   chain->vk_format = pCreateInfo->imageFormat;

   for (uint32_t i = 0; i < num_images; i++) {
      result = wsi_create_image(&chain->base, &chain->base.image_info,
                                &chain->images[i].base);
      if (result != VK_SUCCESS)
         goto fail;
      chain->images[i].busy = false;
   }

   /* ZERO-COPY: register each rendered block-linear image directly as an nwindow
    * scanout buffer (build the NvGraphicBuffer with kind=0xfe + the image's NIL
    * layout, then nwindowConfigureBuffer). slot == image index (identity). If any
    * image can't be wrapped, fall back to the libnx-framebuffer CPU-copy path. */
   nvFenceInit();   /* required before nwindowDequeueBuffer / nvMultiFenceWait */
   nwindowSetDimensions(chain->window, chain->extent.width, chain->extent.height);
   chain->zero_copy = true;
   for (uint32_t i = 0; i < num_images; i++) {
      if (!wsi_switch_build_graphic_buffer(&chain->images[i].gb,
                                           chain->images[i].base.image,
                                           chain->vk_format, chain->extent)) {
         printf("[wsi-zc] img%u build_graphic_buffer FAILED\n", i); fflush(stdout);
         chain->zero_copy = false; break;
      }
      Result rc = nwindowConfigureBuffer(chain->window, (s32)i, &chain->images[i].gb);
      printf("[wsi-zc] img%u nwindowConfigureBuffer -> 0x%x\n", i, (unsigned)rc); fflush(stdout);
      if (R_FAILED(rc)) { chain->zero_copy = false; break; }
      chain->images[i].configured = true;
   }

   if (chain->zero_copy) {
      printf("[wsi-zc] zero-copy ENABLED (block-linear scanout, kind=0xfe)\n"); fflush(stdout);
      dusk_switch_log("[wsi] zero-copy ENABLED (block-linear nwindow scanout, kind=0xfe)\n");
   } else {
      printf("[wsi-zc] zero-copy FAILED -> CPU-copy fallback\n"); fflush(stdout);
      /* Fallback: reset the nwindow, then libnx framebuffer + CPU-copy present. */
      for (uint32_t i = 0; i < num_images; i++) chain->images[i].configured = false;
      nwindowReleaseBuffers(chain->window);
      u32 pixfmt = (chain->vk_format == VK_FORMAT_B8G8R8A8_UNORM)
                   ? PIXEL_FORMAT_BGRA_8888 : PIXEL_FORMAT_RGBA_8888;
      Result rc = framebufferCreate(&chain->fb, chain->window,
                                    chain->extent.width, chain->extent.height,
                                    pixfmt, num_images);
      if (R_FAILED(rc)) { result = VK_ERROR_INITIALIZATION_FAILED; goto fail; }
      framebufferMakeLinear(&chain->fb);
      chain->fb_created = true;
      dusk_switch_log("[wsi] zero-copy FAILED -> CPU-copy fallback\n");
   }

   *swapchain_out = &chain->base;
   return VK_SUCCESS;

fail:
   wsi_switch_swapchain_destroy(&chain->base, pAllocator);
   return result;
}

/* ---- registration ---- */

VkResult
wsi_switch_init_wsi(struct wsi_device *wsi_device,
                    const VkAllocationCallbacks *alloc,
                    VkPhysicalDevice physical_device)
{
   struct wsi_switch *wsi;
   VkResult result;

   wsi = vk_alloc(alloc, sizeof(*wsi), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!wsi) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }

   wsi->wsi = wsi_device;
   wsi->alloc = alloc;
   wsi->physical_device = physical_device;

   wsi->base.get_support = wsi_switch_surface_get_support;
   wsi->base.get_capabilities2 = wsi_switch_surface_get_capabilities2;
   wsi->base.get_formats = wsi_switch_surface_get_formats;
   wsi->base.get_formats2 = wsi_switch_surface_get_formats2;
   wsi->base.get_present_modes = wsi_switch_surface_get_present_modes;
   wsi->base.get_present_rectangles = wsi_switch_surface_get_present_rectangles;
   wsi->base.create_swapchain = wsi_switch_surface_create_swapchain;

   wsi_device->wsi[VK_ICD_WSI_PLATFORM_VI] = &wsi->base;
   return VK_SUCCESS;

fail:
   wsi_device->wsi[VK_ICD_WSI_PLATFORM_VI] = NULL;
   return result;
}

void
wsi_switch_finish_wsi(struct wsi_device *wsi_device,
                      const VkAllocationCallbacks *alloc)
{
   struct wsi_switch *wsi =
      (struct wsi_switch *)wsi_device->wsi[VK_ICD_WSI_PLATFORM_VI];
   if (wsi) {
      vk_free(alloc, wsi);
      wsi_device->wsi[VK_ICD_WSI_PLATFORM_VI] = NULL;
   }
}
