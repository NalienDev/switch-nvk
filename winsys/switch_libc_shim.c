/*
 * switch_libc_shim.c — libc/std glue for the NVK-on-Switch link (M1 group C).
 *
 * Two jobs:
 *
 *  1. Route the DRM render node + BO mappings into the winsys shim:
 *       - open("/dev/dri/renderD128")  -> drm_shim_open()   (via -Wl,--wrap=open)
 *       - close(shim_fd)               -> drm_shim_close()  (via -Wl,--wrap=close)
 *       - mmap(fd, map_handle)         -> drm_shim_mmap()   (mmap is absent in
 *                                         newlib, so we just define it)
 *     Non-DRM paths/fds fall through to the real newlib open()/close().
 *
 *  2. Provide the POSIX/GNU symbols newlib (devkitA64) genuinely lacks but Mesa,
 *     the Vulkan runtime and Rust-std reference. Everything here is a benign,
 *     single-process-homebrew-correct implementation (no multi-user, no fork,
 *     no demand paging). Symbols newlib *does* provide (strrchr, strdup, …) are
 *     deliberately left to -lc.
 *
 * Link: add the NVK static libs + drm_shim.o + this + compat/compat.c, with
 *   -Wl,--wrap=open -Wl,--wrap=close
 */
#ifdef __SWITCH__

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <switch.h>

#include "drm_shim.h"
#include "sys/mman.h"   /* compat: MAP_FAILED, PROT_*                          */

/* ------------------------------------------------------------------------- */
/* 1. open/close/mmap/munmap routing.                                          */
/* ------------------------------------------------------------------------- */
extern int __real_open(const char *path, int flags, ...);
extern int __real_close(int fd);

int __wrap_open(const char *path, int flags, ...)
{
   if (drm_shim_is_render_node(path))
      return drm_shim_open(path, flags);

   /* Forward the optional mode_t for O_CREAT, like real open(2). */
   if (flags & O_CREAT) {
      va_list ap;
      va_start(ap, flags);
      int mode = va_arg(ap, int);
      va_end(ap);
      return __real_open(path, flags, mode);
   }
   return __real_open(path, flags);
}

int __wrap_close(int fd)
{
   if (drm_shim_owns_fd(fd))
      return drm_shim_close(fd);
   return __real_close(fd);
}

/* NVK stat()s the render node to derive its dev_t (render_dev/primary_dev for
 * DRM device identity). newlib stat() would fail on the nonexistent path, so
 * synthesize a char-device stat for the node (major 226, minor 128 = the Linux
 * DRM render-node numbering); everything else falls through. */
extern int __real_stat(const char *path, struct stat *st);
extern int __real_lstat(const char *path, struct stat *st);

static int shim_fake_node_stat(struct stat *st)
{
   memset(st, 0, sizeof(*st));
   st->st_mode = S_IFCHR | 0666;
   st->st_rdev = (226 << 8) | 128;   /* render128 */
   st->st_nlink = 1;
   return 0;
}

int __wrap_stat(const char *path, struct stat *st)
{
   if (drm_shim_is_render_node(path))
      return shim_fake_node_stat(st);
   return __real_stat(path, st);
}

int __wrap_lstat(const char *path, struct stat *st)
{
   if (drm_shim_is_render_node(path))
      return shim_fake_node_stat(st);
   return __real_lstat(path, st);
}

/* newlib has no mmap; NVK only uses it to map a BO (fd is a shim fd) or for the
 * disk cache (which we don't back). BO maps go to the shim; anything else
 * fails cleanly. */
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
   (void)addr; (void)prot;
   {
      char buf[176];
      snprintf(buf, sizeof(buf),
               "libc mmap fd=0x%x len=0x%zx prot=0x%x flags=0x%x off=0x%lx owned=%d\n",
               fd, length, prot, flags, (long)offset, (int)drm_shim_owns_fd(fd));
      drm_shim_dbg(buf);
   }
   if (drm_shim_owns_fd(fd)) {
      void *p = drm_shim_mmap(fd, offset, length);
      return p ? p : MAP_FAILED;
   }
   errno = ENOMEM;
   return MAP_FAILED;
}

int munmap(void *addr, size_t length)
{
   return drm_shim_munmap(addr, length);
}

/* ------------------------------------------------------------------------- */
/* 2. Missing POSIX/GNU libc.                                                  */
/* ------------------------------------------------------------------------- */

/* Aligned allocation (Mesa/anv/nvk). */
int posix_memalign(void **memptr, size_t alignment, size_t size)
{
   if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0)
      return EINVAL;
   void *p = memalign(alignment, size);
   if (!p)
      return ENOMEM;
   *memptr = p;
   return 0;
}

/* CSPRNG -> libnx randomGet (always succeeds). */
ssize_t getrandom(void *buf, size_t buflen, unsigned int flags)
{
   (void)flags;
   randomGet(buf, buflen);
   return (ssize_t)buflen;
}

/* Positional I/O: newlib lacks pread/pwrite. Emulate with lseek; not atomic vs
 * concurrent seeks, which is fine — Mesa uses these only on private cache fds. */
ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
   off_t cur = lseek(fd, 0, SEEK_CUR);
   if (cur < 0) return -1;
   if (lseek(fd, offset, SEEK_SET) < 0) return -1;
   ssize_t n = read(fd, buf, count);
   lseek(fd, cur, SEEK_SET);
   return n;
}

ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset)
{
   off_t cur = lseek(fd, 0, SEEK_CUR);
   if (cur < 0) return -1;
   if (lseek(fd, offset, SEEK_SET) < 0) return -1;
   ssize_t n = write(fd, buf, count);
   lseek(fd, cur, SEEK_SET);
   return n;
}

/* Advisory file locks — single process, so locking is a no-op success. */
int flock(int fd, int operation)
{
   (void)fd; (void)operation;
   return 0;
}

/* No signals worth masking in a homebrew (newlib declares this but ships no
 * impl); report success and leave the old mask untouched. */
int pthread_sigmask(int how, const sigset_t *set, sigset_t *oldset)
{
   (void)how; (void)set;
   if (oldset)
      *oldset = 0;
   return 0;
}

/* Runtime config queries. Return sane Tegra X1 values; -1/EINVAL otherwise. */
long sysconf(int name)
{
   switch (name) {
#ifdef _SC_PAGESIZE
   case _SC_PAGESIZE:          return 0x1000;
#endif
#ifdef _SC_PAGE_SIZE
#if !defined(_SC_PAGESIZE) || (_SC_PAGE_SIZE != _SC_PAGESIZE)
   case _SC_PAGE_SIZE:         return 0x1000;
#endif
#endif
#ifdef _SC_NPROCESSORS_ONLN
   case _SC_NPROCESSORS_ONLN:  return 3;  /* 3 cores available to applications */
#endif
#ifdef _SC_NPROCESSORS_CONF
   case _SC_NPROCESSORS_CONF:  return 4;  /* Tegra X1 has 4 Cortex-A57         */
#endif
#ifdef _SC_PHYS_PAGES
   case _SC_PHYS_PAGES:        return (4ull << 30) / 0x1000;
#endif
   default:
      errno = EINVAL;
      return -1;
   }
}

/* ---- POSIX regex (driconf app-profile matching) -------------------------- *
 * We don't ship driconf profiles, so compile "succeeds" and every match
 * misses — no profiles get applied. Avoids dragging in a regex engine. */
int regcomp(void *preg, const char *regex, int cflags)
{
   (void)preg; (void)regex; (void)cflags;
   return 0;
}
int regexec(const void *preg, const char *str, size_t nmatch,
            void *pmatch, int eflags)
{
   (void)preg; (void)str; (void)nmatch; (void)pmatch; (void)eflags;
   return 1;  /* REG_NOMATCH */
}
void regfree(void *preg)
{
   (void)preg;
}

/* ---- Filesystem odds and ends newlib/devkitA64 omit ---------------------- */

/* No pipes/fifos on Horizon. */
int pipe(int pipefd[2])      { (void)pipefd; errno = ENOSYS; return -1; }
int mkfifo(const char *path, mode_t mode) { (void)path; (void)mode; errno = ENOSYS; return -1; }

/* DIR carries no public fd on devkitA64; cache dir-scan just degrades. */
int dirfd(void *dirp)        { (void)dirp; errno = ENOTSUP; return -1; }

/* Only the AT_FDCWD + absolute-path case is exercised (disk cache). */
int fstatat(int dirfd_, const char *path, struct stat *st, int flags)
{
   (void)dirfd_;
   if (flags & AT_SYMLINK_NOFOLLOW)
      return lstat(path, st);
   return stat(path, st);
}

/* ---- User/credential layer (Rust-std unix + Mesa) ------------------------ *
 * Single-user homebrew: present as root, no passwd db. */
int getpwuid_r(uid_t uid, void *pwd, char *buf, size_t buflen, void **result)
{
   (void)uid; (void)pwd; (void)buf; (void)buflen;
   if (result) *result = NULL;
   return 0;  /* "not found" with no error, per getpwuid_r contract */
}

int chown(const char *path, uid_t o, gid_t g)  { (void)path; (void)o; (void)g; return 0; }
int fchown(int fd, uid_t o, gid_t g)            { (void)fd; (void)o; (void)g; return 0; }
int lchown(const char *path, uid_t o, gid_t g)  { (void)path; (void)o; (void)g; return 0; }
int chroot(const char *path)                    { (void)path; return 0; }

uid_t getuid(void)   { return 0; }
uid_t geteuid(void)  { return 0; }
gid_t getgid(void)   { return 0; }
gid_t getegid(void)  { return 0; }
pid_t getppid(void)  { return 1; }

#endif /* __SWITCH__ */
