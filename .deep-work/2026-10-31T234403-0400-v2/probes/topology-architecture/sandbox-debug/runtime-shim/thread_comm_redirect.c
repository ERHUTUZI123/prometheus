#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <string.h>

static int metadata_only(const char *path, int flags) {
    static const char prefix[] = "/proc/self/task/";
    const char *p;
    if (!path || (flags & O_ACCMODE) != O_WRONLY) return 0;
    if (strncmp(path, prefix, sizeof(prefix)-1) != 0) return 0;
    p = path + sizeof(prefix)-1;
    if (*p < '0' || *p > '9') return 0;
    while (*p >= '0' && *p <= '9') ++p;
    return strcmp(p, "/comm") == 0;
}

static mode_t get_mode(int flags, va_list ap) {
    if ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
        return (mode_t)va_arg(ap, int);
    return 0;
}

static int call_open(const char *symbol, const char *path, int flags, mode_t mode) {
    int saved = errno;
    int (*real)(const char *, int, ...) = dlsym(RTLD_NEXT, symbol);
    if (!real) { errno = ENOSYS; return -1; }
    errno = saved;
    if (metadata_only(path, flags))
        return real("/dev/null", O_WRONLY | (flags & (O_CLOEXEC|O_NONBLOCK)), 0);
    return real(path, flags, mode);
}

static int call_openat(const char *symbol, int dirfd, const char *path, int flags, mode_t mode) {
    int saved = errno;
    int (*real)(int, const char *, int, ...) = dlsym(RTLD_NEXT, symbol);
    if (!real) { errno = ENOSYS; return -1; }
    errno = saved;
    if (metadata_only(path, flags))
        return real(dirfd, "/dev/null", O_WRONLY | (flags & (O_CLOEXEC|O_NONBLOCK)), 0);
    return real(dirfd, path, flags, mode);
}

int open(const char *path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = get_mode(flags, ap); va_end(ap);
    return call_open("open", path, flags, mode);
}
int open64(const char *path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = get_mode(flags, ap); va_end(ap);
    return call_open("open64", path, flags, mode);
}
int openat(int dirfd, const char *path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = get_mode(flags, ap); va_end(ap);
    return call_openat("openat", dirfd, path, flags, mode);
}
int openat64(int dirfd, const char *path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = get_mode(flags, ap); va_end(ap);
    return call_openat("openat64", dirfd, path, flags, mode);
}
