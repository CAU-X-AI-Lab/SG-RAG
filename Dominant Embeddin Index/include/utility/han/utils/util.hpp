#pragma once

#include <cstdio>
#include <iostream>
#include <cerrno>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <x86intrin.h>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <unistd.h>
#endif

// ... existing code ...

// 添加跨平台的内存对齐函数
inline void align_malloc(void **memptr, size_t alignment, size_t size) {
#ifdef _WIN32
    *memptr = _aligned_malloc(size, alignment);
#else
    posix_memalign(memptr, alignment, size);
#endif
}

#ifdef _WIN32
    const int CACHE_LINE_SIZE = 64;
#else
    const int CACHE_LINE_SIZE = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
#endif

// ... rest of the code ... 