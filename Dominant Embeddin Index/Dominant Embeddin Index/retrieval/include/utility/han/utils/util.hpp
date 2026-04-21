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

// 修改 error 函数声明
inline void error(const char* msg, bool exit_flag) {
    fprintf(stderr, "%s", msg);
    if (exit_flag) {
        exit(1);
    }
}


#define SIMD_STATE 4 
#define SIMD_MODE 1 
typedef int PackBase;
#ifdef SI64
typedef long long PackState;
#else
typedef int PackState;
#endif
const int PACK_WIDTH = sizeof(PackState) * 8;
const int PACK_SHIFT = __builtin_ctzll(PACK_WIDTH);
const int PACK_MASK = PACK_WIDTH - 1;




const size_t PACK_NODE_POOL_SIZE = 1024000000;

// 修改 CACHE_LINE_SIZE 的定义
#ifdef _WIN32
    const int CACHE_LINE_SIZE = 64;  // Windows 平台使用固定值
#else
    const int CACHE_LINE_SIZE = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
#endif  // 确保有这个结束标记
struct PackNode
{
    PackBase base;
    PackState state;

    PackNode() {};
    PackNode(PackBase _b, PackState _s): base(_b), state(_s) {};
};

struct UVertex
{
    int start, deg;
    UVertex(): start(-1), deg(0) {};
    UVertex(int _s, int _d): start(_s), deg(_d) {};
};

struct DVertex
{
    int out_start, out_deg;
    int in_start, in_deg;

    DVertex(): out_start(-1), out_deg(0), in_start(-1), in_deg(0) {};
};

typedef std::pair<int, int> Edge;
typedef std::vector<std::pair<int,int>> EdgeVector;

void quit();
std::string extract_filename(const std::string full_filename);
int arg_pos(char *str, int argc, char **argv);
inline void align_malloc(void **memptr, size_t alignment, size_t size) {
#ifdef _WIN32
    *memptr = _aligned_malloc(size, alignment);
#else
    // 在 UNIX/Linux 平台使用 posix_memalign
    if (posix_memalign(memptr, alignment, size) != 0) {
        *memptr = nullptr;  // 处理内存分配失败
    }
#endif
}
EdgeVector load_graph(const std::string path);
void save_graph(const std::string path, const EdgeVector& edge_vec);
std::vector<int> load_vertex_order(const std::string path);
void save_newid(const std::string path, std::vector<int> org2newid);
bool edge_idpair_cmp(const Edge& a, const Edge& b);

