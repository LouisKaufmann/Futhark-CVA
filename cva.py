import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, value) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and self.device.type == device_type:
               if type(value) == str:
                   sizes[size] = self.device.get_info(getattr(cl.device_info,value))
               else:
                   sizes[size] = value
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0
    self.max_local_memory = int(self.device.local_mem_size)
    self.free_list = {}

    self.global_failure = self.pool.allocate(np.int32().itemsize)
    cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)
    self.global_failure_args = self.pool.allocate(np.int32().itemsize *
                                                  (self.global_failure_args_max+1))
    self.failure_is_an_option = np.int32(0)

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            # Bespoke sizes have no limit or default.
            max_value = None
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()

def sync(self):
    failure = np.empty(1, dtype=np.int32)
    cl.enqueue_copy(self.queue, failure, self.global_failure, is_blocking=True)
    self.failure_is_an_option = np.int32(0)
    if failure[0] >= 0:
        failure_args = np.empty(self.global_failure_args_max+1, dtype=np.int32)
        cl.enqueue_copy(self.queue, failure_args, self.global_failure_args, is_blocking=True)
        raise Exception(self.failure_msgs[failure[0]].format(*failure_args))
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
// Start of atomics.h

inline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline float atomic_fadd_f32_global(volatile __global float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline float atomic_fadd_f32_local(volatile __local float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

inline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

// End of atomics.h

__kernel void replicate_2006(__global unsigned char *mem_2002,
                             int32_t num_elems_2003, float val_2004)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_2006;
    int32_t replicate_ltid_2007;
    int32_t replicate_gid_2008;
    
    replicate_gtid_2006 = get_global_id(0);
    replicate_ltid_2007 = get_local_id(0);
    replicate_gid_2008 = get_group_id(0);
    if (slt32(replicate_gtid_2006, num_elems_2003)) {
        ((__global float *) mem_2002)[replicate_gtid_2006] = val_2004;
    }
    
  error_0:
    return;
}
__kernel void replicate_2018(int32_t steps_1286, __global
                             unsigned char *xs_mem_1953, __global
                             unsigned char *mem_1958)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_2018;
    int32_t replicate_ltid_2019;
    int32_t replicate_gid_2020;
    
    replicate_gtid_2018 = get_global_id(0);
    replicate_ltid_2019 = get_local_id(0);
    replicate_gid_2020 = get_group_id(0);
    if (slt32(replicate_gtid_2018, steps_1286)) {
        ((__global float *) mem_1958)[squot32(replicate_gtid_2018, steps_1286) *
                                      steps_1286 + (replicate_gtid_2018 -
                                                    squot32(replicate_gtid_2018,
                                                            steps_1286) *
                                                    steps_1286)] = ((__global
                                                                     float *) xs_mem_1953)[replicate_gtid_2018 -
                                                                                           squot32(replicate_gtid_2018,
                                                                                                   steps_1286) *
                                                                                           steps_1286];
    }
    
  error_0:
    return;
}
__kernel void segmap_1808(__global int *global_failure, int32_t paths_1285,
                          int32_t steps_1286, __global unsigned char *mem_1930,
                          __global unsigned char *mem_1936)
{
    #define segmap_group_sizze_1814 (mainzisegmap_group_sizze_1813)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_1997;
    int32_t local_tid_1998;
    int32_t group_sizze_2001;
    int32_t wave_sizze_2000;
    int32_t group_tid_1999;
    
    global_tid_1997 = get_global_id(0);
    local_tid_1998 = get_local_id(0);
    group_sizze_2001 = get_local_size(0);
    wave_sizze_2000 = LOCKSTEP_WIDTH;
    group_tid_1999 = get_group_id(0);
    
    int32_t phys_tid_1808;
    
    phys_tid_1808 = global_tid_1997;
    
    int32_t gtid_1806;
    
    gtid_1806 = squot32(group_tid_1999 * segmap_group_sizze_1814 +
                        local_tid_1998, steps_1286);
    
    int32_t gtid_1807;
    
    gtid_1807 = group_tid_1999 * segmap_group_sizze_1814 + local_tid_1998 -
        squot32(group_tid_1999 * segmap_group_sizze_1814 + local_tid_1998,
                steps_1286) * steps_1286;
    if (slt32(gtid_1806, paths_1285) && slt32(gtid_1807, steps_1286)) {
        int32_t unsign_arg_1821 = ((__global int32_t *) mem_1930)[gtid_1806];
        int32_t x_1823 = lshr32(gtid_1807, 16);
        int32_t x_1824 = gtid_1807 ^ x_1823;
        int32_t x_1825 = mul32(73244475, x_1824);
        int32_t x_1826 = lshr32(x_1825, 16);
        int32_t x_1827 = x_1825 ^ x_1826;
        int32_t x_1828 = mul32(73244475, x_1827);
        int32_t x_1829 = lshr32(x_1828, 16);
        int32_t x_1830 = x_1828 ^ x_1829;
        int32_t unsign_arg_1831 = unsign_arg_1821 ^ x_1830;
        int32_t unsign_arg_1832 = mul32(48271, unsign_arg_1831);
        int32_t unsign_arg_1833 = umod32(unsign_arg_1832, 2147483647);
        int32_t unsign_arg_1834 = mul32(48271, unsign_arg_1833);
        int32_t unsign_arg_1835 = umod32(unsign_arg_1834, 2147483647);
        float res_1836 = uitofp_i32_f32(unsign_arg_1833);
        float res_1837 = res_1836 / 2.1474836e9F;
        float res_1838 = uitofp_i32_f32(unsign_arg_1835);
        float res_1839 = res_1838 / 2.1474836e9F;
        float res_1840;
        
        res_1840 = futrts_log32(res_1837);
        
        float res_1841 = -2.0F * res_1840;
        float res_1842;
        
        res_1842 = futrts_sqrt32(res_1841);
        
        float res_1843 = 6.2831855F * res_1839;
        float res_1844;
        
        res_1844 = futrts_cos32(res_1843);
        
        float res_1845 = res_1842 * res_1844;
        
        ((__global float *) mem_1936)[gtid_1806 * steps_1286 + gtid_1807] =
            res_1845;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_1814
}
__kernel void segmap_1847(__global int *global_failure, int32_t paths_1285,
                          __global unsigned char *mem_1930)
{
    #define segmap_group_sizze_1851 (mainzisegmap_group_sizze_1850)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_1992;
    int32_t local_tid_1993;
    int32_t group_sizze_1996;
    int32_t wave_sizze_1995;
    int32_t group_tid_1994;
    
    global_tid_1992 = get_global_id(0);
    local_tid_1993 = get_local_id(0);
    group_sizze_1996 = get_local_size(0);
    wave_sizze_1995 = LOCKSTEP_WIDTH;
    group_tid_1994 = get_group_id(0);
    
    int32_t phys_tid_1847;
    
    phys_tid_1847 = global_tid_1992;
    
    int32_t gtid_1846;
    
    gtid_1846 = group_tid_1994 * segmap_group_sizze_1851 + local_tid_1993;
    if (slt32(gtid_1846, paths_1285)) {
        int32_t x_1859 = lshr32(gtid_1846, 16);
        int32_t x_1860 = gtid_1846 ^ x_1859;
        int32_t x_1861 = mul32(73244475, x_1860);
        int32_t x_1862 = lshr32(x_1861, 16);
        int32_t x_1863 = x_1861 ^ x_1862;
        int32_t x_1864 = mul32(73244475, x_1863);
        int32_t x_1865 = lshr32(x_1864, 16);
        int32_t x_1866 = x_1864 ^ x_1865;
        int32_t unsign_arg_1867 = 777822902 ^ x_1866;
        int32_t unsign_arg_1868 = mul32(48271, unsign_arg_1867);
        int32_t unsign_arg_1869 = umod32(unsign_arg_1868, 2147483647);
        
        ((__global int32_t *) mem_1930)[gtid_1846] = unsign_arg_1869;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_1851
}
__kernel void segred_nonseg_1881(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int *global_failure_args, __local volatile
                                 int64_t *red_arr_mem_2037_backing_aligned_0,
                                 __local volatile
                                 int64_t *sync_arr_mem_2035_backing_aligned_1,
                                 int32_t steps_1286, float swap_term_1287,
                                 int32_t payments_1288, float notional_1289,
                                 float a_1290, float sims_per_year_1297,
                                 float swap_period_1315, int32_t sizze_1362,
                                 float x_1518, float x_1520, float y_1522,
                                 float y_1523, float x_1762,
                                 int32_t num_groups_1875,
                                 unsigned char loop_not_taken_1910, __global
                                 unsigned char *mcs_mem_1964, __global
                                 unsigned char *mem_1968, __global
                                 unsigned char *mem_1973, __global
                                 unsigned char *counter_mem_2025, __global
                                 unsigned char *group_res_arr_mem_2027,
                                 int32_t num_threads_2029)
{
    #define segred_group_sizze_1873 (mainzisegred_group_sizze_1872)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_2037_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_2037_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_2035_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_2035_backing_aligned_1;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_2030;
    int32_t local_tid_2031;
    int32_t group_sizze_2034;
    int32_t wave_sizze_2033;
    int32_t group_tid_2032;
    
    global_tid_2030 = get_global_id(0);
    local_tid_2031 = get_local_id(0);
    group_sizze_2034 = get_local_size(0);
    wave_sizze_2033 = LOCKSTEP_WIDTH;
    group_tid_2032 = get_group_id(0);
    
    int32_t phys_tid_1881;
    
    phys_tid_1881 = global_tid_2030;
    
    __local char *sync_arr_mem_2035;
    
    sync_arr_mem_2035 = (__local char *) sync_arr_mem_2035_backing_0;
    
    __local char *red_arr_mem_2037;
    
    red_arr_mem_2037 = (__local char *) red_arr_mem_2037_backing_1;
    
    int32_t dummy_1879;
    
    dummy_1879 = 0;
    
    int32_t gtid_1880;
    
    gtid_1880 = 0;
    
    float x_acc_2039;
    int32_t chunk_sizze_2040;
    
    chunk_sizze_2040 = smin32(squot32(steps_1286 + segred_group_sizze_1873 *
                                      num_groups_1875 - 1,
                                      segred_group_sizze_1873 *
                                      num_groups_1875), squot32(steps_1286 -
                                                                phys_tid_1881 +
                                                                num_threads_2029 -
                                                                1,
                                                                num_threads_2029));
    
    float x_1531;
    float x_1532;
    
    // neutral-initialise the accumulators
    {
        x_acc_2039 = 0.0F;
    }
    for (int32_t i_2044 = 0; i_2044 < chunk_sizze_2040; i_2044++) {
        gtid_1880 = phys_tid_1881 + num_threads_2029 * i_2044;
        // apply map function
        {
            int32_t convop_x_1889 = 1 + gtid_1880;
            float binop_x_1890 = sitofp_i32_f32(convop_x_1889);
            float index_primexp_1891 = binop_x_1890 / sims_per_year_1297;
            bool cond_1657 = index_primexp_1891 < swap_period_1315;
            float ceil_arg_1658 = index_primexp_1891 / swap_term_1287;
            float res_1659;
            
            res_1659 = futrts_ceil32(ceil_arg_1658);
            
            int32_t res_1660 = fptosi_f32_i32(res_1659);
            int32_t remaining_1661 = sub32(payments_1288, res_1660);
            int32_t distance_upwards_exclusive_1662 = sub32(remaining_1661, 1);
            int32_t distance_1663 = add32(1, distance_upwards_exclusive_1662);
            float nextpayment_1665 = swap_term_1287 * res_1659;
            bool bounds_invalid_upwards_1666 = slt32(remaining_1661, 1);
            int32_t j_m_i_1696 = sub32(-1, distance_upwards_exclusive_1662);
            bool j_lte_i_1704 = sle32(-1, distance_upwards_exclusive_1662);
            float binop_y_1711 = sitofp_i32_f32(distance_1663);
            bool valid_1667 = !bounds_invalid_upwards_1666;
            float y_1671 = nextpayment_1665 - index_primexp_1891;
            int32_t n_1697 = squot32(j_m_i_1696, -1);
            float binop_y_1712 = swap_term_1287 * binop_y_1711;
            bool loop_not_taken_1907 = !cond_1657;
            bool protect_assert_disj_1908 = valid_1667 || loop_not_taken_1907;
            bool protect_assert_disj_1911 = protect_assert_disj_1908 ||
                 loop_not_taken_1910;
            bool range_valid_c_1668;
            
            if (!protect_assert_disj_1911) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {
                    global_failure_args[0] = 1;
                    global_failure_args[1] = 2;
                    global_failure_args[2] = remaining_1661;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
            
            float negate_arg_1672 = a_1290 * y_1671;
            bool empty_slice_1698 = n_1697 == 0;
            int32_t m_1699 = sub32(n_1697, 1);
            bool y_1709 = slt32(0, n_1697);
            float index_primexp_1713 = nextpayment_1665 + binop_y_1712;
            bool dim_match_1669 = remaining_1661 == distance_1663;
            float exp_arg_1673 = 0.0F - negate_arg_1672;
            int32_t m_t_s_1700 = mul32(-1, m_1699);
            bool protect_assert_disj_1913 = y_1709 || loop_not_taken_1907;
            bool protect_assert_disj_1916 = loop_not_taken_1910 ||
                 protect_assert_disj_1913;
            bool index_certs_1710;
            
            if (!protect_assert_disj_1916) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 1) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = n_1697;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
            
            bool protect_assert_disj_1918 = dim_match_1669 ||
                 loop_not_taken_1907;
            bool protect_assert_disj_1921 = loop_not_taken_1910 ||
                 protect_assert_disj_1918;
            bool empty_or_match_cert_1670;
            
            if (!protect_assert_disj_1921) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 2) == -1) {
                    ;
                }
                local_failure = true;
                goto error_0;
            }
            
            float res_1674 = fpow32(2.7182817F, exp_arg_1673);
            int32_t i_p_m_t_s_1701 = add32(distance_upwards_exclusive_1662,
                                           m_t_s_1700);
            float x_1675 = 1.0F - res_1674;
            bool zzero_leq_i_p_m_t_s_1702 = sle32(0, i_p_m_t_s_1701);
            bool i_p_m_t_s_leq_w_1703 = sle32(i_p_m_t_s_1701, remaining_1661);
            float B_1676 = x_1675 / a_1290;
            bool y_1705 = zzero_leq_i_p_m_t_s_1702 && i_p_m_t_s_leq_w_1703;
            float x_1677 = B_1676 - nextpayment_1665;
            float y_1686 = fpow32(B_1676, 2.0F);
            bool y_1706 = j_lte_i_1704 && y_1705;
            float x_1678 = x_1677 + index_primexp_1891;
            float x_1687 = x_1520 * y_1686;
            bool ok_or_empty_1707 = empty_slice_1698 || y_1706;
            float x_1684 = y_1522 * x_1678;
            float A2_1689 = x_1687 / y_1523;
            bool protect_assert_disj_1923 = ok_or_empty_1707 ||
                 loop_not_taken_1907;
            bool protect_assert_disj_1926 = loop_not_taken_1910 ||
                 protect_assert_disj_1923;
            bool index_certs_1708;
            
            if (!protect_assert_disj_1926) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 3) == -1) {
                    global_failure_args[0] = -1;
                    global_failure_args[1] = remaining_1661;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
            
            float A1_1685 = x_1684 / x_1518;
            float y_1714 = index_primexp_1713 - index_primexp_1891;
            float exp_arg_1690 = A1_1685 - A2_1689;
            float negate_arg_1715 = a_1290 * y_1714;
            float res_1691 = fpow32(2.7182817F, exp_arg_1690);
            float exp_arg_1716 = 0.0F - negate_arg_1715;
            float res_1717 = fpow32(2.7182817F, exp_arg_1716);
            float x_1718 = 1.0F - res_1717;
            float B_1719 = x_1718 / a_1290;
            float x_1720 = B_1719 - index_primexp_1713;
            float y_1724 = fpow32(B_1719, 2.0F);
            float x_1721 = x_1720 + index_primexp_1891;
            float x_1725 = x_1520 * y_1724;
            float x_1722 = y_1522 * x_1721;
            float A2_1726 = x_1725 / y_1523;
            float A1_1723 = x_1722 / x_1518;
            float exp_arg_1727 = A1_1723 - A2_1726;
            float res_1728 = fpow32(2.7182817F, exp_arg_1727);
            float res_1651;
            float redout_1898 = 0.0F;
            
            for (int32_t i_1899 = 0; i_1899 < sizze_1362; i_1899++) {
                float res_1664;
                
                if (cond_1657) {
                    float x_1655 = ((__global float *) mcs_mem_1964)[i_1899 *
                                                                     steps_1286 +
                                                                     gtid_1880];
                    float negate_arg_1692 = x_1655 * B_1676;
                    float exp_arg_1693 = 0.0F - negate_arg_1692;
                    float res_1694 = fpow32(2.7182817F, exp_arg_1693);
                    float res_1695 = res_1691 * res_1694;
                    float negate_arg_1729 = x_1655 * B_1719;
                    float exp_arg_1730 = 0.0F - negate_arg_1729;
                    float res_1731 = fpow32(2.7182817F, exp_arg_1730);
                    float res_1732 = res_1728 * res_1731;
                    float res_1734;
                    float redout_1886 = 0.0F;
                    
                    for (int32_t i_1887 = 0; i_1887 < remaining_1661;
                         i_1887++) {
                        int32_t index_primexp_1893 = 1 + i_1887;
                        float res_1739 = sitofp_i32_f32(index_primexp_1893);
                        float res_1740 = swap_term_1287 * res_1739;
                        float res_1741 = nextpayment_1665 + res_1740;
                        float y_1742 = res_1741 - index_primexp_1891;
                        float negate_arg_1743 = a_1290 * y_1742;
                        float exp_arg_1744 = 0.0F - negate_arg_1743;
                        float res_1745 = fpow32(2.7182817F, exp_arg_1744);
                        float x_1746 = 1.0F - res_1745;
                        float B_1747 = x_1746 / a_1290;
                        float x_1748 = B_1747 - res_1741;
                        float x_1749 = x_1748 + index_primexp_1891;
                        float x_1750 = y_1522 * x_1749;
                        float A1_1751 = x_1750 / x_1518;
                        float y_1752 = fpow32(B_1747, 2.0F);
                        float x_1753 = x_1520 * y_1752;
                        float A2_1754 = x_1753 / y_1523;
                        float exp_arg_1755 = A1_1751 - A2_1754;
                        float res_1756 = fpow32(2.7182817F, exp_arg_1755);
                        float negate_arg_1757 = x_1655 * B_1747;
                        float exp_arg_1758 = 0.0F - negate_arg_1757;
                        float res_1759 = fpow32(2.7182817F, exp_arg_1758);
                        float res_1760 = res_1756 * res_1759;
                        float res_1737 = res_1760 + redout_1886;
                        float redout_tmp_2046 = res_1737;
                        
                        redout_1886 = redout_tmp_2046;
                    }
                    res_1734 = redout_1886;
                    
                    float x_1761 = res_1695 - res_1732;
                    float y_1763 = res_1734 * x_1762;
                    float y_1764 = x_1761 - y_1763;
                    float res_1765 = notional_1289 * y_1764;
                    
                    res_1664 = res_1765;
                } else {
                    res_1664 = 0.0F;
                }
                
                bool cond_1766 = res_1664 < 0.0F;
                float res_1767;
                
                if (cond_1766) {
                    res_1767 = 0.0F;
                } else {
                    res_1767 = res_1664;
                }
                
                float res_1654 = res_1767 + redout_1898;
                float redout_tmp_2045 = res_1654;
                
                redout_1898 = redout_tmp_2045;
            }
            res_1651 = redout_1898;
            
            float negate_arg_1769 = a_1290 * index_primexp_1891;
            float exp_arg_1770 = 0.0F - negate_arg_1769;
            float res_1771 = fpow32(2.7182817F, exp_arg_1770);
            float x_1772 = 1.0F - res_1771;
            float B_1773 = x_1772 / a_1290;
            float x_1774 = B_1773 - index_primexp_1891;
            float x_1775 = y_1522 * x_1774;
            float A1_1776 = x_1775 / x_1518;
            float y_1777 = fpow32(B_1773, 2.0F);
            float x_1778 = x_1520 * y_1777;
            float A2_1779 = x_1778 / y_1523;
            float exp_arg_1780 = A1_1776 - A2_1779;
            float res_1781 = fpow32(2.7182817F, exp_arg_1780);
            float negate_arg_1782 = 5.0e-2F * B_1773;
            float exp_arg_1783 = 0.0F - negate_arg_1782;
            float res_1784 = fpow32(2.7182817F, exp_arg_1783);
            float res_1785 = res_1781 * res_1784;
            float res_1786 = res_1651 * res_1785;
            
            // save map-out results
            {
                ((__global float *) mem_1973)[dummy_1879 * steps_1286 +
                                              gtid_1880] = res_1786;
            }
            // load accumulator
            {
                x_1531 = x_acc_2039;
            }
            // load new values
            {
                x_1532 = res_1786;
            }
            // apply reduction operator
            {
                float res_1533 = x_1531 + x_1532;
                
                // store in accumulator
                {
                    x_acc_2039 = res_1533;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_1531 = x_acc_2039;
        ((__local float *) red_arr_mem_2037)[local_tid_2031] = x_1531;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_2047;
    int32_t skip_waves_2048;
    float x_2041;
    float x_2042;
    
    offset_2047 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_2031, segred_group_sizze_1873)) {
            x_2041 = ((__local float *) red_arr_mem_2037)[local_tid_2031 +
                                                          offset_2047];
        }
    }
    offset_2047 = 1;
    while (slt32(offset_2047, wave_sizze_2033)) {
        if (slt32(local_tid_2031 + offset_2047, segred_group_sizze_1873) &&
            ((local_tid_2031 - squot32(local_tid_2031, wave_sizze_2033) *
              wave_sizze_2033) & (2 * offset_2047 - 1)) == 0) {
            // read array element
            {
                x_2042 = ((volatile __local
                           float *) red_arr_mem_2037)[local_tid_2031 +
                                                      offset_2047];
            }
            // apply reduction operation
            {
                float res_2043 = x_2041 + x_2042;
                
                x_2041 = res_2043;
            }
            // write result of operation
            {
                ((volatile __local float *) red_arr_mem_2037)[local_tid_2031] =
                    x_2041;
            }
        }
        offset_2047 *= 2;
    }
    skip_waves_2048 = 1;
    while (slt32(skip_waves_2048, squot32(segred_group_sizze_1873 +
                                          wave_sizze_2033 - 1,
                                          wave_sizze_2033))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_2047 = skip_waves_2048 * wave_sizze_2033;
        if (slt32(local_tid_2031 + offset_2047, segred_group_sizze_1873) &&
            ((local_tid_2031 - squot32(local_tid_2031, wave_sizze_2033) *
              wave_sizze_2033) == 0 && (squot32(local_tid_2031,
                                                wave_sizze_2033) & (2 *
                                                                    skip_waves_2048 -
                                                                    1)) == 0)) {
            // read array element
            {
                x_2042 = ((__local float *) red_arr_mem_2037)[local_tid_2031 +
                                                              offset_2047];
            }
            // apply reduction operation
            {
                float res_2043 = x_2041 + x_2042;
                
                x_2041 = res_2043;
            }
            // write result of operation
            {
                ((__local float *) red_arr_mem_2037)[local_tid_2031] = x_2041;
            }
        }
        skip_waves_2048 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_2031 == 0) {
            x_acc_2039 = x_2041;
        }
    }
    
    int32_t old_counter_2049;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_2031 == 0) {
            ((__global float *) group_res_arr_mem_2027)[group_tid_2032 *
                                                        segred_group_sizze_1873] =
                x_acc_2039;
            mem_fence_global();
            old_counter_2049 = atomic_add_i32_global(&((volatile __global
                                                        int *) counter_mem_2025)[0],
                                                     (int) 1);
            ((__local bool *) sync_arr_mem_2035)[0] = old_counter_2049 ==
                num_groups_1875 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_2050;
    
    is_last_group_2050 = ((__local bool *) sync_arr_mem_2035)[0];
    if (is_last_group_2050) {
        if (local_tid_2031 == 0) {
            old_counter_2049 = atomic_add_i32_global(&((volatile __global
                                                        int *) counter_mem_2025)[0],
                                                     (int) (0 -
                                                            num_groups_1875));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_2051 = squot32(num_groups_1875 +
                                                   segred_group_sizze_1873 - 1,
                                                   segred_group_sizze_1873);
            
            x_1531 = 0.0F;
            for (int32_t i_2052 = 0; i_2052 < read_per_thread_2051; i_2052++) {
                int32_t group_res_id_2053 = local_tid_2031 *
                        read_per_thread_2051 + i_2052;
                int32_t index_of_group_res_2054 = group_res_id_2053;
                
                if (slt32(group_res_id_2053, num_groups_1875)) {
                    x_1532 = ((__global
                               float *) group_res_arr_mem_2027)[index_of_group_res_2054 *
                                                                segred_group_sizze_1873];
                    
                    float res_1533;
                    
                    res_1533 = x_1531 + x_1532;
                    x_1531 = res_1533;
                }
            }
        }
        ((__local float *) red_arr_mem_2037)[local_tid_2031] = x_1531;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_2055;
            int32_t skip_waves_2056;
            float x_2041;
            float x_2042;
            
            offset_2055 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_2031, segred_group_sizze_1873)) {
                    x_2041 = ((__local
                               float *) red_arr_mem_2037)[local_tid_2031 +
                                                          offset_2055];
                }
            }
            offset_2055 = 1;
            while (slt32(offset_2055, wave_sizze_2033)) {
                if (slt32(local_tid_2031 + offset_2055,
                          segred_group_sizze_1873) && ((local_tid_2031 -
                                                        squot32(local_tid_2031,
                                                                wave_sizze_2033) *
                                                        wave_sizze_2033) & (2 *
                                                                            offset_2055 -
                                                                            1)) ==
                    0) {
                    // read array element
                    {
                        x_2042 = ((volatile __local
                                   float *) red_arr_mem_2037)[local_tid_2031 +
                                                              offset_2055];
                    }
                    // apply reduction operation
                    {
                        float res_2043 = x_2041 + x_2042;
                        
                        x_2041 = res_2043;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_2037)[local_tid_2031] = x_2041;
                    }
                }
                offset_2055 *= 2;
            }
            skip_waves_2056 = 1;
            while (slt32(skip_waves_2056, squot32(segred_group_sizze_1873 +
                                                  wave_sizze_2033 - 1,
                                                  wave_sizze_2033))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_2055 = skip_waves_2056 * wave_sizze_2033;
                if (slt32(local_tid_2031 + offset_2055,
                          segred_group_sizze_1873) && ((local_tid_2031 -
                                                        squot32(local_tid_2031,
                                                                wave_sizze_2033) *
                                                        wave_sizze_2033) == 0 &&
                                                       (squot32(local_tid_2031,
                                                                wave_sizze_2033) &
                                                        (2 * skip_waves_2056 -
                                                         1)) == 0)) {
                    // read array element
                    {
                        x_2042 = ((__local
                                   float *) red_arr_mem_2037)[local_tid_2031 +
                                                              offset_2055];
                    }
                    // apply reduction operation
                    {
                        float res_2043 = x_2041 + x_2042;
                        
                        x_2041 = res_2043;
                    }
                    // write result of operation
                    {
                        ((__local float *) red_arr_mem_2037)[local_tid_2031] =
                            x_2041;
                    }
                }
                skip_waves_2056 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_2031 == 0) {
                    ((__global float *) mem_1968)[0] = x_2041;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_1873
}
"""
# Start of values.py.

# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    dims = []
    for i in range(rank):
        parse_specific_string(f, '[')
        dims += [int(parse_int(f))]
        parse_specific_string(f, ']')
    if np.product(dims) != 0:
        raise ValueError
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return tuple(dims)

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if type(elems) == tuple:
        # Empty array
        return np.empty(elems, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype=FUTHARK_PRIMTYPES[bin_type_enum]['numpy_type'])
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def end_of_input(entry, f=input_reader):
    skip_spaces(f)
    if f.get_char() != b'':
        panic(1, "Expected EOF on stdin after reading input for \"%s\".", entry)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[{}]'.format(d)
                                                    for d in v.shape]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

# End of values.py.
# Start of memory.py.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, shape):
  # HACK: np.ctypeslib.as_array may fail if the shape contains zeroes,
  # for some reason.
  if any(map(lambda x: x == 0, shape)):
      return np.ndarray(shape, dtype=x._type_)
  else:
      return np.ctypeslib.as_array(x, shape=shape)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset*ct.sizeof(bt), bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset)*ct.sizeof(v), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)

# End of memory.py.
# Start of panic.py.

def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.stderr.write('\n')
    sys.exit(exitcode)

# End of panic.py.
# Start of tuning.py

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

# End of tuning.py.
# Start of scalar.py.

import numpy as np
import math
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def clz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if x < 0:
      break
    n += 1
    x <<= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_mul_hi8(a, b):
  a = np.uint64(np.uint8(a))
  b = np.uint64(np.uint8(b))
  return np.int8((a*b) >> np.uint64(8))

def futhark_mul_hi16(a, b):
  a = np.uint64(np.uint16(a))
  b = np.uint64(np.uint16(b))
  return np.int16((a*b) >> np.uint64(16))

def futhark_mul_hi32(a, b):
  a = np.uint64(np.uint32(a))
  b = np.uint64(np.uint32(b))
  return np.int32((a*b) >> np.uint64(32))

# This one is done with arbitrary-precision integers.
def futhark_mul_hi64(a, b):
  a = int(np.uint64(a))
  b = int(np.uint64(b))
  return np.int64(np.uint64(a*b >> 64))

def futhark_mad_hi8(a, b, c):
  return futhark_mul_hi8(a,b) + c

def futhark_mad_hi16(a, b, c):
  return futhark_mul_hi16(a,b) + c

def futhark_mad_hi32(a, b, c):
  return futhark_mul_hi32(a,b) + c

def futhark_mad_hi64(a, b, c):
  return futhark_mul_hi64(a,b) + c

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_cosh64(x):
  return np.cosh(x)

def futhark_sinh64(x):
  return np.sinh(x)

def futhark_tanh64(x):
  return np.tanh(x)

def futhark_acosh64(x):
  return np.arccosh(x)

def futhark_asinh64(x):
  return np.arcsinh(x)

def futhark_atanh64(x):
  return np.arctanh(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_gamma64(x):
  return np.float64(math.gamma(x))

def futhark_lgamma64(x):
  return np.float64(math.lgamma(x))

def futhark_round64(x):
  return np.round(x)

def futhark_ceil64(x):
  return np.ceil(x)

def futhark_floor64(x):
  return np.floor(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_cosh32(x):
  return np.cosh(x)

def futhark_sinh32(x):
  return np.sinh(x)

def futhark_tanh32(x):
  return np.tanh(x)

def futhark_acosh32(x):
  return np.arccosh(x)

def futhark_asinh32(x):
  return np.arcsinh(x)

def futhark_atanh32(x):
  return np.arctanh(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_gamma32(x):
  return np.float32(math.gamma(x))

def futhark_lgamma32(x):
  return np.float32(math.lgamma(x))

def futhark_round32(x):
  return np.round(x)

def futhark_ceil32(x):
  return np.ceil(x)

def futhark_floor32(x):
  return np.floor(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])

def futhark_lerp32(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_lerp64(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_mad32(a, b, c):
  return a * b + c

def futhark_mad64(a, b, c):
  return a * b + c

def futhark_fma32(a, b, c):
  return a * b + c

def futhark_fma64(a, b, c):
  return a * b + c

# End of scalar.py.
class cva:
  entry_points = {"main": (["i32", "i32", "f32", "i32", "f32", "f32", "f32",
                            "f32", "f32"], ["[]f32", "f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width", 32),
     ("AMD Accelerated Parallel Processing", cl.device_type.GPU, "lockstep_width",
      32), ("", cl.device_type.GPU, "lockstep_width", 1), ("", cl.device_type.GPU,
                                                           "num_groups", 256), ("",
                                                                                cl.device_type.GPU,
                                                                                "group_size",
                                                                                256),
     ("", cl.device_type.GPU, "tile_size", 32), ("", cl.device_type.GPU,
                                                 "threshold", 32768), ("",
                                                                       cl.device_type.CPU,
                                                                       "lockstep_width",
                                                                       1), ("",
                                                                            cl.device_type.CPU,
                                                                            "num_groups",
                                                                            "MAX_COMPUTE_UNITS"),
     ("", cl.device_type.CPU, "group_size", 32), ("", cl.device_type.CPU,
                                                  "tile_size", 4), ("",
                                                                    cl.device_type.CPU,
                                                                    "threshold",
                                                                    "MAX_COMPUTE_UNITS")]
    self.global_failure_args_max = 3
    self.failure_msgs=["Range {}..{}...{} is invalid.\n-> #0  cva.fut:42:30-45\n   #1  cva.fut:50:27-71\n   #2  cva.fut:97:64-89\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:97:26-106\n   #6  cva.fut:95:21-98:31\n   #7  cva.fut:77:1-102:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:52:38-63\n   #1  cva.fut:97:64-89\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:97:26-106\n   #5  cva.fut:95:21-98:31\n   #6  cva.fut:77:1-102:18\n",
     "value cannot match pattern\n-> #0  cva.fut:42:5-44:41\n   #1  cva.fut:50:27-71\n   #2  cva.fut:97:64-89\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:97:26-106\n   #6  cva.fut:95:21-98:31\n   #7  cva.fut:77:1-102:18\n",
     "Index [::{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:52:39-59\n   #1  cva.fut:97:64-89\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:97:26-106\n   #5  cva.fut:95:21-98:31\n   #6  cva.fut:77:1-102:18\n"]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i32", "f32", "bool", "cert"],
                                       user_sizes=sizes,
                                       all_sizes={"main.group_size_2009": {"class": "group_size", "value": None},
                                        "main.group_size_2021": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_1813": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_1850": {"class": "group_size", "value": None},
                                        "main.segred_group_size_1872": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_1874": {"class": "num_groups", "value": None}})
    self.replicate_2006_var = program.replicate_2006
    self.replicate_2018_var = program.replicate_2018
    self.segmap_1808_var = program.segmap_1808
    self.segmap_1847_var = program.segmap_1847
    self.segred_nonseg_1881_var = program.segred_nonseg_1881
    self.constants = {}
    counter_mem_2025 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                 np.int32(0), np.int32(0), np.int32(0),
                                 np.int32(0), np.int32(0), np.int32(0),
                                 np.int32(0)], dtype=np.int32)
    static_mem_2063 = opencl_alloc(self, 40, "static_mem_2063")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_2063,
                      normaliseArray(counter_mem_2025), is_blocking=synchronous)
    self.counter_mem_2025 = static_mem_2063
  def futhark_main(self, paths_1285, steps_1286, swap_term_1287, payments_1288,
                   notional_1289, a_1290, b_1291, sigma_1292, r0_1293):
    res_1294 = sitofp_i32_f32(payments_1288)
    duration_1295 = (swap_term_1287 * res_1294)
    res_1296 = sitofp_i32_f32(steps_1286)
    sims_per_year_1297 = (res_1296 / duration_1295)
    bounds_invalid_upwards_1298 = slt32(steps_1286, np.int32(1))
    distance_upwards_exclusive_1299 = (steps_1286 - np.int32(1))
    valid_1301 = not(bounds_invalid_upwards_1298)
    range_valid_c_1302 = True
    assert valid_1301, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:58:56-67\n   #1  cva.fut:82:17-40\n   #2  cva.fut:77:1-102:18\n" % ("Range ",
                                                                                                                                                 np.int32(1),
                                                                                                                                                 "..",
                                                                                                                                                 np.int32(2),
                                                                                                                                                 "...",
                                                                                                                                                 steps_1286,
                                                                                                                                                 " is invalid."))
    steps_1792 = sext_i32_i64(steps_1286)
    bounds_invalid_upwards_1310 = slt32(paths_1285, np.int32(0))
    valid_1311 = not(bounds_invalid_upwards_1310)
    range_valid_c_1312 = True
    assert valid_1311, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/math.fut:453:23-30\n   #1  /prelude/array.fut:60:3-12\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:42-47\n   #3  cva.fut:83:19-49\n   #4  cva.fut:77:1-102:18\n" % ("Range ",
                                                                                                                                                                                                                                                             np.int32(0),
                                                                                                                                                                                                                                                             "..",
                                                                                                                                                                                                                                                             np.int32(1),
                                                                                                                                                                                                                                                             "..<",
                                                                                                                                                                                                                                                             paths_1285,
                                                                                                                                                                                                                                                             " is invalid."))
    y_1314 = (res_1294 - np.float32(1.0))
    swap_period_1315 = (swap_term_1287 * y_1314)
    delta_t_1316 = (duration_1295 / res_1296)
    paths_1848 = sext_i32_i64(paths_1285)
    segmap_group_sizze_1851 = self.sizes["main.segmap_group_size_1850"]
    segmap_group_sizze_1852 = sext_i32_i64(segmap_group_sizze_1851)
    y_1853 = (segmap_group_sizze_1852 - np.int64(1))
    x_1854 = (paths_1848 + y_1853)
    segmap_usable_groups_64_1856 = squot64(x_1854, segmap_group_sizze_1852)
    segmap_usable_groups_1857 = sext_i64_i32(segmap_usable_groups_64_1856)
    bytes_1928 = (np.int64(4) * paths_1848)
    mem_1930 = opencl_alloc(self, bytes_1928, "mem_1930")
    if ((1 * (np.long(segmap_usable_groups_1857) * np.long(segmap_group_sizze_1851))) != 0):
      self.segmap_1847_var.set_args(self.global_failure, np.int32(paths_1285),
                                    mem_1930)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_1847_var,
                                 ((np.long(segmap_usable_groups_1857) * np.long(segmap_group_sizze_1851)),),
                                 (np.long(segmap_group_sizze_1851),))
      if synchronous:
        sync(self)
    nest_sizze_1812 = (steps_1792 * paths_1848)
    segmap_group_sizze_1814 = self.sizes["main.segmap_group_size_1813"]
    segmap_group_sizze_1815 = sext_i32_i64(segmap_group_sizze_1814)
    y_1816 = (segmap_group_sizze_1815 - np.int64(1))
    x_1817 = (nest_sizze_1812 + y_1816)
    segmap_usable_groups_64_1819 = squot64(x_1817, segmap_group_sizze_1815)
    segmap_usable_groups_1820 = sext_i64_i32(segmap_usable_groups_64_1819)
    binop_x_1935 = (steps_1792 * paths_1848)
    bytes_1932 = (np.int64(4) * binop_x_1935)
    mem_1936 = opencl_alloc(self, bytes_1932, "mem_1936")
    if ((1 * (np.long(segmap_usable_groups_1820) * np.long(segmap_group_sizze_1814))) != 0):
      self.segmap_1808_var.set_args(self.global_failure, np.int32(paths_1285),
                                    np.int32(steps_1286), mem_1930, mem_1936)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_1808_var,
                                 ((np.long(segmap_usable_groups_1820) * np.long(segmap_group_sizze_1814)),),
                                 (np.long(segmap_group_sizze_1814),))
      if synchronous:
        sync(self)
    mem_1930 = None
    mem_1941 = opencl_alloc(self, np.int64(0), "mem_1941")
    mem_1944 = opencl_alloc(self, np.int64(4), "mem_1944")
    self.futhark_builtinzhreplicate_f32(mem_1944, np.int32(1),
                                        np.float32(5.000000074505806e-2))
    res_1361 = futhark_sqrt32(delta_t_1316)
    mem_1949 = opencl_alloc(self, np.int64(4), "mem_1949")
    bytes_1954 = (np.int64(4) * steps_1792)
    mem_1958 = opencl_alloc(self, bytes_1954, "mem_1958")
    loopz2082Uz2082U_1364 = np.int32(0)
    xs_mem_1945 = mem_1941
    i_1366 = np.int32(0)
    one_2062 = np.int32(1)
    for counter_2061 in range(paths_1285):
      loopz2082U_1373 = np.int32(1)
      x_mem_1946 = mem_1944
      i_1375 = np.int32(0)
      one_2060 = np.int32(1)
      for counter_2059 in range(distance_upwards_exclusive_1299):
        read_res_2057 = np.empty(1, dtype=ct.c_float)
        cl.enqueue_copy(self.queue, read_res_2057, mem_1936,
                        device_offset=(np.long(((i_1366 * steps_1286) + i_1375)) * 4),
                        is_blocking=synchronous)
        sync(self)
        shortstep_arg_1380 = read_res_2057[0]
        y_1381 = slt32(i_1375, loopz2082U_1373)
        index_certs_1383 = True
        assert y_1381, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:67:69-72\n   #1  cva.fut:94:21-66\n   #2  cva.fut:77:1-102:18\n" % ("Index [",
                                                                                                                                             i_1375,
                                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                                             loopz2082U_1373,
                                                                                                                                             "]."))
        read_res_2058 = np.empty(1, dtype=ct.c_float)
        cl.enqueue_copy(self.queue, read_res_2058, x_mem_1946,
                        device_offset=(np.long(i_1375) * 4),
                        is_blocking=synchronous)
        sync(self)
        shortstep_arg_1384 = read_res_2058[0]
        y_1385 = (b_1291 - shortstep_arg_1384)
        x_1386 = (a_1290 * y_1385)
        x_1387 = (delta_t_1316 * x_1386)
        x_1388 = (res_1361 * shortstep_arg_1380)
        y_1389 = (sigma_1292 * x_1388)
        delta_r_1390 = (x_1387 + y_1389)
        res_1391 = (shortstep_arg_1384 + delta_r_1390)
        self.futhark_builtinzhreplicate_f32(mem_1949, np.int32(1), res_1391)
        conc_tmp_1393 = (np.int32(1) + loopz2082U_1373)
        binop_x_1951 = sext_i32_i64(conc_tmp_1393)
        bytes_1950 = (np.int64(4) * binop_x_1951)
        mem_1952 = opencl_alloc(self, bytes_1950, "mem_1952")
        tmp_offs_2017 = np.int32(0)
        if ((sext_i32_i64(loopz2082U_1373) * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, mem_1952, x_mem_1946,
                          dest_offset=np.long((tmp_offs_2017 * np.int32(4))),
                          src_offset=np.long(np.int32(0)),
                          byte_count=np.long((sext_i32_i64(loopz2082U_1373) * np.int32(4))))
        if synchronous:
          sync(self)
        tmp_offs_2017 = (tmp_offs_2017 + loopz2082U_1373)
        if ((sext_i32_i64(np.int32(1)) * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, mem_1952, mem_1949,
                          dest_offset=np.long((tmp_offs_2017 * np.int32(4))),
                          src_offset=np.long(np.int32(0)),
                          byte_count=np.long((sext_i32_i64(np.int32(1)) * np.int32(4))))
        if synchronous:
          sync(self)
        tmp_offs_2017 = (tmp_offs_2017 + np.int32(1))
        loopz2082U_tmp_2014 = conc_tmp_1393
        x_mem_tmp_2015 = mem_1952
        loopz2082U_1373 = loopz2082U_tmp_2014
        x_mem_1946 = x_mem_tmp_2015
        i_1375 += one_2060
      sizze_1371 = loopz2082U_1373
      xs_mem_1953 = x_mem_1946
      dim_match_1395 = (steps_1286 == sizze_1371)
      empty_or_match_cert_1396 = True
      assert dim_match_1395, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:68:8-24\n   #1  cva.fut:94:21-66\n   #2  cva.fut:77:1-102:18\n" % ("Value of (core language) shape (",
                                                                                                                                                  sizze_1371,
                                                                                                                                                  ") cannot match shape of type `[",
                                                                                                                                                  steps_1286,
                                                                                                                                                  "]f32`."))
      group_sizze_2021 = self.sizes["main.group_size_2021"]
      num_groups_2022 = squot32(((steps_1286 + sext_i32_i32(group_sizze_2021)) - np.int32(1)),
                                sext_i32_i32(group_sizze_2021))
      if ((1 * (np.long(num_groups_2022) * np.long(group_sizze_2021))) != 0):
        self.replicate_2018_var.set_args(np.int32(steps_1286), xs_mem_1953,
                                         mem_1958)
        cl.enqueue_nd_range_kernel(self.queue, self.replicate_2018_var,
                                   ((np.long(num_groups_2022) * np.long(group_sizze_2021)),),
                                   (np.long(group_sizze_2021),))
        if synchronous:
          sync(self)
      xs_mem_1953 = None
      conc_tmp_1399 = (np.int32(1) + loopz2082Uz2082U_1364)
      binop_x_1960 = sext_i32_i64(conc_tmp_1399)
      binop_x_1962 = (steps_1792 * binop_x_1960)
      bytes_1959 = (np.int64(4) * binop_x_1962)
      mem_1963 = opencl_alloc(self, bytes_1959, "mem_1963")
      tmp_offs_2023 = np.int32(0)
      if ((sext_i32_i64((loopz2082Uz2082U_1364 * steps_1286)) * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_1963, xs_mem_1945,
                        dest_offset=np.long(((steps_1286 * tmp_offs_2023) * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sext_i32_i64((loopz2082Uz2082U_1364 * steps_1286)) * np.int32(4))))
      if synchronous:
        sync(self)
      tmp_offs_2023 = (tmp_offs_2023 + loopz2082Uz2082U_1364)
      if ((sext_i32_i64(steps_1286) * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_1963, mem_1958,
                        dest_offset=np.long(((steps_1286 * tmp_offs_2023) * np.int32(4))),
                        src_offset=np.long(np.int32(0)),
                        byte_count=np.long((sext_i32_i64(steps_1286) * np.int32(4))))
      if synchronous:
        sync(self)
      tmp_offs_2023 = (tmp_offs_2023 + np.int32(1))
      loopz2082Uz2082U_tmp_2011 = conc_tmp_1399
      xs_mem_tmp_2012 = mem_1963
      loopz2082Uz2082U_1364 = loopz2082Uz2082U_tmp_2011
      xs_mem_1945 = xs_mem_tmp_2012
      i_1366 += one_2062
    sizze_1362 = loopz2082Uz2082U_1364
    mcs_mem_1964 = xs_mem_1945
    mem_1936 = None
    mem_1941 = None
    mem_1944 = None
    mem_1949 = None
    mem_1958 = None
    x_1518 = fpow32(a_1290, np.float32(2.0))
    x_1519 = (b_1291 * x_1518)
    x_1520 = fpow32(sigma_1292, np.float32(2.0))
    y_1521 = (x_1520 / np.float32(2.0))
    y_1522 = (x_1519 - y_1521)
    y_1523 = (np.float32(4.0) * a_1290)
    segred_group_sizze_1873 = self.sizes["main.segred_group_size_1872"]
    max_num_groups_2024 = self.sizes["main.segred_num_groups_1874"]
    num_groups_1875 = sext_i64_i32(smax64(np.int32(1),
                                          smin64(squot64(((steps_1792 + sext_i32_i64(segred_group_sizze_1873)) - np.int64(1)),
                                                         sext_i32_i64(segred_group_sizze_1873)),
                                                 sext_i32_i64(max_num_groups_2024))))
    x_1762 = (np.float32(5.06298653781414e-2) * swap_term_1287)
    loop_nonempty_1909 = slt32(np.int32(0), sizze_1362)
    loop_not_taken_1910 = not(loop_nonempty_1909)
    mem_1968 = opencl_alloc(self, np.int64(4), "mem_1968")
    mem_1973 = opencl_alloc(self, bytes_1954, "mem_1973")
    counter_mem_2025 = self.counter_mem_2025
    group_res_arr_mem_2027 = opencl_alloc(self,
                                          (np.int32(4) * (segred_group_sizze_1873 * num_groups_1875)),
                                          "group_res_arr_mem_2027")
    num_threads_2029 = (num_groups_1875 * segred_group_sizze_1873)
    if ((1 * (np.long(num_groups_1875) * np.long(segred_group_sizze_1873))) != 0):
      self.segred_nonseg_1881_var.set_args(self.global_failure,
                                           self.failure_is_an_option,
                                           self.global_failure_args,
                                           cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_1873))),
                                           cl.LocalMemory(np.long(np.int32(1))),
                                           np.int32(steps_1286),
                                           np.float32(swap_term_1287),
                                           np.int32(payments_1288),
                                           np.float32(notional_1289),
                                           np.float32(a_1290),
                                           np.float32(sims_per_year_1297),
                                           np.float32(swap_period_1315),
                                           np.int32(sizze_1362),
                                           np.float32(x_1518),
                                           np.float32(x_1520),
                                           np.float32(y_1522),
                                           np.float32(y_1523),
                                           np.float32(x_1762),
                                           np.int32(num_groups_1875),
                                           np.byte(loop_not_taken_1910),
                                           mcs_mem_1964, mem_1968, mem_1973,
                                           counter_mem_2025,
                                           group_res_arr_mem_2027,
                                           np.int32(num_threads_2029))
      cl.enqueue_nd_range_kernel(self.queue, self.segred_nonseg_1881_var,
                                 ((np.long(num_groups_1875) * np.long(segred_group_sizze_1873)),),
                                 (np.long(segred_group_sizze_1873),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mcs_mem_1964 = None
    read_res_2064 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_2064, mem_1968,
                    device_offset=(np.long(np.int32(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_1529 = read_res_2064[0]
    mem_1968 = None
    CVA_1789 = (np.float32(6.000000052154064e-3) * res_1529)
    mem_1976 = opencl_alloc(self, bytes_1954, "mem_1976")
    if ((sext_i32_i64(steps_1286) * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_1976, mem_1973,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long((sext_i32_i64(steps_1286) * np.int32(4))))
    if synchronous:
      sync(self)
    mem_1973 = None
    out_arrsizze_1990 = steps_1286
    out_mem_1989 = mem_1976
    scalar_out_1991 = CVA_1789
    return (out_mem_1989, out_arrsizze_1990, scalar_out_1991)
  def futhark_builtinzhreplicate_f32(self, mem_2002, num_elems_2003, val_2004):
    group_sizze_2009 = self.sizes["main.group_size_2009"]
    num_groups_2010 = squot32(((num_elems_2003 + sext_i32_i32(group_sizze_2009)) - np.int32(1)),
                              sext_i32_i32(group_sizze_2009))
    if ((1 * (np.long(num_groups_2010) * np.long(group_sizze_2009))) != 0):
      self.replicate_2006_var.set_args(mem_2002, np.int32(num_elems_2003),
                                       np.float32(val_2004))
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_2006_var,
                                 ((np.long(num_groups_2010) * np.long(group_sizze_2009)),),
                                 (np.long(group_sizze_2009),))
      if synchronous:
        sync(self)
    return ()
  def main(self, paths_1285_ext, steps_1286_ext, swap_term_1287_ext,
           payments_1288_ext, notional_1289_ext, a_1290_ext, b_1291_ext,
           sigma_1292_ext, r0_1293_ext):
    try:
      paths_1285 = np.int32(ct.c_int32(paths_1285_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(paths_1285_ext),
                                                                                                                            paths_1285_ext))
    try:
      steps_1286 = np.int32(ct.c_int32(steps_1286_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(steps_1286_ext),
                                                                                                                            steps_1286_ext))
    try:
      swap_term_1287 = np.float32(ct.c_float(swap_term_1287_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(swap_term_1287_ext),
                                                                                                                            swap_term_1287_ext))
    try:
      payments_1288 = np.int32(ct.c_int32(payments_1288_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(payments_1288_ext),
                                                                                                                            payments_1288_ext))
    try:
      notional_1289 = np.float32(ct.c_float(notional_1289_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(notional_1289_ext),
                                                                                                                            notional_1289_ext))
    try:
      a_1290 = np.float32(ct.c_float(a_1290_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_1290_ext),
                                                                                                                            a_1290_ext))
    try:
      b_1291 = np.float32(ct.c_float(b_1291_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_1291_ext),
                                                                                                                            b_1291_ext))
    try:
      sigma_1292 = np.float32(ct.c_float(sigma_1292_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_1292_ext),
                                                                                                                            sigma_1292_ext))
    try:
      r0_1293 = np.float32(ct.c_float(r0_1293_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_1293_ext),
                                                                                                                            r0_1293_ext))
    (out_mem_1989, out_arrsizze_1990,
     scalar_out_1991) = self.futhark_main(paths_1285, steps_1286,
                                          swap_term_1287, payments_1288,
                                          notional_1289, a_1290, b_1291,
                                          sigma_1292, r0_1293)
    return (cl.array.Array(self.queue, (out_arrsizze_1990,), ct.c_float,
                           data=out_mem_1989), np.float32(scalar_out_1991))