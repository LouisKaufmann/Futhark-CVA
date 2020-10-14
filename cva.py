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

__kernel void map_transpose_f32(__local volatile
                                int64_t *block_11_backing_aligned_0,
                                int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t in_elems_7,
                                int32_t out_elems_8, int32_t mulx_9,
                                int32_t muly_10, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,
                                                                 in_elems_7)) {
                ((__local float *) block_11)[(get_local_id_1_39 + j_43 * 8) *
                                             33 + get_local_id_0_38] =
                    ((__global float *) srcmem_2)[idata_offset_34 +
                                                  index_in_35];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,
                                                                 out_elems_8)) {
                ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
                    ((__local float *) block_11)[get_local_id_0_38 * 33 +
                                                 get_local_id_1_39 + j_43 * 8];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f32_low_height(__local volatile
                                           int64_t *block_11_backing_aligned_0,
                                           int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8, int32_t mulx_9,
                                           int32_t muly_10, __global
                                           unsigned char *destmem_0, __global
                                           unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_9) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_9);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        ((__local float *) block_11)[get_local_id_1_39 * 17 +
                                     get_local_id_0_38] = ((__global
                                                            float *) srcmem_2)[idata_offset_34 +
                                                                               index_in_35];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);
    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_9) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__local float *) block_11)[get_local_id_0_38 * 17 +
                                         get_local_id_1_39];
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f32_low_width(__local volatile
                                          int64_t *block_11_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t in_elems_7,
                                          int32_t out_elems_8, int32_t mulx_9,
                                          int32_t muly_10, __global
                                          unsigned char *destmem_0, __global
                                          unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_10);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_10) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        ((__local float *) block_11)[get_local_id_1_39 * 17 +
                                     get_local_id_0_38] = ((__global
                                                            float *) srcmem_2)[idata_offset_34 +
                                                                               index_in_35];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_10) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__local float *) block_11)[get_local_id_0_38 * 17 +
                                         get_local_id_1_39];
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f32_small(__local volatile
                                      int64_t *block_11_backing_aligned_0,
                                      int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t in_elems_7,
                                      int32_t out_elems_8, int32_t mulx_9,
                                      int32_t muly_10, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, in_elems_7)) {
        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__global float *) srcmem_2)[idata_offset_34 + index_in_35];
    }
    
  error_0:
    return;
}
__kernel void segmap_2066(__global int *global_failure,
                          int failure_is_an_option, __global
                          int *global_failure_args, int32_t paths_1765,
                          int32_t steps_1766, float swap_term_1767,
                          int32_t payments_1768, float notional_1769,
                          float a_1770, float b_1771, float sigma_1772,
                          float last_date_1779, float sims_per_year_1780,
                          __global unsigned char *mem_2362, __global
                          unsigned char *mem_2368)
{
    #define segmap_group_sizze_2072 (mainzisegmap_group_sizze_2071)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_2418;
    int32_t local_tid_2419;
    int32_t group_sizze_2422;
    int32_t wave_sizze_2421;
    int32_t group_tid_2420;
    
    global_tid_2418 = get_global_id(0);
    local_tid_2419 = get_local_id(0);
    group_sizze_2422 = get_local_size(0);
    wave_sizze_2421 = LOCKSTEP_WIDTH;
    group_tid_2420 = get_group_id(0);
    
    int32_t phys_tid_2066;
    
    phys_tid_2066 = global_tid_2418;
    
    int32_t gtid_2064;
    
    gtid_2064 = squot32(group_tid_2420 * segmap_group_sizze_2072 +
                        local_tid_2419, steps_1766);
    
    int32_t gtid_2065;
    
    gtid_2065 = group_tid_2420 * segmap_group_sizze_2072 + local_tid_2419 -
        squot32(group_tid_2420 * segmap_group_sizze_2072 + local_tid_2419,
                steps_1766) * steps_1766;
    if (slt32(gtid_2064, paths_1765) && slt32(gtid_2065, steps_1766)) {
        int32_t convop_x_2322 = 1 + gtid_2065;
        float binop_x_2323 = sitofp_i32_f32(convop_x_2322);
        float index_primexp_2324 = binop_x_2323 / sims_per_year_1780;
        bool cond_2081 = index_primexp_2324 < last_date_1779;
        float ceil_arg_2082 = index_primexp_2324 / swap_term_1767;
        float res_2083;
        
        res_2083 = futrts_ceil32(ceil_arg_2082);
        
        int32_t res_2084 = fptosi_f32_i32(res_2083);
        int32_t remaining_2085 = sub32(payments_1768, res_2084);
        int32_t distance_upwards_exclusive_2086 = sub32(remaining_2085, 1);
        int32_t distance_2087 = add32(1, distance_upwards_exclusive_2086);
        float res_2088;
        
        if (cond_2081) {
            float x_2079 = ((__global float *) mem_2362)[gtid_2065 *
                                                         paths_1765 +
                                                         gtid_2064];
            float nextpayment_2089 = swap_term_1767 * res_2083;
            bool bounds_invalid_upwards_2090 = slt32(remaining_2085, 1);
            bool valid_2091 = !bounds_invalid_upwards_2090;
            bool range_valid_c_2092;
            
            if (!valid_2091) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {
                    global_failure_args[0] = 1;
                    global_failure_args[1] = 2;
                    global_failure_args[2] = remaining_2085;
                    ;
                }
                return;
            }
            
            bool dim_match_2093 = remaining_2085 == distance_2087;
            bool empty_or_match_cert_2094;
            
            if (!dim_match_2093) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 1) == -1) {
                    ;
                }
                return;
            }
            
            float y_2095 = nextpayment_2089 - index_primexp_2324;
            float negate_arg_2096 = a_1770 * y_2095;
            float exp_arg_2097 = 0.0F - negate_arg_2096;
            float res_2098 = fpow32(2.7182817F, exp_arg_2097);
            float x_2099 = 1.0F - res_2098;
            float B_2100 = x_2099 / a_1770;
            float x_2101 = B_2100 - nextpayment_2089;
            float x_2102 = x_2101 + index_primexp_2324;
            float x_2103 = fpow32(a_1770, 2.0F);
            float x_2104 = b_1771 * x_2103;
            float x_2105 = fpow32(sigma_1772, 2.0F);
            float y_2106 = x_2105 / 2.0F;
            float y_2107 = x_2104 - y_2106;
            float x_2108 = x_2102 * y_2107;
            float A1_2109 = x_2108 / x_2103;
            float y_2110 = fpow32(B_2100, 2.0F);
            float x_2111 = x_2105 * y_2110;
            float y_2112 = 4.0F * a_1770;
            float A2_2113 = x_2111 / y_2112;
            float exp_arg_2114 = A1_2109 - A2_2113;
            float res_2115 = fpow32(2.7182817F, exp_arg_2114);
            float negate_arg_2116 = x_2079 * B_2100;
            float exp_arg_2117 = 0.0F - negate_arg_2116;
            float res_2118 = fpow32(2.7182817F, exp_arg_2117);
            float res_2119 = res_2115 * res_2118;
            int32_t j_m_i_2120 = sub32(-1, distance_upwards_exclusive_2086);
            int32_t n_2121 = squot32(j_m_i_2120, -1);
            bool empty_slice_2122 = n_2121 == 0;
            int32_t m_2123 = sub32(n_2121, 1);
            int32_t m_t_s_2124 = mul32(-1, m_2123);
            int32_t i_p_m_t_s_2125 = add32(distance_upwards_exclusive_2086,
                                           m_t_s_2124);
            bool zzero_leq_i_p_m_t_s_2126 = sle32(0, i_p_m_t_s_2125);
            bool i_p_m_t_s_leq_w_2127 = sle32(i_p_m_t_s_2125, remaining_2085);
            bool j_lte_i_2128 = sle32(-1, distance_upwards_exclusive_2086);
            bool y_2129 = zzero_leq_i_p_m_t_s_2126 && i_p_m_t_s_leq_w_2127;
            bool y_2130 = j_lte_i_2128 && y_2129;
            bool ok_or_empty_2131 = empty_slice_2122 || y_2130;
            bool index_certs_2132;
            
            if (!ok_or_empty_2131) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 2) == -1) {
                    global_failure_args[0] = -1;
                    global_failure_args[1] = remaining_2085;
                    ;
                }
                return;
            }
            
            bool y_2133 = slt32(0, n_2121);
            bool index_certs_2134;
            
            if (!y_2133) {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 3) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = n_2121;
                    ;
                }
                return;
            }
            
            float binop_y_2135 = sitofp_i32_f32(distance_2087);
            float binop_y_2136 = swap_term_1767 * binop_y_2135;
            float index_primexp_2137 = nextpayment_2089 + binop_y_2136;
            float y_2138 = index_primexp_2137 - index_primexp_2324;
            float negate_arg_2139 = a_1770 * y_2138;
            float exp_arg_2140 = 0.0F - negate_arg_2139;
            float res_2141 = fpow32(2.7182817F, exp_arg_2140);
            float x_2142 = 1.0F - res_2141;
            float B_2143 = x_2142 / a_1770;
            float x_2144 = B_2143 - index_primexp_2137;
            float x_2145 = x_2144 + index_primexp_2324;
            float x_2146 = y_2107 * x_2145;
            float A1_2147 = x_2146 / x_2103;
            float y_2148 = fpow32(B_2143, 2.0F);
            float x_2149 = x_2105 * y_2148;
            float A2_2150 = x_2149 / y_2112;
            float exp_arg_2151 = A1_2147 - A2_2150;
            float res_2152 = fpow32(2.7182817F, exp_arg_2151);
            float negate_arg_2153 = x_2079 * B_2143;
            float exp_arg_2154 = 0.0F - negate_arg_2153;
            float res_2155 = fpow32(2.7182817F, exp_arg_2154);
            float res_2156 = res_2152 * res_2155;
            float res_2158;
            float redout_2319 = 0.0F;
            
            for (int32_t i_2320 = 0; i_2320 < remaining_2085; i_2320++) {
                int32_t index_primexp_2330 = 1 + i_2320;
                float res_2163 = sitofp_i32_f32(index_primexp_2330);
                float res_2164 = swap_term_1767 * res_2163;
                float res_2165 = nextpayment_2089 + res_2164;
                float y_2166 = res_2165 - index_primexp_2324;
                float negate_arg_2167 = a_1770 * y_2166;
                float exp_arg_2168 = 0.0F - negate_arg_2167;
                float res_2169 = fpow32(2.7182817F, exp_arg_2168);
                float x_2170 = 1.0F - res_2169;
                float B_2171 = x_2170 / a_1770;
                float x_2172 = B_2171 - res_2165;
                float x_2173 = x_2172 + index_primexp_2324;
                float x_2174 = y_2107 * x_2173;
                float A1_2175 = x_2174 / x_2103;
                float y_2176 = fpow32(B_2171, 2.0F);
                float x_2177 = x_2105 * y_2176;
                float A2_2178 = x_2177 / y_2112;
                float exp_arg_2179 = A1_2175 - A2_2178;
                float res_2180 = fpow32(2.7182817F, exp_arg_2179);
                float negate_arg_2181 = x_2079 * B_2171;
                float exp_arg_2182 = 0.0F - negate_arg_2181;
                float res_2183 = fpow32(2.7182817F, exp_arg_2182);
                float res_2184 = res_2180 * res_2183;
                float res_2161 = res_2184 + redout_2319;
                float redout_tmp_2423 = res_2161;
                
                redout_2319 = redout_tmp_2423;
            }
            res_2158 = redout_2319;
            
            float x_2185 = res_2119 - res_2156;
            float x_2186 = 5.056644e-2F * swap_term_1767;
            float y_2187 = res_2158 * x_2186;
            float y_2188 = x_2185 - y_2187;
            float res_2189 = notional_1769 * y_2188;
            
            res_2088 = res_2189;
        } else {
            res_2088 = 0.0F;
        }
        
        float res_2190 = fmax32(0.0F, res_2088);
        
        ((__global float *) mem_2368)[gtid_2064 * steps_1766 + gtid_2065] =
            res_2190;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_2072
}
__kernel void segmap_2195(__global int *global_failure, int32_t paths_1765,
                          int32_t steps_1766, float a_1770, float b_1771,
                          float sigma_1772, float r0_1773, float dt_1777,
                          int32_t distance_upwards_exclusive_1782,
                          float res_1803, int32_t num_groups_2201, __global
                          unsigned char *mem_2354, __global
                          unsigned char *mem_2358, __global
                          unsigned char *mem_2362)
{
    #define segmap_group_sizze_2199 (mainzisegmap_group_sizze_2198)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_2407;
    int32_t local_tid_2408;
    int32_t group_sizze_2411;
    int32_t wave_sizze_2410;
    int32_t group_tid_2409;
    
    global_tid_2407 = get_global_id(0);
    local_tid_2408 = get_local_id(0);
    group_sizze_2411 = get_local_size(0);
    wave_sizze_2410 = LOCKSTEP_WIDTH;
    group_tid_2409 = get_group_id(0);
    
    int32_t phys_tid_2195;
    
    phys_tid_2195 = global_tid_2407;
    
    int32_t phys_group_id_2412;
    
    phys_group_id_2412 = get_group_id(0);
    for (int32_t i_2413 = 0; i_2413 < squot32(squot32(paths_1765 +
                                                      segmap_group_sizze_2199 -
                                                      1,
                                                      segmap_group_sizze_2199) -
                                              phys_group_id_2412 +
                                              num_groups_2201 - 1,
                                              num_groups_2201); i_2413++) {
        int32_t virt_group_id_2414 = phys_group_id_2412 + i_2413 *
                num_groups_2201;
        int32_t gtid_2194 = virt_group_id_2414 * segmap_group_sizze_2199 +
                local_tid_2408;
        
        if (slt32(gtid_2194, paths_1765)) {
            for (int32_t i_2415 = 0; i_2415 < steps_1766; i_2415++) {
                ((__global float *) mem_2358)[phys_tid_2195 + i_2415 *
                                              (num_groups_2201 *
                                               segmap_group_sizze_2199)] =
                    r0_1773;
            }
            for (int32_t i_2207 = 0; i_2207 < distance_upwards_exclusive_1782;
                 i_2207++) {
                float shortstep_arg_2208 = ((__global
                                             float *) mem_2354)[i_2207 *
                                                                paths_1765 +
                                                                gtid_2194];
                float shortstep_arg_2209 = ((__global
                                             float *) mem_2358)[phys_tid_2195 +
                                                                i_2207 *
                                                                (num_groups_2201 *
                                                                 segmap_group_sizze_2199)];
                float y_2210 = b_1771 - shortstep_arg_2209;
                float x_2211 = a_1770 * y_2210;
                float x_2212 = dt_1777 * x_2211;
                float x_2213 = res_1803 * shortstep_arg_2208;
                float y_2214 = sigma_1772 * x_2213;
                float delta_r_2215 = x_2212 + y_2214;
                float res_2216 = shortstep_arg_2209 + delta_r_2215;
                int32_t i_2217 = add32(1, i_2207);
                
                ((__global float *) mem_2358)[phys_tid_2195 + i_2217 *
                                              (num_groups_2201 *
                                               segmap_group_sizze_2199)] =
                    res_2216;
            }
            for (int32_t i_2417 = 0; i_2417 < steps_1766; i_2417++) {
                ((__global float *) mem_2362)[i_2417 * paths_1765 + gtid_2194] =
                    ((__global float *) mem_2358)[phys_tid_2195 + i_2417 *
                                                  (num_groups_2201 *
                                                   segmap_group_sizze_2199)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_2199
}
__kernel void segmap_2238(__global int *global_failure, int32_t paths_1765,
                          int32_t steps_1766, __global unsigned char *mem_2344,
                          __global unsigned char *mem_2350)
{
    #define segmap_group_sizze_2244 (mainzisegmap_group_sizze_2243)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_2401;
    int32_t local_tid_2402;
    int32_t group_sizze_2405;
    int32_t wave_sizze_2404;
    int32_t group_tid_2403;
    
    global_tid_2401 = get_global_id(0);
    local_tid_2402 = get_local_id(0);
    group_sizze_2405 = get_local_size(0);
    wave_sizze_2404 = LOCKSTEP_WIDTH;
    group_tid_2403 = get_group_id(0);
    
    int32_t phys_tid_2238;
    
    phys_tid_2238 = global_tid_2401;
    
    int32_t gtid_2236;
    
    gtid_2236 = squot32(group_tid_2403 * segmap_group_sizze_2244 +
                        local_tid_2402, steps_1766);
    
    int32_t gtid_2237;
    
    gtid_2237 = group_tid_2403 * segmap_group_sizze_2244 + local_tid_2402 -
        squot32(group_tid_2403 * segmap_group_sizze_2244 + local_tid_2402,
                steps_1766) * steps_1766;
    if (slt32(gtid_2236, paths_1765) && slt32(gtid_2237, steps_1766)) {
        int32_t unsign_arg_2251 = ((__global int32_t *) mem_2344)[gtid_2236];
        int32_t x_2253 = lshr32(gtid_2237, 16);
        int32_t x_2254 = gtid_2237 ^ x_2253;
        int32_t x_2255 = mul32(73244475, x_2254);
        int32_t x_2256 = lshr32(x_2255, 16);
        int32_t x_2257 = x_2255 ^ x_2256;
        int32_t x_2258 = mul32(73244475, x_2257);
        int32_t x_2259 = lshr32(x_2258, 16);
        int32_t x_2260 = x_2258 ^ x_2259;
        int32_t unsign_arg_2261 = unsign_arg_2251 ^ x_2260;
        int32_t unsign_arg_2262 = mul32(48271, unsign_arg_2261);
        int32_t unsign_arg_2263 = umod32(unsign_arg_2262, 2147483647);
        int32_t unsign_arg_2264 = mul32(48271, unsign_arg_2263);
        int32_t unsign_arg_2265 = umod32(unsign_arg_2264, 2147483647);
        float res_2266 = uitofp_i32_f32(unsign_arg_2263);
        float res_2267 = res_2266 / 2.1474836e9F;
        float res_2268 = uitofp_i32_f32(unsign_arg_2265);
        float res_2269 = res_2268 / 2.1474836e9F;
        float res_2270;
        
        res_2270 = futrts_log32(res_2267);
        
        float res_2271 = -2.0F * res_2270;
        float res_2272;
        
        res_2272 = futrts_sqrt32(res_2271);
        
        float res_2273 = 6.2831855F * res_2269;
        float res_2274;
        
        res_2274 = futrts_cos32(res_2273);
        
        float res_2275 = res_2272 * res_2274;
        
        ((__global float *) mem_2350)[gtid_2236 * steps_1766 + gtid_2237] =
            res_2275;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_2244
}
__kernel void segmap_2277(__global int *global_failure, int32_t paths_1765,
                          __global unsigned char *mem_2344)
{
    #define segmap_group_sizze_2281 (mainzisegmap_group_sizze_2280)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_2396;
    int32_t local_tid_2397;
    int32_t group_sizze_2400;
    int32_t wave_sizze_2399;
    int32_t group_tid_2398;
    
    global_tid_2396 = get_global_id(0);
    local_tid_2397 = get_local_id(0);
    group_sizze_2400 = get_local_size(0);
    wave_sizze_2399 = LOCKSTEP_WIDTH;
    group_tid_2398 = get_group_id(0);
    
    int32_t phys_tid_2277;
    
    phys_tid_2277 = global_tid_2396;
    
    int32_t gtid_2276;
    
    gtid_2276 = group_tid_2398 * segmap_group_sizze_2281 + local_tid_2397;
    if (slt32(gtid_2276, paths_1765)) {
        int32_t x_2289 = lshr32(gtid_2276, 16);
        int32_t x_2290 = gtid_2276 ^ x_2289;
        int32_t x_2291 = mul32(73244475, x_2290);
        int32_t x_2292 = lshr32(x_2291, 16);
        int32_t x_2293 = x_2291 ^ x_2292;
        int32_t x_2294 = mul32(73244475, x_2293);
        int32_t x_2295 = lshr32(x_2294, 16);
        int32_t x_2296 = x_2294 ^ x_2295;
        int32_t unsign_arg_2297 = 777822902 ^ x_2296;
        int32_t unsign_arg_2298 = mul32(48271, unsign_arg_2297);
        int32_t unsign_arg_2299 = umod32(unsign_arg_2298, 2147483647);
        
        ((__global int32_t *) mem_2344)[gtid_2276] = unsign_arg_2299;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_2281
}
__kernel void segred_nonseg_2311(__global int *global_failure, __local volatile
                                 int64_t *red_arr_mem_2437_backing_aligned_0,
                                 __local volatile
                                 int64_t *sync_arr_mem_2435_backing_aligned_1,
                                 int32_t paths_1765, int32_t steps_1766,
                                 float a_1770, float sims_per_year_1780,
                                 float res_1978, float x_1979, float x_1981,
                                 float y_1983, float y_1984,
                                 int32_t num_groups_2305, __global
                                 unsigned char *mem_2368, __global
                                 unsigned char *mem_2372, __global
                                 unsigned char *mem_2377, __global
                                 unsigned char *counter_mem_2425, __global
                                 unsigned char *group_res_arr_mem_2427,
                                 int32_t num_threads_2429)
{
    #define segred_group_sizze_2303 (mainzisegred_group_sizze_2302)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_2437_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_2437_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_2435_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_2435_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_2430;
    int32_t local_tid_2431;
    int32_t group_sizze_2434;
    int32_t wave_sizze_2433;
    int32_t group_tid_2432;
    
    global_tid_2430 = get_global_id(0);
    local_tid_2431 = get_local_id(0);
    group_sizze_2434 = get_local_size(0);
    wave_sizze_2433 = LOCKSTEP_WIDTH;
    group_tid_2432 = get_group_id(0);
    
    int32_t phys_tid_2311;
    
    phys_tid_2311 = global_tid_2430;
    
    __local char *sync_arr_mem_2435;
    
    sync_arr_mem_2435 = (__local char *) sync_arr_mem_2435_backing_0;
    
    __local char *red_arr_mem_2437;
    
    red_arr_mem_2437 = (__local char *) red_arr_mem_2437_backing_1;
    
    int32_t dummy_2309;
    
    dummy_2309 = 0;
    
    int32_t gtid_2310;
    
    gtid_2310 = 0;
    
    float x_acc_2439;
    int32_t chunk_sizze_2440;
    
    chunk_sizze_2440 = smin32(squot32(steps_1766 + segred_group_sizze_2303 *
                                      num_groups_2305 - 1,
                                      segred_group_sizze_2303 *
                                      num_groups_2305), squot32(steps_1766 -
                                                                phys_tid_2311 +
                                                                num_threads_2429 -
                                                                1,
                                                                num_threads_2429));
    
    float x_1988;
    float x_1989;
    
    // neutral-initialise the accumulators
    {
        x_acc_2439 = 0.0F;
    }
    for (int32_t i_2444 = 0; i_2444 < chunk_sizze_2440; i_2444++) {
        gtid_2310 = phys_tid_2311 + num_threads_2429 * i_2444;
        // apply map function
        {
            int32_t convop_x_2326 = 1 + gtid_2310;
            float binop_x_2327 = sitofp_i32_f32(convop_x_2326);
            float index_primexp_2328 = binop_x_2327 / sims_per_year_1780;
            float res_1993;
            float redout_2337 = 0.0F;
            
            for (int32_t i_2338 = 0; i_2338 < paths_1765; i_2338++) {
                float x_1997 = ((__global float *) mem_2368)[i_2338 *
                                                             steps_1766 +
                                                             gtid_2310];
                float res_1996 = x_1997 + redout_2337;
                float redout_tmp_2445 = res_1996;
                
                redout_2337 = redout_tmp_2445;
            }
            res_1993 = redout_2337;
            
            float res_1998 = res_1993 / res_1978;
            float negate_arg_1999 = a_1770 * index_primexp_2328;
            float exp_arg_2000 = 0.0F - negate_arg_1999;
            float res_2001 = fpow32(2.7182817F, exp_arg_2000);
            float x_2002 = 1.0F - res_2001;
            float B_2003 = x_2002 / a_1770;
            float x_2004 = B_2003 - index_primexp_2328;
            float x_2005 = y_1983 * x_2004;
            float A1_2006 = x_2005 / x_1979;
            float y_2007 = fpow32(B_2003, 2.0F);
            float x_2008 = x_1981 * y_2007;
            float A2_2009 = x_2008 / y_1984;
            float exp_arg_2010 = A1_2006 - A2_2009;
            float res_2011 = fpow32(2.7182817F, exp_arg_2010);
            float negate_arg_2012 = 5.0e-2F * B_2003;
            float exp_arg_2013 = 0.0F - negate_arg_2012;
            float res_2014 = fpow32(2.7182817F, exp_arg_2013);
            float res_2015 = res_2011 * res_2014;
            float res_2016 = res_1998 * res_2015;
            
            // save map-out results
            {
                ((__global float *) mem_2377)[dummy_2309 * steps_1766 +
                                              gtid_2310] = res_2016;
            }
            // load accumulator
            {
                x_1988 = x_acc_2439;
            }
            // load new values
            {
                x_1989 = res_2016;
            }
            // apply reduction operator
            {
                float res_1990 = x_1988 + x_1989;
                
                // store in accumulator
                {
                    x_acc_2439 = res_1990;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_1988 = x_acc_2439;
        ((__local float *) red_arr_mem_2437)[local_tid_2431] = x_1988;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_2446;
    int32_t skip_waves_2447;
    float x_2441;
    float x_2442;
    
    offset_2446 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_2431, segred_group_sizze_2303)) {
            x_2441 = ((__local float *) red_arr_mem_2437)[local_tid_2431 +
                                                          offset_2446];
        }
    }
    offset_2446 = 1;
    while (slt32(offset_2446, wave_sizze_2433)) {
        if (slt32(local_tid_2431 + offset_2446, segred_group_sizze_2303) &&
            ((local_tid_2431 - squot32(local_tid_2431, wave_sizze_2433) *
              wave_sizze_2433) & (2 * offset_2446 - 1)) == 0) {
            // read array element
            {
                x_2442 = ((volatile __local
                           float *) red_arr_mem_2437)[local_tid_2431 +
                                                      offset_2446];
            }
            // apply reduction operation
            {
                float res_2443 = x_2441 + x_2442;
                
                x_2441 = res_2443;
            }
            // write result of operation
            {
                ((volatile __local float *) red_arr_mem_2437)[local_tid_2431] =
                    x_2441;
            }
        }
        offset_2446 *= 2;
    }
    skip_waves_2447 = 1;
    while (slt32(skip_waves_2447, squot32(segred_group_sizze_2303 +
                                          wave_sizze_2433 - 1,
                                          wave_sizze_2433))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_2446 = skip_waves_2447 * wave_sizze_2433;
        if (slt32(local_tid_2431 + offset_2446, segred_group_sizze_2303) &&
            ((local_tid_2431 - squot32(local_tid_2431, wave_sizze_2433) *
              wave_sizze_2433) == 0 && (squot32(local_tid_2431,
                                                wave_sizze_2433) & (2 *
                                                                    skip_waves_2447 -
                                                                    1)) == 0)) {
            // read array element
            {
                x_2442 = ((__local float *) red_arr_mem_2437)[local_tid_2431 +
                                                              offset_2446];
            }
            // apply reduction operation
            {
                float res_2443 = x_2441 + x_2442;
                
                x_2441 = res_2443;
            }
            // write result of operation
            {
                ((__local float *) red_arr_mem_2437)[local_tid_2431] = x_2441;
            }
        }
        skip_waves_2447 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_2431 == 0) {
            x_acc_2439 = x_2441;
        }
    }
    
    int32_t old_counter_2448;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_2431 == 0) {
            ((__global float *) group_res_arr_mem_2427)[group_tid_2432 *
                                                        segred_group_sizze_2303] =
                x_acc_2439;
            mem_fence_global();
            old_counter_2448 = atomic_add_i32_global(&((volatile __global
                                                        int *) counter_mem_2425)[0],
                                                     (int) 1);
            ((__local bool *) sync_arr_mem_2435)[0] = old_counter_2448 ==
                num_groups_2305 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_2449;
    
    is_last_group_2449 = ((__local bool *) sync_arr_mem_2435)[0];
    if (is_last_group_2449) {
        if (local_tid_2431 == 0) {
            old_counter_2448 = atomic_add_i32_global(&((volatile __global
                                                        int *) counter_mem_2425)[0],
                                                     (int) (0 -
                                                            num_groups_2305));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_2450 = squot32(num_groups_2305 +
                                                   segred_group_sizze_2303 - 1,
                                                   segred_group_sizze_2303);
            
            x_1988 = 0.0F;
            for (int32_t i_2451 = 0; i_2451 < read_per_thread_2450; i_2451++) {
                int32_t group_res_id_2452 = local_tid_2431 *
                        read_per_thread_2450 + i_2451;
                int32_t index_of_group_res_2453 = group_res_id_2452;
                
                if (slt32(group_res_id_2452, num_groups_2305)) {
                    x_1989 = ((__global
                               float *) group_res_arr_mem_2427)[index_of_group_res_2453 *
                                                                segred_group_sizze_2303];
                    
                    float res_1990;
                    
                    res_1990 = x_1988 + x_1989;
                    x_1988 = res_1990;
                }
            }
        }
        ((__local float *) red_arr_mem_2437)[local_tid_2431] = x_1988;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_2454;
            int32_t skip_waves_2455;
            float x_2441;
            float x_2442;
            
            offset_2454 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_2431, segred_group_sizze_2303)) {
                    x_2441 = ((__local
                               float *) red_arr_mem_2437)[local_tid_2431 +
                                                          offset_2454];
                }
            }
            offset_2454 = 1;
            while (slt32(offset_2454, wave_sizze_2433)) {
                if (slt32(local_tid_2431 + offset_2454,
                          segred_group_sizze_2303) && ((local_tid_2431 -
                                                        squot32(local_tid_2431,
                                                                wave_sizze_2433) *
                                                        wave_sizze_2433) & (2 *
                                                                            offset_2454 -
                                                                            1)) ==
                    0) {
                    // read array element
                    {
                        x_2442 = ((volatile __local
                                   float *) red_arr_mem_2437)[local_tid_2431 +
                                                              offset_2454];
                    }
                    // apply reduction operation
                    {
                        float res_2443 = x_2441 + x_2442;
                        
                        x_2441 = res_2443;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_2437)[local_tid_2431] = x_2441;
                    }
                }
                offset_2454 *= 2;
            }
            skip_waves_2455 = 1;
            while (slt32(skip_waves_2455, squot32(segred_group_sizze_2303 +
                                                  wave_sizze_2433 - 1,
                                                  wave_sizze_2433))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_2454 = skip_waves_2455 * wave_sizze_2433;
                if (slt32(local_tid_2431 + offset_2454,
                          segred_group_sizze_2303) && ((local_tid_2431 -
                                                        squot32(local_tid_2431,
                                                                wave_sizze_2433) *
                                                        wave_sizze_2433) == 0 &&
                                                       (squot32(local_tid_2431,
                                                                wave_sizze_2433) &
                                                        (2 * skip_waves_2455 -
                                                         1)) == 0)) {
                    // read array element
                    {
                        x_2442 = ((__local
                                   float *) red_arr_mem_2437)[local_tid_2431 +
                                                              offset_2454];
                    }
                    // apply reduction operation
                    {
                        float res_2443 = x_2441 + x_2442;
                        
                        x_2441 = res_2443;
                    }
                    // write result of operation
                    {
                        ((__local float *) red_arr_mem_2437)[local_tid_2431] =
                            x_2441;
                    }
                }
                skip_waves_2455 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_2431 == 0) {
                    ((__global float *) mem_2372)[0] = x_2441;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_2303
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
    self.failure_msgs=["Range {}..{}...{} is invalid.\n-> #0  cva.fut:39:30-45\n   #1  cva.fut:54:27-71\n   #2  cva.fut:105:62-87\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:105:26-104\n   #6  cva.fut:103:21-106:31\n   #7  cva.fut:85:1-110:18\n",
     "value cannot match pattern\n-> #0  cva.fut:39:5-41:41\n   #1  cva.fut:54:27-71\n   #2  cva.fut:105:62-87\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:105:26-104\n   #6  cva.fut:103:21-106:31\n   #7  cva.fut:85:1-110:18\n",
     "Index [::{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:56:39-59\n   #1  cva.fut:105:62-87\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:105:26-104\n   #5  cva.fut:103:21-106:31\n   #6  cva.fut:85:1-110:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:56:38-63\n   #1  cva.fut:105:62-87\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:105:26-104\n   #5  cva.fut:103:21-106:31\n   #6  cva.fut:85:1-110:18\n"]
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
                                       all_sizes={"main.segmap_group_size_2071": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_2198": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_2243": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_2280": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_2200": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_2302": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_2304": {"class": "num_groups", "value": None}})
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.segmap_2066_var = program.segmap_2066
    self.segmap_2195_var = program.segmap_2195
    self.segmap_2238_var = program.segmap_2238
    self.segmap_2277_var = program.segmap_2277
    self.segred_nonseg_2311_var = program.segred_nonseg_2311
    self.constants = {}
    counter_mem_2425 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                 np.int32(0), np.int32(0), np.int32(0),
                                 np.int32(0), np.int32(0), np.int32(0),
                                 np.int32(0)], dtype=np.int32)
    static_mem_2456 = opencl_alloc(self, 40, "static_mem_2456")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_2456,
                      normaliseArray(counter_mem_2425), is_blocking=synchronous)
    self.counter_mem_2425 = static_mem_2456
  def futhark_main(self, paths_1765, steps_1766, swap_term_1767, payments_1768,
                   notional_1769, a_1770, b_1771, sigma_1772, r0_1773):
    res_1774 = sitofp_i32_f32(payments_1768)
    x_1775 = (swap_term_1767 * res_1774)
    res_1776 = sitofp_i32_f32(steps_1766)
    dt_1777 = (x_1775 / res_1776)
    y_1778 = (res_1774 - np.float32(1.0))
    last_date_1779 = (swap_term_1767 * y_1778)
    sims_per_year_1780 = (res_1776 / x_1775)
    bounds_invalid_upwards_1781 = slt32(steps_1766, np.int32(1))
    distance_upwards_exclusive_1782 = (steps_1766 - np.int32(1))
    valid_1784 = not(bounds_invalid_upwards_1781)
    range_valid_c_1785 = True
    assert valid_1784, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:66:56-67\n   #1  cva.fut:92:17-40\n   #2  cva.fut:85:1-110:18\n" % ("Range ",
                                                                                                                                                 np.int32(1),
                                                                                                                                                 "..",
                                                                                                                                                 np.int32(2),
                                                                                                                                                 "...",
                                                                                                                                                 steps_1766,
                                                                                                                                                 " is invalid."))
    bounds_invalid_upwards_1794 = slt32(paths_1765, np.int32(0))
    valid_1795 = not(bounds_invalid_upwards_1794)
    range_valid_c_1796 = True
    assert valid_1795, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/math.fut:453:23-30\n   #1  /prelude/array.fut:60:3-12\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:42-47\n   #3  cva.fut:93:19-49\n   #4  cva.fut:85:1-110:18\n" % ("Range ",
                                                                                                                                                                                                                                                             np.int32(0),
                                                                                                                                                                                                                                                             "..",
                                                                                                                                                                                                                                                             np.int32(1),
                                                                                                                                                                                                                                                             "..<",
                                                                                                                                                                                                                                                             paths_1765,
                                                                                                                                                                                                                                                             " is invalid."))
    res_1803 = futhark_sqrt32(dt_1777)
    paths_2278 = sext_i32_i64(paths_1765)
    segmap_group_sizze_2281 = self.sizes["main.segmap_group_size_2280"]
    segmap_group_sizze_2282 = sext_i32_i64(segmap_group_sizze_2281)
    y_2283 = (segmap_group_sizze_2282 - np.int64(1))
    x_2284 = (paths_2278 + y_2283)
    segmap_usable_groups_64_2286 = squot64(x_2284, segmap_group_sizze_2282)
    segmap_usable_groups_2287 = sext_i64_i32(segmap_usable_groups_64_2286)
    bytes_2342 = (np.int64(4) * paths_2278)
    mem_2344 = opencl_alloc(self, bytes_2342, "mem_2344")
    if ((1 * (np.long(segmap_usable_groups_2287) * np.long(segmap_group_sizze_2281))) != 0):
      self.segmap_2277_var.set_args(self.global_failure, np.int32(paths_1765),
                                    mem_2344)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_2277_var,
                                 ((np.long(segmap_usable_groups_2287) * np.long(segmap_group_sizze_2281)),),
                                 (np.long(segmap_group_sizze_2281),))
      if synchronous:
        sync(self)
    steps_2240 = sext_i32_i64(steps_1766)
    nest_sizze_2242 = (steps_2240 * paths_2278)
    segmap_group_sizze_2244 = self.sizes["main.segmap_group_size_2243"]
    segmap_group_sizze_2245 = sext_i32_i64(segmap_group_sizze_2244)
    y_2246 = (segmap_group_sizze_2245 - np.int64(1))
    x_2247 = (nest_sizze_2242 + y_2246)
    segmap_usable_groups_64_2249 = squot64(x_2247, segmap_group_sizze_2245)
    segmap_usable_groups_2250 = sext_i64_i32(segmap_usable_groups_64_2249)
    binop_x_2349 = (steps_2240 * paths_2278)
    bytes_2346 = (np.int64(4) * binop_x_2349)
    mem_2350 = opencl_alloc(self, bytes_2346, "mem_2350")
    if ((1 * (np.long(segmap_usable_groups_2250) * np.long(segmap_group_sizze_2244))) != 0):
      self.segmap_2238_var.set_args(self.global_failure, np.int32(paths_1765),
                                    np.int32(steps_1766), mem_2344, mem_2350)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_2238_var,
                                 ((np.long(segmap_usable_groups_2250) * np.long(segmap_group_sizze_2244)),),
                                 (np.long(segmap_group_sizze_2244),))
      if synchronous:
        sync(self)
    mem_2344 = None
    segmap_group_sizze_2199 = self.sizes["main.segmap_group_size_2198"]
    max_num_groups_2406 = self.sizes["main.segmap_num_groups_2200"]
    num_groups_2201 = sext_i64_i32(smax64(np.int32(1),
                                          smin64(squot64(((paths_2278 + sext_i32_i64(segmap_group_sizze_2199)) - np.int64(1)),
                                                         sext_i32_i64(segmap_group_sizze_2199)),
                                                 sext_i32_i64(max_num_groups_2406))))
    convop_x_2352 = (paths_1765 * steps_1766)
    binop_x_2353 = sext_i32_i64(convop_x_2352)
    bytes_2351 = (np.int64(4) * binop_x_2353)
    mem_2354 = opencl_alloc(self, bytes_2351, "mem_2354")
    self.futhark_builtinzhmap_transpose_f32(mem_2354, np.int32(0), mem_2350,
                                            np.int32(0), np.int32(1),
                                            steps_1766, paths_1765,
                                            (paths_1765 * steps_1766),
                                            (paths_1765 * steps_1766))
    mem_2350 = None
    mem_2362 = opencl_alloc(self, bytes_2351, "mem_2362")
    bytes_2356 = (np.int64(4) * steps_2240)
    num_threads_2386 = (segmap_group_sizze_2199 * num_groups_2201)
    num_threads64_2387 = sext_i32_i64(num_threads_2386)
    total_sizze_2388 = (bytes_2356 * num_threads64_2387)
    mem_2358 = opencl_alloc(self, total_sizze_2388, "mem_2358")
    if ((1 * (np.long(num_groups_2201) * np.long(segmap_group_sizze_2199))) != 0):
      self.segmap_2195_var.set_args(self.global_failure, np.int32(paths_1765),
                                    np.int32(steps_1766), np.float32(a_1770),
                                    np.float32(b_1771), np.float32(sigma_1772),
                                    np.float32(r0_1773), np.float32(dt_1777),
                                    np.int32(distance_upwards_exclusive_1782),
                                    np.float32(res_1803),
                                    np.int32(num_groups_2201), mem_2354,
                                    mem_2358, mem_2362)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_2195_var,
                                 ((np.long(num_groups_2201) * np.long(segmap_group_sizze_2199)),),
                                 (np.long(segmap_group_sizze_2199),))
      if synchronous:
        sync(self)
    mem_2354 = None
    mem_2358 = None
    segmap_group_sizze_2072 = self.sizes["main.segmap_group_size_2071"]
    segmap_group_sizze_2073 = sext_i32_i64(segmap_group_sizze_2072)
    y_2074 = (segmap_group_sizze_2073 - np.int64(1))
    x_2075 = (y_2074 + nest_sizze_2242)
    segmap_usable_groups_64_2077 = squot64(x_2075, segmap_group_sizze_2073)
    segmap_usable_groups_2078 = sext_i64_i32(segmap_usable_groups_64_2077)
    mem_2368 = opencl_alloc(self, bytes_2346, "mem_2368")
    if ((1 * (np.long(segmap_usable_groups_2078) * np.long(segmap_group_sizze_2072))) != 0):
      self.segmap_2066_var.set_args(self.global_failure,
                                    self.failure_is_an_option,
                                    self.global_failure_args,
                                    np.int32(paths_1765), np.int32(steps_1766),
                                    np.float32(swap_term_1767),
                                    np.int32(payments_1768),
                                    np.float32(notional_1769),
                                    np.float32(a_1770), np.float32(b_1771),
                                    np.float32(sigma_1772),
                                    np.float32(last_date_1779),
                                    np.float32(sims_per_year_1780), mem_2362,
                                    mem_2368)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_2066_var,
                                 ((np.long(segmap_usable_groups_2078) * np.long(segmap_group_sizze_2072)),),
                                 (np.long(segmap_group_sizze_2072),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_2362 = None
    res_1978 = sitofp_i32_f32(paths_1765)
    x_1979 = fpow32(a_1770, np.float32(2.0))
    x_1980 = (b_1771 * x_1979)
    x_1981 = fpow32(sigma_1772, np.float32(2.0))
    y_1982 = (x_1981 / np.float32(2.0))
    y_1983 = (x_1980 - y_1982)
    y_1984 = (np.float32(4.0) * a_1770)
    segred_group_sizze_2303 = self.sizes["main.segred_group_size_2302"]
    max_num_groups_2424 = self.sizes["main.segred_num_groups_2304"]
    num_groups_2305 = sext_i64_i32(smax64(np.int32(1),
                                          smin64(squot64(((steps_2240 + sext_i32_i64(segred_group_sizze_2303)) - np.int64(1)),
                                                         sext_i32_i64(segred_group_sizze_2303)),
                                                 sext_i32_i64(max_num_groups_2424))))
    mem_2372 = opencl_alloc(self, np.int64(4), "mem_2372")
    mem_2377 = opencl_alloc(self, bytes_2356, "mem_2377")
    counter_mem_2425 = self.counter_mem_2425
    group_res_arr_mem_2427 = opencl_alloc(self,
                                          (np.int32(4) * (segred_group_sizze_2303 * num_groups_2305)),
                                          "group_res_arr_mem_2427")
    num_threads_2429 = (num_groups_2305 * segred_group_sizze_2303)
    if ((1 * (np.long(num_groups_2305) * np.long(segred_group_sizze_2303))) != 0):
      self.segred_nonseg_2311_var.set_args(self.global_failure,
                                           cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_2303))),
                                           cl.LocalMemory(np.long(np.int32(1))),
                                           np.int32(paths_1765),
                                           np.int32(steps_1766),
                                           np.float32(a_1770),
                                           np.float32(sims_per_year_1780),
                                           np.float32(res_1978),
                                           np.float32(x_1979),
                                           np.float32(x_1981),
                                           np.float32(y_1983),
                                           np.float32(y_1984),
                                           np.int32(num_groups_2305), mem_2368,
                                           mem_2372, mem_2377, counter_mem_2425,
                                           group_res_arr_mem_2427,
                                           np.int32(num_threads_2429))
      cl.enqueue_nd_range_kernel(self.queue, self.segred_nonseg_2311_var,
                                 ((np.long(num_groups_2305) * np.long(segred_group_sizze_2303)),),
                                 (np.long(segred_group_sizze_2303),))
      if synchronous:
        sync(self)
    mem_2368 = None
    read_res_2457 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_2457, mem_2372,
                    device_offset=(np.long(np.int32(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_1986 = read_res_2457[0]
    mem_2372 = None
    CVA_2017 = (np.float32(6.000000052154064e-3) * res_1986)
    mem_2380 = opencl_alloc(self, bytes_2356, "mem_2380")
    if ((sext_i32_i64(steps_1766) * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_2380, mem_2377,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long((sext_i32_i64(steps_1766) * np.int32(4))))
    if synchronous:
      sync(self)
    mem_2377 = None
    out_arrsizze_2394 = steps_1766
    out_mem_2393 = mem_2380
    scalar_out_2395 = CVA_2017
    return (out_mem_2393, out_arrsizze_2394, scalar_out_2395)
  def futhark_builtinzhmap_transpose_f32(self, destmem_0, destoffset_1,
                                         srcmem_2, srcoffset_3, num_arrays_4,
                                         x_elems_5, y_elems_6, in_elems_7,
                                         out_elems_8):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_10 = squot32(np.int32(16), x_elems_5)
      mulx_9 = squot32(np.int32(16), y_elems_6)
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          sync(self)
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                                                            muly_10) + np.int32(16)) - np.int32(1)),
                                                                                                  np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_f32_low_width_var.set_args(cl.LocalMemory(np.long(np.int32(1088))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_f32_low_width_var,
                                       ((np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                   muly_10) + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              sync(self)
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                  mulx_9) + np.int32(16)) - np.int32(1)),
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                                                                    np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_f32_low_height_var.set_args(cl.LocalMemory(np.long(np.int32(1088))),
                                                             np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(in_elems_7),
                                                             np.int32(out_elems_8),
                                                             np.int32(mulx_9),
                                                             np.int32(muly_10),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_f32_low_height_var,
                                         ((np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                                     mulx_9) + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                sync(self)
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                        np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_f32_small_var.set_args(cl.LocalMemory(np.long(np.int32(1))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_small_var,
                                           ((np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                                             np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  sync(self)
            else:
              if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                          np.int32(32))) * np.long(np.int32(32)))) * (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                                                                      np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_f32_var.set_args(cl.LocalMemory(np.long(np.int32(4224))),
                                                    np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(in_elems_7),
                                                    np.int32(out_elems_8),
                                                    np.int32(mulx_9),
                                                    np.int32(muly_10),
                                                    destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_var,
                                           ((np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  sync(self)
    return ()
  def main(self, paths_1765_ext, steps_1766_ext, swap_term_1767_ext,
           payments_1768_ext, notional_1769_ext, a_1770_ext, b_1771_ext,
           sigma_1772_ext, r0_1773_ext):
    try:
      paths_1765 = np.int32(ct.c_int32(paths_1765_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(paths_1765_ext),
                                                                                                                            paths_1765_ext))
    try:
      steps_1766 = np.int32(ct.c_int32(steps_1766_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(steps_1766_ext),
                                                                                                                            steps_1766_ext))
    try:
      swap_term_1767 = np.float32(ct.c_float(swap_term_1767_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(swap_term_1767_ext),
                                                                                                                            swap_term_1767_ext))
    try:
      payments_1768 = np.int32(ct.c_int32(payments_1768_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(payments_1768_ext),
                                                                                                                            payments_1768_ext))
    try:
      notional_1769 = np.float32(ct.c_float(notional_1769_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(notional_1769_ext),
                                                                                                                            notional_1769_ext))
    try:
      a_1770 = np.float32(ct.c_float(a_1770_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_1770_ext),
                                                                                                                            a_1770_ext))
    try:
      b_1771 = np.float32(ct.c_float(b_1771_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_1771_ext),
                                                                                                                            b_1771_ext))
    try:
      sigma_1772 = np.float32(ct.c_float(sigma_1772_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_1772_ext),
                                                                                                                            sigma_1772_ext))
    try:
      r0_1773 = np.float32(ct.c_float(r0_1773_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_1773_ext),
                                                                                                                            r0_1773_ext))
    (out_mem_2393, out_arrsizze_2394,
     scalar_out_2395) = self.futhark_main(paths_1765, steps_1766,
                                          swap_term_1767, payments_1768,
                                          notional_1769, a_1770, b_1771,
                                          sigma_1772, r0_1773)
    return (cl.array.Array(self.queue, (out_arrsizze_2394,), ct.c_float,
                           data=out_mem_2393), np.float32(scalar_out_2395))