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
    for (platform_name, device_type, size, valuef) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and (self.device.type & device_type) == device_type:
               sizes[size] = valuef(self.device)
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

    # Futhark reserves 4 bytes of local memory for its own purposes.
    self.max_local_memory -= 4

    # See comment in rts/c/opencl.h.
    if self.platform.name.find('NVIDIA CUDA') >= 0:
        self.max_local_memory -= 12
    elif self.platform.name.find('AMD') >= 0:
        self.max_local_memory -= 16

    self.free_list = {}

    self.global_failure = self.pool.allocate(np.int32().itemsize)
    cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)
    self.global_failure_args = self.pool.allocate(np.int64().itemsize *
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
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi').replace('#', 'zh'),v) for (s,v) in self.sizes.items()])

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
        # Reset failure information.
        cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)

        # Read failure args.
        failure_args = np.empty(self.global_failure_args_max+1, dtype=np.int64)
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
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
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
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
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
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
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
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
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
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
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
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
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
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzl(x);
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




__kernel void mainzisegmap_14700(__global int *global_failure,
                                 unsigned char loop_nonempty_14817,
                                 float converted_sizze_14823, __global
                                 unsigned char *mem_14843, __global
                                 unsigned char *mem_14849)
{
    #define segmap_group_sizze_14762 (mainzisegmap_group_sizze_14702)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_14939;
    int32_t local_tid_14940;
    int64_t group_sizze_14943;
    int32_t wave_sizze_14942;
    int32_t group_tid_14941;
    
    global_tid_14939 = get_global_id(0);
    local_tid_14940 = get_local_id(0);
    group_sizze_14943 = get_local_size(0);
    wave_sizze_14942 = LOCKSTEP_WIDTH;
    group_tid_14941 = get_group_id(0);
    
    int32_t phys_tid_14700;
    
    phys_tid_14700 = global_tid_14939;
    
    int64_t gtid_14699;
    
    gtid_14699 = sext_i32_i64(group_tid_14941) * segmap_group_sizze_14762 +
        sext_i32_i64(local_tid_14940);
    if (slt64(gtid_14699, 2)) {
        float x_14770;
        
        if (loop_nonempty_14817) {
            float x_14827 = ((__global float *) mem_14843)[gtid_14699];
            
            x_14770 = x_14827;
        } else {
            x_14770 = 0.0F;
        }
        
        float y_14833 = x_14770 * converted_sizze_14823;
        
        ((__global float *) mem_14849)[gtid_14699] = y_14833;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14762
}
__kernel void mainzisegred_large_14730(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_14981_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_14979_backing_aligned_1,
                                       int64_t n_14575,
                                       int64_t num_groups_14774, __global
                                       unsigned char *mem_14843, __global
                                       unsigned char *mem_14852,
                                       int64_t groups_per_segment_14965,
                                       int64_t elements_per_thread_14966,
                                       int64_t virt_num_groups_14967,
                                       int64_t threads_per_segment_14969,
                                       __global
                                       unsigned char *group_res_arr_mem_14970,
                                       __global
                                       unsigned char *mainzicounter_mem_14972)
{
    #define segred_group_sizze_14773 (mainzisegred_group_sizze_14724)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_14981_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_14981_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_14979_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_14979_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_14974;
    int32_t local_tid_14975;
    int64_t group_sizze_14978;
    int32_t wave_sizze_14977;
    int32_t group_tid_14976;
    
    global_tid_14974 = get_global_id(0);
    local_tid_14975 = get_local_id(0);
    group_sizze_14978 = get_local_size(0);
    wave_sizze_14977 = LOCKSTEP_WIDTH;
    group_tid_14976 = get_group_id(0);
    
    int32_t phys_tid_14730;
    
    phys_tid_14730 = global_tid_14974;
    
    __local char *red_arr_mem_14979;
    
    red_arr_mem_14979 = (__local char *) red_arr_mem_14979_backing_0;
    
    __local char *sync_arr_mem_14981;
    
    sync_arr_mem_14981 = (__local char *) sync_arr_mem_14981_backing_1;
    
    int32_t phys_group_id_14983;
    
    phys_group_id_14983 = get_group_id(0);
    for (int32_t i_14984 = 0; i_14984 <
         sdiv_up32(sext_i64_i32(virt_num_groups_14967) - phys_group_id_14983,
                   sext_i64_i32(num_groups_14774)); i_14984++) {
        int32_t virt_group_id_14985 = phys_group_id_14983 + i_14984 *
                sext_i64_i32(num_groups_14774);
        int32_t flat_segment_id_14986 = squot32(virt_group_id_14985,
                                                sext_i64_i32(groups_per_segment_14965));
        int64_t global_tid_14987 = srem64(sext_i32_i64(virt_group_id_14985) *
                                          segred_group_sizze_14773 +
                                          sext_i32_i64(local_tid_14975),
                                          segred_group_sizze_14773 *
                                          groups_per_segment_14965);
        int64_t gtid_14721 = sext_i32_i64(flat_segment_id_14986);
        int64_t gtid_14729;
        float x_acc_14988;
        int64_t chunk_sizze_14989;
        
        chunk_sizze_14989 = smin64(elements_per_thread_14966,
                                   sdiv_up64(n_14575 -
                                             sext_i32_i64(sext_i64_i32(global_tid_14987)),
                                             threads_per_segment_14969));
        
        float x_14777;
        float x_14778;
        
        // neutral-initialise the accumulators
        {
            x_acc_14988 = 0.0F;
        }
        for (int64_t i_14993 = 0; i_14993 < chunk_sizze_14989; i_14993++) {
            gtid_14729 = sext_i32_i64(sext_i64_i32(global_tid_14987)) +
                threads_per_segment_14969 * i_14993;
            // apply map function
            {
                float x_14781 = ((__global float *) mem_14843)[gtid_14721];
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_14777 = x_acc_14988;
                }
                // load new values
                {
                    x_14778 = x_14781;
                }
                // apply reduction operator
                {
                    float res_14779 = x_14777 + x_14778;
                    
                    // store in accumulator
                    {
                        x_acc_14988 = res_14779;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_14777 = x_acc_14988;
            ((__local
              float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975)] =
                x_14777;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_14994;
        int32_t skip_waves_14995;
        
        skip_waves_14995 = 1;
        
        float x_14990;
        float x_14991;
        
        offset_14994 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_14975,
                      sext_i64_i32(segred_group_sizze_14773))) {
                x_14990 = ((__local
                            float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975 +
                                                        offset_14994)];
            }
        }
        offset_14994 = 1;
        while (slt32(offset_14994, wave_sizze_14977)) {
            if (slt32(local_tid_14975 + offset_14994,
                      sext_i64_i32(segred_group_sizze_14773)) &&
                ((local_tid_14975 - squot32(local_tid_14975, wave_sizze_14977) *
                  wave_sizze_14977) & (2 * offset_14994 - 1)) == 0) {
                // read array element
                {
                    x_14991 = ((volatile __local
                                float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975 +
                                                            offset_14994)];
                }
                // apply reduction operation
                {
                    float res_14992 = x_14990 + x_14991;
                    
                    x_14990 = res_14992;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975)] =
                        x_14990;
                }
            }
            offset_14994 *= 2;
        }
        while (slt32(skip_waves_14995,
                     squot32(sext_i64_i32(segred_group_sizze_14773) +
                             wave_sizze_14977 - 1, wave_sizze_14977))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_14994 = skip_waves_14995 * wave_sizze_14977;
            if (slt32(local_tid_14975 + offset_14994,
                      sext_i64_i32(segred_group_sizze_14773)) &&
                ((local_tid_14975 - squot32(local_tid_14975, wave_sizze_14977) *
                  wave_sizze_14977) == 0 && (squot32(local_tid_14975,
                                                     wave_sizze_14977) & (2 *
                                                                          skip_waves_14995 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_14991 = ((__local
                                float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975 +
                                                            offset_14994)];
                }
                // apply reduction operation
                {
                    float res_14992 = x_14990 + x_14991;
                    
                    x_14990 = res_14992;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975)] =
                        x_14990;
                }
            }
            skip_waves_14995 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (sext_i32_i64(local_tid_14975) == 0) {
                x_acc_14988 = x_14990;
            }
        }
        if (groups_per_segment_14965 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_14975 == 0) {
                    ((__global float *) mem_14852)[gtid_14721] = x_acc_14988;
                }
            }
        } else {
            int32_t old_counter_14996;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_14975 == 0) {
                    ((__global
                      float *) group_res_arr_mem_14970)[sext_i32_i64(virt_group_id_14985) *
                                                        segred_group_sizze_14773] =
                        x_acc_14988;
                    mem_fence_global();
                    old_counter_14996 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_14972)[sext_i32_i64(srem32(flat_segment_id_14986,
                                                                                                     10240))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_14981)[0] =
                        old_counter_14996 == groups_per_segment_14965 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_14997;
            
            is_last_group_14997 = ((__local bool *) sync_arr_mem_14981)[0];
            if (is_last_group_14997) {
                if (local_tid_14975 == 0) {
                    old_counter_14996 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_14972)[sext_i32_i64(srem32(flat_segment_id_14986,
                                                                                                     10240))],
                                              (int) (0 -
                                                     groups_per_segment_14965));
                }
                // read in the per-group-results
                {
                    int64_t read_per_thread_14998 =
                            sdiv_up64(groups_per_segment_14965,
                                      segred_group_sizze_14773);
                    
                    x_14777 = 0.0F;
                    for (int64_t i_14999 = 0; i_14999 < read_per_thread_14998;
                         i_14999++) {
                        int64_t group_res_id_15000 =
                                sext_i32_i64(local_tid_14975) *
                                read_per_thread_14998 + i_14999;
                        int64_t index_of_group_res_15001 =
                                sext_i32_i64(flat_segment_id_14986) *
                                groups_per_segment_14965 + group_res_id_15000;
                        
                        if (slt64(group_res_id_15000,
                                  groups_per_segment_14965)) {
                            x_14778 = ((__global
                                        float *) group_res_arr_mem_14970)[index_of_group_res_15001 *
                                                                          segred_group_sizze_14773];
                            
                            float res_14779;
                            
                            res_14779 = x_14777 + x_14778;
                            x_14777 = res_14779;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975)] =
                    x_14777;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_15002;
                    int32_t skip_waves_15003;
                    
                    skip_waves_15003 = 1;
                    
                    float x_14990;
                    float x_14991;
                    
                    offset_15002 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_14975,
                                  sext_i64_i32(segred_group_sizze_14773))) {
                            x_14990 = ((__local
                                        float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975 +
                                                                    offset_15002)];
                        }
                    }
                    offset_15002 = 1;
                    while (slt32(offset_15002, wave_sizze_14977)) {
                        if (slt32(local_tid_14975 + offset_15002,
                                  sext_i64_i32(segred_group_sizze_14773)) &&
                            ((local_tid_14975 - squot32(local_tid_14975,
                                                        wave_sizze_14977) *
                              wave_sizze_14977) & (2 * offset_15002 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_14991 = ((volatile __local
                                            float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975 +
                                                                        offset_15002)];
                            }
                            // apply reduction operation
                            {
                                float res_14992 = x_14990 + x_14991;
                                
                                x_14990 = res_14992;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975)] =
                                    x_14990;
                            }
                        }
                        offset_15002 *= 2;
                    }
                    while (slt32(skip_waves_15003,
                                 squot32(sext_i64_i32(segred_group_sizze_14773) +
                                         wave_sizze_14977 - 1,
                                         wave_sizze_14977))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_15002 = skip_waves_15003 * wave_sizze_14977;
                        if (slt32(local_tid_14975 + offset_15002,
                                  sext_i64_i32(segred_group_sizze_14773)) &&
                            ((local_tid_14975 - squot32(local_tid_14975,
                                                        wave_sizze_14977) *
                              wave_sizze_14977) == 0 &&
                             (squot32(local_tid_14975, wave_sizze_14977) & (2 *
                                                                            skip_waves_15003 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_14991 = ((__local
                                            float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975 +
                                                                        offset_15002)];
                            }
                            // apply reduction operation
                            {
                                float res_14992 = x_14990 + x_14991;
                                
                                x_14990 = res_14992;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_14979)[sext_i32_i64(local_tid_14975)] =
                                    x_14990;
                            }
                        }
                        skip_waves_15003 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_14975 == 0) {
                            ((__global float *) mem_14852)[gtid_14721] =
                                x_14990;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14773
}
__kernel void mainzisegred_nonseg_14667(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_14887_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_14885_backing_aligned_1,
                                        int64_t n_14575,
                                        int64_t num_groups_14662, __global
                                        unsigned char *swap_term_mem_14836,
                                        __global
                                        unsigned char *payments_mem_14837,
                                        __global unsigned char *mem_14841,
                                        __global
                                        unsigned char *mainzicounter_mem_14875,
                                        __global
                                        unsigned char *group_res_arr_mem_14877,
                                        int64_t num_threads_14879)
{
    #define segred_group_sizze_14660 (mainzisegred_group_sizze_14659)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_14887_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_14887_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_14885_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_14885_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_14880;
    int32_t local_tid_14881;
    int64_t group_sizze_14884;
    int32_t wave_sizze_14883;
    int32_t group_tid_14882;
    
    global_tid_14880 = get_global_id(0);
    local_tid_14881 = get_local_id(0);
    group_sizze_14884 = get_local_size(0);
    wave_sizze_14883 = LOCKSTEP_WIDTH;
    group_tid_14882 = get_group_id(0);
    
    int32_t phys_tid_14667;
    
    phys_tid_14667 = global_tid_14880;
    
    __local char *sync_arr_mem_14885;
    
    sync_arr_mem_14885 = (__local char *) sync_arr_mem_14885_backing_0;
    
    __local char *red_arr_mem_14887;
    
    red_arr_mem_14887 = (__local char *) red_arr_mem_14887_backing_1;
    
    int64_t dummy_14665;
    
    dummy_14665 = 0;
    
    int64_t gtid_14666;
    
    gtid_14666 = 0;
    
    float x_acc_14889;
    int64_t chunk_sizze_14890;
    
    chunk_sizze_14890 = smin64(sdiv_up64(n_14575,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_14660 *
                                         num_groups_14662))),
                               sdiv_up64(n_14575 - sext_i32_i64(phys_tid_14667),
                                         num_threads_14879));
    
    float x_14596;
    float x_14597;
    
    // neutral-initialise the accumulators
    {
        x_acc_14889 = -INFINITY;
    }
    for (int64_t i_14894 = 0; i_14894 < chunk_sizze_14890; i_14894++) {
        gtid_14666 = sext_i32_i64(phys_tid_14667) + num_threads_14879 * i_14894;
        // apply map function
        {
            float x_14599 = ((__global
                              float *) swap_term_mem_14836)[gtid_14666];
            int64_t x_14600 = ((__global
                                int64_t *) payments_mem_14837)[gtid_14666];
            float res_14601 = sitofp_i64_f32(x_14600);
            float res_14602 = x_14599 * res_14601;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_14596 = x_acc_14889;
            }
            // load new values
            {
                x_14597 = res_14602;
            }
            // apply reduction operator
            {
                float res_14598 = fmax32(x_14596, x_14597);
                
                // store in accumulator
                {
                    x_acc_14889 = res_14598;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_14596 = x_acc_14889;
        ((__local float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881)] =
            x_14596;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_14895;
    int32_t skip_waves_14896;
    
    skip_waves_14896 = 1;
    
    float x_14891;
    float x_14892;
    
    offset_14895 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_14881, sext_i64_i32(segred_group_sizze_14660))) {
            x_14891 = ((__local
                        float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881 +
                                                    offset_14895)];
        }
    }
    offset_14895 = 1;
    while (slt32(offset_14895, wave_sizze_14883)) {
        if (slt32(local_tid_14881 + offset_14895,
                  sext_i64_i32(segred_group_sizze_14660)) && ((local_tid_14881 -
                                                               squot32(local_tid_14881,
                                                                       wave_sizze_14883) *
                                                               wave_sizze_14883) &
                                                              (2 *
                                                               offset_14895 -
                                                               1)) == 0) {
            // read array element
            {
                x_14892 = ((volatile __local
                            float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881 +
                                                        offset_14895)];
            }
            // apply reduction operation
            {
                float res_14893 = fmax32(x_14891, x_14892);
                
                x_14891 = res_14893;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881)] =
                    x_14891;
            }
        }
        offset_14895 *= 2;
    }
    while (slt32(skip_waves_14896,
                 squot32(sext_i64_i32(segred_group_sizze_14660) +
                         wave_sizze_14883 - 1, wave_sizze_14883))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_14895 = skip_waves_14896 * wave_sizze_14883;
        if (slt32(local_tid_14881 + offset_14895,
                  sext_i64_i32(segred_group_sizze_14660)) && ((local_tid_14881 -
                                                               squot32(local_tid_14881,
                                                                       wave_sizze_14883) *
                                                               wave_sizze_14883) ==
                                                              0 &&
                                                              (squot32(local_tid_14881,
                                                                       wave_sizze_14883) &
                                                               (2 *
                                                                skip_waves_14896 -
                                                                1)) == 0)) {
            // read array element
            {
                x_14892 = ((__local
                            float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881 +
                                                        offset_14895)];
            }
            // apply reduction operation
            {
                float res_14893 = fmax32(x_14891, x_14892);
                
                x_14891 = res_14893;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881)] =
                    x_14891;
            }
        }
        skip_waves_14896 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_14881) == 0) {
            x_acc_14889 = x_14891;
        }
    }
    
    int32_t old_counter_14897;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_14881 == 0) {
            ((__global
              float *) group_res_arr_mem_14877)[sext_i32_i64(group_tid_14882) *
                                                segred_group_sizze_14660] =
                x_acc_14889;
            mem_fence_global();
            old_counter_14897 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_14875)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_14885)[0] = old_counter_14897 ==
                num_groups_14662 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_14898;
    
    is_last_group_14898 = ((__local bool *) sync_arr_mem_14885)[0];
    if (is_last_group_14898) {
        if (local_tid_14881 == 0) {
            old_counter_14897 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_14875)[0],
                                                      (int) (0 -
                                                             num_groups_14662));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_14899 = sdiv_up64(num_groups_14662,
                                                      segred_group_sizze_14660);
            
            x_14596 = -INFINITY;
            for (int64_t i_14900 = 0; i_14900 < read_per_thread_14899;
                 i_14900++) {
                int64_t group_res_id_14901 = sext_i32_i64(local_tid_14881) *
                        read_per_thread_14899 + i_14900;
                int64_t index_of_group_res_14902 = group_res_id_14901;
                
                if (slt64(group_res_id_14901, num_groups_14662)) {
                    x_14597 = ((__global
                                float *) group_res_arr_mem_14877)[index_of_group_res_14902 *
                                                                  segred_group_sizze_14660];
                    
                    float res_14598;
                    
                    res_14598 = fmax32(x_14596, x_14597);
                    x_14596 = res_14598;
                }
            }
        }
        ((__local float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881)] =
            x_14596;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_14903;
            int32_t skip_waves_14904;
            
            skip_waves_14904 = 1;
            
            float x_14891;
            float x_14892;
            
            offset_14903 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_14881,
                          sext_i64_i32(segred_group_sizze_14660))) {
                    x_14891 = ((__local
                                float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881 +
                                                            offset_14903)];
                }
            }
            offset_14903 = 1;
            while (slt32(offset_14903, wave_sizze_14883)) {
                if (slt32(local_tid_14881 + offset_14903,
                          sext_i64_i32(segred_group_sizze_14660)) &&
                    ((local_tid_14881 - squot32(local_tid_14881,
                                                wave_sizze_14883) *
                      wave_sizze_14883) & (2 * offset_14903 - 1)) == 0) {
                    // read array element
                    {
                        x_14892 = ((volatile __local
                                    float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881 +
                                                                offset_14903)];
                    }
                    // apply reduction operation
                    {
                        float res_14893 = fmax32(x_14891, x_14892);
                        
                        x_14891 = res_14893;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881)] =
                            x_14891;
                    }
                }
                offset_14903 *= 2;
            }
            while (slt32(skip_waves_14904,
                         squot32(sext_i64_i32(segred_group_sizze_14660) +
                                 wave_sizze_14883 - 1, wave_sizze_14883))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_14903 = skip_waves_14904 * wave_sizze_14883;
                if (slt32(local_tid_14881 + offset_14903,
                          sext_i64_i32(segred_group_sizze_14660)) &&
                    ((local_tid_14881 - squot32(local_tid_14881,
                                                wave_sizze_14883) *
                      wave_sizze_14883) == 0 && (squot32(local_tid_14881,
                                                         wave_sizze_14883) &
                                                 (2 * skip_waves_14904 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_14892 = ((__local
                                    float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881 +
                                                                offset_14903)];
                    }
                    // apply reduction operation
                    {
                        float res_14893 = fmax32(x_14891, x_14892);
                        
                        x_14891 = res_14893;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_14887)[sext_i32_i64(local_tid_14881)] =
                            x_14891;
                    }
                }
                skip_waves_14904 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_14881 == 0) {
                    ((__global float *) mem_14841)[0] = x_14891;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14660
}
__kernel void mainzisegred_nonseg_14679(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_14921_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_14919_backing_aligned_1,
                                        int64_t num_groups_14682,
                                        unsigned char loop_nonempty_14817,
                                        float converted_sizze_14823, __global
                                        unsigned char *mem_14843, __global
                                        unsigned char *mem_14846, __global
                                        unsigned char *mainzicounter_mem_14909,
                                        __global
                                        unsigned char *group_res_arr_mem_14911,
                                        int64_t num_threads_14913)
{
    #define segred_group_sizze_14681 (mainzisegred_group_sizze_14671)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_14921_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_14921_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_14919_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_14919_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_14914;
    int32_t local_tid_14915;
    int64_t group_sizze_14918;
    int32_t wave_sizze_14917;
    int32_t group_tid_14916;
    
    global_tid_14914 = get_global_id(0);
    local_tid_14915 = get_local_id(0);
    group_sizze_14918 = get_local_size(0);
    wave_sizze_14917 = LOCKSTEP_WIDTH;
    group_tid_14916 = get_group_id(0);
    
    int32_t phys_tid_14679;
    
    phys_tid_14679 = global_tid_14914;
    
    __local char *sync_arr_mem_14919;
    
    sync_arr_mem_14919 = (__local char *) sync_arr_mem_14919_backing_0;
    
    __local char *red_arr_mem_14921;
    
    red_arr_mem_14921 = (__local char *) red_arr_mem_14921_backing_1;
    
    int64_t dummy_14677;
    
    dummy_14677 = 0;
    
    int64_t gtid_14678;
    
    gtid_14678 = 0;
    
    float x_acc_14923;
    int64_t chunk_sizze_14924;
    
    chunk_sizze_14924 = smin64(sdiv_up64(2,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_14681 *
                                         num_groups_14682))), sdiv_up64(2 -
                                                                        sext_i32_i64(phys_tid_14679),
                                                                        num_threads_14913));
    
    float x_14685;
    float x_14686;
    
    // neutral-initialise the accumulators
    {
        x_acc_14923 = 0.0F;
    }
    for (int64_t i_14928 = 0; i_14928 < chunk_sizze_14924; i_14928++) {
        gtid_14678 = sext_i32_i64(phys_tid_14679) + num_threads_14913 * i_14928;
        // apply map function
        {
            float x_14693;
            
            if (loop_nonempty_14817) {
                float x_14818 = ((__global float *) mem_14843)[gtid_14678];
                
                x_14693 = x_14818;
            } else {
                x_14693 = 0.0F;
            }
            
            float y_14824 = x_14693 * converted_sizze_14823;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_14685 = x_acc_14923;
            }
            // load new values
            {
                x_14686 = y_14824;
            }
            // apply reduction operator
            {
                float res_14687 = x_14685 + x_14686;
                
                // store in accumulator
                {
                    x_acc_14923 = res_14687;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_14685 = x_acc_14923;
        ((__local float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915)] =
            x_14685;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_14929;
    int32_t skip_waves_14930;
    
    skip_waves_14930 = 1;
    
    float x_14925;
    float x_14926;
    
    offset_14929 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_14915, sext_i64_i32(segred_group_sizze_14681))) {
            x_14925 = ((__local
                        float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915 +
                                                    offset_14929)];
        }
    }
    offset_14929 = 1;
    while (slt32(offset_14929, wave_sizze_14917)) {
        if (slt32(local_tid_14915 + offset_14929,
                  sext_i64_i32(segred_group_sizze_14681)) && ((local_tid_14915 -
                                                               squot32(local_tid_14915,
                                                                       wave_sizze_14917) *
                                                               wave_sizze_14917) &
                                                              (2 *
                                                               offset_14929 -
                                                               1)) == 0) {
            // read array element
            {
                x_14926 = ((volatile __local
                            float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915 +
                                                        offset_14929)];
            }
            // apply reduction operation
            {
                float res_14927 = x_14925 + x_14926;
                
                x_14925 = res_14927;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915)] =
                    x_14925;
            }
        }
        offset_14929 *= 2;
    }
    while (slt32(skip_waves_14930,
                 squot32(sext_i64_i32(segred_group_sizze_14681) +
                         wave_sizze_14917 - 1, wave_sizze_14917))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_14929 = skip_waves_14930 * wave_sizze_14917;
        if (slt32(local_tid_14915 + offset_14929,
                  sext_i64_i32(segred_group_sizze_14681)) && ((local_tid_14915 -
                                                               squot32(local_tid_14915,
                                                                       wave_sizze_14917) *
                                                               wave_sizze_14917) ==
                                                              0 &&
                                                              (squot32(local_tid_14915,
                                                                       wave_sizze_14917) &
                                                               (2 *
                                                                skip_waves_14930 -
                                                                1)) == 0)) {
            // read array element
            {
                x_14926 = ((__local
                            float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915 +
                                                        offset_14929)];
            }
            // apply reduction operation
            {
                float res_14927 = x_14925 + x_14926;
                
                x_14925 = res_14927;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915)] =
                    x_14925;
            }
        }
        skip_waves_14930 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_14915) == 0) {
            x_acc_14923 = x_14925;
        }
    }
    
    int32_t old_counter_14931;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_14915 == 0) {
            ((__global
              float *) group_res_arr_mem_14911)[sext_i32_i64(group_tid_14916) *
                                                segred_group_sizze_14681] =
                x_acc_14923;
            mem_fence_global();
            old_counter_14931 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_14909)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_14919)[0] = old_counter_14931 ==
                num_groups_14682 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_14932;
    
    is_last_group_14932 = ((__local bool *) sync_arr_mem_14919)[0];
    if (is_last_group_14932) {
        if (local_tid_14915 == 0) {
            old_counter_14931 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_14909)[0],
                                                      (int) (0 -
                                                             num_groups_14682));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_14933 = sdiv_up64(num_groups_14682,
                                                      segred_group_sizze_14681);
            
            x_14685 = 0.0F;
            for (int64_t i_14934 = 0; i_14934 < read_per_thread_14933;
                 i_14934++) {
                int64_t group_res_id_14935 = sext_i32_i64(local_tid_14915) *
                        read_per_thread_14933 + i_14934;
                int64_t index_of_group_res_14936 = group_res_id_14935;
                
                if (slt64(group_res_id_14935, num_groups_14682)) {
                    x_14686 = ((__global
                                float *) group_res_arr_mem_14911)[index_of_group_res_14936 *
                                                                  segred_group_sizze_14681];
                    
                    float res_14687;
                    
                    res_14687 = x_14685 + x_14686;
                    x_14685 = res_14687;
                }
            }
        }
        ((__local float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915)] =
            x_14685;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_14937;
            int32_t skip_waves_14938;
            
            skip_waves_14938 = 1;
            
            float x_14925;
            float x_14926;
            
            offset_14937 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_14915,
                          sext_i64_i32(segred_group_sizze_14681))) {
                    x_14925 = ((__local
                                float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915 +
                                                            offset_14937)];
                }
            }
            offset_14937 = 1;
            while (slt32(offset_14937, wave_sizze_14917)) {
                if (slt32(local_tid_14915 + offset_14937,
                          sext_i64_i32(segred_group_sizze_14681)) &&
                    ((local_tid_14915 - squot32(local_tid_14915,
                                                wave_sizze_14917) *
                      wave_sizze_14917) & (2 * offset_14937 - 1)) == 0) {
                    // read array element
                    {
                        x_14926 = ((volatile __local
                                    float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915 +
                                                                offset_14937)];
                    }
                    // apply reduction operation
                    {
                        float res_14927 = x_14925 + x_14926;
                        
                        x_14925 = res_14927;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915)] =
                            x_14925;
                    }
                }
                offset_14937 *= 2;
            }
            while (slt32(skip_waves_14938,
                         squot32(sext_i64_i32(segred_group_sizze_14681) +
                                 wave_sizze_14917 - 1, wave_sizze_14917))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_14937 = skip_waves_14938 * wave_sizze_14917;
                if (slt32(local_tid_14915 + offset_14937,
                          sext_i64_i32(segred_group_sizze_14681)) &&
                    ((local_tid_14915 - squot32(local_tid_14915,
                                                wave_sizze_14917) *
                      wave_sizze_14917) == 0 && (squot32(local_tid_14915,
                                                         wave_sizze_14917) &
                                                 (2 * skip_waves_14938 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_14926 = ((__local
                                    float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915 +
                                                                offset_14937)];
                    }
                    // apply reduction operation
                    {
                        float res_14927 = x_14925 + x_14926;
                        
                        x_14925 = res_14927;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_14921)[sext_i32_i64(local_tid_14915)] =
                            x_14925;
                    }
                }
                skip_waves_14938 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_14915 == 0) {
                    ((__global float *) mem_14846)[0] = x_14925;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14681
}
__kernel void mainzisegred_nonseg_14757(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_15017_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_15015_backing_aligned_1,
                                        int64_t num_groups_14785, __global
                                        unsigned char *res_map_acc_mem_14853,
                                        __global unsigned char *mem_14856,
                                        __global
                                        unsigned char *mainzicounter_mem_15005,
                                        __global
                                        unsigned char *group_res_arr_mem_15007,
                                        int64_t num_threads_15009)
{
    #define segred_group_sizze_14784 (mainzisegred_group_sizze_14749)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_15017_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_15017_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_15015_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_15015_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_15010;
    int32_t local_tid_15011;
    int64_t group_sizze_15014;
    int32_t wave_sizze_15013;
    int32_t group_tid_15012;
    
    global_tid_15010 = get_global_id(0);
    local_tid_15011 = get_local_id(0);
    group_sizze_15014 = get_local_size(0);
    wave_sizze_15013 = LOCKSTEP_WIDTH;
    group_tid_15012 = get_group_id(0);
    
    int32_t phys_tid_14757;
    
    phys_tid_14757 = global_tid_15010;
    
    __local char *sync_arr_mem_15015;
    
    sync_arr_mem_15015 = (__local char *) sync_arr_mem_15015_backing_0;
    
    __local char *red_arr_mem_15017;
    
    red_arr_mem_15017 = (__local char *) red_arr_mem_15017_backing_1;
    
    int64_t dummy_14755;
    
    dummy_14755 = 0;
    
    int64_t gtid_14756;
    
    gtid_14756 = 0;
    
    float x_acc_15019;
    int64_t chunk_sizze_15020;
    
    chunk_sizze_15020 = smin64(sdiv_up64(2,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_14784 *
                                         num_groups_14785))), sdiv_up64(2 -
                                                                        sext_i32_i64(phys_tid_14757),
                                                                        num_threads_15009));
    
    float x_14788;
    float x_14789;
    
    // neutral-initialise the accumulators
    {
        x_acc_15019 = 0.0F;
    }
    for (int64_t i_15024 = 0; i_15024 < chunk_sizze_15020; i_15024++) {
        gtid_14756 = sext_i32_i64(phys_tid_14757) + num_threads_15009 * i_15024;
        // apply map function
        {
            float x_14791 = ((__global
                              float *) res_map_acc_mem_14853)[gtid_14756];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_14788 = x_acc_15019;
            }
            // load new values
            {
                x_14789 = x_14791;
            }
            // apply reduction operator
            {
                float res_14790 = x_14788 + x_14789;
                
                // store in accumulator
                {
                    x_acc_15019 = res_14790;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_14788 = x_acc_15019;
        ((__local float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011)] =
            x_14788;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_15025;
    int32_t skip_waves_15026;
    
    skip_waves_15026 = 1;
    
    float x_15021;
    float x_15022;
    
    offset_15025 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_15011, sext_i64_i32(segred_group_sizze_14784))) {
            x_15021 = ((__local
                        float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011 +
                                                    offset_15025)];
        }
    }
    offset_15025 = 1;
    while (slt32(offset_15025, wave_sizze_15013)) {
        if (slt32(local_tid_15011 + offset_15025,
                  sext_i64_i32(segred_group_sizze_14784)) && ((local_tid_15011 -
                                                               squot32(local_tid_15011,
                                                                       wave_sizze_15013) *
                                                               wave_sizze_15013) &
                                                              (2 *
                                                               offset_15025 -
                                                               1)) == 0) {
            // read array element
            {
                x_15022 = ((volatile __local
                            float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011 +
                                                        offset_15025)];
            }
            // apply reduction operation
            {
                float res_15023 = x_15021 + x_15022;
                
                x_15021 = res_15023;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011)] =
                    x_15021;
            }
        }
        offset_15025 *= 2;
    }
    while (slt32(skip_waves_15026,
                 squot32(sext_i64_i32(segred_group_sizze_14784) +
                         wave_sizze_15013 - 1, wave_sizze_15013))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_15025 = skip_waves_15026 * wave_sizze_15013;
        if (slt32(local_tid_15011 + offset_15025,
                  sext_i64_i32(segred_group_sizze_14784)) && ((local_tid_15011 -
                                                               squot32(local_tid_15011,
                                                                       wave_sizze_15013) *
                                                               wave_sizze_15013) ==
                                                              0 &&
                                                              (squot32(local_tid_15011,
                                                                       wave_sizze_15013) &
                                                               (2 *
                                                                skip_waves_15026 -
                                                                1)) == 0)) {
            // read array element
            {
                x_15022 = ((__local
                            float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011 +
                                                        offset_15025)];
            }
            // apply reduction operation
            {
                float res_15023 = x_15021 + x_15022;
                
                x_15021 = res_15023;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011)] =
                    x_15021;
            }
        }
        skip_waves_15026 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_15011) == 0) {
            x_acc_15019 = x_15021;
        }
    }
    
    int32_t old_counter_15027;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_15011 == 0) {
            ((__global
              float *) group_res_arr_mem_15007)[sext_i32_i64(group_tid_15012) *
                                                segred_group_sizze_14784] =
                x_acc_15019;
            mem_fence_global();
            old_counter_15027 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_15005)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_15015)[0] = old_counter_15027 ==
                num_groups_14785 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_15028;
    
    is_last_group_15028 = ((__local bool *) sync_arr_mem_15015)[0];
    if (is_last_group_15028) {
        if (local_tid_15011 == 0) {
            old_counter_15027 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_15005)[0],
                                                      (int) (0 -
                                                             num_groups_14785));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_15029 = sdiv_up64(num_groups_14785,
                                                      segred_group_sizze_14784);
            
            x_14788 = 0.0F;
            for (int64_t i_15030 = 0; i_15030 < read_per_thread_15029;
                 i_15030++) {
                int64_t group_res_id_15031 = sext_i32_i64(local_tid_15011) *
                        read_per_thread_15029 + i_15030;
                int64_t index_of_group_res_15032 = group_res_id_15031;
                
                if (slt64(group_res_id_15031, num_groups_14785)) {
                    x_14789 = ((__global
                                float *) group_res_arr_mem_15007)[index_of_group_res_15032 *
                                                                  segred_group_sizze_14784];
                    
                    float res_14790;
                    
                    res_14790 = x_14788 + x_14789;
                    x_14788 = res_14790;
                }
            }
        }
        ((__local float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011)] =
            x_14788;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_15033;
            int32_t skip_waves_15034;
            
            skip_waves_15034 = 1;
            
            float x_15021;
            float x_15022;
            
            offset_15033 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_15011,
                          sext_i64_i32(segred_group_sizze_14784))) {
                    x_15021 = ((__local
                                float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011 +
                                                            offset_15033)];
                }
            }
            offset_15033 = 1;
            while (slt32(offset_15033, wave_sizze_15013)) {
                if (slt32(local_tid_15011 + offset_15033,
                          sext_i64_i32(segred_group_sizze_14784)) &&
                    ((local_tid_15011 - squot32(local_tid_15011,
                                                wave_sizze_15013) *
                      wave_sizze_15013) & (2 * offset_15033 - 1)) == 0) {
                    // read array element
                    {
                        x_15022 = ((volatile __local
                                    float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011 +
                                                                offset_15033)];
                    }
                    // apply reduction operation
                    {
                        float res_15023 = x_15021 + x_15022;
                        
                        x_15021 = res_15023;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011)] =
                            x_15021;
                    }
                }
                offset_15033 *= 2;
            }
            while (slt32(skip_waves_15034,
                         squot32(sext_i64_i32(segred_group_sizze_14784) +
                                 wave_sizze_15013 - 1, wave_sizze_15013))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_15033 = skip_waves_15034 * wave_sizze_15013;
                if (slt32(local_tid_15011 + offset_15033,
                          sext_i64_i32(segred_group_sizze_14784)) &&
                    ((local_tid_15011 - squot32(local_tid_15011,
                                                wave_sizze_15013) *
                      wave_sizze_15013) == 0 && (squot32(local_tid_15011,
                                                         wave_sizze_15013) &
                                                 (2 * skip_waves_15034 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_15022 = ((__local
                                    float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011 +
                                                                offset_15033)];
                    }
                    // apply reduction operation
                    {
                        float res_15023 = x_15021 + x_15022;
                        
                        x_15021 = res_15023;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_15017)[sext_i32_i64(local_tid_15011)] =
                            x_15021;
                    }
                }
                skip_waves_15034 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_15011 == 0) {
                    ((__global float *) mem_14856)[0] = x_15021;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14784
}
__kernel void mainzisegred_nonseg_14804(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_15049_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_15047_backing_aligned_1,
                                        int64_t steps_14580, float a_14585,
                                        float sims_per_year_14604,
                                        float res_14623, float x_14624,
                                        float x_14626, float y_14628,
                                        float y_14629, int64_t num_groups_14798,
                                        __global unsigned char *mem_14859,
                                        __global unsigned char *mem_14861,
                                        __global
                                        unsigned char *mainzicounter_mem_15037,
                                        __global
                                        unsigned char *group_res_arr_mem_15039,
                                        int64_t num_threads_15041)
{
    #define segred_group_sizze_14796 (mainzisegred_group_sizze_14795)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_15049_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_15049_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_15047_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_15047_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_15042;
    int32_t local_tid_15043;
    int64_t group_sizze_15046;
    int32_t wave_sizze_15045;
    int32_t group_tid_15044;
    
    global_tid_15042 = get_global_id(0);
    local_tid_15043 = get_local_id(0);
    group_sizze_15046 = get_local_size(0);
    wave_sizze_15045 = LOCKSTEP_WIDTH;
    group_tid_15044 = get_group_id(0);
    
    int32_t phys_tid_14804;
    
    phys_tid_14804 = global_tid_15042;
    
    __local char *sync_arr_mem_15047;
    
    sync_arr_mem_15047 = (__local char *) sync_arr_mem_15047_backing_0;
    
    __local char *red_arr_mem_15049;
    
    red_arr_mem_15049 = (__local char *) red_arr_mem_15049_backing_1;
    
    int64_t dummy_14802;
    
    dummy_14802 = 0;
    
    int64_t gtid_14803;
    
    gtid_14803 = 0;
    
    float x_acc_15051;
    int64_t chunk_sizze_15052;
    
    chunk_sizze_15052 = smin64(sdiv_up64(steps_14580,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_14796 *
                                         num_groups_14798))),
                               sdiv_up64(steps_14580 -
                                         sext_i32_i64(phys_tid_14804),
                                         num_threads_15041));
    
    float x_14632;
    float x_14633;
    
    // neutral-initialise the accumulators
    {
        x_acc_15051 = 0.0F;
    }
    for (int64_t i_15056 = 0; i_15056 < chunk_sizze_15052; i_15056++) {
        gtid_14803 = sext_i32_i64(phys_tid_14804) + num_threads_15041 * i_15056;
        // apply map function
        {
            int64_t index_primexp_14806 = add64(1, gtid_14803);
            float res_14636 = sitofp_i64_f32(index_primexp_14806);
            float res_14637 = res_14636 / sims_per_year_14604;
            float negate_arg_14638 = a_14585 * res_14637;
            float exp_arg_14639 = 0.0F - negate_arg_14638;
            float res_14640 = fpow32(2.7182817F, exp_arg_14639);
            float x_14641 = 1.0F - res_14640;
            float B_14642 = x_14641 / a_14585;
            float x_14643 = B_14642 - res_14637;
            float x_14644 = y_14628 * x_14643;
            float A1_14645 = x_14644 / x_14624;
            float y_14646 = fpow32(B_14642, 2.0F);
            float x_14647 = x_14626 * y_14646;
            float A2_14648 = x_14647 / y_14629;
            float exp_arg_14649 = A1_14645 - A2_14648;
            float res_14650 = fpow32(2.7182817F, exp_arg_14649);
            float negate_arg_14651 = 5.0e-2F * B_14642;
            float exp_arg_14652 = 0.0F - negate_arg_14651;
            float res_14653 = fpow32(2.7182817F, exp_arg_14652);
            float res_14654 = res_14650 * res_14653;
            float res_14655 = res_14623 * res_14654;
            
            // save map-out results
            {
                ((__global float *) mem_14861)[dummy_14802 * steps_14580 +
                                               gtid_14803] = res_14655;
            }
            // load accumulator
            {
                x_14632 = x_acc_15051;
            }
            // load new values
            {
                x_14633 = res_14655;
            }
            // apply reduction operator
            {
                float res_14634 = x_14632 + x_14633;
                
                // store in accumulator
                {
                    x_acc_15051 = res_14634;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_14632 = x_acc_15051;
        ((__local float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043)] =
            x_14632;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_15057;
    int32_t skip_waves_15058;
    
    skip_waves_15058 = 1;
    
    float x_15053;
    float x_15054;
    
    offset_15057 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_15043, sext_i64_i32(segred_group_sizze_14796))) {
            x_15053 = ((__local
                        float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043 +
                                                    offset_15057)];
        }
    }
    offset_15057 = 1;
    while (slt32(offset_15057, wave_sizze_15045)) {
        if (slt32(local_tid_15043 + offset_15057,
                  sext_i64_i32(segred_group_sizze_14796)) && ((local_tid_15043 -
                                                               squot32(local_tid_15043,
                                                                       wave_sizze_15045) *
                                                               wave_sizze_15045) &
                                                              (2 *
                                                               offset_15057 -
                                                               1)) == 0) {
            // read array element
            {
                x_15054 = ((volatile __local
                            float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043 +
                                                        offset_15057)];
            }
            // apply reduction operation
            {
                float res_15055 = x_15053 + x_15054;
                
                x_15053 = res_15055;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043)] =
                    x_15053;
            }
        }
        offset_15057 *= 2;
    }
    while (slt32(skip_waves_15058,
                 squot32(sext_i64_i32(segred_group_sizze_14796) +
                         wave_sizze_15045 - 1, wave_sizze_15045))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_15057 = skip_waves_15058 * wave_sizze_15045;
        if (slt32(local_tid_15043 + offset_15057,
                  sext_i64_i32(segred_group_sizze_14796)) && ((local_tid_15043 -
                                                               squot32(local_tid_15043,
                                                                       wave_sizze_15045) *
                                                               wave_sizze_15045) ==
                                                              0 &&
                                                              (squot32(local_tid_15043,
                                                                       wave_sizze_15045) &
                                                               (2 *
                                                                skip_waves_15058 -
                                                                1)) == 0)) {
            // read array element
            {
                x_15054 = ((__local
                            float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043 +
                                                        offset_15057)];
            }
            // apply reduction operation
            {
                float res_15055 = x_15053 + x_15054;
                
                x_15053 = res_15055;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043)] =
                    x_15053;
            }
        }
        skip_waves_15058 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_15043) == 0) {
            x_acc_15051 = x_15053;
        }
    }
    
    int32_t old_counter_15059;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_15043 == 0) {
            ((__global
              float *) group_res_arr_mem_15039)[sext_i32_i64(group_tid_15044) *
                                                segred_group_sizze_14796] =
                x_acc_15051;
            mem_fence_global();
            old_counter_15059 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_15037)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_15047)[0] = old_counter_15059 ==
                num_groups_14798 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_15060;
    
    is_last_group_15060 = ((__local bool *) sync_arr_mem_15047)[0];
    if (is_last_group_15060) {
        if (local_tid_15043 == 0) {
            old_counter_15059 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_15037)[0],
                                                      (int) (0 -
                                                             num_groups_14798));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_15061 = sdiv_up64(num_groups_14798,
                                                      segred_group_sizze_14796);
            
            x_14632 = 0.0F;
            for (int64_t i_15062 = 0; i_15062 < read_per_thread_15061;
                 i_15062++) {
                int64_t group_res_id_15063 = sext_i32_i64(local_tid_15043) *
                        read_per_thread_15061 + i_15062;
                int64_t index_of_group_res_15064 = group_res_id_15063;
                
                if (slt64(group_res_id_15063, num_groups_14798)) {
                    x_14633 = ((__global
                                float *) group_res_arr_mem_15039)[index_of_group_res_15064 *
                                                                  segred_group_sizze_14796];
                    
                    float res_14634;
                    
                    res_14634 = x_14632 + x_14633;
                    x_14632 = res_14634;
                }
            }
        }
        ((__local float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043)] =
            x_14632;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_15065;
            int32_t skip_waves_15066;
            
            skip_waves_15066 = 1;
            
            float x_15053;
            float x_15054;
            
            offset_15065 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_15043,
                          sext_i64_i32(segred_group_sizze_14796))) {
                    x_15053 = ((__local
                                float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043 +
                                                            offset_15065)];
                }
            }
            offset_15065 = 1;
            while (slt32(offset_15065, wave_sizze_15045)) {
                if (slt32(local_tid_15043 + offset_15065,
                          sext_i64_i32(segred_group_sizze_14796)) &&
                    ((local_tid_15043 - squot32(local_tid_15043,
                                                wave_sizze_15045) *
                      wave_sizze_15045) & (2 * offset_15065 - 1)) == 0) {
                    // read array element
                    {
                        x_15054 = ((volatile __local
                                    float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043 +
                                                                offset_15065)];
                    }
                    // apply reduction operation
                    {
                        float res_15055 = x_15053 + x_15054;
                        
                        x_15053 = res_15055;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043)] =
                            x_15053;
                    }
                }
                offset_15065 *= 2;
            }
            while (slt32(skip_waves_15066,
                         squot32(sext_i64_i32(segred_group_sizze_14796) +
                                 wave_sizze_15045 - 1, wave_sizze_15045))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_15065 = skip_waves_15066 * wave_sizze_15045;
                if (slt32(local_tid_15043 + offset_15065,
                          sext_i64_i32(segred_group_sizze_14796)) &&
                    ((local_tid_15043 - squot32(local_tid_15043,
                                                wave_sizze_15045) *
                      wave_sizze_15045) == 0 && (squot32(local_tid_15043,
                                                         wave_sizze_15045) &
                                                 (2 * skip_waves_15066 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_15054 = ((__local
                                    float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043 +
                                                                offset_15065)];
                    }
                    // apply reduction operation
                    {
                        float res_15055 = x_15053 + x_15054;
                        
                        x_15053 = res_15055;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_15049)[sext_i32_i64(local_tid_15043)] =
                            x_15053;
                    }
                }
                skip_waves_15066 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_15043 == 0) {
                    ((__global float *) mem_14859)[0] = x_15053;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14796
}
__kernel void mainzisegred_small_14730(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_14951_backing_aligned_0,
                                       int64_t n_14575,
                                       int64_t num_groups_14774, __global
                                       unsigned char *mem_14843, __global
                                       unsigned char *mem_14852,
                                       int64_t segment_sizze_nonzzero_14944)
{
    #define segred_group_sizze_14773 (mainzisegred_group_sizze_14724)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_14951_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_14951_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_14946;
    int32_t local_tid_14947;
    int64_t group_sizze_14950;
    int32_t wave_sizze_14949;
    int32_t group_tid_14948;
    
    global_tid_14946 = get_global_id(0);
    local_tid_14947 = get_local_id(0);
    group_sizze_14950 = get_local_size(0);
    wave_sizze_14949 = LOCKSTEP_WIDTH;
    group_tid_14948 = get_group_id(0);
    
    int32_t phys_tid_14730;
    
    phys_tid_14730 = global_tid_14946;
    
    __local char *red_arr_mem_14951;
    
    red_arr_mem_14951 = (__local char *) red_arr_mem_14951_backing_0;
    
    int32_t phys_group_id_14953;
    
    phys_group_id_14953 = get_group_id(0);
    for (int32_t i_14954 = 0; i_14954 < sdiv_up32(sext_i64_i32(sdiv_up64(2,
                                                                         squot64(segred_group_sizze_14773,
                                                                                 segment_sizze_nonzzero_14944))) -
                                                  phys_group_id_14953,
                                                  sext_i64_i32(num_groups_14774));
         i_14954++) {
        int32_t virt_group_id_14955 = phys_group_id_14953 + i_14954 *
                sext_i64_i32(num_groups_14774);
        int64_t gtid_14721 = squot64(sext_i32_i64(local_tid_14947),
                                     segment_sizze_nonzzero_14944) +
                sext_i32_i64(virt_group_id_14955) *
                squot64(segred_group_sizze_14773, segment_sizze_nonzzero_14944);
        int64_t gtid_14729 = srem64(sext_i32_i64(local_tid_14947), n_14575);
        
        // apply map function if in bounds
        {
            if (slt64(0, n_14575) && (slt64(gtid_14721, 2) &&
                                      slt64(sext_i32_i64(local_tid_14947),
                                            n_14575 *
                                            squot64(segred_group_sizze_14773,
                                                    segment_sizze_nonzzero_14944)))) {
                float x_14781 = ((__global float *) mem_14843)[gtid_14721];
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)] =
                        x_14781;
                }
            } else {
                ((__local
                  float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt64(0, n_14575)) {
            // perform segmented scan to imitate reduction
            {
                float x_14777;
                float x_14778;
                float x_14956;
                float x_14957;
                bool ltid_in_bounds_14959;
                
                ltid_in_bounds_14959 = slt64(sext_i32_i64(local_tid_14947),
                                             n_14575 *
                                             squot64(segred_group_sizze_14773,
                                                     segment_sizze_nonzzero_14944));
                
                int32_t skip_threads_14960;
                
                // read input for in-block scan
                {
                    if (ltid_in_bounds_14959) {
                        x_14778 = ((volatile __local
                                    float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)];
                        if ((local_tid_14947 - squot32(local_tid_14947, 32) *
                             32) == 0) {
                            x_14777 = x_14778;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_14960 = 1;
                    while (slt32(skip_threads_14960, 32)) {
                        if (sle32(skip_threads_14960, local_tid_14947 -
                                  squot32(local_tid_14947, 32) * 32) &&
                            ltid_in_bounds_14959) {
                            // read operands
                            {
                                x_14777 = ((volatile __local
                                            float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947) -
                                                                        sext_i32_i64(skip_threads_14960)];
                            }
                            // perform operation
                            {
                                bool inactive_14961 =
                                     slt64(srem64(sext_i32_i64(local_tid_14947),
                                                  n_14575),
                                           sext_i32_i64(local_tid_14947) -
                                           sext_i32_i64(local_tid_14947 -
                                           skip_threads_14960));
                                
                                if (inactive_14961) {
                                    x_14777 = x_14778;
                                }
                                if (!inactive_14961) {
                                    float res_14779 = x_14777 + x_14778;
                                    
                                    x_14777 = res_14779;
                                }
                            }
                        }
                        if (sle32(wave_sizze_14949, skip_threads_14960)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_14960, local_tid_14947 -
                                  squot32(local_tid_14947, 32) * 32) &&
                            ltid_in_bounds_14959) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)] =
                                    x_14777;
                                x_14778 = x_14777;
                            }
                        }
                        if (sle32(wave_sizze_14949, skip_threads_14960)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_14960 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_14947 - squot32(local_tid_14947, 32) * 32) ==
                        31 && ltid_in_bounds_14959) {
                        ((volatile __local
                          float *) red_arr_mem_14951)[sext_i32_i64(squot32(local_tid_14947,
                                                                           32))] =
                            x_14777;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_14962;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_14947, 32) == 0 &&
                            ltid_in_bounds_14959) {
                            x_14957 = ((volatile __local
                                        float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)];
                            if ((local_tid_14947 - squot32(local_tid_14947,
                                                           32) * 32) == 0) {
                                x_14956 = x_14957;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_14962 = 1;
                        while (slt32(skip_threads_14962, 32)) {
                            if (sle32(skip_threads_14962, local_tid_14947 -
                                      squot32(local_tid_14947, 32) * 32) &&
                                (squot32(local_tid_14947, 32) == 0 &&
                                 ltid_in_bounds_14959)) {
                                // read operands
                                {
                                    x_14956 = ((volatile __local
                                                float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947) -
                                                                            sext_i32_i64(skip_threads_14962)];
                                }
                                // perform operation
                                {
                                    bool inactive_14963 =
                                         slt64(srem64(sext_i32_i64(local_tid_14947 *
                                                      32 + 32 - 1), n_14575),
                                               sext_i32_i64(local_tid_14947 *
                                               32 + 32 - 1) -
                                               sext_i32_i64((local_tid_14947 -
                                                             skip_threads_14962) *
                                               32 + 32 - 1));
                                    
                                    if (inactive_14963) {
                                        x_14956 = x_14957;
                                    }
                                    if (!inactive_14963) {
                                        float res_14958 = x_14956 + x_14957;
                                        
                                        x_14956 = res_14958;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_14949, skip_threads_14962)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_14962, local_tid_14947 -
                                      squot32(local_tid_14947, 32) * 32) &&
                                (squot32(local_tid_14947, 32) == 0 &&
                                 ltid_in_bounds_14959)) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)] =
                                        x_14956;
                                    x_14957 = x_14956;
                                }
                            }
                            if (sle32(wave_sizze_14949, skip_threads_14962)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_14962 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_14947, 32) == 0 ||
                          !ltid_in_bounds_14959)) {
                        // read operands
                        {
                            x_14778 = x_14777;
                            x_14777 = ((__local
                                        float *) red_arr_mem_14951)[sext_i32_i64(squot32(local_tid_14947,
                                                                                         32)) -
                                                                    1];
                        }
                        // perform operation
                        {
                            bool inactive_14964 =
                                 slt64(srem64(sext_i32_i64(local_tid_14947),
                                              n_14575),
                                       sext_i32_i64(local_tid_14947) -
                                       sext_i32_i64(squot32(local_tid_14947,
                                                            32) * 32 - 1));
                            
                            if (inactive_14964) {
                                x_14777 = x_14778;
                            }
                            if (!inactive_14964) {
                                float res_14779 = x_14777 + x_14778;
                                
                                x_14777 = res_14779;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)] =
                                x_14777;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_14947, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_14951)[sext_i32_i64(local_tid_14947)] =
                            x_14778;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt64(sext_i32_i64(virt_group_id_14955) *
                      squot64(segred_group_sizze_14773,
                              segment_sizze_nonzzero_14944) +
                      sext_i32_i64(local_tid_14947), 2) &&
                slt64(sext_i32_i64(local_tid_14947),
                      squot64(segred_group_sizze_14773,
                              segment_sizze_nonzzero_14944))) {
                ((__global
                  float *) mem_14852)[sext_i32_i64(virt_group_id_14955) *
                                      squot64(segred_group_sizze_14773,
                                              segment_sizze_nonzzero_14944) +
                                      sext_i32_i64(local_tid_14947)] = ((__local
                                                                         float *) red_arr_mem_14951)[(sext_i32_i64(local_tid_14947) +
                                                                                                      1) *
                                                                                                     segment_sizze_nonzzero_14944 -
                                                                                                     1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14773
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

def intlit(t, x):
  if t == np.int8:
    return np.int8(x)
  elif t == np.int16:
    return np.int16(x)
  elif t == np.int32:
    return np.int32(x)
  else:
    return np.int64(x)

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

# Python is so slow that we just make all the unsafe operations safe,
# always.

def sdivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x // y

def sdiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return (x+y-intlit(type(x), 1)) // y

def smodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x % y

def udivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) // unsigned(y))

def udiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed((unsigned(x)+unsigned(y)-unsigned(intlit(type(x),1))) // unsigned(y))

def umodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
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

sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
sdiv_up8 = sdiv1_up6 = sdiv_up32 = sdiv_up64 = sdiv_upN
sdiv_safe8 = sdiv1_safe6 = sdiv_safe32 = sdiv_safe64 = sdivN
sdiv_up_safe8 = sdiv_up1_safe6 = sdiv_up_safe32 = sdiv_up_safe64 = sdiv_upN
smod8 = smod16 = smod32 = smod64 = smodN
smod_safe8 = smod_safe16 = smod_safe32 = smod_safe64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
udiv_up8 = udiv_up16 = udiv_up32 = udiv_up64 = udivN
udiv_safe8 = udiv_safe16 = udiv_safe32 = udiv_safe64 = udiv_upN
udiv_up_safe8 = udiv_up_safe16 = udiv_up_safe32 = udiv_up_safe64 = udiv_upN
umod8 = umod16 = umod32 = umod64 = umodN
umod_safe8 = umod_safe16 = umod_safe32 = umod_safe64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
squot_safe8 = squot_safe16 = squot_safe32 = squot_safe64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
srem_safe8 = srem_safe16 = srem_safe32 = srem_safe64 = sremN

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
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

def ctz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if (x & 1) == 1:
      break
    n += 1
    x >>= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T
futhark_ctzz8 = futhark_ctzz16 = futhark_ctzz32 = futhark_ctzz64 = ctz_T

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
  entry_points = {"main": (["i64", "i64", "[]i64", "[]f32", "[]i64", "[]f32",
                            "f32", "f32", "f32", "f32"], ["f32", "[]f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width",
      lambda device: np.int32(32)), ("AMD Accelerated Parallel Processing",
                                     cl.device_type.GPU, "lockstep_width",
                                     lambda device: np.int32(32)), ("",
                                                                    cl.device_type.GPU,
                                                                    "lockstep_width",
                                                                    lambda device: np.int32(1)),
     ("", cl.device_type.GPU, "num_groups",
      lambda device: (np.int32(4) * device.get_info(getattr(cl.device_info,
                                                            "MAX_COMPUTE_UNITS")))),
     ("", cl.device_type.GPU, "group_size", lambda device: np.int32(256)), ("",
                                                                            cl.device_type.GPU,
                                                                            "tile_size",
                                                                            lambda device: np.int32(32)),
     ("", cl.device_type.GPU, "threshold", lambda device: np.int32(32768)), ("",
                                                                             cl.device_type.CPU,
                                                                             "lockstep_width",
                                                                             lambda device: np.int32(1)),
     ("", cl.device_type.CPU, "num_groups",
      lambda device: device.get_info(getattr(cl.device_info, "MAX_COMPUTE_UNITS"))),
     ("", cl.device_type.CPU, "group_size", lambda device: np.int32(32)), ("",
                                                                           cl.device_type.CPU,
                                                                           "tile_size",
                                                                           lambda device: np.int32(4)),
     ("", cl.device_type.CPU, "threshold",
      lambda device: device.get_info(getattr(cl.device_info, "MAX_COMPUTE_UNITS")))]
    self.global_failure_args_max = 0
    self.failure_msgs=[]
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
                                       required_types=["i32", "i64", "f32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"main.segmap_group_size_14702": {"class": "group_size", "value": None},
                                        "main.segred_group_size_14659": {"class": "group_size", "value": None},
                                        "main.segred_group_size_14671": {"class": "group_size", "value": None},
                                        "main.segred_group_size_14724": {"class": "group_size", "value": None},
                                        "main.segred_group_size_14749": {"class": "group_size", "value": None},
                                        "main.segred_group_size_14795": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_14661": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_14673": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_14726": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_14751": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_14797": {"class": "num_groups", "value": None},
                                        "main.suff_outer_par_1": {"class": "threshold (!main.suff_outer_redomap_0)",
                                                                  "value": None},
                                        "main.suff_outer_redomap_0": {"class": "threshold ()", "value": None}})
    self.mainzisegmap_14700_var = program.mainzisegmap_14700
    self.mainzisegred_large_14730_var = program.mainzisegred_large_14730
    self.mainzisegred_nonseg_14667_var = program.mainzisegred_nonseg_14667
    self.mainzisegred_nonseg_14679_var = program.mainzisegred_nonseg_14679
    self.mainzisegred_nonseg_14757_var = program.mainzisegred_nonseg_14757
    self.mainzisegred_nonseg_14804_var = program.mainzisegred_nonseg_14804
    self.mainzisegred_small_14730_var = program.mainzisegred_small_14730
    self.constants = {}
    mainzicounter_mem_14875 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_15067 = opencl_alloc(self, 40, "static_mem_15067")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_15067,
                      normaliseArray(mainzicounter_mem_14875),
                      is_blocking=synchronous)
    self.mainzicounter_mem_14875 = static_mem_15067
    mainzistatic_array_14905 = np.array([np.float32(1.0), np.float32(2.0)],
                                        dtype=np.float32)
    static_mem_15069 = opencl_alloc(self, 8, "static_mem_15069")
    if (8 != 0):
      cl.enqueue_copy(self.queue, static_mem_15069,
                      normaliseArray(mainzistatic_array_14905),
                      is_blocking=synchronous)
    self.mainzistatic_array_14905 = static_mem_15069
    mainzicounter_mem_14909 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_15070 = opencl_alloc(self, 40, "static_mem_15070")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_15070,
                      normaliseArray(mainzicounter_mem_14909),
                      is_blocking=synchronous)
    self.mainzicounter_mem_14909 = static_mem_15070
    mainzicounter_mem_14972 = np.zeros(10240, dtype=np.int32)
    static_mem_15072 = opencl_alloc(self, 40960, "static_mem_15072")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_15072,
                      normaliseArray(mainzicounter_mem_14972),
                      is_blocking=synchronous)
    self.mainzicounter_mem_14972 = static_mem_15072
    mainzicounter_mem_15005 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_15073 = opencl_alloc(self, 40, "static_mem_15073")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_15073,
                      normaliseArray(mainzicounter_mem_15005),
                      is_blocking=synchronous)
    self.mainzicounter_mem_15005 = static_mem_15073
    mainzicounter_mem_15037 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_15075 = opencl_alloc(self, 40, "static_mem_15075")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_15075,
                      normaliseArray(mainzicounter_mem_15037),
                      is_blocking=synchronous)
    self.mainzicounter_mem_15037 = static_mem_15075
  def futhark_main(self, netting_mem_14835, swap_term_mem_14836,
                   payments_mem_14837, notional_mem_14838, n_14575, n_14576,
                   n_14577, n_14578, paths_14579, steps_14580, a_14585, b_14586,
                   sigma_14587, r0_14588):
    dim_match_14589 = (n_14575 == n_14576)
    empty_or_match_cert_14590 = True
    assert dim_match_14589, ("Error: %s\n\nBacktrace:\n-> #0  cva.fut:97:1-153:18\n" % ("function arguments of wrong shape",))
    dim_match_14591 = (n_14575 == n_14577)
    empty_or_match_cert_14592 = True
    assert dim_match_14591, ("Error: %s\n\nBacktrace:\n-> #0  cva.fut:97:1-153:18\n" % ("function arguments of wrong shape",))
    segred_group_sizze_14660 = self.sizes["main.segred_group_size_14659"]
    max_num_groups_14874 = self.sizes["main.segred_num_groups_14661"]
    num_groups_14662 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(n_14575,
                                                            sext_i32_i64(segred_group_sizze_14660)),
                                                  sext_i32_i64(max_num_groups_14874))))
    mem_14841 = opencl_alloc(self, np.int64(4), "mem_14841")
    mainzicounter_mem_14875 = self.mainzicounter_mem_14875
    group_res_arr_mem_14877 = opencl_alloc(self,
                                           (np.int32(4) * (segred_group_sizze_14660 * num_groups_14662)),
                                           "group_res_arr_mem_14877")
    num_threads_14879 = (num_groups_14662 * segred_group_sizze_14660)
    if ((1 * (np.long(num_groups_14662) * np.long(segred_group_sizze_14660))) != 0):
      self.mainzisegred_nonseg_14667_var.set_args(self.global_failure,
                                                  cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14660))),
                                                  cl.LocalMemory(np.long(np.int32(1))),
                                                  np.int64(n_14575),
                                                  np.int64(num_groups_14662),
                                                  swap_term_mem_14836,
                                                  payments_mem_14837, mem_14841,
                                                  mainzicounter_mem_14875,
                                                  group_res_arr_mem_14877,
                                                  np.int64(num_threads_14879))
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegred_nonseg_14667_var,
                                 ((np.long(num_groups_14662) * np.long(segred_group_sizze_14660)),),
                                 (np.long(segred_group_sizze_14660),))
      if synchronous:
        sync(self)
    read_res_15068 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_15068, mem_14841,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_14595 = read_res_15068[0]
    mem_14841 = None
    res_14603 = sitofp_i64_f32(steps_14580)
    sims_per_year_14604 = (res_14603 / res_14595)
    bounds_invalid_upwards_14605 = slt64(steps_14580, np.int64(1))
    valid_14606 = not(bounds_invalid_upwards_14605)
    range_valid_c_14607 = True
    assert valid_14606, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:61:56-67\n   #1  cva.fut:108:17-44\n   #2  cva.fut:97:1-153:18\n" % ("Range ",
                                                                                                                                                   np.int64(1),
                                                                                                                                                   "..",
                                                                                                                                                   np.int64(2),
                                                                                                                                                   "...",
                                                                                                                                                   steps_14580,
                                                                                                                                                   " is invalid."))
    mem_14843 = opencl_alloc(self, np.int64(8), "mem_14843")
    mainzistatic_array_14905 = self.mainzistatic_array_14905
    if ((np.int64(2) * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_14843, mainzistatic_array_14905,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((np.int64(2) * np.int32(4))))
    if synchronous:
      sync(self)
    suff_outer_redomap_14669 = (self.sizes["main.suff_outer_redomap_0"] <= np.int64(2))
    segred_group_sizze_14681 = self.sizes["main.segred_group_size_14671"]
    max_num_groups_14906 = self.sizes["main.segred_num_groups_14673"]
    num_groups_14682 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(np.int64(2),
                                                            sext_i32_i64(segred_group_sizze_14681)),
                                                  sext_i32_i64(max_num_groups_14906))))
    suff_outer_par_14759 = (self.sizes["main.suff_outer_par_1"] <= np.int64(2))
    segmap_group_sizze_14762 = self.sizes["main.segmap_group_size_14702"]
    nest_sizze_14772 = (np.int64(2) * n_14575)
    segred_group_sizze_14773 = self.sizes["main.segred_group_size_14724"]
    max_num_groups_14907 = self.sizes["main.segred_num_groups_14726"]
    num_groups_14774 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_14772,
                                                            sext_i32_i64(segred_group_sizze_14773)),
                                                  sext_i32_i64(max_num_groups_14907))))
    segred_group_sizze_14784 = self.sizes["main.segred_group_size_14749"]
    max_num_groups_14908 = self.sizes["main.segred_num_groups_14751"]
    num_groups_14785 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(np.int64(2),
                                                            sext_i32_i64(segred_group_sizze_14784)),
                                                  sext_i32_i64(max_num_groups_14908))))
    loop_nonempty_14817 = slt64(np.int64(0), n_14575)
    converted_sizze_14823 = sitofp_i64_f32(n_14575)
    local_memory_capacity_15035 = self.max_local_memory
    if (sle64((np.int32(1) + (np.int32(4) * segred_group_sizze_14681)),
              sext_i32_i64(local_memory_capacity_15035)) and suff_outer_redomap_14669):
      mem_14846 = opencl_alloc(self, np.int64(4), "mem_14846")
      mainzicounter_mem_14909 = self.mainzicounter_mem_14909
      group_res_arr_mem_14911 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_14681 * num_groups_14682)),
                                             "group_res_arr_mem_14911")
      num_threads_14913 = (num_groups_14682 * segred_group_sizze_14681)
      if ((1 * (np.long(num_groups_14682) * np.long(segred_group_sizze_14681))) != 0):
        self.mainzisegred_nonseg_14679_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14681))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(num_groups_14682),
                                                    np.byte(loop_nonempty_14817),
                                                    np.float32(converted_sizze_14823),
                                                    mem_14843, mem_14846,
                                                    mainzicounter_mem_14909,
                                                    group_res_arr_mem_14911,
                                                    np.int64(num_threads_14913))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_14679_var,
                                   ((np.long(num_groups_14682) * np.long(segred_group_sizze_14681)),),
                                   (np.long(segred_group_sizze_14681),))
        if synchronous:
          sync(self)
      read_res_15071 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_15071, mem_14846,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_14694 = read_res_15071[0]
      mem_14846 = None
      res_14612 = res_14694
    else:
      local_memory_capacity_15004 = self.max_local_memory
      if (sle64(np.int64(0),
                sext_i32_i64(local_memory_capacity_15004)) and suff_outer_par_14759):
        segmap_usable_groups_14763 = sdiv_up64(np.int64(2),
                                               segmap_group_sizze_14762)
        mem_14849 = opencl_alloc(self, np.int64(8), "mem_14849")
        if ((1 * (np.long(segmap_usable_groups_14763) * np.long(segmap_group_sizze_14762))) != 0):
          self.mainzisegmap_14700_var.set_args(self.global_failure,
                                               np.byte(loop_nonempty_14817),
                                               np.float32(converted_sizze_14823),
                                               mem_14843, mem_14849)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_14700_var,
                                     ((np.long(segmap_usable_groups_14763) * np.long(segmap_group_sizze_14762)),),
                                     (np.long(segmap_group_sizze_14762),))
          if synchronous:
            sync(self)
        res_map_acc_mem_14853 = mem_14849
      else:
        mem_14852 = opencl_alloc(self, np.int64(8), "mem_14852")
        if slt64((n_14575 * np.int64(2)), segred_group_sizze_14773):
          segment_sizze_nonzzero_14944 = smax64(np.int64(1), n_14575)
          num_threads_14945 = (num_groups_14774 * segred_group_sizze_14773)
          if ((1 * (np.long(num_groups_14774) * np.long(segred_group_sizze_14773))) != 0):
            self.mainzisegred_small_14730_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14773))),
                                                       np.int64(n_14575),
                                                       np.int64(num_groups_14774),
                                                       mem_14843, mem_14852,
                                                       np.int64(segment_sizze_nonzzero_14944))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_14730_var,
                                       ((np.long(num_groups_14774) * np.long(segred_group_sizze_14773)),),
                                       (np.long(segred_group_sizze_14773),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_14965 = sdiv_up64(num_groups_14774, np.int64(2))
          elements_per_thread_14966 = sdiv_up64(n_14575,
                                                (segred_group_sizze_14773 * groups_per_segment_14965))
          virt_num_groups_14967 = (groups_per_segment_14965 * np.int64(2))
          num_threads_14968 = (num_groups_14774 * segred_group_sizze_14773)
          threads_per_segment_14969 = (groups_per_segment_14965 * segred_group_sizze_14773)
          group_res_arr_mem_14970 = opencl_alloc(self,
                                                 (np.int32(4) * (segred_group_sizze_14773 * virt_num_groups_14967)),
                                                 "group_res_arr_mem_14970")
          mainzicounter_mem_14972 = self.mainzicounter_mem_14972
          if ((1 * (np.long(num_groups_14774) * np.long(segred_group_sizze_14773))) != 0):
            self.mainzisegred_large_14730_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14773))),
                                                       np.int64(n_14575),
                                                       np.int64(num_groups_14774),
                                                       mem_14843, mem_14852,
                                                       np.int64(groups_per_segment_14965),
                                                       np.int64(elements_per_thread_14966),
                                                       np.int64(virt_num_groups_14967),
                                                       np.int64(threads_per_segment_14969),
                                                       group_res_arr_mem_14970,
                                                       mainzicounter_mem_14972)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_14730_var,
                                       ((np.long(num_groups_14774) * np.long(segred_group_sizze_14773)),),
                                       (np.long(segred_group_sizze_14773),))
            if synchronous:
              sync(self)
        res_map_acc_mem_14853 = mem_14852
      mem_14856 = opencl_alloc(self, np.int64(4), "mem_14856")
      mainzicounter_mem_15005 = self.mainzicounter_mem_15005
      group_res_arr_mem_15007 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_14784 * num_groups_14785)),
                                             "group_res_arr_mem_15007")
      num_threads_15009 = (num_groups_14785 * segred_group_sizze_14784)
      if ((1 * (np.long(num_groups_14785) * np.long(segred_group_sizze_14784))) != 0):
        self.mainzisegred_nonseg_14757_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14784))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(num_groups_14785),
                                                    res_map_acc_mem_14853,
                                                    mem_14856,
                                                    mainzicounter_mem_15005,
                                                    group_res_arr_mem_15007,
                                                    np.int64(num_threads_15009))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_14757_var,
                                   ((np.long(num_groups_14785) * np.long(segred_group_sizze_14784)),),
                                   (np.long(segred_group_sizze_14784),))
        if synchronous:
          sync(self)
      res_map_acc_mem_14853 = None
      read_res_15074 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_15074, mem_14856,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_14792 = read_res_15074[0]
      mem_14856 = None
      res_14612 = res_14792
    mem_14843 = None
    res_14622 = sitofp_i64_f32(paths_14579)
    res_14623 = (res_14612 / res_14622)
    x_14624 = fpow32(a_14585, np.float32(2.0))
    x_14625 = (b_14586 * x_14624)
    x_14626 = fpow32(sigma_14587, np.float32(2.0))
    y_14627 = (x_14626 / np.float32(2.0))
    y_14628 = (x_14625 - y_14627)
    y_14629 = (np.float32(4.0) * a_14585)
    segred_group_sizze_14796 = self.sizes["main.segred_group_size_14795"]
    max_num_groups_15036 = self.sizes["main.segred_num_groups_14797"]
    num_groups_14798 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(steps_14580,
                                                            sext_i32_i64(segred_group_sizze_14796)),
                                                  sext_i32_i64(max_num_groups_15036))))
    mem_14859 = opencl_alloc(self, np.int64(4), "mem_14859")
    bytes_14860 = (np.int64(4) * steps_14580)
    mem_14861 = opencl_alloc(self, bytes_14860, "mem_14861")
    mainzicounter_mem_15037 = self.mainzicounter_mem_15037
    group_res_arr_mem_15039 = opencl_alloc(self,
                                           (np.int32(4) * (segred_group_sizze_14796 * num_groups_14798)),
                                           "group_res_arr_mem_15039")
    num_threads_15041 = (num_groups_14798 * segred_group_sizze_14796)
    if ((1 * (np.long(num_groups_14798) * np.long(segred_group_sizze_14796))) != 0):
      self.mainzisegred_nonseg_14804_var.set_args(self.global_failure,
                                                  cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14796))),
                                                  cl.LocalMemory(np.long(np.int32(1))),
                                                  np.int64(steps_14580),
                                                  np.float32(a_14585),
                                                  np.float32(sims_per_year_14604),
                                                  np.float32(res_14623),
                                                  np.float32(x_14624),
                                                  np.float32(x_14626),
                                                  np.float32(y_14628),
                                                  np.float32(y_14629),
                                                  np.int64(num_groups_14798),
                                                  mem_14859, mem_14861,
                                                  mainzicounter_mem_15037,
                                                  group_res_arr_mem_15039,
                                                  np.int64(num_threads_15041))
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegred_nonseg_14804_var,
                                 ((np.long(num_groups_14798) * np.long(segred_group_sizze_14796)),),
                                 (np.long(segred_group_sizze_14796),))
      if synchronous:
        sync(self)
    read_res_15076 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_15076, mem_14859,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_14630 = read_res_15076[0]
    mem_14859 = None
    CVA_14656 = (np.float32(6.000000052154064e-3) * res_14630)
    mem_14863 = opencl_alloc(self, bytes_14860, "mem_14863")
    if ((steps_14580 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_14863, mem_14861,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((steps_14580 * np.int32(4))))
    if synchronous:
      sync(self)
    mem_14861 = None
    out_arrsizze_14873 = steps_14580
    out_mem_14872 = mem_14863
    scalar_out_14871 = CVA_14656
    return (scalar_out_14871, out_mem_14872, out_arrsizze_14873)
  def main(self, paths_14579_ext, steps_14580_ext, netting_mem_14835_ext,
           swap_term_mem_14836_ext, payments_mem_14837_ext,
           notional_mem_14838_ext, a_14585_ext, b_14586_ext, sigma_14587_ext,
           r0_14588_ext):
    try:
      paths_14579 = np.int64(ct.c_int64(paths_14579_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(paths_14579_ext),
                                                                                                                            paths_14579_ext))
    try:
      steps_14580 = np.int64(ct.c_int64(steps_14580_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(steps_14580_ext),
                                                                                                                            steps_14580_ext))
    try:
      assert ((type(netting_mem_14835_ext) in [np.ndarray,
                                               cl.array.Array]) and (netting_mem_14835_ext.dtype == np.int64)), "Parameter has unexpected type"
      n_14575 = np.int32(netting_mem_14835_ext.shape[0])
      if (type(netting_mem_14835_ext) == cl.array.Array):
        netting_mem_14835 = netting_mem_14835_ext.data
      else:
        netting_mem_14835 = opencl_alloc(self,
                                         np.int64(netting_mem_14835_ext.nbytes),
                                         "netting_mem_14835")
        if (np.int64(netting_mem_14835_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, netting_mem_14835,
                          normaliseArray(netting_mem_14835_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i64",
                                                                                                                            type(netting_mem_14835_ext),
                                                                                                                            netting_mem_14835_ext))
    try:
      assert ((type(swap_term_mem_14836_ext) in [np.ndarray,
                                                 cl.array.Array]) and (swap_term_mem_14836_ext.dtype == np.float32)), "Parameter has unexpected type"
      n_14576 = np.int32(swap_term_mem_14836_ext.shape[0])
      if (type(swap_term_mem_14836_ext) == cl.array.Array):
        swap_term_mem_14836 = swap_term_mem_14836_ext.data
      else:
        swap_term_mem_14836 = opencl_alloc(self,
                                           np.int64(swap_term_mem_14836_ext.nbytes),
                                           "swap_term_mem_14836")
        if (np.int64(swap_term_mem_14836_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, swap_term_mem_14836,
                          normaliseArray(swap_term_mem_14836_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(swap_term_mem_14836_ext),
                                                                                                                            swap_term_mem_14836_ext))
    try:
      assert ((type(payments_mem_14837_ext) in [np.ndarray,
                                                cl.array.Array]) and (payments_mem_14837_ext.dtype == np.int64)), "Parameter has unexpected type"
      n_14577 = np.int32(payments_mem_14837_ext.shape[0])
      if (type(payments_mem_14837_ext) == cl.array.Array):
        payments_mem_14837 = payments_mem_14837_ext.data
      else:
        payments_mem_14837 = opencl_alloc(self,
                                          np.int64(payments_mem_14837_ext.nbytes),
                                          "payments_mem_14837")
        if (np.int64(payments_mem_14837_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, payments_mem_14837,
                          normaliseArray(payments_mem_14837_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i64",
                                                                                                                            type(payments_mem_14837_ext),
                                                                                                                            payments_mem_14837_ext))
    try:
      assert ((type(notional_mem_14838_ext) in [np.ndarray,
                                                cl.array.Array]) and (notional_mem_14838_ext.dtype == np.float32)), "Parameter has unexpected type"
      n_14578 = np.int32(notional_mem_14838_ext.shape[0])
      if (type(notional_mem_14838_ext) == cl.array.Array):
        notional_mem_14838 = notional_mem_14838_ext.data
      else:
        notional_mem_14838 = opencl_alloc(self,
                                          np.int64(notional_mem_14838_ext.nbytes),
                                          "notional_mem_14838")
        if (np.int64(notional_mem_14838_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, notional_mem_14838,
                          normaliseArray(notional_mem_14838_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(notional_mem_14838_ext),
                                                                                                                            notional_mem_14838_ext))
    try:
      a_14585 = np.float32(ct.c_float(a_14585_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_14585_ext),
                                                                                                                            a_14585_ext))
    try:
      b_14586 = np.float32(ct.c_float(b_14586_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_14586_ext),
                                                                                                                            b_14586_ext))
    try:
      sigma_14587 = np.float32(ct.c_float(sigma_14587_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_14587_ext),
                                                                                                                            sigma_14587_ext))
    try:
      r0_14588 = np.float32(ct.c_float(r0_14588_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #9 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_14588_ext),
                                                                                                                            r0_14588_ext))
    (scalar_out_14871, out_mem_14872,
     out_arrsizze_14873) = self.futhark_main(netting_mem_14835,
                                             swap_term_mem_14836,
                                             payments_mem_14837,
                                             notional_mem_14838, n_14575,
                                             n_14576, n_14577, n_14578,
                                             paths_14579, steps_14580, a_14585,
                                             b_14586, sigma_14587, r0_14588)
    sync(self)
    return (np.float32(scalar_out_14871), cl.array.Array(self.queue,
                                                         (out_arrsizze_14873,),
                                                         ct.c_float,
                                                         data=out_mem_14872))