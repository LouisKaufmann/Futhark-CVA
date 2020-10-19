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




__kernel void gpu_map_transpose_f32(__local volatile
                                    int64_t *block_9_backing_aligned_0,
                                    int32_t destoffset_1, int32_t srcoffset_3,
                                    int32_t num_arrays_4, int32_t x_elems_5,
                                    int32_t y_elems_6, int32_t mulx_7,
                                    int32_t muly_8, __global
                                    unsigned char *destmem_0, __global
                                    unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
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
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local float *) block_9)[sext_i32_i64((get_local_id_1_39 +
                                                          j_43 * 8) * 33 +
                                            get_local_id_0_38)] = ((__global
                                                                    float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                       index_in_35)];
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
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                               index_out_36)] = ((__local
                                                                  float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                    33 +
                                                                                    get_local_id_1_39 +
                                                                                    j_43 *
                                                                                    8)];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_height(__local volatile
                                               int64_t *block_9_backing_aligned_0,
                                               int32_t destoffset_1,
                                               int32_t srcoffset_3,
                                               int32_t num_arrays_4,
                                               int32_t x_elems_5,
                                               int32_t y_elems_6,
                                               int32_t mulx_7, int32_t muly_8,
                                               __global
                                               unsigned char *destmem_0,
                                               __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
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
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_width(__local volatile
                                              int64_t *block_9_backing_aligned_0,
                                              int32_t destoffset_1,
                                              int32_t srcoffset_3,
                                              int32_t num_arrays_4,
                                              int32_t x_elems_5,
                                              int32_t y_elems_6, int32_t mulx_7,
                                              int32_t muly_8, __global
                                              unsigned char *destmem_0, __global
                                              unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
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
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_small(__local volatile
                                          int64_t *block_9_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t mulx_7, int32_t muly_8,
                                          __global unsigned char *destmem_0,
                                          __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
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
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__global
                                                          float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                             index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void mainzisegmap_14120(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args, int64_t n_13650,
                                 float a_13658, float r0_13661, float x_13678,
                                 float x_13680, float y_13682, float y_13683,
                                 __global unsigned char *swap_term_mem_16319,
                                 __global unsigned char *payments_mem_16320,
                                 __global unsigned char *mem_16333, __global
                                 unsigned char *mem_16347,
                                 int64_t num_threads_16439)
{
    #define segmap_group_sizze_14198 (mainzisegmap_group_sizze_14122)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16577;
    int32_t local_tid_16578;
    int64_t group_sizze_16581;
    int32_t wave_sizze_16580;
    int32_t group_tid_16579;
    
    global_tid_16577 = get_global_id(0);
    local_tid_16578 = get_local_id(0);
    group_sizze_16581 = get_local_size(0);
    wave_sizze_16580 = LOCKSTEP_WIDTH;
    group_tid_16579 = get_group_id(0);
    
    int32_t phys_tid_14120;
    
    phys_tid_14120 = global_tid_16577;
    
    int64_t gtid_14119;
    
    gtid_14119 = sext_i32_i64(group_tid_16579) * segmap_group_sizze_14198 +
        sext_i32_i64(local_tid_16578);
    if (slt64(gtid_14119, n_13650)) {
        float res_14209 = ((__global float *) swap_term_mem_16319)[gtid_14119];
        int64_t res_14210 = ((__global
                              int64_t *) payments_mem_16320)[gtid_14119];
        int64_t range_end_14212 = sub64(res_14210, 1);
        bool bounds_invalid_upwards_14213 = slt64(range_end_14212, 0);
        bool valid_14214 = !bounds_invalid_upwards_14213;
        bool range_valid_c_14215;
        
        if (!valid_14214) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = 1;
                    global_failure_args[2] = range_end_14212;
                    ;
                }
                return;
            }
        }
        for (int64_t i_16293 = 0; i_16293 < res_14210; i_16293++) {
            float res_14219 = sitofp_i64_f32(i_16293);
            float res_14220 = res_14209 * res_14219;
            
            ((__global float *) mem_16333)[phys_tid_14120 + i_16293 *
                                           num_threads_16439] = res_14220;
        }
        
        bool y_14221 = slt64(0, res_14210);
        bool index_certs_14222;
        
        if (!y_14221) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 1) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = res_14210;
                    ;
                }
                return;
            }
        }
        
        float binop_y_14223 = sitofp_i64_f32(range_end_14212);
        float index_primexp_14224 = res_14209 * binop_y_14223;
        float negate_arg_14225 = a_13658 * index_primexp_14224;
        float exp_arg_14226 = 0.0F - negate_arg_14225;
        float res_14227 = fpow32(2.7182817F, exp_arg_14226);
        float x_14228 = 1.0F - res_14227;
        float B_14229 = x_14228 / a_13658;
        float x_14230 = B_14229 - index_primexp_14224;
        float x_14231 = y_13682 * x_14230;
        float A1_14232 = x_14231 / x_13678;
        float y_14233 = fpow32(B_14229, 2.0F);
        float x_14234 = x_13680 * y_14233;
        float A2_14235 = x_14234 / y_13683;
        float exp_arg_14236 = A1_14232 - A2_14235;
        float res_14237 = fpow32(2.7182817F, exp_arg_14236);
        float negate_arg_14238 = r0_13661 * B_14229;
        float exp_arg_14239 = 0.0F - negate_arg_14238;
        float res_14240 = fpow32(2.7182817F, exp_arg_14239);
        float res_14241 = res_14237 * res_14240;
        bool empty_slice_14242 = range_end_14212 == 0;
        bool zzero_leq_i_p_m_t_s_14243 = sle64(0, range_end_14212);
        bool i_p_m_t_s_leq_w_14244 = slt64(range_end_14212, res_14210);
        bool i_lte_j_14245 = sle64(1, res_14210);
        bool y_14246 = zzero_leq_i_p_m_t_s_14243 && i_p_m_t_s_leq_w_14244;
        bool y_14247 = i_lte_j_14245 && y_14246;
        bool ok_or_empty_14248 = empty_slice_14242 || y_14247;
        bool index_certs_14249;
        
        if (!ok_or_empty_14248) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 2) == -1) {
                    global_failure_args[0] = 1;
                    global_failure_args[1] = res_14210;
                    ;
                }
                return;
            }
        }
        
        float res_14251;
        float redout_16295 = 0.0F;
        
        for (int64_t i_16296 = 0; i_16296 < range_end_14212; i_16296++) {
            int64_t slice_16309 = 1 + i_16296;
            float x_14255 = ((__global float *) mem_16333)[phys_tid_14120 +
                                                           slice_16309 *
                                                           num_threads_16439];
            float negate_arg_14256 = a_13658 * x_14255;
            float exp_arg_14257 = 0.0F - negate_arg_14256;
            float res_14258 = fpow32(2.7182817F, exp_arg_14257);
            float x_14259 = 1.0F - res_14258;
            float B_14260 = x_14259 / a_13658;
            float x_14261 = B_14260 - x_14255;
            float x_14262 = y_13682 * x_14261;
            float A1_14263 = x_14262 / x_13678;
            float y_14264 = fpow32(B_14260, 2.0F);
            float x_14265 = x_13680 * y_14264;
            float A2_14266 = x_14265 / y_13683;
            float exp_arg_14267 = A1_14263 - A2_14266;
            float res_14268 = fpow32(2.7182817F, exp_arg_14267);
            float negate_arg_14269 = r0_13661 * B_14260;
            float exp_arg_14270 = 0.0F - negate_arg_14269;
            float res_14271 = fpow32(2.7182817F, exp_arg_14270);
            float res_14272 = res_14268 * res_14271;
            float res_14254 = res_14272 + redout_16295;
            float redout_tmp_16583 = res_14254;
            
            redout_16295 = redout_tmp_16583;
        }
        res_14251 = redout_16295;
        
        float x_14273 = 1.0F - res_14241;
        float y_14274 = res_14209 * res_14251;
        float res_14275 = x_14273 / y_14274;
        
        ((__global float *) mem_16347)[gtid_14119] = res_14275;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14198
}
__kernel void mainzisegmap_14296(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args, int64_t n_13650,
                                 float a_13658, float r0_13661, float x_13678,
                                 float x_13680, float y_13682, float y_13683,
                                 __global unsigned char *swap_term_mem_16319,
                                 __global unsigned char *payments_mem_16320,
                                 __global unsigned char *mem_16356, __global
                                 unsigned char *mem_16370,
                                 int64_t num_threads_16471)
{
    #define segmap_group_sizze_14374 (mainzisegmap_group_sizze_14298)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16614;
    int32_t local_tid_16615;
    int64_t group_sizze_16618;
    int32_t wave_sizze_16617;
    int32_t group_tid_16616;
    
    global_tid_16614 = get_global_id(0);
    local_tid_16615 = get_local_id(0);
    group_sizze_16618 = get_local_size(0);
    wave_sizze_16617 = LOCKSTEP_WIDTH;
    group_tid_16616 = get_group_id(0);
    
    int32_t phys_tid_14296;
    
    phys_tid_14296 = global_tid_16614;
    
    int64_t gtid_14295;
    
    gtid_14295 = sext_i32_i64(group_tid_16616) * segmap_group_sizze_14374 +
        sext_i32_i64(local_tid_16615);
    if (slt64(gtid_14295, n_13650)) {
        float res_14385 = ((__global float *) swap_term_mem_16319)[gtid_14295];
        int64_t res_14386 = ((__global
                              int64_t *) payments_mem_16320)[gtid_14295];
        int64_t range_end_14388 = sub64(res_14386, 1);
        bool bounds_invalid_upwards_14389 = slt64(range_end_14388, 0);
        bool valid_14390 = !bounds_invalid_upwards_14389;
        bool range_valid_c_14391;
        
        if (!valid_14390) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 3) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = 1;
                    global_failure_args[2] = range_end_14388;
                    ;
                }
                return;
            }
        }
        for (int64_t i_16299 = 0; i_16299 < res_14386; i_16299++) {
            float res_14395 = sitofp_i64_f32(i_16299);
            float res_14396 = res_14385 * res_14395;
            
            ((__global float *) mem_16356)[phys_tid_14296 + i_16299 *
                                           num_threads_16471] = res_14396;
        }
        
        bool y_14397 = slt64(0, res_14386);
        bool index_certs_14398;
        
        if (!y_14397) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 4) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = res_14386;
                    ;
                }
                return;
            }
        }
        
        float binop_y_14399 = sitofp_i64_f32(range_end_14388);
        float index_primexp_14400 = res_14385 * binop_y_14399;
        float negate_arg_14401 = a_13658 * index_primexp_14400;
        float exp_arg_14402 = 0.0F - negate_arg_14401;
        float res_14403 = fpow32(2.7182817F, exp_arg_14402);
        float x_14404 = 1.0F - res_14403;
        float B_14405 = x_14404 / a_13658;
        float x_14406 = B_14405 - index_primexp_14400;
        float x_14407 = y_13682 * x_14406;
        float A1_14408 = x_14407 / x_13678;
        float y_14409 = fpow32(B_14405, 2.0F);
        float x_14410 = x_13680 * y_14409;
        float A2_14411 = x_14410 / y_13683;
        float exp_arg_14412 = A1_14408 - A2_14411;
        float res_14413 = fpow32(2.7182817F, exp_arg_14412);
        float negate_arg_14414 = r0_13661 * B_14405;
        float exp_arg_14415 = 0.0F - negate_arg_14414;
        float res_14416 = fpow32(2.7182817F, exp_arg_14415);
        float res_14417 = res_14413 * res_14416;
        bool empty_slice_14418 = range_end_14388 == 0;
        bool zzero_leq_i_p_m_t_s_14419 = sle64(0, range_end_14388);
        bool i_p_m_t_s_leq_w_14420 = slt64(range_end_14388, res_14386);
        bool i_lte_j_14421 = sle64(1, res_14386);
        bool y_14422 = zzero_leq_i_p_m_t_s_14419 && i_p_m_t_s_leq_w_14420;
        bool y_14423 = i_lte_j_14421 && y_14422;
        bool ok_or_empty_14424 = empty_slice_14418 || y_14423;
        bool index_certs_14425;
        
        if (!ok_or_empty_14424) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 5) == -1) {
                    global_failure_args[0] = 1;
                    global_failure_args[1] = res_14386;
                    ;
                }
                return;
            }
        }
        
        float res_14427;
        float redout_16301 = 0.0F;
        
        for (int64_t i_16302 = 0; i_16302 < range_end_14388; i_16302++) {
            int64_t slice_16312 = 1 + i_16302;
            float x_14431 = ((__global float *) mem_16356)[phys_tid_14296 +
                                                           slice_16312 *
                                                           num_threads_16471];
            float negate_arg_14432 = a_13658 * x_14431;
            float exp_arg_14433 = 0.0F - negate_arg_14432;
            float res_14434 = fpow32(2.7182817F, exp_arg_14433);
            float x_14435 = 1.0F - res_14434;
            float B_14436 = x_14435 / a_13658;
            float x_14437 = B_14436 - x_14431;
            float x_14438 = y_13682 * x_14437;
            float A1_14439 = x_14438 / x_13678;
            float y_14440 = fpow32(B_14436, 2.0F);
            float x_14441 = x_13680 * y_14440;
            float A2_14442 = x_14441 / y_13683;
            float exp_arg_14443 = A1_14439 - A2_14442;
            float res_14444 = fpow32(2.7182817F, exp_arg_14443);
            float negate_arg_14445 = r0_13661 * B_14436;
            float exp_arg_14446 = 0.0F - negate_arg_14445;
            float res_14447 = fpow32(2.7182817F, exp_arg_14446);
            float res_14448 = res_14444 * res_14447;
            float res_14430 = res_14448 + redout_16301;
            float redout_tmp_16620 = res_14430;
            
            redout_16301 = redout_tmp_16620;
        }
        res_14427 = redout_16301;
        
        float x_14449 = 1.0F - res_14417;
        float y_14450 = res_14385 * res_14427;
        float res_14451 = x_14449 / y_14450;
        
        ((__global float *) mem_16370)[gtid_14295] = res_14451;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14374
}
__kernel void mainzisegmap_14629(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args,
                                 int64_t paths_13653, int64_t steps_13654,
                                 float a_13658, float b_13659,
                                 float sigma_13660, float r0_13661,
                                 float dt_13676, int64_t upper_bound_13781,
                                 float res_13782, int64_t num_groups_14881,
                                 __global unsigned char *mem_16384, __global
                                 unsigned char *mem_16387, __global
                                 unsigned char *mem_16402)
{
    #define segmap_group_sizze_14880 (mainzisegmap_group_sizze_14631)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16633;
    int32_t local_tid_16634;
    int64_t group_sizze_16637;
    int32_t wave_sizze_16636;
    int32_t group_tid_16635;
    
    global_tid_16633 = get_global_id(0);
    local_tid_16634 = get_local_id(0);
    group_sizze_16637 = get_local_size(0);
    wave_sizze_16636 = LOCKSTEP_WIDTH;
    group_tid_16635 = get_group_id(0);
    
    int32_t phys_tid_14629;
    
    phys_tid_14629 = global_tid_16633;
    
    int32_t phys_group_id_16638;
    
    phys_group_id_16638 = get_group_id(0);
    for (int32_t i_16639 = 0; i_16639 <
         sdiv_up32(sext_i64_i32(sdiv_up64(paths_13653,
                                          segmap_group_sizze_14880)) -
                   phys_group_id_16638, sext_i64_i32(num_groups_14881));
         i_16639++) {
        int32_t virt_group_id_16640 = phys_group_id_16638 + i_16639 *
                sext_i64_i32(num_groups_14881);
        int64_t gtid_14628 = sext_i32_i64(virt_group_id_16640) *
                segmap_group_sizze_14880 + sext_i32_i64(local_tid_16634);
        
        if (slt64(gtid_14628, paths_13653)) {
            for (int64_t i_16641 = 0; i_16641 < steps_13654; i_16641++) {
                ((__global float *) mem_16387)[phys_tid_14629 + i_16641 *
                                               (num_groups_14881 *
                                                segmap_group_sizze_14880)] =
                    r0_13661;
            }
            for (int64_t i_14887 = 0; i_14887 < upper_bound_13781; i_14887++) {
                bool y_14889 = slt64(i_14887, steps_13654);
                bool index_certs_14890;
                
                if (!y_14889) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 6) ==
                            -1) {
                            global_failure_args[0] = i_14887;
                            global_failure_args[1] = steps_13654;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float shortstep_arg_14891 = ((__global
                                              float *) mem_16384)[i_14887 *
                                                                  paths_13653 +
                                                                  gtid_14628];
                float shortstep_arg_14892 = ((__global
                                              float *) mem_16387)[phys_tid_14629 +
                                                                  i_14887 *
                                                                  (num_groups_14881 *
                                                                   segmap_group_sizze_14880)];
                float y_14893 = b_13659 - shortstep_arg_14892;
                float x_14894 = a_13658 * y_14893;
                float x_14895 = dt_13676 * x_14894;
                float x_14896 = res_13782 * shortstep_arg_14891;
                float y_14897 = sigma_13660 * x_14896;
                float delta_r_14898 = x_14895 + y_14897;
                float res_14899 = shortstep_arg_14892 + delta_r_14898;
                int64_t i_14900 = add64(1, i_14887);
                bool x_14901 = sle64(0, i_14900);
                bool y_14902 = slt64(i_14900, steps_13654);
                bool bounds_check_14903 = x_14901 && y_14902;
                bool index_certs_14904;
                
                if (!bounds_check_14903) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 7) ==
                            -1) {
                            global_failure_args[0] = i_14900;
                            global_failure_args[1] = steps_13654;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                ((__global float *) mem_16387)[phys_tid_14629 + i_14900 *
                                               (num_groups_14881 *
                                                segmap_group_sizze_14880)] =
                    res_14899;
            }
            for (int64_t i_16643 = 0; i_16643 < steps_13654; i_16643++) {
                ((__global float *) mem_16402)[i_16643 * paths_13653 +
                                               gtid_14628] = ((__global
                                                               float *) mem_16387)[phys_tid_14629 +
                                                                                   i_16643 *
                                                                                   (num_groups_14881 *
                                                                                    segmap_group_sizze_14880)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14880
}
__kernel void mainzisegmap_14727(__global int *global_failure,
                                 int64_t paths_13653, int64_t steps_13654,
                                 __global unsigned char *mem_16377, __global
                                 unsigned char *mem_16381)
{
    #define segmap_group_sizze_14835 (mainzisegmap_group_sizze_14730)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16627;
    int32_t local_tid_16628;
    int64_t group_sizze_16631;
    int32_t wave_sizze_16630;
    int32_t group_tid_16629;
    
    global_tid_16627 = get_global_id(0);
    local_tid_16628 = get_local_id(0);
    group_sizze_16631 = get_local_size(0);
    wave_sizze_16630 = LOCKSTEP_WIDTH;
    group_tid_16629 = get_group_id(0);
    
    int32_t phys_tid_14727;
    
    phys_tid_14727 = global_tid_16627;
    
    int64_t gtid_14725;
    
    gtid_14725 = squot64(sext_i32_i64(group_tid_16629) *
                         segmap_group_sizze_14835 +
                         sext_i32_i64(local_tid_16628), steps_13654);
    
    int64_t gtid_14726;
    
    gtid_14726 = sext_i32_i64(group_tid_16629) * segmap_group_sizze_14835 +
        sext_i32_i64(local_tid_16628) - squot64(sext_i32_i64(group_tid_16629) *
                                                segmap_group_sizze_14835 +
                                                sext_i32_i64(local_tid_16628),
                                                steps_13654) * steps_13654;
    if (slt64(gtid_14725, paths_13653) && slt64(gtid_14726, steps_13654)) {
        int32_t unsign_arg_14838 = ((__global int32_t *) mem_16377)[gtid_14725];
        int32_t res_14840 = sext_i64_i32(gtid_14726);
        int32_t x_14841 = lshr32(res_14840, 16);
        int32_t x_14842 = res_14840 ^ x_14841;
        int32_t x_14843 = mul32(73244475, x_14842);
        int32_t x_14844 = lshr32(x_14843, 16);
        int32_t x_14845 = x_14843 ^ x_14844;
        int32_t x_14846 = mul32(73244475, x_14845);
        int32_t x_14847 = lshr32(x_14846, 16);
        int32_t x_14848 = x_14846 ^ x_14847;
        int32_t unsign_arg_14849 = unsign_arg_14838 ^ x_14848;
        int32_t unsign_arg_14850 = mul32(48271, unsign_arg_14849);
        int32_t unsign_arg_14851 = umod32(unsign_arg_14850, 2147483647);
        int32_t unsign_arg_14852 = mul32(48271, unsign_arg_14851);
        int32_t unsign_arg_14853 = umod32(unsign_arg_14852, 2147483647);
        float res_14854 = uitofp_i32_f32(unsign_arg_14851);
        float res_14855 = res_14854 / 2.1474836e9F;
        float res_14856 = uitofp_i32_f32(unsign_arg_14853);
        float res_14857 = res_14856 / 2.1474836e9F;
        float res_14858;
        
        res_14858 = futrts_log32(res_14855);
        
        float res_14859 = -2.0F * res_14858;
        float res_14860;
        
        res_14860 = futrts_sqrt32(res_14859);
        
        float res_14861 = 6.2831855F * res_14857;
        float res_14862;
        
        res_14862 = futrts_cos32(res_14861);
        
        float res_14863 = res_14860 * res_14862;
        
        ((__global float *) mem_16381)[gtid_14725 * steps_13654 + gtid_14726] =
            res_14863;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14835
}
__kernel void mainzisegmap_14791(__global int *global_failure,
                                 int64_t paths_13653, __global
                                 unsigned char *mem_16377)
{
    #define segmap_group_sizze_14810 (mainzisegmap_group_sizze_14793)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16622;
    int32_t local_tid_16623;
    int64_t group_sizze_16626;
    int32_t wave_sizze_16625;
    int32_t group_tid_16624;
    
    global_tid_16622 = get_global_id(0);
    local_tid_16623 = get_local_id(0);
    group_sizze_16626 = get_local_size(0);
    wave_sizze_16625 = LOCKSTEP_WIDTH;
    group_tid_16624 = get_group_id(0);
    
    int32_t phys_tid_14791;
    
    phys_tid_14791 = global_tid_16622;
    
    int64_t gtid_14790;
    
    gtid_14790 = sext_i32_i64(group_tid_16624) * segmap_group_sizze_14810 +
        sext_i32_i64(local_tid_16623);
    if (slt64(gtid_14790, paths_13653)) {
        int32_t res_14814 = sext_i64_i32(gtid_14790);
        int32_t x_14815 = lshr32(res_14814, 16);
        int32_t x_14816 = res_14814 ^ x_14815;
        int32_t x_14817 = mul32(73244475, x_14816);
        int32_t x_14818 = lshr32(x_14817, 16);
        int32_t x_14819 = x_14817 ^ x_14818;
        int32_t x_14820 = mul32(73244475, x_14819);
        int32_t x_14821 = lshr32(x_14820, 16);
        int32_t x_14822 = x_14820 ^ x_14821;
        int32_t unsign_arg_14823 = 777822902 ^ x_14822;
        int32_t unsign_arg_14824 = mul32(48271, unsign_arg_14823);
        int32_t unsign_arg_14825 = umod32(unsign_arg_14824, 2147483647);
        
        ((__global int32_t *) mem_16377)[gtid_14790] = unsign_arg_14825;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14810
}
__kernel void mainzisegmap_15070(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args, int64_t n_13650,
                                 int64_t paths_13653, int64_t steps_13654,
                                 float a_13658, float b_13659,
                                 float sigma_13660, float x_13678,
                                 float x_13680, float y_13682, float y_13683,
                                 float sims_per_year_13760,
                                 float last_date_13775, float res_13844,
                                 __global unsigned char *res_mem_16371, __global
                                 unsigned char *res_mem_16372, __global
                                 unsigned char *res_mem_16373, __global
                                 unsigned char *res_mem_16374, __global
                                 unsigned char *mem_16402, __global
                                 unsigned char *mem_16410, __global
                                 unsigned char *mem_16412)
{
    #define segmap_group_sizze_15799 (mainzisegmap_group_sizze_15072)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16679;
    int32_t local_tid_16680;
    int64_t group_sizze_16683;
    int32_t wave_sizze_16682;
    int32_t group_tid_16681;
    
    global_tid_16679 = get_global_id(0);
    local_tid_16680 = get_local_id(0);
    group_sizze_16683 = get_local_size(0);
    wave_sizze_16682 = LOCKSTEP_WIDTH;
    group_tid_16681 = get_group_id(0);
    
    int32_t phys_tid_15070;
    
    phys_tid_15070 = global_tid_16679;
    
    int64_t gtid_15069;
    
    gtid_15069 = sext_i32_i64(group_tid_16681) * segmap_group_sizze_15799 +
        sext_i32_i64(local_tid_16680);
    if (slt64(gtid_15069, steps_13654)) {
        int64_t index_primexp_16242 = add64(1, gtid_15069);
        float res_15806 = sitofp_i64_f32(index_primexp_16242);
        float res_15807 = res_15806 / sims_per_year_13760;
        bool cond_15814 = last_date_13775 < res_15807;
        float res_15808;
        float redout_16305 = 0.0F;
        
        for (int64_t i_16306 = 0; i_16306 < paths_13653; i_16306++) {
            float max_arg_15815;
            
            if (cond_15814) {
                max_arg_15815 = 0.0F;
            } else {
                float x_15812 = ((__global float *) mem_16402)[gtid_15069 *
                                                               paths_13653 +
                                                               i_16306];
                bool y_15816 = slt64(0, n_13650);
                bool index_certs_15817;
                
                if (!y_15816) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 11) ==
                            -1) {
                            global_failure_args[0] = 0;
                            global_failure_args[1] = n_13650;
                            ;
                        }
                        return;
                    }
                }
                
                float swapprice_arg_15818 = ((__global
                                              float *) res_mem_16371)[0];
                float swapprice_arg_15819 = ((__global
                                              float *) res_mem_16372)[0];
                int64_t swapprice_arg_15820 = ((__global
                                                int64_t *) res_mem_16373)[0];
                float swapprice_arg_15821 = ((__global
                                              float *) res_mem_16374)[0];
                float ceil_arg_15822 = res_15807 / swapprice_arg_15821;
                float res_15823;
                
                res_15823 = futrts_ceil32(ceil_arg_15822);
                
                float nextpayment_15824 = swapprice_arg_15821 * res_15823;
                int64_t res_15825 = fptosi_f32_i64(res_15823);
                int64_t remaining_15826 = sub64(swapprice_arg_15820, res_15825);
                bool bounds_invalid_upwards_15827 = slt64(remaining_15826, 1);
                bool valid_15828 = !bounds_invalid_upwards_15827;
                bool range_valid_c_15829;
                
                if (!valid_15828) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 12) ==
                            -1) {
                            global_failure_args[0] = 1;
                            global_failure_args[1] = 2;
                            global_failure_args[2] = remaining_15826;
                            ;
                        }
                        return;
                    }
                }
                
                float y_15831 = nextpayment_15824 - res_15807;
                float negate_arg_15832 = a_13658 * y_15831;
                float exp_arg_15833 = 0.0F - negate_arg_15832;
                float res_15834 = fpow32(2.7182817F, exp_arg_15833);
                float x_15835 = 1.0F - res_15834;
                float B_15836 = x_15835 / a_13658;
                float x_15837 = B_15836 - nextpayment_15824;
                float x_15838 = res_15807 + x_15837;
                float x_15839 = fpow32(a_13658, 2.0F);
                float x_15840 = b_13659 * x_15839;
                float x_15841 = fpow32(sigma_13660, 2.0F);
                float y_15842 = x_15841 / 2.0F;
                float y_15843 = x_15840 - y_15842;
                float x_15844 = x_15838 * y_15843;
                float A1_15845 = x_15844 / x_15839;
                float y_15846 = fpow32(B_15836, 2.0F);
                float x_15847 = x_15841 * y_15846;
                float y_15848 = 4.0F * a_13658;
                float A2_15849 = x_15847 / y_15848;
                float exp_arg_15850 = A1_15845 - A2_15849;
                float res_15851 = fpow32(2.7182817F, exp_arg_15850);
                float negate_arg_15852 = x_15812 * B_15836;
                float exp_arg_15853 = 0.0F - negate_arg_15852;
                float res_15854 = fpow32(2.7182817F, exp_arg_15853);
                float res_15855 = res_15851 * res_15854;
                bool y_15856 = slt64(0, remaining_15826);
                bool index_certs_15857;
                
                if (!y_15856) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 13) ==
                            -1) {
                            global_failure_args[0] = 0;
                            global_failure_args[1] = remaining_15826;
                            ;
                        }
                        return;
                    }
                }
                
                float binop_y_15858 = sitofp_i64_f32(remaining_15826);
                float binop_y_15859 = swapprice_arg_15821 * binop_y_15858;
                float index_primexp_15860 = nextpayment_15824 + binop_y_15859;
                float y_15861 = index_primexp_15860 - res_15807;
                float negate_arg_15862 = a_13658 * y_15861;
                float exp_arg_15863 = 0.0F - negate_arg_15862;
                float res_15864 = fpow32(2.7182817F, exp_arg_15863);
                float x_15865 = 1.0F - res_15864;
                float B_15866 = x_15865 / a_13658;
                float x_15867 = B_15866 - index_primexp_15860;
                float x_15868 = res_15807 + x_15867;
                float x_15869 = y_15843 * x_15868;
                float A1_15870 = x_15869 / x_15839;
                float y_15871 = fpow32(B_15866, 2.0F);
                float x_15872 = x_15841 * y_15871;
                float A2_15873 = x_15872 / y_15848;
                float exp_arg_15874 = A1_15870 - A2_15873;
                float res_15875 = fpow32(2.7182817F, exp_arg_15874);
                float negate_arg_15876 = x_15812 * B_15866;
                float exp_arg_15877 = 0.0F - negate_arg_15876;
                float res_15878 = fpow32(2.7182817F, exp_arg_15877);
                float res_15879 = res_15875 * res_15878;
                float res_15880;
                float redout_16239 = 0.0F;
                
                for (int64_t i_16240 = 0; i_16240 < remaining_15826;
                     i_16240++) {
                    int64_t index_primexp_16272 = add64(1, i_16240);
                    float res_15885 = sitofp_i64_f32(index_primexp_16272);
                    float res_15886 = swapprice_arg_15821 * res_15885;
                    float res_15887 = nextpayment_15824 + res_15886;
                    float y_15888 = res_15887 - res_15807;
                    float negate_arg_15889 = a_13658 * y_15888;
                    float exp_arg_15890 = 0.0F - negate_arg_15889;
                    float res_15891 = fpow32(2.7182817F, exp_arg_15890);
                    float x_15892 = 1.0F - res_15891;
                    float B_15893 = x_15892 / a_13658;
                    float x_15894 = B_15893 - res_15887;
                    float x_15895 = res_15807 + x_15894;
                    float x_15896 = y_15843 * x_15895;
                    float A1_15897 = x_15896 / x_15839;
                    float y_15898 = fpow32(B_15893, 2.0F);
                    float x_15899 = x_15841 * y_15898;
                    float A2_15900 = x_15899 / y_15848;
                    float exp_arg_15901 = A1_15897 - A2_15900;
                    float res_15902 = fpow32(2.7182817F, exp_arg_15901);
                    float negate_arg_15903 = x_15812 * B_15893;
                    float exp_arg_15904 = 0.0F - negate_arg_15903;
                    float res_15905 = fpow32(2.7182817F, exp_arg_15904);
                    float res_15906 = res_15902 * res_15905;
                    float res_15883 = res_15906 + redout_16239;
                    float redout_tmp_16685 = res_15883;
                    
                    redout_16239 = redout_tmp_16685;
                }
                res_15880 = redout_16239;
                
                float x_15907 = res_15855 - res_15879;
                float x_15908 = swapprice_arg_15818 * swapprice_arg_15821;
                float y_15909 = res_15880 * x_15908;
                float y_15910 = x_15907 - y_15909;
                float res_15911 = swapprice_arg_15819 * y_15910;
                
                max_arg_15815 = res_15911;
            }
            
            float res_15912 = fmax32(0.0F, max_arg_15815);
            float res_15811 = res_15912 + redout_16305;
            float redout_tmp_16684 = res_15811;
            
            redout_16305 = redout_tmp_16684;
        }
        res_15808 = redout_16305;
        
        float res_15913 = res_15808 / res_13844;
        float negate_arg_15914 = a_13658 * res_15807;
        float exp_arg_15915 = 0.0F - negate_arg_15914;
        float res_15916 = fpow32(2.7182817F, exp_arg_15915);
        float x_15917 = 1.0F - res_15916;
        float B_15918 = x_15917 / a_13658;
        float x_15919 = B_15918 - res_15807;
        float x_15920 = y_13682 * x_15919;
        float A1_15921 = x_15920 / x_13678;
        float y_15922 = fpow32(B_15918, 2.0F);
        float x_15923 = x_13680 * y_15922;
        float A2_15924 = x_15923 / y_13683;
        float exp_arg_15925 = A1_15921 - A2_15924;
        float res_15926 = fpow32(2.7182817F, exp_arg_15925);
        float negate_arg_15927 = 5.0e-2F * B_15918;
        float exp_arg_15928 = 0.0F - negate_arg_15927;
        float res_15929 = fpow32(2.7182817F, exp_arg_15928);
        float res_15930 = res_15926 * res_15929;
        float res_15931 = res_15913 * res_15930;
        
        ((__global float *) mem_16410)[gtid_15069] = res_15931;
        ((__global float *) mem_16412)[gtid_15069] = res_15913;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15799
}
__kernel void mainzisegmap_15477(__global int *global_failure,
                                 int64_t steps_13654, float a_13658,
                                 float x_13678, float x_13680, float y_13682,
                                 float y_13683, float sims_per_year_13760,
                                 float res_13844, __global
                                 unsigned char *mem_16421, __global
                                 unsigned char *mem_16424, __global
                                 unsigned char *mem_16426)
{
    #define segmap_group_sizze_16185 (mainzisegmap_group_sizze_15479)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16759;
    int32_t local_tid_16760;
    int64_t group_sizze_16763;
    int32_t wave_sizze_16762;
    int32_t group_tid_16761;
    
    global_tid_16759 = get_global_id(0);
    local_tid_16760 = get_local_id(0);
    group_sizze_16763 = get_local_size(0);
    wave_sizze_16762 = LOCKSTEP_WIDTH;
    group_tid_16761 = get_group_id(0);
    
    int32_t phys_tid_15477;
    
    phys_tid_15477 = global_tid_16759;
    
    int64_t gtid_15476;
    
    gtid_15476 = sext_i32_i64(group_tid_16761) * segmap_group_sizze_16185 +
        sext_i32_i64(local_tid_16760);
    if (slt64(gtid_15476, steps_13654)) {
        int64_t convop_x_16260 = add64(1, gtid_15476);
        float binop_x_16261 = sitofp_i64_f32(convop_x_16260);
        float index_primexp_16262 = binop_x_16261 / sims_per_year_13760;
        float res_16190 = ((__global float *) mem_16421)[gtid_15476];
        float res_16191 = res_16190 / res_13844;
        float negate_arg_16192 = a_13658 * index_primexp_16262;
        float exp_arg_16193 = 0.0F - negate_arg_16192;
        float res_16194 = fpow32(2.7182817F, exp_arg_16193);
        float x_16195 = 1.0F - res_16194;
        float B_16196 = x_16195 / a_13658;
        float x_16197 = B_16196 - index_primexp_16262;
        float x_16198 = y_13682 * x_16197;
        float A1_16199 = x_16198 / x_13678;
        float y_16200 = fpow32(B_16196, 2.0F);
        float x_16201 = x_13680 * y_16200;
        float A2_16202 = x_16201 / y_13683;
        float exp_arg_16203 = A1_16199 - A2_16202;
        float res_16204 = fpow32(2.7182817F, exp_arg_16203);
        float negate_arg_16205 = 5.0e-2F * B_16196;
        float exp_arg_16206 = 0.0F - negate_arg_16205;
        float res_16207 = fpow32(2.7182817F, exp_arg_16206);
        float res_16208 = res_16204 * res_16207;
        float res_16209 = res_16191 * res_16208;
        
        ((__global float *) mem_16424)[gtid_15476] = res_16209;
        ((__global float *) mem_16426)[gtid_15476] = res_16191;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16185
}
__kernel void mainzisegmap_intragroup_15068(__global int *global_failure,
                                            int failure_is_an_option, __global
                                            int64_t *global_failure_args,
                                            __local volatile
                                            int64_t *red_arr_mem_16692_backing_aligned_0,
                                            int64_t n_13650,
                                            int64_t paths_13653, float a_13658,
                                            float b_13659, float sigma_13660,
                                            float x_13678, float x_13680,
                                            float y_13682, float y_13683,
                                            float sims_per_year_13760,
                                            float last_date_13775,
                                            float res_13844, __global
                                            unsigned char *res_mem_16371,
                                            __global
                                            unsigned char *res_mem_16372,
                                            __global
                                            unsigned char *res_mem_16373,
                                            __global
                                            unsigned char *res_mem_16374,
                                            __global unsigned char *mem_16402,
                                            __global unsigned char *mem_16416,
                                            __global unsigned char *mem_16418)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16692_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_16692_backing_aligned_0;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16686;
    int32_t local_tid_16687;
    int64_t group_sizze_16690;
    int32_t wave_sizze_16689;
    int32_t group_tid_16688;
    
    global_tid_16686 = get_global_id(0);
    local_tid_16687 = get_local_id(0);
    group_sizze_16690 = get_local_size(0);
    wave_sizze_16689 = LOCKSTEP_WIDTH;
    group_tid_16688 = get_group_id(0);
    
    int32_t phys_tid_15068;
    
    phys_tid_15068 = group_tid_16688;
    
    int32_t ltid_pre_16691;
    
    ltid_pre_16691 = local_tid_16687;
    
    int64_t gtid_15063;
    
    gtid_15063 = sext_i32_i64(group_tid_16688);
    
    int64_t index_primexp_16250;
    
    index_primexp_16250 = add64(1, gtid_15063);
    
    float res_15939 = sitofp_i64_f32(index_primexp_16250);
    float res_15940 = res_15939 / sims_per_year_13760;
    bool cond_15947 = last_date_13775 < res_15940;
    float res_15941;
    int64_t gtid_15066 = sext_i32_i64(ltid_pre_16691);
    int32_t phys_tid_15067 = local_tid_16687;
    __local char *red_arr_mem_16692;
    
    red_arr_mem_16692 = (__local char *) red_arr_mem_16692_backing_0;
    if (slt64(gtid_15066, paths_13653)) {
        float max_arg_15948;
        
        if (cond_15947) {
            max_arg_15948 = 0.0F;
        } else {
            float x_15945 = ((__global float *) mem_16402)[gtid_15063 *
                                                           paths_13653 +
                                                           gtid_15066];
            bool y_15949 = slt64(0, n_13650);
            bool index_certs_15950;
            
            if (!y_15949) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 14) ==
                        -1) {
                        global_failure_args[0] = 0;
                        global_failure_args[1] = n_13650;
                        ;
                    }
                    local_failure = true;
                    goto error_0;
                }
            }
            
            float swapprice_arg_15951 = ((__global float *) res_mem_16371)[0];
            float swapprice_arg_15952 = ((__global float *) res_mem_16372)[0];
            int64_t swapprice_arg_15953 = ((__global
                                            int64_t *) res_mem_16373)[0];
            float swapprice_arg_15954 = ((__global float *) res_mem_16374)[0];
            float ceil_arg_15955 = res_15940 / swapprice_arg_15954;
            float res_15956;
            
            res_15956 = futrts_ceil32(ceil_arg_15955);
            
            float nextpayment_15957 = swapprice_arg_15954 * res_15956;
            int64_t res_15958 = fptosi_f32_i64(res_15956);
            int64_t remaining_15959 = sub64(swapprice_arg_15953, res_15958);
            bool bounds_invalid_upwards_15960 = slt64(remaining_15959, 1);
            bool valid_15961 = !bounds_invalid_upwards_15960;
            bool range_valid_c_15962;
            
            if (!valid_15961) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 15) ==
                        -1) {
                        global_failure_args[0] = 1;
                        global_failure_args[1] = 2;
                        global_failure_args[2] = remaining_15959;
                        ;
                    }
                    local_failure = true;
                    goto error_0;
                }
            }
            
            float y_15964 = nextpayment_15957 - res_15940;
            float negate_arg_15965 = a_13658 * y_15964;
            float exp_arg_15966 = 0.0F - negate_arg_15965;
            float res_15967 = fpow32(2.7182817F, exp_arg_15966);
            float x_15968 = 1.0F - res_15967;
            float B_15969 = x_15968 / a_13658;
            float x_15970 = B_15969 - nextpayment_15957;
            float x_15971 = res_15940 + x_15970;
            float x_15972 = fpow32(a_13658, 2.0F);
            float x_15973 = b_13659 * x_15972;
            float x_15974 = fpow32(sigma_13660, 2.0F);
            float y_15975 = x_15974 / 2.0F;
            float y_15976 = x_15973 - y_15975;
            float x_15977 = x_15971 * y_15976;
            float A1_15978 = x_15977 / x_15972;
            float y_15979 = fpow32(B_15969, 2.0F);
            float x_15980 = x_15974 * y_15979;
            float y_15981 = 4.0F * a_13658;
            float A2_15982 = x_15980 / y_15981;
            float exp_arg_15983 = A1_15978 - A2_15982;
            float res_15984 = fpow32(2.7182817F, exp_arg_15983);
            float negate_arg_15985 = x_15945 * B_15969;
            float exp_arg_15986 = 0.0F - negate_arg_15985;
            float res_15987 = fpow32(2.7182817F, exp_arg_15986);
            float res_15988 = res_15984 * res_15987;
            bool y_15989 = slt64(0, remaining_15959);
            bool index_certs_15990;
            
            if (!y_15989) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 16) ==
                        -1) {
                        global_failure_args[0] = 0;
                        global_failure_args[1] = remaining_15959;
                        ;
                    }
                    local_failure = true;
                    goto error_0;
                }
            }
            
            float binop_y_15991 = sitofp_i64_f32(remaining_15959);
            float binop_y_15992 = swapprice_arg_15954 * binop_y_15991;
            float index_primexp_15993 = nextpayment_15957 + binop_y_15992;
            float y_15994 = index_primexp_15993 - res_15940;
            float negate_arg_15995 = a_13658 * y_15994;
            float exp_arg_15996 = 0.0F - negate_arg_15995;
            float res_15997 = fpow32(2.7182817F, exp_arg_15996);
            float x_15998 = 1.0F - res_15997;
            float B_15999 = x_15998 / a_13658;
            float x_16000 = B_15999 - index_primexp_15993;
            float x_16001 = res_15940 + x_16000;
            float x_16002 = y_15976 * x_16001;
            float A1_16003 = x_16002 / x_15972;
            float y_16004 = fpow32(B_15999, 2.0F);
            float x_16005 = x_15974 * y_16004;
            float A2_16006 = x_16005 / y_15981;
            float exp_arg_16007 = A1_16003 - A2_16006;
            float res_16008 = fpow32(2.7182817F, exp_arg_16007);
            float negate_arg_16009 = x_15945 * B_15999;
            float exp_arg_16010 = 0.0F - negate_arg_16009;
            float res_16011 = fpow32(2.7182817F, exp_arg_16010);
            float res_16012 = res_16008 * res_16011;
            float res_16013;
            float redout_16243 = 0.0F;
            
            for (int64_t i_16244 = 0; i_16244 < remaining_15959; i_16244++) {
                int64_t index_primexp_16274 = add64(1, i_16244);
                float res_16018 = sitofp_i64_f32(index_primexp_16274);
                float res_16019 = swapprice_arg_15954 * res_16018;
                float res_16020 = nextpayment_15957 + res_16019;
                float y_16021 = res_16020 - res_15940;
                float negate_arg_16022 = a_13658 * y_16021;
                float exp_arg_16023 = 0.0F - negate_arg_16022;
                float res_16024 = fpow32(2.7182817F, exp_arg_16023);
                float x_16025 = 1.0F - res_16024;
                float B_16026 = x_16025 / a_13658;
                float x_16027 = B_16026 - res_16020;
                float x_16028 = res_15940 + x_16027;
                float x_16029 = y_15976 * x_16028;
                float A1_16030 = x_16029 / x_15972;
                float y_16031 = fpow32(B_16026, 2.0F);
                float x_16032 = x_15974 * y_16031;
                float A2_16033 = x_16032 / y_15981;
                float exp_arg_16034 = A1_16030 - A2_16033;
                float res_16035 = fpow32(2.7182817F, exp_arg_16034);
                float negate_arg_16036 = x_15945 * B_16026;
                float exp_arg_16037 = 0.0F - negate_arg_16036;
                float res_16038 = fpow32(2.7182817F, exp_arg_16037);
                float res_16039 = res_16035 * res_16038;
                float res_16016 = res_16039 + redout_16243;
                float redout_tmp_16694 = res_16016;
                
                redout_16243 = redout_tmp_16694;
            }
            res_16013 = redout_16243;
            
            float x_16040 = res_15988 - res_16012;
            float x_16041 = swapprice_arg_15951 * swapprice_arg_15954;
            float y_16042 = res_16013 * x_16041;
            float y_16043 = x_16040 - y_16042;
            float res_16044 = swapprice_arg_15952 * y_16043;
            
            max_arg_15948 = res_16044;
        }
        
        float res_16045 = fmax32(0.0F, max_arg_15948);
        
        ((__local float *) red_arr_mem_16692)[gtid_15066] = res_16045;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_16695;
    int32_t skip_waves_16696;
    
    skip_waves_16696 = 1;
    
    float x_15942;
    float x_15943;
    
    offset_16695 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_16687, sext_i64_i32(paths_13653))) {
            x_15942 = ((__local
                        float *) red_arr_mem_16692)[sext_i32_i64(local_tid_16687 +
                                                    offset_16695)];
        }
    }
    offset_16695 = 1;
    while (slt32(offset_16695, wave_sizze_16689)) {
        if (slt32(local_tid_16687 + offset_16695, sext_i64_i32(paths_13653)) &&
            ((local_tid_16687 - squot32(local_tid_16687, wave_sizze_16689) *
              wave_sizze_16689) & (2 * offset_16695 - 1)) == 0) {
            // read array element
            {
                x_15943 = ((volatile __local
                            float *) red_arr_mem_16692)[sext_i32_i64(local_tid_16687 +
                                                        offset_16695)];
            }
            // apply reduction operation
            {
                float res_15944 = x_15942 + x_15943;
                
                x_15942 = res_15944;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_16692)[sext_i32_i64(local_tid_16687)] =
                    x_15942;
            }
        }
        offset_16695 *= 2;
    }
    while (slt32(skip_waves_16696, squot32(sext_i64_i32(paths_13653) +
                                           wave_sizze_16689 - 1,
                                           wave_sizze_16689))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_16695 = skip_waves_16696 * wave_sizze_16689;
        if (slt32(local_tid_16687 + offset_16695, sext_i64_i32(paths_13653)) &&
            ((local_tid_16687 - squot32(local_tid_16687, wave_sizze_16689) *
              wave_sizze_16689) == 0 && (squot32(local_tid_16687,
                                                 wave_sizze_16689) & (2 *
                                                                      skip_waves_16696 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_15943 = ((__local
                            float *) red_arr_mem_16692)[sext_i32_i64(local_tid_16687 +
                                                        offset_16695)];
            }
            // apply reduction operation
            {
                float res_15944 = x_15942 + x_15943;
                
                x_15942 = res_15944;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_16692)[sext_i32_i64(local_tid_16687)] =
                    x_15942;
            }
        }
        skip_waves_16696 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_15941 = ((__local float *) red_arr_mem_16692)[0];
    
    float res_16046 = res_15941 / res_13844;
    float negate_arg_16047 = a_13658 * res_15940;
    float exp_arg_16048 = 0.0F - negate_arg_16047;
    float res_16049 = fpow32(2.7182817F, exp_arg_16048);
    float x_16050 = 1.0F - res_16049;
    float B_16051 = x_16050 / a_13658;
    float x_16052 = B_16051 - res_15940;
    float x_16053 = y_13682 * x_16052;
    float A1_16054 = x_16053 / x_13678;
    float y_16055 = fpow32(B_16051, 2.0F);
    float x_16056 = x_13680 * y_16055;
    float A2_16057 = x_16056 / y_13683;
    float exp_arg_16058 = A1_16054 - A2_16057;
    float res_16059 = fpow32(2.7182817F, exp_arg_16058);
    float negate_arg_16060 = 5.0e-2F * B_16051;
    float exp_arg_16061 = 0.0F - negate_arg_16060;
    float res_16062 = fpow32(2.7182817F, exp_arg_16061);
    float res_16063 = res_16059 * res_16062;
    float res_16064 = res_16046 * res_16063;
    
    if (local_tid_16687 == 0) {
        ((__global float *) mem_16416)[gtid_15063] = res_16064;
    }
    if (local_tid_16687 == 0) {
        ((__global float *) mem_16418)[gtid_15063] = res_16046;
    }
    
  error_2:
    return;
}
__kernel void mainzisegred_large_15512(__global int *global_failure,
                                       int failure_is_an_option, __global
                                       int64_t *global_failure_args,
                                       __local volatile
                                       int64_t *sync_arr_mem_16735_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_16733_backing_aligned_1,
                                       int64_t n_13650, int64_t paths_13653,
                                       float a_13658, float b_13659,
                                       float sigma_13660,
                                       float sims_per_year_13760,
                                       float last_date_13775,
                                       int64_t num_groups_16075, __global
                                       unsigned char *res_mem_16371, __global
                                       unsigned char *res_mem_16372, __global
                                       unsigned char *res_mem_16373, __global
                                       unsigned char *res_mem_16374, __global
                                       unsigned char *mem_16402, __global
                                       unsigned char *mem_16421,
                                       int64_t groups_per_segment_16719,
                                       int64_t elements_per_thread_16720,
                                       int64_t virt_num_groups_16721,
                                       int64_t threads_per_segment_16723,
                                       __global
                                       unsigned char *group_res_arr_mem_16724,
                                       __global
                                       unsigned char *mainzicounter_mem_16726)
{
    #define segred_group_sizze_16074 (mainzisegred_group_sizze_15506)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_16735_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_16735_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_16733_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_16733_backing_aligned_1;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16728;
    int32_t local_tid_16729;
    int64_t group_sizze_16732;
    int32_t wave_sizze_16731;
    int32_t group_tid_16730;
    
    global_tid_16728 = get_global_id(0);
    local_tid_16729 = get_local_id(0);
    group_sizze_16732 = get_local_size(0);
    wave_sizze_16731 = LOCKSTEP_WIDTH;
    group_tid_16730 = get_group_id(0);
    
    int32_t phys_tid_15512;
    
    phys_tid_15512 = global_tid_16728;
    
    __local char *red_arr_mem_16733;
    
    red_arr_mem_16733 = (__local char *) red_arr_mem_16733_backing_0;
    
    __local char *sync_arr_mem_16735;
    
    sync_arr_mem_16735 = (__local char *) sync_arr_mem_16735_backing_1;
    
    int32_t phys_group_id_16737;
    
    phys_group_id_16737 = get_group_id(0);
    for (int32_t i_16738 = 0; i_16738 <
         sdiv_up32(sext_i64_i32(virt_num_groups_16721) - phys_group_id_16737,
                   sext_i64_i32(num_groups_16075)); i_16738++) {
        int32_t virt_group_id_16739 = phys_group_id_16737 + i_16738 *
                sext_i64_i32(num_groups_16075);
        int32_t flat_segment_id_16740 = squot32(virt_group_id_16739,
                                                sext_i64_i32(groups_per_segment_16719));
        int64_t global_tid_16741 = srem64(sext_i32_i64(virt_group_id_16739) *
                                          segred_group_sizze_16074 +
                                          sext_i32_i64(local_tid_16729),
                                          segred_group_sizze_16074 *
                                          groups_per_segment_16719);
        int64_t gtid_15503 = sext_i32_i64(flat_segment_id_16740);
        int64_t gtid_15511;
        float x_acc_16742;
        int64_t chunk_sizze_16743;
        
        chunk_sizze_16743 = smin64(elements_per_thread_16720,
                                   sdiv_up64(paths_13653 -
                                             sext_i32_i64(sext_i64_i32(global_tid_16741)),
                                             threads_per_segment_16723));
        
        float x_16078;
        float x_16079;
        
        // neutral-initialise the accumulators
        {
            x_acc_16742 = 0.0F;
        }
        for (int64_t i_16747 = 0; i_16747 < chunk_sizze_16743; i_16747++) {
            gtid_15511 = sext_i32_i64(sext_i64_i32(global_tid_16741)) +
                threads_per_segment_16723 * i_16747;
            // apply map function
            {
                int64_t convop_x_16256 = add64(1, gtid_15503);
                float binop_x_16257 = sitofp_i64_f32(convop_x_16256);
                float index_primexp_16258 = binop_x_16257 / sims_per_year_13760;
                bool cond_16085 = last_date_13775 < index_primexp_16258;
                float max_arg_16086;
                
                if (cond_16085) {
                    max_arg_16086 = 0.0F;
                } else {
                    float x_16083 = ((__global float *) mem_16402)[gtid_15503 *
                                                                   paths_13653 +
                                                                   gtid_15511];
                    bool y_16087 = slt64(0, n_13650);
                    bool index_certs_16088;
                    
                    if (!y_16087) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          20) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = n_13650;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float swapprice_arg_16089 = ((__global
                                                  float *) res_mem_16371)[0];
                    float swapprice_arg_16090 = ((__global
                                                  float *) res_mem_16372)[0];
                    int64_t swapprice_arg_16091 = ((__global
                                                    int64_t *) res_mem_16373)[0];
                    float swapprice_arg_16092 = ((__global
                                                  float *) res_mem_16374)[0];
                    float ceil_arg_16093 = index_primexp_16258 /
                          swapprice_arg_16092;
                    float res_16094;
                    
                    res_16094 = futrts_ceil32(ceil_arg_16093);
                    
                    float nextpayment_16095 = swapprice_arg_16092 * res_16094;
                    int64_t res_16096 = fptosi_f32_i64(res_16094);
                    int64_t remaining_16097 = sub64(swapprice_arg_16091,
                                                    res_16096);
                    bool bounds_invalid_upwards_16098 = slt64(remaining_16097,
                                                              1);
                    bool valid_16099 = !bounds_invalid_upwards_16098;
                    bool range_valid_c_16100;
                    
                    if (!valid_16099) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          21) == -1) {
                                global_failure_args[0] = 1;
                                global_failure_args[1] = 2;
                                global_failure_args[2] = remaining_16097;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_16102 = nextpayment_16095 - index_primexp_16258;
                    float negate_arg_16103 = a_13658 * y_16102;
                    float exp_arg_16104 = 0.0F - negate_arg_16103;
                    float res_16105 = fpow32(2.7182817F, exp_arg_16104);
                    float x_16106 = 1.0F - res_16105;
                    float B_16107 = x_16106 / a_13658;
                    float x_16108 = B_16107 - nextpayment_16095;
                    float x_16109 = x_16108 + index_primexp_16258;
                    float x_16110 = fpow32(a_13658, 2.0F);
                    float x_16111 = b_13659 * x_16110;
                    float x_16112 = fpow32(sigma_13660, 2.0F);
                    float y_16113 = x_16112 / 2.0F;
                    float y_16114 = x_16111 - y_16113;
                    float x_16115 = x_16109 * y_16114;
                    float A1_16116 = x_16115 / x_16110;
                    float y_16117 = fpow32(B_16107, 2.0F);
                    float x_16118 = x_16112 * y_16117;
                    float y_16119 = 4.0F * a_13658;
                    float A2_16120 = x_16118 / y_16119;
                    float exp_arg_16121 = A1_16116 - A2_16120;
                    float res_16122 = fpow32(2.7182817F, exp_arg_16121);
                    float negate_arg_16123 = x_16083 * B_16107;
                    float exp_arg_16124 = 0.0F - negate_arg_16123;
                    float res_16125 = fpow32(2.7182817F, exp_arg_16124);
                    float res_16126 = res_16122 * res_16125;
                    bool y_16127 = slt64(0, remaining_16097);
                    bool index_certs_16128;
                    
                    if (!y_16127) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          22) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = remaining_16097;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float binop_y_16129 = sitofp_i64_f32(remaining_16097);
                    float binop_y_16130 = swapprice_arg_16092 * binop_y_16129;
                    float index_primexp_16131 = nextpayment_16095 +
                          binop_y_16130;
                    float y_16132 = index_primexp_16131 - index_primexp_16258;
                    float negate_arg_16133 = a_13658 * y_16132;
                    float exp_arg_16134 = 0.0F - negate_arg_16133;
                    float res_16135 = fpow32(2.7182817F, exp_arg_16134);
                    float x_16136 = 1.0F - res_16135;
                    float B_16137 = x_16136 / a_13658;
                    float x_16138 = B_16137 - index_primexp_16131;
                    float x_16139 = x_16138 + index_primexp_16258;
                    float x_16140 = y_16114 * x_16139;
                    float A1_16141 = x_16140 / x_16110;
                    float y_16142 = fpow32(B_16137, 2.0F);
                    float x_16143 = x_16112 * y_16142;
                    float A2_16144 = x_16143 / y_16119;
                    float exp_arg_16145 = A1_16141 - A2_16144;
                    float res_16146 = fpow32(2.7182817F, exp_arg_16145);
                    float negate_arg_16147 = x_16083 * B_16137;
                    float exp_arg_16148 = 0.0F - negate_arg_16147;
                    float res_16149 = fpow32(2.7182817F, exp_arg_16148);
                    float res_16150 = res_16146 * res_16149;
                    float res_16151;
                    float redout_16253 = 0.0F;
                    
                    for (int64_t i_16254 = 0; i_16254 < remaining_16097;
                         i_16254++) {
                        int64_t index_primexp_16276 = add64(1, i_16254);
                        float res_16156 = sitofp_i64_f32(index_primexp_16276);
                        float res_16157 = swapprice_arg_16092 * res_16156;
                        float res_16158 = nextpayment_16095 + res_16157;
                        float y_16159 = res_16158 - index_primexp_16258;
                        float negate_arg_16160 = a_13658 * y_16159;
                        float exp_arg_16161 = 0.0F - negate_arg_16160;
                        float res_16162 = fpow32(2.7182817F, exp_arg_16161);
                        float x_16163 = 1.0F - res_16162;
                        float B_16164 = x_16163 / a_13658;
                        float x_16165 = B_16164 - res_16158;
                        float x_16166 = x_16165 + index_primexp_16258;
                        float x_16167 = y_16114 * x_16166;
                        float A1_16168 = x_16167 / x_16110;
                        float y_16169 = fpow32(B_16164, 2.0F);
                        float x_16170 = x_16112 * y_16169;
                        float A2_16171 = x_16170 / y_16119;
                        float exp_arg_16172 = A1_16168 - A2_16171;
                        float res_16173 = fpow32(2.7182817F, exp_arg_16172);
                        float negate_arg_16174 = x_16083 * B_16164;
                        float exp_arg_16175 = 0.0F - negate_arg_16174;
                        float res_16176 = fpow32(2.7182817F, exp_arg_16175);
                        float res_16177 = res_16173 * res_16176;
                        float res_16154 = res_16177 + redout_16253;
                        float redout_tmp_16748 = res_16154;
                        
                        redout_16253 = redout_tmp_16748;
                    }
                    res_16151 = redout_16253;
                    
                    float x_16178 = res_16126 - res_16150;
                    float x_16179 = swapprice_arg_16089 * swapprice_arg_16092;
                    float y_16180 = res_16151 * x_16179;
                    float y_16181 = x_16178 - y_16180;
                    float res_16182 = swapprice_arg_16090 * y_16181;
                    
                    max_arg_16086 = res_16182;
                }
                
                float res_16183 = fmax32(0.0F, max_arg_16086);
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_16078 = x_acc_16742;
                }
                // load new values
                {
                    x_16079 = res_16183;
                }
                // apply reduction operator
                {
                    float res_16080 = x_16078 + x_16079;
                    
                    // store in accumulator
                    {
                        x_acc_16742 = res_16080;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_16078 = x_acc_16742;
            ((__local
              float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729)] =
                x_16078;
        }
        
      error_0:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_16749;
        int32_t skip_waves_16750;
        
        skip_waves_16750 = 1;
        
        float x_16744;
        float x_16745;
        
        offset_16749 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_16729,
                      sext_i64_i32(segred_group_sizze_16074))) {
                x_16744 = ((__local
                            float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729 +
                                                        offset_16749)];
            }
        }
        offset_16749 = 1;
        while (slt32(offset_16749, wave_sizze_16731)) {
            if (slt32(local_tid_16729 + offset_16749,
                      sext_i64_i32(segred_group_sizze_16074)) &&
                ((local_tid_16729 - squot32(local_tid_16729, wave_sizze_16731) *
                  wave_sizze_16731) & (2 * offset_16749 - 1)) == 0) {
                // read array element
                {
                    x_16745 = ((volatile __local
                                float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729 +
                                                            offset_16749)];
                }
                // apply reduction operation
                {
                    float res_16746 = x_16744 + x_16745;
                    
                    x_16744 = res_16746;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729)] =
                        x_16744;
                }
            }
            offset_16749 *= 2;
        }
        while (slt32(skip_waves_16750,
                     squot32(sext_i64_i32(segred_group_sizze_16074) +
                             wave_sizze_16731 - 1, wave_sizze_16731))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_16749 = skip_waves_16750 * wave_sizze_16731;
            if (slt32(local_tid_16729 + offset_16749,
                      sext_i64_i32(segred_group_sizze_16074)) &&
                ((local_tid_16729 - squot32(local_tid_16729, wave_sizze_16731) *
                  wave_sizze_16731) == 0 && (squot32(local_tid_16729,
                                                     wave_sizze_16731) & (2 *
                                                                          skip_waves_16750 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_16745 = ((__local
                                float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729 +
                                                            offset_16749)];
                }
                // apply reduction operation
                {
                    float res_16746 = x_16744 + x_16745;
                    
                    x_16744 = res_16746;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729)] =
                        x_16744;
                }
            }
            skip_waves_16750 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (sext_i32_i64(local_tid_16729) == 0) {
                x_acc_16742 = x_16744;
            }
        }
        if (groups_per_segment_16719 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_16729 == 0) {
                    ((__global float *) mem_16421)[gtid_15503] = x_acc_16742;
                }
            }
        } else {
            int32_t old_counter_16751;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_16729 == 0) {
                    ((__global
                      float *) group_res_arr_mem_16724)[sext_i32_i64(virt_group_id_16739) *
                                                        segred_group_sizze_16074] =
                        x_acc_16742;
                    mem_fence_global();
                    old_counter_16751 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_16726)[sext_i32_i64(srem32(flat_segment_id_16740,
                                                                                                     10240))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_16735)[0] =
                        old_counter_16751 == groups_per_segment_16719 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_16752;
            
            is_last_group_16752 = ((__local bool *) sync_arr_mem_16735)[0];
            if (is_last_group_16752) {
                if (local_tid_16729 == 0) {
                    old_counter_16751 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_16726)[sext_i32_i64(srem32(flat_segment_id_16740,
                                                                                                     10240))],
                                              (int) (0 -
                                                     groups_per_segment_16719));
                }
                // read in the per-group-results
                {
                    int64_t read_per_thread_16753 =
                            sdiv_up64(groups_per_segment_16719,
                                      segred_group_sizze_16074);
                    
                    x_16078 = 0.0F;
                    for (int64_t i_16754 = 0; i_16754 < read_per_thread_16753;
                         i_16754++) {
                        int64_t group_res_id_16755 =
                                sext_i32_i64(local_tid_16729) *
                                read_per_thread_16753 + i_16754;
                        int64_t index_of_group_res_16756 =
                                sext_i32_i64(flat_segment_id_16740) *
                                groups_per_segment_16719 + group_res_id_16755;
                        
                        if (slt64(group_res_id_16755,
                                  groups_per_segment_16719)) {
                            x_16079 = ((__global
                                        float *) group_res_arr_mem_16724)[index_of_group_res_16756 *
                                                                          segred_group_sizze_16074];
                            
                            float res_16080;
                            
                            res_16080 = x_16078 + x_16079;
                            x_16078 = res_16080;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729)] =
                    x_16078;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_16757;
                    int32_t skip_waves_16758;
                    
                    skip_waves_16758 = 1;
                    
                    float x_16744;
                    float x_16745;
                    
                    offset_16757 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_16729,
                                  sext_i64_i32(segred_group_sizze_16074))) {
                            x_16744 = ((__local
                                        float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729 +
                                                                    offset_16757)];
                        }
                    }
                    offset_16757 = 1;
                    while (slt32(offset_16757, wave_sizze_16731)) {
                        if (slt32(local_tid_16729 + offset_16757,
                                  sext_i64_i32(segred_group_sizze_16074)) &&
                            ((local_tid_16729 - squot32(local_tid_16729,
                                                        wave_sizze_16731) *
                              wave_sizze_16731) & (2 * offset_16757 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_16745 = ((volatile __local
                                            float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729 +
                                                                        offset_16757)];
                            }
                            // apply reduction operation
                            {
                                float res_16746 = x_16744 + x_16745;
                                
                                x_16744 = res_16746;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729)] =
                                    x_16744;
                            }
                        }
                        offset_16757 *= 2;
                    }
                    while (slt32(skip_waves_16758,
                                 squot32(sext_i64_i32(segred_group_sizze_16074) +
                                         wave_sizze_16731 - 1,
                                         wave_sizze_16731))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_16757 = skip_waves_16758 * wave_sizze_16731;
                        if (slt32(local_tid_16729 + offset_16757,
                                  sext_i64_i32(segred_group_sizze_16074)) &&
                            ((local_tid_16729 - squot32(local_tid_16729,
                                                        wave_sizze_16731) *
                              wave_sizze_16731) == 0 &&
                             (squot32(local_tid_16729, wave_sizze_16731) & (2 *
                                                                            skip_waves_16758 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_16745 = ((__local
                                            float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729 +
                                                                        offset_16757)];
                            }
                            // apply reduction operation
                            {
                                float res_16746 = x_16744 + x_16745;
                                
                                x_16744 = res_16746;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_16733)[sext_i32_i64(local_tid_16729)] =
                                    x_16744;
                            }
                        }
                        skip_waves_16758 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_16729 == 0) {
                            ((__global float *) mem_16421)[gtid_15503] =
                                x_16744;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_16074
}
__kernel void mainzisegred_nonseg_14106(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_16529_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_16527_backing_aligned_1,
                                        int64_t n_13650,
                                        int64_t num_groups_14101, __global
                                        unsigned char *swap_term_mem_16319,
                                        __global
                                        unsigned char *payments_mem_16320,
                                        __global unsigned char *mem_16324,
                                        __global
                                        unsigned char *mainzicounter_mem_16517,
                                        __global
                                        unsigned char *group_res_arr_mem_16519,
                                        int64_t num_threads_16521)
{
    #define segred_group_sizze_14099 (mainzisegred_group_sizze_14098)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16529_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_16529_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_16527_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_16527_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16522;
    int32_t local_tid_16523;
    int64_t group_sizze_16526;
    int32_t wave_sizze_16525;
    int32_t group_tid_16524;
    
    global_tid_16522 = get_global_id(0);
    local_tid_16523 = get_local_id(0);
    group_sizze_16526 = get_local_size(0);
    wave_sizze_16525 = LOCKSTEP_WIDTH;
    group_tid_16524 = get_group_id(0);
    
    int32_t phys_tid_14106;
    
    phys_tid_14106 = global_tid_16522;
    
    __local char *sync_arr_mem_16527;
    
    sync_arr_mem_16527 = (__local char *) sync_arr_mem_16527_backing_0;
    
    __local char *red_arr_mem_16529;
    
    red_arr_mem_16529 = (__local char *) red_arr_mem_16529_backing_1;
    
    int64_t dummy_14104;
    
    dummy_14104 = 0;
    
    int64_t gtid_14105;
    
    gtid_14105 = 0;
    
    float x_acc_16531;
    int64_t chunk_sizze_16532;
    
    chunk_sizze_16532 = smin64(sdiv_up64(n_13650,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_14099 *
                                         num_groups_14101))),
                               sdiv_up64(n_13650 - sext_i32_i64(phys_tid_14106),
                                         num_threads_16521));
    
    float x_13668;
    float x_13669;
    
    // neutral-initialise the accumulators
    {
        x_acc_16531 = -INFINITY;
    }
    for (int64_t i_16536 = 0; i_16536 < chunk_sizze_16532; i_16536++) {
        gtid_14105 = sext_i32_i64(phys_tid_14106) + num_threads_16521 * i_16536;
        // apply map function
        {
            float x_13671 = ((__global
                              float *) swap_term_mem_16319)[gtid_14105];
            int64_t x_13672 = ((__global
                                int64_t *) payments_mem_16320)[gtid_14105];
            float res_13673 = sitofp_i64_f32(x_13672);
            float res_13674 = x_13671 * res_13673;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_13668 = x_acc_16531;
            }
            // load new values
            {
                x_13669 = res_13674;
            }
            // apply reduction operator
            {
                float res_13670 = fmax32(x_13668, x_13669);
                
                // store in accumulator
                {
                    x_acc_16531 = res_13670;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_13668 = x_acc_16531;
        ((__local float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523)] =
            x_13668;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_16537;
    int32_t skip_waves_16538;
    
    skip_waves_16538 = 1;
    
    float x_16533;
    float x_16534;
    
    offset_16537 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_16523, sext_i64_i32(segred_group_sizze_14099))) {
            x_16533 = ((__local
                        float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523 +
                                                    offset_16537)];
        }
    }
    offset_16537 = 1;
    while (slt32(offset_16537, wave_sizze_16525)) {
        if (slt32(local_tid_16523 + offset_16537,
                  sext_i64_i32(segred_group_sizze_14099)) && ((local_tid_16523 -
                                                               squot32(local_tid_16523,
                                                                       wave_sizze_16525) *
                                                               wave_sizze_16525) &
                                                              (2 *
                                                               offset_16537 -
                                                               1)) == 0) {
            // read array element
            {
                x_16534 = ((volatile __local
                            float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523 +
                                                        offset_16537)];
            }
            // apply reduction operation
            {
                float res_16535 = fmax32(x_16533, x_16534);
                
                x_16533 = res_16535;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523)] =
                    x_16533;
            }
        }
        offset_16537 *= 2;
    }
    while (slt32(skip_waves_16538,
                 squot32(sext_i64_i32(segred_group_sizze_14099) +
                         wave_sizze_16525 - 1, wave_sizze_16525))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_16537 = skip_waves_16538 * wave_sizze_16525;
        if (slt32(local_tid_16523 + offset_16537,
                  sext_i64_i32(segred_group_sizze_14099)) && ((local_tid_16523 -
                                                               squot32(local_tid_16523,
                                                                       wave_sizze_16525) *
                                                               wave_sizze_16525) ==
                                                              0 &&
                                                              (squot32(local_tid_16523,
                                                                       wave_sizze_16525) &
                                                               (2 *
                                                                skip_waves_16538 -
                                                                1)) == 0)) {
            // read array element
            {
                x_16534 = ((__local
                            float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523 +
                                                        offset_16537)];
            }
            // apply reduction operation
            {
                float res_16535 = fmax32(x_16533, x_16534);
                
                x_16533 = res_16535;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523)] =
                    x_16533;
            }
        }
        skip_waves_16538 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_16523) == 0) {
            x_acc_16531 = x_16533;
        }
    }
    
    int32_t old_counter_16539;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_16523 == 0) {
            ((__global
              float *) group_res_arr_mem_16519)[sext_i32_i64(group_tid_16524) *
                                                segred_group_sizze_14099] =
                x_acc_16531;
            mem_fence_global();
            old_counter_16539 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16517)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_16527)[0] = old_counter_16539 ==
                num_groups_14101 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_16540;
    
    is_last_group_16540 = ((__local bool *) sync_arr_mem_16527)[0];
    if (is_last_group_16540) {
        if (local_tid_16523 == 0) {
            old_counter_16539 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16517)[0],
                                                      (int) (0 -
                                                             num_groups_14101));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_16541 = sdiv_up64(num_groups_14101,
                                                      segred_group_sizze_14099);
            
            x_13668 = -INFINITY;
            for (int64_t i_16542 = 0; i_16542 < read_per_thread_16541;
                 i_16542++) {
                int64_t group_res_id_16543 = sext_i32_i64(local_tid_16523) *
                        read_per_thread_16541 + i_16542;
                int64_t index_of_group_res_16544 = group_res_id_16543;
                
                if (slt64(group_res_id_16543, num_groups_14101)) {
                    x_13669 = ((__global
                                float *) group_res_arr_mem_16519)[index_of_group_res_16544 *
                                                                  segred_group_sizze_14099];
                    
                    float res_13670;
                    
                    res_13670 = fmax32(x_13668, x_13669);
                    x_13668 = res_13670;
                }
            }
        }
        ((__local float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523)] =
            x_13668;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_16545;
            int32_t skip_waves_16546;
            
            skip_waves_16546 = 1;
            
            float x_16533;
            float x_16534;
            
            offset_16545 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_16523,
                          sext_i64_i32(segred_group_sizze_14099))) {
                    x_16533 = ((__local
                                float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523 +
                                                            offset_16545)];
                }
            }
            offset_16545 = 1;
            while (slt32(offset_16545, wave_sizze_16525)) {
                if (slt32(local_tid_16523 + offset_16545,
                          sext_i64_i32(segred_group_sizze_14099)) &&
                    ((local_tid_16523 - squot32(local_tid_16523,
                                                wave_sizze_16525) *
                      wave_sizze_16525) & (2 * offset_16545 - 1)) == 0) {
                    // read array element
                    {
                        x_16534 = ((volatile __local
                                    float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523 +
                                                                offset_16545)];
                    }
                    // apply reduction operation
                    {
                        float res_16535 = fmax32(x_16533, x_16534);
                        
                        x_16533 = res_16535;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523)] =
                            x_16533;
                    }
                }
                offset_16545 *= 2;
            }
            while (slt32(skip_waves_16546,
                         squot32(sext_i64_i32(segred_group_sizze_14099) +
                                 wave_sizze_16525 - 1, wave_sizze_16525))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_16545 = skip_waves_16546 * wave_sizze_16525;
                if (slt32(local_tid_16523 + offset_16545,
                          sext_i64_i32(segred_group_sizze_14099)) &&
                    ((local_tid_16523 - squot32(local_tid_16523,
                                                wave_sizze_16525) *
                      wave_sizze_16525) == 0 && (squot32(local_tid_16523,
                                                         wave_sizze_16525) &
                                                 (2 * skip_waves_16546 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_16534 = ((__local
                                    float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523 +
                                                                offset_16545)];
                    }
                    // apply reduction operation
                    {
                        float res_16535 = fmax32(x_16533, x_16534);
                        
                        x_16533 = res_16535;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_16529)[sext_i32_i64(local_tid_16523)] =
                            x_16533;
                    }
                }
                skip_waves_16546 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_16523 == 0) {
                    ((__global float *) mem_16324)[0] = x_16533;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14099
}
__kernel void mainzisegred_nonseg_14918(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int64_t *global_failure_args,
                                        __local volatile
                                        int64_t *red_arr_mem_16659_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_16657_backing_aligned_1,
                                        int64_t n_13650, int64_t paths_13653,
                                        int64_t steps_13654, float a_13658,
                                        float b_13659, float sigma_13660,
                                        float x_13678, float x_13680,
                                        float y_13682, float y_13683,
                                        float sims_per_year_13760,
                                        float last_date_13775, float res_13844,
                                        int64_t num_groups_14921, __global
                                        unsigned char *res_mem_16371, __global
                                        unsigned char *res_mem_16372, __global
                                        unsigned char *res_mem_16373, __global
                                        unsigned char *res_mem_16374, __global
                                        unsigned char *mem_16402, __global
                                        unsigned char *mem_16405, __global
                                        unsigned char *mem_16407, __global
                                        unsigned char *mainzicounter_mem_16647,
                                        __global
                                        unsigned char *group_res_arr_mem_16649,
                                        int64_t num_threads_16651)
{
    #define segred_group_sizze_14920 (mainzisegred_group_sizze_14909)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16659_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_16659_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_16657_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_16657_backing_aligned_1;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16652;
    int32_t local_tid_16653;
    int64_t group_sizze_16656;
    int32_t wave_sizze_16655;
    int32_t group_tid_16654;
    
    global_tid_16652 = get_global_id(0);
    local_tid_16653 = get_local_id(0);
    group_sizze_16656 = get_local_size(0);
    wave_sizze_16655 = LOCKSTEP_WIDTH;
    group_tid_16654 = get_group_id(0);
    
    int32_t phys_tid_14918;
    
    phys_tid_14918 = global_tid_16652;
    
    __local char *sync_arr_mem_16657;
    
    sync_arr_mem_16657 = (__local char *) sync_arr_mem_16657_backing_0;
    
    __local char *red_arr_mem_16659;
    
    red_arr_mem_16659 = (__local char *) red_arr_mem_16659_backing_1;
    
    int64_t dummy_14916;
    
    dummy_14916 = 0;
    
    int64_t gtid_14917;
    
    gtid_14917 = 0;
    
    float x_acc_16661;
    int64_t chunk_sizze_16662;
    
    chunk_sizze_16662 = smin64(sdiv_up64(steps_13654,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_14920 *
                                         num_groups_14921))),
                               sdiv_up64(steps_13654 -
                                         sext_i32_i64(phys_tid_14918),
                                         num_threads_16651));
    
    float x_14925;
    float x_14926;
    
    // neutral-initialise the accumulators
    {
        x_acc_16661 = 0.0F;
    }
    for (int64_t i_16666 = 0; i_16666 < chunk_sizze_16662; i_16666++) {
        gtid_14917 = sext_i32_i64(phys_tid_14918) + num_threads_16651 * i_16666;
        // apply map function
        {
            int64_t index_primexp_16238 = add64(1, gtid_14917);
            float res_14931 = sitofp_i64_f32(index_primexp_16238);
            float res_14932 = res_14931 / sims_per_year_13760;
            bool cond_14939 = last_date_13775 < res_14932;
            float res_14933;
            float redout_16303 = 0.0F;
            
            for (int64_t i_16304 = 0; i_16304 < paths_13653; i_16304++) {
                float max_arg_14940;
                
                if (cond_14939) {
                    max_arg_14940 = 0.0F;
                } else {
                    float x_14937 = ((__global float *) mem_16402)[gtid_14917 *
                                                                   paths_13653 +
                                                                   i_16304];
                    bool y_14941 = slt64(0, n_13650);
                    bool index_certs_14942;
                    
                    if (!y_14941) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          8) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = n_13650;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float swapprice_arg_14943 = ((__global
                                                  float *) res_mem_16371)[0];
                    float swapprice_arg_14944 = ((__global
                                                  float *) res_mem_16372)[0];
                    int64_t swapprice_arg_14945 = ((__global
                                                    int64_t *) res_mem_16373)[0];
                    float swapprice_arg_14946 = ((__global
                                                  float *) res_mem_16374)[0];
                    float ceil_arg_14947 = res_14932 / swapprice_arg_14946;
                    float res_14948;
                    
                    res_14948 = futrts_ceil32(ceil_arg_14947);
                    
                    float nextpayment_14949 = swapprice_arg_14946 * res_14948;
                    int64_t res_14950 = fptosi_f32_i64(res_14948);
                    int64_t remaining_14951 = sub64(swapprice_arg_14945,
                                                    res_14950);
                    bool bounds_invalid_upwards_14952 = slt64(remaining_14951,
                                                              1);
                    bool valid_14953 = !bounds_invalid_upwards_14952;
                    bool range_valid_c_14954;
                    
                    if (!valid_14953) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          9) == -1) {
                                global_failure_args[0] = 1;
                                global_failure_args[1] = 2;
                                global_failure_args[2] = remaining_14951;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_14956 = nextpayment_14949 - res_14932;
                    float negate_arg_14957 = a_13658 * y_14956;
                    float exp_arg_14958 = 0.0F - negate_arg_14957;
                    float res_14959 = fpow32(2.7182817F, exp_arg_14958);
                    float x_14960 = 1.0F - res_14959;
                    float B_14961 = x_14960 / a_13658;
                    float x_14962 = B_14961 - nextpayment_14949;
                    float x_14963 = res_14932 + x_14962;
                    float x_14964 = fpow32(a_13658, 2.0F);
                    float x_14965 = b_13659 * x_14964;
                    float x_14966 = fpow32(sigma_13660, 2.0F);
                    float y_14967 = x_14966 / 2.0F;
                    float y_14968 = x_14965 - y_14967;
                    float x_14969 = x_14963 * y_14968;
                    float A1_14970 = x_14969 / x_14964;
                    float y_14971 = fpow32(B_14961, 2.0F);
                    float x_14972 = x_14966 * y_14971;
                    float y_14973 = 4.0F * a_13658;
                    float A2_14974 = x_14972 / y_14973;
                    float exp_arg_14975 = A1_14970 - A2_14974;
                    float res_14976 = fpow32(2.7182817F, exp_arg_14975);
                    float negate_arg_14977 = x_14937 * B_14961;
                    float exp_arg_14978 = 0.0F - negate_arg_14977;
                    float res_14979 = fpow32(2.7182817F, exp_arg_14978);
                    float res_14980 = res_14976 * res_14979;
                    bool y_14981 = slt64(0, remaining_14951);
                    bool index_certs_14982;
                    
                    if (!y_14981) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          10) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = remaining_14951;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float binop_y_14983 = sitofp_i64_f32(remaining_14951);
                    float binop_y_14984 = swapprice_arg_14946 * binop_y_14983;
                    float index_primexp_14985 = nextpayment_14949 +
                          binop_y_14984;
                    float y_14986 = index_primexp_14985 - res_14932;
                    float negate_arg_14987 = a_13658 * y_14986;
                    float exp_arg_14988 = 0.0F - negate_arg_14987;
                    float res_14989 = fpow32(2.7182817F, exp_arg_14988);
                    float x_14990 = 1.0F - res_14989;
                    float B_14991 = x_14990 / a_13658;
                    float x_14992 = B_14991 - index_primexp_14985;
                    float x_14993 = res_14932 + x_14992;
                    float x_14994 = y_14968 * x_14993;
                    float A1_14995 = x_14994 / x_14964;
                    float y_14996 = fpow32(B_14991, 2.0F);
                    float x_14997 = x_14966 * y_14996;
                    float A2_14998 = x_14997 / y_14973;
                    float exp_arg_14999 = A1_14995 - A2_14998;
                    float res_15000 = fpow32(2.7182817F, exp_arg_14999);
                    float negate_arg_15001 = x_14937 * B_14991;
                    float exp_arg_15002 = 0.0F - negate_arg_15001;
                    float res_15003 = fpow32(2.7182817F, exp_arg_15002);
                    float res_15004 = res_15000 * res_15003;
                    float res_15005;
                    float redout_16235 = 0.0F;
                    
                    for (int64_t i_16236 = 0; i_16236 < remaining_14951;
                         i_16236++) {
                        int64_t index_primexp_16270 = add64(1, i_16236);
                        float res_15010 = sitofp_i64_f32(index_primexp_16270);
                        float res_15011 = swapprice_arg_14946 * res_15010;
                        float res_15012 = nextpayment_14949 + res_15011;
                        float y_15013 = res_15012 - res_14932;
                        float negate_arg_15014 = a_13658 * y_15013;
                        float exp_arg_15015 = 0.0F - negate_arg_15014;
                        float res_15016 = fpow32(2.7182817F, exp_arg_15015);
                        float x_15017 = 1.0F - res_15016;
                        float B_15018 = x_15017 / a_13658;
                        float x_15019 = B_15018 - res_15012;
                        float x_15020 = res_14932 + x_15019;
                        float x_15021 = y_14968 * x_15020;
                        float A1_15022 = x_15021 / x_14964;
                        float y_15023 = fpow32(B_15018, 2.0F);
                        float x_15024 = x_14966 * y_15023;
                        float A2_15025 = x_15024 / y_14973;
                        float exp_arg_15026 = A1_15022 - A2_15025;
                        float res_15027 = fpow32(2.7182817F, exp_arg_15026);
                        float negate_arg_15028 = x_14937 * B_15018;
                        float exp_arg_15029 = 0.0F - negate_arg_15028;
                        float res_15030 = fpow32(2.7182817F, exp_arg_15029);
                        float res_15031 = res_15027 * res_15030;
                        float res_15008 = res_15031 + redout_16235;
                        float redout_tmp_16668 = res_15008;
                        
                        redout_16235 = redout_tmp_16668;
                    }
                    res_15005 = redout_16235;
                    
                    float x_15032 = res_14980 - res_15004;
                    float x_15033 = swapprice_arg_14943 * swapprice_arg_14946;
                    float y_15034 = res_15005 * x_15033;
                    float y_15035 = x_15032 - y_15034;
                    float res_15036 = swapprice_arg_14944 * y_15035;
                    
                    max_arg_14940 = res_15036;
                }
                
                float res_15037 = fmax32(0.0F, max_arg_14940);
                float res_14936 = res_15037 + redout_16303;
                float redout_tmp_16667 = res_14936;
                
                redout_16303 = redout_tmp_16667;
            }
            res_14933 = redout_16303;
            
            float res_15038 = res_14933 / res_13844;
            float negate_arg_15039 = a_13658 * res_14932;
            float exp_arg_15040 = 0.0F - negate_arg_15039;
            float res_15041 = fpow32(2.7182817F, exp_arg_15040);
            float x_15042 = 1.0F - res_15041;
            float B_15043 = x_15042 / a_13658;
            float x_15044 = B_15043 - res_14932;
            float x_15045 = y_13682 * x_15044;
            float A1_15046 = x_15045 / x_13678;
            float y_15047 = fpow32(B_15043, 2.0F);
            float x_15048 = x_13680 * y_15047;
            float A2_15049 = x_15048 / y_13683;
            float exp_arg_15050 = A1_15046 - A2_15049;
            float res_15051 = fpow32(2.7182817F, exp_arg_15050);
            float negate_arg_15052 = 5.0e-2F * B_15043;
            float exp_arg_15053 = 0.0F - negate_arg_15052;
            float res_15054 = fpow32(2.7182817F, exp_arg_15053);
            float res_15055 = res_15051 * res_15054;
            float res_15056 = res_15038 * res_15055;
            
            // save map-out results
            {
                ((__global float *) mem_16407)[dummy_14916 * steps_13654 +
                                               gtid_14917] = res_15038;
            }
            // load accumulator
            {
                x_14925 = x_acc_16661;
            }
            // load new values
            {
                x_14926 = res_15056;
            }
            // apply reduction operator
            {
                float res_14927 = x_14925 + x_14926;
                
                // store in accumulator
                {
                    x_acc_16661 = res_14927;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_14925 = x_acc_16661;
        ((__local float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653)] =
            x_14925;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_16669;
    int32_t skip_waves_16670;
    
    skip_waves_16670 = 1;
    
    float x_16663;
    float x_16664;
    
    offset_16669 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_16653, sext_i64_i32(segred_group_sizze_14920))) {
            x_16663 = ((__local
                        float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653 +
                                                    offset_16669)];
        }
    }
    offset_16669 = 1;
    while (slt32(offset_16669, wave_sizze_16655)) {
        if (slt32(local_tid_16653 + offset_16669,
                  sext_i64_i32(segred_group_sizze_14920)) && ((local_tid_16653 -
                                                               squot32(local_tid_16653,
                                                                       wave_sizze_16655) *
                                                               wave_sizze_16655) &
                                                              (2 *
                                                               offset_16669 -
                                                               1)) == 0) {
            // read array element
            {
                x_16664 = ((volatile __local
                            float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653 +
                                                        offset_16669)];
            }
            // apply reduction operation
            {
                float res_16665 = x_16663 + x_16664;
                
                x_16663 = res_16665;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653)] =
                    x_16663;
            }
        }
        offset_16669 *= 2;
    }
    while (slt32(skip_waves_16670,
                 squot32(sext_i64_i32(segred_group_sizze_14920) +
                         wave_sizze_16655 - 1, wave_sizze_16655))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_16669 = skip_waves_16670 * wave_sizze_16655;
        if (slt32(local_tid_16653 + offset_16669,
                  sext_i64_i32(segred_group_sizze_14920)) && ((local_tid_16653 -
                                                               squot32(local_tid_16653,
                                                                       wave_sizze_16655) *
                                                               wave_sizze_16655) ==
                                                              0 &&
                                                              (squot32(local_tid_16653,
                                                                       wave_sizze_16655) &
                                                               (2 *
                                                                skip_waves_16670 -
                                                                1)) == 0)) {
            // read array element
            {
                x_16664 = ((__local
                            float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653 +
                                                        offset_16669)];
            }
            // apply reduction operation
            {
                float res_16665 = x_16663 + x_16664;
                
                x_16663 = res_16665;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653)] =
                    x_16663;
            }
        }
        skip_waves_16670 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_16653) == 0) {
            x_acc_16661 = x_16663;
        }
    }
    
    int32_t old_counter_16671;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_16653 == 0) {
            ((__global
              float *) group_res_arr_mem_16649)[sext_i32_i64(group_tid_16654) *
                                                segred_group_sizze_14920] =
                x_acc_16661;
            mem_fence_global();
            old_counter_16671 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16647)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_16657)[0] = old_counter_16671 ==
                num_groups_14921 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_16672;
    
    is_last_group_16672 = ((__local bool *) sync_arr_mem_16657)[0];
    if (is_last_group_16672) {
        if (local_tid_16653 == 0) {
            old_counter_16671 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16647)[0],
                                                      (int) (0 -
                                                             num_groups_14921));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_16673 = sdiv_up64(num_groups_14921,
                                                      segred_group_sizze_14920);
            
            x_14925 = 0.0F;
            for (int64_t i_16674 = 0; i_16674 < read_per_thread_16673;
                 i_16674++) {
                int64_t group_res_id_16675 = sext_i32_i64(local_tid_16653) *
                        read_per_thread_16673 + i_16674;
                int64_t index_of_group_res_16676 = group_res_id_16675;
                
                if (slt64(group_res_id_16675, num_groups_14921)) {
                    x_14926 = ((__global
                                float *) group_res_arr_mem_16649)[index_of_group_res_16676 *
                                                                  segred_group_sizze_14920];
                    
                    float res_14927;
                    
                    res_14927 = x_14925 + x_14926;
                    x_14925 = res_14927;
                }
            }
        }
        ((__local float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653)] =
            x_14925;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_16677;
            int32_t skip_waves_16678;
            
            skip_waves_16678 = 1;
            
            float x_16663;
            float x_16664;
            
            offset_16677 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_16653,
                          sext_i64_i32(segred_group_sizze_14920))) {
                    x_16663 = ((__local
                                float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653 +
                                                            offset_16677)];
                }
            }
            offset_16677 = 1;
            while (slt32(offset_16677, wave_sizze_16655)) {
                if (slt32(local_tid_16653 + offset_16677,
                          sext_i64_i32(segred_group_sizze_14920)) &&
                    ((local_tid_16653 - squot32(local_tid_16653,
                                                wave_sizze_16655) *
                      wave_sizze_16655) & (2 * offset_16677 - 1)) == 0) {
                    // read array element
                    {
                        x_16664 = ((volatile __local
                                    float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653 +
                                                                offset_16677)];
                    }
                    // apply reduction operation
                    {
                        float res_16665 = x_16663 + x_16664;
                        
                        x_16663 = res_16665;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653)] =
                            x_16663;
                    }
                }
                offset_16677 *= 2;
            }
            while (slt32(skip_waves_16678,
                         squot32(sext_i64_i32(segred_group_sizze_14920) +
                                 wave_sizze_16655 - 1, wave_sizze_16655))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_16677 = skip_waves_16678 * wave_sizze_16655;
                if (slt32(local_tid_16653 + offset_16677,
                          sext_i64_i32(segred_group_sizze_14920)) &&
                    ((local_tid_16653 - squot32(local_tid_16653,
                                                wave_sizze_16655) *
                      wave_sizze_16655) == 0 && (squot32(local_tid_16653,
                                                         wave_sizze_16655) &
                                                 (2 * skip_waves_16678 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_16664 = ((__local
                                    float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653 +
                                                                offset_16677)];
                    }
                    // apply reduction operation
                    {
                        float res_16665 = x_16663 + x_16664;
                        
                        x_16663 = res_16665;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_16659)[sext_i32_i64(local_tid_16653)] =
                            x_16663;
                    }
                }
                skip_waves_16678 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_16653 == 0) {
                    ((__global float *) mem_16405)[0] = x_16663;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_14920
}
__kernel void mainzisegred_nonseg_15786(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_16778_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_16776_backing_aligned_1,
                                        int64_t steps_13654,
                                        int64_t num_groups_16214, __global
                                        unsigned char *res_map_acc_mem_16429,
                                        __global unsigned char *mem_16433,
                                        __global
                                        unsigned char *mainzicounter_mem_16766,
                                        __global
                                        unsigned char *group_res_arr_mem_16768,
                                        int64_t num_threads_16770)
{
    #define segred_group_sizze_16213 (mainzisegred_group_sizze_15778)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16778_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_16778_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_16776_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_16776_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16771;
    int32_t local_tid_16772;
    int64_t group_sizze_16775;
    int32_t wave_sizze_16774;
    int32_t group_tid_16773;
    
    global_tid_16771 = get_global_id(0);
    local_tid_16772 = get_local_id(0);
    group_sizze_16775 = get_local_size(0);
    wave_sizze_16774 = LOCKSTEP_WIDTH;
    group_tid_16773 = get_group_id(0);
    
    int32_t phys_tid_15786;
    
    phys_tid_15786 = global_tid_16771;
    
    __local char *sync_arr_mem_16776;
    
    sync_arr_mem_16776 = (__local char *) sync_arr_mem_16776_backing_0;
    
    __local char *red_arr_mem_16778;
    
    red_arr_mem_16778 = (__local char *) red_arr_mem_16778_backing_1;
    
    int64_t dummy_15784;
    
    dummy_15784 = 0;
    
    int64_t gtid_15785;
    
    gtid_15785 = 0;
    
    float x_acc_16780;
    int64_t chunk_sizze_16781;
    
    chunk_sizze_16781 = smin64(sdiv_up64(steps_13654,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_16213 *
                                         num_groups_16214))),
                               sdiv_up64(steps_13654 -
                                         sext_i32_i64(phys_tid_15786),
                                         num_threads_16770));
    
    float x_16217;
    float x_16218;
    
    // neutral-initialise the accumulators
    {
        x_acc_16780 = 0.0F;
    }
    for (int64_t i_16785 = 0; i_16785 < chunk_sizze_16781; i_16785++) {
        gtid_15785 = sext_i32_i64(phys_tid_15786) + num_threads_16770 * i_16785;
        // apply map function
        {
            float x_16220 = ((__global
                              float *) res_map_acc_mem_16429)[gtid_15785];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_16217 = x_acc_16780;
            }
            // load new values
            {
                x_16218 = x_16220;
            }
            // apply reduction operator
            {
                float res_16219 = x_16217 + x_16218;
                
                // store in accumulator
                {
                    x_acc_16780 = res_16219;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_16217 = x_acc_16780;
        ((__local float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772)] =
            x_16217;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_16786;
    int32_t skip_waves_16787;
    
    skip_waves_16787 = 1;
    
    float x_16782;
    float x_16783;
    
    offset_16786 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_16772, sext_i64_i32(segred_group_sizze_16213))) {
            x_16782 = ((__local
                        float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772 +
                                                    offset_16786)];
        }
    }
    offset_16786 = 1;
    while (slt32(offset_16786, wave_sizze_16774)) {
        if (slt32(local_tid_16772 + offset_16786,
                  sext_i64_i32(segred_group_sizze_16213)) && ((local_tid_16772 -
                                                               squot32(local_tid_16772,
                                                                       wave_sizze_16774) *
                                                               wave_sizze_16774) &
                                                              (2 *
                                                               offset_16786 -
                                                               1)) == 0) {
            // read array element
            {
                x_16783 = ((volatile __local
                            float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772 +
                                                        offset_16786)];
            }
            // apply reduction operation
            {
                float res_16784 = x_16782 + x_16783;
                
                x_16782 = res_16784;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772)] =
                    x_16782;
            }
        }
        offset_16786 *= 2;
    }
    while (slt32(skip_waves_16787,
                 squot32(sext_i64_i32(segred_group_sizze_16213) +
                         wave_sizze_16774 - 1, wave_sizze_16774))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_16786 = skip_waves_16787 * wave_sizze_16774;
        if (slt32(local_tid_16772 + offset_16786,
                  sext_i64_i32(segred_group_sizze_16213)) && ((local_tid_16772 -
                                                               squot32(local_tid_16772,
                                                                       wave_sizze_16774) *
                                                               wave_sizze_16774) ==
                                                              0 &&
                                                              (squot32(local_tid_16772,
                                                                       wave_sizze_16774) &
                                                               (2 *
                                                                skip_waves_16787 -
                                                                1)) == 0)) {
            // read array element
            {
                x_16783 = ((__local
                            float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772 +
                                                        offset_16786)];
            }
            // apply reduction operation
            {
                float res_16784 = x_16782 + x_16783;
                
                x_16782 = res_16784;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772)] =
                    x_16782;
            }
        }
        skip_waves_16787 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_16772) == 0) {
            x_acc_16780 = x_16782;
        }
    }
    
    int32_t old_counter_16788;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_16772 == 0) {
            ((__global
              float *) group_res_arr_mem_16768)[sext_i32_i64(group_tid_16773) *
                                                segred_group_sizze_16213] =
                x_acc_16780;
            mem_fence_global();
            old_counter_16788 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16766)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_16776)[0] = old_counter_16788 ==
                num_groups_16214 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_16789;
    
    is_last_group_16789 = ((__local bool *) sync_arr_mem_16776)[0];
    if (is_last_group_16789) {
        if (local_tid_16772 == 0) {
            old_counter_16788 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16766)[0],
                                                      (int) (0 -
                                                             num_groups_16214));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_16790 = sdiv_up64(num_groups_16214,
                                                      segred_group_sizze_16213);
            
            x_16217 = 0.0F;
            for (int64_t i_16791 = 0; i_16791 < read_per_thread_16790;
                 i_16791++) {
                int64_t group_res_id_16792 = sext_i32_i64(local_tid_16772) *
                        read_per_thread_16790 + i_16791;
                int64_t index_of_group_res_16793 = group_res_id_16792;
                
                if (slt64(group_res_id_16792, num_groups_16214)) {
                    x_16218 = ((__global
                                float *) group_res_arr_mem_16768)[index_of_group_res_16793 *
                                                                  segred_group_sizze_16213];
                    
                    float res_16219;
                    
                    res_16219 = x_16217 + x_16218;
                    x_16217 = res_16219;
                }
            }
        }
        ((__local float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772)] =
            x_16217;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_16794;
            int32_t skip_waves_16795;
            
            skip_waves_16795 = 1;
            
            float x_16782;
            float x_16783;
            
            offset_16794 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_16772,
                          sext_i64_i32(segred_group_sizze_16213))) {
                    x_16782 = ((__local
                                float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772 +
                                                            offset_16794)];
                }
            }
            offset_16794 = 1;
            while (slt32(offset_16794, wave_sizze_16774)) {
                if (slt32(local_tid_16772 + offset_16794,
                          sext_i64_i32(segred_group_sizze_16213)) &&
                    ((local_tid_16772 - squot32(local_tid_16772,
                                                wave_sizze_16774) *
                      wave_sizze_16774) & (2 * offset_16794 - 1)) == 0) {
                    // read array element
                    {
                        x_16783 = ((volatile __local
                                    float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772 +
                                                                offset_16794)];
                    }
                    // apply reduction operation
                    {
                        float res_16784 = x_16782 + x_16783;
                        
                        x_16782 = res_16784;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772)] =
                            x_16782;
                    }
                }
                offset_16794 *= 2;
            }
            while (slt32(skip_waves_16795,
                         squot32(sext_i64_i32(segred_group_sizze_16213) +
                                 wave_sizze_16774 - 1, wave_sizze_16774))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_16794 = skip_waves_16795 * wave_sizze_16774;
                if (slt32(local_tid_16772 + offset_16794,
                          sext_i64_i32(segred_group_sizze_16213)) &&
                    ((local_tid_16772 - squot32(local_tid_16772,
                                                wave_sizze_16774) *
                      wave_sizze_16774) == 0 && (squot32(local_tid_16772,
                                                         wave_sizze_16774) &
                                                 (2 * skip_waves_16795 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_16783 = ((__local
                                    float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772 +
                                                                offset_16794)];
                    }
                    // apply reduction operation
                    {
                        float res_16784 = x_16782 + x_16783;
                        
                        x_16782 = res_16784;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_16778)[sext_i32_i64(local_tid_16772)] =
                            x_16782;
                    }
                }
                skip_waves_16795 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_16772 == 0) {
                    ((__global float *) mem_16433)[0] = x_16782;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_16213
}
__kernel void mainzisegred_nonseg_16455(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_16559_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_16557_backing_aligned_1,
                                        int64_t n_13650, __global
                                        unsigned char *payments_mem_16320,
                                        __global unsigned char *mem_16467,
                                        __global
                                        unsigned char *mainzicounter_mem_16547,
                                        __global
                                        unsigned char *group_res_arr_mem_16549,
                                        int64_t num_threads_16551)
{
    #define segred_num_groups_16449 (mainzisegred_num_groups_16448)
    #define segred_group_sizze_16451 (mainzisegred_group_sizze_16450)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16559_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_16559_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_16557_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_16557_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16552;
    int32_t local_tid_16553;
    int64_t group_sizze_16556;
    int32_t wave_sizze_16555;
    int32_t group_tid_16554;
    
    global_tid_16552 = get_global_id(0);
    local_tid_16553 = get_local_id(0);
    group_sizze_16556 = get_local_size(0);
    wave_sizze_16555 = LOCKSTEP_WIDTH;
    group_tid_16554 = get_group_id(0);
    
    int32_t phys_tid_16455;
    
    phys_tid_16455 = global_tid_16552;
    
    __local char *sync_arr_mem_16557;
    
    sync_arr_mem_16557 = (__local char *) sync_arr_mem_16557_backing_0;
    
    __local char *red_arr_mem_16559;
    
    red_arr_mem_16559 = (__local char *) red_arr_mem_16559_backing_1;
    
    int64_t dummy_16453;
    
    dummy_16453 = 0;
    
    int64_t gtid_16454;
    
    gtid_16454 = 0;
    
    int64_t x_acc_16561;
    int64_t chunk_sizze_16562;
    
    chunk_sizze_16562 = smin64(sdiv_up64(n_13650,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_16451 *
                                         segred_num_groups_16449))),
                               sdiv_up64(n_13650 - sext_i32_i64(phys_tid_16455),
                                         num_threads_16551));
    
    int64_t x_16456;
    int64_t y_16457;
    
    // neutral-initialise the accumulators
    {
        x_acc_16561 = 0;
    }
    for (int64_t i_16566 = 0; i_16566 < chunk_sizze_16562; i_16566++) {
        gtid_16454 = sext_i32_i64(phys_tid_16455) + num_threads_16551 * i_16566;
        // apply map function
        {
            int64_t res_16460 = ((__global
                                  int64_t *) payments_mem_16320)[gtid_16454];
            int64_t bytes_16461 = 4 * res_16460;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_16456 = x_acc_16561;
            }
            // load new values
            {
                y_16457 = bytes_16461;
            }
            // apply reduction operator
            {
                int64_t zz_16458 = smax64(x_16456, y_16457);
                
                // store in accumulator
                {
                    x_acc_16561 = zz_16458;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_16456 = x_acc_16561;
        ((__local int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553)] =
            x_16456;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_16567;
    int32_t skip_waves_16568;
    
    skip_waves_16568 = 1;
    
    int64_t x_16563;
    int64_t y_16564;
    
    offset_16567 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_16553, sext_i64_i32(segred_group_sizze_16451))) {
            x_16563 = ((__local
                        int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553 +
                                                      offset_16567)];
        }
    }
    offset_16567 = 1;
    while (slt32(offset_16567, wave_sizze_16555)) {
        if (slt32(local_tid_16553 + offset_16567,
                  sext_i64_i32(segred_group_sizze_16451)) && ((local_tid_16553 -
                                                               squot32(local_tid_16553,
                                                                       wave_sizze_16555) *
                                                               wave_sizze_16555) &
                                                              (2 *
                                                               offset_16567 -
                                                               1)) == 0) {
            // read array element
            {
                y_16564 = ((volatile __local
                            int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553 +
                                                          offset_16567)];
            }
            // apply reduction operation
            {
                int64_t zz_16565 = smax64(x_16563, y_16564);
                
                x_16563 = zz_16565;
            }
            // write result of operation
            {
                ((volatile __local
                  int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553)] =
                    x_16563;
            }
        }
        offset_16567 *= 2;
    }
    while (slt32(skip_waves_16568,
                 squot32(sext_i64_i32(segred_group_sizze_16451) +
                         wave_sizze_16555 - 1, wave_sizze_16555))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_16567 = skip_waves_16568 * wave_sizze_16555;
        if (slt32(local_tid_16553 + offset_16567,
                  sext_i64_i32(segred_group_sizze_16451)) && ((local_tid_16553 -
                                                               squot32(local_tid_16553,
                                                                       wave_sizze_16555) *
                                                               wave_sizze_16555) ==
                                                              0 &&
                                                              (squot32(local_tid_16553,
                                                                       wave_sizze_16555) &
                                                               (2 *
                                                                skip_waves_16568 -
                                                                1)) == 0)) {
            // read array element
            {
                y_16564 = ((__local
                            int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553 +
                                                          offset_16567)];
            }
            // apply reduction operation
            {
                int64_t zz_16565 = smax64(x_16563, y_16564);
                
                x_16563 = zz_16565;
            }
            // write result of operation
            {
                ((__local
                  int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553)] =
                    x_16563;
            }
        }
        skip_waves_16568 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_16553) == 0) {
            x_acc_16561 = x_16563;
        }
    }
    
    int32_t old_counter_16569;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_16553 == 0) {
            ((__global
              int64_t *) group_res_arr_mem_16549)[sext_i32_i64(group_tid_16554) *
                                                  segred_group_sizze_16451] =
                x_acc_16561;
            mem_fence_global();
            old_counter_16569 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16547)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_16557)[0] = old_counter_16569 ==
                segred_num_groups_16449 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_16570;
    
    is_last_group_16570 = ((__local bool *) sync_arr_mem_16557)[0];
    if (is_last_group_16570) {
        if (local_tid_16553 == 0) {
            old_counter_16569 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16547)[0],
                                                      (int) (0 -
                                                             segred_num_groups_16449));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_16571 = sdiv_up64(segred_num_groups_16449,
                                                      segred_group_sizze_16451);
            
            x_16456 = 0;
            for (int64_t i_16572 = 0; i_16572 < read_per_thread_16571;
                 i_16572++) {
                int64_t group_res_id_16573 = sext_i32_i64(local_tid_16553) *
                        read_per_thread_16571 + i_16572;
                int64_t index_of_group_res_16574 = group_res_id_16573;
                
                if (slt64(group_res_id_16573, segred_num_groups_16449)) {
                    y_16457 = ((__global
                                int64_t *) group_res_arr_mem_16549)[index_of_group_res_16574 *
                                                                    segred_group_sizze_16451];
                    
                    int64_t zz_16458;
                    
                    zz_16458 = smax64(x_16456, y_16457);
                    x_16456 = zz_16458;
                }
            }
        }
        ((__local int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553)] =
            x_16456;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_16575;
            int32_t skip_waves_16576;
            
            skip_waves_16576 = 1;
            
            int64_t x_16563;
            int64_t y_16564;
            
            offset_16575 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_16553,
                          sext_i64_i32(segred_group_sizze_16451))) {
                    x_16563 = ((__local
                                int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553 +
                                                              offset_16575)];
                }
            }
            offset_16575 = 1;
            while (slt32(offset_16575, wave_sizze_16555)) {
                if (slt32(local_tid_16553 + offset_16575,
                          sext_i64_i32(segred_group_sizze_16451)) &&
                    ((local_tid_16553 - squot32(local_tid_16553,
                                                wave_sizze_16555) *
                      wave_sizze_16555) & (2 * offset_16575 - 1)) == 0) {
                    // read array element
                    {
                        y_16564 = ((volatile __local
                                    int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553 +
                                                                  offset_16575)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_16565 = smax64(x_16563, y_16564);
                        
                        x_16563 = zz_16565;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553)] =
                            x_16563;
                    }
                }
                offset_16575 *= 2;
            }
            while (slt32(skip_waves_16576,
                         squot32(sext_i64_i32(segred_group_sizze_16451) +
                                 wave_sizze_16555 - 1, wave_sizze_16555))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_16575 = skip_waves_16576 * wave_sizze_16555;
                if (slt32(local_tid_16553 + offset_16575,
                          sext_i64_i32(segred_group_sizze_16451)) &&
                    ((local_tid_16553 - squot32(local_tid_16553,
                                                wave_sizze_16555) *
                      wave_sizze_16555) == 0 && (squot32(local_tid_16553,
                                                         wave_sizze_16555) &
                                                 (2 * skip_waves_16576 - 1)) ==
                     0)) {
                    // read array element
                    {
                        y_16564 = ((__local
                                    int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553 +
                                                                  offset_16575)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_16565 = smax64(x_16563, y_16564);
                        
                        x_16563 = zz_16565;
                    }
                    // write result of operation
                    {
                        ((__local
                          int64_t *) red_arr_mem_16559)[sext_i32_i64(local_tid_16553)] =
                            x_16563;
                    }
                }
                skip_waves_16576 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_16553 == 0) {
                    ((__global int64_t *) mem_16467)[0] = x_16563;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_num_groups_16449
    #undef segred_group_sizze_16451
}
__kernel void mainzisegred_nonseg_16487(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_16596_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_16594_backing_aligned_1,
                                        int64_t n_13650, __global
                                        unsigned char *payments_mem_16320,
                                        __global unsigned char *mem_16499,
                                        __global
                                        unsigned char *mainzicounter_mem_16584,
                                        __global
                                        unsigned char *group_res_arr_mem_16586,
                                        int64_t num_threads_16588)
{
    #define segred_num_groups_16481 (mainzisegred_num_groups_16480)
    #define segred_group_sizze_16483 (mainzisegred_group_sizze_16482)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16596_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_16596_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_16594_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_16594_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16589;
    int32_t local_tid_16590;
    int64_t group_sizze_16593;
    int32_t wave_sizze_16592;
    int32_t group_tid_16591;
    
    global_tid_16589 = get_global_id(0);
    local_tid_16590 = get_local_id(0);
    group_sizze_16593 = get_local_size(0);
    wave_sizze_16592 = LOCKSTEP_WIDTH;
    group_tid_16591 = get_group_id(0);
    
    int32_t phys_tid_16487;
    
    phys_tid_16487 = global_tid_16589;
    
    __local char *sync_arr_mem_16594;
    
    sync_arr_mem_16594 = (__local char *) sync_arr_mem_16594_backing_0;
    
    __local char *red_arr_mem_16596;
    
    red_arr_mem_16596 = (__local char *) red_arr_mem_16596_backing_1;
    
    int64_t dummy_16485;
    
    dummy_16485 = 0;
    
    int64_t gtid_16486;
    
    gtid_16486 = 0;
    
    int64_t x_acc_16598;
    int64_t chunk_sizze_16599;
    
    chunk_sizze_16599 = smin64(sdiv_up64(n_13650,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_16483 *
                                         segred_num_groups_16481))),
                               sdiv_up64(n_13650 - sext_i32_i64(phys_tid_16487),
                                         num_threads_16588));
    
    int64_t x_16488;
    int64_t y_16489;
    
    // neutral-initialise the accumulators
    {
        x_acc_16598 = 0;
    }
    for (int64_t i_16603 = 0; i_16603 < chunk_sizze_16599; i_16603++) {
        gtid_16486 = sext_i32_i64(phys_tid_16487) + num_threads_16588 * i_16603;
        // apply map function
        {
            int64_t res_16492 = ((__global
                                  int64_t *) payments_mem_16320)[gtid_16486];
            int64_t bytes_16493 = 4 * res_16492;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_16488 = x_acc_16598;
            }
            // load new values
            {
                y_16489 = bytes_16493;
            }
            // apply reduction operator
            {
                int64_t zz_16490 = smax64(x_16488, y_16489);
                
                // store in accumulator
                {
                    x_acc_16598 = zz_16490;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_16488 = x_acc_16598;
        ((__local int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590)] =
            x_16488;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_16604;
    int32_t skip_waves_16605;
    
    skip_waves_16605 = 1;
    
    int64_t x_16600;
    int64_t y_16601;
    
    offset_16604 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_16590, sext_i64_i32(segred_group_sizze_16483))) {
            x_16600 = ((__local
                        int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590 +
                                                      offset_16604)];
        }
    }
    offset_16604 = 1;
    while (slt32(offset_16604, wave_sizze_16592)) {
        if (slt32(local_tid_16590 + offset_16604,
                  sext_i64_i32(segred_group_sizze_16483)) && ((local_tid_16590 -
                                                               squot32(local_tid_16590,
                                                                       wave_sizze_16592) *
                                                               wave_sizze_16592) &
                                                              (2 *
                                                               offset_16604 -
                                                               1)) == 0) {
            // read array element
            {
                y_16601 = ((volatile __local
                            int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590 +
                                                          offset_16604)];
            }
            // apply reduction operation
            {
                int64_t zz_16602 = smax64(x_16600, y_16601);
                
                x_16600 = zz_16602;
            }
            // write result of operation
            {
                ((volatile __local
                  int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590)] =
                    x_16600;
            }
        }
        offset_16604 *= 2;
    }
    while (slt32(skip_waves_16605,
                 squot32(sext_i64_i32(segred_group_sizze_16483) +
                         wave_sizze_16592 - 1, wave_sizze_16592))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_16604 = skip_waves_16605 * wave_sizze_16592;
        if (slt32(local_tid_16590 + offset_16604,
                  sext_i64_i32(segred_group_sizze_16483)) && ((local_tid_16590 -
                                                               squot32(local_tid_16590,
                                                                       wave_sizze_16592) *
                                                               wave_sizze_16592) ==
                                                              0 &&
                                                              (squot32(local_tid_16590,
                                                                       wave_sizze_16592) &
                                                               (2 *
                                                                skip_waves_16605 -
                                                                1)) == 0)) {
            // read array element
            {
                y_16601 = ((__local
                            int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590 +
                                                          offset_16604)];
            }
            // apply reduction operation
            {
                int64_t zz_16602 = smax64(x_16600, y_16601);
                
                x_16600 = zz_16602;
            }
            // write result of operation
            {
                ((__local
                  int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590)] =
                    x_16600;
            }
        }
        skip_waves_16605 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_16590) == 0) {
            x_acc_16598 = x_16600;
        }
    }
    
    int32_t old_counter_16606;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_16590 == 0) {
            ((__global
              int64_t *) group_res_arr_mem_16586)[sext_i32_i64(group_tid_16591) *
                                                  segred_group_sizze_16483] =
                x_acc_16598;
            mem_fence_global();
            old_counter_16606 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16584)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_16594)[0] = old_counter_16606 ==
                segred_num_groups_16481 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_16607;
    
    is_last_group_16607 = ((__local bool *) sync_arr_mem_16594)[0];
    if (is_last_group_16607) {
        if (local_tid_16590 == 0) {
            old_counter_16606 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_16584)[0],
                                                      (int) (0 -
                                                             segred_num_groups_16481));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_16608 = sdiv_up64(segred_num_groups_16481,
                                                      segred_group_sizze_16483);
            
            x_16488 = 0;
            for (int64_t i_16609 = 0; i_16609 < read_per_thread_16608;
                 i_16609++) {
                int64_t group_res_id_16610 = sext_i32_i64(local_tid_16590) *
                        read_per_thread_16608 + i_16609;
                int64_t index_of_group_res_16611 = group_res_id_16610;
                
                if (slt64(group_res_id_16610, segred_num_groups_16481)) {
                    y_16489 = ((__global
                                int64_t *) group_res_arr_mem_16586)[index_of_group_res_16611 *
                                                                    segred_group_sizze_16483];
                    
                    int64_t zz_16490;
                    
                    zz_16490 = smax64(x_16488, y_16489);
                    x_16488 = zz_16490;
                }
            }
        }
        ((__local int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590)] =
            x_16488;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_16612;
            int32_t skip_waves_16613;
            
            skip_waves_16613 = 1;
            
            int64_t x_16600;
            int64_t y_16601;
            
            offset_16612 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_16590,
                          sext_i64_i32(segred_group_sizze_16483))) {
                    x_16600 = ((__local
                                int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590 +
                                                              offset_16612)];
                }
            }
            offset_16612 = 1;
            while (slt32(offset_16612, wave_sizze_16592)) {
                if (slt32(local_tid_16590 + offset_16612,
                          sext_i64_i32(segred_group_sizze_16483)) &&
                    ((local_tid_16590 - squot32(local_tid_16590,
                                                wave_sizze_16592) *
                      wave_sizze_16592) & (2 * offset_16612 - 1)) == 0) {
                    // read array element
                    {
                        y_16601 = ((volatile __local
                                    int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590 +
                                                                  offset_16612)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_16602 = smax64(x_16600, y_16601);
                        
                        x_16600 = zz_16602;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590)] =
                            x_16600;
                    }
                }
                offset_16612 *= 2;
            }
            while (slt32(skip_waves_16613,
                         squot32(sext_i64_i32(segred_group_sizze_16483) +
                                 wave_sizze_16592 - 1, wave_sizze_16592))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_16612 = skip_waves_16613 * wave_sizze_16592;
                if (slt32(local_tid_16590 + offset_16612,
                          sext_i64_i32(segred_group_sizze_16483)) &&
                    ((local_tid_16590 - squot32(local_tid_16590,
                                                wave_sizze_16592) *
                      wave_sizze_16592) == 0 && (squot32(local_tid_16590,
                                                         wave_sizze_16592) &
                                                 (2 * skip_waves_16613 - 1)) ==
                     0)) {
                    // read array element
                    {
                        y_16601 = ((__local
                                    int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590 +
                                                                  offset_16612)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_16602 = smax64(x_16600, y_16601);
                        
                        x_16600 = zz_16602;
                    }
                    // write result of operation
                    {
                        ((__local
                          int64_t *) red_arr_mem_16596)[sext_i32_i64(local_tid_16590)] =
                            x_16600;
                    }
                }
                skip_waves_16613 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_16590 == 0) {
                    ((__global int64_t *) mem_16499)[0] = x_16600;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_num_groups_16481
    #undef segred_group_sizze_16483
}
__kernel void mainzisegred_small_15512(__global int *global_failure,
                                       int failure_is_an_option, __global
                                       int64_t *global_failure_args,
                                       __local volatile
                                       int64_t *red_arr_mem_16704_backing_aligned_0,
                                       int64_t n_13650, int64_t paths_13653,
                                       int64_t steps_13654, float a_13658,
                                       float b_13659, float sigma_13660,
                                       float sims_per_year_13760,
                                       float last_date_13775,
                                       int64_t num_groups_16075, __global
                                       unsigned char *res_mem_16371, __global
                                       unsigned char *res_mem_16372, __global
                                       unsigned char *res_mem_16373, __global
                                       unsigned char *res_mem_16374, __global
                                       unsigned char *mem_16402, __global
                                       unsigned char *mem_16421,
                                       int64_t segment_sizze_nonzzero_16697)
{
    #define segred_group_sizze_16074 (mainzisegred_group_sizze_15506)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_16704_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_16704_backing_aligned_0;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16699;
    int32_t local_tid_16700;
    int64_t group_sizze_16703;
    int32_t wave_sizze_16702;
    int32_t group_tid_16701;
    
    global_tid_16699 = get_global_id(0);
    local_tid_16700 = get_local_id(0);
    group_sizze_16703 = get_local_size(0);
    wave_sizze_16702 = LOCKSTEP_WIDTH;
    group_tid_16701 = get_group_id(0);
    
    int32_t phys_tid_15512;
    
    phys_tid_15512 = global_tid_16699;
    
    __local char *red_arr_mem_16704;
    
    red_arr_mem_16704 = (__local char *) red_arr_mem_16704_backing_0;
    
    int32_t phys_group_id_16706;
    
    phys_group_id_16706 = get_group_id(0);
    for (int32_t i_16707 = 0; i_16707 <
         sdiv_up32(sext_i64_i32(sdiv_up64(steps_13654,
                                          squot64(segred_group_sizze_16074,
                                                  segment_sizze_nonzzero_16697))) -
                   phys_group_id_16706, sext_i64_i32(num_groups_16075));
         i_16707++) {
        int32_t virt_group_id_16708 = phys_group_id_16706 + i_16707 *
                sext_i64_i32(num_groups_16075);
        int64_t gtid_15503 = squot64(sext_i32_i64(local_tid_16700),
                                     segment_sizze_nonzzero_16697) +
                sext_i32_i64(virt_group_id_16708) *
                squot64(segred_group_sizze_16074, segment_sizze_nonzzero_16697);
        int64_t gtid_15511 = srem64(sext_i32_i64(local_tid_16700), paths_13653);
        
        // apply map function if in bounds
        {
            if (slt64(0, paths_13653) && (slt64(gtid_15503, steps_13654) &&
                                          slt64(sext_i32_i64(local_tid_16700),
                                                paths_13653 *
                                                squot64(segred_group_sizze_16074,
                                                        segment_sizze_nonzzero_16697)))) {
                int64_t convop_x_16256 = add64(1, gtid_15503);
                float binop_x_16257 = sitofp_i64_f32(convop_x_16256);
                float index_primexp_16258 = binop_x_16257 / sims_per_year_13760;
                bool cond_16085 = last_date_13775 < index_primexp_16258;
                float max_arg_16086;
                
                if (cond_16085) {
                    max_arg_16086 = 0.0F;
                } else {
                    float x_16083 = ((__global float *) mem_16402)[gtid_15503 *
                                                                   paths_13653 +
                                                                   gtid_15511];
                    bool y_16087 = slt64(0, n_13650);
                    bool index_certs_16088;
                    
                    if (!y_16087) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          17) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = n_13650;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float swapprice_arg_16089 = ((__global
                                                  float *) res_mem_16371)[0];
                    float swapprice_arg_16090 = ((__global
                                                  float *) res_mem_16372)[0];
                    int64_t swapprice_arg_16091 = ((__global
                                                    int64_t *) res_mem_16373)[0];
                    float swapprice_arg_16092 = ((__global
                                                  float *) res_mem_16374)[0];
                    float ceil_arg_16093 = index_primexp_16258 /
                          swapprice_arg_16092;
                    float res_16094;
                    
                    res_16094 = futrts_ceil32(ceil_arg_16093);
                    
                    float nextpayment_16095 = swapprice_arg_16092 * res_16094;
                    int64_t res_16096 = fptosi_f32_i64(res_16094);
                    int64_t remaining_16097 = sub64(swapprice_arg_16091,
                                                    res_16096);
                    bool bounds_invalid_upwards_16098 = slt64(remaining_16097,
                                                              1);
                    bool valid_16099 = !bounds_invalid_upwards_16098;
                    bool range_valid_c_16100;
                    
                    if (!valid_16099) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          18) == -1) {
                                global_failure_args[0] = 1;
                                global_failure_args[1] = 2;
                                global_failure_args[2] = remaining_16097;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_16102 = nextpayment_16095 - index_primexp_16258;
                    float negate_arg_16103 = a_13658 * y_16102;
                    float exp_arg_16104 = 0.0F - negate_arg_16103;
                    float res_16105 = fpow32(2.7182817F, exp_arg_16104);
                    float x_16106 = 1.0F - res_16105;
                    float B_16107 = x_16106 / a_13658;
                    float x_16108 = B_16107 - nextpayment_16095;
                    float x_16109 = x_16108 + index_primexp_16258;
                    float x_16110 = fpow32(a_13658, 2.0F);
                    float x_16111 = b_13659 * x_16110;
                    float x_16112 = fpow32(sigma_13660, 2.0F);
                    float y_16113 = x_16112 / 2.0F;
                    float y_16114 = x_16111 - y_16113;
                    float x_16115 = x_16109 * y_16114;
                    float A1_16116 = x_16115 / x_16110;
                    float y_16117 = fpow32(B_16107, 2.0F);
                    float x_16118 = x_16112 * y_16117;
                    float y_16119 = 4.0F * a_13658;
                    float A2_16120 = x_16118 / y_16119;
                    float exp_arg_16121 = A1_16116 - A2_16120;
                    float res_16122 = fpow32(2.7182817F, exp_arg_16121);
                    float negate_arg_16123 = x_16083 * B_16107;
                    float exp_arg_16124 = 0.0F - negate_arg_16123;
                    float res_16125 = fpow32(2.7182817F, exp_arg_16124);
                    float res_16126 = res_16122 * res_16125;
                    bool y_16127 = slt64(0, remaining_16097);
                    bool index_certs_16128;
                    
                    if (!y_16127) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          19) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = remaining_16097;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float binop_y_16129 = sitofp_i64_f32(remaining_16097);
                    float binop_y_16130 = swapprice_arg_16092 * binop_y_16129;
                    float index_primexp_16131 = nextpayment_16095 +
                          binop_y_16130;
                    float y_16132 = index_primexp_16131 - index_primexp_16258;
                    float negate_arg_16133 = a_13658 * y_16132;
                    float exp_arg_16134 = 0.0F - negate_arg_16133;
                    float res_16135 = fpow32(2.7182817F, exp_arg_16134);
                    float x_16136 = 1.0F - res_16135;
                    float B_16137 = x_16136 / a_13658;
                    float x_16138 = B_16137 - index_primexp_16131;
                    float x_16139 = x_16138 + index_primexp_16258;
                    float x_16140 = y_16114 * x_16139;
                    float A1_16141 = x_16140 / x_16110;
                    float y_16142 = fpow32(B_16137, 2.0F);
                    float x_16143 = x_16112 * y_16142;
                    float A2_16144 = x_16143 / y_16119;
                    float exp_arg_16145 = A1_16141 - A2_16144;
                    float res_16146 = fpow32(2.7182817F, exp_arg_16145);
                    float negate_arg_16147 = x_16083 * B_16137;
                    float exp_arg_16148 = 0.0F - negate_arg_16147;
                    float res_16149 = fpow32(2.7182817F, exp_arg_16148);
                    float res_16150 = res_16146 * res_16149;
                    float res_16151;
                    float redout_16253 = 0.0F;
                    
                    for (int64_t i_16254 = 0; i_16254 < remaining_16097;
                         i_16254++) {
                        int64_t index_primexp_16276 = add64(1, i_16254);
                        float res_16156 = sitofp_i64_f32(index_primexp_16276);
                        float res_16157 = swapprice_arg_16092 * res_16156;
                        float res_16158 = nextpayment_16095 + res_16157;
                        float y_16159 = res_16158 - index_primexp_16258;
                        float negate_arg_16160 = a_13658 * y_16159;
                        float exp_arg_16161 = 0.0F - negate_arg_16160;
                        float res_16162 = fpow32(2.7182817F, exp_arg_16161);
                        float x_16163 = 1.0F - res_16162;
                        float B_16164 = x_16163 / a_13658;
                        float x_16165 = B_16164 - res_16158;
                        float x_16166 = x_16165 + index_primexp_16258;
                        float x_16167 = y_16114 * x_16166;
                        float A1_16168 = x_16167 / x_16110;
                        float y_16169 = fpow32(B_16164, 2.0F);
                        float x_16170 = x_16112 * y_16169;
                        float A2_16171 = x_16170 / y_16119;
                        float exp_arg_16172 = A1_16168 - A2_16171;
                        float res_16173 = fpow32(2.7182817F, exp_arg_16172);
                        float negate_arg_16174 = x_16083 * B_16164;
                        float exp_arg_16175 = 0.0F - negate_arg_16174;
                        float res_16176 = fpow32(2.7182817F, exp_arg_16175);
                        float res_16177 = res_16173 * res_16176;
                        float res_16154 = res_16177 + redout_16253;
                        float redout_tmp_16709 = res_16154;
                        
                        redout_16253 = redout_tmp_16709;
                    }
                    res_16151 = redout_16253;
                    
                    float x_16178 = res_16126 - res_16150;
                    float x_16179 = swapprice_arg_16089 * swapprice_arg_16092;
                    float y_16180 = res_16151 * x_16179;
                    float y_16181 = x_16178 - y_16180;
                    float res_16182 = swapprice_arg_16090 * y_16181;
                    
                    max_arg_16086 = res_16182;
                }
                
                float res_16183 = fmax32(0.0F, max_arg_16086);
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)] =
                        res_16183;
                }
            } else {
                ((__local
                  float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)] =
                    0.0F;
            }
        }
        
      error_0:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt64(0, paths_13653)) {
            // perform segmented scan to imitate reduction
            {
                float x_16078;
                float x_16079;
                float x_16710;
                float x_16711;
                bool ltid_in_bounds_16713;
                
                ltid_in_bounds_16713 = slt64(sext_i32_i64(local_tid_16700),
                                             paths_13653 *
                                             squot64(segred_group_sizze_16074,
                                                     segment_sizze_nonzzero_16697));
                
                int32_t skip_threads_16714;
                
                // read input for in-block scan
                {
                    if (ltid_in_bounds_16713) {
                        x_16079 = ((volatile __local
                                    float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)];
                        if ((local_tid_16700 - squot32(local_tid_16700, 32) *
                             32) == 0) {
                            x_16078 = x_16079;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16714 = 1;
                    while (slt32(skip_threads_16714, 32)) {
                        if (sle32(skip_threads_16714, local_tid_16700 -
                                  squot32(local_tid_16700, 32) * 32) &&
                            ltid_in_bounds_16713) {
                            // read operands
                            {
                                x_16078 = ((volatile __local
                                            float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700) -
                                                                        sext_i32_i64(skip_threads_16714)];
                            }
                            // perform operation
                            {
                                bool inactive_16715 =
                                     slt64(srem64(sext_i32_i64(local_tid_16700),
                                                  paths_13653),
                                           sext_i32_i64(local_tid_16700) -
                                           sext_i32_i64(local_tid_16700 -
                                           skip_threads_16714));
                                
                                if (inactive_16715) {
                                    x_16078 = x_16079;
                                }
                                if (!inactive_16715) {
                                    float res_16080 = x_16078 + x_16079;
                                    
                                    x_16078 = res_16080;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16702, skip_threads_16714)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16714, local_tid_16700 -
                                  squot32(local_tid_16700, 32) * 32) &&
                            ltid_in_bounds_16713) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)] =
                                    x_16078;
                                x_16079 = x_16078;
                            }
                        }
                        if (sle32(wave_sizze_16702, skip_threads_16714)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16714 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_16700 - squot32(local_tid_16700, 32) * 32) ==
                        31 && ltid_in_bounds_16713) {
                        ((volatile __local
                          float *) red_arr_mem_16704)[sext_i32_i64(squot32(local_tid_16700,
                                                                           32))] =
                            x_16078;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_16716;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_16700, 32) == 0 &&
                            ltid_in_bounds_16713) {
                            x_16711 = ((volatile __local
                                        float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)];
                            if ((local_tid_16700 - squot32(local_tid_16700,
                                                           32) * 32) == 0) {
                                x_16710 = x_16711;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_16716 = 1;
                        while (slt32(skip_threads_16716, 32)) {
                            if (sle32(skip_threads_16716, local_tid_16700 -
                                      squot32(local_tid_16700, 32) * 32) &&
                                (squot32(local_tid_16700, 32) == 0 &&
                                 ltid_in_bounds_16713)) {
                                // read operands
                                {
                                    x_16710 = ((volatile __local
                                                float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700) -
                                                                            sext_i32_i64(skip_threads_16716)];
                                }
                                // perform operation
                                {
                                    bool inactive_16717 =
                                         slt64(srem64(sext_i32_i64(local_tid_16700 *
                                                      32 + 32 - 1),
                                                      paths_13653),
                                               sext_i32_i64(local_tid_16700 *
                                               32 + 32 - 1) -
                                               sext_i32_i64((local_tid_16700 -
                                                             skip_threads_16716) *
                                               32 + 32 - 1));
                                    
                                    if (inactive_16717) {
                                        x_16710 = x_16711;
                                    }
                                    if (!inactive_16717) {
                                        float res_16712 = x_16710 + x_16711;
                                        
                                        x_16710 = res_16712;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_16702, skip_threads_16716)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_16716, local_tid_16700 -
                                      squot32(local_tid_16700, 32) * 32) &&
                                (squot32(local_tid_16700, 32) == 0 &&
                                 ltid_in_bounds_16713)) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)] =
                                        x_16710;
                                    x_16711 = x_16710;
                                }
                            }
                            if (sle32(wave_sizze_16702, skip_threads_16716)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_16716 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_16700, 32) == 0 ||
                          !ltid_in_bounds_16713)) {
                        // read operands
                        {
                            x_16079 = x_16078;
                            x_16078 = ((__local
                                        float *) red_arr_mem_16704)[sext_i32_i64(squot32(local_tid_16700,
                                                                                         32)) -
                                                                    1];
                        }
                        // perform operation
                        {
                            bool inactive_16718 =
                                 slt64(srem64(sext_i32_i64(local_tid_16700),
                                              paths_13653),
                                       sext_i32_i64(local_tid_16700) -
                                       sext_i32_i64(squot32(local_tid_16700,
                                                            32) * 32 - 1));
                            
                            if (inactive_16718) {
                                x_16078 = x_16079;
                            }
                            if (!inactive_16718) {
                                float res_16080 = x_16078 + x_16079;
                                
                                x_16078 = res_16080;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)] =
                                x_16078;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_16700, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_16704)[sext_i32_i64(local_tid_16700)] =
                            x_16079;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt64(sext_i32_i64(virt_group_id_16708) *
                      squot64(segred_group_sizze_16074,
                              segment_sizze_nonzzero_16697) +
                      sext_i32_i64(local_tid_16700), steps_13654) &&
                slt64(sext_i32_i64(local_tid_16700),
                      squot64(segred_group_sizze_16074,
                              segment_sizze_nonzzero_16697))) {
                ((__global
                  float *) mem_16421)[sext_i32_i64(virt_group_id_16708) *
                                      squot64(segred_group_sizze_16074,
                                              segment_sizze_nonzzero_16697) +
                                      sext_i32_i64(local_tid_16700)] = ((__local
                                                                         float *) red_arr_mem_16704)[(sext_i32_i64(local_tid_16700) +
                                                                                                      1) *
                                                                                                     segment_sizze_nonzzero_16697 -
                                                                                                     1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_16074
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
  entry_points = {"main": (["i64", "i64", "[]f32", "[]i64", "[]f32", "f32",
                            "f32", "f32", "f32"], ["f32", "[]f32"])}
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
    self.global_failure_args_max = 3
    self.failure_msgs=["Range {}..{}...{} is invalid.\n-> #0  cva.fut:54:29-52\n   #1  cva.fut:102:25-65\n   #2  cva.fut:116:16-62\n   #3  cva.fut:112:17-116:85\n   #4  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:103:47-70\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-181:20\n",
     "Index [{}:] out of bounds for array of shape [{}].\n-> #0  cva.fut:104:74-90\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-181:20\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:54:29-52\n   #1  cva.fut:102:25-65\n   #2  cva.fut:116:16-62\n   #3  cva.fut:112:17-116:85\n   #4  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:103:47-70\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-181:20\n",
     "Index [{}:] out of bounds for array of shape [{}].\n-> #0  cva.fut:104:74-90\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:71:97-104\n   #1  cva.fut:130:32-62\n   #2  cva.fut:130:22-69\n   #3  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:71:58-105\n   #1  cva.fut:130:32-62\n   #2  cva.fut:130:22-69\n   #3  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:174:90-97\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  cva.fut:174:25-119\n   #4  cva.fut:173:21-174:131\n   #5  cva.fut:107:1-181:20\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:39:30-45\n   #1  cva.fut:47:27-71\n   #2  cva.fut:174:80-109\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:174:25-119\n   #6  cva.fut:173:21-174:131\n   #7  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:49:38-63\n   #1  cva.fut:174:80-109\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:174:25-119\n   #5  cva.fut:173:21-174:131\n   #6  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:174:90-97\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  cva.fut:174:25-119\n   #4  cva.fut:173:21-174:131\n   #5  cva.fut:107:1-181:20\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:39:30-45\n   #1  cva.fut:47:27-71\n   #2  cva.fut:174:80-109\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:174:25-119\n   #6  cva.fut:173:21-174:131\n   #7  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:49:38-63\n   #1  cva.fut:174:80-109\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:174:25-119\n   #5  cva.fut:173:21-174:131\n   #6  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:174:90-97\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  cva.fut:174:25-119\n   #4  cva.fut:173:21-174:131\n   #5  cva.fut:107:1-181:20\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:39:30-45\n   #1  cva.fut:47:27-71\n   #2  cva.fut:174:80-109\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:174:25-119\n   #6  cva.fut:173:21-174:131\n   #7  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:49:38-63\n   #1  cva.fut:174:80-109\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:174:25-119\n   #5  cva.fut:173:21-174:131\n   #6  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:174:90-97\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  cva.fut:174:25-119\n   #4  cva.fut:173:21-174:131\n   #5  cva.fut:107:1-181:20\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:39:30-45\n   #1  cva.fut:47:27-71\n   #2  cva.fut:174:80-109\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:174:25-119\n   #6  cva.fut:173:21-174:131\n   #7  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:49:38-63\n   #1  cva.fut:174:80-109\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:174:25-119\n   #5  cva.fut:173:21-174:131\n   #6  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:174:90-97\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  cva.fut:174:25-119\n   #4  cva.fut:173:21-174:131\n   #5  cva.fut:107:1-181:20\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:39:30-45\n   #1  cva.fut:47:27-71\n   #2  cva.fut:174:80-109\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cva.fut:174:25-119\n   #6  cva.fut:173:21-174:131\n   #7  cva.fut:107:1-181:20\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:49:38-63\n   #1  cva.fut:174:80-109\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:174:25-119\n   #5  cva.fut:173:21-174:131\n   #6  cva.fut:107:1-181:20\n"]
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
                                       required_types=["i32", "i64", "f32", "bool", "cert"],
                                       user_sizes=sizes,
                                       all_sizes={"main.segmap_group_size_14122": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_14298": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_14631": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_14730": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_14793": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_15072": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_15479": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_14633": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_14098": {"class": "group_size", "value": None},
                                        "main.segred_group_size_14909": {"class": "group_size", "value": None},
                                        "main.segred_group_size_15506": {"class": "group_size", "value": None},
                                        "main.segred_group_size_15778": {"class": "group_size", "value": None},
                                        "main.segred_group_size_16450": {"class": "group_size", "value": None},
                                        "main.segred_group_size_16482": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_14100": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_14911": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_15508": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_15780": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_16448": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_16480": {"class": "num_groups", "value": None},
                                        "main.suff_intra_par_7": {"class": "threshold (!main.suff_outer_par_6 !main.suff_outer_redomap_5)",
                                                                  "value": 32},
                                        "main.suff_outer_par_0": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_6": {"class": "threshold (!main.suff_outer_redomap_5)",
                                                                  "value": None},
                                        "main.suff_outer_redomap_5": {"class": "threshold ()", "value": None}})
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainzisegmap_14120_var = program.mainzisegmap_14120
    self.mainzisegmap_14296_var = program.mainzisegmap_14296
    self.mainzisegmap_14629_var = program.mainzisegmap_14629
    self.mainzisegmap_14727_var = program.mainzisegmap_14727
    self.mainzisegmap_14791_var = program.mainzisegmap_14791
    self.mainzisegmap_15070_var = program.mainzisegmap_15070
    self.mainzisegmap_15477_var = program.mainzisegmap_15477
    self.mainzisegmap_intragroup_15068_var = program.mainzisegmap_intragroup_15068
    self.mainzisegred_large_15512_var = program.mainzisegred_large_15512
    self.mainzisegred_nonseg_14106_var = program.mainzisegred_nonseg_14106
    self.mainzisegred_nonseg_14918_var = program.mainzisegred_nonseg_14918
    self.mainzisegred_nonseg_15786_var = program.mainzisegred_nonseg_15786
    self.mainzisegred_nonseg_16455_var = program.mainzisegred_nonseg_16455
    self.mainzisegred_nonseg_16487_var = program.mainzisegred_nonseg_16487
    self.mainzisegred_small_15512_var = program.mainzisegred_small_15512
    self.constants = {}
    mainzicounter_mem_16517 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_16797 = opencl_alloc(self, 40, "static_mem_16797")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_16797,
                      normaliseArray(mainzicounter_mem_16517),
                      is_blocking=synchronous)
    self.mainzicounter_mem_16517 = static_mem_16797
    mainzicounter_mem_16547 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_16799 = opencl_alloc(self, 40, "static_mem_16799")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_16799,
                      normaliseArray(mainzicounter_mem_16547),
                      is_blocking=synchronous)
    self.mainzicounter_mem_16547 = static_mem_16799
    mainzicounter_mem_16584 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_16801 = opencl_alloc(self, 40, "static_mem_16801")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_16801,
                      normaliseArray(mainzicounter_mem_16584),
                      is_blocking=synchronous)
    self.mainzicounter_mem_16584 = static_mem_16801
    mainzicounter_mem_16647 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_16805 = opencl_alloc(self, 40, "static_mem_16805")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_16805,
                      normaliseArray(mainzicounter_mem_16647),
                      is_blocking=synchronous)
    self.mainzicounter_mem_16647 = static_mem_16805
    mainzicounter_mem_16726 = np.zeros(10240, dtype=np.int32)
    static_mem_16807 = opencl_alloc(self, 40960, "static_mem_16807")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_16807,
                      normaliseArray(mainzicounter_mem_16726),
                      is_blocking=synchronous)
    self.mainzicounter_mem_16726 = static_mem_16807
    mainzicounter_mem_16766 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_16808 = opencl_alloc(self, 40, "static_mem_16808")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_16808,
                      normaliseArray(mainzicounter_mem_16766),
                      is_blocking=synchronous)
    self.mainzicounter_mem_16766 = static_mem_16808
  def futhark_builtinzhgpu_map_transpose_f32(self, destmem_0, destoffset_1,
                                             srcmem_2, srcoffset_3,
                                             num_arrays_4, x_elems_5,
                                             y_elems_6):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_8 = squot32(np.int32(16), x_elems_5)
      mulx_7 = squot32(np.int32(16), y_elems_6)
      if ((num_arrays_4 == np.int32(1)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1)))):
        if (sext_i32_i64(((x_elems_5 * y_elems_6) * np.int32(4))) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(sext_i32_i64(destoffset_1)),
                          src_offset=np.long(sext_i32_i64(srcoffset_3)),
                          byte_count=np.long(sext_i32_i64(((x_elems_5 * y_elems_6) * np.int32(4)))))
        if synchronous:
          sync(self)
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                                                                muly_8),
                                                                                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.gpu_map_transpose_f32_low_width_var.set_args(cl.LocalMemory(np.long(np.int64(1088))),
                                                              np.int32(destoffset_1),
                                                              np.int32(srcoffset_3),
                                                              np.int32(num_arrays_4),
                                                              np.int32(x_elems_5),
                                                              np.int32(y_elems_6),
                                                              np.int32(mulx_7),
                                                              np.int32(muly_8),
                                                              destmem_0,
                                                              srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.gpu_map_transpose_f32_low_width_var,
                                       ((np.long(sdiv_up32(x_elems_5,
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                     muly_8),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              sync(self)
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(sdiv_up32(sdiv_up32(x_elems_5, mulx_7),
                                          np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.gpu_map_transpose_f32_low_height_var.set_args(cl.LocalMemory(np.long(np.int64(1088))),
                                                                 np.int32(destoffset_1),
                                                                 np.int32(srcoffset_3),
                                                                 np.int32(num_arrays_4),
                                                                 np.int32(x_elems_5),
                                                                 np.int32(y_elems_6),
                                                                 np.int32(mulx_7),
                                                                 np.int32(muly_8),
                                                                 destmem_0,
                                                                 srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.gpu_map_transpose_f32_low_height_var,
                                         ((np.long(sdiv_up32(sdiv_up32(x_elems_5,
                                                                       mulx_7),
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(sdiv_up32(y_elems_6,
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
              if ((1 * (np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                          np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.gpu_map_transpose_f32_small_var.set_args(cl.LocalMemory(np.long(np.int64(1))),
                                                              np.int32(destoffset_1),
                                                              np.int32(srcoffset_3),
                                                              np.int32(num_arrays_4),
                                                              np.int32(x_elems_5),
                                                              np.int32(y_elems_6),
                                                              np.int32(mulx_7),
                                                              np.int32(muly_8),
                                                              destmem_0,
                                                              srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.gpu_map_transpose_f32_small_var,
                                           ((np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                               np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  sync(self)
            else:
              if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                            np.int32(32))) * np.long(np.int32(32)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                          np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.gpu_map_transpose_f32_var.set_args(cl.LocalMemory(np.long(np.int64(4224))),
                                                        np.int32(destoffset_1),
                                                        np.int32(srcoffset_3),
                                                        np.int32(num_arrays_4),
                                                        np.int32(x_elems_5),
                                                        np.int32(y_elems_6),
                                                        np.int32(mulx_7),
                                                        np.int32(muly_8),
                                                        destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.gpu_map_transpose_f32_var,
                                           ((np.long(sdiv_up32(x_elems_5,
                                                               np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(sdiv_up32(y_elems_6,
                                                               np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  sync(self)
    return ()
  def futhark_main(self, swap_term_mem_16319, payments_mem_16320,
                   notional_mem_16321, n_13650, n_13651, n_13652, paths_13653,
                   steps_13654, a_13658, b_13659, sigma_13660, r0_13661):
    dim_match_13662 = (n_13650 == n_13651)
    empty_or_match_cert_13663 = True
    assert dim_match_13662, ("Error: %s\n\nBacktrace:\n-> #0  cva.fut:107:1-181:20\n" % ("function arguments of wrong shape",))
    segred_group_sizze_14099 = self.sizes["main.segred_group_size_14098"]
    max_num_groups_16516 = self.sizes["main.segred_num_groups_14100"]
    num_groups_14101 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(n_13650,
                                                            sext_i32_i64(segred_group_sizze_14099)),
                                                  sext_i32_i64(max_num_groups_16516))))
    mem_16324 = opencl_alloc(self, np.int64(4), "mem_16324")
    mainzicounter_mem_16517 = self.mainzicounter_mem_16517
    group_res_arr_mem_16519 = opencl_alloc(self,
                                           (np.int32(4) * (segred_group_sizze_14099 * num_groups_14101)),
                                           "group_res_arr_mem_16519")
    num_threads_16521 = (num_groups_14101 * segred_group_sizze_14099)
    if ((1 * (np.long(num_groups_14101) * np.long(segred_group_sizze_14099))) != 0):
      self.mainzisegred_nonseg_14106_var.set_args(self.global_failure,
                                                  cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14099))),
                                                  cl.LocalMemory(np.long(np.int32(1))),
                                                  np.int64(n_13650),
                                                  np.int64(num_groups_14101),
                                                  swap_term_mem_16319,
                                                  payments_mem_16320, mem_16324,
                                                  mainzicounter_mem_16517,
                                                  group_res_arr_mem_16519,
                                                  np.int64(num_threads_16521))
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegred_nonseg_14106_var,
                                 ((np.long(num_groups_14101) * np.long(segred_group_sizze_14099)),),
                                 (np.long(segred_group_sizze_14099),))
      if synchronous:
        sync(self)
    read_res_16798 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_16798, mem_16324,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_13667 = read_res_16798[0]
    mem_16324 = None
    res_13675 = sitofp_i64_f32(steps_13654)
    dt_13676 = (res_13667 / res_13675)
    x_13678 = fpow32(a_13658, np.float32(2.0))
    x_13679 = (b_13659 * x_13678)
    x_13680 = fpow32(sigma_13660, np.float32(2.0))
    y_13681 = (x_13680 / np.float32(2.0))
    y_13682 = (x_13679 - y_13681)
    y_13683 = (np.float32(4.0) * a_13658)
    suff_outer_par_14108 = (self.sizes["main.suff_outer_par_0"] <= n_13650)
    segmap_group_sizze_14198 = self.sizes["main.segmap_group_size_14122"]
    segmap_group_sizze_14374 = self.sizes["main.segmap_group_size_14298"]
    bytes_16325 = (np.int64(4) * n_13650)
    bytes_16327 = (np.int64(8) * n_13650)
    segred_num_groups_16449 = self.sizes["main.segred_num_groups_16448"]
    segred_group_sizze_16451 = self.sizes["main.segred_group_size_16450"]
    segred_num_groups_16481 = self.sizes["main.segred_num_groups_16480"]
    segred_group_sizze_16483 = self.sizes["main.segred_group_size_16482"]
    local_memory_capacity_16621 = self.max_local_memory
    if ((sle64((np.int32(1) + (np.int32(8) * segred_group_sizze_16451)),
               sext_i32_i64(local_memory_capacity_16621)) and sle64(np.int64(0),
                                                                    sext_i32_i64(local_memory_capacity_16621))) and suff_outer_par_14108):
      segmap_usable_groups_14199 = sdiv_up64(n_13650, segmap_group_sizze_14198)
      mem_16326 = opencl_alloc(self, bytes_16325, "mem_16326")
      if ((n_13650 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_16326, swap_term_mem_16319,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_13650 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_16328 = opencl_alloc(self, bytes_16327, "mem_16328")
      if ((n_13650 * np.int32(8)) != 0):
        cl.enqueue_copy(self.queue, mem_16328, payments_mem_16320,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_13650 * np.int32(8))))
      if synchronous:
        sync(self)
      mem_16330 = opencl_alloc(self, bytes_16325, "mem_16330")
      if ((n_13650 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_16330, notional_mem_16321,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_13650 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_16347 = opencl_alloc(self, bytes_16325, "mem_16347")
      num_threads_16439 = (segmap_group_sizze_14198 * segmap_usable_groups_14199)
      mem_16467 = opencl_alloc(self, np.int64(8), "mem_16467")
      mainzicounter_mem_16547 = self.mainzicounter_mem_16547
      group_res_arr_mem_16549 = opencl_alloc(self,
                                             (np.int32(8) * (segred_group_sizze_16451 * segred_num_groups_16449)),
                                             "group_res_arr_mem_16549")
      num_threads_16551 = (segred_num_groups_16449 * segred_group_sizze_16451)
      if ((1 * (np.long(segred_num_groups_16449) * np.long(segred_group_sizze_16451))) != 0):
        self.mainzisegred_nonseg_16455_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(8) * segred_group_sizze_16451))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(n_13650),
                                                    payments_mem_16320,
                                                    mem_16467,
                                                    mainzicounter_mem_16547,
                                                    group_res_arr_mem_16549,
                                                    np.int64(num_threads_16551))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_16455_var,
                                   ((np.long(segred_num_groups_16449) * np.long(segred_group_sizze_16451)),),
                                   (np.long(segred_group_sizze_16451),))
        if synchronous:
          sync(self)
      read_res_16800 = np.empty(1, dtype=ct.c_int64)
      cl.enqueue_copy(self.queue, read_res_16800, mem_16467,
                      device_offset=(np.long(np.int64(0)) * 8),
                      is_blocking=synchronous)
      sync(self)
      max_per_thread_16445 = read_res_16800[0]
      mem_16467 = None
      sizze_sum_16462 = (num_threads_16439 * max_per_thread_16445)
      mem_16333 = opencl_alloc(self, sizze_sum_16462, "mem_16333")
      if ((1 * (np.long(segmap_usable_groups_14199) * np.long(segmap_group_sizze_14198))) != 0):
        self.mainzisegmap_14120_var.set_args(self.global_failure,
                                             self.failure_is_an_option,
                                             self.global_failure_args,
                                             np.int64(n_13650),
                                             np.float32(a_13658),
                                             np.float32(r0_13661),
                                             np.float32(x_13678),
                                             np.float32(x_13680),
                                             np.float32(y_13682),
                                             np.float32(y_13683),
                                             swap_term_mem_16319,
                                             payments_mem_16320, mem_16333,
                                             mem_16347,
                                             np.int64(num_threads_16439))
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_14120_var,
                                   ((np.long(segmap_usable_groups_14199) * np.long(segmap_group_sizze_14198)),),
                                   (np.long(segmap_group_sizze_14198),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      mem_16333 = None
      res_mem_16371 = mem_16347
      res_mem_16372 = mem_16330
      res_mem_16373 = mem_16328
      res_mem_16374 = mem_16326
    else:
      segmap_usable_groups_14375 = sdiv_up64(n_13650, segmap_group_sizze_14374)
      mem_16349 = opencl_alloc(self, bytes_16325, "mem_16349")
      if ((n_13650 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_16349, swap_term_mem_16319,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_13650 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_16351 = opencl_alloc(self, bytes_16327, "mem_16351")
      if ((n_13650 * np.int32(8)) != 0):
        cl.enqueue_copy(self.queue, mem_16351, payments_mem_16320,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_13650 * np.int32(8))))
      if synchronous:
        sync(self)
      mem_16353 = opencl_alloc(self, bytes_16325, "mem_16353")
      if ((n_13650 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_16353, notional_mem_16321,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_13650 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_16370 = opencl_alloc(self, bytes_16325, "mem_16370")
      num_threads_16471 = (segmap_group_sizze_14374 * segmap_usable_groups_14375)
      mem_16499 = opencl_alloc(self, np.int64(8), "mem_16499")
      mainzicounter_mem_16584 = self.mainzicounter_mem_16584
      group_res_arr_mem_16586 = opencl_alloc(self,
                                             (np.int32(8) * (segred_group_sizze_16483 * segred_num_groups_16481)),
                                             "group_res_arr_mem_16586")
      num_threads_16588 = (segred_num_groups_16481 * segred_group_sizze_16483)
      if ((1 * (np.long(segred_num_groups_16481) * np.long(segred_group_sizze_16483))) != 0):
        self.mainzisegred_nonseg_16487_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(8) * segred_group_sizze_16483))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(n_13650),
                                                    payments_mem_16320,
                                                    mem_16499,
                                                    mainzicounter_mem_16584,
                                                    group_res_arr_mem_16586,
                                                    np.int64(num_threads_16588))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_16487_var,
                                   ((np.long(segred_num_groups_16481) * np.long(segred_group_sizze_16483)),),
                                   (np.long(segred_group_sizze_16483),))
        if synchronous:
          sync(self)
      read_res_16802 = np.empty(1, dtype=ct.c_int64)
      cl.enqueue_copy(self.queue, read_res_16802, mem_16499,
                      device_offset=(np.long(np.int64(0)) * 8),
                      is_blocking=synchronous)
      sync(self)
      max_per_thread_16477 = read_res_16802[0]
      mem_16499 = None
      sizze_sum_16494 = (num_threads_16471 * max_per_thread_16477)
      mem_16356 = opencl_alloc(self, sizze_sum_16494, "mem_16356")
      if ((1 * (np.long(segmap_usable_groups_14375) * np.long(segmap_group_sizze_14374))) != 0):
        self.mainzisegmap_14296_var.set_args(self.global_failure,
                                             self.failure_is_an_option,
                                             self.global_failure_args,
                                             np.int64(n_13650),
                                             np.float32(a_13658),
                                             np.float32(r0_13661),
                                             np.float32(x_13678),
                                             np.float32(x_13680),
                                             np.float32(y_13682),
                                             np.float32(y_13683),
                                             swap_term_mem_16319,
                                             payments_mem_16320, mem_16356,
                                             mem_16370,
                                             np.int64(num_threads_16471))
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_14296_var,
                                   ((np.long(segmap_usable_groups_14375) * np.long(segmap_group_sizze_14374)),),
                                   (np.long(segmap_group_sizze_14374),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      mem_16356 = None
      res_mem_16371 = mem_16370
      res_mem_16372 = mem_16353
      res_mem_16373 = mem_16351
      res_mem_16374 = mem_16349
    sims_per_year_13760 = (res_13675 / res_13667)
    bounds_invalid_upwards_13761 = slt64(steps_13654, np.int64(1))
    valid_13762 = not(bounds_invalid_upwards_13761)
    range_valid_c_13763 = True
    assert valid_13762, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:60:56-67\n   #1  cva.fut:118:17-44\n   #2  cva.fut:107:1-181:20\n" % ("Range ",
                                                                                                                                                    np.int64(1),
                                                                                                                                                    "..",
                                                                                                                                                    np.int64(2),
                                                                                                                                                    "...",
                                                                                                                                                    steps_13654,
                                                                                                                                                    " is invalid."))
    y_13769 = slt64(np.int64(0), n_13650)
    index_certs_13770 = True
    assert y_13769, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:119:21-32\n   #1  cva.fut:107:1-181:20\n" % ("Index [",
                                                                                                                   np.int64(0),
                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                   n_13650,
                                                                                                                   "]."))
    read_res_16803 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_16803, swap_term_mem_16319,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    x_13771 = read_res_16803[0]
    read_res_16804 = np.empty(1, dtype=ct.c_int64)
    cl.enqueue_copy(self.queue, read_res_16804, payments_mem_16320,
                    device_offset=(np.long(np.int64(0)) * 8),
                    is_blocking=synchronous)
    sync(self)
    x_13772 = read_res_16804[0]
    i64_arg_13773 = (x_13772 - np.int64(1))
    res_13774 = sitofp_i64_f32(i64_arg_13773)
    last_date_13775 = (x_13771 * res_13774)
    bounds_invalid_upwards_13776 = slt64(paths_13653, np.int64(0))
    valid_13777 = not(bounds_invalid_upwards_13776)
    range_valid_c_13778 = True
    assert valid_13777, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:122:19-49\n   #4  cva.fut:107:1-181:20\n" % ("Range ",
                                                                                                                                                                                                                                                                np.int64(0),
                                                                                                                                                                                                                                                                "..",
                                                                                                                                                                                                                                                                np.int64(1),
                                                                                                                                                                                                                                                                "..<",
                                                                                                                                                                                                                                                                paths_13653,
                                                                                                                                                                                                                                                                " is invalid."))
    upper_bound_13781 = (steps_13654 - np.int64(1))
    res_13782 = futhark_sqrt32(dt_13676)
    segmap_group_sizze_14810 = self.sizes["main.segmap_group_size_14793"]
    segmap_usable_groups_14811 = sdiv_up64(paths_13653,
                                           segmap_group_sizze_14810)
    bytes_16376 = (np.int64(4) * paths_13653)
    mem_16377 = opencl_alloc(self, bytes_16376, "mem_16377")
    if ((1 * (np.long(segmap_usable_groups_14811) * np.long(segmap_group_sizze_14810))) != 0):
      self.mainzisegmap_14791_var.set_args(self.global_failure,
                                           np.int64(paths_13653), mem_16377)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_14791_var,
                                 ((np.long(segmap_usable_groups_14811) * np.long(segmap_group_sizze_14810)),),
                                 (np.long(segmap_group_sizze_14810),))
      if synchronous:
        sync(self)
    nest_sizze_14834 = (paths_13653 * steps_13654)
    segmap_group_sizze_14835 = self.sizes["main.segmap_group_size_14730"]
    segmap_usable_groups_14836 = sdiv_up64(nest_sizze_14834,
                                           segmap_group_sizze_14835)
    bytes_16379 = (np.int64(4) * nest_sizze_14834)
    mem_16381 = opencl_alloc(self, bytes_16379, "mem_16381")
    if ((1 * (np.long(segmap_usable_groups_14836) * np.long(segmap_group_sizze_14835))) != 0):
      self.mainzisegmap_14727_var.set_args(self.global_failure,
                                           np.int64(paths_13653),
                                           np.int64(steps_13654), mem_16377,
                                           mem_16381)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_14727_var,
                                 ((np.long(segmap_usable_groups_14836) * np.long(segmap_group_sizze_14835)),),
                                 (np.long(segmap_group_sizze_14835),))
      if synchronous:
        sync(self)
    mem_16377 = None
    segmap_group_sizze_14880 = self.sizes["main.segmap_group_size_14631"]
    max_num_groups_16632 = self.sizes["main.segmap_num_groups_14633"]
    num_groups_14881 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_13653,
                                                            sext_i32_i64(segmap_group_sizze_14880)),
                                                  sext_i32_i64(max_num_groups_16632))))
    mem_16384 = opencl_alloc(self, bytes_16379, "mem_16384")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_16384, np.int64(0),
                                                mem_16381, np.int64(0),
                                                np.int64(1), steps_13654,
                                                paths_13653)
    mem_16381 = None
    mem_16402 = opencl_alloc(self, bytes_16379, "mem_16402")
    bytes_16386 = (np.int64(4) * steps_13654)
    num_threads_16505 = (segmap_group_sizze_14880 * num_groups_14881)
    total_sizze_16506 = (bytes_16386 * num_threads_16505)
    mem_16387 = opencl_alloc(self, total_sizze_16506, "mem_16387")
    if ((1 * (np.long(num_groups_14881) * np.long(segmap_group_sizze_14880))) != 0):
      self.mainzisegmap_14629_var.set_args(self.global_failure,
                                           self.failure_is_an_option,
                                           self.global_failure_args,
                                           np.int64(paths_13653),
                                           np.int64(steps_13654),
                                           np.float32(a_13658),
                                           np.float32(b_13659),
                                           np.float32(sigma_13660),
                                           np.float32(r0_13661),
                                           np.float32(dt_13676),
                                           np.int64(upper_bound_13781),
                                           np.float32(res_13782),
                                           np.int64(num_groups_14881),
                                           mem_16384, mem_16387, mem_16402)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_14629_var,
                                 ((np.long(num_groups_14881) * np.long(segmap_group_sizze_14880)),),
                                 (np.long(segmap_group_sizze_14880),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_16384 = None
    mem_16387 = None
    res_13844 = sitofp_i64_f32(paths_13653)
    suff_outer_redomap_14907 = (self.sizes["main.suff_outer_redomap_5"] <= steps_13654)
    segred_group_sizze_14920 = self.sizes["main.segred_group_size_14909"]
    max_num_groups_16644 = self.sizes["main.segred_num_groups_14911"]
    num_groups_14921 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(steps_13654,
                                                            sext_i32_i64(segred_group_sizze_14920)),
                                                  sext_i32_i64(max_num_groups_16644))))
    suff_outer_par_15788 = (self.sizes["main.suff_outer_par_6"] <= steps_13654)
    max_group_sizze_15791 = self.max_group_size
    fits_15792 = sle64(paths_13653, max_group_sizze_15791)
    suff_intra_par_15794 = (self.sizes["main.suff_intra_par_7"] <= paths_13653)
    intra_suff_and_fits_15795 = (fits_15792 and suff_intra_par_15794)
    segmap_group_sizze_15799 = self.sizes["main.segmap_group_size_15072"]
    segred_group_sizze_16074 = self.sizes["main.segred_group_size_15506"]
    max_num_groups_16645 = self.sizes["main.segred_num_groups_15508"]
    num_groups_16075 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_14834,
                                                            sext_i32_i64(segred_group_sizze_16074)),
                                                  sext_i32_i64(max_num_groups_16645))))
    segmap_group_sizze_16185 = self.sizes["main.segmap_group_size_15479"]
    segred_group_sizze_16213 = self.sizes["main.segred_group_size_15778"]
    max_num_groups_16646 = self.sizes["main.segred_num_groups_15780"]
    num_groups_16214 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(steps_13654,
                                                            sext_i32_i64(segred_group_sizze_16213)),
                                                  sext_i32_i64(max_num_groups_16646))))
    local_memory_capacity_16796 = self.max_local_memory
    if (sle64((np.int32(1) + (np.int32(4) * segred_group_sizze_14920)),
              sext_i32_i64(local_memory_capacity_16796)) and suff_outer_redomap_14907):
      mem_16405 = opencl_alloc(self, np.int64(4), "mem_16405")
      mem_16407 = opencl_alloc(self, bytes_16386, "mem_16407")
      mainzicounter_mem_16647 = self.mainzicounter_mem_16647
      group_res_arr_mem_16649 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_14920 * num_groups_14921)),
                                             "group_res_arr_mem_16649")
      num_threads_16651 = (num_groups_14921 * segred_group_sizze_14920)
      if ((1 * (np.long(num_groups_14921) * np.long(segred_group_sizze_14920))) != 0):
        self.mainzisegred_nonseg_14918_var.set_args(self.global_failure,
                                                    self.failure_is_an_option,
                                                    self.global_failure_args,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_14920))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(n_13650),
                                                    np.int64(paths_13653),
                                                    np.int64(steps_13654),
                                                    np.float32(a_13658),
                                                    np.float32(b_13659),
                                                    np.float32(sigma_13660),
                                                    np.float32(x_13678),
                                                    np.float32(x_13680),
                                                    np.float32(y_13682),
                                                    np.float32(y_13683),
                                                    np.float32(sims_per_year_13760),
                                                    np.float32(last_date_13775),
                                                    np.float32(res_13844),
                                                    np.int64(num_groups_14921),
                                                    res_mem_16371,
                                                    res_mem_16372,
                                                    res_mem_16373,
                                                    res_mem_16374, mem_16402,
                                                    mem_16405, mem_16407,
                                                    mainzicounter_mem_16647,
                                                    group_res_arr_mem_16649,
                                                    np.int64(num_threads_16651))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_14918_var,
                                   ((np.long(num_groups_14921) * np.long(segred_group_sizze_14920)),),
                                   (np.long(segred_group_sizze_14920),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      read_res_16806 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_16806, mem_16405,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_15057 = read_res_16806[0]
      mem_16405 = None
      mem_16435 = opencl_alloc(self, bytes_16386, "mem_16435")
      if ((steps_13654 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_16435, mem_16407,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((steps_13654 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_16407 = None
      res_mem_16437 = mem_16435
      res_13960 = res_15057
    else:
      local_memory_capacity_16765 = self.max_local_memory
      if (sle64(np.int64(0),
                sext_i32_i64(local_memory_capacity_16765)) and suff_outer_par_15788):
        segmap_usable_groups_15800 = sdiv_up64(steps_13654,
                                               segmap_group_sizze_15799)
        mem_16410 = opencl_alloc(self, bytes_16386, "mem_16410")
        mem_16412 = opencl_alloc(self, bytes_16386, "mem_16412")
        if ((1 * (np.long(segmap_usable_groups_15800) * np.long(segmap_group_sizze_15799))) != 0):
          self.mainzisegmap_15070_var.set_args(self.global_failure,
                                               self.failure_is_an_option,
                                               self.global_failure_args,
                                               np.int64(n_13650),
                                               np.int64(paths_13653),
                                               np.int64(steps_13654),
                                               np.float32(a_13658),
                                               np.float32(b_13659),
                                               np.float32(sigma_13660),
                                               np.float32(x_13678),
                                               np.float32(x_13680),
                                               np.float32(y_13682),
                                               np.float32(y_13683),
                                               np.float32(sims_per_year_13760),
                                               np.float32(last_date_13775),
                                               np.float32(res_13844),
                                               res_mem_16371, res_mem_16372,
                                               res_mem_16373, res_mem_16374,
                                               mem_16402, mem_16410, mem_16412)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_15070_var,
                                     ((np.long(segmap_usable_groups_15800) * np.long(segmap_group_sizze_15799)),),
                                     (np.long(segmap_group_sizze_15799),))
          if synchronous:
            sync(self)
        self.failure_is_an_option = np.int32(1)
        res_map_acc_mem_16429 = mem_16410
        res_mem_16430 = mem_16412
      else:
        local_memory_capacity_16764 = self.max_local_memory
        if (sle64((np.int32(4) * paths_13653),
                  sext_i32_i64(local_memory_capacity_16764)) and intra_suff_and_fits_15795):
          mem_16416 = opencl_alloc(self, bytes_16386, "mem_16416")
          mem_16418 = opencl_alloc(self, bytes_16386, "mem_16418")
          if ((1 * (np.long(steps_13654) * np.long(paths_13653))) != 0):
            self.mainzisegmap_intragroup_15068_var.set_args(self.global_failure,
                                                            self.failure_is_an_option,
                                                            self.global_failure_args,
                                                            cl.LocalMemory(np.long((np.int32(4) * paths_13653))),
                                                            np.int64(n_13650),
                                                            np.int64(paths_13653),
                                                            np.float32(a_13658),
                                                            np.float32(b_13659),
                                                            np.float32(sigma_13660),
                                                            np.float32(x_13678),
                                                            np.float32(x_13680),
                                                            np.float32(y_13682),
                                                            np.float32(y_13683),
                                                            np.float32(sims_per_year_13760),
                                                            np.float32(last_date_13775),
                                                            np.float32(res_13844),
                                                            res_mem_16371,
                                                            res_mem_16372,
                                                            res_mem_16373,
                                                            res_mem_16374,
                                                            mem_16402,
                                                            mem_16416,
                                                            mem_16418)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegmap_intragroup_15068_var,
                                       ((np.long(steps_13654) * np.long(paths_13653)),),
                                       (np.long(paths_13653),))
            if synchronous:
              sync(self)
          self.failure_is_an_option = np.int32(1)
          res_map_acc_mem_16427 = mem_16416
          res_mem_16428 = mem_16418
        else:
          mem_16421 = opencl_alloc(self, bytes_16386, "mem_16421")
          if slt64((paths_13653 * np.int64(2)), segred_group_sizze_16074):
            segment_sizze_nonzzero_16697 = smax64(np.int64(1), paths_13653)
            num_threads_16698 = (num_groups_16075 * segred_group_sizze_16074)
            if ((1 * (np.long(num_groups_16075) * np.long(segred_group_sizze_16074))) != 0):
              self.mainzisegred_small_15512_var.set_args(self.global_failure,
                                                         self.failure_is_an_option,
                                                         self.global_failure_args,
                                                         cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_16074))),
                                                         np.int64(n_13650),
                                                         np.int64(paths_13653),
                                                         np.int64(steps_13654),
                                                         np.float32(a_13658),
                                                         np.float32(b_13659),
                                                         np.float32(sigma_13660),
                                                         np.float32(sims_per_year_13760),
                                                         np.float32(last_date_13775),
                                                         np.int64(num_groups_16075),
                                                         res_mem_16371,
                                                         res_mem_16372,
                                                         res_mem_16373,
                                                         res_mem_16374,
                                                         mem_16402, mem_16421,
                                                         np.int64(segment_sizze_nonzzero_16697))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_small_15512_var,
                                         ((np.long(num_groups_16075) * np.long(segred_group_sizze_16074)),),
                                         (np.long(segred_group_sizze_16074),))
              if synchronous:
                sync(self)
            self.failure_is_an_option = np.int32(1)
          else:
            groups_per_segment_16719 = sdiv_up64(num_groups_16075,
                                                 smax64(np.int64(1),
                                                        steps_13654))
            elements_per_thread_16720 = sdiv_up64(paths_13653,
                                                  (segred_group_sizze_16074 * groups_per_segment_16719))
            virt_num_groups_16721 = (groups_per_segment_16719 * steps_13654)
            num_threads_16722 = (num_groups_16075 * segred_group_sizze_16074)
            threads_per_segment_16723 = (groups_per_segment_16719 * segred_group_sizze_16074)
            group_res_arr_mem_16724 = opencl_alloc(self,
                                                   (np.int32(4) * (segred_group_sizze_16074 * virt_num_groups_16721)),
                                                   "group_res_arr_mem_16724")
            mainzicounter_mem_16726 = self.mainzicounter_mem_16726
            if ((1 * (np.long(num_groups_16075) * np.long(segred_group_sizze_16074))) != 0):
              self.mainzisegred_large_15512_var.set_args(self.global_failure,
                                                         self.failure_is_an_option,
                                                         self.global_failure_args,
                                                         cl.LocalMemory(np.long(np.int32(1))),
                                                         cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_16074))),
                                                         np.int64(n_13650),
                                                         np.int64(paths_13653),
                                                         np.float32(a_13658),
                                                         np.float32(b_13659),
                                                         np.float32(sigma_13660),
                                                         np.float32(sims_per_year_13760),
                                                         np.float32(last_date_13775),
                                                         np.int64(num_groups_16075),
                                                         res_mem_16371,
                                                         res_mem_16372,
                                                         res_mem_16373,
                                                         res_mem_16374,
                                                         mem_16402, mem_16421,
                                                         np.int64(groups_per_segment_16719),
                                                         np.int64(elements_per_thread_16720),
                                                         np.int64(virt_num_groups_16721),
                                                         np.int64(threads_per_segment_16723),
                                                         group_res_arr_mem_16724,
                                                         mainzicounter_mem_16726)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_large_15512_var,
                                         ((np.long(num_groups_16075) * np.long(segred_group_sizze_16074)),),
                                         (np.long(segred_group_sizze_16074),))
              if synchronous:
                sync(self)
            self.failure_is_an_option = np.int32(1)
          segmap_usable_groups_16186 = sdiv_up64(steps_13654,
                                                 segmap_group_sizze_16185)
          mem_16424 = opencl_alloc(self, bytes_16386, "mem_16424")
          mem_16426 = opencl_alloc(self, bytes_16386, "mem_16426")
          if ((1 * (np.long(segmap_usable_groups_16186) * np.long(segmap_group_sizze_16185))) != 0):
            self.mainzisegmap_15477_var.set_args(self.global_failure,
                                                 np.int64(steps_13654),
                                                 np.float32(a_13658),
                                                 np.float32(x_13678),
                                                 np.float32(x_13680),
                                                 np.float32(y_13682),
                                                 np.float32(y_13683),
                                                 np.float32(sims_per_year_13760),
                                                 np.float32(res_13844),
                                                 mem_16421, mem_16424,
                                                 mem_16426)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_15477_var,
                                       ((np.long(segmap_usable_groups_16186) * np.long(segmap_group_sizze_16185)),),
                                       (np.long(segmap_group_sizze_16185),))
            if synchronous:
              sync(self)
          mem_16421 = None
          res_map_acc_mem_16427 = mem_16424
          res_mem_16428 = mem_16426
        res_map_acc_mem_16429 = res_map_acc_mem_16427
        res_mem_16430 = res_mem_16428
      mem_16433 = opencl_alloc(self, np.int64(4), "mem_16433")
      mainzicounter_mem_16766 = self.mainzicounter_mem_16766
      group_res_arr_mem_16768 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_16213 * num_groups_16214)),
                                             "group_res_arr_mem_16768")
      num_threads_16770 = (num_groups_16214 * segred_group_sizze_16213)
      if ((1 * (np.long(num_groups_16214) * np.long(segred_group_sizze_16213))) != 0):
        self.mainzisegred_nonseg_15786_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_16213))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(steps_13654),
                                                    np.int64(num_groups_16214),
                                                    res_map_acc_mem_16429,
                                                    mem_16433,
                                                    mainzicounter_mem_16766,
                                                    group_res_arr_mem_16768,
                                                    np.int64(num_threads_16770))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_15786_var,
                                   ((np.long(num_groups_16214) * np.long(segred_group_sizze_16213)),),
                                   (np.long(segred_group_sizze_16213),))
        if synchronous:
          sync(self)
      res_map_acc_mem_16429 = None
      read_res_16809 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_16809, mem_16433,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_16221 = read_res_16809[0]
      mem_16433 = None
      res_mem_16437 = res_mem_16430
      res_13960 = res_16221
    res_mem_16371 = None
    res_mem_16372 = None
    res_mem_16373 = None
    res_mem_16374 = None
    mem_16402 = None
    CVA_14096 = (np.float32(6.000000052154064e-3) * res_13960)
    out_arrsizze_16515 = steps_13654
    out_mem_16514 = res_mem_16437
    scalar_out_16513 = CVA_14096
    return (scalar_out_16513, out_mem_16514, out_arrsizze_16515)
  def main(self, paths_13653_ext, steps_13654_ext, swap_term_mem_16319_ext,
           payments_mem_16320_ext, notional_mem_16321_ext, a_13658_ext,
           b_13659_ext, sigma_13660_ext, r0_13661_ext):
    try:
      paths_13653 = np.int64(ct.c_int64(paths_13653_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(paths_13653_ext),
                                                                                                                            paths_13653_ext))
    try:
      steps_13654 = np.int64(ct.c_int64(steps_13654_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(steps_13654_ext),
                                                                                                                            steps_13654_ext))
    try:
      assert ((type(swap_term_mem_16319_ext) in [np.ndarray,
                                                 cl.array.Array]) and (swap_term_mem_16319_ext.dtype == np.float32)), "Parameter has unexpected type"
      n_13650 = np.int32(swap_term_mem_16319_ext.shape[0])
      if (type(swap_term_mem_16319_ext) == cl.array.Array):
        swap_term_mem_16319 = swap_term_mem_16319_ext.data
      else:
        swap_term_mem_16319 = opencl_alloc(self,
                                           np.int64(swap_term_mem_16319_ext.nbytes),
                                           "swap_term_mem_16319")
        if (np.int64(swap_term_mem_16319_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, swap_term_mem_16319,
                          normaliseArray(swap_term_mem_16319_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(swap_term_mem_16319_ext),
                                                                                                                            swap_term_mem_16319_ext))
    try:
      assert ((type(payments_mem_16320_ext) in [np.ndarray,
                                                cl.array.Array]) and (payments_mem_16320_ext.dtype == np.int64)), "Parameter has unexpected type"
      n_13651 = np.int32(payments_mem_16320_ext.shape[0])
      if (type(payments_mem_16320_ext) == cl.array.Array):
        payments_mem_16320 = payments_mem_16320_ext.data
      else:
        payments_mem_16320 = opencl_alloc(self,
                                          np.int64(payments_mem_16320_ext.nbytes),
                                          "payments_mem_16320")
        if (np.int64(payments_mem_16320_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, payments_mem_16320,
                          normaliseArray(payments_mem_16320_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i64",
                                                                                                                            type(payments_mem_16320_ext),
                                                                                                                            payments_mem_16320_ext))
    try:
      assert ((type(notional_mem_16321_ext) in [np.ndarray,
                                                cl.array.Array]) and (notional_mem_16321_ext.dtype == np.float32)), "Parameter has unexpected type"
      n_13652 = np.int32(notional_mem_16321_ext.shape[0])
      if (type(notional_mem_16321_ext) == cl.array.Array):
        notional_mem_16321 = notional_mem_16321_ext.data
      else:
        notional_mem_16321 = opencl_alloc(self,
                                          np.int64(notional_mem_16321_ext.nbytes),
                                          "notional_mem_16321")
        if (np.int64(notional_mem_16321_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, notional_mem_16321,
                          normaliseArray(notional_mem_16321_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(notional_mem_16321_ext),
                                                                                                                            notional_mem_16321_ext))
    try:
      a_13658 = np.float32(ct.c_float(a_13658_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_13658_ext),
                                                                                                                            a_13658_ext))
    try:
      b_13659 = np.float32(ct.c_float(b_13659_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_13659_ext),
                                                                                                                            b_13659_ext))
    try:
      sigma_13660 = np.float32(ct.c_float(sigma_13660_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_13660_ext),
                                                                                                                            sigma_13660_ext))
    try:
      r0_13661 = np.float32(ct.c_float(r0_13661_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_13661_ext),
                                                                                                                            r0_13661_ext))
    (scalar_out_16513, out_mem_16514,
     out_arrsizze_16515) = self.futhark_main(swap_term_mem_16319,
                                             payments_mem_16320,
                                             notional_mem_16321, n_13650,
                                             n_13651, n_13652, paths_13653,
                                             steps_13654, a_13658, b_13659,
                                             sigma_13660, r0_13661)
    sync(self)
    return (np.float32(scalar_out_16513), cl.array.Array(self.queue,
                                                         (out_arrsizze_16515,),
                                                         ct.c_float,
                                                         data=out_mem_16514))