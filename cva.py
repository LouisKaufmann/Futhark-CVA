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




__kernel void builtinzhreplicate_f32zireplicate_19089(__global
                                                      unsigned char *mem_19085,
                                                      int32_t num_elems_19086,
                                                      float val_19087)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_19089;
    int32_t replicate_ltid_19090;
    int32_t replicate_gid_19091;
    
    replicate_gtid_19089 = get_global_id(0);
    replicate_ltid_19090 = get_local_id(0);
    replicate_gid_19091 = get_group_id(0);
    if (slt64(replicate_gtid_19089, num_elems_19086)) {
        ((__global float *) mem_19085)[sext_i32_i64(replicate_gtid_19089)] =
            val_19087;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i64zireplicate_18681(__global
                                                      unsigned char *mem_18677,
                                                      int32_t num_elems_18678,
                                                      int64_t val_18679)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_18681;
    int32_t replicate_ltid_18682;
    int32_t replicate_gid_18683;
    
    replicate_gtid_18681 = get_global_id(0);
    replicate_ltid_18682 = get_local_id(0);
    replicate_gid_18683 = get_group_id(0);
    if (slt64(replicate_gtid_18681, num_elems_18678)) {
        ((__global int64_t *) mem_18677)[sext_i32_i64(replicate_gtid_18681)] =
            val_18679;
    }
    
  error_0:
    return;
}
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
__kernel void mainziscan_stage1_18308(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_18605_backing_aligned_0,
                                      int64_t paths_17456, int64_t res_17605,
                                      __global unsigned char *mem_18505,
                                      __global unsigned char *mem_18507,
                                      int32_t num_threads_18599)
{
    #define segscan_group_sizze_18303 (mainzisegscan_group_sizze_18302)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18605_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18605_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18600;
    int32_t local_tid_18601;
    int64_t group_sizze_18604;
    int32_t wave_sizze_18603;
    int32_t group_tid_18602;
    
    global_tid_18600 = get_global_id(0);
    local_tid_18601 = get_local_id(0);
    group_sizze_18604 = get_local_size(0);
    wave_sizze_18603 = LOCKSTEP_WIDTH;
    group_tid_18602 = get_group_id(0);
    
    int32_t phys_tid_18308;
    
    phys_tid_18308 = global_tid_18600;
    
    __local char *scan_arr_mem_18605;
    
    scan_arr_mem_18605 = (__local char *) scan_arr_mem_18605_backing_0;
    
    int64_t x_17593;
    int64_t x_17594;
    
    x_17593 = 0;
    for (int64_t j_18607 = 0; j_18607 < sdiv_up64(paths_17456,
                                                  sext_i32_i64(num_threads_18599));
         j_18607++) {
        int64_t chunk_offset_18608 = segscan_group_sizze_18303 * j_18607 +
                sext_i32_i64(group_tid_18602) * (segscan_group_sizze_18303 *
                                                 sdiv_up64(paths_17456,
                                                           sext_i32_i64(num_threads_18599)));
        int64_t flat_idx_18609 = chunk_offset_18608 +
                sext_i32_i64(local_tid_18601);
        int64_t gtid_18307 = flat_idx_18609;
        
        // threads in bounds read input
        {
            if (slt64(gtid_18307, paths_17456)) {
                // write to-scan values to parameters
                {
                    x_17594 = res_17605;
                }
                // write mapped values results to global memory
                {
                    ((__global int64_t *) mem_18507)[gtid_18307] = res_17605;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_18307, paths_17456)) {
                    x_17594 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int64_t res_17595 = add64(x_17593, x_17594);
                
                ((__local
                  int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)] =
                    res_17595;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int64_t x_18610;
            int64_t x_18611;
            int64_t x_18613;
            int64_t x_18614;
            bool ltid_in_bounds_18616;
            
            ltid_in_bounds_18616 = slt64(sext_i32_i64(local_tid_18601),
                                         segscan_group_sizze_18303);
            
            int32_t skip_threads_18617;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_18616) {
                    x_18611 = ((volatile __local
                                int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)];
                    if ((local_tid_18601 - squot32(local_tid_18601, 32) * 32) ==
                        0) {
                        x_18610 = x_18611;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_18617 = 1;
                while (slt32(skip_threads_18617, 32)) {
                    if (sle32(skip_threads_18617, local_tid_18601 -
                              squot32(local_tid_18601, 32) * 32) &&
                        ltid_in_bounds_18616) {
                        // read operands
                        {
                            x_18610 = ((volatile __local
                                        int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601) -
                                                                       sext_i32_i64(skip_threads_18617)];
                        }
                        // perform operation
                        {
                            int64_t res_18612 = add64(x_18610, x_18611);
                            
                            x_18610 = res_18612;
                        }
                    }
                    if (sle32(wave_sizze_18603, skip_threads_18617)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_18617, local_tid_18601 -
                              squot32(local_tid_18601, 32) * 32) &&
                        ltid_in_bounds_18616) {
                        // write result
                        {
                            ((volatile __local
                              int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)] =
                                x_18610;
                            x_18611 = x_18610;
                        }
                    }
                    if (sle32(wave_sizze_18603, skip_threads_18617)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_18617 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_18601 - squot32(local_tid_18601, 32) * 32) ==
                    31 && ltid_in_bounds_18616) {
                    ((volatile __local
                      int64_t *) scan_arr_mem_18605)[sext_i32_i64(squot32(local_tid_18601,
                                                                          32))] =
                        x_18610;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_18618;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_18601, 32) == 0 &&
                        ltid_in_bounds_18616) {
                        x_18614 = ((volatile __local
                                    int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)];
                        if ((local_tid_18601 - squot32(local_tid_18601, 32) *
                             32) == 0) {
                            x_18613 = x_18614;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_18618 = 1;
                    while (slt32(skip_threads_18618, 32)) {
                        if (sle32(skip_threads_18618, local_tid_18601 -
                                  squot32(local_tid_18601, 32) * 32) &&
                            (squot32(local_tid_18601, 32) == 0 &&
                             ltid_in_bounds_18616)) {
                            // read operands
                            {
                                x_18613 = ((volatile __local
                                            int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601) -
                                                                           sext_i32_i64(skip_threads_18618)];
                            }
                            // perform operation
                            {
                                int64_t res_18615 = add64(x_18613, x_18614);
                                
                                x_18613 = res_18615;
                            }
                        }
                        if (sle32(wave_sizze_18603, skip_threads_18618)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_18618, local_tid_18601 -
                                  squot32(local_tid_18601, 32) * 32) &&
                            (squot32(local_tid_18601, 32) == 0 &&
                             ltid_in_bounds_18616)) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)] =
                                    x_18613;
                                x_18614 = x_18613;
                            }
                        }
                        if (sle32(wave_sizze_18603, skip_threads_18618)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_18618 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_18601, 32) == 0 ||
                      !ltid_in_bounds_18616)) {
                    // read operands
                    {
                        x_18611 = x_18610;
                        x_18610 = ((__local
                                    int64_t *) scan_arr_mem_18605)[sext_i32_i64(squot32(local_tid_18601,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        int64_t res_18612 = add64(x_18610, x_18611);
                        
                        x_18610 = res_18612;
                    }
                    // write final result
                    {
                        ((__local
                          int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)] =
                            x_18610;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_18601, 32) == 0) {
                    ((__local
                      int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)] =
                        x_18611;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_18307, paths_17456)) {
                    ((__global int64_t *) mem_18505)[gtid_18307] = ((__local
                                                                     int64_t *) scan_arr_mem_18605)[sext_i32_i64(local_tid_18601)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_18619 = 0;
                bool should_load_carry_18620 = local_tid_18601 == 0 &&
                     !crosses_segment_18619;
                
                if (should_load_carry_18620) {
                    x_17593 = ((__local
                                int64_t *) scan_arr_mem_18605)[segscan_group_sizze_18303 -
                                                               1];
                }
                if (!should_load_carry_18620) {
                    x_17593 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_18303
}
__kernel void mainziscan_stage1_18341(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_18844_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_18842_backing_aligned_1,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18512, __global
                                      unsigned char *mem_18516, __global
                                      unsigned char *mem_18518,
                                      int32_t num_threads_18836)
{
    #define segscan_group_sizze_18336 (mainzisegscan_group_sizze_18335)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18844_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_18844_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_18842_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18842_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18837;
    int32_t local_tid_18838;
    int64_t group_sizze_18841;
    int32_t wave_sizze_18840;
    int32_t group_tid_18839;
    
    global_tid_18837 = get_global_id(0);
    local_tid_18838 = get_local_id(0);
    group_sizze_18841 = get_local_size(0);
    wave_sizze_18840 = LOCKSTEP_WIDTH;
    group_tid_18839 = get_group_id(0);
    
    int32_t phys_tid_18341;
    
    phys_tid_18341 = global_tid_18837;
    
    __local char *scan_arr_mem_18842;
    __local char *scan_arr_mem_18844;
    
    scan_arr_mem_18842 = (__local char *) scan_arr_mem_18842_backing_0;
    scan_arr_mem_18844 = (__local char *) scan_arr_mem_18844_backing_1;
    
    bool x_17626;
    int64_t x_17627;
    bool x_17628;
    int64_t x_17629;
    
    x_17626 = 0;
    x_17627 = 0;
    for (int64_t j_18846 = 0; j_18846 < sdiv_up64(res_17607,
                                                  sext_i32_i64(num_threads_18836));
         j_18846++) {
        int64_t chunk_offset_18847 = segscan_group_sizze_18336 * j_18846 +
                sext_i32_i64(group_tid_18839) * (segscan_group_sizze_18336 *
                                                 sdiv_up64(res_17607,
                                                           sext_i32_i64(num_threads_18836)));
        int64_t flat_idx_18848 = chunk_offset_18847 +
                sext_i32_i64(local_tid_18838);
        int64_t gtid_18340 = flat_idx_18848;
        
        // threads in bounds read input
        {
            if (slt64(gtid_18340, res_17607)) {
                int64_t x_17633 = ((__global int64_t *) mem_18512)[gtid_18340];
                bool res_17634 = slt64(0, x_17633);
                
                // write to-scan values to parameters
                {
                    x_17628 = res_17634;
                    x_17629 = x_17633;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_18340, res_17607)) {
                    x_17628 = 0;
                    x_17629 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_17630 = x_17626 || x_17628;
                int64_t res_17631;
                
                if (x_17628) {
                    res_17631 = x_17629;
                } else {
                    int64_t res_17632 = add64(x_17627, x_17629);
                    
                    res_17631 = res_17632;
                }
                ((__local
                  bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)] =
                    res_17630;
                ((__local
                  int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)] =
                    res_17631;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_18849;
            int64_t x_18850;
            bool x_18851;
            int64_t x_18852;
            bool x_18856;
            int64_t x_18857;
            bool x_18858;
            int64_t x_18859;
            bool ltid_in_bounds_18863;
            
            ltid_in_bounds_18863 = slt64(sext_i32_i64(local_tid_18838),
                                         segscan_group_sizze_18336);
            
            int32_t skip_threads_18864;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_18863) {
                    x_18851 = ((volatile __local
                                bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)];
                    x_18852 = ((volatile __local
                                int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)];
                    if ((local_tid_18838 - squot32(local_tid_18838, 32) * 32) ==
                        0) {
                        x_18849 = x_18851;
                        x_18850 = x_18852;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_18864 = 1;
                while (slt32(skip_threads_18864, 32)) {
                    if (sle32(skip_threads_18864, local_tid_18838 -
                              squot32(local_tid_18838, 32) * 32) &&
                        ltid_in_bounds_18863) {
                        // read operands
                        {
                            x_18849 = ((volatile __local
                                        bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838) -
                                                                    sext_i32_i64(skip_threads_18864)];
                            x_18850 = ((volatile __local
                                        int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838) -
                                                                       sext_i32_i64(skip_threads_18864)];
                        }
                        // perform operation
                        {
                            bool res_18853 = x_18849 || x_18851;
                            int64_t res_18854;
                            
                            if (x_18851) {
                                res_18854 = x_18852;
                            } else {
                                int64_t res_18855 = add64(x_18850, x_18852);
                                
                                res_18854 = res_18855;
                            }
                            x_18849 = res_18853;
                            x_18850 = res_18854;
                        }
                    }
                    if (sle32(wave_sizze_18840, skip_threads_18864)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_18864, local_tid_18838 -
                              squot32(local_tid_18838, 32) * 32) &&
                        ltid_in_bounds_18863) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)] =
                                x_18849;
                            x_18851 = x_18849;
                            ((volatile __local
                              int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)] =
                                x_18850;
                            x_18852 = x_18850;
                        }
                    }
                    if (sle32(wave_sizze_18840, skip_threads_18864)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_18864 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_18838 - squot32(local_tid_18838, 32) * 32) ==
                    31 && ltid_in_bounds_18863) {
                    ((volatile __local
                      bool *) scan_arr_mem_18842)[sext_i32_i64(squot32(local_tid_18838,
                                                                       32))] =
                        x_18849;
                    ((volatile __local
                      int64_t *) scan_arr_mem_18844)[sext_i32_i64(squot32(local_tid_18838,
                                                                          32))] =
                        x_18850;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_18865;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_18838, 32) == 0 &&
                        ltid_in_bounds_18863) {
                        x_18858 = ((volatile __local
                                    bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)];
                        x_18859 = ((volatile __local
                                    int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)];
                        if ((local_tid_18838 - squot32(local_tid_18838, 32) *
                             32) == 0) {
                            x_18856 = x_18858;
                            x_18857 = x_18859;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_18865 = 1;
                    while (slt32(skip_threads_18865, 32)) {
                        if (sle32(skip_threads_18865, local_tid_18838 -
                                  squot32(local_tid_18838, 32) * 32) &&
                            (squot32(local_tid_18838, 32) == 0 &&
                             ltid_in_bounds_18863)) {
                            // read operands
                            {
                                x_18856 = ((volatile __local
                                            bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838) -
                                                                        sext_i32_i64(skip_threads_18865)];
                                x_18857 = ((volatile __local
                                            int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838) -
                                                                           sext_i32_i64(skip_threads_18865)];
                            }
                            // perform operation
                            {
                                bool res_18860 = x_18856 || x_18858;
                                int64_t res_18861;
                                
                                if (x_18858) {
                                    res_18861 = x_18859;
                                } else {
                                    int64_t res_18862 = add64(x_18857, x_18859);
                                    
                                    res_18861 = res_18862;
                                }
                                x_18856 = res_18860;
                                x_18857 = res_18861;
                            }
                        }
                        if (sle32(wave_sizze_18840, skip_threads_18865)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_18865, local_tid_18838 -
                                  squot32(local_tid_18838, 32) * 32) &&
                            (squot32(local_tid_18838, 32) == 0 &&
                             ltid_in_bounds_18863)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)] =
                                    x_18856;
                                x_18858 = x_18856;
                                ((volatile __local
                                  int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)] =
                                    x_18857;
                                x_18859 = x_18857;
                            }
                        }
                        if (sle32(wave_sizze_18840, skip_threads_18865)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_18865 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_18838, 32) == 0 ||
                      !ltid_in_bounds_18863)) {
                    // read operands
                    {
                        x_18851 = x_18849;
                        x_18852 = x_18850;
                        x_18849 = ((__local
                                    bool *) scan_arr_mem_18842)[sext_i32_i64(squot32(local_tid_18838,
                                                                                     32)) -
                                                                1];
                        x_18850 = ((__local
                                    int64_t *) scan_arr_mem_18844)[sext_i32_i64(squot32(local_tid_18838,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        bool res_18853 = x_18849 || x_18851;
                        int64_t res_18854;
                        
                        if (x_18851) {
                            res_18854 = x_18852;
                        } else {
                            int64_t res_18855 = add64(x_18850, x_18852);
                            
                            res_18854 = res_18855;
                        }
                        x_18849 = res_18853;
                        x_18850 = res_18854;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)] =
                            x_18849;
                        ((__local
                          int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)] =
                            x_18850;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_18838, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)] =
                        x_18851;
                    ((__local
                      int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)] =
                        x_18852;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_18340, res_17607)) {
                    ((__global bool *) mem_18516)[gtid_18340] = ((__local
                                                                  bool *) scan_arr_mem_18842)[sext_i32_i64(local_tid_18838)];
                    ((__global int64_t *) mem_18518)[gtid_18340] = ((__local
                                                                     int64_t *) scan_arr_mem_18844)[sext_i32_i64(local_tid_18838)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_18866 = 0;
                bool should_load_carry_18867 = local_tid_18838 == 0 &&
                     !crosses_segment_18866;
                
                if (should_load_carry_18867) {
                    x_17626 = ((__local
                                bool *) scan_arr_mem_18842)[segscan_group_sizze_18336 -
                                                            1];
                    x_17627 = ((__local
                                int64_t *) scan_arr_mem_18844)[segscan_group_sizze_18336 -
                                                               1];
                }
                if (!should_load_carry_18867) {
                    x_17626 = 0;
                    x_17627 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_18336
}
__kernel void mainziscan_stage1_18349(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_18911_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_18909_backing_aligned_1,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18518, __global
                                      unsigned char *mem_18521, __global
                                      unsigned char *mem_18523, __global
                                      unsigned char *mem_18525,
                                      int32_t num_threads_18903)
{
    #define segscan_group_sizze_18344 (mainzisegscan_group_sizze_18343)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18911_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_18911_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_18909_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18909_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18904;
    int32_t local_tid_18905;
    int64_t group_sizze_18908;
    int32_t wave_sizze_18907;
    int32_t group_tid_18906;
    
    global_tid_18904 = get_global_id(0);
    local_tid_18905 = get_local_id(0);
    group_sizze_18908 = get_local_size(0);
    wave_sizze_18907 = LOCKSTEP_WIDTH;
    group_tid_18906 = get_group_id(0);
    
    int32_t phys_tid_18349;
    
    phys_tid_18349 = global_tid_18904;
    
    __local char *scan_arr_mem_18909;
    __local char *scan_arr_mem_18911;
    
    scan_arr_mem_18909 = (__local char *) scan_arr_mem_18909_backing_0;
    scan_arr_mem_18911 = (__local char *) scan_arr_mem_18911_backing_1;
    
    bool x_17665;
    int64_t x_17666;
    bool x_17667;
    int64_t x_17668;
    
    x_17665 = 0;
    x_17666 = 0;
    for (int64_t j_18913 = 0; j_18913 < sdiv_up64(res_17607,
                                                  sext_i32_i64(num_threads_18903));
         j_18913++) {
        int64_t chunk_offset_18914 = segscan_group_sizze_18344 * j_18913 +
                sext_i32_i64(group_tid_18906) * (segscan_group_sizze_18344 *
                                                 sdiv_up64(res_17607,
                                                           sext_i32_i64(num_threads_18903)));
        int64_t flat_idx_18915 = chunk_offset_18914 +
                sext_i32_i64(local_tid_18905);
        int64_t gtid_18348 = flat_idx_18915;
        
        // threads in bounds read input
        {
            if (slt64(gtid_18348, res_17607)) {
                int64_t x_17672 = ((__global int64_t *) mem_18518)[gtid_18348];
                int64_t i_p_o_18441 = add64(-1, gtid_18348);
                int64_t rot_i_18442 = smod64(i_p_o_18441, res_17607);
                int64_t x_17673 = ((__global int64_t *) mem_18518)[rot_i_18442];
                bool res_17675 = x_17672 == x_17673;
                bool res_17676 = !res_17675;
                
                // write to-scan values to parameters
                {
                    x_17667 = res_17676;
                    x_17668 = 1;
                }
                // write mapped values results to global memory
                {
                    ((__global bool *) mem_18525)[gtid_18348] = res_17676;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_18348, res_17607)) {
                    x_17667 = 0;
                    x_17668 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_17669 = x_17665 || x_17667;
                int64_t res_17670;
                
                if (x_17667) {
                    res_17670 = x_17668;
                } else {
                    int64_t res_17671 = add64(x_17666, x_17668);
                    
                    res_17670 = res_17671;
                }
                ((__local
                  bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)] =
                    res_17669;
                ((__local
                  int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)] =
                    res_17670;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_18916;
            int64_t x_18917;
            bool x_18918;
            int64_t x_18919;
            bool x_18923;
            int64_t x_18924;
            bool x_18925;
            int64_t x_18926;
            bool ltid_in_bounds_18930;
            
            ltid_in_bounds_18930 = slt64(sext_i32_i64(local_tid_18905),
                                         segscan_group_sizze_18344);
            
            int32_t skip_threads_18931;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_18930) {
                    x_18918 = ((volatile __local
                                bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)];
                    x_18919 = ((volatile __local
                                int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)];
                    if ((local_tid_18905 - squot32(local_tid_18905, 32) * 32) ==
                        0) {
                        x_18916 = x_18918;
                        x_18917 = x_18919;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_18931 = 1;
                while (slt32(skip_threads_18931, 32)) {
                    if (sle32(skip_threads_18931, local_tid_18905 -
                              squot32(local_tid_18905, 32) * 32) &&
                        ltid_in_bounds_18930) {
                        // read operands
                        {
                            x_18916 = ((volatile __local
                                        bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905) -
                                                                    sext_i32_i64(skip_threads_18931)];
                            x_18917 = ((volatile __local
                                        int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905) -
                                                                       sext_i32_i64(skip_threads_18931)];
                        }
                        // perform operation
                        {
                            bool res_18920 = x_18916 || x_18918;
                            int64_t res_18921;
                            
                            if (x_18918) {
                                res_18921 = x_18919;
                            } else {
                                int64_t res_18922 = add64(x_18917, x_18919);
                                
                                res_18921 = res_18922;
                            }
                            x_18916 = res_18920;
                            x_18917 = res_18921;
                        }
                    }
                    if (sle32(wave_sizze_18907, skip_threads_18931)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_18931, local_tid_18905 -
                              squot32(local_tid_18905, 32) * 32) &&
                        ltid_in_bounds_18930) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)] =
                                x_18916;
                            x_18918 = x_18916;
                            ((volatile __local
                              int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)] =
                                x_18917;
                            x_18919 = x_18917;
                        }
                    }
                    if (sle32(wave_sizze_18907, skip_threads_18931)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_18931 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_18905 - squot32(local_tid_18905, 32) * 32) ==
                    31 && ltid_in_bounds_18930) {
                    ((volatile __local
                      bool *) scan_arr_mem_18909)[sext_i32_i64(squot32(local_tid_18905,
                                                                       32))] =
                        x_18916;
                    ((volatile __local
                      int64_t *) scan_arr_mem_18911)[sext_i32_i64(squot32(local_tid_18905,
                                                                          32))] =
                        x_18917;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_18932;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_18905, 32) == 0 &&
                        ltid_in_bounds_18930) {
                        x_18925 = ((volatile __local
                                    bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)];
                        x_18926 = ((volatile __local
                                    int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)];
                        if ((local_tid_18905 - squot32(local_tid_18905, 32) *
                             32) == 0) {
                            x_18923 = x_18925;
                            x_18924 = x_18926;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_18932 = 1;
                    while (slt32(skip_threads_18932, 32)) {
                        if (sle32(skip_threads_18932, local_tid_18905 -
                                  squot32(local_tid_18905, 32) * 32) &&
                            (squot32(local_tid_18905, 32) == 0 &&
                             ltid_in_bounds_18930)) {
                            // read operands
                            {
                                x_18923 = ((volatile __local
                                            bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905) -
                                                                        sext_i32_i64(skip_threads_18932)];
                                x_18924 = ((volatile __local
                                            int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905) -
                                                                           sext_i32_i64(skip_threads_18932)];
                            }
                            // perform operation
                            {
                                bool res_18927 = x_18923 || x_18925;
                                int64_t res_18928;
                                
                                if (x_18925) {
                                    res_18928 = x_18926;
                                } else {
                                    int64_t res_18929 = add64(x_18924, x_18926);
                                    
                                    res_18928 = res_18929;
                                }
                                x_18923 = res_18927;
                                x_18924 = res_18928;
                            }
                        }
                        if (sle32(wave_sizze_18907, skip_threads_18932)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_18932, local_tid_18905 -
                                  squot32(local_tid_18905, 32) * 32) &&
                            (squot32(local_tid_18905, 32) == 0 &&
                             ltid_in_bounds_18930)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)] =
                                    x_18923;
                                x_18925 = x_18923;
                                ((volatile __local
                                  int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)] =
                                    x_18924;
                                x_18926 = x_18924;
                            }
                        }
                        if (sle32(wave_sizze_18907, skip_threads_18932)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_18932 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_18905, 32) == 0 ||
                      !ltid_in_bounds_18930)) {
                    // read operands
                    {
                        x_18918 = x_18916;
                        x_18919 = x_18917;
                        x_18916 = ((__local
                                    bool *) scan_arr_mem_18909)[sext_i32_i64(squot32(local_tid_18905,
                                                                                     32)) -
                                                                1];
                        x_18917 = ((__local
                                    int64_t *) scan_arr_mem_18911)[sext_i32_i64(squot32(local_tid_18905,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        bool res_18920 = x_18916 || x_18918;
                        int64_t res_18921;
                        
                        if (x_18918) {
                            res_18921 = x_18919;
                        } else {
                            int64_t res_18922 = add64(x_18917, x_18919);
                            
                            res_18921 = res_18922;
                        }
                        x_18916 = res_18920;
                        x_18917 = res_18921;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)] =
                            x_18916;
                        ((__local
                          int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)] =
                            x_18917;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_18905, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)] =
                        x_18918;
                    ((__local
                      int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)] =
                        x_18919;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_18348, res_17607)) {
                    ((__global bool *) mem_18521)[gtid_18348] = ((__local
                                                                  bool *) scan_arr_mem_18909)[sext_i32_i64(local_tid_18905)];
                    ((__global int64_t *) mem_18523)[gtid_18348] = ((__local
                                                                     int64_t *) scan_arr_mem_18911)[sext_i32_i64(local_tid_18905)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_18933 = 0;
                bool should_load_carry_18934 = local_tid_18905 == 0 &&
                     !crosses_segment_18933;
                
                if (should_load_carry_18934) {
                    x_17665 = ((__local
                                bool *) scan_arr_mem_18909)[segscan_group_sizze_18344 -
                                                            1];
                    x_17666 = ((__local
                                int64_t *) scan_arr_mem_18911)[segscan_group_sizze_18344 -
                                                               1];
                }
                if (!should_load_carry_18934) {
                    x_17665 = 0;
                    x_17666 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_18344
}
__kernel void mainziscan_stage1_18357(__global int *global_failure,
                                      int failure_is_an_option, __global
                                      int64_t *global_failure_args,
                                      __local volatile
                                      int64_t *scan_arr_mem_18978_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_18976_backing_aligned_1,
                                      int64_t paths_17456,
                                      float swap_term_17458,
                                      int64_t payments_17459,
                                      float notional_17460, float a_17461,
                                      float b_17462, float sigma_17463,
                                      float res_17590, int64_t res_17607,
                                      int64_t i_18299, __global
                                      unsigned char *mem_18493, __global
                                      unsigned char *mem_18518, __global
                                      unsigned char *mem_18523, __global
                                      unsigned char *mem_18525, __global
                                      unsigned char *mem_18528, __global
                                      unsigned char *mem_18530,
                                      int32_t num_threads_18970)
{
    #define segscan_group_sizze_18352 (mainzisegscan_group_sizze_18351)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18978_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_18978_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_18976_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18976_backing_aligned_1;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_18971;
    int32_t local_tid_18972;
    int64_t group_sizze_18975;
    int32_t wave_sizze_18974;
    int32_t group_tid_18973;
    
    global_tid_18971 = get_global_id(0);
    local_tid_18972 = get_local_id(0);
    group_sizze_18975 = get_local_size(0);
    wave_sizze_18974 = LOCKSTEP_WIDTH;
    group_tid_18973 = get_group_id(0);
    
    int32_t phys_tid_18357;
    
    phys_tid_18357 = global_tid_18971;
    
    __local char *scan_arr_mem_18976;
    __local char *scan_arr_mem_18978;
    
    scan_arr_mem_18976 = (__local char *) scan_arr_mem_18976_backing_0;
    scan_arr_mem_18978 = (__local char *) scan_arr_mem_18978_backing_1;
    
    bool x_17691;
    float x_17692;
    bool x_17693;
    float x_17694;
    
    x_17691 = 0;
    x_17692 = 0.0F;
    for (int64_t j_18980 = 0; j_18980 < sdiv_up64(res_17607,
                                                  sext_i32_i64(num_threads_18970));
         j_18980++) {
        int64_t chunk_offset_18981 = segscan_group_sizze_18352 * j_18980 +
                sext_i32_i64(group_tid_18973) * (segscan_group_sizze_18352 *
                                                 sdiv_up64(res_17607,
                                                           sext_i32_i64(num_threads_18970)));
        int64_t flat_idx_18982 = chunk_offset_18981 +
                sext_i32_i64(local_tid_18972);
        int64_t gtid_18356 = flat_idx_18982;
        
        // threads in bounds read input
        {
            if (slt64(gtid_18356, res_17607)) {
                int64_t x_17699 = ((__global int64_t *) mem_18523)[gtid_18356];
                int64_t x_17700 = ((__global int64_t *) mem_18518)[gtid_18356];
                bool x_17701 = ((__global bool *) mem_18525)[gtid_18356];
                int64_t res_17704 = sub64(x_17699, 1);
                bool x_17705 = sle64(0, x_17700);
                bool y_17706 = slt64(x_17700, paths_17456);
                bool bounds_check_17707 = x_17705 && y_17706;
                bool index_certs_17708;
                
                if (!bounds_check_17707) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 2) ==
                            -1) {
                            global_failure_args[0] = x_17700;
                            global_failure_args[1] = paths_17456;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float x_17718 = res_17590 / swap_term_17458;
                float ceil_arg_17719 = x_17718 - 1.0F;
                float res_17720;
                
                res_17720 = futrts_ceil32(ceil_arg_17719);
                
                int64_t res_17721 = fptosi_f32_i64(res_17720);
                int64_t res_17722 = sub64(payments_17459, res_17721);
                bool cond_17723 = res_17722 == 0;
                float res_17724;
                
                if (cond_17723) {
                    res_17724 = 0.0F;
                } else {
                    float lifted_0_get_arg_17709 = ((__global
                                                     float *) mem_18493)[i_18299 *
                                                                         paths_17456 +
                                                                         x_17700];
                    float res_17725;
                    
                    res_17725 = futrts_ceil32(x_17718);
                    
                    float start_17726 = swap_term_17458 * res_17725;
                    float res_17727;
                    
                    res_17727 = futrts_ceil32(ceil_arg_17719);
                    
                    int64_t res_17728 = fptosi_f32_i64(res_17727);
                    int64_t res_17729 = sub64(payments_17459, res_17728);
                    int64_t sizze_17730 = sub64(res_17729, 1);
                    bool cond_17731 = res_17704 == 0;
                    float res_17732;
                    
                    if (cond_17731) {
                        res_17732 = 1.0F;
                    } else {
                        res_17732 = 0.0F;
                    }
                    
                    bool cond_17733 = slt64(0, res_17704);
                    float res_17734;
                    
                    if (cond_17733) {
                        float y_17735 = 5.056644e-2F * swap_term_17458;
                        float res_17736 = res_17732 - y_17735;
                        
                        res_17734 = res_17736;
                    } else {
                        res_17734 = res_17732;
                    }
                    
                    bool cond_17737 = res_17704 == sizze_17730;
                    float res_17738;
                    
                    if (cond_17737) {
                        float res_17739 = res_17734 - 1.0F;
                        
                        res_17738 = res_17739;
                    } else {
                        res_17738 = res_17734;
                    }
                    
                    float res_17740 = notional_17460 * res_17738;
                    float res_17741 = sitofp_i64_f32(res_17704);
                    float y_17742 = swap_term_17458 * res_17741;
                    float bondprice_arg_17743 = start_17726 + y_17742;
                    float y_17744 = bondprice_arg_17743 - res_17590;
                    float negate_arg_17745 = a_17461 * y_17744;
                    float exp_arg_17746 = 0.0F - negate_arg_17745;
                    float res_17747 = fpow32(2.7182817F, exp_arg_17746);
                    float x_17748 = 1.0F - res_17747;
                    float B_17749 = x_17748 / a_17461;
                    float x_17750 = B_17749 - bondprice_arg_17743;
                    float x_17751 = res_17590 + x_17750;
                    float x_17752 = fpow32(a_17461, 2.0F);
                    float x_17753 = b_17462 * x_17752;
                    float x_17754 = fpow32(sigma_17463, 2.0F);
                    float y_17755 = x_17754 / 2.0F;
                    float y_17756 = x_17753 - y_17755;
                    float x_17757 = x_17751 * y_17756;
                    float A1_17758 = x_17757 / x_17752;
                    float y_17759 = fpow32(B_17749, 2.0F);
                    float x_17760 = x_17754 * y_17759;
                    float y_17761 = 4.0F * a_17461;
                    float A2_17762 = x_17760 / y_17761;
                    float exp_arg_17763 = A1_17758 - A2_17762;
                    float res_17764 = fpow32(2.7182817F, exp_arg_17763);
                    float negate_arg_17765 = lifted_0_get_arg_17709 * B_17749;
                    float exp_arg_17766 = 0.0F - negate_arg_17765;
                    float res_17767 = fpow32(2.7182817F, exp_arg_17766);
                    float res_17768 = res_17764 * res_17767;
                    float res_17769 = res_17740 * res_17768;
                    
                    res_17724 = res_17769;
                }
                // write to-scan values to parameters
                {
                    x_17693 = x_17701;
                    x_17694 = res_17724;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_18356, res_17607)) {
                    x_17693 = 0;
                    x_17694 = 0.0F;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_17695 = x_17691 || x_17693;
                float res_17696;
                
                if (x_17693) {
                    res_17696 = x_17694;
                } else {
                    float res_17697 = x_17692 + x_17694;
                    
                    res_17696 = res_17697;
                }
                ((__local
                  bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)] =
                    res_17695;
                ((__local
                  float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)] =
                    res_17696;
            }
            
          error_0:
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_failure)
                return;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_18983;
            float x_18984;
            bool x_18985;
            float x_18986;
            bool x_18990;
            float x_18991;
            bool x_18992;
            float x_18993;
            bool ltid_in_bounds_18997;
            
            ltid_in_bounds_18997 = slt64(sext_i32_i64(local_tid_18972),
                                         segscan_group_sizze_18352);
            
            int32_t skip_threads_18998;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_18997) {
                    x_18985 = ((volatile __local
                                bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)];
                    x_18986 = ((volatile __local
                                float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)];
                    if ((local_tid_18972 - squot32(local_tid_18972, 32) * 32) ==
                        0) {
                        x_18983 = x_18985;
                        x_18984 = x_18986;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_18998 = 1;
                while (slt32(skip_threads_18998, 32)) {
                    if (sle32(skip_threads_18998, local_tid_18972 -
                              squot32(local_tid_18972, 32) * 32) &&
                        ltid_in_bounds_18997) {
                        // read operands
                        {
                            x_18983 = ((volatile __local
                                        bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972) -
                                                                    sext_i32_i64(skip_threads_18998)];
                            x_18984 = ((volatile __local
                                        float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972) -
                                                                     sext_i32_i64(skip_threads_18998)];
                        }
                        // perform operation
                        {
                            bool res_18987 = x_18983 || x_18985;
                            float res_18988;
                            
                            if (x_18985) {
                                res_18988 = x_18986;
                            } else {
                                float res_18989 = x_18984 + x_18986;
                                
                                res_18988 = res_18989;
                            }
                            x_18983 = res_18987;
                            x_18984 = res_18988;
                        }
                    }
                    if (sle32(wave_sizze_18974, skip_threads_18998)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_18998, local_tid_18972 -
                              squot32(local_tid_18972, 32) * 32) &&
                        ltid_in_bounds_18997) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)] =
                                x_18983;
                            x_18985 = x_18983;
                            ((volatile __local
                              float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)] =
                                x_18984;
                            x_18986 = x_18984;
                        }
                    }
                    if (sle32(wave_sizze_18974, skip_threads_18998)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_18998 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_18972 - squot32(local_tid_18972, 32) * 32) ==
                    31 && ltid_in_bounds_18997) {
                    ((volatile __local
                      bool *) scan_arr_mem_18976)[sext_i32_i64(squot32(local_tid_18972,
                                                                       32))] =
                        x_18983;
                    ((volatile __local
                      float *) scan_arr_mem_18978)[sext_i32_i64(squot32(local_tid_18972,
                                                                        32))] =
                        x_18984;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_18999;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_18972, 32) == 0 &&
                        ltid_in_bounds_18997) {
                        x_18992 = ((volatile __local
                                    bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)];
                        x_18993 = ((volatile __local
                                    float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)];
                        if ((local_tid_18972 - squot32(local_tid_18972, 32) *
                             32) == 0) {
                            x_18990 = x_18992;
                            x_18991 = x_18993;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_18999 = 1;
                    while (slt32(skip_threads_18999, 32)) {
                        if (sle32(skip_threads_18999, local_tid_18972 -
                                  squot32(local_tid_18972, 32) * 32) &&
                            (squot32(local_tid_18972, 32) == 0 &&
                             ltid_in_bounds_18997)) {
                            // read operands
                            {
                                x_18990 = ((volatile __local
                                            bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972) -
                                                                        sext_i32_i64(skip_threads_18999)];
                                x_18991 = ((volatile __local
                                            float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972) -
                                                                         sext_i32_i64(skip_threads_18999)];
                            }
                            // perform operation
                            {
                                bool res_18994 = x_18990 || x_18992;
                                float res_18995;
                                
                                if (x_18992) {
                                    res_18995 = x_18993;
                                } else {
                                    float res_18996 = x_18991 + x_18993;
                                    
                                    res_18995 = res_18996;
                                }
                                x_18990 = res_18994;
                                x_18991 = res_18995;
                            }
                        }
                        if (sle32(wave_sizze_18974, skip_threads_18999)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_18999, local_tid_18972 -
                                  squot32(local_tid_18972, 32) * 32) &&
                            (squot32(local_tid_18972, 32) == 0 &&
                             ltid_in_bounds_18997)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)] =
                                    x_18990;
                                x_18992 = x_18990;
                                ((volatile __local
                                  float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)] =
                                    x_18991;
                                x_18993 = x_18991;
                            }
                        }
                        if (sle32(wave_sizze_18974, skip_threads_18999)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_18999 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_18972, 32) == 0 ||
                      !ltid_in_bounds_18997)) {
                    // read operands
                    {
                        x_18985 = x_18983;
                        x_18986 = x_18984;
                        x_18983 = ((__local
                                    bool *) scan_arr_mem_18976)[sext_i32_i64(squot32(local_tid_18972,
                                                                                     32)) -
                                                                1];
                        x_18984 = ((__local
                                    float *) scan_arr_mem_18978)[sext_i32_i64(squot32(local_tid_18972,
                                                                                      32)) -
                                                                 1];
                    }
                    // perform operation
                    {
                        bool res_18987 = x_18983 || x_18985;
                        float res_18988;
                        
                        if (x_18985) {
                            res_18988 = x_18986;
                        } else {
                            float res_18989 = x_18984 + x_18986;
                            
                            res_18988 = res_18989;
                        }
                        x_18983 = res_18987;
                        x_18984 = res_18988;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)] =
                            x_18983;
                        ((__local
                          float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)] =
                            x_18984;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_18972, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)] =
                        x_18985;
                    ((__local
                      float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)] =
                        x_18986;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_18356, res_17607)) {
                    ((__global bool *) mem_18528)[gtid_18356] = ((__local
                                                                  bool *) scan_arr_mem_18976)[sext_i32_i64(local_tid_18972)];
                    ((__global float *) mem_18530)[gtid_18356] = ((__local
                                                                   float *) scan_arr_mem_18978)[sext_i32_i64(local_tid_18972)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_19000 = 0;
                bool should_load_carry_19001 = local_tid_18972 == 0 &&
                     !crosses_segment_19000;
                
                if (should_load_carry_19001) {
                    x_17691 = ((__local
                                bool *) scan_arr_mem_18976)[segscan_group_sizze_18352 -
                                                            1];
                    x_17692 = ((__local
                                float *) scan_arr_mem_18978)[segscan_group_sizze_18352 -
                                                             1];
                }
                if (!should_load_carry_19001) {
                    x_17691 = 0;
                    x_17692 = 0.0F;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_18352
}
__kernel void mainziscan_stage1_18409(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_19043_backing_aligned_0,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18525, __global
                                      unsigned char *mem_18533,
                                      int32_t num_threads_19037)
{
    #define segscan_group_sizze_18404 (mainzisegscan_group_sizze_18403)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_19043_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_19043_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19038;
    int32_t local_tid_19039;
    int64_t group_sizze_19042;
    int32_t wave_sizze_19041;
    int32_t group_tid_19040;
    
    global_tid_19038 = get_global_id(0);
    local_tid_19039 = get_local_id(0);
    group_sizze_19042 = get_local_size(0);
    wave_sizze_19041 = LOCKSTEP_WIDTH;
    group_tid_19040 = get_group_id(0);
    
    int32_t phys_tid_18409;
    
    phys_tid_18409 = global_tid_19038;
    
    __local char *scan_arr_mem_19043;
    
    scan_arr_mem_19043 = (__local char *) scan_arr_mem_19043_backing_0;
    
    int64_t x_17793;
    int64_t x_17794;
    
    x_17793 = 0;
    for (int64_t j_19045 = 0; j_19045 < sdiv_up64(res_17607,
                                                  sext_i32_i64(num_threads_19037));
         j_19045++) {
        int64_t chunk_offset_19046 = segscan_group_sizze_18404 * j_19045 +
                sext_i32_i64(group_tid_19040) * (segscan_group_sizze_18404 *
                                                 sdiv_up64(res_17607,
                                                           sext_i32_i64(num_threads_19037)));
        int64_t flat_idx_19047 = chunk_offset_19046 +
                sext_i32_i64(local_tid_19039);
        int64_t gtid_18408 = flat_idx_19047;
        
        // threads in bounds read input
        {
            if (slt64(gtid_18408, res_17607)) {
                int64_t i_p_o_18447 = add64(1, gtid_18408);
                int64_t rot_i_18448 = smod64(i_p_o_18447, res_17607);
                bool x_17796 = ((__global bool *) mem_18525)[rot_i_18448];
                int64_t res_17797 = btoi_bool_i64(x_17796);
                
                // write to-scan values to parameters
                {
                    x_17794 = res_17797;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_18408, res_17607)) {
                    x_17794 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int64_t res_17795 = add64(x_17793, x_17794);
                
                ((__local
                  int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)] =
                    res_17795;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int64_t x_19048;
            int64_t x_19049;
            int64_t x_19051;
            int64_t x_19052;
            bool ltid_in_bounds_19054;
            
            ltid_in_bounds_19054 = slt64(sext_i32_i64(local_tid_19039),
                                         segscan_group_sizze_18404);
            
            int32_t skip_threads_19055;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_19054) {
                    x_19049 = ((volatile __local
                                int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)];
                    if ((local_tid_19039 - squot32(local_tid_19039, 32) * 32) ==
                        0) {
                        x_19048 = x_19049;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_19055 = 1;
                while (slt32(skip_threads_19055, 32)) {
                    if (sle32(skip_threads_19055, local_tid_19039 -
                              squot32(local_tid_19039, 32) * 32) &&
                        ltid_in_bounds_19054) {
                        // read operands
                        {
                            x_19048 = ((volatile __local
                                        int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039) -
                                                                       sext_i32_i64(skip_threads_19055)];
                        }
                        // perform operation
                        {
                            int64_t res_19050 = add64(x_19048, x_19049);
                            
                            x_19048 = res_19050;
                        }
                    }
                    if (sle32(wave_sizze_19041, skip_threads_19055)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_19055, local_tid_19039 -
                              squot32(local_tid_19039, 32) * 32) &&
                        ltid_in_bounds_19054) {
                        // write result
                        {
                            ((volatile __local
                              int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)] =
                                x_19048;
                            x_19049 = x_19048;
                        }
                    }
                    if (sle32(wave_sizze_19041, skip_threads_19055)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_19055 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_19039 - squot32(local_tid_19039, 32) * 32) ==
                    31 && ltid_in_bounds_19054) {
                    ((volatile __local
                      int64_t *) scan_arr_mem_19043)[sext_i32_i64(squot32(local_tid_19039,
                                                                          32))] =
                        x_19048;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_19056;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_19039, 32) == 0 &&
                        ltid_in_bounds_19054) {
                        x_19052 = ((volatile __local
                                    int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)];
                        if ((local_tid_19039 - squot32(local_tid_19039, 32) *
                             32) == 0) {
                            x_19051 = x_19052;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_19056 = 1;
                    while (slt32(skip_threads_19056, 32)) {
                        if (sle32(skip_threads_19056, local_tid_19039 -
                                  squot32(local_tid_19039, 32) * 32) &&
                            (squot32(local_tid_19039, 32) == 0 &&
                             ltid_in_bounds_19054)) {
                            // read operands
                            {
                                x_19051 = ((volatile __local
                                            int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039) -
                                                                           sext_i32_i64(skip_threads_19056)];
                            }
                            // perform operation
                            {
                                int64_t res_19053 = add64(x_19051, x_19052);
                                
                                x_19051 = res_19053;
                            }
                        }
                        if (sle32(wave_sizze_19041, skip_threads_19056)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_19056, local_tid_19039 -
                                  squot32(local_tid_19039, 32) * 32) &&
                            (squot32(local_tid_19039, 32) == 0 &&
                             ltid_in_bounds_19054)) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)] =
                                    x_19051;
                                x_19052 = x_19051;
                            }
                        }
                        if (sle32(wave_sizze_19041, skip_threads_19056)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_19056 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_19039, 32) == 0 ||
                      !ltid_in_bounds_19054)) {
                    // read operands
                    {
                        x_19049 = x_19048;
                        x_19048 = ((__local
                                    int64_t *) scan_arr_mem_19043)[sext_i32_i64(squot32(local_tid_19039,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        int64_t res_19050 = add64(x_19048, x_19049);
                        
                        x_19048 = res_19050;
                    }
                    // write final result
                    {
                        ((__local
                          int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)] =
                            x_19048;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_19039, 32) == 0) {
                    ((__local
                      int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)] =
                        x_19049;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_18408, res_17607)) {
                    ((__global int64_t *) mem_18533)[gtid_18408] = ((__local
                                                                     int64_t *) scan_arr_mem_19043)[sext_i32_i64(local_tid_19039)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_19057 = 0;
                bool should_load_carry_19058 = local_tid_19039 == 0 &&
                     !crosses_segment_19057;
                
                if (should_load_carry_19058) {
                    x_17793 = ((__local
                                int64_t *) scan_arr_mem_19043)[segscan_group_sizze_18404 -
                                                               1];
                }
                if (!should_load_carry_19058) {
                    x_17793 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_18404
}
__kernel void mainziscan_stage2_18308(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_18626_backing_aligned_0,
                                      int64_t paths_17456, __global
                                      unsigned char *mem_18505,
                                      int64_t stage1_num_groups_18598,
                                      int32_t num_threads_18599)
{
    #define segscan_group_sizze_18303 (mainzisegscan_group_sizze_18302)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18626_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18626_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18621;
    int32_t local_tid_18622;
    int64_t group_sizze_18625;
    int32_t wave_sizze_18624;
    int32_t group_tid_18623;
    
    global_tid_18621 = get_global_id(0);
    local_tid_18622 = get_local_id(0);
    group_sizze_18625 = get_local_size(0);
    wave_sizze_18624 = LOCKSTEP_WIDTH;
    group_tid_18623 = get_group_id(0);
    
    int32_t phys_tid_18308;
    
    phys_tid_18308 = global_tid_18621;
    
    __local char *scan_arr_mem_18626;
    
    scan_arr_mem_18626 = (__local char *) scan_arr_mem_18626_backing_0;
    
    int64_t flat_idx_18628;
    
    flat_idx_18628 = (sext_i32_i64(local_tid_18622) + 1) *
        (segscan_group_sizze_18303 * sdiv_up64(paths_17456,
                                               sext_i32_i64(num_threads_18599))) -
        1;
    
    int64_t gtid_18307;
    
    gtid_18307 = flat_idx_18628;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_18307, paths_17456)) {
            ((__local
              int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)] =
                ((__global int64_t *) mem_18505)[gtid_18307];
        } else {
            ((__local
              int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int64_t x_17593;
    int64_t x_17594;
    int64_t x_18629;
    int64_t x_18630;
    bool ltid_in_bounds_18632;
    
    ltid_in_bounds_18632 = slt64(sext_i32_i64(local_tid_18622),
                                 stage1_num_groups_18598);
    
    int32_t skip_threads_18633;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_18632) {
            x_17594 = ((volatile __local
                        int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)];
            if ((local_tid_18622 - squot32(local_tid_18622, 32) * 32) == 0) {
                x_17593 = x_17594;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_18633 = 1;
        while (slt32(skip_threads_18633, 32)) {
            if (sle32(skip_threads_18633, local_tid_18622 -
                      squot32(local_tid_18622, 32) * 32) &&
                ltid_in_bounds_18632) {
                // read operands
                {
                    x_17593 = ((volatile __local
                                int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622) -
                                                               sext_i32_i64(skip_threads_18633)];
                }
                // perform operation
                {
                    int64_t res_17595 = add64(x_17593, x_17594);
                    
                    x_17593 = res_17595;
                }
            }
            if (sle32(wave_sizze_18624, skip_threads_18633)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_18633, local_tid_18622 -
                      squot32(local_tid_18622, 32) * 32) &&
                ltid_in_bounds_18632) {
                // write result
                {
                    ((volatile __local
                      int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)] =
                        x_17593;
                    x_17594 = x_17593;
                }
            }
            if (sle32(wave_sizze_18624, skip_threads_18633)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_18633 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_18622 - squot32(local_tid_18622, 32) * 32) == 31 &&
            ltid_in_bounds_18632) {
            ((volatile __local
              int64_t *) scan_arr_mem_18626)[sext_i32_i64(squot32(local_tid_18622,
                                                                  32))] =
                x_17593;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_18634;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_18622, 32) == 0 && ltid_in_bounds_18632) {
                x_18630 = ((volatile __local
                            int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)];
                if ((local_tid_18622 - squot32(local_tid_18622, 32) * 32) ==
                    0) {
                    x_18629 = x_18630;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_18634 = 1;
            while (slt32(skip_threads_18634, 32)) {
                if (sle32(skip_threads_18634, local_tid_18622 -
                          squot32(local_tid_18622, 32) * 32) &&
                    (squot32(local_tid_18622, 32) == 0 &&
                     ltid_in_bounds_18632)) {
                    // read operands
                    {
                        x_18629 = ((volatile __local
                                    int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622) -
                                                                   sext_i32_i64(skip_threads_18634)];
                    }
                    // perform operation
                    {
                        int64_t res_18631 = add64(x_18629, x_18630);
                        
                        x_18629 = res_18631;
                    }
                }
                if (sle32(wave_sizze_18624, skip_threads_18634)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_18634, local_tid_18622 -
                          squot32(local_tid_18622, 32) * 32) &&
                    (squot32(local_tid_18622, 32) == 0 &&
                     ltid_in_bounds_18632)) {
                    // write result
                    {
                        ((volatile __local
                          int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)] =
                            x_18629;
                        x_18630 = x_18629;
                    }
                }
                if (sle32(wave_sizze_18624, skip_threads_18634)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_18634 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_18622, 32) == 0 || !ltid_in_bounds_18632)) {
            // read operands
            {
                x_17594 = x_17593;
                x_17593 = ((__local
                            int64_t *) scan_arr_mem_18626)[sext_i32_i64(squot32(local_tid_18622,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                int64_t res_17595 = add64(x_17593, x_17594);
                
                x_17593 = res_17595;
            }
            // write final result
            {
                ((__local
                  int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)] =
                    x_17593;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_18622, 32) == 0) {
            ((__local
              int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)] =
                x_17594;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_18307, paths_17456)) {
            ((__global int64_t *) mem_18505)[gtid_18307] = ((__local
                                                             int64_t *) scan_arr_mem_18626)[sext_i32_i64(local_tid_18622)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18303
}
__kernel void mainziscan_stage2_18341(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_18875_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_18873_backing_aligned_1,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18516, __global
                                      unsigned char *mem_18518,
                                      int64_t stage1_num_groups_18835,
                                      int32_t num_threads_18836)
{
    #define segscan_group_sizze_18336 (mainzisegscan_group_sizze_18335)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18875_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_18875_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_18873_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18873_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18868;
    int32_t local_tid_18869;
    int64_t group_sizze_18872;
    int32_t wave_sizze_18871;
    int32_t group_tid_18870;
    
    global_tid_18868 = get_global_id(0);
    local_tid_18869 = get_local_id(0);
    group_sizze_18872 = get_local_size(0);
    wave_sizze_18871 = LOCKSTEP_WIDTH;
    group_tid_18870 = get_group_id(0);
    
    int32_t phys_tid_18341;
    
    phys_tid_18341 = global_tid_18868;
    
    __local char *scan_arr_mem_18873;
    __local char *scan_arr_mem_18875;
    
    scan_arr_mem_18873 = (__local char *) scan_arr_mem_18873_backing_0;
    scan_arr_mem_18875 = (__local char *) scan_arr_mem_18875_backing_1;
    
    int64_t flat_idx_18877;
    
    flat_idx_18877 = (sext_i32_i64(local_tid_18869) + 1) *
        (segscan_group_sizze_18336 * sdiv_up64(res_17607,
                                               sext_i32_i64(num_threads_18836))) -
        1;
    
    int64_t gtid_18340;
    
    gtid_18340 = flat_idx_18877;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_18340, res_17607)) {
            ((__local
              bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)] =
                ((__global bool *) mem_18516)[gtid_18340];
            ((__local
              int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)] =
                ((__global int64_t *) mem_18518)[gtid_18340];
        } else {
            ((__local
              bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)] = 0;
            ((__local
              int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_17626;
    int64_t x_17627;
    bool x_17628;
    int64_t x_17629;
    bool x_18878;
    int64_t x_18879;
    bool x_18880;
    int64_t x_18881;
    bool ltid_in_bounds_18885;
    
    ltid_in_bounds_18885 = slt64(sext_i32_i64(local_tid_18869),
                                 stage1_num_groups_18835);
    
    int32_t skip_threads_18886;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_18885) {
            x_17628 = ((volatile __local
                        bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)];
            x_17629 = ((volatile __local
                        int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)];
            if ((local_tid_18869 - squot32(local_tid_18869, 32) * 32) == 0) {
                x_17626 = x_17628;
                x_17627 = x_17629;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_18886 = 1;
        while (slt32(skip_threads_18886, 32)) {
            if (sle32(skip_threads_18886, local_tid_18869 -
                      squot32(local_tid_18869, 32) * 32) &&
                ltid_in_bounds_18885) {
                // read operands
                {
                    x_17626 = ((volatile __local
                                bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869) -
                                                            sext_i32_i64(skip_threads_18886)];
                    x_17627 = ((volatile __local
                                int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869) -
                                                               sext_i32_i64(skip_threads_18886)];
                }
                // perform operation
                {
                    bool res_17630 = x_17626 || x_17628;
                    int64_t res_17631;
                    
                    if (x_17628) {
                        res_17631 = x_17629;
                    } else {
                        int64_t res_17632 = add64(x_17627, x_17629);
                        
                        res_17631 = res_17632;
                    }
                    x_17626 = res_17630;
                    x_17627 = res_17631;
                }
            }
            if (sle32(wave_sizze_18871, skip_threads_18886)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_18886, local_tid_18869 -
                      squot32(local_tid_18869, 32) * 32) &&
                ltid_in_bounds_18885) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)] =
                        x_17626;
                    x_17628 = x_17626;
                    ((volatile __local
                      int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)] =
                        x_17627;
                    x_17629 = x_17627;
                }
            }
            if (sle32(wave_sizze_18871, skip_threads_18886)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_18886 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_18869 - squot32(local_tid_18869, 32) * 32) == 31 &&
            ltid_in_bounds_18885) {
            ((volatile __local
              bool *) scan_arr_mem_18873)[sext_i32_i64(squot32(local_tid_18869,
                                                               32))] = x_17626;
            ((volatile __local
              int64_t *) scan_arr_mem_18875)[sext_i32_i64(squot32(local_tid_18869,
                                                                  32))] =
                x_17627;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_18887;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_18869, 32) == 0 && ltid_in_bounds_18885) {
                x_18880 = ((volatile __local
                            bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)];
                x_18881 = ((volatile __local
                            int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)];
                if ((local_tid_18869 - squot32(local_tid_18869, 32) * 32) ==
                    0) {
                    x_18878 = x_18880;
                    x_18879 = x_18881;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_18887 = 1;
            while (slt32(skip_threads_18887, 32)) {
                if (sle32(skip_threads_18887, local_tid_18869 -
                          squot32(local_tid_18869, 32) * 32) &&
                    (squot32(local_tid_18869, 32) == 0 &&
                     ltid_in_bounds_18885)) {
                    // read operands
                    {
                        x_18878 = ((volatile __local
                                    bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869) -
                                                                sext_i32_i64(skip_threads_18887)];
                        x_18879 = ((volatile __local
                                    int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869) -
                                                                   sext_i32_i64(skip_threads_18887)];
                    }
                    // perform operation
                    {
                        bool res_18882 = x_18878 || x_18880;
                        int64_t res_18883;
                        
                        if (x_18880) {
                            res_18883 = x_18881;
                        } else {
                            int64_t res_18884 = add64(x_18879, x_18881);
                            
                            res_18883 = res_18884;
                        }
                        x_18878 = res_18882;
                        x_18879 = res_18883;
                    }
                }
                if (sle32(wave_sizze_18871, skip_threads_18887)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_18887, local_tid_18869 -
                          squot32(local_tid_18869, 32) * 32) &&
                    (squot32(local_tid_18869, 32) == 0 &&
                     ltid_in_bounds_18885)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)] =
                            x_18878;
                        x_18880 = x_18878;
                        ((volatile __local
                          int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)] =
                            x_18879;
                        x_18881 = x_18879;
                    }
                }
                if (sle32(wave_sizze_18871, skip_threads_18887)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_18887 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_18869, 32) == 0 || !ltid_in_bounds_18885)) {
            // read operands
            {
                x_17628 = x_17626;
                x_17629 = x_17627;
                x_17626 = ((__local
                            bool *) scan_arr_mem_18873)[sext_i32_i64(squot32(local_tid_18869,
                                                                             32)) -
                                                        1];
                x_17627 = ((__local
                            int64_t *) scan_arr_mem_18875)[sext_i32_i64(squot32(local_tid_18869,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                bool res_17630 = x_17626 || x_17628;
                int64_t res_17631;
                
                if (x_17628) {
                    res_17631 = x_17629;
                } else {
                    int64_t res_17632 = add64(x_17627, x_17629);
                    
                    res_17631 = res_17632;
                }
                x_17626 = res_17630;
                x_17627 = res_17631;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)] =
                    x_17626;
                ((__local
                  int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)] =
                    x_17627;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_18869, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)] =
                x_17628;
            ((__local
              int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)] =
                x_17629;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_18340, res_17607)) {
            ((__global bool *) mem_18516)[gtid_18340] = ((__local
                                                          bool *) scan_arr_mem_18873)[sext_i32_i64(local_tid_18869)];
            ((__global int64_t *) mem_18518)[gtid_18340] = ((__local
                                                             int64_t *) scan_arr_mem_18875)[sext_i32_i64(local_tid_18869)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18336
}
__kernel void mainziscan_stage2_18349(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_18942_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_18940_backing_aligned_1,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18521, __global
                                      unsigned char *mem_18523,
                                      int64_t stage1_num_groups_18902,
                                      int32_t num_threads_18903)
{
    #define segscan_group_sizze_18344 (mainzisegscan_group_sizze_18343)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_18942_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_18942_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_18940_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_18940_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18935;
    int32_t local_tid_18936;
    int64_t group_sizze_18939;
    int32_t wave_sizze_18938;
    int32_t group_tid_18937;
    
    global_tid_18935 = get_global_id(0);
    local_tid_18936 = get_local_id(0);
    group_sizze_18939 = get_local_size(0);
    wave_sizze_18938 = LOCKSTEP_WIDTH;
    group_tid_18937 = get_group_id(0);
    
    int32_t phys_tid_18349;
    
    phys_tid_18349 = global_tid_18935;
    
    __local char *scan_arr_mem_18940;
    __local char *scan_arr_mem_18942;
    
    scan_arr_mem_18940 = (__local char *) scan_arr_mem_18940_backing_0;
    scan_arr_mem_18942 = (__local char *) scan_arr_mem_18942_backing_1;
    
    int64_t flat_idx_18944;
    
    flat_idx_18944 = (sext_i32_i64(local_tid_18936) + 1) *
        (segscan_group_sizze_18344 * sdiv_up64(res_17607,
                                               sext_i32_i64(num_threads_18903))) -
        1;
    
    int64_t gtid_18348;
    
    gtid_18348 = flat_idx_18944;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_18348, res_17607)) {
            ((__local
              bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)] =
                ((__global bool *) mem_18521)[gtid_18348];
            ((__local
              int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)] =
                ((__global int64_t *) mem_18523)[gtid_18348];
        } else {
            ((__local
              bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)] = 0;
            ((__local
              int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_17665;
    int64_t x_17666;
    bool x_17667;
    int64_t x_17668;
    bool x_18945;
    int64_t x_18946;
    bool x_18947;
    int64_t x_18948;
    bool ltid_in_bounds_18952;
    
    ltid_in_bounds_18952 = slt64(sext_i32_i64(local_tid_18936),
                                 stage1_num_groups_18902);
    
    int32_t skip_threads_18953;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_18952) {
            x_17667 = ((volatile __local
                        bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)];
            x_17668 = ((volatile __local
                        int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)];
            if ((local_tid_18936 - squot32(local_tid_18936, 32) * 32) == 0) {
                x_17665 = x_17667;
                x_17666 = x_17668;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_18953 = 1;
        while (slt32(skip_threads_18953, 32)) {
            if (sle32(skip_threads_18953, local_tid_18936 -
                      squot32(local_tid_18936, 32) * 32) &&
                ltid_in_bounds_18952) {
                // read operands
                {
                    x_17665 = ((volatile __local
                                bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936) -
                                                            sext_i32_i64(skip_threads_18953)];
                    x_17666 = ((volatile __local
                                int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936) -
                                                               sext_i32_i64(skip_threads_18953)];
                }
                // perform operation
                {
                    bool res_17669 = x_17665 || x_17667;
                    int64_t res_17670;
                    
                    if (x_17667) {
                        res_17670 = x_17668;
                    } else {
                        int64_t res_17671 = add64(x_17666, x_17668);
                        
                        res_17670 = res_17671;
                    }
                    x_17665 = res_17669;
                    x_17666 = res_17670;
                }
            }
            if (sle32(wave_sizze_18938, skip_threads_18953)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_18953, local_tid_18936 -
                      squot32(local_tid_18936, 32) * 32) &&
                ltid_in_bounds_18952) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)] =
                        x_17665;
                    x_17667 = x_17665;
                    ((volatile __local
                      int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)] =
                        x_17666;
                    x_17668 = x_17666;
                }
            }
            if (sle32(wave_sizze_18938, skip_threads_18953)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_18953 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_18936 - squot32(local_tid_18936, 32) * 32) == 31 &&
            ltid_in_bounds_18952) {
            ((volatile __local
              bool *) scan_arr_mem_18940)[sext_i32_i64(squot32(local_tid_18936,
                                                               32))] = x_17665;
            ((volatile __local
              int64_t *) scan_arr_mem_18942)[sext_i32_i64(squot32(local_tid_18936,
                                                                  32))] =
                x_17666;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_18954;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_18936, 32) == 0 && ltid_in_bounds_18952) {
                x_18947 = ((volatile __local
                            bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)];
                x_18948 = ((volatile __local
                            int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)];
                if ((local_tid_18936 - squot32(local_tid_18936, 32) * 32) ==
                    0) {
                    x_18945 = x_18947;
                    x_18946 = x_18948;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_18954 = 1;
            while (slt32(skip_threads_18954, 32)) {
                if (sle32(skip_threads_18954, local_tid_18936 -
                          squot32(local_tid_18936, 32) * 32) &&
                    (squot32(local_tid_18936, 32) == 0 &&
                     ltid_in_bounds_18952)) {
                    // read operands
                    {
                        x_18945 = ((volatile __local
                                    bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936) -
                                                                sext_i32_i64(skip_threads_18954)];
                        x_18946 = ((volatile __local
                                    int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936) -
                                                                   sext_i32_i64(skip_threads_18954)];
                    }
                    // perform operation
                    {
                        bool res_18949 = x_18945 || x_18947;
                        int64_t res_18950;
                        
                        if (x_18947) {
                            res_18950 = x_18948;
                        } else {
                            int64_t res_18951 = add64(x_18946, x_18948);
                            
                            res_18950 = res_18951;
                        }
                        x_18945 = res_18949;
                        x_18946 = res_18950;
                    }
                }
                if (sle32(wave_sizze_18938, skip_threads_18954)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_18954, local_tid_18936 -
                          squot32(local_tid_18936, 32) * 32) &&
                    (squot32(local_tid_18936, 32) == 0 &&
                     ltid_in_bounds_18952)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)] =
                            x_18945;
                        x_18947 = x_18945;
                        ((volatile __local
                          int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)] =
                            x_18946;
                        x_18948 = x_18946;
                    }
                }
                if (sle32(wave_sizze_18938, skip_threads_18954)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_18954 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_18936, 32) == 0 || !ltid_in_bounds_18952)) {
            // read operands
            {
                x_17667 = x_17665;
                x_17668 = x_17666;
                x_17665 = ((__local
                            bool *) scan_arr_mem_18940)[sext_i32_i64(squot32(local_tid_18936,
                                                                             32)) -
                                                        1];
                x_17666 = ((__local
                            int64_t *) scan_arr_mem_18942)[sext_i32_i64(squot32(local_tid_18936,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                bool res_17669 = x_17665 || x_17667;
                int64_t res_17670;
                
                if (x_17667) {
                    res_17670 = x_17668;
                } else {
                    int64_t res_17671 = add64(x_17666, x_17668);
                    
                    res_17670 = res_17671;
                }
                x_17665 = res_17669;
                x_17666 = res_17670;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)] =
                    x_17665;
                ((__local
                  int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)] =
                    x_17666;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_18936, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)] =
                x_17667;
            ((__local
              int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)] =
                x_17668;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_18348, res_17607)) {
            ((__global bool *) mem_18521)[gtid_18348] = ((__local
                                                          bool *) scan_arr_mem_18940)[sext_i32_i64(local_tid_18936)];
            ((__global int64_t *) mem_18523)[gtid_18348] = ((__local
                                                             int64_t *) scan_arr_mem_18942)[sext_i32_i64(local_tid_18936)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18344
}
__kernel void mainziscan_stage2_18357(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_19009_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_19007_backing_aligned_1,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18528, __global
                                      unsigned char *mem_18530,
                                      int64_t stage1_num_groups_18969,
                                      int32_t num_threads_18970)
{
    #define segscan_group_sizze_18352 (mainzisegscan_group_sizze_18351)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_19009_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_19009_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_19007_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_19007_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19002;
    int32_t local_tid_19003;
    int64_t group_sizze_19006;
    int32_t wave_sizze_19005;
    int32_t group_tid_19004;
    
    global_tid_19002 = get_global_id(0);
    local_tid_19003 = get_local_id(0);
    group_sizze_19006 = get_local_size(0);
    wave_sizze_19005 = LOCKSTEP_WIDTH;
    group_tid_19004 = get_group_id(0);
    
    int32_t phys_tid_18357;
    
    phys_tid_18357 = global_tid_19002;
    
    __local char *scan_arr_mem_19007;
    __local char *scan_arr_mem_19009;
    
    scan_arr_mem_19007 = (__local char *) scan_arr_mem_19007_backing_0;
    scan_arr_mem_19009 = (__local char *) scan_arr_mem_19009_backing_1;
    
    int64_t flat_idx_19011;
    
    flat_idx_19011 = (sext_i32_i64(local_tid_19003) + 1) *
        (segscan_group_sizze_18352 * sdiv_up64(res_17607,
                                               sext_i32_i64(num_threads_18970))) -
        1;
    
    int64_t gtid_18356;
    
    gtid_18356 = flat_idx_19011;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_18356, res_17607)) {
            ((__local
              bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)] =
                ((__global bool *) mem_18528)[gtid_18356];
            ((__local
              float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)] =
                ((__global float *) mem_18530)[gtid_18356];
        } else {
            ((__local
              bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)] = 0;
            ((__local
              float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)] =
                0.0F;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_17691;
    float x_17692;
    bool x_17693;
    float x_17694;
    bool x_19012;
    float x_19013;
    bool x_19014;
    float x_19015;
    bool ltid_in_bounds_19019;
    
    ltid_in_bounds_19019 = slt64(sext_i32_i64(local_tid_19003),
                                 stage1_num_groups_18969);
    
    int32_t skip_threads_19020;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_19019) {
            x_17693 = ((volatile __local
                        bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)];
            x_17694 = ((volatile __local
                        float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)];
            if ((local_tid_19003 - squot32(local_tid_19003, 32) * 32) == 0) {
                x_17691 = x_17693;
                x_17692 = x_17694;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_19020 = 1;
        while (slt32(skip_threads_19020, 32)) {
            if (sle32(skip_threads_19020, local_tid_19003 -
                      squot32(local_tid_19003, 32) * 32) &&
                ltid_in_bounds_19019) {
                // read operands
                {
                    x_17691 = ((volatile __local
                                bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003) -
                                                            sext_i32_i64(skip_threads_19020)];
                    x_17692 = ((volatile __local
                                float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003) -
                                                             sext_i32_i64(skip_threads_19020)];
                }
                // perform operation
                {
                    bool res_17695 = x_17691 || x_17693;
                    float res_17696;
                    
                    if (x_17693) {
                        res_17696 = x_17694;
                    } else {
                        float res_17697 = x_17692 + x_17694;
                        
                        res_17696 = res_17697;
                    }
                    x_17691 = res_17695;
                    x_17692 = res_17696;
                }
            }
            if (sle32(wave_sizze_19005, skip_threads_19020)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_19020, local_tid_19003 -
                      squot32(local_tid_19003, 32) * 32) &&
                ltid_in_bounds_19019) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)] =
                        x_17691;
                    x_17693 = x_17691;
                    ((volatile __local
                      float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)] =
                        x_17692;
                    x_17694 = x_17692;
                }
            }
            if (sle32(wave_sizze_19005, skip_threads_19020)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_19020 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_19003 - squot32(local_tid_19003, 32) * 32) == 31 &&
            ltid_in_bounds_19019) {
            ((volatile __local
              bool *) scan_arr_mem_19007)[sext_i32_i64(squot32(local_tid_19003,
                                                               32))] = x_17691;
            ((volatile __local
              float *) scan_arr_mem_19009)[sext_i32_i64(squot32(local_tid_19003,
                                                                32))] = x_17692;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_19021;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_19003, 32) == 0 && ltid_in_bounds_19019) {
                x_19014 = ((volatile __local
                            bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)];
                x_19015 = ((volatile __local
                            float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)];
                if ((local_tid_19003 - squot32(local_tid_19003, 32) * 32) ==
                    0) {
                    x_19012 = x_19014;
                    x_19013 = x_19015;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_19021 = 1;
            while (slt32(skip_threads_19021, 32)) {
                if (sle32(skip_threads_19021, local_tid_19003 -
                          squot32(local_tid_19003, 32) * 32) &&
                    (squot32(local_tid_19003, 32) == 0 &&
                     ltid_in_bounds_19019)) {
                    // read operands
                    {
                        x_19012 = ((volatile __local
                                    bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003) -
                                                                sext_i32_i64(skip_threads_19021)];
                        x_19013 = ((volatile __local
                                    float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003) -
                                                                 sext_i32_i64(skip_threads_19021)];
                    }
                    // perform operation
                    {
                        bool res_19016 = x_19012 || x_19014;
                        float res_19017;
                        
                        if (x_19014) {
                            res_19017 = x_19015;
                        } else {
                            float res_19018 = x_19013 + x_19015;
                            
                            res_19017 = res_19018;
                        }
                        x_19012 = res_19016;
                        x_19013 = res_19017;
                    }
                }
                if (sle32(wave_sizze_19005, skip_threads_19021)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_19021, local_tid_19003 -
                          squot32(local_tid_19003, 32) * 32) &&
                    (squot32(local_tid_19003, 32) == 0 &&
                     ltid_in_bounds_19019)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)] =
                            x_19012;
                        x_19014 = x_19012;
                        ((volatile __local
                          float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)] =
                            x_19013;
                        x_19015 = x_19013;
                    }
                }
                if (sle32(wave_sizze_19005, skip_threads_19021)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_19021 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_19003, 32) == 0 || !ltid_in_bounds_19019)) {
            // read operands
            {
                x_17693 = x_17691;
                x_17694 = x_17692;
                x_17691 = ((__local
                            bool *) scan_arr_mem_19007)[sext_i32_i64(squot32(local_tid_19003,
                                                                             32)) -
                                                        1];
                x_17692 = ((__local
                            float *) scan_arr_mem_19009)[sext_i32_i64(squot32(local_tid_19003,
                                                                              32)) -
                                                         1];
            }
            // perform operation
            {
                bool res_17695 = x_17691 || x_17693;
                float res_17696;
                
                if (x_17693) {
                    res_17696 = x_17694;
                } else {
                    float res_17697 = x_17692 + x_17694;
                    
                    res_17696 = res_17697;
                }
                x_17691 = res_17695;
                x_17692 = res_17696;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)] =
                    x_17691;
                ((__local
                  float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)] =
                    x_17692;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_19003, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)] =
                x_17693;
            ((__local
              float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)] =
                x_17694;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_18356, res_17607)) {
            ((__global bool *) mem_18528)[gtid_18356] = ((__local
                                                          bool *) scan_arr_mem_19007)[sext_i32_i64(local_tid_19003)];
            ((__global float *) mem_18530)[gtid_18356] = ((__local
                                                           float *) scan_arr_mem_19009)[sext_i32_i64(local_tid_19003)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18352
}
__kernel void mainziscan_stage2_18409(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_19064_backing_aligned_0,
                                      int64_t res_17607, __global
                                      unsigned char *mem_18533,
                                      int64_t stage1_num_groups_19036,
                                      int32_t num_threads_19037)
{
    #define segscan_group_sizze_18404 (mainzisegscan_group_sizze_18403)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_19064_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_19064_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19059;
    int32_t local_tid_19060;
    int64_t group_sizze_19063;
    int32_t wave_sizze_19062;
    int32_t group_tid_19061;
    
    global_tid_19059 = get_global_id(0);
    local_tid_19060 = get_local_id(0);
    group_sizze_19063 = get_local_size(0);
    wave_sizze_19062 = LOCKSTEP_WIDTH;
    group_tid_19061 = get_group_id(0);
    
    int32_t phys_tid_18409;
    
    phys_tid_18409 = global_tid_19059;
    
    __local char *scan_arr_mem_19064;
    
    scan_arr_mem_19064 = (__local char *) scan_arr_mem_19064_backing_0;
    
    int64_t flat_idx_19066;
    
    flat_idx_19066 = (sext_i32_i64(local_tid_19060) + 1) *
        (segscan_group_sizze_18404 * sdiv_up64(res_17607,
                                               sext_i32_i64(num_threads_19037))) -
        1;
    
    int64_t gtid_18408;
    
    gtid_18408 = flat_idx_19066;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_18408, res_17607)) {
            ((__local
              int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)] =
                ((__global int64_t *) mem_18533)[gtid_18408];
        } else {
            ((__local
              int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int64_t x_17793;
    int64_t x_17794;
    int64_t x_19067;
    int64_t x_19068;
    bool ltid_in_bounds_19070;
    
    ltid_in_bounds_19070 = slt64(sext_i32_i64(local_tid_19060),
                                 stage1_num_groups_19036);
    
    int32_t skip_threads_19071;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_19070) {
            x_17794 = ((volatile __local
                        int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)];
            if ((local_tid_19060 - squot32(local_tid_19060, 32) * 32) == 0) {
                x_17793 = x_17794;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_19071 = 1;
        while (slt32(skip_threads_19071, 32)) {
            if (sle32(skip_threads_19071, local_tid_19060 -
                      squot32(local_tid_19060, 32) * 32) &&
                ltid_in_bounds_19070) {
                // read operands
                {
                    x_17793 = ((volatile __local
                                int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060) -
                                                               sext_i32_i64(skip_threads_19071)];
                }
                // perform operation
                {
                    int64_t res_17795 = add64(x_17793, x_17794);
                    
                    x_17793 = res_17795;
                }
            }
            if (sle32(wave_sizze_19062, skip_threads_19071)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_19071, local_tid_19060 -
                      squot32(local_tid_19060, 32) * 32) &&
                ltid_in_bounds_19070) {
                // write result
                {
                    ((volatile __local
                      int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)] =
                        x_17793;
                    x_17794 = x_17793;
                }
            }
            if (sle32(wave_sizze_19062, skip_threads_19071)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_19071 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_19060 - squot32(local_tid_19060, 32) * 32) == 31 &&
            ltid_in_bounds_19070) {
            ((volatile __local
              int64_t *) scan_arr_mem_19064)[sext_i32_i64(squot32(local_tid_19060,
                                                                  32))] =
                x_17793;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_19072;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_19060, 32) == 0 && ltid_in_bounds_19070) {
                x_19068 = ((volatile __local
                            int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)];
                if ((local_tid_19060 - squot32(local_tid_19060, 32) * 32) ==
                    0) {
                    x_19067 = x_19068;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_19072 = 1;
            while (slt32(skip_threads_19072, 32)) {
                if (sle32(skip_threads_19072, local_tid_19060 -
                          squot32(local_tid_19060, 32) * 32) &&
                    (squot32(local_tid_19060, 32) == 0 &&
                     ltid_in_bounds_19070)) {
                    // read operands
                    {
                        x_19067 = ((volatile __local
                                    int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060) -
                                                                   sext_i32_i64(skip_threads_19072)];
                    }
                    // perform operation
                    {
                        int64_t res_19069 = add64(x_19067, x_19068);
                        
                        x_19067 = res_19069;
                    }
                }
                if (sle32(wave_sizze_19062, skip_threads_19072)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_19072, local_tid_19060 -
                          squot32(local_tid_19060, 32) * 32) &&
                    (squot32(local_tid_19060, 32) == 0 &&
                     ltid_in_bounds_19070)) {
                    // write result
                    {
                        ((volatile __local
                          int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)] =
                            x_19067;
                        x_19068 = x_19067;
                    }
                }
                if (sle32(wave_sizze_19062, skip_threads_19072)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_19072 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_19060, 32) == 0 || !ltid_in_bounds_19070)) {
            // read operands
            {
                x_17794 = x_17793;
                x_17793 = ((__local
                            int64_t *) scan_arr_mem_19064)[sext_i32_i64(squot32(local_tid_19060,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                int64_t res_17795 = add64(x_17793, x_17794);
                
                x_17793 = res_17795;
            }
            // write final result
            {
                ((__local
                  int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)] =
                    x_17793;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_19060, 32) == 0) {
            ((__local
              int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)] =
                x_17794;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_18408, res_17607)) {
            ((__global int64_t *) mem_18533)[gtid_18408] = ((__local
                                                             int64_t *) scan_arr_mem_19064)[sext_i32_i64(local_tid_19060)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18404
}
__kernel void mainziscan_stage3_18308(__global int *global_failure,
                                      int64_t paths_17456,
                                      int64_t num_groups_18305, __global
                                      unsigned char *mem_18505,
                                      int32_t num_threads_18599,
                                      int32_t required_groups_18635)
{
    #define segscan_group_sizze_18303 (mainzisegscan_group_sizze_18302)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18636;
    int32_t local_tid_18637;
    int64_t group_sizze_18640;
    int32_t wave_sizze_18639;
    int32_t group_tid_18638;
    
    global_tid_18636 = get_global_id(0);
    local_tid_18637 = get_local_id(0);
    group_sizze_18640 = get_local_size(0);
    wave_sizze_18639 = LOCKSTEP_WIDTH;
    group_tid_18638 = get_group_id(0);
    
    int32_t phys_tid_18308;
    
    phys_tid_18308 = global_tid_18636;
    
    int32_t phys_group_id_18641;
    
    phys_group_id_18641 = get_group_id(0);
    for (int32_t i_18642 = 0; i_18642 < sdiv_up32(required_groups_18635 -
                                                  phys_group_id_18641,
                                                  sext_i64_i32(num_groups_18305));
         i_18642++) {
        int32_t virt_group_id_18643 = phys_group_id_18641 + i_18642 *
                sext_i64_i32(num_groups_18305);
        int64_t flat_idx_18644 = sext_i32_i64(virt_group_id_18643) *
                segscan_group_sizze_18303 + sext_i32_i64(local_tid_18637);
        int64_t gtid_18307 = flat_idx_18644;
        int64_t orig_group_18645 = squot64(flat_idx_18644,
                                           segscan_group_sizze_18303 *
                                           sdiv_up64(paths_17456,
                                                     sext_i32_i64(num_threads_18599)));
        int64_t carry_in_flat_idx_18646 = orig_group_18645 *
                (segscan_group_sizze_18303 * sdiv_up64(paths_17456,
                                                       sext_i32_i64(num_threads_18599))) -
                1;
        
        if (slt64(gtid_18307, paths_17456)) {
            if (!(orig_group_18645 == 0 || flat_idx_18644 == (orig_group_18645 +
                                                              1) *
                  (segscan_group_sizze_18303 * sdiv_up64(paths_17456,
                                                         sext_i32_i64(num_threads_18599))) -
                  1)) {
                int64_t x_17593;
                int64_t x_17594;
                
                x_17593 = ((__global
                            int64_t *) mem_18505)[carry_in_flat_idx_18646];
                x_17594 = ((__global int64_t *) mem_18505)[gtid_18307];
                
                int64_t res_17595;
                
                res_17595 = add64(x_17593, x_17594);
                x_17593 = res_17595;
                ((__global int64_t *) mem_18505)[gtid_18307] = x_17593;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18303
}
__kernel void mainziscan_stage3_18341(__global int *global_failure,
                                      int64_t res_17607,
                                      int64_t num_groups_18338, __global
                                      unsigned char *mem_18516, __global
                                      unsigned char *mem_18518,
                                      int32_t num_threads_18836,
                                      int32_t required_groups_18888)
{
    #define segscan_group_sizze_18336 (mainzisegscan_group_sizze_18335)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18889;
    int32_t local_tid_18890;
    int64_t group_sizze_18893;
    int32_t wave_sizze_18892;
    int32_t group_tid_18891;
    
    global_tid_18889 = get_global_id(0);
    local_tid_18890 = get_local_id(0);
    group_sizze_18893 = get_local_size(0);
    wave_sizze_18892 = LOCKSTEP_WIDTH;
    group_tid_18891 = get_group_id(0);
    
    int32_t phys_tid_18341;
    
    phys_tid_18341 = global_tid_18889;
    
    int32_t phys_group_id_18894;
    
    phys_group_id_18894 = get_group_id(0);
    for (int32_t i_18895 = 0; i_18895 < sdiv_up32(required_groups_18888 -
                                                  phys_group_id_18894,
                                                  sext_i64_i32(num_groups_18338));
         i_18895++) {
        int32_t virt_group_id_18896 = phys_group_id_18894 + i_18895 *
                sext_i64_i32(num_groups_18338);
        int64_t flat_idx_18897 = sext_i32_i64(virt_group_id_18896) *
                segscan_group_sizze_18336 + sext_i32_i64(local_tid_18890);
        int64_t gtid_18340 = flat_idx_18897;
        int64_t orig_group_18898 = squot64(flat_idx_18897,
                                           segscan_group_sizze_18336 *
                                           sdiv_up64(res_17607,
                                                     sext_i32_i64(num_threads_18836)));
        int64_t carry_in_flat_idx_18899 = orig_group_18898 *
                (segscan_group_sizze_18336 * sdiv_up64(res_17607,
                                                       sext_i32_i64(num_threads_18836))) -
                1;
        
        if (slt64(gtid_18340, res_17607)) {
            if (!(orig_group_18898 == 0 || flat_idx_18897 == (orig_group_18898 +
                                                              1) *
                  (segscan_group_sizze_18336 * sdiv_up64(res_17607,
                                                         sext_i32_i64(num_threads_18836))) -
                  1)) {
                bool x_17626;
                int64_t x_17627;
                bool x_17628;
                int64_t x_17629;
                
                x_17626 = ((__global
                            bool *) mem_18516)[carry_in_flat_idx_18899];
                x_17627 = ((__global
                            int64_t *) mem_18518)[carry_in_flat_idx_18899];
                x_17628 = ((__global bool *) mem_18516)[gtid_18340];
                x_17629 = ((__global int64_t *) mem_18518)[gtid_18340];
                
                bool res_17630;
                
                res_17630 = x_17626 || x_17628;
                
                int64_t res_17631;
                
                if (x_17628) {
                    res_17631 = x_17629;
                } else {
                    int64_t res_17632 = add64(x_17627, x_17629);
                    
                    res_17631 = res_17632;
                }
                x_17626 = res_17630;
                x_17627 = res_17631;
                ((__global bool *) mem_18516)[gtid_18340] = x_17626;
                ((__global int64_t *) mem_18518)[gtid_18340] = x_17627;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18336
}
__kernel void mainziscan_stage3_18349(__global int *global_failure,
                                      int64_t res_17607,
                                      int64_t num_groups_18346, __global
                                      unsigned char *mem_18521, __global
                                      unsigned char *mem_18523,
                                      int32_t num_threads_18903,
                                      int32_t required_groups_18955)
{
    #define segscan_group_sizze_18344 (mainzisegscan_group_sizze_18343)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18956;
    int32_t local_tid_18957;
    int64_t group_sizze_18960;
    int32_t wave_sizze_18959;
    int32_t group_tid_18958;
    
    global_tid_18956 = get_global_id(0);
    local_tid_18957 = get_local_id(0);
    group_sizze_18960 = get_local_size(0);
    wave_sizze_18959 = LOCKSTEP_WIDTH;
    group_tid_18958 = get_group_id(0);
    
    int32_t phys_tid_18349;
    
    phys_tid_18349 = global_tid_18956;
    
    int32_t phys_group_id_18961;
    
    phys_group_id_18961 = get_group_id(0);
    for (int32_t i_18962 = 0; i_18962 < sdiv_up32(required_groups_18955 -
                                                  phys_group_id_18961,
                                                  sext_i64_i32(num_groups_18346));
         i_18962++) {
        int32_t virt_group_id_18963 = phys_group_id_18961 + i_18962 *
                sext_i64_i32(num_groups_18346);
        int64_t flat_idx_18964 = sext_i32_i64(virt_group_id_18963) *
                segscan_group_sizze_18344 + sext_i32_i64(local_tid_18957);
        int64_t gtid_18348 = flat_idx_18964;
        int64_t orig_group_18965 = squot64(flat_idx_18964,
                                           segscan_group_sizze_18344 *
                                           sdiv_up64(res_17607,
                                                     sext_i32_i64(num_threads_18903)));
        int64_t carry_in_flat_idx_18966 = orig_group_18965 *
                (segscan_group_sizze_18344 * sdiv_up64(res_17607,
                                                       sext_i32_i64(num_threads_18903))) -
                1;
        
        if (slt64(gtid_18348, res_17607)) {
            if (!(orig_group_18965 == 0 || flat_idx_18964 == (orig_group_18965 +
                                                              1) *
                  (segscan_group_sizze_18344 * sdiv_up64(res_17607,
                                                         sext_i32_i64(num_threads_18903))) -
                  1)) {
                bool x_17665;
                int64_t x_17666;
                bool x_17667;
                int64_t x_17668;
                
                x_17665 = ((__global
                            bool *) mem_18521)[carry_in_flat_idx_18966];
                x_17666 = ((__global
                            int64_t *) mem_18523)[carry_in_flat_idx_18966];
                x_17667 = ((__global bool *) mem_18521)[gtid_18348];
                x_17668 = ((__global int64_t *) mem_18523)[gtid_18348];
                
                bool res_17669;
                
                res_17669 = x_17665 || x_17667;
                
                int64_t res_17670;
                
                if (x_17667) {
                    res_17670 = x_17668;
                } else {
                    int64_t res_17671 = add64(x_17666, x_17668);
                    
                    res_17670 = res_17671;
                }
                x_17665 = res_17669;
                x_17666 = res_17670;
                ((__global bool *) mem_18521)[gtid_18348] = x_17665;
                ((__global int64_t *) mem_18523)[gtid_18348] = x_17666;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18344
}
__kernel void mainziscan_stage3_18357(__global int *global_failure,
                                      int64_t res_17607,
                                      int64_t num_groups_18354, __global
                                      unsigned char *mem_18528, __global
                                      unsigned char *mem_18530,
                                      int32_t num_threads_18970,
                                      int32_t required_groups_19022)
{
    #define segscan_group_sizze_18352 (mainzisegscan_group_sizze_18351)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19023;
    int32_t local_tid_19024;
    int64_t group_sizze_19027;
    int32_t wave_sizze_19026;
    int32_t group_tid_19025;
    
    global_tid_19023 = get_global_id(0);
    local_tid_19024 = get_local_id(0);
    group_sizze_19027 = get_local_size(0);
    wave_sizze_19026 = LOCKSTEP_WIDTH;
    group_tid_19025 = get_group_id(0);
    
    int32_t phys_tid_18357;
    
    phys_tid_18357 = global_tid_19023;
    
    int32_t phys_group_id_19028;
    
    phys_group_id_19028 = get_group_id(0);
    for (int32_t i_19029 = 0; i_19029 < sdiv_up32(required_groups_19022 -
                                                  phys_group_id_19028,
                                                  sext_i64_i32(num_groups_18354));
         i_19029++) {
        int32_t virt_group_id_19030 = phys_group_id_19028 + i_19029 *
                sext_i64_i32(num_groups_18354);
        int64_t flat_idx_19031 = sext_i32_i64(virt_group_id_19030) *
                segscan_group_sizze_18352 + sext_i32_i64(local_tid_19024);
        int64_t gtid_18356 = flat_idx_19031;
        int64_t orig_group_19032 = squot64(flat_idx_19031,
                                           segscan_group_sizze_18352 *
                                           sdiv_up64(res_17607,
                                                     sext_i32_i64(num_threads_18970)));
        int64_t carry_in_flat_idx_19033 = orig_group_19032 *
                (segscan_group_sizze_18352 * sdiv_up64(res_17607,
                                                       sext_i32_i64(num_threads_18970))) -
                1;
        
        if (slt64(gtid_18356, res_17607)) {
            if (!(orig_group_19032 == 0 || flat_idx_19031 == (orig_group_19032 +
                                                              1) *
                  (segscan_group_sizze_18352 * sdiv_up64(res_17607,
                                                         sext_i32_i64(num_threads_18970))) -
                  1)) {
                bool x_17691;
                float x_17692;
                bool x_17693;
                float x_17694;
                
                x_17691 = ((__global
                            bool *) mem_18528)[carry_in_flat_idx_19033];
                x_17692 = ((__global
                            float *) mem_18530)[carry_in_flat_idx_19033];
                x_17693 = ((__global bool *) mem_18528)[gtid_18356];
                x_17694 = ((__global float *) mem_18530)[gtid_18356];
                
                bool res_17695;
                
                res_17695 = x_17691 || x_17693;
                
                float res_17696;
                
                if (x_17693) {
                    res_17696 = x_17694;
                } else {
                    float res_17697 = x_17692 + x_17694;
                    
                    res_17696 = res_17697;
                }
                x_17691 = res_17695;
                x_17692 = res_17696;
                ((__global bool *) mem_18528)[gtid_18356] = x_17691;
                ((__global float *) mem_18530)[gtid_18356] = x_17692;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18352
}
__kernel void mainziscan_stage3_18409(__global int *global_failure,
                                      int64_t res_17607,
                                      int64_t num_groups_18406, __global
                                      unsigned char *mem_18533,
                                      int32_t num_threads_19037,
                                      int32_t required_groups_19073)
{
    #define segscan_group_sizze_18404 (mainzisegscan_group_sizze_18403)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19074;
    int32_t local_tid_19075;
    int64_t group_sizze_19078;
    int32_t wave_sizze_19077;
    int32_t group_tid_19076;
    
    global_tid_19074 = get_global_id(0);
    local_tid_19075 = get_local_id(0);
    group_sizze_19078 = get_local_size(0);
    wave_sizze_19077 = LOCKSTEP_WIDTH;
    group_tid_19076 = get_group_id(0);
    
    int32_t phys_tid_18409;
    
    phys_tid_18409 = global_tid_19074;
    
    int32_t phys_group_id_19079;
    
    phys_group_id_19079 = get_group_id(0);
    for (int32_t i_19080 = 0; i_19080 < sdiv_up32(required_groups_19073 -
                                                  phys_group_id_19079,
                                                  sext_i64_i32(num_groups_18406));
         i_19080++) {
        int32_t virt_group_id_19081 = phys_group_id_19079 + i_19080 *
                sext_i64_i32(num_groups_18406);
        int64_t flat_idx_19082 = sext_i32_i64(virt_group_id_19081) *
                segscan_group_sizze_18404 + sext_i32_i64(local_tid_19075);
        int64_t gtid_18408 = flat_idx_19082;
        int64_t orig_group_19083 = squot64(flat_idx_19082,
                                           segscan_group_sizze_18404 *
                                           sdiv_up64(res_17607,
                                                     sext_i32_i64(num_threads_19037)));
        int64_t carry_in_flat_idx_19084 = orig_group_19083 *
                (segscan_group_sizze_18404 * sdiv_up64(res_17607,
                                                       sext_i32_i64(num_threads_19037))) -
                1;
        
        if (slt64(gtid_18408, res_17607)) {
            if (!(orig_group_19083 == 0 || flat_idx_19082 == (orig_group_19083 +
                                                              1) *
                  (segscan_group_sizze_18404 * sdiv_up64(res_17607,
                                                         sext_i32_i64(num_threads_19037))) -
                  1)) {
                int64_t x_17793;
                int64_t x_17794;
                
                x_17793 = ((__global
                            int64_t *) mem_18533)[carry_in_flat_idx_19084];
                x_17794 = ((__global int64_t *) mem_18533)[gtid_18408];
                
                int64_t res_17795;
                
                res_17795 = add64(x_17793, x_17794);
                x_17793 = res_17795;
                ((__global int64_t *) mem_18533)[gtid_18408] = x_17793;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_18404
}
__kernel void mainziseghist_global_18326(__global int *global_failure,
                                         int64_t paths_17456, int64_t res_17607,
                                         int64_t num_groups_18323, __global
                                         unsigned char *mem_18505,
                                         int32_t num_subhistos_18686, __global
                                         unsigned char *res_subhistos_mem_18687,
                                         __global
                                         unsigned char *mainzihist_locks_mem_18757,
                                         int32_t chk_i_18759,
                                         int64_t hist_H_chk_18760)
{
    #define seghist_group_sizze_18321 (mainziseghist_group_sizze_18320)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18761;
    int32_t local_tid_18762;
    int64_t group_sizze_18765;
    int32_t wave_sizze_18764;
    int32_t group_tid_18763;
    
    global_tid_18761 = get_global_id(0);
    local_tid_18762 = get_local_id(0);
    group_sizze_18765 = get_local_size(0);
    wave_sizze_18764 = LOCKSTEP_WIDTH;
    group_tid_18763 = get_group_id(0);
    
    int32_t phys_tid_18326;
    
    phys_tid_18326 = global_tid_18761;
    
    int32_t subhisto_ind_18766;
    
    subhisto_ind_18766 = squot32(global_tid_18761,
                                 sdiv_up32(sext_i64_i32(seghist_group_sizze_18321 *
                                           num_groups_18323),
                                           num_subhistos_18686));
    for (int64_t i_18767 = 0; i_18767 < sdiv_up64(paths_17456 -
                                                  sext_i32_i64(global_tid_18761),
                                                  sext_i32_i64(sext_i64_i32(seghist_group_sizze_18321 *
                                                  num_groups_18323)));
         i_18767++) {
        int32_t gtid_18325 = sext_i64_i32(i_18767 *
                sext_i32_i64(sext_i64_i32(seghist_group_sizze_18321 *
                num_groups_18323)) + sext_i32_i64(global_tid_18761));
        
        if (slt64(i_18767 *
                  sext_i32_i64(sext_i64_i32(seghist_group_sizze_18321 *
                  num_groups_18323)) + sext_i32_i64(global_tid_18761),
                  paths_17456)) {
            int64_t i_p_o_18437 = add64(-1, gtid_18325);
            int64_t rot_i_18438 = smod64(i_p_o_18437, paths_17456);
            bool cond_18332 = gtid_18325 == 0;
            int64_t res_18333;
            
            if (cond_18332) {
                res_18333 = 0;
            } else {
                int64_t x_18331 = ((__global int64_t *) mem_18505)[rot_i_18438];
                
                res_18333 = x_18331;
            }
            // save map-out results
            { }
            // perform atomic updates
            {
                if (sle64(sext_i32_i64(chk_i_18759) * hist_H_chk_18760,
                          res_18333) && (slt64(res_18333,
                                               sext_i32_i64(chk_i_18759) *
                                               hist_H_chk_18760 +
                                               hist_H_chk_18760) &&
                                         slt64(res_18333, res_17607))) {
                    int64_t x_18327;
                    int64_t x_18328;
                    
                    x_18328 = gtid_18325;
                    
                    int32_t old_18768;
                    volatile bool continue_18769;
                    
                    continue_18769 = 1;
                    while (continue_18769) {
                        old_18768 =
                            atomic_cmpxchg_i32_global(&((volatile __global
                                                         int *) mainzihist_locks_mem_18757)[srem64(sext_i32_i64(subhisto_ind_18766) *
                                                                                                   res_17607 +
                                                                                                   res_18333,
                                                                                                   100151)],
                                                      0, 1);
                        if (old_18768 == 0) {
                            int64_t x_18327;
                            
                            // bind lhs
                            {
                                x_18327 = ((volatile __global
                                            int64_t *) res_subhistos_mem_18687)[sext_i32_i64(subhisto_ind_18766) *
                                                                                res_17607 +
                                                                                res_18333];
                            }
                            // execute operation
                            {
                                int64_t res_18329 = smax64(x_18327, x_18328);
                                
                                x_18327 = res_18329;
                            }
                            // update global result
                            {
                                ((volatile __global
                                  int64_t *) res_subhistos_mem_18687)[sext_i32_i64(subhisto_ind_18766) *
                                                                      res_17607 +
                                                                      res_18333] =
                                    x_18327;
                            }
                            mem_fence_global();
                            old_18768 =
                                atomic_cmpxchg_i32_global(&((volatile __global
                                                             int *) mainzihist_locks_mem_18757)[srem64(sext_i32_i64(subhisto_ind_18766) *
                                                                                                       res_17607 +
                                                                                                       res_18333,
                                                                                                       100151)],
                                                          1, 0);
                            continue_18769 = 0;
                        }
                        mem_fence_global();
                    }
                }
            }
        }
    }
    
  error_0:
    return;
    #undef seghist_group_sizze_18321
}
__kernel void mainziseghist_local_18326(__global int *global_failure,
                                        __local volatile
                                        int64_t *locks_mem_18727_backing_aligned_0,
                                        __local volatile
                                        int64_t *subhistogram_local_mem_18725_backing_aligned_1,
                                        int64_t paths_17456, int64_t res_17607,
                                        __global unsigned char *mem_18505,
                                        __global
                                        unsigned char *res_subhistos_mem_18687,
                                        int32_t max_group_sizze_18696,
                                        int64_t num_groups_18697,
                                        int32_t hist_M_18703,
                                        int32_t chk_i_18708,
                                        int64_t num_segments_18709,
                                        int64_t hist_H_chk_18710,
                                        int64_t histo_sizze_18711,
                                        int32_t init_per_thread_18712)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict locks_mem_18727_backing_1 =
                          (__local volatile
                           char *) locks_mem_18727_backing_aligned_0;
    __local volatile char *restrict subhistogram_local_mem_18725_backing_0 =
                          (__local volatile
                           char *) subhistogram_local_mem_18725_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18713;
    int32_t local_tid_18714;
    int64_t group_sizze_18717;
    int32_t wave_sizze_18716;
    int32_t group_tid_18715;
    
    global_tid_18713 = get_global_id(0);
    local_tid_18714 = get_local_id(0);
    group_sizze_18717 = get_local_size(0);
    wave_sizze_18716 = LOCKSTEP_WIDTH;
    group_tid_18715 = get_group_id(0);
    
    int32_t phys_tid_18326;
    
    phys_tid_18326 = global_tid_18713;
    
    int32_t phys_group_id_18718;
    
    phys_group_id_18718 = get_group_id(0);
    for (int32_t i_18719 = 0; i_18719 <
         sdiv_up32(sext_i64_i32(num_groups_18697 * num_segments_18709) -
                   phys_group_id_18718, sext_i64_i32(num_groups_18697));
         i_18719++) {
        int32_t virt_group_id_18720 = phys_group_id_18718 + i_18719 *
                sext_i64_i32(num_groups_18697);
        int32_t flat_segment_id_18721 = squot32(virt_group_id_18720,
                                                sext_i64_i32(num_groups_18697));
        int32_t gid_in_segment_18722 = srem32(virt_group_id_18720,
                                              sext_i64_i32(num_groups_18697));
        int32_t pgtid_in_segment_18723 = gid_in_segment_18722 *
                sext_i64_i32(max_group_sizze_18696) + local_tid_18714;
        int32_t threads_per_segment_18724 = sext_i64_i32(num_groups_18697 *
                max_group_sizze_18696);
        __local char *subhistogram_local_mem_18725;
        
        subhistogram_local_mem_18725 = (__local
                                        char *) subhistogram_local_mem_18725_backing_0;
        
        __local char *locks_mem_18727;
        
        locks_mem_18727 = (__local char *) locks_mem_18727_backing_1;
        // All locks start out unlocked
        {
            for (int64_t i_18729 = 0; i_18729 < sdiv_up64(hist_M_18703 *
                                                          hist_H_chk_18710 -
                                                          sext_i32_i64(local_tid_18714),
                                                          max_group_sizze_18696);
                 i_18729++) {
                ((__local int32_t *) locks_mem_18727)[squot64(i_18729 *
                                                              max_group_sizze_18696 +
                                                              sext_i32_i64(local_tid_18714),
                                                              hist_H_chk_18710) *
                                                      hist_H_chk_18710 +
                                                      (i_18729 *
                                                       max_group_sizze_18696 +
                                                       sext_i32_i64(local_tid_18714) -
                                                       squot64(i_18729 *
                                                               max_group_sizze_18696 +
                                                               sext_i32_i64(local_tid_18714),
                                                               hist_H_chk_18710) *
                                                       hist_H_chk_18710)] = 0;
            }
        }
        
        int32_t thread_local_subhisto_i_18730;
        
        thread_local_subhisto_i_18730 = srem32(local_tid_18714, hist_M_18703);
        // initialize histograms in local memory
        {
            for (int32_t local_i_18731 = 0; local_i_18731 <
                 init_per_thread_18712; local_i_18731++) {
                int32_t j_18732 = local_i_18731 *
                        sext_i64_i32(max_group_sizze_18696) + local_tid_18714;
                int32_t j_offset_18733 = hist_M_18703 *
                        sext_i64_i32(histo_sizze_18711) * gid_in_segment_18722 +
                        j_18732;
                int32_t local_subhisto_i_18734 = squot32(j_18732,
                                                         sext_i64_i32(histo_sizze_18711));
                int32_t global_subhisto_i_18735 = squot32(j_offset_18733,
                                                          sext_i64_i32(histo_sizze_18711));
                
                if (slt32(j_18732, hist_M_18703 *
                          sext_i64_i32(histo_sizze_18711))) {
                    // First subhistogram is initialised from global memory; others with neutral element.
                    {
                        if (global_subhisto_i_18735 == 0) {
                            ((__local
                              int64_t *) subhistogram_local_mem_18725)[sext_i32_i64(local_subhisto_i_18734) *
                                                                       hist_H_chk_18710 +
                                                                       sext_i32_i64(srem32(j_18732,
                                                                                           sext_i64_i32(histo_sizze_18711)))] =
                                ((__global
                                  int64_t *) res_subhistos_mem_18687)[sext_i32_i64(srem32(j_18732,
                                                                                          sext_i64_i32(histo_sizze_18711))) +
                                                                      sext_i32_i64(chk_i_18708) *
                                                                      hist_H_chk_18710];
                        } else {
                            ((__local
                              int64_t *) subhistogram_local_mem_18725)[sext_i32_i64(local_subhisto_i_18734) *
                                                                       hist_H_chk_18710 +
                                                                       sext_i32_i64(srem32(j_18732,
                                                                                           sext_i64_i32(histo_sizze_18711)))] =
                                0;
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_18736 = 0; i_18736 <
             sdiv_up32(sext_i64_i32(paths_17456) - pgtid_in_segment_18723,
                       threads_per_segment_18724); i_18736++) {
            int32_t gtid_18325 = i_18736 * threads_per_segment_18724 +
                    pgtid_in_segment_18723;
            int64_t i_p_o_18437 = add64(-1, gtid_18325);
            int64_t rot_i_18438 = smod64(i_p_o_18437, paths_17456);
            bool cond_18332 = gtid_18325 == 0;
            int64_t res_18333;
            
            if (cond_18332) {
                res_18333 = 0;
            } else {
                int64_t x_18331 = ((__global int64_t *) mem_18505)[rot_i_18438];
                
                res_18333 = x_18331;
            }
            if (chk_i_18708 == 0) {
                // save map-out results
                { }
            }
            // perform atomic updates
            {
                if (slt64(res_18333, res_17607) &&
                    (sle64(sext_i32_i64(chk_i_18708) * hist_H_chk_18710,
                           res_18333) && slt64(res_18333,
                                               sext_i32_i64(chk_i_18708) *
                                               hist_H_chk_18710 +
                                               hist_H_chk_18710))) {
                    int64_t x_18327;
                    int64_t x_18328;
                    
                    x_18328 = gtid_18325;
                    
                    int32_t old_18737;
                    volatile bool continue_18738;
                    
                    continue_18738 = 1;
                    while (continue_18738) {
                        old_18737 = atomic_cmpxchg_i32_local(&((volatile __local
                                                                int *) locks_mem_18727)[sext_i32_i64(thread_local_subhisto_i_18730) *
                                                                                        hist_H_chk_18710 +
                                                                                        (res_18333 -
                                                                                         sext_i32_i64(chk_i_18708) *
                                                                                         hist_H_chk_18710)],
                                                             0, 1);
                        if (old_18737 == 0) {
                            int64_t x_18327;
                            
                            // bind lhs
                            {
                                x_18327 = ((volatile __local
                                            int64_t *) subhistogram_local_mem_18725)[sext_i32_i64(thread_local_subhisto_i_18730) *
                                                                                     hist_H_chk_18710 +
                                                                                     (res_18333 -
                                                                                      sext_i32_i64(chk_i_18708) *
                                                                                      hist_H_chk_18710)];
                            }
                            // execute operation
                            {
                                int64_t res_18329 = smax64(x_18327, x_18328);
                                
                                x_18327 = res_18329;
                            }
                            // update global result
                            {
                                ((volatile __local
                                  int64_t *) subhistogram_local_mem_18725)[sext_i32_i64(thread_local_subhisto_i_18730) *
                                                                           hist_H_chk_18710 +
                                                                           (res_18333 -
                                                                            sext_i32_i64(chk_i_18708) *
                                                                            hist_H_chk_18710)] =
                                    x_18327;
                            }
                            mem_fence_local();
                            old_18737 =
                                atomic_cmpxchg_i32_local(&((volatile __local
                                                            int *) locks_mem_18727)[sext_i32_i64(thread_local_subhisto_i_18730) *
                                                                                    hist_H_chk_18710 +
                                                                                    (res_18333 -
                                                                                     sext_i32_i64(chk_i_18708) *
                                                                                     hist_H_chk_18710)],
                                                         1, 0);
                            continue_18738 = 0;
                        }
                        mem_fence_local();
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compact the multiple local memory subhistograms to result in global memory
        {
            int64_t trunc_H_18739 = smin64(hist_H_chk_18710, res_17607 -
                                           sext_i32_i64(chk_i_18708) *
                                           hist_H_chk_18710);
            int32_t histo_sizze_18740 = sext_i64_i32(trunc_H_18739);
            
            for (int32_t local_i_18741 = 0; local_i_18741 <
                 init_per_thread_18712; local_i_18741++) {
                int32_t j_18742 = local_i_18741 *
                        sext_i64_i32(max_group_sizze_18696) + local_tid_18714;
                
                if (slt32(j_18742, histo_sizze_18740)) {
                    int64_t x_18327;
                    int64_t x_18328;
                    
                    // Read values from subhistogram 0.
                    {
                        x_18327 = ((__local
                                    int64_t *) subhistogram_local_mem_18725)[sext_i32_i64(j_18742)];
                    }
                    // Accumulate based on values in other subhistograms.
                    {
                        for (int32_t subhisto_id_18743 = 0; subhisto_id_18743 <
                             hist_M_18703 - 1; subhisto_id_18743++) {
                            x_18328 = ((__local
                                        int64_t *) subhistogram_local_mem_18725)[(sext_i32_i64(subhisto_id_18743) +
                                                                                  1) *
                                                                                 hist_H_chk_18710 +
                                                                                 sext_i32_i64(j_18742)];
                            
                            int64_t res_18329;
                            
                            res_18329 = smax64(x_18327, x_18328);
                            x_18327 = res_18329;
                        }
                    }
                    // Put final bucket value in global memory.
                    {
                        ((__global
                          int64_t *) res_subhistos_mem_18687)[srem64(sext_i32_i64(virt_group_id_18720),
                                                                     num_groups_18697) *
                                                              res_17607 +
                                                              (sext_i32_i64(j_18742) +
                                                               sext_i32_i64(chk_i_18708) *
                                                               hist_H_chk_18710)] =
                            x_18327;
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
}
__kernel void mainzisegmap_18019(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args,
                                 int64_t paths_17456, int64_t steps_17457,
                                 float a_17461, float b_17462,
                                 float sigma_17463, float r0_17464,
                                 float dt_17468, int64_t upper_bound_17487,
                                 float res_17488, int64_t num_groups_18271,
                                 __global unsigned char *mem_18475, __global
                                 unsigned char *mem_18478, __global
                                 unsigned char *mem_18493)
{
    #define segmap_group_sizze_18270 (mainzisegmap_group_sizze_18021)
    
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
    
    int32_t global_tid_18580;
    int32_t local_tid_18581;
    int64_t group_sizze_18584;
    int32_t wave_sizze_18583;
    int32_t group_tid_18582;
    
    global_tid_18580 = get_global_id(0);
    local_tid_18581 = get_local_id(0);
    group_sizze_18584 = get_local_size(0);
    wave_sizze_18583 = LOCKSTEP_WIDTH;
    group_tid_18582 = get_group_id(0);
    
    int32_t phys_tid_18019;
    
    phys_tid_18019 = global_tid_18580;
    
    int32_t phys_group_id_18585;
    
    phys_group_id_18585 = get_group_id(0);
    for (int32_t i_18586 = 0; i_18586 <
         sdiv_up32(sext_i64_i32(sdiv_up64(paths_17456,
                                          segmap_group_sizze_18270)) -
                   phys_group_id_18585, sext_i64_i32(num_groups_18271));
         i_18586++) {
        int32_t virt_group_id_18587 = phys_group_id_18585 + i_18586 *
                sext_i64_i32(num_groups_18271);
        int64_t gtid_18018 = sext_i32_i64(virt_group_id_18587) *
                segmap_group_sizze_18270 + sext_i32_i64(local_tid_18581);
        
        if (slt64(gtid_18018, paths_17456)) {
            for (int64_t i_18588 = 0; i_18588 < steps_17457; i_18588++) {
                ((__global float *) mem_18478)[phys_tid_18019 + i_18588 *
                                               (num_groups_18271 *
                                                segmap_group_sizze_18270)] =
                    r0_17464;
            }
            for (int64_t i_18277 = 0; i_18277 < upper_bound_17487; i_18277++) {
                bool y_18279 = slt64(i_18277, steps_17457);
                bool index_certs_18280;
                
                if (!y_18279) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 0) ==
                            -1) {
                            global_failure_args[0] = i_18277;
                            global_failure_args[1] = steps_17457;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float shortstep_arg_18281 = ((__global
                                              float *) mem_18475)[i_18277 *
                                                                  paths_17456 +
                                                                  gtid_18018];
                float shortstep_arg_18282 = ((__global
                                              float *) mem_18478)[phys_tid_18019 +
                                                                  i_18277 *
                                                                  (num_groups_18271 *
                                                                   segmap_group_sizze_18270)];
                float y_18283 = b_17462 - shortstep_arg_18282;
                float x_18284 = a_17461 * y_18283;
                float x_18285 = dt_17468 * x_18284;
                float x_18286 = res_17488 * shortstep_arg_18281;
                float y_18287 = sigma_17463 * x_18286;
                float delta_r_18288 = x_18285 + y_18287;
                float res_18289 = shortstep_arg_18282 + delta_r_18288;
                int64_t i_18290 = add64(1, i_18277);
                bool x_18291 = sle64(0, i_18290);
                bool y_18292 = slt64(i_18290, steps_17457);
                bool bounds_check_18293 = x_18291 && y_18292;
                bool index_certs_18294;
                
                if (!bounds_check_18293) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 1) ==
                            -1) {
                            global_failure_args[0] = i_18290;
                            global_failure_args[1] = steps_17457;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                ((__global float *) mem_18478)[phys_tid_18019 + i_18290 *
                                               (num_groups_18271 *
                                                segmap_group_sizze_18270)] =
                    res_18289;
            }
            for (int64_t i_18590 = 0; i_18590 < steps_17457; i_18590++) {
                ((__global float *) mem_18493)[i_18590 * paths_17456 +
                                               gtid_18018] = ((__global
                                                               float *) mem_18478)[phys_tid_18019 +
                                                                                   i_18590 *
                                                                                   (num_groups_18271 *
                                                                                    segmap_group_sizze_18270)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18270
}
__kernel void mainzisegmap_18117(__global int *global_failure,
                                 int64_t paths_17456, int64_t steps_17457,
                                 __global unsigned char *mem_18468, __global
                                 unsigned char *mem_18472)
{
    #define segmap_group_sizze_18225 (mainzisegmap_group_sizze_18120)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18574;
    int32_t local_tid_18575;
    int64_t group_sizze_18578;
    int32_t wave_sizze_18577;
    int32_t group_tid_18576;
    
    global_tid_18574 = get_global_id(0);
    local_tid_18575 = get_local_id(0);
    group_sizze_18578 = get_local_size(0);
    wave_sizze_18577 = LOCKSTEP_WIDTH;
    group_tid_18576 = get_group_id(0);
    
    int32_t phys_tid_18117;
    
    phys_tid_18117 = global_tid_18574;
    
    int64_t gtid_18115;
    
    gtid_18115 = squot64(sext_i32_i64(group_tid_18576) *
                         segmap_group_sizze_18225 +
                         sext_i32_i64(local_tid_18575), steps_17457);
    
    int64_t gtid_18116;
    
    gtid_18116 = sext_i32_i64(group_tid_18576) * segmap_group_sizze_18225 +
        sext_i32_i64(local_tid_18575) - squot64(sext_i32_i64(group_tid_18576) *
                                                segmap_group_sizze_18225 +
                                                sext_i32_i64(local_tid_18575),
                                                steps_17457) * steps_17457;
    if (slt64(gtid_18115, paths_17456) && slt64(gtid_18116, steps_17457)) {
        int32_t unsign_arg_18228 = ((__global int32_t *) mem_18468)[gtid_18115];
        int32_t res_18230 = sext_i64_i32(gtid_18116);
        int32_t x_18231 = lshr32(res_18230, 16);
        int32_t x_18232 = res_18230 ^ x_18231;
        int32_t x_18233 = mul32(73244475, x_18232);
        int32_t x_18234 = lshr32(x_18233, 16);
        int32_t x_18235 = x_18233 ^ x_18234;
        int32_t x_18236 = mul32(73244475, x_18235);
        int32_t x_18237 = lshr32(x_18236, 16);
        int32_t x_18238 = x_18236 ^ x_18237;
        int32_t unsign_arg_18239 = unsign_arg_18228 ^ x_18238;
        int32_t unsign_arg_18240 = mul32(48271, unsign_arg_18239);
        int32_t unsign_arg_18241 = umod32(unsign_arg_18240, 2147483647);
        int32_t unsign_arg_18242 = mul32(48271, unsign_arg_18241);
        int32_t unsign_arg_18243 = umod32(unsign_arg_18242, 2147483647);
        float res_18244 = uitofp_i32_f32(unsign_arg_18241);
        float res_18245 = res_18244 / 2.1474836e9F;
        float res_18246 = uitofp_i32_f32(unsign_arg_18243);
        float res_18247 = res_18246 / 2.1474836e9F;
        float res_18248;
        
        res_18248 = futrts_log32(res_18245);
        
        float res_18249 = -2.0F * res_18248;
        float res_18250;
        
        res_18250 = futrts_sqrt32(res_18249);
        
        float res_18251 = 6.2831855F * res_18247;
        float res_18252;
        
        res_18252 = futrts_cos32(res_18251);
        
        float res_18253 = res_18250 * res_18252;
        
        ((__global float *) mem_18472)[gtid_18115 * steps_17457 + gtid_18116] =
            res_18253;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18225
}
__kernel void mainzisegmap_18181(__global int *global_failure,
                                 int64_t paths_17456, __global
                                 unsigned char *mem_18468)
{
    #define segmap_group_sizze_18200 (mainzisegmap_group_sizze_18183)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18569;
    int32_t local_tid_18570;
    int64_t group_sizze_18573;
    int32_t wave_sizze_18572;
    int32_t group_tid_18571;
    
    global_tid_18569 = get_global_id(0);
    local_tid_18570 = get_local_id(0);
    group_sizze_18573 = get_local_size(0);
    wave_sizze_18572 = LOCKSTEP_WIDTH;
    group_tid_18571 = get_group_id(0);
    
    int32_t phys_tid_18181;
    
    phys_tid_18181 = global_tid_18569;
    
    int64_t gtid_18180;
    
    gtid_18180 = sext_i32_i64(group_tid_18571) * segmap_group_sizze_18200 +
        sext_i32_i64(local_tid_18570);
    if (slt64(gtid_18180, paths_17456)) {
        int32_t res_18204 = sext_i64_i32(gtid_18180);
        int32_t x_18205 = lshr32(res_18204, 16);
        int32_t x_18206 = res_18204 ^ x_18205;
        int32_t x_18207 = mul32(73244475, x_18206);
        int32_t x_18208 = lshr32(x_18207, 16);
        int32_t x_18209 = x_18207 ^ x_18208;
        int32_t x_18210 = mul32(73244475, x_18209);
        int32_t x_18211 = lshr32(x_18210, 16);
        int32_t x_18212 = x_18210 ^ x_18211;
        int32_t unsign_arg_18213 = 777822902 ^ x_18212;
        int32_t unsign_arg_18214 = mul32(48271, unsign_arg_18213);
        int32_t unsign_arg_18215 = umod32(unsign_arg_18214, 2147483647);
        
        ((__global int32_t *) mem_18468)[gtid_18180] = unsign_arg_18215;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18200
}
__kernel void mainzisegmap_18411(__global int *global_failure,
                                 int64_t res_17607, int64_t num_segments_17799,
                                 __global unsigned char *mem_18525, __global
                                 unsigned char *mem_18530, __global
                                 unsigned char *mem_18533, __global
                                 unsigned char *mem_18535)
{
    #define segmap_group_sizze_18414 (mainzisegmap_group_sizze_18413)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19094;
    int32_t local_tid_19095;
    int64_t group_sizze_19098;
    int32_t wave_sizze_19097;
    int32_t group_tid_19096;
    
    global_tid_19094 = get_global_id(0);
    local_tid_19095 = get_local_id(0);
    group_sizze_19098 = get_local_size(0);
    wave_sizze_19097 = LOCKSTEP_WIDTH;
    group_tid_19096 = get_group_id(0);
    
    int32_t phys_tid_18411;
    
    phys_tid_18411 = global_tid_19094;
    
    int64_t write_i_18410;
    
    write_i_18410 = sext_i32_i64(group_tid_19096) * segmap_group_sizze_18414 +
        sext_i32_i64(local_tid_19095);
    if (slt64(write_i_18410, res_17607)) {
        int64_t i_p_o_18449 = add64(1, write_i_18410);
        int64_t rot_i_18450 = smod64(i_p_o_18449, res_17607);
        bool x_17812 = ((__global bool *) mem_18525)[rot_i_18450];
        float write_value_17813 = ((__global float *) mem_18530)[write_i_18410];
        int64_t res_17814;
        
        if (x_17812) {
            int64_t x_17811 = ((__global int64_t *) mem_18533)[write_i_18410];
            int64_t res_17815 = sub64(x_17811, 1);
            
            res_17814 = res_17815;
        } else {
            res_17814 = -1;
        }
        if (sle64(0, res_17814) && slt64(res_17814, num_segments_17799)) {
            ((__global float *) mem_18535)[res_17814] = write_value_17813;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18414
}
__kernel void mainzisegred_large_18772(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_18810_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_18808_backing_aligned_1,
                                       int64_t res_17607,
                                       int64_t num_groups_18323, __global
                                       unsigned char *mem_18512,
                                       int32_t num_subhistos_18686, __global
                                       unsigned char *res_subhistos_mem_18687,
                                       int64_t groups_per_segment_18794,
                                       int64_t elements_per_thread_18795,
                                       int64_t virt_num_groups_18796,
                                       int64_t threads_per_segment_18798,
                                       __global
                                       unsigned char *group_res_arr_mem_18799,
                                       __global
                                       unsigned char *mainzicounter_mem_18801)
{
    #define seghist_group_sizze_18321 (mainziseghist_group_sizze_18320)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_18810_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_18810_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_18808_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_18808_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18803;
    int32_t local_tid_18804;
    int64_t group_sizze_18807;
    int32_t wave_sizze_18806;
    int32_t group_tid_18805;
    
    global_tid_18803 = get_global_id(0);
    local_tid_18804 = get_local_id(0);
    group_sizze_18807 = get_local_size(0);
    wave_sizze_18806 = LOCKSTEP_WIDTH;
    group_tid_18805 = get_group_id(0);
    
    int32_t flat_gtid_18772;
    
    flat_gtid_18772 = global_tid_18803;
    
    __local char *red_arr_mem_18808;
    
    red_arr_mem_18808 = (__local char *) red_arr_mem_18808_backing_0;
    
    __local char *sync_arr_mem_18810;
    
    sync_arr_mem_18810 = (__local char *) sync_arr_mem_18810_backing_1;
    
    int32_t phys_group_id_18812;
    
    phys_group_id_18812 = get_group_id(0);
    for (int32_t i_18813 = 0; i_18813 <
         sdiv_up32(sext_i64_i32(virt_num_groups_18796) - phys_group_id_18812,
                   sext_i64_i32(num_groups_18323)); i_18813++) {
        int32_t virt_group_id_18814 = phys_group_id_18812 + i_18813 *
                sext_i64_i32(num_groups_18323);
        int32_t flat_segment_id_18815 = squot32(virt_group_id_18814,
                                                sext_i64_i32(groups_per_segment_18794));
        int64_t global_tid_18816 = srem64(sext_i32_i64(virt_group_id_18814) *
                                          seghist_group_sizze_18321 +
                                          sext_i32_i64(local_tid_18804),
                                          seghist_group_sizze_18321 *
                                          groups_per_segment_18794);
        int64_t bucket_id_18770 = sext_i32_i64(flat_segment_id_18815);
        int64_t subhistogram_id_18771;
        int64_t x_acc_18817;
        int64_t chunk_sizze_18818;
        
        chunk_sizze_18818 = smin64(elements_per_thread_18795,
                                   sdiv_up64(num_subhistos_18686 -
                                             sext_i32_i64(sext_i64_i32(global_tid_18816)),
                                             threads_per_segment_18798));
        
        int64_t x_18327;
        int64_t x_18328;
        
        // neutral-initialise the accumulators
        {
            x_acc_18817 = 0;
        }
        for (int64_t i_18822 = 0; i_18822 < chunk_sizze_18818; i_18822++) {
            subhistogram_id_18771 =
                sext_i32_i64(sext_i64_i32(global_tid_18816)) +
                threads_per_segment_18798 * i_18822;
            // apply map function
            {
                // load accumulator
                {
                    x_18327 = x_acc_18817;
                }
                // load new values
                {
                    x_18328 = ((__global
                                int64_t *) res_subhistos_mem_18687)[subhistogram_id_18771 *
                                                                    res_17607 +
                                                                    bucket_id_18770];
                }
                // apply reduction operator
                {
                    int64_t res_18329 = smax64(x_18327, x_18328);
                    
                    // store in accumulator
                    {
                        x_acc_18817 = res_18329;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_18327 = x_acc_18817;
            ((__local
              int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804)] =
                x_18327;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_18823;
        int32_t skip_waves_18824;
        
        skip_waves_18824 = 1;
        
        int64_t x_18819;
        int64_t x_18820;
        
        offset_18823 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_18804,
                      sext_i64_i32(seghist_group_sizze_18321))) {
                x_18819 = ((__local
                            int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804 +
                                                          offset_18823)];
            }
        }
        offset_18823 = 1;
        while (slt32(offset_18823, wave_sizze_18806)) {
            if (slt32(local_tid_18804 + offset_18823,
                      sext_i64_i32(seghist_group_sizze_18321)) &&
                ((local_tid_18804 - squot32(local_tid_18804, wave_sizze_18806) *
                  wave_sizze_18806) & (2 * offset_18823 - 1)) == 0) {
                // read array element
                {
                    x_18820 = ((volatile __local
                                int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804 +
                                                              offset_18823)];
                }
                // apply reduction operation
                {
                    int64_t res_18821 = smax64(x_18819, x_18820);
                    
                    x_18819 = res_18821;
                }
                // write result of operation
                {
                    ((volatile __local
                      int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804)] =
                        x_18819;
                }
            }
            offset_18823 *= 2;
        }
        while (slt32(skip_waves_18824,
                     squot32(sext_i64_i32(seghist_group_sizze_18321) +
                             wave_sizze_18806 - 1, wave_sizze_18806))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_18823 = skip_waves_18824 * wave_sizze_18806;
            if (slt32(local_tid_18804 + offset_18823,
                      sext_i64_i32(seghist_group_sizze_18321)) &&
                ((local_tid_18804 - squot32(local_tid_18804, wave_sizze_18806) *
                  wave_sizze_18806) == 0 && (squot32(local_tid_18804,
                                                     wave_sizze_18806) & (2 *
                                                                          skip_waves_18824 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_18820 = ((__local
                                int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804 +
                                                              offset_18823)];
                }
                // apply reduction operation
                {
                    int64_t res_18821 = smax64(x_18819, x_18820);
                    
                    x_18819 = res_18821;
                }
                // write result of operation
                {
                    ((__local
                      int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804)] =
                        x_18819;
                }
            }
            skip_waves_18824 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (sext_i32_i64(local_tid_18804) == 0) {
                x_acc_18817 = x_18819;
            }
        }
        if (groups_per_segment_18794 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_18804 == 0) {
                    ((__global int64_t *) mem_18512)[bucket_id_18770] =
                        x_acc_18817;
                }
            }
        } else {
            int32_t old_counter_18825;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_18804 == 0) {
                    ((__global
                      int64_t *) group_res_arr_mem_18799)[sext_i32_i64(virt_group_id_18814) *
                                                          seghist_group_sizze_18321] =
                        x_acc_18817;
                    mem_fence_global();
                    old_counter_18825 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_18801)[sext_i32_i64(srem32(flat_segment_id_18815,
                                                                                                     10240))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_18810)[0] =
                        old_counter_18825 == groups_per_segment_18794 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_18826;
            
            is_last_group_18826 = ((__local bool *) sync_arr_mem_18810)[0];
            if (is_last_group_18826) {
                if (local_tid_18804 == 0) {
                    old_counter_18825 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_18801)[sext_i32_i64(srem32(flat_segment_id_18815,
                                                                                                     10240))],
                                              (int) (0 -
                                                     groups_per_segment_18794));
                }
                // read in the per-group-results
                {
                    int64_t read_per_thread_18827 =
                            sdiv_up64(groups_per_segment_18794,
                                      seghist_group_sizze_18321);
                    
                    x_18327 = 0;
                    for (int64_t i_18828 = 0; i_18828 < read_per_thread_18827;
                         i_18828++) {
                        int64_t group_res_id_18829 =
                                sext_i32_i64(local_tid_18804) *
                                read_per_thread_18827 + i_18828;
                        int64_t index_of_group_res_18830 =
                                sext_i32_i64(flat_segment_id_18815) *
                                groups_per_segment_18794 + group_res_id_18829;
                        
                        if (slt64(group_res_id_18829,
                                  groups_per_segment_18794)) {
                            x_18328 = ((__global
                                        int64_t *) group_res_arr_mem_18799)[index_of_group_res_18830 *
                                                                            seghist_group_sizze_18321];
                            
                            int64_t res_18329;
                            
                            res_18329 = smax64(x_18327, x_18328);
                            x_18327 = res_18329;
                        }
                    }
                }
                ((__local
                  int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804)] =
                    x_18327;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_18831;
                    int32_t skip_waves_18832;
                    
                    skip_waves_18832 = 1;
                    
                    int64_t x_18819;
                    int64_t x_18820;
                    
                    offset_18831 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_18804,
                                  sext_i64_i32(seghist_group_sizze_18321))) {
                            x_18819 = ((__local
                                        int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804 +
                                                                      offset_18831)];
                        }
                    }
                    offset_18831 = 1;
                    while (slt32(offset_18831, wave_sizze_18806)) {
                        if (slt32(local_tid_18804 + offset_18831,
                                  sext_i64_i32(seghist_group_sizze_18321)) &&
                            ((local_tid_18804 - squot32(local_tid_18804,
                                                        wave_sizze_18806) *
                              wave_sizze_18806) & (2 * offset_18831 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_18820 = ((volatile __local
                                            int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804 +
                                                                          offset_18831)];
                            }
                            // apply reduction operation
                            {
                                int64_t res_18821 = smax64(x_18819, x_18820);
                                
                                x_18819 = res_18821;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804)] =
                                    x_18819;
                            }
                        }
                        offset_18831 *= 2;
                    }
                    while (slt32(skip_waves_18832,
                                 squot32(sext_i64_i32(seghist_group_sizze_18321) +
                                         wave_sizze_18806 - 1,
                                         wave_sizze_18806))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_18831 = skip_waves_18832 * wave_sizze_18806;
                        if (slt32(local_tid_18804 + offset_18831,
                                  sext_i64_i32(seghist_group_sizze_18321)) &&
                            ((local_tid_18804 - squot32(local_tid_18804,
                                                        wave_sizze_18806) *
                              wave_sizze_18806) == 0 &&
                             (squot32(local_tid_18804, wave_sizze_18806) & (2 *
                                                                            skip_waves_18832 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_18820 = ((__local
                                            int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804 +
                                                                          offset_18831)];
                            }
                            // apply reduction operation
                            {
                                int64_t res_18821 = smax64(x_18819, x_18820);
                                
                                x_18819 = res_18821;
                            }
                            // write result of operation
                            {
                                ((__local
                                  int64_t *) red_arr_mem_18808)[sext_i32_i64(local_tid_18804)] =
                                    x_18819;
                            }
                        }
                        skip_waves_18832 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_18804 == 0) {
                            ((__global int64_t *) mem_18512)[bucket_id_18770] =
                                x_18819;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef seghist_group_sizze_18321
}
__kernel void mainzisegred_nonseg_18318(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_18659_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_18657_backing_aligned_1,
                                        int64_t paths_17456,
                                        int64_t num_groups_18313, __global
                                        unsigned char *mem_18507, __global
                                        unsigned char *mem_18510, __global
                                        unsigned char *mainzicounter_mem_18647,
                                        __global
                                        unsigned char *group_res_arr_mem_18649,
                                        int64_t num_threads_18651)
{
    #define segred_group_sizze_18311 (mainzisegred_group_sizze_18310)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_18659_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_18659_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_18657_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_18657_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18652;
    int32_t local_tid_18653;
    int64_t group_sizze_18656;
    int32_t wave_sizze_18655;
    int32_t group_tid_18654;
    
    global_tid_18652 = get_global_id(0);
    local_tid_18653 = get_local_id(0);
    group_sizze_18656 = get_local_size(0);
    wave_sizze_18655 = LOCKSTEP_WIDTH;
    group_tid_18654 = get_group_id(0);
    
    int32_t phys_tid_18318;
    
    phys_tid_18318 = global_tid_18652;
    
    __local char *sync_arr_mem_18657;
    
    sync_arr_mem_18657 = (__local char *) sync_arr_mem_18657_backing_0;
    
    __local char *red_arr_mem_18659;
    
    red_arr_mem_18659 = (__local char *) red_arr_mem_18659_backing_1;
    
    int64_t dummy_18316;
    
    dummy_18316 = 0;
    
    int64_t gtid_18317;
    
    gtid_18317 = 0;
    
    int64_t x_acc_18661;
    int64_t chunk_sizze_18662;
    
    chunk_sizze_18662 = smin64(sdiv_up64(paths_17456,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_18311 *
                                         num_groups_18313))),
                               sdiv_up64(paths_17456 -
                                         sext_i32_i64(phys_tid_18318),
                                         num_threads_18651));
    
    int64_t x_17608;
    int64_t x_17609;
    
    // neutral-initialise the accumulators
    {
        x_acc_18661 = 0;
    }
    for (int64_t i_18666 = 0; i_18666 < chunk_sizze_18662; i_18666++) {
        gtid_18317 = sext_i32_i64(phys_tid_18318) + num_threads_18651 * i_18666;
        // apply map function
        {
            int64_t x_17611 = ((__global int64_t *) mem_18507)[gtid_18317];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_17608 = x_acc_18661;
            }
            // load new values
            {
                x_17609 = x_17611;
            }
            // apply reduction operator
            {
                int64_t res_17610 = add64(x_17608, x_17609);
                
                // store in accumulator
                {
                    x_acc_18661 = res_17610;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_17608 = x_acc_18661;
        ((__local int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653)] =
            x_17608;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_18667;
    int32_t skip_waves_18668;
    
    skip_waves_18668 = 1;
    
    int64_t x_18663;
    int64_t x_18664;
    
    offset_18667 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_18653, sext_i64_i32(segred_group_sizze_18311))) {
            x_18663 = ((__local
                        int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653 +
                                                      offset_18667)];
        }
    }
    offset_18667 = 1;
    while (slt32(offset_18667, wave_sizze_18655)) {
        if (slt32(local_tid_18653 + offset_18667,
                  sext_i64_i32(segred_group_sizze_18311)) && ((local_tid_18653 -
                                                               squot32(local_tid_18653,
                                                                       wave_sizze_18655) *
                                                               wave_sizze_18655) &
                                                              (2 *
                                                               offset_18667 -
                                                               1)) == 0) {
            // read array element
            {
                x_18664 = ((volatile __local
                            int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653 +
                                                          offset_18667)];
            }
            // apply reduction operation
            {
                int64_t res_18665 = add64(x_18663, x_18664);
                
                x_18663 = res_18665;
            }
            // write result of operation
            {
                ((volatile __local
                  int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653)] =
                    x_18663;
            }
        }
        offset_18667 *= 2;
    }
    while (slt32(skip_waves_18668,
                 squot32(sext_i64_i32(segred_group_sizze_18311) +
                         wave_sizze_18655 - 1, wave_sizze_18655))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_18667 = skip_waves_18668 * wave_sizze_18655;
        if (slt32(local_tid_18653 + offset_18667,
                  sext_i64_i32(segred_group_sizze_18311)) && ((local_tid_18653 -
                                                               squot32(local_tid_18653,
                                                                       wave_sizze_18655) *
                                                               wave_sizze_18655) ==
                                                              0 &&
                                                              (squot32(local_tid_18653,
                                                                       wave_sizze_18655) &
                                                               (2 *
                                                                skip_waves_18668 -
                                                                1)) == 0)) {
            // read array element
            {
                x_18664 = ((__local
                            int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653 +
                                                          offset_18667)];
            }
            // apply reduction operation
            {
                int64_t res_18665 = add64(x_18663, x_18664);
                
                x_18663 = res_18665;
            }
            // write result of operation
            {
                ((__local
                  int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653)] =
                    x_18663;
            }
        }
        skip_waves_18668 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_18653) == 0) {
            x_acc_18661 = x_18663;
        }
    }
    
    int32_t old_counter_18669;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_18653 == 0) {
            ((__global
              int64_t *) group_res_arr_mem_18649)[sext_i32_i64(group_tid_18654) *
                                                  segred_group_sizze_18311] =
                x_acc_18661;
            mem_fence_global();
            old_counter_18669 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_18647)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_18657)[0] = old_counter_18669 ==
                num_groups_18313 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_18670;
    
    is_last_group_18670 = ((__local bool *) sync_arr_mem_18657)[0];
    if (is_last_group_18670) {
        if (local_tid_18653 == 0) {
            old_counter_18669 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_18647)[0],
                                                      (int) (0 -
                                                             num_groups_18313));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_18671 = sdiv_up64(num_groups_18313,
                                                      segred_group_sizze_18311);
            
            x_17608 = 0;
            for (int64_t i_18672 = 0; i_18672 < read_per_thread_18671;
                 i_18672++) {
                int64_t group_res_id_18673 = sext_i32_i64(local_tid_18653) *
                        read_per_thread_18671 + i_18672;
                int64_t index_of_group_res_18674 = group_res_id_18673;
                
                if (slt64(group_res_id_18673, num_groups_18313)) {
                    x_17609 = ((__global
                                int64_t *) group_res_arr_mem_18649)[index_of_group_res_18674 *
                                                                    segred_group_sizze_18311];
                    
                    int64_t res_17610;
                    
                    res_17610 = add64(x_17608, x_17609);
                    x_17608 = res_17610;
                }
            }
        }
        ((__local int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653)] =
            x_17608;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_18675;
            int32_t skip_waves_18676;
            
            skip_waves_18676 = 1;
            
            int64_t x_18663;
            int64_t x_18664;
            
            offset_18675 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_18653,
                          sext_i64_i32(segred_group_sizze_18311))) {
                    x_18663 = ((__local
                                int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653 +
                                                              offset_18675)];
                }
            }
            offset_18675 = 1;
            while (slt32(offset_18675, wave_sizze_18655)) {
                if (slt32(local_tid_18653 + offset_18675,
                          sext_i64_i32(segred_group_sizze_18311)) &&
                    ((local_tid_18653 - squot32(local_tid_18653,
                                                wave_sizze_18655) *
                      wave_sizze_18655) & (2 * offset_18675 - 1)) == 0) {
                    // read array element
                    {
                        x_18664 = ((volatile __local
                                    int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653 +
                                                                  offset_18675)];
                    }
                    // apply reduction operation
                    {
                        int64_t res_18665 = add64(x_18663, x_18664);
                        
                        x_18663 = res_18665;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653)] =
                            x_18663;
                    }
                }
                offset_18675 *= 2;
            }
            while (slt32(skip_waves_18676,
                         squot32(sext_i64_i32(segred_group_sizze_18311) +
                                 wave_sizze_18655 - 1, wave_sizze_18655))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_18675 = skip_waves_18676 * wave_sizze_18655;
                if (slt32(local_tid_18653 + offset_18675,
                          sext_i64_i32(segred_group_sizze_18311)) &&
                    ((local_tid_18653 - squot32(local_tid_18653,
                                                wave_sizze_18655) *
                      wave_sizze_18655) == 0 && (squot32(local_tid_18653,
                                                         wave_sizze_18655) &
                                                 (2 * skip_waves_18676 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_18664 = ((__local
                                    int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653 +
                                                                  offset_18675)];
                    }
                    // apply reduction operation
                    {
                        int64_t res_18665 = add64(x_18663, x_18664);
                        
                        x_18663 = res_18665;
                    }
                    // write result of operation
                    {
                        ((__local
                          int64_t *) red_arr_mem_18659)[sext_i32_i64(local_tid_18653)] =
                            x_18663;
                    }
                }
                skip_waves_18676 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_18653 == 0) {
                    ((__global int64_t *) mem_18510)[0] = x_18663;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_18311
}
__kernel void mainzisegred_nonseg_18425(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_19111_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_19109_backing_aligned_1,
                                        int64_t paths_17456,
                                        int64_t num_groups_18420, __global
                                        unsigned char *mem_18535, __global
                                        unsigned char *mem_18539, __global
                                        unsigned char *mainzicounter_mem_19099,
                                        __global
                                        unsigned char *group_res_arr_mem_19101,
                                        int64_t num_threads_19103)
{
    #define segred_group_sizze_18418 (mainzisegred_group_sizze_18417)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_19111_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_19111_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_19109_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_19109_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_19104;
    int32_t local_tid_19105;
    int64_t group_sizze_19108;
    int32_t wave_sizze_19107;
    int32_t group_tid_19106;
    
    global_tid_19104 = get_global_id(0);
    local_tid_19105 = get_local_id(0);
    group_sizze_19108 = get_local_size(0);
    wave_sizze_19107 = LOCKSTEP_WIDTH;
    group_tid_19106 = get_group_id(0);
    
    int32_t phys_tid_18425;
    
    phys_tid_18425 = global_tid_19104;
    
    __local char *sync_arr_mem_19109;
    
    sync_arr_mem_19109 = (__local char *) sync_arr_mem_19109_backing_0;
    
    __local char *red_arr_mem_19111;
    
    red_arr_mem_19111 = (__local char *) red_arr_mem_19111_backing_1;
    
    int64_t dummy_18423;
    
    dummy_18423 = 0;
    
    int64_t gtid_18424;
    
    gtid_18424 = 0;
    
    float x_acc_19113;
    int64_t chunk_sizze_19114;
    
    chunk_sizze_19114 = smin64(sdiv_up64(paths_17456,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_18418 *
                                         num_groups_18420))),
                               sdiv_up64(paths_17456 -
                                         sext_i32_i64(phys_tid_18425),
                                         num_threads_19103));
    
    float x_17820;
    float x_17821;
    
    // neutral-initialise the accumulators
    {
        x_acc_19113 = 0.0F;
    }
    for (int64_t i_19118 = 0; i_19118 < chunk_sizze_19114; i_19118++) {
        gtid_18424 = sext_i32_i64(phys_tid_18425) + num_threads_19103 * i_19118;
        // apply map function
        {
            float x_17823 = ((__global float *) mem_18535)[gtid_18424];
            float res_17824 = fmax32(0.0F, x_17823);
            
            // save map-out results
            { }
            // load accumulator
            {
                x_17820 = x_acc_19113;
            }
            // load new values
            {
                x_17821 = res_17824;
            }
            // apply reduction operator
            {
                float res_17822 = x_17820 + x_17821;
                
                // store in accumulator
                {
                    x_acc_19113 = res_17822;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_17820 = x_acc_19113;
        ((__local float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105)] =
            x_17820;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_19119;
    int32_t skip_waves_19120;
    
    skip_waves_19120 = 1;
    
    float x_19115;
    float x_19116;
    
    offset_19119 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_19105, sext_i64_i32(segred_group_sizze_18418))) {
            x_19115 = ((__local
                        float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105 +
                                                    offset_19119)];
        }
    }
    offset_19119 = 1;
    while (slt32(offset_19119, wave_sizze_19107)) {
        if (slt32(local_tid_19105 + offset_19119,
                  sext_i64_i32(segred_group_sizze_18418)) && ((local_tid_19105 -
                                                               squot32(local_tid_19105,
                                                                       wave_sizze_19107) *
                                                               wave_sizze_19107) &
                                                              (2 *
                                                               offset_19119 -
                                                               1)) == 0) {
            // read array element
            {
                x_19116 = ((volatile __local
                            float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105 +
                                                        offset_19119)];
            }
            // apply reduction operation
            {
                float res_19117 = x_19115 + x_19116;
                
                x_19115 = res_19117;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105)] =
                    x_19115;
            }
        }
        offset_19119 *= 2;
    }
    while (slt32(skip_waves_19120,
                 squot32(sext_i64_i32(segred_group_sizze_18418) +
                         wave_sizze_19107 - 1, wave_sizze_19107))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_19119 = skip_waves_19120 * wave_sizze_19107;
        if (slt32(local_tid_19105 + offset_19119,
                  sext_i64_i32(segred_group_sizze_18418)) && ((local_tid_19105 -
                                                               squot32(local_tid_19105,
                                                                       wave_sizze_19107) *
                                                               wave_sizze_19107) ==
                                                              0 &&
                                                              (squot32(local_tid_19105,
                                                                       wave_sizze_19107) &
                                                               (2 *
                                                                skip_waves_19120 -
                                                                1)) == 0)) {
            // read array element
            {
                x_19116 = ((__local
                            float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105 +
                                                        offset_19119)];
            }
            // apply reduction operation
            {
                float res_19117 = x_19115 + x_19116;
                
                x_19115 = res_19117;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105)] =
                    x_19115;
            }
        }
        skip_waves_19120 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_19105) == 0) {
            x_acc_19113 = x_19115;
        }
    }
    
    int32_t old_counter_19121;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_19105 == 0) {
            ((__global
              float *) group_res_arr_mem_19101)[sext_i32_i64(group_tid_19106) *
                                                segred_group_sizze_18418] =
                x_acc_19113;
            mem_fence_global();
            old_counter_19121 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_19099)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_19109)[0] = old_counter_19121 ==
                num_groups_18420 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_19122;
    
    is_last_group_19122 = ((__local bool *) sync_arr_mem_19109)[0];
    if (is_last_group_19122) {
        if (local_tid_19105 == 0) {
            old_counter_19121 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_19099)[0],
                                                      (int) (0 -
                                                             num_groups_18420));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_19123 = sdiv_up64(num_groups_18420,
                                                      segred_group_sizze_18418);
            
            x_17820 = 0.0F;
            for (int64_t i_19124 = 0; i_19124 < read_per_thread_19123;
                 i_19124++) {
                int64_t group_res_id_19125 = sext_i32_i64(local_tid_19105) *
                        read_per_thread_19123 + i_19124;
                int64_t index_of_group_res_19126 = group_res_id_19125;
                
                if (slt64(group_res_id_19125, num_groups_18420)) {
                    x_17821 = ((__global
                                float *) group_res_arr_mem_19101)[index_of_group_res_19126 *
                                                                  segred_group_sizze_18418];
                    
                    float res_17822;
                    
                    res_17822 = x_17820 + x_17821;
                    x_17820 = res_17822;
                }
            }
        }
        ((__local float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105)] =
            x_17820;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_19127;
            int32_t skip_waves_19128;
            
            skip_waves_19128 = 1;
            
            float x_19115;
            float x_19116;
            
            offset_19127 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_19105,
                          sext_i64_i32(segred_group_sizze_18418))) {
                    x_19115 = ((__local
                                float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105 +
                                                            offset_19127)];
                }
            }
            offset_19127 = 1;
            while (slt32(offset_19127, wave_sizze_19107)) {
                if (slt32(local_tid_19105 + offset_19127,
                          sext_i64_i32(segred_group_sizze_18418)) &&
                    ((local_tid_19105 - squot32(local_tid_19105,
                                                wave_sizze_19107) *
                      wave_sizze_19107) & (2 * offset_19127 - 1)) == 0) {
                    // read array element
                    {
                        x_19116 = ((volatile __local
                                    float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105 +
                                                                offset_19127)];
                    }
                    // apply reduction operation
                    {
                        float res_19117 = x_19115 + x_19116;
                        
                        x_19115 = res_19117;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105)] =
                            x_19115;
                    }
                }
                offset_19127 *= 2;
            }
            while (slt32(skip_waves_19128,
                         squot32(sext_i64_i32(segred_group_sizze_18418) +
                                 wave_sizze_19107 - 1, wave_sizze_19107))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_19127 = skip_waves_19128 * wave_sizze_19107;
                if (slt32(local_tid_19105 + offset_19127,
                          sext_i64_i32(segred_group_sizze_18418)) &&
                    ((local_tid_19105 - squot32(local_tid_19105,
                                                wave_sizze_19107) *
                      wave_sizze_19107) == 0 && (squot32(local_tid_19105,
                                                         wave_sizze_19107) &
                                                 (2 * skip_waves_19128 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_19116 = ((__local
                                    float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105 +
                                                                offset_19127)];
                    }
                    // apply reduction operation
                    {
                        float res_19117 = x_19115 + x_19116;
                        
                        x_19115 = res_19117;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_19111)[sext_i32_i64(local_tid_19105)] =
                            x_19115;
                    }
                }
                skip_waves_19128 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_19105 == 0) {
                    ((__global float *) mem_18539)[0] = x_19115;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_18418
}
__kernel void mainzisegred_small_18772(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_18780_backing_aligned_0,
                                       int64_t res_17607,
                                       int64_t num_groups_18323, __global
                                       unsigned char *mem_18512,
                                       int32_t num_subhistos_18686, __global
                                       unsigned char *res_subhistos_mem_18687,
                                       int64_t segment_sizze_nonzzero_18773)
{
    #define seghist_group_sizze_18321 (mainziseghist_group_sizze_18320)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_18780_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_18780_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_18775;
    int32_t local_tid_18776;
    int64_t group_sizze_18779;
    int32_t wave_sizze_18778;
    int32_t group_tid_18777;
    
    global_tid_18775 = get_global_id(0);
    local_tid_18776 = get_local_id(0);
    group_sizze_18779 = get_local_size(0);
    wave_sizze_18778 = LOCKSTEP_WIDTH;
    group_tid_18777 = get_group_id(0);
    
    int32_t flat_gtid_18772;
    
    flat_gtid_18772 = global_tid_18775;
    
    __local char *red_arr_mem_18780;
    
    red_arr_mem_18780 = (__local char *) red_arr_mem_18780_backing_0;
    
    int32_t phys_group_id_18782;
    
    phys_group_id_18782 = get_group_id(0);
    for (int32_t i_18783 = 0; i_18783 <
         sdiv_up32(sext_i64_i32(sdiv_up64(res_17607,
                                          squot64(seghist_group_sizze_18321,
                                                  segment_sizze_nonzzero_18773))) -
                   phys_group_id_18782, sext_i64_i32(num_groups_18323));
         i_18783++) {
        int32_t virt_group_id_18784 = phys_group_id_18782 + i_18783 *
                sext_i64_i32(num_groups_18323);
        int64_t bucket_id_18770 = squot64(sext_i32_i64(local_tid_18776),
                                          segment_sizze_nonzzero_18773) +
                sext_i32_i64(virt_group_id_18784) *
                squot64(seghist_group_sizze_18321,
                        segment_sizze_nonzzero_18773);
        int64_t subhistogram_id_18771 = srem64(sext_i32_i64(local_tid_18776),
                                               num_subhistos_18686);
        
        // apply map function if in bounds
        {
            if (slt64(0, num_subhistos_18686) && (slt64(bucket_id_18770,
                                                        res_17607) &&
                                                  slt64(sext_i32_i64(local_tid_18776),
                                                        num_subhistos_18686 *
                                                        squot64(seghist_group_sizze_18321,
                                                                segment_sizze_nonzzero_18773)))) {
                // save results to be reduced
                {
                    ((__local
                      int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)] =
                        ((__global
                          int64_t *) res_subhistos_mem_18687)[subhistogram_id_18771 *
                                                              res_17607 +
                                                              bucket_id_18770];
                }
            } else {
                ((__local
                  int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)] =
                    0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt64(0, num_subhistos_18686)) {
            // perform segmented scan to imitate reduction
            {
                int64_t x_18327;
                int64_t x_18328;
                int64_t x_18785;
                int64_t x_18786;
                bool ltid_in_bounds_18788;
                
                ltid_in_bounds_18788 = slt64(sext_i32_i64(local_tid_18776),
                                             num_subhistos_18686 *
                                             squot64(seghist_group_sizze_18321,
                                                     segment_sizze_nonzzero_18773));
                
                int32_t skip_threads_18789;
                
                // read input for in-block scan
                {
                    if (ltid_in_bounds_18788) {
                        x_18328 = ((volatile __local
                                    int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)];
                        if ((local_tid_18776 - squot32(local_tid_18776, 32) *
                             32) == 0) {
                            x_18327 = x_18328;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_18789 = 1;
                    while (slt32(skip_threads_18789, 32)) {
                        if (sle32(skip_threads_18789, local_tid_18776 -
                                  squot32(local_tid_18776, 32) * 32) &&
                            ltid_in_bounds_18788) {
                            // read operands
                            {
                                x_18327 = ((volatile __local
                                            int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776) -
                                                                          sext_i32_i64(skip_threads_18789)];
                            }
                            // perform operation
                            {
                                bool inactive_18790 =
                                     slt64(srem64(sext_i32_i64(local_tid_18776),
                                                  num_subhistos_18686),
                                           sext_i32_i64(local_tid_18776) -
                                           sext_i32_i64(local_tid_18776 -
                                           skip_threads_18789));
                                
                                if (inactive_18790) {
                                    x_18327 = x_18328;
                                }
                                if (!inactive_18790) {
                                    int64_t res_18329 = smax64(x_18327,
                                                               x_18328);
                                    
                                    x_18327 = res_18329;
                                }
                            }
                        }
                        if (sle32(wave_sizze_18778, skip_threads_18789)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_18789, local_tid_18776 -
                                  squot32(local_tid_18776, 32) * 32) &&
                            ltid_in_bounds_18788) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)] =
                                    x_18327;
                                x_18328 = x_18327;
                            }
                        }
                        if (sle32(wave_sizze_18778, skip_threads_18789)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_18789 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_18776 - squot32(local_tid_18776, 32) * 32) ==
                        31 && ltid_in_bounds_18788) {
                        ((volatile __local
                          int64_t *) red_arr_mem_18780)[sext_i32_i64(squot32(local_tid_18776,
                                                                             32))] =
                            x_18327;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_18791;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_18776, 32) == 0 &&
                            ltid_in_bounds_18788) {
                            x_18786 = ((volatile __local
                                        int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)];
                            if ((local_tid_18776 - squot32(local_tid_18776,
                                                           32) * 32) == 0) {
                                x_18785 = x_18786;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_18791 = 1;
                        while (slt32(skip_threads_18791, 32)) {
                            if (sle32(skip_threads_18791, local_tid_18776 -
                                      squot32(local_tid_18776, 32) * 32) &&
                                (squot32(local_tid_18776, 32) == 0 &&
                                 ltid_in_bounds_18788)) {
                                // read operands
                                {
                                    x_18785 = ((volatile __local
                                                int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776) -
                                                                              sext_i32_i64(skip_threads_18791)];
                                }
                                // perform operation
                                {
                                    bool inactive_18792 =
                                         slt64(srem64(sext_i32_i64(local_tid_18776 *
                                                      32 + 32 - 1),
                                                      num_subhistos_18686),
                                               sext_i32_i64(local_tid_18776 *
                                               32 + 32 - 1) -
                                               sext_i32_i64((local_tid_18776 -
                                                             skip_threads_18791) *
                                               32 + 32 - 1));
                                    
                                    if (inactive_18792) {
                                        x_18785 = x_18786;
                                    }
                                    if (!inactive_18792) {
                                        int64_t res_18787 = smax64(x_18785,
                                                                   x_18786);
                                        
                                        x_18785 = res_18787;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_18778, skip_threads_18791)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_18791, local_tid_18776 -
                                      squot32(local_tid_18776, 32) * 32) &&
                                (squot32(local_tid_18776, 32) == 0 &&
                                 ltid_in_bounds_18788)) {
                                // write result
                                {
                                    ((volatile __local
                                      int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)] =
                                        x_18785;
                                    x_18786 = x_18785;
                                }
                            }
                            if (sle32(wave_sizze_18778, skip_threads_18791)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_18791 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_18776, 32) == 0 ||
                          !ltid_in_bounds_18788)) {
                        // read operands
                        {
                            x_18328 = x_18327;
                            x_18327 = ((__local
                                        int64_t *) red_arr_mem_18780)[sext_i32_i64(squot32(local_tid_18776,
                                                                                           32)) -
                                                                      1];
                        }
                        // perform operation
                        {
                            bool inactive_18793 =
                                 slt64(srem64(sext_i32_i64(local_tid_18776),
                                              num_subhistos_18686),
                                       sext_i32_i64(local_tid_18776) -
                                       sext_i32_i64(squot32(local_tid_18776,
                                                            32) * 32 - 1));
                            
                            if (inactive_18793) {
                                x_18327 = x_18328;
                            }
                            if (!inactive_18793) {
                                int64_t res_18329 = smax64(x_18327, x_18328);
                                
                                x_18327 = res_18329;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)] =
                                x_18327;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_18776, 32) == 0) {
                        ((__local
                          int64_t *) red_arr_mem_18780)[sext_i32_i64(local_tid_18776)] =
                            x_18328;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt64(sext_i32_i64(virt_group_id_18784) *
                      squot64(seghist_group_sizze_18321,
                              segment_sizze_nonzzero_18773) +
                      sext_i32_i64(local_tid_18776), res_17607) &&
                slt64(sext_i32_i64(local_tid_18776),
                      squot64(seghist_group_sizze_18321,
                              segment_sizze_nonzzero_18773))) {
                ((__global
                  int64_t *) mem_18512)[sext_i32_i64(virt_group_id_18784) *
                                        squot64(seghist_group_sizze_18321,
                                                segment_sizze_nonzzero_18773) +
                                        sext_i32_i64(local_tid_18776)] =
                    ((__local
                      int64_t *) red_arr_mem_18780)[(sext_i32_i64(local_tid_18776) +
                                                     1) *
                                                    segment_sizze_nonzzero_18773 -
                                                    1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef seghist_group_sizze_18321
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
  entry_points = {"main": (["i64", "i64", "f32", "i64", "f32", "f32", "f32",
                            "f32", "f32"], ["f32", "[]f32"])}
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
    self.global_failure_args_max = 2
    self.failure_msgs=["Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:84:112-119\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:105:1-139:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:84:58-120\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:105:1-139:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:130:37-88\n   #6  cva.fut:129:18-132:79\n   #7  cva.fut:105:1-139:18\n"]
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
                                       all_sizes={"builtin#replicate_f32.group_size_19092": {"class": "group_size",
                                                                                   "value": None},
                                        "builtin#replicate_i64.group_size_18684": {"class": "group_size",
                                                                                   "value": None},
                                        "main.L2_size_18749": {"class": "L2_for_histogram", "value": 4194304},
                                        "main.seghist_group_size_18320": {"class": "group_size", "value": None},
                                        "main.seghist_num_groups_18322": {"class": "num_groups", "value": None},
                                        "main.segmap_group_size_18021": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_18120": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_18183": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_18413": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_18023": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_18310": {"class": "group_size", "value": None},
                                        "main.segred_group_size_18417": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_18312": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_18419": {"class": "num_groups", "value": None},
                                        "main.segscan_group_size_18302": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_18335": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_18343": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_18351": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_18403": {"class": "group_size", "value": None},
                                        "main.segscan_num_groups_18304": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_18337": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_18345": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_18353": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_18405": {"class": "num_groups", "value": None}})
    self.builtinzhreplicate_f32zireplicate_19089_var = program.builtinzhreplicate_f32zireplicate_19089
    self.builtinzhreplicate_i64zireplicate_18681_var = program.builtinzhreplicate_i64zireplicate_18681
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainziscan_stage1_18308_var = program.mainziscan_stage1_18308
    self.mainziscan_stage1_18341_var = program.mainziscan_stage1_18341
    self.mainziscan_stage1_18349_var = program.mainziscan_stage1_18349
    self.mainziscan_stage1_18357_var = program.mainziscan_stage1_18357
    self.mainziscan_stage1_18409_var = program.mainziscan_stage1_18409
    self.mainziscan_stage2_18308_var = program.mainziscan_stage2_18308
    self.mainziscan_stage2_18341_var = program.mainziscan_stage2_18341
    self.mainziscan_stage2_18349_var = program.mainziscan_stage2_18349
    self.mainziscan_stage2_18357_var = program.mainziscan_stage2_18357
    self.mainziscan_stage2_18409_var = program.mainziscan_stage2_18409
    self.mainziscan_stage3_18308_var = program.mainziscan_stage3_18308
    self.mainziscan_stage3_18341_var = program.mainziscan_stage3_18341
    self.mainziscan_stage3_18349_var = program.mainziscan_stage3_18349
    self.mainziscan_stage3_18357_var = program.mainziscan_stage3_18357
    self.mainziscan_stage3_18409_var = program.mainziscan_stage3_18409
    self.mainziseghist_global_18326_var = program.mainziseghist_global_18326
    self.mainziseghist_local_18326_var = program.mainziseghist_local_18326
    self.mainzisegmap_18019_var = program.mainzisegmap_18019
    self.mainzisegmap_18117_var = program.mainzisegmap_18117
    self.mainzisegmap_18181_var = program.mainzisegmap_18181
    self.mainzisegmap_18411_var = program.mainzisegmap_18411
    self.mainzisegred_large_18772_var = program.mainzisegred_large_18772
    self.mainzisegred_nonseg_18318_var = program.mainzisegred_nonseg_18318
    self.mainzisegred_nonseg_18425_var = program.mainzisegred_nonseg_18425
    self.mainzisegred_small_18772_var = program.mainzisegred_small_18772
    self.constants = {}
    mainzicounter_mem_18647 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_19129 = opencl_alloc(self, 40, "static_mem_19129")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_19129,
                      normaliseArray(mainzicounter_mem_18647),
                      is_blocking=synchronous)
    self.mainzicounter_mem_18647 = static_mem_19129
    mainzihist_locks_mem_18757 = np.zeros(100151, dtype=np.int32)
    static_mem_19133 = opencl_alloc(self, 400604, "static_mem_19133")
    if (400604 != 0):
      cl.enqueue_copy(self.queue, static_mem_19133,
                      normaliseArray(mainzihist_locks_mem_18757),
                      is_blocking=synchronous)
    self.mainzihist_locks_mem_18757 = static_mem_19133
    mainzicounter_mem_18801 = np.zeros(10240, dtype=np.int32)
    static_mem_19136 = opencl_alloc(self, 40960, "static_mem_19136")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_19136,
                      normaliseArray(mainzicounter_mem_18801),
                      is_blocking=synchronous)
    self.mainzicounter_mem_18801 = static_mem_19136
    mainzicounter_mem_19099 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_19138 = opencl_alloc(self, 40, "static_mem_19138")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_19138,
                      normaliseArray(mainzicounter_mem_19099),
                      is_blocking=synchronous)
    self.mainzicounter_mem_19099 = static_mem_19138
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
  def futhark_builtinzhreplicate_f32(self, mem_19085, num_elems_19086,
                                     val_19087):
    group_sizze_19092 = self.sizes["builtin#replicate_f32.group_size_19092"]
    num_groups_19093 = sdiv_up64(num_elems_19086, group_sizze_19092)
    if ((1 * (np.long(num_groups_19093) * np.long(group_sizze_19092))) != 0):
      self.builtinzhreplicate_f32zireplicate_19089_var.set_args(mem_19085,
                                                                np.int32(num_elems_19086),
                                                                np.float32(val_19087))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f32zireplicate_19089_var,
                                 ((np.long(num_groups_19093) * np.long(group_sizze_19092)),),
                                 (np.long(group_sizze_19092),))
      if synchronous:
        sync(self)
    return ()
  def futhark_builtinzhreplicate_i64(self, mem_18677, num_elems_18678,
                                     val_18679):
    group_sizze_18684 = self.sizes["builtin#replicate_i64.group_size_18684"]
    num_groups_18685 = sdiv_up64(num_elems_18678, group_sizze_18684)
    if ((1 * (np.long(num_groups_18685) * np.long(group_sizze_18684))) != 0):
      self.builtinzhreplicate_i64zireplicate_18681_var.set_args(mem_18677,
                                                                np.int32(num_elems_18678),
                                                                np.int64(val_18679))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_i64zireplicate_18681_var,
                                 ((np.long(num_groups_18685) * np.long(group_sizze_18684)),),
                                 (np.long(group_sizze_18684),))
      if synchronous:
        sync(self)
    return ()
  def futhark_main(self, paths_17456, steps_17457, swap_term_17458,
                   payments_17459, notional_17460, a_17461, b_17462,
                   sigma_17463, r0_17464):
    res_17465 = sitofp_i64_f32(payments_17459)
    x_17466 = (swap_term_17458 * res_17465)
    res_17467 = sitofp_i64_f32(steps_17457)
    dt_17468 = (x_17466 / res_17467)
    sims_per_year_17469 = (res_17467 / x_17466)
    bounds_invalid_upwards_17470 = slt64(steps_17457, np.int64(1))
    valid_17471 = not(bounds_invalid_upwards_17470)
    range_valid_c_17472 = True
    assert valid_17471, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:75:56-67\n   #1  cva.fut:112:17-40\n   #2  cva.fut:105:1-139:18\n" % ("Range ",
                                                                                                                                                    np.int64(1),
                                                                                                                                                    "..",
                                                                                                                                                    np.int64(2),
                                                                                                                                                    "...",
                                                                                                                                                    steps_17457,
                                                                                                                                                    " is invalid."))
    bounds_invalid_upwards_17482 = slt64(paths_17456, np.int64(0))
    valid_17483 = not(bounds_invalid_upwards_17482)
    range_valid_c_17484 = True
    assert valid_17483, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:113:19-49\n   #4  cva.fut:105:1-139:18\n" % ("Range ",
                                                                                                                                                                                                                                                                np.int64(0),
                                                                                                                                                                                                                                                                "..",
                                                                                                                                                                                                                                                                np.int64(1),
                                                                                                                                                                                                                                                                "..<",
                                                                                                                                                                                                                                                                paths_17456,
                                                                                                                                                                                                                                                                " is invalid."))
    upper_bound_17487 = (steps_17457 - np.int64(1))
    res_17488 = futhark_sqrt32(dt_17468)
    segmap_group_sizze_18200 = self.sizes["main.segmap_group_size_18183"]
    segmap_usable_groups_18201 = sdiv_up64(paths_17456,
                                           segmap_group_sizze_18200)
    bytes_18467 = (np.int64(4) * paths_17456)
    mem_18468 = opencl_alloc(self, bytes_18467, "mem_18468")
    if ((1 * (np.long(segmap_usable_groups_18201) * np.long(segmap_group_sizze_18200))) != 0):
      self.mainzisegmap_18181_var.set_args(self.global_failure,
                                           np.int64(paths_17456), mem_18468)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18181_var,
                                 ((np.long(segmap_usable_groups_18201) * np.long(segmap_group_sizze_18200)),),
                                 (np.long(segmap_group_sizze_18200),))
      if synchronous:
        sync(self)
    nest_sizze_18224 = (paths_17456 * steps_17457)
    segmap_group_sizze_18225 = self.sizes["main.segmap_group_size_18120"]
    segmap_usable_groups_18226 = sdiv_up64(nest_sizze_18224,
                                           segmap_group_sizze_18225)
    bytes_18470 = (np.int64(4) * nest_sizze_18224)
    mem_18472 = opencl_alloc(self, bytes_18470, "mem_18472")
    if ((1 * (np.long(segmap_usable_groups_18226) * np.long(segmap_group_sizze_18225))) != 0):
      self.mainzisegmap_18117_var.set_args(self.global_failure,
                                           np.int64(paths_17456),
                                           np.int64(steps_17457), mem_18468,
                                           mem_18472)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18117_var,
                                 ((np.long(segmap_usable_groups_18226) * np.long(segmap_group_sizze_18225)),),
                                 (np.long(segmap_group_sizze_18225),))
      if synchronous:
        sync(self)
    mem_18468 = None
    segmap_group_sizze_18270 = self.sizes["main.segmap_group_size_18021"]
    max_num_groups_18579 = self.sizes["main.segmap_num_groups_18023"]
    num_groups_18271 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_17456,
                                                            sext_i32_i64(segmap_group_sizze_18270)),
                                                  sext_i32_i64(max_num_groups_18579))))
    mem_18475 = opencl_alloc(self, bytes_18470, "mem_18475")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_18475, np.int64(0),
                                                mem_18472, np.int64(0),
                                                np.int64(1), steps_17457,
                                                paths_17456)
    mem_18472 = None
    mem_18493 = opencl_alloc(self, bytes_18470, "mem_18493")
    bytes_18477 = (np.int64(4) * steps_17457)
    num_threads_18555 = (segmap_group_sizze_18270 * num_groups_18271)
    total_sizze_18556 = (bytes_18477 * num_threads_18555)
    mem_18478 = opencl_alloc(self, total_sizze_18556, "mem_18478")
    if ((1 * (np.long(num_groups_18271) * np.long(segmap_group_sizze_18270))) != 0):
      self.mainzisegmap_18019_var.set_args(self.global_failure,
                                           self.failure_is_an_option,
                                           self.global_failure_args,
                                           np.int64(paths_17456),
                                           np.int64(steps_17457),
                                           np.float32(a_17461),
                                           np.float32(b_17462),
                                           np.float32(sigma_17463),
                                           np.float32(r0_17464),
                                           np.float32(dt_17468),
                                           np.int64(upper_bound_17487),
                                           np.float32(res_17488),
                                           np.int64(num_groups_18271),
                                           mem_18475, mem_18478, mem_18493)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18019_var,
                                 ((np.long(num_groups_18271) * np.long(segmap_group_sizze_18270)),),
                                 (np.long(segmap_group_sizze_18270),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_18475 = None
    mem_18478 = None
    res_17557 = sitofp_i64_f32(paths_17456)
    x_17558 = fpow32(a_17461, np.float32(2.0))
    x_17559 = (b_17462 * x_17558)
    x_17560 = fpow32(sigma_17463, np.float32(2.0))
    y_17561 = (x_17560 / np.float32(2.0))
    y_17562 = (x_17559 - y_17561)
    y_17563 = (np.float32(4.0) * a_17461)
    mem_18495 = opencl_alloc(self, bytes_18477, "mem_18495")
    segscan_group_sizze_18303 = self.sizes["main.segscan_group_size_18302"]
    max_num_groups_18591 = self.sizes["main.segscan_num_groups_18304"]
    num_groups_18305 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_17456,
                                                            sext_i32_i64(segscan_group_sizze_18303)),
                                                  sext_i32_i64(max_num_groups_18591))))
    segred_group_sizze_18311 = self.sizes["main.segred_group_size_18310"]
    max_num_groups_18592 = self.sizes["main.segred_num_groups_18312"]
    num_groups_18313 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_17456,
                                                            sext_i32_i64(segred_group_sizze_18311)),
                                                  sext_i32_i64(max_num_groups_18592))))
    seghist_group_sizze_18321 = self.sizes["main.seghist_group_size_18320"]
    max_num_groups_18593 = self.sizes["main.seghist_num_groups_18322"]
    num_groups_18323 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_17456,
                                                            sext_i32_i64(seghist_group_sizze_18321)),
                                                  sext_i32_i64(max_num_groups_18593))))
    segscan_group_sizze_18336 = self.sizes["main.segscan_group_size_18335"]
    segscan_group_sizze_18344 = self.sizes["main.segscan_group_size_18343"]
    segscan_group_sizze_18352 = self.sizes["main.segscan_group_size_18351"]
    segscan_group_sizze_18404 = self.sizes["main.segscan_group_size_18403"]
    segred_group_sizze_18418 = self.sizes["main.segred_group_size_18417"]
    max_num_groups_18594 = self.sizes["main.segred_num_groups_18419"]
    num_groups_18420 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_17456,
                                                            sext_i32_i64(segred_group_sizze_18418)),
                                                  sext_i32_i64(max_num_groups_18594))))
    bytes_18504 = (np.int64(8) * paths_17456)
    mem_18505 = opencl_alloc(self, bytes_18504, "mem_18505")
    mem_18507 = opencl_alloc(self, bytes_18504, "mem_18507")
    mem_18510 = opencl_alloc(self, np.int64(8), "mem_18510")
    mem_18539 = opencl_alloc(self, np.int64(4), "mem_18539")
    redout_18297 = np.float32(0.0)
    i_18299 = np.int64(0)
    one_19141 = np.int64(1)
    for counter_19140 in range(steps_17457):
      index_primexp_18452 = (np.int64(1) + i_18299)
      res_17589 = sitofp_i64_f32(index_primexp_18452)
      res_17590 = (res_17589 / sims_per_year_17469)
      x_17599 = (res_17590 / swap_term_17458)
      ceil_arg_17600 = (x_17599 - np.float32(1.0))
      res_17601 = futhark_ceil32(ceil_arg_17600)
      res_17602 = fptosi_f32_i64(res_17601)
      res_17603 = (payments_17459 - res_17602)
      cond_17604 = (res_17603 == np.int64(0))
      if cond_17604:
        res_17605 = np.int64(1)
      else:
        res_17605 = res_17603
      if slt64(np.int64(0), paths_17456):
        stage1_max_num_groups_18597 = self.max_group_size
        stage1_num_groups_18598 = smin64(stage1_max_num_groups_18597,
                                         num_groups_18305)
        num_threads_18599 = sext_i64_i32((stage1_num_groups_18598 * segscan_group_sizze_18303))
        if ((1 * (np.long(stage1_num_groups_18598) * np.long(segscan_group_sizze_18303))) != 0):
          self.mainziscan_stage1_18308_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_18303)))),
                                                    np.int64(paths_17456),
                                                    np.int64(res_17605),
                                                    mem_18505, mem_18507,
                                                    np.int32(num_threads_18599))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_18308_var,
                                     ((np.long(stage1_num_groups_18598) * np.long(segscan_group_sizze_18303)),),
                                     (np.long(segscan_group_sizze_18303),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_18598))) != 0):
          self.mainziscan_stage2_18308_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_18598)))),
                                                    np.int64(paths_17456),
                                                    mem_18505,
                                                    np.int64(stage1_num_groups_18598),
                                                    np.int32(num_threads_18599))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_18308_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_18598)),),
                                     (np.long(stage1_num_groups_18598),))
          if synchronous:
            sync(self)
        required_groups_18635 = sext_i64_i32(sdiv_up64(paths_17456,
                                                       segscan_group_sizze_18303))
        if ((1 * (np.long(num_groups_18305) * np.long(segscan_group_sizze_18303))) != 0):
          self.mainziscan_stage3_18308_var.set_args(self.global_failure,
                                                    np.int64(paths_17456),
                                                    np.int64(num_groups_18305),
                                                    mem_18505,
                                                    np.int32(num_threads_18599),
                                                    np.int32(required_groups_18635))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_18308_var,
                                     ((np.long(num_groups_18305) * np.long(segscan_group_sizze_18303)),),
                                     (np.long(segscan_group_sizze_18303),))
          if synchronous:
            sync(self)
      mainzicounter_mem_18647 = self.mainzicounter_mem_18647
      group_res_arr_mem_18649 = opencl_alloc(self,
                                             (np.int32(8) * (segred_group_sizze_18311 * num_groups_18313)),
                                             "group_res_arr_mem_18649")
      num_threads_18651 = (num_groups_18313 * segred_group_sizze_18311)
      if ((1 * (np.long(num_groups_18313) * np.long(segred_group_sizze_18311))) != 0):
        self.mainzisegred_nonseg_18318_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(8) * segred_group_sizze_18311))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(paths_17456),
                                                    np.int64(num_groups_18313),
                                                    mem_18507, mem_18510,
                                                    mainzicounter_mem_18647,
                                                    group_res_arr_mem_18649,
                                                    np.int64(num_threads_18651))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_18318_var,
                                   ((np.long(num_groups_18313) * np.long(segred_group_sizze_18311)),),
                                   (np.long(segred_group_sizze_18311),))
        if synchronous:
          sync(self)
      read_res_19130 = np.empty(1, dtype=ct.c_int64)
      cl.enqueue_copy(self.queue, read_res_19130, mem_18510,
                      device_offset=(np.long(np.int64(0)) * 8),
                      is_blocking=synchronous)
      sync(self)
      res_17607 = read_res_19130[0]
      bounds_invalid_upwards_17612 = slt64(res_17607, np.int64(0))
      valid_17613 = not(bounds_invalid_upwards_17612)
      range_valid_c_17614 = True
      assert valid_17613, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:130:37-88\n   #6  cva.fut:129:18-132:79\n   #7  cva.fut:105:1-139:18\n" % ("Range ",
                                                                                                                                                                                                                                                                                                                                                                                                                                  np.int64(0),
                                                                                                                                                                                                                                                                                                                                                                                                                                  "..",
                                                                                                                                                                                                                                                                                                                                                                                                                                  np.int64(1),
                                                                                                                                                                                                                                                                                                                                                                                                                                  "..<",
                                                                                                                                                                                                                                                                                                                                                                                                                                  res_17607,
                                                                                                                                                                                                                                                                                                                                                                                                                                  " is invalid."))
      bytes_18511 = (np.int64(8) * res_17607)
      mem_18512 = opencl_alloc(self, bytes_18511, "mem_18512")
      self.futhark_builtinzhreplicate_i64(mem_18512, res_17607, np.int64(0))
      h_18689 = (np.int32(8) * res_17607)
      seg_h_18690 = (np.int32(8) * res_17607)
      if (seg_h_18690 == np.int64(0)):
        pass
      else:
        hist_H_18691 = res_17607
        hist_el_sizze_18692 = (sdiv_up64(h_18689, hist_H_18691) + np.int64(4))
        hist_N_18693 = paths_17456
        hist_RF_18694 = np.int64(1)
        hist_L_18695 = self.max_local_memory
        max_group_sizze_18696 = self.max_group_size
        num_groups_18697 = sdiv_up64(sext_i32_i64(sext_i64_i32((num_groups_18323 * seghist_group_sizze_18321))),
                                     max_group_sizze_18696)
        hist_m_prime_18698 = (sitofp_i64_f64(smin64(sext_i32_i64(squot32(hist_L_18695,
                                                                         hist_el_sizze_18692)),
                                                    sdiv_up64(hist_N_18693,
                                                              num_groups_18697))) / sitofp_i64_f64(hist_H_18691))
        hist_M0_18699 = smax64(np.int64(1),
                               smin64(fptosi_f64_i64(hist_m_prime_18698),
                                      max_group_sizze_18696))
        hist_Nout_18700 = np.int64(1)
        hist_Nin_18701 = paths_17456
        work_asymp_M_max_18702 = squot64((hist_Nout_18700 * hist_N_18693),
                                         ((np.int64(2) * num_groups_18697) * hist_H_18691))
        hist_M_18703 = sext_i64_i32(smin64(hist_M0_18699,
                                           work_asymp_M_max_18702))
        hist_C_18704 = sdiv_up64(max_group_sizze_18696,
                                 sext_i32_i64(smax32(np.int32(1),
                                                     hist_M_18703)))
        local_mem_needed_18705 = (hist_el_sizze_18692 * sext_i32_i64(hist_M_18703))
        hist_S_18706 = sext_i64_i32(sdiv_up64((hist_H_18691 * local_mem_needed_18705),
                                              hist_L_18695))
        if (sle64(hist_H_18691,
                  hist_Nin_18701) and (sle64(local_mem_needed_18705,
                                             hist_L_18695) and (sle32(hist_S_18706,
                                                                      np.int32(6)) and (sle64(hist_C_18704,
                                                                                              max_group_sizze_18696) and slt32(np.int32(0),
                                                                                                                               hist_M_18703))))):
          num_segments_18707 = np.int64(1)
          num_subhistos_18686 = (num_groups_18697 * num_segments_18707)
          if (num_subhistos_18686 == np.int64(1)):
            res_subhistos_mem_18687 = mem_18512
          else:
            res_subhistos_mem_18687 = opencl_alloc(self,
                                                   ((sext_i32_i64(num_subhistos_18686) * res_17607) * np.int32(8)),
                                                   "res_subhistos_mem_18687")
            self.futhark_builtinzhreplicate_i64(res_subhistos_mem_18687,
                                                (num_subhistos_18686 * res_17607),
                                                np.int64(0))
            if ((res_17607 * np.int32(8)) != 0):
              cl.enqueue_copy(self.queue, res_subhistos_mem_18687, mem_18512,
                              dest_offset=np.long(np.int64(0)),
                              src_offset=np.long(np.int64(0)),
                              byte_count=np.long((res_17607 * np.int32(8))))
            if synchronous:
              sync(self)
          chk_i_18708 = np.int32(0)
          one_19132 = np.int32(1)
          for counter_19131 in range(hist_S_18706):
            num_segments_18709 = np.int64(1)
            hist_H_chk_18710 = sdiv_up64(res_17607, sext_i32_i64(hist_S_18706))
            histo_sizze_18711 = hist_H_chk_18710
            init_per_thread_18712 = sext_i64_i32(sdiv_up64((sext_i32_i64(hist_M_18703) * histo_sizze_18711),
                                                           max_group_sizze_18696))
            if ((1 * (np.long(num_groups_18697) * np.long(max_group_sizze_18696))) != 0):
              self.mainziseghist_local_18326_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long((np.int32(4) * (hist_M_18703 * hist_H_chk_18710)))),
                                                          cl.LocalMemory(np.long((np.int32(8) * (hist_M_18703 * hist_H_chk_18710)))),
                                                          np.int64(paths_17456),
                                                          np.int64(res_17607),
                                                          mem_18505,
                                                          res_subhistos_mem_18687,
                                                          np.int32(max_group_sizze_18696),
                                                          np.int64(num_groups_18697),
                                                          np.int32(hist_M_18703),
                                                          np.int32(chk_i_18708),
                                                          np.int64(num_segments_18709),
                                                          np.int64(hist_H_chk_18710),
                                                          np.int64(histo_sizze_18711),
                                                          np.int32(init_per_thread_18712))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainziseghist_local_18326_var,
                                         ((np.long(num_groups_18697) * np.long(max_group_sizze_18696)),),
                                         (np.long(max_group_sizze_18696),))
              if synchronous:
                sync(self)
            chk_i_18708 += one_19132
        else:
          hist_H_18744 = res_17607
          hist_RF_18745 = ((np.float64(0.0) + sitofp_i32_f64(np.int64(1))) / np.float64(1.0))
          hist_el_sizze_18746 = squot32(sext_i64_i32((np.int32(4) + np.int32(8))),
                                        np.int32(2))
          hist_C_max_18747 = fmin64(sitofp_i32_f64(sext_i64_i32((num_groups_18323 * seghist_group_sizze_18321))),
                                    (sitofp_i32_f64(hist_H_18744) / np.float64(2.0)))
          hist_M_min_18748 = smax32(np.int32(1),
                                    sext_i64_i32(fptosi_f64_i64((sitofp_i32_f64(sext_i64_i32((num_groups_18323 * seghist_group_sizze_18321))) / hist_C_max_18747))))
          L2_sizze_18749 = self.sizes["main.L2_size_18749"]
          hist_RACE_exp_18750 = fmax64(np.float64(1.0),
                                       ((np.float64(0.75) * hist_RF_18745) / (np.float64(64.0) / sitofp_i32_f64(hist_el_sizze_18746))))
          if slt64(paths_17456, hist_H_18744):
            hist_S_18751 = np.int32(1)
          else:
            hist_S_18751 = sext_i64_i32(sdiv_up64(((sext_i32_i64(hist_M_min_18748) * hist_H_18744) * sext_i32_i64(hist_el_sizze_18746)),
                                                  fptosi_f64_i64(((np.float64(0.4) * sitofp_i32_f64(L2_sizze_18749)) * hist_RACE_exp_18750))))
          hist_H_chk_18752 = sdiv_up64(res_17607, sext_i32_i64(hist_S_18751))
          hist_k_max_18753 = (fmin64(((np.float64(0.4) * (sitofp_i32_f64(L2_sizze_18749) / sitofp_i32_f64(sext_i64_i32((np.int32(4) + np.int32(8)))))) * hist_RACE_exp_18750),
                                     sitofp_i32_f64(paths_17456)) / sitofp_i32_f64(sext_i64_i32((num_groups_18323 * seghist_group_sizze_18321))))
          hist_u_18754 = np.int64(1)
          hist_C_18755 = fmin64(sitofp_i32_f64(sext_i64_i32((num_groups_18323 * seghist_group_sizze_18321))),
                                (sitofp_i32_f64((hist_u_18754 * hist_H_chk_18752)) / hist_k_max_18753))
          hist_M_18756 = smax32(hist_M_min_18748,
                                sext_i64_i32(fptosi_f64_i64((sitofp_i32_f64(sext_i64_i32((num_groups_18323 * seghist_group_sizze_18321))) / hist_C_18755))))
          num_subhistos_18686 = sext_i32_i64(hist_M_18756)
          if (hist_M_18756 == np.int32(1)):
            res_subhistos_mem_18687 = mem_18512
          else:
            if (num_subhistos_18686 == np.int64(1)):
              res_subhistos_mem_18687 = mem_18512
            else:
              res_subhistos_mem_18687 = opencl_alloc(self,
                                                     ((sext_i32_i64(num_subhistos_18686) * res_17607) * np.int32(8)),
                                                     "res_subhistos_mem_18687")
              self.futhark_builtinzhreplicate_i64(res_subhistos_mem_18687,
                                                  (num_subhistos_18686 * res_17607),
                                                  np.int64(0))
              if ((res_17607 * np.int32(8)) != 0):
                cl.enqueue_copy(self.queue, res_subhistos_mem_18687, mem_18512,
                                dest_offset=np.long(np.int64(0)),
                                src_offset=np.long(np.int64(0)),
                                byte_count=np.long((res_17607 * np.int32(8))))
              if synchronous:
                sync(self)
          mainzihist_locks_mem_18757 = self.mainzihist_locks_mem_18757
          chk_i_18759 = np.int32(0)
          one_19135 = np.int32(1)
          for counter_19134 in range(hist_S_18751):
            hist_H_chk_18760 = sdiv_up64(res_17607, sext_i32_i64(hist_S_18751))
            if ((1 * (np.long(num_groups_18323) * np.long(seghist_group_sizze_18321))) != 0):
              self.mainziseghist_global_18326_var.set_args(self.global_failure,
                                                           np.int64(paths_17456),
                                                           np.int64(res_17607),
                                                           np.int64(num_groups_18323),
                                                           mem_18505,
                                                           np.int32(num_subhistos_18686),
                                                           res_subhistos_mem_18687,
                                                           mainzihist_locks_mem_18757,
                                                           np.int32(chk_i_18759),
                                                           np.int64(hist_H_chk_18760))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainziseghist_global_18326_var,
                                         ((np.long(num_groups_18323) * np.long(seghist_group_sizze_18321)),),
                                         (np.long(seghist_group_sizze_18321),))
              if synchronous:
                sync(self)
            chk_i_18759 += one_19135
        if (num_subhistos_18686 == np.int64(1)):
          mem_18512 = res_subhistos_mem_18687
        else:
          if slt64((num_subhistos_18686 * np.int64(2)),
                   seghist_group_sizze_18321):
            segment_sizze_nonzzero_18773 = smax64(np.int64(1),
                                                  num_subhistos_18686)
            num_threads_18774 = (num_groups_18323 * seghist_group_sizze_18321)
            if ((1 * (np.long(num_groups_18323) * np.long(seghist_group_sizze_18321))) != 0):
              self.mainzisegred_small_18772_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long((np.int32(8) * seghist_group_sizze_18321))),
                                                         np.int64(res_17607),
                                                         np.int64(num_groups_18323),
                                                         mem_18512,
                                                         np.int32(num_subhistos_18686),
                                                         res_subhistos_mem_18687,
                                                         np.int64(segment_sizze_nonzzero_18773))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_small_18772_var,
                                         ((np.long(num_groups_18323) * np.long(seghist_group_sizze_18321)),),
                                         (np.long(seghist_group_sizze_18321),))
              if synchronous:
                sync(self)
          else:
            groups_per_segment_18794 = sdiv_up64(num_groups_18323,
                                                 smax64(np.int64(1), res_17607))
            elements_per_thread_18795 = sdiv_up64(num_subhistos_18686,
                                                  (seghist_group_sizze_18321 * groups_per_segment_18794))
            virt_num_groups_18796 = (groups_per_segment_18794 * res_17607)
            num_threads_18797 = (num_groups_18323 * seghist_group_sizze_18321)
            threads_per_segment_18798 = (groups_per_segment_18794 * seghist_group_sizze_18321)
            group_res_arr_mem_18799 = opencl_alloc(self,
                                                   (np.int32(8) * (seghist_group_sizze_18321 * virt_num_groups_18796)),
                                                   "group_res_arr_mem_18799")
            mainzicounter_mem_18801 = self.mainzicounter_mem_18801
            if ((1 * (np.long(num_groups_18323) * np.long(seghist_group_sizze_18321))) != 0):
              self.mainzisegred_large_18772_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long(np.int32(1))),
                                                         cl.LocalMemory(np.long((np.int32(8) * seghist_group_sizze_18321))),
                                                         np.int64(res_17607),
                                                         np.int64(num_groups_18323),
                                                         mem_18512,
                                                         np.int32(num_subhistos_18686),
                                                         res_subhistos_mem_18687,
                                                         np.int64(groups_per_segment_18794),
                                                         np.int64(elements_per_thread_18795),
                                                         np.int64(virt_num_groups_18796),
                                                         np.int64(threads_per_segment_18798),
                                                         group_res_arr_mem_18799,
                                                         mainzicounter_mem_18801)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_large_18772_var,
                                         ((np.long(num_groups_18323) * np.long(seghist_group_sizze_18321)),),
                                         (np.long(seghist_group_sizze_18321),))
              if synchronous:
                sync(self)
      max_num_groups_18833 = self.sizes["main.segscan_num_groups_18337"]
      num_groups_18338 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_17607,
                                                              sext_i32_i64(segscan_group_sizze_18336)),
                                                    sext_i32_i64(max_num_groups_18833))))
      mem_18516 = opencl_alloc(self, res_17607, "mem_18516")
      mem_18518 = opencl_alloc(self, bytes_18511, "mem_18518")
      if slt64(np.int64(0), res_17607):
        stage1_max_num_groups_18834 = self.max_group_size
        stage1_num_groups_18835 = smin64(stage1_max_num_groups_18834,
                                         num_groups_18338)
        num_threads_18836 = sext_i64_i32((stage1_num_groups_18835 * segscan_group_sizze_18336))
        if ((1 * (np.long(stage1_num_groups_18835) * np.long(segscan_group_sizze_18336))) != 0):
          self.mainziscan_stage1_18341_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_18336)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_18336)))),
                                                    np.int64(res_17607),
                                                    mem_18512, mem_18516,
                                                    mem_18518,
                                                    np.int32(num_threads_18836))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_18341_var,
                                     ((np.long(stage1_num_groups_18835) * np.long(segscan_group_sizze_18336)),),
                                     (np.long(segscan_group_sizze_18336),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_18835))) != 0):
          self.mainziscan_stage2_18341_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_18835)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_18835)))),
                                                    np.int64(res_17607),
                                                    mem_18516, mem_18518,
                                                    np.int64(stage1_num_groups_18835),
                                                    np.int32(num_threads_18836))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_18341_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_18835)),),
                                     (np.long(stage1_num_groups_18835),))
          if synchronous:
            sync(self)
        required_groups_18888 = sext_i64_i32(sdiv_up64(res_17607,
                                                       segscan_group_sizze_18336))
        if ((1 * (np.long(num_groups_18338) * np.long(segscan_group_sizze_18336))) != 0):
          self.mainziscan_stage3_18341_var.set_args(self.global_failure,
                                                    np.int64(res_17607),
                                                    np.int64(num_groups_18338),
                                                    mem_18516, mem_18518,
                                                    np.int32(num_threads_18836),
                                                    np.int32(required_groups_18888))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_18341_var,
                                     ((np.long(num_groups_18338) * np.long(segscan_group_sizze_18336)),),
                                     (np.long(segscan_group_sizze_18336),))
          if synchronous:
            sync(self)
      mem_18512 = None
      mem_18516 = None
      max_num_groups_18900 = self.sizes["main.segscan_num_groups_18345"]
      num_groups_18346 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_17607,
                                                              sext_i32_i64(segscan_group_sizze_18344)),
                                                    sext_i32_i64(max_num_groups_18900))))
      mem_18521 = opencl_alloc(self, res_17607, "mem_18521")
      mem_18523 = opencl_alloc(self, bytes_18511, "mem_18523")
      mem_18525 = opencl_alloc(self, res_17607, "mem_18525")
      if slt64(np.int64(0), res_17607):
        stage1_max_num_groups_18901 = self.max_group_size
        stage1_num_groups_18902 = smin64(stage1_max_num_groups_18901,
                                         num_groups_18346)
        num_threads_18903 = sext_i64_i32((stage1_num_groups_18902 * segscan_group_sizze_18344))
        if ((1 * (np.long(stage1_num_groups_18902) * np.long(segscan_group_sizze_18344))) != 0):
          self.mainziscan_stage1_18349_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_18344)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_18344)))),
                                                    np.int64(res_17607),
                                                    mem_18518, mem_18521,
                                                    mem_18523, mem_18525,
                                                    np.int32(num_threads_18903))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_18349_var,
                                     ((np.long(stage1_num_groups_18902) * np.long(segscan_group_sizze_18344)),),
                                     (np.long(segscan_group_sizze_18344),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_18902))) != 0):
          self.mainziscan_stage2_18349_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_18902)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_18902)))),
                                                    np.int64(res_17607),
                                                    mem_18521, mem_18523,
                                                    np.int64(stage1_num_groups_18902),
                                                    np.int32(num_threads_18903))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_18349_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_18902)),),
                                     (np.long(stage1_num_groups_18902),))
          if synchronous:
            sync(self)
        required_groups_18955 = sext_i64_i32(sdiv_up64(res_17607,
                                                       segscan_group_sizze_18344))
        if ((1 * (np.long(num_groups_18346) * np.long(segscan_group_sizze_18344))) != 0):
          self.mainziscan_stage3_18349_var.set_args(self.global_failure,
                                                    np.int64(res_17607),
                                                    np.int64(num_groups_18346),
                                                    mem_18521, mem_18523,
                                                    np.int32(num_threads_18903),
                                                    np.int32(required_groups_18955))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_18349_var,
                                     ((np.long(num_groups_18346) * np.long(segscan_group_sizze_18344)),),
                                     (np.long(segscan_group_sizze_18344),))
          if synchronous:
            sync(self)
      mem_18521 = None
      max_num_groups_18967 = self.sizes["main.segscan_num_groups_18353"]
      num_groups_18354 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_17607,
                                                              sext_i32_i64(segscan_group_sizze_18352)),
                                                    sext_i32_i64(max_num_groups_18967))))
      mem_18528 = opencl_alloc(self, res_17607, "mem_18528")
      bytes_18529 = (np.int64(4) * res_17607)
      mem_18530 = opencl_alloc(self, bytes_18529, "mem_18530")
      if slt64(np.int64(0), res_17607):
        stage1_max_num_groups_18968 = self.max_group_size
        stage1_num_groups_18969 = smin64(stage1_max_num_groups_18968,
                                         num_groups_18354)
        num_threads_18970 = sext_i64_i32((stage1_num_groups_18969 * segscan_group_sizze_18352))
        if ((1 * (np.long(stage1_num_groups_18969) * np.long(segscan_group_sizze_18352))) != 0):
          self.mainziscan_stage1_18357_var.set_args(self.global_failure,
                                                    self.failure_is_an_option,
                                                    self.global_failure_args,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * segscan_group_sizze_18352)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_18352)))),
                                                    np.int64(paths_17456),
                                                    np.float32(swap_term_17458),
                                                    np.int64(payments_17459),
                                                    np.float32(notional_17460),
                                                    np.float32(a_17461),
                                                    np.float32(b_17462),
                                                    np.float32(sigma_17463),
                                                    np.float32(res_17590),
                                                    np.int64(res_17607),
                                                    np.int64(i_18299),
                                                    mem_18493, mem_18518,
                                                    mem_18523, mem_18525,
                                                    mem_18528, mem_18530,
                                                    np.int32(num_threads_18970))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_18357_var,
                                     ((np.long(stage1_num_groups_18969) * np.long(segscan_group_sizze_18352)),),
                                     (np.long(segscan_group_sizze_18352),))
          if synchronous:
            sync(self)
        self.failure_is_an_option = np.int32(1)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_18969))) != 0):
          self.mainziscan_stage2_18357_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * stage1_num_groups_18969)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_18969)))),
                                                    np.int64(res_17607),
                                                    mem_18528, mem_18530,
                                                    np.int64(stage1_num_groups_18969),
                                                    np.int32(num_threads_18970))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_18357_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_18969)),),
                                     (np.long(stage1_num_groups_18969),))
          if synchronous:
            sync(self)
        required_groups_19022 = sext_i64_i32(sdiv_up64(res_17607,
                                                       segscan_group_sizze_18352))
        if ((1 * (np.long(num_groups_18354) * np.long(segscan_group_sizze_18352))) != 0):
          self.mainziscan_stage3_18357_var.set_args(self.global_failure,
                                                    np.int64(res_17607),
                                                    np.int64(num_groups_18354),
                                                    mem_18528, mem_18530,
                                                    np.int32(num_threads_18970),
                                                    np.int32(required_groups_19022))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_18357_var,
                                     ((np.long(num_groups_18354) * np.long(segscan_group_sizze_18352)),),
                                     (np.long(segscan_group_sizze_18352),))
          if synchronous:
            sync(self)
      mem_18518 = None
      mem_18523 = None
      mem_18528 = None
      max_num_groups_19034 = self.sizes["main.segscan_num_groups_18405"]
      num_groups_18406 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_17607,
                                                              sext_i32_i64(segscan_group_sizze_18404)),
                                                    sext_i32_i64(max_num_groups_19034))))
      mem_18533 = opencl_alloc(self, bytes_18511, "mem_18533")
      if slt64(np.int64(0), res_17607):
        stage1_max_num_groups_19035 = self.max_group_size
        stage1_num_groups_19036 = smin64(stage1_max_num_groups_19035,
                                         num_groups_18406)
        num_threads_19037 = sext_i64_i32((stage1_num_groups_19036 * segscan_group_sizze_18404))
        if ((1 * (np.long(stage1_num_groups_19036) * np.long(segscan_group_sizze_18404))) != 0):
          self.mainziscan_stage1_18409_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_18404)))),
                                                    np.int64(res_17607),
                                                    mem_18525, mem_18533,
                                                    np.int32(num_threads_19037))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_18409_var,
                                     ((np.long(stage1_num_groups_19036) * np.long(segscan_group_sizze_18404)),),
                                     (np.long(segscan_group_sizze_18404),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_19036))) != 0):
          self.mainziscan_stage2_18409_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_19036)))),
                                                    np.int64(res_17607),
                                                    mem_18533,
                                                    np.int64(stage1_num_groups_19036),
                                                    np.int32(num_threads_19037))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_18409_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_19036)),),
                                     (np.long(stage1_num_groups_19036),))
          if synchronous:
            sync(self)
        required_groups_19073 = sext_i64_i32(sdiv_up64(res_17607,
                                                       segscan_group_sizze_18404))
        if ((1 * (np.long(num_groups_18406) * np.long(segscan_group_sizze_18404))) != 0):
          self.mainziscan_stage3_18409_var.set_args(self.global_failure,
                                                    np.int64(res_17607),
                                                    np.int64(num_groups_18406),
                                                    mem_18533,
                                                    np.int32(num_threads_19037),
                                                    np.int32(required_groups_19073))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_18409_var,
                                     ((np.long(num_groups_18406) * np.long(segscan_group_sizze_18404)),),
                                     (np.long(segscan_group_sizze_18404),))
          if synchronous:
            sync(self)
      cond_17798 = slt64(np.int64(0), res_17607)
      if cond_17798:
        i_17800 = (res_17607 - np.int64(1))
        x_17801 = sle64(np.int64(0), i_17800)
        y_17802 = slt64(i_17800, res_17607)
        bounds_check_17803 = (x_17801 and y_17802)
        index_certs_17804 = True
        assert bounds_check_17803, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:130:37-88\n   #5  cva.fut:129:18-132:79\n   #6  cva.fut:105:1-139:18\n" % ("Index [",
                                                                                                                                                                                                                                                                                                                                                                                                   i_17800,
                                                                                                                                                                                                                                                                                                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                                                                                                                                                   res_17607,
                                                                                                                                                                                                                                                                                                                                                                                                   "]."))
        read_res_19137 = np.empty(1, dtype=ct.c_int64)
        cl.enqueue_copy(self.queue, read_res_19137, mem_18533,
                        device_offset=(np.long(i_17800) * 8),
                        is_blocking=synchronous)
        sync(self)
        res_17805 = read_res_19137[0]
        num_segments_17799 = res_17805
      else:
        num_segments_17799 = np.int64(0)
      bounds_invalid_upwards_17806 = slt64(num_segments_17799, np.int64(0))
      valid_17807 = not(bounds_invalid_upwards_17806)
      range_valid_c_17808 = True
      assert valid_17807, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:130:37-88\n   #6  cva.fut:129:18-132:79\n   #7  cva.fut:105:1-139:18\n" % ("Range ",
                                                                                                                                                                                                                                                                                                                                                                                                                                 np.int64(0),
                                                                                                                                                                                                                                                                                                                                                                                                                                 "..",
                                                                                                                                                                                                                                                                                                                                                                                                                                 np.int64(1),
                                                                                                                                                                                                                                                                                                                                                                                                                                 "..<",
                                                                                                                                                                                                                                                                                                                                                                                                                                 num_segments_17799,
                                                                                                                                                                                                                                                                                                                                                                                                                                 " is invalid."))
      bytes_18534 = (np.int64(4) * num_segments_17799)
      mem_18535 = opencl_alloc(self, bytes_18534, "mem_18535")
      self.futhark_builtinzhreplicate_f32(mem_18535, num_segments_17799,
                                          np.float32(0.0))
      segmap_group_sizze_18414 = self.sizes["main.segmap_group_size_18413"]
      segmap_usable_groups_18415 = sdiv_up64(res_17607,
                                             segmap_group_sizze_18414)
      if ((1 * (np.long(segmap_usable_groups_18415) * np.long(segmap_group_sizze_18414))) != 0):
        self.mainzisegmap_18411_var.set_args(self.global_failure,
                                             np.int64(res_17607),
                                             np.int64(num_segments_17799),
                                             mem_18525, mem_18530, mem_18533,
                                             mem_18535)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18411_var,
                                   ((np.long(segmap_usable_groups_18415) * np.long(segmap_group_sizze_18414)),),
                                   (np.long(segmap_group_sizze_18414),))
        if synchronous:
          sync(self)
      mem_18525 = None
      mem_18530 = None
      mem_18533 = None
      dim_match_17816 = (paths_17456 == num_segments_17799)
      empty_or_match_cert_17817 = True
      assert dim_match_17816, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:130:37-88\n   #2  cva.fut:129:18-132:79\n   #3  cva.fut:105:1-139:18\n" % ("Value of (core language) shape (",
                                                                                                                                                                                                                           num_segments_17799,
                                                                                                                                                                                                                           ") cannot match shape of type `[",
                                                                                                                                                                                                                           paths_17456,
                                                                                                                                                                                                                           "]b`."))
      mainzicounter_mem_19099 = self.mainzicounter_mem_19099
      group_res_arr_mem_19101 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_18418 * num_groups_18420)),
                                             "group_res_arr_mem_19101")
      num_threads_19103 = (num_groups_18420 * segred_group_sizze_18418)
      if ((1 * (np.long(num_groups_18420) * np.long(segred_group_sizze_18418))) != 0):
        self.mainzisegred_nonseg_18425_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_18418))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(paths_17456),
                                                    np.int64(num_groups_18420),
                                                    mem_18535, mem_18539,
                                                    mainzicounter_mem_19099,
                                                    group_res_arr_mem_19101,
                                                    np.int64(num_threads_19103))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_18425_var,
                                   ((np.long(num_groups_18420) * np.long(segred_group_sizze_18418)),),
                                   (np.long(segred_group_sizze_18418),))
        if synchronous:
          sync(self)
      mem_18535 = None
      read_res_19139 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_19139, mem_18539,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_17819 = read_res_19139[0]
      res_17825 = (res_17819 / res_17557)
      negate_arg_17826 = (a_17461 * res_17590)
      exp_arg_17827 = (np.float32(0.0) - negate_arg_17826)
      res_17828 = fpow32(np.float32(2.7182817459106445), exp_arg_17827)
      x_17829 = (np.float32(1.0) - res_17828)
      B_17830 = (x_17829 / a_17461)
      x_17831 = (B_17830 - res_17590)
      x_17832 = (y_17562 * x_17831)
      A1_17833 = (x_17832 / x_17558)
      y_17834 = fpow32(B_17830, np.float32(2.0))
      x_17835 = (x_17560 * y_17834)
      A2_17836 = (x_17835 / y_17563)
      exp_arg_17837 = (A1_17833 - A2_17836)
      res_17838 = fpow32(np.float32(2.7182817459106445), exp_arg_17837)
      negate_arg_17839 = (np.float32(5.000000074505806e-2) * B_17830)
      exp_arg_17840 = (np.float32(0.0) - negate_arg_17839)
      res_17841 = fpow32(np.float32(2.7182817459106445), exp_arg_17840)
      res_17842 = (res_17838 * res_17841)
      res_17843 = (res_17825 * res_17842)
      res_17578 = (res_17843 + redout_18297)
      cl.enqueue_copy(self.queue, mem_18495, np.array(res_17843,
                                                      dtype=ct.c_float),
                      device_offset=(np.long(i_18299) * 4),
                      is_blocking=synchronous)
      redout_tmp_18595 = res_17578
      redout_18297 = redout_tmp_18595
      i_18299 += one_19141
    res_17574 = redout_18297
    mem_18493 = None
    mem_18505 = None
    mem_18507 = None
    mem_18510 = None
    mem_18539 = None
    CVA_17844 = (np.float32(6.000000052154064e-3) * res_17574)
    mem_18546 = opencl_alloc(self, bytes_18477, "mem_18546")
    if ((steps_17457 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_18546, mem_18495,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((steps_17457 * np.int32(4))))
    if synchronous:
      sync(self)
    mem_18495 = None
    out_arrsizze_18568 = steps_17457
    out_mem_18567 = mem_18546
    scalar_out_18566 = CVA_17844
    return (scalar_out_18566, out_mem_18567, out_arrsizze_18568)
  def main(self, paths_17456_ext, steps_17457_ext, swap_term_17458_ext,
           payments_17459_ext, notional_17460_ext, a_17461_ext, b_17462_ext,
           sigma_17463_ext, r0_17464_ext):
    try:
      paths_17456 = np.int64(ct.c_int64(paths_17456_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(paths_17456_ext),
                                                                                                                            paths_17456_ext))
    try:
      steps_17457 = np.int64(ct.c_int64(steps_17457_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(steps_17457_ext),
                                                                                                                            steps_17457_ext))
    try:
      swap_term_17458 = np.float32(ct.c_float(swap_term_17458_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(swap_term_17458_ext),
                                                                                                                            swap_term_17458_ext))
    try:
      payments_17459 = np.int64(ct.c_int64(payments_17459_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(payments_17459_ext),
                                                                                                                            payments_17459_ext))
    try:
      notional_17460 = np.float32(ct.c_float(notional_17460_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(notional_17460_ext),
                                                                                                                            notional_17460_ext))
    try:
      a_17461 = np.float32(ct.c_float(a_17461_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_17461_ext),
                                                                                                                            a_17461_ext))
    try:
      b_17462 = np.float32(ct.c_float(b_17462_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_17462_ext),
                                                                                                                            b_17462_ext))
    try:
      sigma_17463 = np.float32(ct.c_float(sigma_17463_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_17463_ext),
                                                                                                                            sigma_17463_ext))
    try:
      r0_17464 = np.float32(ct.c_float(r0_17464_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_17464_ext),
                                                                                                                            r0_17464_ext))
    (scalar_out_18566, out_mem_18567,
     out_arrsizze_18568) = self.futhark_main(paths_17456, steps_17457,
                                             swap_term_17458, payments_17459,
                                             notional_17460, a_17461, b_17462,
                                             sigma_17463, r0_17464)
    sync(self)
    return (np.float32(scalar_out_18566), cl.array.Array(self.queue,
                                                         (out_arrsizze_18568,),
                                                         ct.c_float,
                                                         data=out_mem_18567))