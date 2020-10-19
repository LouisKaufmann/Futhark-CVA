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




__kernel void builtinzhreplicate_f32zireplicate_20946(__global
                                                      unsigned char *mem_20942,
                                                      int32_t num_elems_20943,
                                                      float val_20944)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_20946;
    int32_t replicate_ltid_20947;
    int32_t replicate_gid_20948;
    
    replicate_gtid_20946 = get_global_id(0);
    replicate_ltid_20947 = get_local_id(0);
    replicate_gid_20948 = get_group_id(0);
    if (slt64(replicate_gtid_20946, num_elems_20943)) {
        ((__global float *) mem_20942)[sext_i32_i64(replicate_gtid_20946)] =
            val_20944;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i64zireplicate_20538(__global
                                                      unsigned char *mem_20534,
                                                      int32_t num_elems_20535,
                                                      int64_t val_20536)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_20538;
    int32_t replicate_ltid_20539;
    int32_t replicate_gid_20540;
    
    replicate_gtid_20538 = get_global_id(0);
    replicate_ltid_20539 = get_local_id(0);
    replicate_gid_20540 = get_group_id(0);
    if (slt64(replicate_gtid_20538, num_elems_20535)) {
        ((__global int64_t *) mem_20534)[sext_i32_i64(replicate_gtid_20538)] =
            val_20536;
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
__kernel void mainziscan_stage1_19907(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20462_backing_aligned_0,
                                      int64_t paths_18599, int64_t res_18843,
                                      __global unsigned char *mem_20191,
                                      __global unsigned char *mem_20193,
                                      int32_t num_threads_20456)
{
    #define segscan_group_sizze_19902 (mainzisegscan_group_sizze_19901)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20462_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20462_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20457;
    int32_t local_tid_20458;
    int64_t group_sizze_20461;
    int32_t wave_sizze_20460;
    int32_t group_tid_20459;
    
    global_tid_20457 = get_global_id(0);
    local_tid_20458 = get_local_id(0);
    group_sizze_20461 = get_local_size(0);
    wave_sizze_20460 = LOCKSTEP_WIDTH;
    group_tid_20459 = get_group_id(0);
    
    int32_t phys_tid_19907;
    
    phys_tid_19907 = global_tid_20457;
    
    __local char *scan_arr_mem_20462;
    
    scan_arr_mem_20462 = (__local char *) scan_arr_mem_20462_backing_0;
    
    int64_t x_18830;
    int64_t x_18831;
    
    x_18830 = 0;
    for (int64_t j_20464 = 0; j_20464 < sdiv_up64(paths_18599,
                                                  sext_i32_i64(num_threads_20456));
         j_20464++) {
        int64_t chunk_offset_20465 = segscan_group_sizze_19902 * j_20464 +
                sext_i32_i64(group_tid_20459) * (segscan_group_sizze_19902 *
                                                 sdiv_up64(paths_18599,
                                                           sext_i32_i64(num_threads_20456)));
        int64_t flat_idx_20466 = chunk_offset_20465 +
                sext_i32_i64(local_tid_20458);
        int64_t gtid_19906 = flat_idx_20466;
        
        // threads in bounds read input
        {
            if (slt64(gtid_19906, paths_18599)) {
                // write to-scan values to parameters
                {
                    x_18831 = res_18843;
                }
                // write mapped values results to global memory
                {
                    ((__global int64_t *) mem_20193)[gtid_19906] = res_18843;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_19906, paths_18599)) {
                    x_18831 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int64_t res_18832 = add64(x_18830, x_18831);
                
                ((__local
                  int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)] =
                    res_18832;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int64_t x_20467;
            int64_t x_20468;
            int64_t x_20470;
            int64_t x_20471;
            bool ltid_in_bounds_20473;
            
            ltid_in_bounds_20473 = slt64(sext_i32_i64(local_tid_20458),
                                         segscan_group_sizze_19902);
            
            int32_t skip_threads_20474;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_20473) {
                    x_20468 = ((volatile __local
                                int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)];
                    if ((local_tid_20458 - squot32(local_tid_20458, 32) * 32) ==
                        0) {
                        x_20467 = x_20468;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_20474 = 1;
                while (slt32(skip_threads_20474, 32)) {
                    if (sle32(skip_threads_20474, local_tid_20458 -
                              squot32(local_tid_20458, 32) * 32) &&
                        ltid_in_bounds_20473) {
                        // read operands
                        {
                            x_20467 = ((volatile __local
                                        int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458) -
                                                                       sext_i32_i64(skip_threads_20474)];
                        }
                        // perform operation
                        {
                            int64_t res_20469 = add64(x_20467, x_20468);
                            
                            x_20467 = res_20469;
                        }
                    }
                    if (sle32(wave_sizze_20460, skip_threads_20474)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_20474, local_tid_20458 -
                              squot32(local_tid_20458, 32) * 32) &&
                        ltid_in_bounds_20473) {
                        // write result
                        {
                            ((volatile __local
                              int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)] =
                                x_20467;
                            x_20468 = x_20467;
                        }
                    }
                    if (sle32(wave_sizze_20460, skip_threads_20474)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_20474 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_20458 - squot32(local_tid_20458, 32) * 32) ==
                    31 && ltid_in_bounds_20473) {
                    ((volatile __local
                      int64_t *) scan_arr_mem_20462)[sext_i32_i64(squot32(local_tid_20458,
                                                                          32))] =
                        x_20467;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_20475;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_20458, 32) == 0 &&
                        ltid_in_bounds_20473) {
                        x_20471 = ((volatile __local
                                    int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)];
                        if ((local_tid_20458 - squot32(local_tid_20458, 32) *
                             32) == 0) {
                            x_20470 = x_20471;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_20475 = 1;
                    while (slt32(skip_threads_20475, 32)) {
                        if (sle32(skip_threads_20475, local_tid_20458 -
                                  squot32(local_tid_20458, 32) * 32) &&
                            (squot32(local_tid_20458, 32) == 0 &&
                             ltid_in_bounds_20473)) {
                            // read operands
                            {
                                x_20470 = ((volatile __local
                                            int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458) -
                                                                           sext_i32_i64(skip_threads_20475)];
                            }
                            // perform operation
                            {
                                int64_t res_20472 = add64(x_20470, x_20471);
                                
                                x_20470 = res_20472;
                            }
                        }
                        if (sle32(wave_sizze_20460, skip_threads_20475)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_20475, local_tid_20458 -
                                  squot32(local_tid_20458, 32) * 32) &&
                            (squot32(local_tid_20458, 32) == 0 &&
                             ltid_in_bounds_20473)) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)] =
                                    x_20470;
                                x_20471 = x_20470;
                            }
                        }
                        if (sle32(wave_sizze_20460, skip_threads_20475)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_20475 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_20458, 32) == 0 ||
                      !ltid_in_bounds_20473)) {
                    // read operands
                    {
                        x_20468 = x_20467;
                        x_20467 = ((__local
                                    int64_t *) scan_arr_mem_20462)[sext_i32_i64(squot32(local_tid_20458,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        int64_t res_20469 = add64(x_20467, x_20468);
                        
                        x_20467 = res_20469;
                    }
                    // write final result
                    {
                        ((__local
                          int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)] =
                            x_20467;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_20458, 32) == 0) {
                    ((__local
                      int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)] =
                        x_20468;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_19906, paths_18599)) {
                    ((__global int64_t *) mem_20191)[gtid_19906] = ((__local
                                                                     int64_t *) scan_arr_mem_20462)[sext_i32_i64(local_tid_20458)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_20476 = 0;
                bool should_load_carry_20477 = local_tid_20458 == 0 &&
                     !crosses_segment_20476;
                
                if (should_load_carry_20477) {
                    x_18830 = ((__local
                                int64_t *) scan_arr_mem_20462)[segscan_group_sizze_19902 -
                                                               1];
                }
                if (!should_load_carry_20477) {
                    x_18830 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_19902
}
__kernel void mainziscan_stage1_19940(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20701_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_20699_backing_aligned_1,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20198, __global
                                      unsigned char *mem_20202, __global
                                      unsigned char *mem_20204,
                                      int32_t num_threads_20693)
{
    #define segscan_group_sizze_19935 (mainzisegscan_group_sizze_19934)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20701_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_20701_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_20699_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20699_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20694;
    int32_t local_tid_20695;
    int64_t group_sizze_20698;
    int32_t wave_sizze_20697;
    int32_t group_tid_20696;
    
    global_tid_20694 = get_global_id(0);
    local_tid_20695 = get_local_id(0);
    group_sizze_20698 = get_local_size(0);
    wave_sizze_20697 = LOCKSTEP_WIDTH;
    group_tid_20696 = get_group_id(0);
    
    int32_t phys_tid_19940;
    
    phys_tid_19940 = global_tid_20694;
    
    __local char *scan_arr_mem_20699;
    __local char *scan_arr_mem_20701;
    
    scan_arr_mem_20699 = (__local char *) scan_arr_mem_20699_backing_0;
    scan_arr_mem_20701 = (__local char *) scan_arr_mem_20701_backing_1;
    
    bool x_18864;
    int64_t x_18865;
    bool x_18866;
    int64_t x_18867;
    
    x_18864 = 0;
    x_18865 = 0;
    for (int64_t j_20703 = 0; j_20703 < sdiv_up64(res_18845,
                                                  sext_i32_i64(num_threads_20693));
         j_20703++) {
        int64_t chunk_offset_20704 = segscan_group_sizze_19935 * j_20703 +
                sext_i32_i64(group_tid_20696) * (segscan_group_sizze_19935 *
                                                 sdiv_up64(res_18845,
                                                           sext_i32_i64(num_threads_20693)));
        int64_t flat_idx_20705 = chunk_offset_20704 +
                sext_i32_i64(local_tid_20695);
        int64_t gtid_19939 = flat_idx_20705;
        
        // threads in bounds read input
        {
            if (slt64(gtid_19939, res_18845)) {
                int64_t x_18871 = ((__global int64_t *) mem_20198)[gtid_19939];
                bool res_18872 = slt64(0, x_18871);
                
                // write to-scan values to parameters
                {
                    x_18866 = res_18872;
                    x_18867 = x_18871;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_19939, res_18845)) {
                    x_18866 = 0;
                    x_18867 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_18868 = x_18864 || x_18866;
                int64_t res_18869;
                
                if (x_18866) {
                    res_18869 = x_18867;
                } else {
                    int64_t res_18870 = add64(x_18865, x_18867);
                    
                    res_18869 = res_18870;
                }
                ((__local
                  bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)] =
                    res_18868;
                ((__local
                  int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)] =
                    res_18869;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_20706;
            int64_t x_20707;
            bool x_20708;
            int64_t x_20709;
            bool x_20713;
            int64_t x_20714;
            bool x_20715;
            int64_t x_20716;
            bool ltid_in_bounds_20720;
            
            ltid_in_bounds_20720 = slt64(sext_i32_i64(local_tid_20695),
                                         segscan_group_sizze_19935);
            
            int32_t skip_threads_20721;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_20720) {
                    x_20708 = ((volatile __local
                                bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)];
                    x_20709 = ((volatile __local
                                int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)];
                    if ((local_tid_20695 - squot32(local_tid_20695, 32) * 32) ==
                        0) {
                        x_20706 = x_20708;
                        x_20707 = x_20709;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_20721 = 1;
                while (slt32(skip_threads_20721, 32)) {
                    if (sle32(skip_threads_20721, local_tid_20695 -
                              squot32(local_tid_20695, 32) * 32) &&
                        ltid_in_bounds_20720) {
                        // read operands
                        {
                            x_20706 = ((volatile __local
                                        bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695) -
                                                                    sext_i32_i64(skip_threads_20721)];
                            x_20707 = ((volatile __local
                                        int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695) -
                                                                       sext_i32_i64(skip_threads_20721)];
                        }
                        // perform operation
                        {
                            bool res_20710 = x_20706 || x_20708;
                            int64_t res_20711;
                            
                            if (x_20708) {
                                res_20711 = x_20709;
                            } else {
                                int64_t res_20712 = add64(x_20707, x_20709);
                                
                                res_20711 = res_20712;
                            }
                            x_20706 = res_20710;
                            x_20707 = res_20711;
                        }
                    }
                    if (sle32(wave_sizze_20697, skip_threads_20721)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_20721, local_tid_20695 -
                              squot32(local_tid_20695, 32) * 32) &&
                        ltid_in_bounds_20720) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)] =
                                x_20706;
                            x_20708 = x_20706;
                            ((volatile __local
                              int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)] =
                                x_20707;
                            x_20709 = x_20707;
                        }
                    }
                    if (sle32(wave_sizze_20697, skip_threads_20721)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_20721 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_20695 - squot32(local_tid_20695, 32) * 32) ==
                    31 && ltid_in_bounds_20720) {
                    ((volatile __local
                      bool *) scan_arr_mem_20699)[sext_i32_i64(squot32(local_tid_20695,
                                                                       32))] =
                        x_20706;
                    ((volatile __local
                      int64_t *) scan_arr_mem_20701)[sext_i32_i64(squot32(local_tid_20695,
                                                                          32))] =
                        x_20707;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_20722;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_20695, 32) == 0 &&
                        ltid_in_bounds_20720) {
                        x_20715 = ((volatile __local
                                    bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)];
                        x_20716 = ((volatile __local
                                    int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)];
                        if ((local_tid_20695 - squot32(local_tid_20695, 32) *
                             32) == 0) {
                            x_20713 = x_20715;
                            x_20714 = x_20716;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_20722 = 1;
                    while (slt32(skip_threads_20722, 32)) {
                        if (sle32(skip_threads_20722, local_tid_20695 -
                                  squot32(local_tid_20695, 32) * 32) &&
                            (squot32(local_tid_20695, 32) == 0 &&
                             ltid_in_bounds_20720)) {
                            // read operands
                            {
                                x_20713 = ((volatile __local
                                            bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695) -
                                                                        sext_i32_i64(skip_threads_20722)];
                                x_20714 = ((volatile __local
                                            int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695) -
                                                                           sext_i32_i64(skip_threads_20722)];
                            }
                            // perform operation
                            {
                                bool res_20717 = x_20713 || x_20715;
                                int64_t res_20718;
                                
                                if (x_20715) {
                                    res_20718 = x_20716;
                                } else {
                                    int64_t res_20719 = add64(x_20714, x_20716);
                                    
                                    res_20718 = res_20719;
                                }
                                x_20713 = res_20717;
                                x_20714 = res_20718;
                            }
                        }
                        if (sle32(wave_sizze_20697, skip_threads_20722)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_20722, local_tid_20695 -
                                  squot32(local_tid_20695, 32) * 32) &&
                            (squot32(local_tid_20695, 32) == 0 &&
                             ltid_in_bounds_20720)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)] =
                                    x_20713;
                                x_20715 = x_20713;
                                ((volatile __local
                                  int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)] =
                                    x_20714;
                                x_20716 = x_20714;
                            }
                        }
                        if (sle32(wave_sizze_20697, skip_threads_20722)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_20722 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_20695, 32) == 0 ||
                      !ltid_in_bounds_20720)) {
                    // read operands
                    {
                        x_20708 = x_20706;
                        x_20709 = x_20707;
                        x_20706 = ((__local
                                    bool *) scan_arr_mem_20699)[sext_i32_i64(squot32(local_tid_20695,
                                                                                     32)) -
                                                                1];
                        x_20707 = ((__local
                                    int64_t *) scan_arr_mem_20701)[sext_i32_i64(squot32(local_tid_20695,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        bool res_20710 = x_20706 || x_20708;
                        int64_t res_20711;
                        
                        if (x_20708) {
                            res_20711 = x_20709;
                        } else {
                            int64_t res_20712 = add64(x_20707, x_20709);
                            
                            res_20711 = res_20712;
                        }
                        x_20706 = res_20710;
                        x_20707 = res_20711;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)] =
                            x_20706;
                        ((__local
                          int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)] =
                            x_20707;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_20695, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)] =
                        x_20708;
                    ((__local
                      int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)] =
                        x_20709;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_19939, res_18845)) {
                    ((__global bool *) mem_20202)[gtid_19939] = ((__local
                                                                  bool *) scan_arr_mem_20699)[sext_i32_i64(local_tid_20695)];
                    ((__global int64_t *) mem_20204)[gtid_19939] = ((__local
                                                                     int64_t *) scan_arr_mem_20701)[sext_i32_i64(local_tid_20695)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_20723 = 0;
                bool should_load_carry_20724 = local_tid_20695 == 0 &&
                     !crosses_segment_20723;
                
                if (should_load_carry_20724) {
                    x_18864 = ((__local
                                bool *) scan_arr_mem_20699)[segscan_group_sizze_19935 -
                                                            1];
                    x_18865 = ((__local
                                int64_t *) scan_arr_mem_20701)[segscan_group_sizze_19935 -
                                                               1];
                }
                if (!should_load_carry_20724) {
                    x_18864 = 0;
                    x_18865 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_19935
}
__kernel void mainziscan_stage1_19948(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20768_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_20766_backing_aligned_1,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20204, __global
                                      unsigned char *mem_20207, __global
                                      unsigned char *mem_20209, __global
                                      unsigned char *mem_20211,
                                      int32_t num_threads_20760)
{
    #define segscan_group_sizze_19943 (mainzisegscan_group_sizze_19942)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20768_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_20768_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_20766_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20766_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20761;
    int32_t local_tid_20762;
    int64_t group_sizze_20765;
    int32_t wave_sizze_20764;
    int32_t group_tid_20763;
    
    global_tid_20761 = get_global_id(0);
    local_tid_20762 = get_local_id(0);
    group_sizze_20765 = get_local_size(0);
    wave_sizze_20764 = LOCKSTEP_WIDTH;
    group_tid_20763 = get_group_id(0);
    
    int32_t phys_tid_19948;
    
    phys_tid_19948 = global_tid_20761;
    
    __local char *scan_arr_mem_20766;
    __local char *scan_arr_mem_20768;
    
    scan_arr_mem_20766 = (__local char *) scan_arr_mem_20766_backing_0;
    scan_arr_mem_20768 = (__local char *) scan_arr_mem_20768_backing_1;
    
    bool x_18903;
    int64_t x_18904;
    bool x_18905;
    int64_t x_18906;
    
    x_18903 = 0;
    x_18904 = 0;
    for (int64_t j_20770 = 0; j_20770 < sdiv_up64(res_18845,
                                                  sext_i32_i64(num_threads_20760));
         j_20770++) {
        int64_t chunk_offset_20771 = segscan_group_sizze_19943 * j_20770 +
                sext_i32_i64(group_tid_20763) * (segscan_group_sizze_19943 *
                                                 sdiv_up64(res_18845,
                                                           sext_i32_i64(num_threads_20760)));
        int64_t flat_idx_20772 = chunk_offset_20771 +
                sext_i32_i64(local_tid_20762);
        int64_t gtid_19947 = flat_idx_20772;
        
        // threads in bounds read input
        {
            if (slt64(gtid_19947, res_18845)) {
                int64_t x_18910 = ((__global int64_t *) mem_20204)[gtid_19947];
                int64_t i_p_o_20044 = add64(-1, gtid_19947);
                int64_t rot_i_20045 = smod64(i_p_o_20044, res_18845);
                int64_t x_18911 = ((__global int64_t *) mem_20204)[rot_i_20045];
                bool res_18913 = x_18910 == x_18911;
                bool res_18914 = !res_18913;
                
                // write to-scan values to parameters
                {
                    x_18905 = res_18914;
                    x_18906 = 1;
                }
                // write mapped values results to global memory
                {
                    ((__global bool *) mem_20211)[gtid_19947] = res_18914;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_19947, res_18845)) {
                    x_18905 = 0;
                    x_18906 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_18907 = x_18903 || x_18905;
                int64_t res_18908;
                
                if (x_18905) {
                    res_18908 = x_18906;
                } else {
                    int64_t res_18909 = add64(x_18904, x_18906);
                    
                    res_18908 = res_18909;
                }
                ((__local
                  bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)] =
                    res_18907;
                ((__local
                  int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)] =
                    res_18908;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_20773;
            int64_t x_20774;
            bool x_20775;
            int64_t x_20776;
            bool x_20780;
            int64_t x_20781;
            bool x_20782;
            int64_t x_20783;
            bool ltid_in_bounds_20787;
            
            ltid_in_bounds_20787 = slt64(sext_i32_i64(local_tid_20762),
                                         segscan_group_sizze_19943);
            
            int32_t skip_threads_20788;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_20787) {
                    x_20775 = ((volatile __local
                                bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)];
                    x_20776 = ((volatile __local
                                int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)];
                    if ((local_tid_20762 - squot32(local_tid_20762, 32) * 32) ==
                        0) {
                        x_20773 = x_20775;
                        x_20774 = x_20776;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_20788 = 1;
                while (slt32(skip_threads_20788, 32)) {
                    if (sle32(skip_threads_20788, local_tid_20762 -
                              squot32(local_tid_20762, 32) * 32) &&
                        ltid_in_bounds_20787) {
                        // read operands
                        {
                            x_20773 = ((volatile __local
                                        bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762) -
                                                                    sext_i32_i64(skip_threads_20788)];
                            x_20774 = ((volatile __local
                                        int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762) -
                                                                       sext_i32_i64(skip_threads_20788)];
                        }
                        // perform operation
                        {
                            bool res_20777 = x_20773 || x_20775;
                            int64_t res_20778;
                            
                            if (x_20775) {
                                res_20778 = x_20776;
                            } else {
                                int64_t res_20779 = add64(x_20774, x_20776);
                                
                                res_20778 = res_20779;
                            }
                            x_20773 = res_20777;
                            x_20774 = res_20778;
                        }
                    }
                    if (sle32(wave_sizze_20764, skip_threads_20788)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_20788, local_tid_20762 -
                              squot32(local_tid_20762, 32) * 32) &&
                        ltid_in_bounds_20787) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)] =
                                x_20773;
                            x_20775 = x_20773;
                            ((volatile __local
                              int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)] =
                                x_20774;
                            x_20776 = x_20774;
                        }
                    }
                    if (sle32(wave_sizze_20764, skip_threads_20788)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_20788 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_20762 - squot32(local_tid_20762, 32) * 32) ==
                    31 && ltid_in_bounds_20787) {
                    ((volatile __local
                      bool *) scan_arr_mem_20766)[sext_i32_i64(squot32(local_tid_20762,
                                                                       32))] =
                        x_20773;
                    ((volatile __local
                      int64_t *) scan_arr_mem_20768)[sext_i32_i64(squot32(local_tid_20762,
                                                                          32))] =
                        x_20774;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_20789;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_20762, 32) == 0 &&
                        ltid_in_bounds_20787) {
                        x_20782 = ((volatile __local
                                    bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)];
                        x_20783 = ((volatile __local
                                    int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)];
                        if ((local_tid_20762 - squot32(local_tid_20762, 32) *
                             32) == 0) {
                            x_20780 = x_20782;
                            x_20781 = x_20783;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_20789 = 1;
                    while (slt32(skip_threads_20789, 32)) {
                        if (sle32(skip_threads_20789, local_tid_20762 -
                                  squot32(local_tid_20762, 32) * 32) &&
                            (squot32(local_tid_20762, 32) == 0 &&
                             ltid_in_bounds_20787)) {
                            // read operands
                            {
                                x_20780 = ((volatile __local
                                            bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762) -
                                                                        sext_i32_i64(skip_threads_20789)];
                                x_20781 = ((volatile __local
                                            int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762) -
                                                                           sext_i32_i64(skip_threads_20789)];
                            }
                            // perform operation
                            {
                                bool res_20784 = x_20780 || x_20782;
                                int64_t res_20785;
                                
                                if (x_20782) {
                                    res_20785 = x_20783;
                                } else {
                                    int64_t res_20786 = add64(x_20781, x_20783);
                                    
                                    res_20785 = res_20786;
                                }
                                x_20780 = res_20784;
                                x_20781 = res_20785;
                            }
                        }
                        if (sle32(wave_sizze_20764, skip_threads_20789)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_20789, local_tid_20762 -
                                  squot32(local_tid_20762, 32) * 32) &&
                            (squot32(local_tid_20762, 32) == 0 &&
                             ltid_in_bounds_20787)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)] =
                                    x_20780;
                                x_20782 = x_20780;
                                ((volatile __local
                                  int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)] =
                                    x_20781;
                                x_20783 = x_20781;
                            }
                        }
                        if (sle32(wave_sizze_20764, skip_threads_20789)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_20789 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_20762, 32) == 0 ||
                      !ltid_in_bounds_20787)) {
                    // read operands
                    {
                        x_20775 = x_20773;
                        x_20776 = x_20774;
                        x_20773 = ((__local
                                    bool *) scan_arr_mem_20766)[sext_i32_i64(squot32(local_tid_20762,
                                                                                     32)) -
                                                                1];
                        x_20774 = ((__local
                                    int64_t *) scan_arr_mem_20768)[sext_i32_i64(squot32(local_tid_20762,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        bool res_20777 = x_20773 || x_20775;
                        int64_t res_20778;
                        
                        if (x_20775) {
                            res_20778 = x_20776;
                        } else {
                            int64_t res_20779 = add64(x_20774, x_20776);
                            
                            res_20778 = res_20779;
                        }
                        x_20773 = res_20777;
                        x_20774 = res_20778;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)] =
                            x_20773;
                        ((__local
                          int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)] =
                            x_20774;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_20762, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)] =
                        x_20775;
                    ((__local
                      int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)] =
                        x_20776;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_19947, res_18845)) {
                    ((__global bool *) mem_20207)[gtid_19947] = ((__local
                                                                  bool *) scan_arr_mem_20766)[sext_i32_i64(local_tid_20762)];
                    ((__global int64_t *) mem_20209)[gtid_19947] = ((__local
                                                                     int64_t *) scan_arr_mem_20768)[sext_i32_i64(local_tid_20762)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_20790 = 0;
                bool should_load_carry_20791 = local_tid_20762 == 0 &&
                     !crosses_segment_20790;
                
                if (should_load_carry_20791) {
                    x_18903 = ((__local
                                bool *) scan_arr_mem_20766)[segscan_group_sizze_19943 -
                                                            1];
                    x_18904 = ((__local
                                int64_t *) scan_arr_mem_20768)[segscan_group_sizze_19943 -
                                                               1];
                }
                if (!should_load_carry_20791) {
                    x_18903 = 0;
                    x_18904 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_19943
}
__kernel void mainziscan_stage1_19956(__global int *global_failure,
                                      int failure_is_an_option, __global
                                      int64_t *global_failure_args,
                                      __local volatile
                                      int64_t *scan_arr_mem_20835_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_20833_backing_aligned_1,
                                      int64_t paths_18599, float a_18604,
                                      float b_18605, float sigma_18606,
                                      float res_18789, float res_18790,
                                      int64_t res_18791, float res_18792,
                                      float res_18827, int64_t res_18845,
                                      int64_t i_19898, __global
                                      unsigned char *mem_20179, __global
                                      unsigned char *mem_20204, __global
                                      unsigned char *mem_20209, __global
                                      unsigned char *mem_20211, __global
                                      unsigned char *mem_20214, __global
                                      unsigned char *mem_20216,
                                      int32_t num_threads_20827)
{
    #define segscan_group_sizze_19951 (mainzisegscan_group_sizze_19950)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20835_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_20835_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_20833_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20833_backing_aligned_1;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_20828;
    int32_t local_tid_20829;
    int64_t group_sizze_20832;
    int32_t wave_sizze_20831;
    int32_t group_tid_20830;
    
    global_tid_20828 = get_global_id(0);
    local_tid_20829 = get_local_id(0);
    group_sizze_20832 = get_local_size(0);
    wave_sizze_20831 = LOCKSTEP_WIDTH;
    group_tid_20830 = get_group_id(0);
    
    int32_t phys_tid_19956;
    
    phys_tid_19956 = global_tid_20828;
    
    __local char *scan_arr_mem_20833;
    __local char *scan_arr_mem_20835;
    
    scan_arr_mem_20833 = (__local char *) scan_arr_mem_20833_backing_0;
    scan_arr_mem_20835 = (__local char *) scan_arr_mem_20835_backing_1;
    
    bool x_18929;
    float x_18930;
    bool x_18931;
    float x_18932;
    
    x_18929 = 0;
    x_18930 = 0.0F;
    for (int64_t j_20837 = 0; j_20837 < sdiv_up64(res_18845,
                                                  sext_i32_i64(num_threads_20827));
         j_20837++) {
        int64_t chunk_offset_20838 = segscan_group_sizze_19951 * j_20837 +
                sext_i32_i64(group_tid_20830) * (segscan_group_sizze_19951 *
                                                 sdiv_up64(res_18845,
                                                           sext_i32_i64(num_threads_20827)));
        int64_t flat_idx_20839 = chunk_offset_20838 +
                sext_i32_i64(local_tid_20829);
        int64_t gtid_19955 = flat_idx_20839;
        
        // threads in bounds read input
        {
            if (slt64(gtid_19955, res_18845)) {
                int64_t x_18937 = ((__global int64_t *) mem_20209)[gtid_19955];
                int64_t x_18938 = ((__global int64_t *) mem_20204)[gtid_19955];
                bool x_18939 = ((__global bool *) mem_20211)[gtid_19955];
                int64_t res_18942 = sub64(x_18937, 1);
                bool x_18943 = sle64(0, x_18938);
                bool y_18944 = slt64(x_18938, paths_18599);
                bool bounds_check_18945 = x_18943 && y_18944;
                bool index_certs_18946;
                
                if (!bounds_check_18945) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 8) ==
                            -1) {
                            global_failure_args[0] = x_18938;
                            global_failure_args[1] = paths_18599;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float x_18956 = res_18827 / res_18792;
                float ceil_arg_18957 = x_18956 - 1.0F;
                float res_18958;
                
                res_18958 = futrts_ceil32(ceil_arg_18957);
                
                int64_t res_18959 = fptosi_f32_i64(res_18958);
                int64_t max_arg_18960 = sub64(res_18791, res_18959);
                int64_t res_18961 = smax64(0, max_arg_18960);
                bool cond_18962 = res_18961 == 0;
                float res_18963;
                
                if (cond_18962) {
                    res_18963 = 0.0F;
                } else {
                    float lifted_0_get_arg_18947 = ((__global
                                                     float *) mem_20179)[i_19898 *
                                                                         paths_18599 +
                                                                         x_18938];
                    float res_18964;
                    
                    res_18964 = futrts_ceil32(x_18956);
                    
                    float start_18965 = res_18792 * res_18964;
                    float res_18966;
                    
                    res_18966 = futrts_ceil32(ceil_arg_18957);
                    
                    int64_t res_18967 = fptosi_f32_i64(res_18966);
                    int64_t max_arg_18968 = sub64(res_18791, res_18967);
                    int64_t res_18969 = smax64(0, max_arg_18968);
                    int64_t sizze_18970 = sub64(res_18969, 1);
                    bool cond_18971 = res_18942 == 0;
                    float res_18972;
                    
                    if (cond_18971) {
                        res_18972 = 1.0F;
                    } else {
                        res_18972 = 0.0F;
                    }
                    
                    bool cond_18973 = slt64(0, res_18942);
                    float res_18974;
                    
                    if (cond_18973) {
                        float y_18975 = res_18789 * res_18792;
                        float res_18976 = res_18972 - y_18975;
                        
                        res_18974 = res_18976;
                    } else {
                        res_18974 = res_18972;
                    }
                    
                    bool cond_18977 = res_18942 == sizze_18970;
                    float res_18978;
                    
                    if (cond_18977) {
                        float res_18979 = res_18974 - 1.0F;
                        
                        res_18978 = res_18979;
                    } else {
                        res_18978 = res_18974;
                    }
                    
                    float res_18980 = res_18790 * res_18978;
                    float res_18981 = sitofp_i64_f32(res_18942);
                    float y_18982 = res_18792 * res_18981;
                    float bondprice_arg_18983 = start_18965 + y_18982;
                    float y_18984 = bondprice_arg_18983 - res_18827;
                    float negate_arg_18985 = a_18604 * y_18984;
                    float exp_arg_18986 = 0.0F - negate_arg_18985;
                    float res_18987 = fpow32(2.7182817F, exp_arg_18986);
                    float x_18988 = 1.0F - res_18987;
                    float B_18989 = x_18988 / a_18604;
                    float x_18990 = B_18989 - bondprice_arg_18983;
                    float x_18991 = res_18827 + x_18990;
                    float x_18992 = fpow32(a_18604, 2.0F);
                    float x_18993 = b_18605 * x_18992;
                    float x_18994 = fpow32(sigma_18606, 2.0F);
                    float y_18995 = x_18994 / 2.0F;
                    float y_18996 = x_18993 - y_18995;
                    float x_18997 = x_18991 * y_18996;
                    float A1_18998 = x_18997 / x_18992;
                    float y_18999 = fpow32(B_18989, 2.0F);
                    float x_19000 = x_18994 * y_18999;
                    float y_19001 = 4.0F * a_18604;
                    float A2_19002 = x_19000 / y_19001;
                    float exp_arg_19003 = A1_18998 - A2_19002;
                    float res_19004 = fpow32(2.7182817F, exp_arg_19003);
                    float negate_arg_19005 = lifted_0_get_arg_18947 * B_18989;
                    float exp_arg_19006 = 0.0F - negate_arg_19005;
                    float res_19007 = fpow32(2.7182817F, exp_arg_19006);
                    float res_19008 = res_19004 * res_19007;
                    float res_19009 = res_18980 * res_19008;
                    
                    res_18963 = res_19009;
                }
                // write to-scan values to parameters
                {
                    x_18931 = x_18939;
                    x_18932 = res_18963;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_19955, res_18845)) {
                    x_18931 = 0;
                    x_18932 = 0.0F;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_18933 = x_18929 || x_18931;
                float res_18934;
                
                if (x_18931) {
                    res_18934 = x_18932;
                } else {
                    float res_18935 = x_18930 + x_18932;
                    
                    res_18934 = res_18935;
                }
                ((__local
                  bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)] =
                    res_18933;
                ((__local
                  float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)] =
                    res_18934;
            }
            
          error_0:
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_failure)
                return;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_20840;
            float x_20841;
            bool x_20842;
            float x_20843;
            bool x_20847;
            float x_20848;
            bool x_20849;
            float x_20850;
            bool ltid_in_bounds_20854;
            
            ltid_in_bounds_20854 = slt64(sext_i32_i64(local_tid_20829),
                                         segscan_group_sizze_19951);
            
            int32_t skip_threads_20855;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_20854) {
                    x_20842 = ((volatile __local
                                bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)];
                    x_20843 = ((volatile __local
                                float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)];
                    if ((local_tid_20829 - squot32(local_tid_20829, 32) * 32) ==
                        0) {
                        x_20840 = x_20842;
                        x_20841 = x_20843;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_20855 = 1;
                while (slt32(skip_threads_20855, 32)) {
                    if (sle32(skip_threads_20855, local_tid_20829 -
                              squot32(local_tid_20829, 32) * 32) &&
                        ltid_in_bounds_20854) {
                        // read operands
                        {
                            x_20840 = ((volatile __local
                                        bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829) -
                                                                    sext_i32_i64(skip_threads_20855)];
                            x_20841 = ((volatile __local
                                        float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829) -
                                                                     sext_i32_i64(skip_threads_20855)];
                        }
                        // perform operation
                        {
                            bool res_20844 = x_20840 || x_20842;
                            float res_20845;
                            
                            if (x_20842) {
                                res_20845 = x_20843;
                            } else {
                                float res_20846 = x_20841 + x_20843;
                                
                                res_20845 = res_20846;
                            }
                            x_20840 = res_20844;
                            x_20841 = res_20845;
                        }
                    }
                    if (sle32(wave_sizze_20831, skip_threads_20855)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_20855, local_tid_20829 -
                              squot32(local_tid_20829, 32) * 32) &&
                        ltid_in_bounds_20854) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)] =
                                x_20840;
                            x_20842 = x_20840;
                            ((volatile __local
                              float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)] =
                                x_20841;
                            x_20843 = x_20841;
                        }
                    }
                    if (sle32(wave_sizze_20831, skip_threads_20855)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_20855 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_20829 - squot32(local_tid_20829, 32) * 32) ==
                    31 && ltid_in_bounds_20854) {
                    ((volatile __local
                      bool *) scan_arr_mem_20833)[sext_i32_i64(squot32(local_tid_20829,
                                                                       32))] =
                        x_20840;
                    ((volatile __local
                      float *) scan_arr_mem_20835)[sext_i32_i64(squot32(local_tid_20829,
                                                                        32))] =
                        x_20841;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_20856;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_20829, 32) == 0 &&
                        ltid_in_bounds_20854) {
                        x_20849 = ((volatile __local
                                    bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)];
                        x_20850 = ((volatile __local
                                    float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)];
                        if ((local_tid_20829 - squot32(local_tid_20829, 32) *
                             32) == 0) {
                            x_20847 = x_20849;
                            x_20848 = x_20850;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_20856 = 1;
                    while (slt32(skip_threads_20856, 32)) {
                        if (sle32(skip_threads_20856, local_tid_20829 -
                                  squot32(local_tid_20829, 32) * 32) &&
                            (squot32(local_tid_20829, 32) == 0 &&
                             ltid_in_bounds_20854)) {
                            // read operands
                            {
                                x_20847 = ((volatile __local
                                            bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829) -
                                                                        sext_i32_i64(skip_threads_20856)];
                                x_20848 = ((volatile __local
                                            float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829) -
                                                                         sext_i32_i64(skip_threads_20856)];
                            }
                            // perform operation
                            {
                                bool res_20851 = x_20847 || x_20849;
                                float res_20852;
                                
                                if (x_20849) {
                                    res_20852 = x_20850;
                                } else {
                                    float res_20853 = x_20848 + x_20850;
                                    
                                    res_20852 = res_20853;
                                }
                                x_20847 = res_20851;
                                x_20848 = res_20852;
                            }
                        }
                        if (sle32(wave_sizze_20831, skip_threads_20856)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_20856, local_tid_20829 -
                                  squot32(local_tid_20829, 32) * 32) &&
                            (squot32(local_tid_20829, 32) == 0 &&
                             ltid_in_bounds_20854)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)] =
                                    x_20847;
                                x_20849 = x_20847;
                                ((volatile __local
                                  float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)] =
                                    x_20848;
                                x_20850 = x_20848;
                            }
                        }
                        if (sle32(wave_sizze_20831, skip_threads_20856)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_20856 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_20829, 32) == 0 ||
                      !ltid_in_bounds_20854)) {
                    // read operands
                    {
                        x_20842 = x_20840;
                        x_20843 = x_20841;
                        x_20840 = ((__local
                                    bool *) scan_arr_mem_20833)[sext_i32_i64(squot32(local_tid_20829,
                                                                                     32)) -
                                                                1];
                        x_20841 = ((__local
                                    float *) scan_arr_mem_20835)[sext_i32_i64(squot32(local_tid_20829,
                                                                                      32)) -
                                                                 1];
                    }
                    // perform operation
                    {
                        bool res_20844 = x_20840 || x_20842;
                        float res_20845;
                        
                        if (x_20842) {
                            res_20845 = x_20843;
                        } else {
                            float res_20846 = x_20841 + x_20843;
                            
                            res_20845 = res_20846;
                        }
                        x_20840 = res_20844;
                        x_20841 = res_20845;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)] =
                            x_20840;
                        ((__local
                          float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)] =
                            x_20841;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_20829, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)] =
                        x_20842;
                    ((__local
                      float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)] =
                        x_20843;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_19955, res_18845)) {
                    ((__global bool *) mem_20214)[gtid_19955] = ((__local
                                                                  bool *) scan_arr_mem_20833)[sext_i32_i64(local_tid_20829)];
                    ((__global float *) mem_20216)[gtid_19955] = ((__local
                                                                   float *) scan_arr_mem_20835)[sext_i32_i64(local_tid_20829)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_20857 = 0;
                bool should_load_carry_20858 = local_tid_20829 == 0 &&
                     !crosses_segment_20857;
                
                if (should_load_carry_20858) {
                    x_18929 = ((__local
                                bool *) scan_arr_mem_20833)[segscan_group_sizze_19951 -
                                                            1];
                    x_18930 = ((__local
                                float *) scan_arr_mem_20835)[segscan_group_sizze_19951 -
                                                             1];
                }
                if (!should_load_carry_20858) {
                    x_18929 = 0;
                    x_18930 = 0.0F;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_19951
}
__kernel void mainziscan_stage1_20008(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20900_backing_aligned_0,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20211, __global
                                      unsigned char *mem_20219,
                                      int32_t num_threads_20894)
{
    #define segscan_group_sizze_20003 (mainzisegscan_group_sizze_20002)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20900_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20900_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20895;
    int32_t local_tid_20896;
    int64_t group_sizze_20899;
    int32_t wave_sizze_20898;
    int32_t group_tid_20897;
    
    global_tid_20895 = get_global_id(0);
    local_tid_20896 = get_local_id(0);
    group_sizze_20899 = get_local_size(0);
    wave_sizze_20898 = LOCKSTEP_WIDTH;
    group_tid_20897 = get_group_id(0);
    
    int32_t phys_tid_20008;
    
    phys_tid_20008 = global_tid_20895;
    
    __local char *scan_arr_mem_20900;
    
    scan_arr_mem_20900 = (__local char *) scan_arr_mem_20900_backing_0;
    
    int64_t x_19033;
    int64_t x_19034;
    
    x_19033 = 0;
    for (int64_t j_20902 = 0; j_20902 < sdiv_up64(res_18845,
                                                  sext_i32_i64(num_threads_20894));
         j_20902++) {
        int64_t chunk_offset_20903 = segscan_group_sizze_20003 * j_20902 +
                sext_i32_i64(group_tid_20897) * (segscan_group_sizze_20003 *
                                                 sdiv_up64(res_18845,
                                                           sext_i32_i64(num_threads_20894)));
        int64_t flat_idx_20904 = chunk_offset_20903 +
                sext_i32_i64(local_tid_20896);
        int64_t gtid_20007 = flat_idx_20904;
        
        // threads in bounds read input
        {
            if (slt64(gtid_20007, res_18845)) {
                int64_t i_p_o_20050 = add64(1, gtid_20007);
                int64_t rot_i_20051 = smod64(i_p_o_20050, res_18845);
                bool x_19036 = ((__global bool *) mem_20211)[rot_i_20051];
                int64_t res_19037 = btoi_bool_i64(x_19036);
                
                // write to-scan values to parameters
                {
                    x_19034 = res_19037;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_20007, res_18845)) {
                    x_19034 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int64_t res_19035 = add64(x_19033, x_19034);
                
                ((__local
                  int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)] =
                    res_19035;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int64_t x_20905;
            int64_t x_20906;
            int64_t x_20908;
            int64_t x_20909;
            bool ltid_in_bounds_20911;
            
            ltid_in_bounds_20911 = slt64(sext_i32_i64(local_tid_20896),
                                         segscan_group_sizze_20003);
            
            int32_t skip_threads_20912;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_20911) {
                    x_20906 = ((volatile __local
                                int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)];
                    if ((local_tid_20896 - squot32(local_tid_20896, 32) * 32) ==
                        0) {
                        x_20905 = x_20906;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_20912 = 1;
                while (slt32(skip_threads_20912, 32)) {
                    if (sle32(skip_threads_20912, local_tid_20896 -
                              squot32(local_tid_20896, 32) * 32) &&
                        ltid_in_bounds_20911) {
                        // read operands
                        {
                            x_20905 = ((volatile __local
                                        int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896) -
                                                                       sext_i32_i64(skip_threads_20912)];
                        }
                        // perform operation
                        {
                            int64_t res_20907 = add64(x_20905, x_20906);
                            
                            x_20905 = res_20907;
                        }
                    }
                    if (sle32(wave_sizze_20898, skip_threads_20912)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_20912, local_tid_20896 -
                              squot32(local_tid_20896, 32) * 32) &&
                        ltid_in_bounds_20911) {
                        // write result
                        {
                            ((volatile __local
                              int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)] =
                                x_20905;
                            x_20906 = x_20905;
                        }
                    }
                    if (sle32(wave_sizze_20898, skip_threads_20912)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_20912 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_20896 - squot32(local_tid_20896, 32) * 32) ==
                    31 && ltid_in_bounds_20911) {
                    ((volatile __local
                      int64_t *) scan_arr_mem_20900)[sext_i32_i64(squot32(local_tid_20896,
                                                                          32))] =
                        x_20905;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_20913;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_20896, 32) == 0 &&
                        ltid_in_bounds_20911) {
                        x_20909 = ((volatile __local
                                    int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)];
                        if ((local_tid_20896 - squot32(local_tid_20896, 32) *
                             32) == 0) {
                            x_20908 = x_20909;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_20913 = 1;
                    while (slt32(skip_threads_20913, 32)) {
                        if (sle32(skip_threads_20913, local_tid_20896 -
                                  squot32(local_tid_20896, 32) * 32) &&
                            (squot32(local_tid_20896, 32) == 0 &&
                             ltid_in_bounds_20911)) {
                            // read operands
                            {
                                x_20908 = ((volatile __local
                                            int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896) -
                                                                           sext_i32_i64(skip_threads_20913)];
                            }
                            // perform operation
                            {
                                int64_t res_20910 = add64(x_20908, x_20909);
                                
                                x_20908 = res_20910;
                            }
                        }
                        if (sle32(wave_sizze_20898, skip_threads_20913)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_20913, local_tid_20896 -
                                  squot32(local_tid_20896, 32) * 32) &&
                            (squot32(local_tid_20896, 32) == 0 &&
                             ltid_in_bounds_20911)) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)] =
                                    x_20908;
                                x_20909 = x_20908;
                            }
                        }
                        if (sle32(wave_sizze_20898, skip_threads_20913)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_20913 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_20896, 32) == 0 ||
                      !ltid_in_bounds_20911)) {
                    // read operands
                    {
                        x_20906 = x_20905;
                        x_20905 = ((__local
                                    int64_t *) scan_arr_mem_20900)[sext_i32_i64(squot32(local_tid_20896,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        int64_t res_20907 = add64(x_20905, x_20906);
                        
                        x_20905 = res_20907;
                    }
                    // write final result
                    {
                        ((__local
                          int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)] =
                            x_20905;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_20896, 32) == 0) {
                    ((__local
                      int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)] =
                        x_20906;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_20007, res_18845)) {
                    ((__global int64_t *) mem_20219)[gtid_20007] = ((__local
                                                                     int64_t *) scan_arr_mem_20900)[sext_i32_i64(local_tid_20896)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_20914 = 0;
                bool should_load_carry_20915 = local_tid_20896 == 0 &&
                     !crosses_segment_20914;
                
                if (should_load_carry_20915) {
                    x_19033 = ((__local
                                int64_t *) scan_arr_mem_20900)[segscan_group_sizze_20003 -
                                                               1];
                }
                if (!should_load_carry_20915) {
                    x_19033 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_20003
}
__kernel void mainziscan_stage2_19907(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20483_backing_aligned_0,
                                      int64_t paths_18599, __global
                                      unsigned char *mem_20191,
                                      int64_t stage1_num_groups_20455,
                                      int32_t num_threads_20456)
{
    #define segscan_group_sizze_19902 (mainzisegscan_group_sizze_19901)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20483_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20483_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20478;
    int32_t local_tid_20479;
    int64_t group_sizze_20482;
    int32_t wave_sizze_20481;
    int32_t group_tid_20480;
    
    global_tid_20478 = get_global_id(0);
    local_tid_20479 = get_local_id(0);
    group_sizze_20482 = get_local_size(0);
    wave_sizze_20481 = LOCKSTEP_WIDTH;
    group_tid_20480 = get_group_id(0);
    
    int32_t phys_tid_19907;
    
    phys_tid_19907 = global_tid_20478;
    
    __local char *scan_arr_mem_20483;
    
    scan_arr_mem_20483 = (__local char *) scan_arr_mem_20483_backing_0;
    
    int64_t flat_idx_20485;
    
    flat_idx_20485 = (sext_i32_i64(local_tid_20479) + 1) *
        (segscan_group_sizze_19902 * sdiv_up64(paths_18599,
                                               sext_i32_i64(num_threads_20456))) -
        1;
    
    int64_t gtid_19906;
    
    gtid_19906 = flat_idx_20485;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_19906, paths_18599)) {
            ((__local
              int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)] =
                ((__global int64_t *) mem_20191)[gtid_19906];
        } else {
            ((__local
              int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int64_t x_18830;
    int64_t x_18831;
    int64_t x_20486;
    int64_t x_20487;
    bool ltid_in_bounds_20489;
    
    ltid_in_bounds_20489 = slt64(sext_i32_i64(local_tid_20479),
                                 stage1_num_groups_20455);
    
    int32_t skip_threads_20490;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_20489) {
            x_18831 = ((volatile __local
                        int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)];
            if ((local_tid_20479 - squot32(local_tid_20479, 32) * 32) == 0) {
                x_18830 = x_18831;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_20490 = 1;
        while (slt32(skip_threads_20490, 32)) {
            if (sle32(skip_threads_20490, local_tid_20479 -
                      squot32(local_tid_20479, 32) * 32) &&
                ltid_in_bounds_20489) {
                // read operands
                {
                    x_18830 = ((volatile __local
                                int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479) -
                                                               sext_i32_i64(skip_threads_20490)];
                }
                // perform operation
                {
                    int64_t res_18832 = add64(x_18830, x_18831);
                    
                    x_18830 = res_18832;
                }
            }
            if (sle32(wave_sizze_20481, skip_threads_20490)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_20490, local_tid_20479 -
                      squot32(local_tid_20479, 32) * 32) &&
                ltid_in_bounds_20489) {
                // write result
                {
                    ((volatile __local
                      int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)] =
                        x_18830;
                    x_18831 = x_18830;
                }
            }
            if (sle32(wave_sizze_20481, skip_threads_20490)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_20490 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_20479 - squot32(local_tid_20479, 32) * 32) == 31 &&
            ltid_in_bounds_20489) {
            ((volatile __local
              int64_t *) scan_arr_mem_20483)[sext_i32_i64(squot32(local_tid_20479,
                                                                  32))] =
                x_18830;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_20491;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_20479, 32) == 0 && ltid_in_bounds_20489) {
                x_20487 = ((volatile __local
                            int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)];
                if ((local_tid_20479 - squot32(local_tid_20479, 32) * 32) ==
                    0) {
                    x_20486 = x_20487;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_20491 = 1;
            while (slt32(skip_threads_20491, 32)) {
                if (sle32(skip_threads_20491, local_tid_20479 -
                          squot32(local_tid_20479, 32) * 32) &&
                    (squot32(local_tid_20479, 32) == 0 &&
                     ltid_in_bounds_20489)) {
                    // read operands
                    {
                        x_20486 = ((volatile __local
                                    int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479) -
                                                                   sext_i32_i64(skip_threads_20491)];
                    }
                    // perform operation
                    {
                        int64_t res_20488 = add64(x_20486, x_20487);
                        
                        x_20486 = res_20488;
                    }
                }
                if (sle32(wave_sizze_20481, skip_threads_20491)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_20491, local_tid_20479 -
                          squot32(local_tid_20479, 32) * 32) &&
                    (squot32(local_tid_20479, 32) == 0 &&
                     ltid_in_bounds_20489)) {
                    // write result
                    {
                        ((volatile __local
                          int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)] =
                            x_20486;
                        x_20487 = x_20486;
                    }
                }
                if (sle32(wave_sizze_20481, skip_threads_20491)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_20491 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_20479, 32) == 0 || !ltid_in_bounds_20489)) {
            // read operands
            {
                x_18831 = x_18830;
                x_18830 = ((__local
                            int64_t *) scan_arr_mem_20483)[sext_i32_i64(squot32(local_tid_20479,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                int64_t res_18832 = add64(x_18830, x_18831);
                
                x_18830 = res_18832;
            }
            // write final result
            {
                ((__local
                  int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)] =
                    x_18830;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_20479, 32) == 0) {
            ((__local
              int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)] =
                x_18831;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_19906, paths_18599)) {
            ((__global int64_t *) mem_20191)[gtid_19906] = ((__local
                                                             int64_t *) scan_arr_mem_20483)[sext_i32_i64(local_tid_20479)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19902
}
__kernel void mainziscan_stage2_19940(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20732_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_20730_backing_aligned_1,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20202, __global
                                      unsigned char *mem_20204,
                                      int64_t stage1_num_groups_20692,
                                      int32_t num_threads_20693)
{
    #define segscan_group_sizze_19935 (mainzisegscan_group_sizze_19934)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20732_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_20732_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_20730_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20730_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20725;
    int32_t local_tid_20726;
    int64_t group_sizze_20729;
    int32_t wave_sizze_20728;
    int32_t group_tid_20727;
    
    global_tid_20725 = get_global_id(0);
    local_tid_20726 = get_local_id(0);
    group_sizze_20729 = get_local_size(0);
    wave_sizze_20728 = LOCKSTEP_WIDTH;
    group_tid_20727 = get_group_id(0);
    
    int32_t phys_tid_19940;
    
    phys_tid_19940 = global_tid_20725;
    
    __local char *scan_arr_mem_20730;
    __local char *scan_arr_mem_20732;
    
    scan_arr_mem_20730 = (__local char *) scan_arr_mem_20730_backing_0;
    scan_arr_mem_20732 = (__local char *) scan_arr_mem_20732_backing_1;
    
    int64_t flat_idx_20734;
    
    flat_idx_20734 = (sext_i32_i64(local_tid_20726) + 1) *
        (segscan_group_sizze_19935 * sdiv_up64(res_18845,
                                               sext_i32_i64(num_threads_20693))) -
        1;
    
    int64_t gtid_19939;
    
    gtid_19939 = flat_idx_20734;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_19939, res_18845)) {
            ((__local
              bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)] =
                ((__global bool *) mem_20202)[gtid_19939];
            ((__local
              int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)] =
                ((__global int64_t *) mem_20204)[gtid_19939];
        } else {
            ((__local
              bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)] = 0;
            ((__local
              int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_18864;
    int64_t x_18865;
    bool x_18866;
    int64_t x_18867;
    bool x_20735;
    int64_t x_20736;
    bool x_20737;
    int64_t x_20738;
    bool ltid_in_bounds_20742;
    
    ltid_in_bounds_20742 = slt64(sext_i32_i64(local_tid_20726),
                                 stage1_num_groups_20692);
    
    int32_t skip_threads_20743;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_20742) {
            x_18866 = ((volatile __local
                        bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)];
            x_18867 = ((volatile __local
                        int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)];
            if ((local_tid_20726 - squot32(local_tid_20726, 32) * 32) == 0) {
                x_18864 = x_18866;
                x_18865 = x_18867;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_20743 = 1;
        while (slt32(skip_threads_20743, 32)) {
            if (sle32(skip_threads_20743, local_tid_20726 -
                      squot32(local_tid_20726, 32) * 32) &&
                ltid_in_bounds_20742) {
                // read operands
                {
                    x_18864 = ((volatile __local
                                bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726) -
                                                            sext_i32_i64(skip_threads_20743)];
                    x_18865 = ((volatile __local
                                int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726) -
                                                               sext_i32_i64(skip_threads_20743)];
                }
                // perform operation
                {
                    bool res_18868 = x_18864 || x_18866;
                    int64_t res_18869;
                    
                    if (x_18866) {
                        res_18869 = x_18867;
                    } else {
                        int64_t res_18870 = add64(x_18865, x_18867);
                        
                        res_18869 = res_18870;
                    }
                    x_18864 = res_18868;
                    x_18865 = res_18869;
                }
            }
            if (sle32(wave_sizze_20728, skip_threads_20743)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_20743, local_tid_20726 -
                      squot32(local_tid_20726, 32) * 32) &&
                ltid_in_bounds_20742) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)] =
                        x_18864;
                    x_18866 = x_18864;
                    ((volatile __local
                      int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)] =
                        x_18865;
                    x_18867 = x_18865;
                }
            }
            if (sle32(wave_sizze_20728, skip_threads_20743)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_20743 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_20726 - squot32(local_tid_20726, 32) * 32) == 31 &&
            ltid_in_bounds_20742) {
            ((volatile __local
              bool *) scan_arr_mem_20730)[sext_i32_i64(squot32(local_tid_20726,
                                                               32))] = x_18864;
            ((volatile __local
              int64_t *) scan_arr_mem_20732)[sext_i32_i64(squot32(local_tid_20726,
                                                                  32))] =
                x_18865;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_20744;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_20726, 32) == 0 && ltid_in_bounds_20742) {
                x_20737 = ((volatile __local
                            bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)];
                x_20738 = ((volatile __local
                            int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)];
                if ((local_tid_20726 - squot32(local_tid_20726, 32) * 32) ==
                    0) {
                    x_20735 = x_20737;
                    x_20736 = x_20738;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_20744 = 1;
            while (slt32(skip_threads_20744, 32)) {
                if (sle32(skip_threads_20744, local_tid_20726 -
                          squot32(local_tid_20726, 32) * 32) &&
                    (squot32(local_tid_20726, 32) == 0 &&
                     ltid_in_bounds_20742)) {
                    // read operands
                    {
                        x_20735 = ((volatile __local
                                    bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726) -
                                                                sext_i32_i64(skip_threads_20744)];
                        x_20736 = ((volatile __local
                                    int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726) -
                                                                   sext_i32_i64(skip_threads_20744)];
                    }
                    // perform operation
                    {
                        bool res_20739 = x_20735 || x_20737;
                        int64_t res_20740;
                        
                        if (x_20737) {
                            res_20740 = x_20738;
                        } else {
                            int64_t res_20741 = add64(x_20736, x_20738);
                            
                            res_20740 = res_20741;
                        }
                        x_20735 = res_20739;
                        x_20736 = res_20740;
                    }
                }
                if (sle32(wave_sizze_20728, skip_threads_20744)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_20744, local_tid_20726 -
                          squot32(local_tid_20726, 32) * 32) &&
                    (squot32(local_tid_20726, 32) == 0 &&
                     ltid_in_bounds_20742)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)] =
                            x_20735;
                        x_20737 = x_20735;
                        ((volatile __local
                          int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)] =
                            x_20736;
                        x_20738 = x_20736;
                    }
                }
                if (sle32(wave_sizze_20728, skip_threads_20744)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_20744 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_20726, 32) == 0 || !ltid_in_bounds_20742)) {
            // read operands
            {
                x_18866 = x_18864;
                x_18867 = x_18865;
                x_18864 = ((__local
                            bool *) scan_arr_mem_20730)[sext_i32_i64(squot32(local_tid_20726,
                                                                             32)) -
                                                        1];
                x_18865 = ((__local
                            int64_t *) scan_arr_mem_20732)[sext_i32_i64(squot32(local_tid_20726,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                bool res_18868 = x_18864 || x_18866;
                int64_t res_18869;
                
                if (x_18866) {
                    res_18869 = x_18867;
                } else {
                    int64_t res_18870 = add64(x_18865, x_18867);
                    
                    res_18869 = res_18870;
                }
                x_18864 = res_18868;
                x_18865 = res_18869;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)] =
                    x_18864;
                ((__local
                  int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)] =
                    x_18865;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_20726, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)] =
                x_18866;
            ((__local
              int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)] =
                x_18867;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_19939, res_18845)) {
            ((__global bool *) mem_20202)[gtid_19939] = ((__local
                                                          bool *) scan_arr_mem_20730)[sext_i32_i64(local_tid_20726)];
            ((__global int64_t *) mem_20204)[gtid_19939] = ((__local
                                                             int64_t *) scan_arr_mem_20732)[sext_i32_i64(local_tid_20726)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19935
}
__kernel void mainziscan_stage2_19948(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20799_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_20797_backing_aligned_1,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20207, __global
                                      unsigned char *mem_20209,
                                      int64_t stage1_num_groups_20759,
                                      int32_t num_threads_20760)
{
    #define segscan_group_sizze_19943 (mainzisegscan_group_sizze_19942)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20799_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_20799_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_20797_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20797_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20792;
    int32_t local_tid_20793;
    int64_t group_sizze_20796;
    int32_t wave_sizze_20795;
    int32_t group_tid_20794;
    
    global_tid_20792 = get_global_id(0);
    local_tid_20793 = get_local_id(0);
    group_sizze_20796 = get_local_size(0);
    wave_sizze_20795 = LOCKSTEP_WIDTH;
    group_tid_20794 = get_group_id(0);
    
    int32_t phys_tid_19948;
    
    phys_tid_19948 = global_tid_20792;
    
    __local char *scan_arr_mem_20797;
    __local char *scan_arr_mem_20799;
    
    scan_arr_mem_20797 = (__local char *) scan_arr_mem_20797_backing_0;
    scan_arr_mem_20799 = (__local char *) scan_arr_mem_20799_backing_1;
    
    int64_t flat_idx_20801;
    
    flat_idx_20801 = (sext_i32_i64(local_tid_20793) + 1) *
        (segscan_group_sizze_19943 * sdiv_up64(res_18845,
                                               sext_i32_i64(num_threads_20760))) -
        1;
    
    int64_t gtid_19947;
    
    gtid_19947 = flat_idx_20801;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_19947, res_18845)) {
            ((__local
              bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)] =
                ((__global bool *) mem_20207)[gtid_19947];
            ((__local
              int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)] =
                ((__global int64_t *) mem_20209)[gtid_19947];
        } else {
            ((__local
              bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)] = 0;
            ((__local
              int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_18903;
    int64_t x_18904;
    bool x_18905;
    int64_t x_18906;
    bool x_20802;
    int64_t x_20803;
    bool x_20804;
    int64_t x_20805;
    bool ltid_in_bounds_20809;
    
    ltid_in_bounds_20809 = slt64(sext_i32_i64(local_tid_20793),
                                 stage1_num_groups_20759);
    
    int32_t skip_threads_20810;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_20809) {
            x_18905 = ((volatile __local
                        bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)];
            x_18906 = ((volatile __local
                        int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)];
            if ((local_tid_20793 - squot32(local_tid_20793, 32) * 32) == 0) {
                x_18903 = x_18905;
                x_18904 = x_18906;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_20810 = 1;
        while (slt32(skip_threads_20810, 32)) {
            if (sle32(skip_threads_20810, local_tid_20793 -
                      squot32(local_tid_20793, 32) * 32) &&
                ltid_in_bounds_20809) {
                // read operands
                {
                    x_18903 = ((volatile __local
                                bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793) -
                                                            sext_i32_i64(skip_threads_20810)];
                    x_18904 = ((volatile __local
                                int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793) -
                                                               sext_i32_i64(skip_threads_20810)];
                }
                // perform operation
                {
                    bool res_18907 = x_18903 || x_18905;
                    int64_t res_18908;
                    
                    if (x_18905) {
                        res_18908 = x_18906;
                    } else {
                        int64_t res_18909 = add64(x_18904, x_18906);
                        
                        res_18908 = res_18909;
                    }
                    x_18903 = res_18907;
                    x_18904 = res_18908;
                }
            }
            if (sle32(wave_sizze_20795, skip_threads_20810)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_20810, local_tid_20793 -
                      squot32(local_tid_20793, 32) * 32) &&
                ltid_in_bounds_20809) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)] =
                        x_18903;
                    x_18905 = x_18903;
                    ((volatile __local
                      int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)] =
                        x_18904;
                    x_18906 = x_18904;
                }
            }
            if (sle32(wave_sizze_20795, skip_threads_20810)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_20810 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_20793 - squot32(local_tid_20793, 32) * 32) == 31 &&
            ltid_in_bounds_20809) {
            ((volatile __local
              bool *) scan_arr_mem_20797)[sext_i32_i64(squot32(local_tid_20793,
                                                               32))] = x_18903;
            ((volatile __local
              int64_t *) scan_arr_mem_20799)[sext_i32_i64(squot32(local_tid_20793,
                                                                  32))] =
                x_18904;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_20811;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_20793, 32) == 0 && ltid_in_bounds_20809) {
                x_20804 = ((volatile __local
                            bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)];
                x_20805 = ((volatile __local
                            int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)];
                if ((local_tid_20793 - squot32(local_tid_20793, 32) * 32) ==
                    0) {
                    x_20802 = x_20804;
                    x_20803 = x_20805;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_20811 = 1;
            while (slt32(skip_threads_20811, 32)) {
                if (sle32(skip_threads_20811, local_tid_20793 -
                          squot32(local_tid_20793, 32) * 32) &&
                    (squot32(local_tid_20793, 32) == 0 &&
                     ltid_in_bounds_20809)) {
                    // read operands
                    {
                        x_20802 = ((volatile __local
                                    bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793) -
                                                                sext_i32_i64(skip_threads_20811)];
                        x_20803 = ((volatile __local
                                    int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793) -
                                                                   sext_i32_i64(skip_threads_20811)];
                    }
                    // perform operation
                    {
                        bool res_20806 = x_20802 || x_20804;
                        int64_t res_20807;
                        
                        if (x_20804) {
                            res_20807 = x_20805;
                        } else {
                            int64_t res_20808 = add64(x_20803, x_20805);
                            
                            res_20807 = res_20808;
                        }
                        x_20802 = res_20806;
                        x_20803 = res_20807;
                    }
                }
                if (sle32(wave_sizze_20795, skip_threads_20811)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_20811, local_tid_20793 -
                          squot32(local_tid_20793, 32) * 32) &&
                    (squot32(local_tid_20793, 32) == 0 &&
                     ltid_in_bounds_20809)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)] =
                            x_20802;
                        x_20804 = x_20802;
                        ((volatile __local
                          int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)] =
                            x_20803;
                        x_20805 = x_20803;
                    }
                }
                if (sle32(wave_sizze_20795, skip_threads_20811)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_20811 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_20793, 32) == 0 || !ltid_in_bounds_20809)) {
            // read operands
            {
                x_18905 = x_18903;
                x_18906 = x_18904;
                x_18903 = ((__local
                            bool *) scan_arr_mem_20797)[sext_i32_i64(squot32(local_tid_20793,
                                                                             32)) -
                                                        1];
                x_18904 = ((__local
                            int64_t *) scan_arr_mem_20799)[sext_i32_i64(squot32(local_tid_20793,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                bool res_18907 = x_18903 || x_18905;
                int64_t res_18908;
                
                if (x_18905) {
                    res_18908 = x_18906;
                } else {
                    int64_t res_18909 = add64(x_18904, x_18906);
                    
                    res_18908 = res_18909;
                }
                x_18903 = res_18907;
                x_18904 = res_18908;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)] =
                    x_18903;
                ((__local
                  int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)] =
                    x_18904;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_20793, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)] =
                x_18905;
            ((__local
              int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)] =
                x_18906;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_19947, res_18845)) {
            ((__global bool *) mem_20207)[gtid_19947] = ((__local
                                                          bool *) scan_arr_mem_20797)[sext_i32_i64(local_tid_20793)];
            ((__global int64_t *) mem_20209)[gtid_19947] = ((__local
                                                             int64_t *) scan_arr_mem_20799)[sext_i32_i64(local_tid_20793)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19943
}
__kernel void mainziscan_stage2_19956(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20866_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_20864_backing_aligned_1,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20214, __global
                                      unsigned char *mem_20216,
                                      int64_t stage1_num_groups_20826,
                                      int32_t num_threads_20827)
{
    #define segscan_group_sizze_19951 (mainzisegscan_group_sizze_19950)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20866_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_20866_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_20864_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20864_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20859;
    int32_t local_tid_20860;
    int64_t group_sizze_20863;
    int32_t wave_sizze_20862;
    int32_t group_tid_20861;
    
    global_tid_20859 = get_global_id(0);
    local_tid_20860 = get_local_id(0);
    group_sizze_20863 = get_local_size(0);
    wave_sizze_20862 = LOCKSTEP_WIDTH;
    group_tid_20861 = get_group_id(0);
    
    int32_t phys_tid_19956;
    
    phys_tid_19956 = global_tid_20859;
    
    __local char *scan_arr_mem_20864;
    __local char *scan_arr_mem_20866;
    
    scan_arr_mem_20864 = (__local char *) scan_arr_mem_20864_backing_0;
    scan_arr_mem_20866 = (__local char *) scan_arr_mem_20866_backing_1;
    
    int64_t flat_idx_20868;
    
    flat_idx_20868 = (sext_i32_i64(local_tid_20860) + 1) *
        (segscan_group_sizze_19951 * sdiv_up64(res_18845,
                                               sext_i32_i64(num_threads_20827))) -
        1;
    
    int64_t gtid_19955;
    
    gtid_19955 = flat_idx_20868;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_19955, res_18845)) {
            ((__local
              bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)] =
                ((__global bool *) mem_20214)[gtid_19955];
            ((__local
              float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)] =
                ((__global float *) mem_20216)[gtid_19955];
        } else {
            ((__local
              bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)] = 0;
            ((__local
              float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)] =
                0.0F;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_18929;
    float x_18930;
    bool x_18931;
    float x_18932;
    bool x_20869;
    float x_20870;
    bool x_20871;
    float x_20872;
    bool ltid_in_bounds_20876;
    
    ltid_in_bounds_20876 = slt64(sext_i32_i64(local_tid_20860),
                                 stage1_num_groups_20826);
    
    int32_t skip_threads_20877;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_20876) {
            x_18931 = ((volatile __local
                        bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)];
            x_18932 = ((volatile __local
                        float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)];
            if ((local_tid_20860 - squot32(local_tid_20860, 32) * 32) == 0) {
                x_18929 = x_18931;
                x_18930 = x_18932;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_20877 = 1;
        while (slt32(skip_threads_20877, 32)) {
            if (sle32(skip_threads_20877, local_tid_20860 -
                      squot32(local_tid_20860, 32) * 32) &&
                ltid_in_bounds_20876) {
                // read operands
                {
                    x_18929 = ((volatile __local
                                bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860) -
                                                            sext_i32_i64(skip_threads_20877)];
                    x_18930 = ((volatile __local
                                float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860) -
                                                             sext_i32_i64(skip_threads_20877)];
                }
                // perform operation
                {
                    bool res_18933 = x_18929 || x_18931;
                    float res_18934;
                    
                    if (x_18931) {
                        res_18934 = x_18932;
                    } else {
                        float res_18935 = x_18930 + x_18932;
                        
                        res_18934 = res_18935;
                    }
                    x_18929 = res_18933;
                    x_18930 = res_18934;
                }
            }
            if (sle32(wave_sizze_20862, skip_threads_20877)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_20877, local_tid_20860 -
                      squot32(local_tid_20860, 32) * 32) &&
                ltid_in_bounds_20876) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)] =
                        x_18929;
                    x_18931 = x_18929;
                    ((volatile __local
                      float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)] =
                        x_18930;
                    x_18932 = x_18930;
                }
            }
            if (sle32(wave_sizze_20862, skip_threads_20877)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_20877 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_20860 - squot32(local_tid_20860, 32) * 32) == 31 &&
            ltid_in_bounds_20876) {
            ((volatile __local
              bool *) scan_arr_mem_20864)[sext_i32_i64(squot32(local_tid_20860,
                                                               32))] = x_18929;
            ((volatile __local
              float *) scan_arr_mem_20866)[sext_i32_i64(squot32(local_tid_20860,
                                                                32))] = x_18930;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_20878;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_20860, 32) == 0 && ltid_in_bounds_20876) {
                x_20871 = ((volatile __local
                            bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)];
                x_20872 = ((volatile __local
                            float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)];
                if ((local_tid_20860 - squot32(local_tid_20860, 32) * 32) ==
                    0) {
                    x_20869 = x_20871;
                    x_20870 = x_20872;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_20878 = 1;
            while (slt32(skip_threads_20878, 32)) {
                if (sle32(skip_threads_20878, local_tid_20860 -
                          squot32(local_tid_20860, 32) * 32) &&
                    (squot32(local_tid_20860, 32) == 0 &&
                     ltid_in_bounds_20876)) {
                    // read operands
                    {
                        x_20869 = ((volatile __local
                                    bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860) -
                                                                sext_i32_i64(skip_threads_20878)];
                        x_20870 = ((volatile __local
                                    float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860) -
                                                                 sext_i32_i64(skip_threads_20878)];
                    }
                    // perform operation
                    {
                        bool res_20873 = x_20869 || x_20871;
                        float res_20874;
                        
                        if (x_20871) {
                            res_20874 = x_20872;
                        } else {
                            float res_20875 = x_20870 + x_20872;
                            
                            res_20874 = res_20875;
                        }
                        x_20869 = res_20873;
                        x_20870 = res_20874;
                    }
                }
                if (sle32(wave_sizze_20862, skip_threads_20878)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_20878, local_tid_20860 -
                          squot32(local_tid_20860, 32) * 32) &&
                    (squot32(local_tid_20860, 32) == 0 &&
                     ltid_in_bounds_20876)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)] =
                            x_20869;
                        x_20871 = x_20869;
                        ((volatile __local
                          float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)] =
                            x_20870;
                        x_20872 = x_20870;
                    }
                }
                if (sle32(wave_sizze_20862, skip_threads_20878)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_20878 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_20860, 32) == 0 || !ltid_in_bounds_20876)) {
            // read operands
            {
                x_18931 = x_18929;
                x_18932 = x_18930;
                x_18929 = ((__local
                            bool *) scan_arr_mem_20864)[sext_i32_i64(squot32(local_tid_20860,
                                                                             32)) -
                                                        1];
                x_18930 = ((__local
                            float *) scan_arr_mem_20866)[sext_i32_i64(squot32(local_tid_20860,
                                                                              32)) -
                                                         1];
            }
            // perform operation
            {
                bool res_18933 = x_18929 || x_18931;
                float res_18934;
                
                if (x_18931) {
                    res_18934 = x_18932;
                } else {
                    float res_18935 = x_18930 + x_18932;
                    
                    res_18934 = res_18935;
                }
                x_18929 = res_18933;
                x_18930 = res_18934;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)] =
                    x_18929;
                ((__local
                  float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)] =
                    x_18930;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_20860, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)] =
                x_18931;
            ((__local
              float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)] =
                x_18932;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_19955, res_18845)) {
            ((__global bool *) mem_20214)[gtid_19955] = ((__local
                                                          bool *) scan_arr_mem_20864)[sext_i32_i64(local_tid_20860)];
            ((__global float *) mem_20216)[gtid_19955] = ((__local
                                                           float *) scan_arr_mem_20866)[sext_i32_i64(local_tid_20860)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19951
}
__kernel void mainziscan_stage2_20008(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_20921_backing_aligned_0,
                                      int64_t res_18845, __global
                                      unsigned char *mem_20219,
                                      int64_t stage1_num_groups_20893,
                                      int32_t num_threads_20894)
{
    #define segscan_group_sizze_20003 (mainzisegscan_group_sizze_20002)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_20921_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_20921_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20916;
    int32_t local_tid_20917;
    int64_t group_sizze_20920;
    int32_t wave_sizze_20919;
    int32_t group_tid_20918;
    
    global_tid_20916 = get_global_id(0);
    local_tid_20917 = get_local_id(0);
    group_sizze_20920 = get_local_size(0);
    wave_sizze_20919 = LOCKSTEP_WIDTH;
    group_tid_20918 = get_group_id(0);
    
    int32_t phys_tid_20008;
    
    phys_tid_20008 = global_tid_20916;
    
    __local char *scan_arr_mem_20921;
    
    scan_arr_mem_20921 = (__local char *) scan_arr_mem_20921_backing_0;
    
    int64_t flat_idx_20923;
    
    flat_idx_20923 = (sext_i32_i64(local_tid_20917) + 1) *
        (segscan_group_sizze_20003 * sdiv_up64(res_18845,
                                               sext_i32_i64(num_threads_20894))) -
        1;
    
    int64_t gtid_20007;
    
    gtid_20007 = flat_idx_20923;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_20007, res_18845)) {
            ((__local
              int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)] =
                ((__global int64_t *) mem_20219)[gtid_20007];
        } else {
            ((__local
              int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int64_t x_19033;
    int64_t x_19034;
    int64_t x_20924;
    int64_t x_20925;
    bool ltid_in_bounds_20927;
    
    ltid_in_bounds_20927 = slt64(sext_i32_i64(local_tid_20917),
                                 stage1_num_groups_20893);
    
    int32_t skip_threads_20928;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_20927) {
            x_19034 = ((volatile __local
                        int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)];
            if ((local_tid_20917 - squot32(local_tid_20917, 32) * 32) == 0) {
                x_19033 = x_19034;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_20928 = 1;
        while (slt32(skip_threads_20928, 32)) {
            if (sle32(skip_threads_20928, local_tid_20917 -
                      squot32(local_tid_20917, 32) * 32) &&
                ltid_in_bounds_20927) {
                // read operands
                {
                    x_19033 = ((volatile __local
                                int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917) -
                                                               sext_i32_i64(skip_threads_20928)];
                }
                // perform operation
                {
                    int64_t res_19035 = add64(x_19033, x_19034);
                    
                    x_19033 = res_19035;
                }
            }
            if (sle32(wave_sizze_20919, skip_threads_20928)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_20928, local_tid_20917 -
                      squot32(local_tid_20917, 32) * 32) &&
                ltid_in_bounds_20927) {
                // write result
                {
                    ((volatile __local
                      int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)] =
                        x_19033;
                    x_19034 = x_19033;
                }
            }
            if (sle32(wave_sizze_20919, skip_threads_20928)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_20928 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_20917 - squot32(local_tid_20917, 32) * 32) == 31 &&
            ltid_in_bounds_20927) {
            ((volatile __local
              int64_t *) scan_arr_mem_20921)[sext_i32_i64(squot32(local_tid_20917,
                                                                  32))] =
                x_19033;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_20929;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_20917, 32) == 0 && ltid_in_bounds_20927) {
                x_20925 = ((volatile __local
                            int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)];
                if ((local_tid_20917 - squot32(local_tid_20917, 32) * 32) ==
                    0) {
                    x_20924 = x_20925;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_20929 = 1;
            while (slt32(skip_threads_20929, 32)) {
                if (sle32(skip_threads_20929, local_tid_20917 -
                          squot32(local_tid_20917, 32) * 32) &&
                    (squot32(local_tid_20917, 32) == 0 &&
                     ltid_in_bounds_20927)) {
                    // read operands
                    {
                        x_20924 = ((volatile __local
                                    int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917) -
                                                                   sext_i32_i64(skip_threads_20929)];
                    }
                    // perform operation
                    {
                        int64_t res_20926 = add64(x_20924, x_20925);
                        
                        x_20924 = res_20926;
                    }
                }
                if (sle32(wave_sizze_20919, skip_threads_20929)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_20929, local_tid_20917 -
                          squot32(local_tid_20917, 32) * 32) &&
                    (squot32(local_tid_20917, 32) == 0 &&
                     ltid_in_bounds_20927)) {
                    // write result
                    {
                        ((volatile __local
                          int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)] =
                            x_20924;
                        x_20925 = x_20924;
                    }
                }
                if (sle32(wave_sizze_20919, skip_threads_20929)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_20929 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_20917, 32) == 0 || !ltid_in_bounds_20927)) {
            // read operands
            {
                x_19034 = x_19033;
                x_19033 = ((__local
                            int64_t *) scan_arr_mem_20921)[sext_i32_i64(squot32(local_tid_20917,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                int64_t res_19035 = add64(x_19033, x_19034);
                
                x_19033 = res_19035;
            }
            // write final result
            {
                ((__local
                  int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)] =
                    x_19033;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_20917, 32) == 0) {
            ((__local
              int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)] =
                x_19034;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_20007, res_18845)) {
            ((__global int64_t *) mem_20219)[gtid_20007] = ((__local
                                                             int64_t *) scan_arr_mem_20921)[sext_i32_i64(local_tid_20917)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_20003
}
__kernel void mainziscan_stage3_19907(__global int *global_failure,
                                      int64_t paths_18599,
                                      int64_t num_groups_19904, __global
                                      unsigned char *mem_20191,
                                      int32_t num_threads_20456,
                                      int32_t required_groups_20492)
{
    #define segscan_group_sizze_19902 (mainzisegscan_group_sizze_19901)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20493;
    int32_t local_tid_20494;
    int64_t group_sizze_20497;
    int32_t wave_sizze_20496;
    int32_t group_tid_20495;
    
    global_tid_20493 = get_global_id(0);
    local_tid_20494 = get_local_id(0);
    group_sizze_20497 = get_local_size(0);
    wave_sizze_20496 = LOCKSTEP_WIDTH;
    group_tid_20495 = get_group_id(0);
    
    int32_t phys_tid_19907;
    
    phys_tid_19907 = global_tid_20493;
    
    int32_t phys_group_id_20498;
    
    phys_group_id_20498 = get_group_id(0);
    for (int32_t i_20499 = 0; i_20499 < sdiv_up32(required_groups_20492 -
                                                  phys_group_id_20498,
                                                  sext_i64_i32(num_groups_19904));
         i_20499++) {
        int32_t virt_group_id_20500 = phys_group_id_20498 + i_20499 *
                sext_i64_i32(num_groups_19904);
        int64_t flat_idx_20501 = sext_i32_i64(virt_group_id_20500) *
                segscan_group_sizze_19902 + sext_i32_i64(local_tid_20494);
        int64_t gtid_19906 = flat_idx_20501;
        int64_t orig_group_20502 = squot64(flat_idx_20501,
                                           segscan_group_sizze_19902 *
                                           sdiv_up64(paths_18599,
                                                     sext_i32_i64(num_threads_20456)));
        int64_t carry_in_flat_idx_20503 = orig_group_20502 *
                (segscan_group_sizze_19902 * sdiv_up64(paths_18599,
                                                       sext_i32_i64(num_threads_20456))) -
                1;
        
        if (slt64(gtid_19906, paths_18599)) {
            if (!(orig_group_20502 == 0 || flat_idx_20501 == (orig_group_20502 +
                                                              1) *
                  (segscan_group_sizze_19902 * sdiv_up64(paths_18599,
                                                         sext_i32_i64(num_threads_20456))) -
                  1)) {
                int64_t x_18830;
                int64_t x_18831;
                
                x_18830 = ((__global
                            int64_t *) mem_20191)[carry_in_flat_idx_20503];
                x_18831 = ((__global int64_t *) mem_20191)[gtid_19906];
                
                int64_t res_18832;
                
                res_18832 = add64(x_18830, x_18831);
                x_18830 = res_18832;
                ((__global int64_t *) mem_20191)[gtid_19906] = x_18830;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19902
}
__kernel void mainziscan_stage3_19940(__global int *global_failure,
                                      int64_t res_18845,
                                      int64_t num_groups_19937, __global
                                      unsigned char *mem_20202, __global
                                      unsigned char *mem_20204,
                                      int32_t num_threads_20693,
                                      int32_t required_groups_20745)
{
    #define segscan_group_sizze_19935 (mainzisegscan_group_sizze_19934)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20746;
    int32_t local_tid_20747;
    int64_t group_sizze_20750;
    int32_t wave_sizze_20749;
    int32_t group_tid_20748;
    
    global_tid_20746 = get_global_id(0);
    local_tid_20747 = get_local_id(0);
    group_sizze_20750 = get_local_size(0);
    wave_sizze_20749 = LOCKSTEP_WIDTH;
    group_tid_20748 = get_group_id(0);
    
    int32_t phys_tid_19940;
    
    phys_tid_19940 = global_tid_20746;
    
    int32_t phys_group_id_20751;
    
    phys_group_id_20751 = get_group_id(0);
    for (int32_t i_20752 = 0; i_20752 < sdiv_up32(required_groups_20745 -
                                                  phys_group_id_20751,
                                                  sext_i64_i32(num_groups_19937));
         i_20752++) {
        int32_t virt_group_id_20753 = phys_group_id_20751 + i_20752 *
                sext_i64_i32(num_groups_19937);
        int64_t flat_idx_20754 = sext_i32_i64(virt_group_id_20753) *
                segscan_group_sizze_19935 + sext_i32_i64(local_tid_20747);
        int64_t gtid_19939 = flat_idx_20754;
        int64_t orig_group_20755 = squot64(flat_idx_20754,
                                           segscan_group_sizze_19935 *
                                           sdiv_up64(res_18845,
                                                     sext_i32_i64(num_threads_20693)));
        int64_t carry_in_flat_idx_20756 = orig_group_20755 *
                (segscan_group_sizze_19935 * sdiv_up64(res_18845,
                                                       sext_i32_i64(num_threads_20693))) -
                1;
        
        if (slt64(gtid_19939, res_18845)) {
            if (!(orig_group_20755 == 0 || flat_idx_20754 == (orig_group_20755 +
                                                              1) *
                  (segscan_group_sizze_19935 * sdiv_up64(res_18845,
                                                         sext_i32_i64(num_threads_20693))) -
                  1)) {
                bool x_18864;
                int64_t x_18865;
                bool x_18866;
                int64_t x_18867;
                
                x_18864 = ((__global
                            bool *) mem_20202)[carry_in_flat_idx_20756];
                x_18865 = ((__global
                            int64_t *) mem_20204)[carry_in_flat_idx_20756];
                x_18866 = ((__global bool *) mem_20202)[gtid_19939];
                x_18867 = ((__global int64_t *) mem_20204)[gtid_19939];
                
                bool res_18868;
                
                res_18868 = x_18864 || x_18866;
                
                int64_t res_18869;
                
                if (x_18866) {
                    res_18869 = x_18867;
                } else {
                    int64_t res_18870 = add64(x_18865, x_18867);
                    
                    res_18869 = res_18870;
                }
                x_18864 = res_18868;
                x_18865 = res_18869;
                ((__global bool *) mem_20202)[gtid_19939] = x_18864;
                ((__global int64_t *) mem_20204)[gtid_19939] = x_18865;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19935
}
__kernel void mainziscan_stage3_19948(__global int *global_failure,
                                      int64_t res_18845,
                                      int64_t num_groups_19945, __global
                                      unsigned char *mem_20207, __global
                                      unsigned char *mem_20209,
                                      int32_t num_threads_20760,
                                      int32_t required_groups_20812)
{
    #define segscan_group_sizze_19943 (mainzisegscan_group_sizze_19942)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20813;
    int32_t local_tid_20814;
    int64_t group_sizze_20817;
    int32_t wave_sizze_20816;
    int32_t group_tid_20815;
    
    global_tid_20813 = get_global_id(0);
    local_tid_20814 = get_local_id(0);
    group_sizze_20817 = get_local_size(0);
    wave_sizze_20816 = LOCKSTEP_WIDTH;
    group_tid_20815 = get_group_id(0);
    
    int32_t phys_tid_19948;
    
    phys_tid_19948 = global_tid_20813;
    
    int32_t phys_group_id_20818;
    
    phys_group_id_20818 = get_group_id(0);
    for (int32_t i_20819 = 0; i_20819 < sdiv_up32(required_groups_20812 -
                                                  phys_group_id_20818,
                                                  sext_i64_i32(num_groups_19945));
         i_20819++) {
        int32_t virt_group_id_20820 = phys_group_id_20818 + i_20819 *
                sext_i64_i32(num_groups_19945);
        int64_t flat_idx_20821 = sext_i32_i64(virt_group_id_20820) *
                segscan_group_sizze_19943 + sext_i32_i64(local_tid_20814);
        int64_t gtid_19947 = flat_idx_20821;
        int64_t orig_group_20822 = squot64(flat_idx_20821,
                                           segscan_group_sizze_19943 *
                                           sdiv_up64(res_18845,
                                                     sext_i32_i64(num_threads_20760)));
        int64_t carry_in_flat_idx_20823 = orig_group_20822 *
                (segscan_group_sizze_19943 * sdiv_up64(res_18845,
                                                       sext_i32_i64(num_threads_20760))) -
                1;
        
        if (slt64(gtid_19947, res_18845)) {
            if (!(orig_group_20822 == 0 || flat_idx_20821 == (orig_group_20822 +
                                                              1) *
                  (segscan_group_sizze_19943 * sdiv_up64(res_18845,
                                                         sext_i32_i64(num_threads_20760))) -
                  1)) {
                bool x_18903;
                int64_t x_18904;
                bool x_18905;
                int64_t x_18906;
                
                x_18903 = ((__global
                            bool *) mem_20207)[carry_in_flat_idx_20823];
                x_18904 = ((__global
                            int64_t *) mem_20209)[carry_in_flat_idx_20823];
                x_18905 = ((__global bool *) mem_20207)[gtid_19947];
                x_18906 = ((__global int64_t *) mem_20209)[gtid_19947];
                
                bool res_18907;
                
                res_18907 = x_18903 || x_18905;
                
                int64_t res_18908;
                
                if (x_18905) {
                    res_18908 = x_18906;
                } else {
                    int64_t res_18909 = add64(x_18904, x_18906);
                    
                    res_18908 = res_18909;
                }
                x_18903 = res_18907;
                x_18904 = res_18908;
                ((__global bool *) mem_20207)[gtid_19947] = x_18903;
                ((__global int64_t *) mem_20209)[gtid_19947] = x_18904;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19943
}
__kernel void mainziscan_stage3_19956(__global int *global_failure,
                                      int64_t res_18845,
                                      int64_t num_groups_19953, __global
                                      unsigned char *mem_20214, __global
                                      unsigned char *mem_20216,
                                      int32_t num_threads_20827,
                                      int32_t required_groups_20879)
{
    #define segscan_group_sizze_19951 (mainzisegscan_group_sizze_19950)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20880;
    int32_t local_tid_20881;
    int64_t group_sizze_20884;
    int32_t wave_sizze_20883;
    int32_t group_tid_20882;
    
    global_tid_20880 = get_global_id(0);
    local_tid_20881 = get_local_id(0);
    group_sizze_20884 = get_local_size(0);
    wave_sizze_20883 = LOCKSTEP_WIDTH;
    group_tid_20882 = get_group_id(0);
    
    int32_t phys_tid_19956;
    
    phys_tid_19956 = global_tid_20880;
    
    int32_t phys_group_id_20885;
    
    phys_group_id_20885 = get_group_id(0);
    for (int32_t i_20886 = 0; i_20886 < sdiv_up32(required_groups_20879 -
                                                  phys_group_id_20885,
                                                  sext_i64_i32(num_groups_19953));
         i_20886++) {
        int32_t virt_group_id_20887 = phys_group_id_20885 + i_20886 *
                sext_i64_i32(num_groups_19953);
        int64_t flat_idx_20888 = sext_i32_i64(virt_group_id_20887) *
                segscan_group_sizze_19951 + sext_i32_i64(local_tid_20881);
        int64_t gtid_19955 = flat_idx_20888;
        int64_t orig_group_20889 = squot64(flat_idx_20888,
                                           segscan_group_sizze_19951 *
                                           sdiv_up64(res_18845,
                                                     sext_i32_i64(num_threads_20827)));
        int64_t carry_in_flat_idx_20890 = orig_group_20889 *
                (segscan_group_sizze_19951 * sdiv_up64(res_18845,
                                                       sext_i32_i64(num_threads_20827))) -
                1;
        
        if (slt64(gtid_19955, res_18845)) {
            if (!(orig_group_20889 == 0 || flat_idx_20888 == (orig_group_20889 +
                                                              1) *
                  (segscan_group_sizze_19951 * sdiv_up64(res_18845,
                                                         sext_i32_i64(num_threads_20827))) -
                  1)) {
                bool x_18929;
                float x_18930;
                bool x_18931;
                float x_18932;
                
                x_18929 = ((__global
                            bool *) mem_20214)[carry_in_flat_idx_20890];
                x_18930 = ((__global
                            float *) mem_20216)[carry_in_flat_idx_20890];
                x_18931 = ((__global bool *) mem_20214)[gtid_19955];
                x_18932 = ((__global float *) mem_20216)[gtid_19955];
                
                bool res_18933;
                
                res_18933 = x_18929 || x_18931;
                
                float res_18934;
                
                if (x_18931) {
                    res_18934 = x_18932;
                } else {
                    float res_18935 = x_18930 + x_18932;
                    
                    res_18934 = res_18935;
                }
                x_18929 = res_18933;
                x_18930 = res_18934;
                ((__global bool *) mem_20214)[gtid_19955] = x_18929;
                ((__global float *) mem_20216)[gtid_19955] = x_18930;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_19951
}
__kernel void mainziscan_stage3_20008(__global int *global_failure,
                                      int64_t res_18845,
                                      int64_t num_groups_20005, __global
                                      unsigned char *mem_20219,
                                      int32_t num_threads_20894,
                                      int32_t required_groups_20930)
{
    #define segscan_group_sizze_20003 (mainzisegscan_group_sizze_20002)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20931;
    int32_t local_tid_20932;
    int64_t group_sizze_20935;
    int32_t wave_sizze_20934;
    int32_t group_tid_20933;
    
    global_tid_20931 = get_global_id(0);
    local_tid_20932 = get_local_id(0);
    group_sizze_20935 = get_local_size(0);
    wave_sizze_20934 = LOCKSTEP_WIDTH;
    group_tid_20933 = get_group_id(0);
    
    int32_t phys_tid_20008;
    
    phys_tid_20008 = global_tid_20931;
    
    int32_t phys_group_id_20936;
    
    phys_group_id_20936 = get_group_id(0);
    for (int32_t i_20937 = 0; i_20937 < sdiv_up32(required_groups_20930 -
                                                  phys_group_id_20936,
                                                  sext_i64_i32(num_groups_20005));
         i_20937++) {
        int32_t virt_group_id_20938 = phys_group_id_20936 + i_20937 *
                sext_i64_i32(num_groups_20005);
        int64_t flat_idx_20939 = sext_i32_i64(virt_group_id_20938) *
                segscan_group_sizze_20003 + sext_i32_i64(local_tid_20932);
        int64_t gtid_20007 = flat_idx_20939;
        int64_t orig_group_20940 = squot64(flat_idx_20939,
                                           segscan_group_sizze_20003 *
                                           sdiv_up64(res_18845,
                                                     sext_i32_i64(num_threads_20894)));
        int64_t carry_in_flat_idx_20941 = orig_group_20940 *
                (segscan_group_sizze_20003 * sdiv_up64(res_18845,
                                                       sext_i32_i64(num_threads_20894))) -
                1;
        
        if (slt64(gtid_20007, res_18845)) {
            if (!(orig_group_20940 == 0 || flat_idx_20939 == (orig_group_20940 +
                                                              1) *
                  (segscan_group_sizze_20003 * sdiv_up64(res_18845,
                                                         sext_i32_i64(num_threads_20894))) -
                  1)) {
                int64_t x_19033;
                int64_t x_19034;
                
                x_19033 = ((__global
                            int64_t *) mem_20219)[carry_in_flat_idx_20941];
                x_19034 = ((__global int64_t *) mem_20219)[gtid_20007];
                
                int64_t res_19035;
                
                res_19035 = add64(x_19033, x_19034);
                x_19033 = res_19035;
                ((__global int64_t *) mem_20219)[gtid_20007] = x_19033;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_20003
}
__kernel void mainziseghist_global_19925(__global int *global_failure,
                                         int64_t paths_18599, int64_t res_18845,
                                         int64_t num_groups_19922, __global
                                         unsigned char *mem_20191,
                                         int32_t num_subhistos_20543, __global
                                         unsigned char *res_subhistos_mem_20544,
                                         __global
                                         unsigned char *mainzihist_locks_mem_20614,
                                         int32_t chk_i_20616,
                                         int64_t hist_H_chk_20617)
{
    #define seghist_group_sizze_19920 (mainziseghist_group_sizze_19919)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20618;
    int32_t local_tid_20619;
    int64_t group_sizze_20622;
    int32_t wave_sizze_20621;
    int32_t group_tid_20620;
    
    global_tid_20618 = get_global_id(0);
    local_tid_20619 = get_local_id(0);
    group_sizze_20622 = get_local_size(0);
    wave_sizze_20621 = LOCKSTEP_WIDTH;
    group_tid_20620 = get_group_id(0);
    
    int32_t phys_tid_19925;
    
    phys_tid_19925 = global_tid_20618;
    
    int32_t subhisto_ind_20623;
    
    subhisto_ind_20623 = squot32(global_tid_20618,
                                 sdiv_up32(sext_i64_i32(seghist_group_sizze_19920 *
                                           num_groups_19922),
                                           num_subhistos_20543));
    for (int64_t i_20624 = 0; i_20624 < sdiv_up64(paths_18599 -
                                                  sext_i32_i64(global_tid_20618),
                                                  sext_i32_i64(sext_i64_i32(seghist_group_sizze_19920 *
                                                  num_groups_19922)));
         i_20624++) {
        int32_t gtid_19924 = sext_i64_i32(i_20624 *
                sext_i32_i64(sext_i64_i32(seghist_group_sizze_19920 *
                num_groups_19922)) + sext_i32_i64(global_tid_20618));
        
        if (slt64(i_20624 *
                  sext_i32_i64(sext_i64_i32(seghist_group_sizze_19920 *
                  num_groups_19922)) + sext_i32_i64(global_tid_20618),
                  paths_18599)) {
            int64_t i_p_o_20040 = add64(-1, gtid_19924);
            int64_t rot_i_20041 = smod64(i_p_o_20040, paths_18599);
            bool cond_19931 = gtid_19924 == 0;
            int64_t res_19932;
            
            if (cond_19931) {
                res_19932 = 0;
            } else {
                int64_t x_19930 = ((__global int64_t *) mem_20191)[rot_i_20041];
                
                res_19932 = x_19930;
            }
            // save map-out results
            { }
            // perform atomic updates
            {
                if (sle64(sext_i32_i64(chk_i_20616) * hist_H_chk_20617,
                          res_19932) && (slt64(res_19932,
                                               sext_i32_i64(chk_i_20616) *
                                               hist_H_chk_20617 +
                                               hist_H_chk_20617) &&
                                         slt64(res_19932, res_18845))) {
                    int64_t x_19926;
                    int64_t x_19927;
                    
                    x_19927 = gtid_19924;
                    
                    int32_t old_20625;
                    volatile bool continue_20626;
                    
                    continue_20626 = 1;
                    while (continue_20626) {
                        old_20625 =
                            atomic_cmpxchg_i32_global(&((volatile __global
                                                         int *) mainzihist_locks_mem_20614)[srem64(sext_i32_i64(subhisto_ind_20623) *
                                                                                                   res_18845 +
                                                                                                   res_19932,
                                                                                                   100151)],
                                                      0, 1);
                        if (old_20625 == 0) {
                            int64_t x_19926;
                            
                            // bind lhs
                            {
                                x_19926 = ((volatile __global
                                            int64_t *) res_subhistos_mem_20544)[sext_i32_i64(subhisto_ind_20623) *
                                                                                res_18845 +
                                                                                res_19932];
                            }
                            // execute operation
                            {
                                int64_t res_19928 = smax64(x_19926, x_19927);
                                
                                x_19926 = res_19928;
                            }
                            // update global result
                            {
                                ((volatile __global
                                  int64_t *) res_subhistos_mem_20544)[sext_i32_i64(subhisto_ind_20623) *
                                                                      res_18845 +
                                                                      res_19932] =
                                    x_19926;
                            }
                            mem_fence_global();
                            old_20625 =
                                atomic_cmpxchg_i32_global(&((volatile __global
                                                             int *) mainzihist_locks_mem_20614)[srem64(sext_i32_i64(subhisto_ind_20623) *
                                                                                                       res_18845 +
                                                                                                       res_19932,
                                                                                                       100151)],
                                                          1, 0);
                            continue_20626 = 0;
                        }
                        mem_fence_global();
                    }
                }
            }
        }
    }
    
  error_0:
    return;
    #undef seghist_group_sizze_19920
}
__kernel void mainziseghist_local_19925(__global int *global_failure,
                                        __local volatile
                                        int64_t *locks_mem_20584_backing_aligned_0,
                                        __local volatile
                                        int64_t *subhistogram_local_mem_20582_backing_aligned_1,
                                        int64_t paths_18599, int64_t res_18845,
                                        __global unsigned char *mem_20191,
                                        __global
                                        unsigned char *res_subhistos_mem_20544,
                                        int32_t max_group_sizze_20553,
                                        int64_t num_groups_20554,
                                        int32_t hist_M_20560,
                                        int32_t chk_i_20565,
                                        int64_t num_segments_20566,
                                        int64_t hist_H_chk_20567,
                                        int64_t histo_sizze_20568,
                                        int32_t init_per_thread_20569)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict locks_mem_20584_backing_1 =
                          (__local volatile
                           char *) locks_mem_20584_backing_aligned_0;
    __local volatile char *restrict subhistogram_local_mem_20582_backing_0 =
                          (__local volatile
                           char *) subhistogram_local_mem_20582_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20570;
    int32_t local_tid_20571;
    int64_t group_sizze_20574;
    int32_t wave_sizze_20573;
    int32_t group_tid_20572;
    
    global_tid_20570 = get_global_id(0);
    local_tid_20571 = get_local_id(0);
    group_sizze_20574 = get_local_size(0);
    wave_sizze_20573 = LOCKSTEP_WIDTH;
    group_tid_20572 = get_group_id(0);
    
    int32_t phys_tid_19925;
    
    phys_tid_19925 = global_tid_20570;
    
    int32_t phys_group_id_20575;
    
    phys_group_id_20575 = get_group_id(0);
    for (int32_t i_20576 = 0; i_20576 <
         sdiv_up32(sext_i64_i32(num_groups_20554 * num_segments_20566) -
                   phys_group_id_20575, sext_i64_i32(num_groups_20554));
         i_20576++) {
        int32_t virt_group_id_20577 = phys_group_id_20575 + i_20576 *
                sext_i64_i32(num_groups_20554);
        int32_t flat_segment_id_20578 = squot32(virt_group_id_20577,
                                                sext_i64_i32(num_groups_20554));
        int32_t gid_in_segment_20579 = srem32(virt_group_id_20577,
                                              sext_i64_i32(num_groups_20554));
        int32_t pgtid_in_segment_20580 = gid_in_segment_20579 *
                sext_i64_i32(max_group_sizze_20553) + local_tid_20571;
        int32_t threads_per_segment_20581 = sext_i64_i32(num_groups_20554 *
                max_group_sizze_20553);
        __local char *subhistogram_local_mem_20582;
        
        subhistogram_local_mem_20582 = (__local
                                        char *) subhistogram_local_mem_20582_backing_0;
        
        __local char *locks_mem_20584;
        
        locks_mem_20584 = (__local char *) locks_mem_20584_backing_1;
        // All locks start out unlocked
        {
            for (int64_t i_20586 = 0; i_20586 < sdiv_up64(hist_M_20560 *
                                                          hist_H_chk_20567 -
                                                          sext_i32_i64(local_tid_20571),
                                                          max_group_sizze_20553);
                 i_20586++) {
                ((__local int32_t *) locks_mem_20584)[squot64(i_20586 *
                                                              max_group_sizze_20553 +
                                                              sext_i32_i64(local_tid_20571),
                                                              hist_H_chk_20567) *
                                                      hist_H_chk_20567 +
                                                      (i_20586 *
                                                       max_group_sizze_20553 +
                                                       sext_i32_i64(local_tid_20571) -
                                                       squot64(i_20586 *
                                                               max_group_sizze_20553 +
                                                               sext_i32_i64(local_tid_20571),
                                                               hist_H_chk_20567) *
                                                       hist_H_chk_20567)] = 0;
            }
        }
        
        int32_t thread_local_subhisto_i_20587;
        
        thread_local_subhisto_i_20587 = srem32(local_tid_20571, hist_M_20560);
        // initialize histograms in local memory
        {
            for (int32_t local_i_20588 = 0; local_i_20588 <
                 init_per_thread_20569; local_i_20588++) {
                int32_t j_20589 = local_i_20588 *
                        sext_i64_i32(max_group_sizze_20553) + local_tid_20571;
                int32_t j_offset_20590 = hist_M_20560 *
                        sext_i64_i32(histo_sizze_20568) * gid_in_segment_20579 +
                        j_20589;
                int32_t local_subhisto_i_20591 = squot32(j_20589,
                                                         sext_i64_i32(histo_sizze_20568));
                int32_t global_subhisto_i_20592 = squot32(j_offset_20590,
                                                          sext_i64_i32(histo_sizze_20568));
                
                if (slt32(j_20589, hist_M_20560 *
                          sext_i64_i32(histo_sizze_20568))) {
                    // First subhistogram is initialised from global memory; others with neutral element.
                    {
                        if (global_subhisto_i_20592 == 0) {
                            ((__local
                              int64_t *) subhistogram_local_mem_20582)[sext_i32_i64(local_subhisto_i_20591) *
                                                                       hist_H_chk_20567 +
                                                                       sext_i32_i64(srem32(j_20589,
                                                                                           sext_i64_i32(histo_sizze_20568)))] =
                                ((__global
                                  int64_t *) res_subhistos_mem_20544)[sext_i32_i64(srem32(j_20589,
                                                                                          sext_i64_i32(histo_sizze_20568))) +
                                                                      sext_i32_i64(chk_i_20565) *
                                                                      hist_H_chk_20567];
                        } else {
                            ((__local
                              int64_t *) subhistogram_local_mem_20582)[sext_i32_i64(local_subhisto_i_20591) *
                                                                       hist_H_chk_20567 +
                                                                       sext_i32_i64(srem32(j_20589,
                                                                                           sext_i64_i32(histo_sizze_20568)))] =
                                0;
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_20593 = 0; i_20593 <
             sdiv_up32(sext_i64_i32(paths_18599) - pgtid_in_segment_20580,
                       threads_per_segment_20581); i_20593++) {
            int32_t gtid_19924 = i_20593 * threads_per_segment_20581 +
                    pgtid_in_segment_20580;
            int64_t i_p_o_20040 = add64(-1, gtid_19924);
            int64_t rot_i_20041 = smod64(i_p_o_20040, paths_18599);
            bool cond_19931 = gtid_19924 == 0;
            int64_t res_19932;
            
            if (cond_19931) {
                res_19932 = 0;
            } else {
                int64_t x_19930 = ((__global int64_t *) mem_20191)[rot_i_20041];
                
                res_19932 = x_19930;
            }
            if (chk_i_20565 == 0) {
                // save map-out results
                { }
            }
            // perform atomic updates
            {
                if (slt64(res_19932, res_18845) &&
                    (sle64(sext_i32_i64(chk_i_20565) * hist_H_chk_20567,
                           res_19932) && slt64(res_19932,
                                               sext_i32_i64(chk_i_20565) *
                                               hist_H_chk_20567 +
                                               hist_H_chk_20567))) {
                    int64_t x_19926;
                    int64_t x_19927;
                    
                    x_19927 = gtid_19924;
                    
                    int32_t old_20594;
                    volatile bool continue_20595;
                    
                    continue_20595 = 1;
                    while (continue_20595) {
                        old_20594 = atomic_cmpxchg_i32_local(&((volatile __local
                                                                int *) locks_mem_20584)[sext_i32_i64(thread_local_subhisto_i_20587) *
                                                                                        hist_H_chk_20567 +
                                                                                        (res_19932 -
                                                                                         sext_i32_i64(chk_i_20565) *
                                                                                         hist_H_chk_20567)],
                                                             0, 1);
                        if (old_20594 == 0) {
                            int64_t x_19926;
                            
                            // bind lhs
                            {
                                x_19926 = ((volatile __local
                                            int64_t *) subhistogram_local_mem_20582)[sext_i32_i64(thread_local_subhisto_i_20587) *
                                                                                     hist_H_chk_20567 +
                                                                                     (res_19932 -
                                                                                      sext_i32_i64(chk_i_20565) *
                                                                                      hist_H_chk_20567)];
                            }
                            // execute operation
                            {
                                int64_t res_19928 = smax64(x_19926, x_19927);
                                
                                x_19926 = res_19928;
                            }
                            // update global result
                            {
                                ((volatile __local
                                  int64_t *) subhistogram_local_mem_20582)[sext_i32_i64(thread_local_subhisto_i_20587) *
                                                                           hist_H_chk_20567 +
                                                                           (res_19932 -
                                                                            sext_i32_i64(chk_i_20565) *
                                                                            hist_H_chk_20567)] =
                                    x_19926;
                            }
                            mem_fence_local();
                            old_20594 =
                                atomic_cmpxchg_i32_local(&((volatile __local
                                                            int *) locks_mem_20584)[sext_i32_i64(thread_local_subhisto_i_20587) *
                                                                                    hist_H_chk_20567 +
                                                                                    (res_19932 -
                                                                                     sext_i32_i64(chk_i_20565) *
                                                                                     hist_H_chk_20567)],
                                                         1, 0);
                            continue_20595 = 0;
                        }
                        mem_fence_local();
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compact the multiple local memory subhistograms to result in global memory
        {
            int64_t trunc_H_20596 = smin64(hist_H_chk_20567, res_18845 -
                                           sext_i32_i64(chk_i_20565) *
                                           hist_H_chk_20567);
            int32_t histo_sizze_20597 = sext_i64_i32(trunc_H_20596);
            
            for (int32_t local_i_20598 = 0; local_i_20598 <
                 init_per_thread_20569; local_i_20598++) {
                int32_t j_20599 = local_i_20598 *
                        sext_i64_i32(max_group_sizze_20553) + local_tid_20571;
                
                if (slt32(j_20599, histo_sizze_20597)) {
                    int64_t x_19926;
                    int64_t x_19927;
                    
                    // Read values from subhistogram 0.
                    {
                        x_19926 = ((__local
                                    int64_t *) subhistogram_local_mem_20582)[sext_i32_i64(j_20599)];
                    }
                    // Accumulate based on values in other subhistograms.
                    {
                        for (int32_t subhisto_id_20600 = 0; subhisto_id_20600 <
                             hist_M_20560 - 1; subhisto_id_20600++) {
                            x_19927 = ((__local
                                        int64_t *) subhistogram_local_mem_20582)[(sext_i32_i64(subhisto_id_20600) +
                                                                                  1) *
                                                                                 hist_H_chk_20567 +
                                                                                 sext_i32_i64(j_20599)];
                            
                            int64_t res_19928;
                            
                            res_19928 = smax64(x_19926, x_19927);
                            x_19926 = res_19928;
                        }
                    }
                    // Put final bucket value in global memory.
                    {
                        ((__global
                          int64_t *) res_subhistos_mem_20544)[srem64(sext_i32_i64(virt_group_id_20577),
                                                                     num_groups_20554) *
                                                              res_18845 +
                                                              (sext_i32_i64(j_20599) +
                                                               sext_i32_i64(chk_i_20565) *
                                                               hist_H_chk_20567)] =
                            x_19926;
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
}
__kernel void mainzisegmap_19109(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args, int64_t n_18596,
                                 float a_18604, float r0_18607, float x_18624,
                                 float x_18626, float y_18628, float y_18629,
                                 __global unsigned char *swap_term_mem_20096,
                                 __global unsigned char *payments_mem_20097,
                                 __global unsigned char *mem_20110, __global
                                 unsigned char *mem_20124,
                                 int64_t num_threads_20240)
{
    #define segmap_group_sizze_19187 (mainzisegmap_group_sizze_19111)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20381;
    int32_t local_tid_20382;
    int64_t group_sizze_20385;
    int32_t wave_sizze_20384;
    int32_t group_tid_20383;
    
    global_tid_20381 = get_global_id(0);
    local_tid_20382 = get_local_id(0);
    group_sizze_20385 = get_local_size(0);
    wave_sizze_20384 = LOCKSTEP_WIDTH;
    group_tid_20383 = get_group_id(0);
    
    int32_t phys_tid_19109;
    
    phys_tid_19109 = global_tid_20381;
    
    int64_t gtid_19108;
    
    gtid_19108 = sext_i32_i64(group_tid_20383) * segmap_group_sizze_19187 +
        sext_i32_i64(local_tid_20382);
    if (slt64(gtid_19108, n_18596)) {
        float res_19198 = ((__global float *) swap_term_mem_20096)[gtid_19108];
        int64_t res_19199 = ((__global
                              int64_t *) payments_mem_20097)[gtid_19108];
        int64_t range_end_19201 = sub64(res_19199, 1);
        bool bounds_invalid_upwards_19202 = slt64(range_end_19201, 0);
        bool valid_19203 = !bounds_invalid_upwards_19202;
        bool range_valid_c_19204;
        
        if (!valid_19203) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = 1;
                    global_failure_args[2] = range_end_19201;
                    ;
                }
                return;
            }
        }
        for (int64_t i_20080 = 0; i_20080 < res_19199; i_20080++) {
            float res_19208 = sitofp_i64_f32(i_20080);
            float res_19209 = res_19198 * res_19208;
            
            ((__global float *) mem_20110)[phys_tid_19109 + i_20080 *
                                           num_threads_20240] = res_19209;
        }
        
        bool y_19210 = slt64(0, res_19199);
        bool index_certs_19211;
        
        if (!y_19210) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 1) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = res_19199;
                    ;
                }
                return;
            }
        }
        
        float binop_y_19212 = sitofp_i64_f32(range_end_19201);
        float index_primexp_19213 = res_19198 * binop_y_19212;
        float negate_arg_19214 = a_18604 * index_primexp_19213;
        float exp_arg_19215 = 0.0F - negate_arg_19214;
        float res_19216 = fpow32(2.7182817F, exp_arg_19215);
        float x_19217 = 1.0F - res_19216;
        float B_19218 = x_19217 / a_18604;
        float x_19219 = B_19218 - index_primexp_19213;
        float x_19220 = y_18628 * x_19219;
        float A1_19221 = x_19220 / x_18624;
        float y_19222 = fpow32(B_19218, 2.0F);
        float x_19223 = x_18626 * y_19222;
        float A2_19224 = x_19223 / y_18629;
        float exp_arg_19225 = A1_19221 - A2_19224;
        float res_19226 = fpow32(2.7182817F, exp_arg_19225);
        float negate_arg_19227 = r0_18607 * B_19218;
        float exp_arg_19228 = 0.0F - negate_arg_19227;
        float res_19229 = fpow32(2.7182817F, exp_arg_19228);
        float res_19230 = res_19226 * res_19229;
        bool empty_slice_19231 = range_end_19201 == 0;
        bool zzero_leq_i_p_m_t_s_19232 = sle64(0, range_end_19201);
        bool i_p_m_t_s_leq_w_19233 = slt64(range_end_19201, res_19199);
        bool i_lte_j_19234 = sle64(1, res_19199);
        bool y_19235 = zzero_leq_i_p_m_t_s_19232 && i_p_m_t_s_leq_w_19233;
        bool y_19236 = i_lte_j_19234 && y_19235;
        bool ok_or_empty_19237 = empty_slice_19231 || y_19236;
        bool index_certs_19238;
        
        if (!ok_or_empty_19237) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 2) == -1) {
                    global_failure_args[0] = 1;
                    global_failure_args[1] = res_19199;
                    ;
                }
                return;
            }
        }
        
        float res_19240;
        float redout_20082 = 0.0F;
        
        for (int64_t i_20083 = 0; i_20083 < range_end_19201; i_20083++) {
            int64_t slice_20092 = 1 + i_20083;
            float x_19244 = ((__global float *) mem_20110)[phys_tid_19109 +
                                                           slice_20092 *
                                                           num_threads_20240];
            float negate_arg_19245 = a_18604 * x_19244;
            float exp_arg_19246 = 0.0F - negate_arg_19245;
            float res_19247 = fpow32(2.7182817F, exp_arg_19246);
            float x_19248 = 1.0F - res_19247;
            float B_19249 = x_19248 / a_18604;
            float x_19250 = B_19249 - x_19244;
            float x_19251 = y_18628 * x_19250;
            float A1_19252 = x_19251 / x_18624;
            float y_19253 = fpow32(B_19249, 2.0F);
            float x_19254 = x_18626 * y_19253;
            float A2_19255 = x_19254 / y_18629;
            float exp_arg_19256 = A1_19252 - A2_19255;
            float res_19257 = fpow32(2.7182817F, exp_arg_19256);
            float negate_arg_19258 = r0_18607 * B_19249;
            float exp_arg_19259 = 0.0F - negate_arg_19258;
            float res_19260 = fpow32(2.7182817F, exp_arg_19259);
            float res_19261 = res_19257 * res_19260;
            float res_19243 = res_19261 + redout_20082;
            float redout_tmp_20387 = res_19243;
            
            redout_20082 = redout_tmp_20387;
        }
        res_19240 = redout_20082;
        
        float x_19262 = 1.0F - res_19230;
        float y_19263 = res_19198 * res_19240;
        float res_19264 = x_19262 / y_19263;
        
        ((__global float *) mem_20124)[gtid_19108] = res_19264;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19187
}
__kernel void mainzisegmap_19285(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args, int64_t n_18596,
                                 float a_18604, float r0_18607, float x_18624,
                                 float x_18626, float y_18628, float y_18629,
                                 __global unsigned char *swap_term_mem_20096,
                                 __global unsigned char *payments_mem_20097,
                                 __global unsigned char *mem_20133, __global
                                 unsigned char *mem_20147,
                                 int64_t num_threads_20272)
{
    #define segmap_group_sizze_19363 (mainzisegmap_group_sizze_19287)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20418;
    int32_t local_tid_20419;
    int64_t group_sizze_20422;
    int32_t wave_sizze_20421;
    int32_t group_tid_20420;
    
    global_tid_20418 = get_global_id(0);
    local_tid_20419 = get_local_id(0);
    group_sizze_20422 = get_local_size(0);
    wave_sizze_20421 = LOCKSTEP_WIDTH;
    group_tid_20420 = get_group_id(0);
    
    int32_t phys_tid_19285;
    
    phys_tid_19285 = global_tid_20418;
    
    int64_t gtid_19284;
    
    gtid_19284 = sext_i32_i64(group_tid_20420) * segmap_group_sizze_19363 +
        sext_i32_i64(local_tid_20419);
    if (slt64(gtid_19284, n_18596)) {
        float res_19374 = ((__global float *) swap_term_mem_20096)[gtid_19284];
        int64_t res_19375 = ((__global
                              int64_t *) payments_mem_20097)[gtid_19284];
        int64_t range_end_19377 = sub64(res_19375, 1);
        bool bounds_invalid_upwards_19378 = slt64(range_end_19377, 0);
        bool valid_19379 = !bounds_invalid_upwards_19378;
        bool range_valid_c_19380;
        
        if (!valid_19379) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 3) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = 1;
                    global_failure_args[2] = range_end_19377;
                    ;
                }
                return;
            }
        }
        for (int64_t i_20086 = 0; i_20086 < res_19375; i_20086++) {
            float res_19384 = sitofp_i64_f32(i_20086);
            float res_19385 = res_19374 * res_19384;
            
            ((__global float *) mem_20133)[phys_tid_19285 + i_20086 *
                                           num_threads_20272] = res_19385;
        }
        
        bool y_19386 = slt64(0, res_19375);
        bool index_certs_19387;
        
        if (!y_19386) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 4) == -1) {
                    global_failure_args[0] = 0;
                    global_failure_args[1] = res_19375;
                    ;
                }
                return;
            }
        }
        
        float binop_y_19388 = sitofp_i64_f32(range_end_19377);
        float index_primexp_19389 = res_19374 * binop_y_19388;
        float negate_arg_19390 = a_18604 * index_primexp_19389;
        float exp_arg_19391 = 0.0F - negate_arg_19390;
        float res_19392 = fpow32(2.7182817F, exp_arg_19391);
        float x_19393 = 1.0F - res_19392;
        float B_19394 = x_19393 / a_18604;
        float x_19395 = B_19394 - index_primexp_19389;
        float x_19396 = y_18628 * x_19395;
        float A1_19397 = x_19396 / x_18624;
        float y_19398 = fpow32(B_19394, 2.0F);
        float x_19399 = x_18626 * y_19398;
        float A2_19400 = x_19399 / y_18629;
        float exp_arg_19401 = A1_19397 - A2_19400;
        float res_19402 = fpow32(2.7182817F, exp_arg_19401);
        float negate_arg_19403 = r0_18607 * B_19394;
        float exp_arg_19404 = 0.0F - negate_arg_19403;
        float res_19405 = fpow32(2.7182817F, exp_arg_19404);
        float res_19406 = res_19402 * res_19405;
        bool empty_slice_19407 = range_end_19377 == 0;
        bool zzero_leq_i_p_m_t_s_19408 = sle64(0, range_end_19377);
        bool i_p_m_t_s_leq_w_19409 = slt64(range_end_19377, res_19375);
        bool i_lte_j_19410 = sle64(1, res_19375);
        bool y_19411 = zzero_leq_i_p_m_t_s_19408 && i_p_m_t_s_leq_w_19409;
        bool y_19412 = i_lte_j_19410 && y_19411;
        bool ok_or_empty_19413 = empty_slice_19407 || y_19412;
        bool index_certs_19414;
        
        if (!ok_or_empty_19413) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 5) == -1) {
                    global_failure_args[0] = 1;
                    global_failure_args[1] = res_19375;
                    ;
                }
                return;
            }
        }
        
        float res_19416;
        float redout_20088 = 0.0F;
        
        for (int64_t i_20089 = 0; i_20089 < range_end_19377; i_20089++) {
            int64_t slice_20095 = 1 + i_20089;
            float x_19420 = ((__global float *) mem_20133)[phys_tid_19285 +
                                                           slice_20095 *
                                                           num_threads_20272];
            float negate_arg_19421 = a_18604 * x_19420;
            float exp_arg_19422 = 0.0F - negate_arg_19421;
            float res_19423 = fpow32(2.7182817F, exp_arg_19422);
            float x_19424 = 1.0F - res_19423;
            float B_19425 = x_19424 / a_18604;
            float x_19426 = B_19425 - x_19420;
            float x_19427 = y_18628 * x_19426;
            float A1_19428 = x_19427 / x_18624;
            float y_19429 = fpow32(B_19425, 2.0F);
            float x_19430 = x_18626 * y_19429;
            float A2_19431 = x_19430 / y_18629;
            float exp_arg_19432 = A1_19428 - A2_19431;
            float res_19433 = fpow32(2.7182817F, exp_arg_19432);
            float negate_arg_19434 = r0_18607 * B_19425;
            float exp_arg_19435 = 0.0F - negate_arg_19434;
            float res_19436 = fpow32(2.7182817F, exp_arg_19435);
            float res_19437 = res_19433 * res_19436;
            float res_19419 = res_19437 + redout_20088;
            float redout_tmp_20424 = res_19419;
            
            redout_20088 = redout_tmp_20424;
        }
        res_19416 = redout_20088;
        
        float x_19438 = 1.0F - res_19406;
        float y_19439 = res_19374 * res_19416;
        float res_19440 = x_19438 / y_19439;
        
        ((__global float *) mem_20147)[gtid_19284] = res_19440;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19363
}
__kernel void mainzisegmap_19618(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args,
                                 int64_t paths_18599, int64_t steps_18600,
                                 float a_18604, float b_18605,
                                 float sigma_18606, float r0_18607,
                                 float dt_18622, int64_t upper_bound_18724,
                                 float res_18725, int64_t num_groups_19870,
                                 __global unsigned char *mem_20161, __global
                                 unsigned char *mem_20164, __global
                                 unsigned char *mem_20179)
{
    #define segmap_group_sizze_19869 (mainzisegmap_group_sizze_19620)
    
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
    
    int32_t global_tid_20437;
    int32_t local_tid_20438;
    int64_t group_sizze_20441;
    int32_t wave_sizze_20440;
    int32_t group_tid_20439;
    
    global_tid_20437 = get_global_id(0);
    local_tid_20438 = get_local_id(0);
    group_sizze_20441 = get_local_size(0);
    wave_sizze_20440 = LOCKSTEP_WIDTH;
    group_tid_20439 = get_group_id(0);
    
    int32_t phys_tid_19618;
    
    phys_tid_19618 = global_tid_20437;
    
    int32_t phys_group_id_20442;
    
    phys_group_id_20442 = get_group_id(0);
    for (int32_t i_20443 = 0; i_20443 <
         sdiv_up32(sext_i64_i32(sdiv_up64(paths_18599,
                                          segmap_group_sizze_19869)) -
                   phys_group_id_20442, sext_i64_i32(num_groups_19870));
         i_20443++) {
        int32_t virt_group_id_20444 = phys_group_id_20442 + i_20443 *
                sext_i64_i32(num_groups_19870);
        int64_t gtid_19617 = sext_i32_i64(virt_group_id_20444) *
                segmap_group_sizze_19869 + sext_i32_i64(local_tid_20438);
        
        if (slt64(gtid_19617, paths_18599)) {
            for (int64_t i_20445 = 0; i_20445 < steps_18600; i_20445++) {
                ((__global float *) mem_20164)[phys_tid_19618 + i_20445 *
                                               (num_groups_19870 *
                                                segmap_group_sizze_19869)] =
                    r0_18607;
            }
            for (int64_t i_19876 = 0; i_19876 < upper_bound_18724; i_19876++) {
                bool y_19878 = slt64(i_19876, steps_18600);
                bool index_certs_19879;
                
                if (!y_19878) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 6) ==
                            -1) {
                            global_failure_args[0] = i_19876;
                            global_failure_args[1] = steps_18600;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float shortstep_arg_19880 = ((__global
                                              float *) mem_20161)[i_19876 *
                                                                  paths_18599 +
                                                                  gtid_19617];
                float shortstep_arg_19881 = ((__global
                                              float *) mem_20164)[phys_tid_19618 +
                                                                  i_19876 *
                                                                  (num_groups_19870 *
                                                                   segmap_group_sizze_19869)];
                float y_19882 = b_18605 - shortstep_arg_19881;
                float x_19883 = a_18604 * y_19882;
                float x_19884 = dt_18622 * x_19883;
                float x_19885 = res_18725 * shortstep_arg_19880;
                float y_19886 = sigma_18606 * x_19885;
                float delta_r_19887 = x_19884 + y_19886;
                float res_19888 = shortstep_arg_19881 + delta_r_19887;
                int64_t i_19889 = add64(1, i_19876);
                bool x_19890 = sle64(0, i_19889);
                bool y_19891 = slt64(i_19889, steps_18600);
                bool bounds_check_19892 = x_19890 && y_19891;
                bool index_certs_19893;
                
                if (!bounds_check_19892) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 7) ==
                            -1) {
                            global_failure_args[0] = i_19889;
                            global_failure_args[1] = steps_18600;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                ((__global float *) mem_20164)[phys_tid_19618 + i_19889 *
                                               (num_groups_19870 *
                                                segmap_group_sizze_19869)] =
                    res_19888;
            }
            for (int64_t i_20447 = 0; i_20447 < steps_18600; i_20447++) {
                ((__global float *) mem_20179)[i_20447 * paths_18599 +
                                               gtid_19617] = ((__global
                                                               float *) mem_20164)[phys_tid_19618 +
                                                                                   i_20447 *
                                                                                   (num_groups_19870 *
                                                                                    segmap_group_sizze_19869)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19869
}
__kernel void mainzisegmap_19716(__global int *global_failure,
                                 int64_t paths_18599, int64_t steps_18600,
                                 __global unsigned char *mem_20154, __global
                                 unsigned char *mem_20158)
{
    #define segmap_group_sizze_19824 (mainzisegmap_group_sizze_19719)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20431;
    int32_t local_tid_20432;
    int64_t group_sizze_20435;
    int32_t wave_sizze_20434;
    int32_t group_tid_20433;
    
    global_tid_20431 = get_global_id(0);
    local_tid_20432 = get_local_id(0);
    group_sizze_20435 = get_local_size(0);
    wave_sizze_20434 = LOCKSTEP_WIDTH;
    group_tid_20433 = get_group_id(0);
    
    int32_t phys_tid_19716;
    
    phys_tid_19716 = global_tid_20431;
    
    int64_t gtid_19714;
    
    gtid_19714 = squot64(sext_i32_i64(group_tid_20433) *
                         segmap_group_sizze_19824 +
                         sext_i32_i64(local_tid_20432), steps_18600);
    
    int64_t gtid_19715;
    
    gtid_19715 = sext_i32_i64(group_tid_20433) * segmap_group_sizze_19824 +
        sext_i32_i64(local_tid_20432) - squot64(sext_i32_i64(group_tid_20433) *
                                                segmap_group_sizze_19824 +
                                                sext_i32_i64(local_tid_20432),
                                                steps_18600) * steps_18600;
    if (slt64(gtid_19714, paths_18599) && slt64(gtid_19715, steps_18600)) {
        int32_t unsign_arg_19827 = ((__global int32_t *) mem_20154)[gtid_19714];
        int32_t res_19829 = sext_i64_i32(gtid_19715);
        int32_t x_19830 = lshr32(res_19829, 16);
        int32_t x_19831 = res_19829 ^ x_19830;
        int32_t x_19832 = mul32(73244475, x_19831);
        int32_t x_19833 = lshr32(x_19832, 16);
        int32_t x_19834 = x_19832 ^ x_19833;
        int32_t x_19835 = mul32(73244475, x_19834);
        int32_t x_19836 = lshr32(x_19835, 16);
        int32_t x_19837 = x_19835 ^ x_19836;
        int32_t unsign_arg_19838 = unsign_arg_19827 ^ x_19837;
        int32_t unsign_arg_19839 = mul32(48271, unsign_arg_19838);
        int32_t unsign_arg_19840 = umod32(unsign_arg_19839, 2147483647);
        int32_t unsign_arg_19841 = mul32(48271, unsign_arg_19840);
        int32_t unsign_arg_19842 = umod32(unsign_arg_19841, 2147483647);
        float res_19843 = uitofp_i32_f32(unsign_arg_19840);
        float res_19844 = res_19843 / 2.1474836e9F;
        float res_19845 = uitofp_i32_f32(unsign_arg_19842);
        float res_19846 = res_19845 / 2.1474836e9F;
        float res_19847;
        
        res_19847 = futrts_log32(res_19844);
        
        float res_19848 = -2.0F * res_19847;
        float res_19849;
        
        res_19849 = futrts_sqrt32(res_19848);
        
        float res_19850 = 6.2831855F * res_19846;
        float res_19851;
        
        res_19851 = futrts_cos32(res_19850);
        
        float res_19852 = res_19849 * res_19851;
        
        ((__global float *) mem_20158)[gtid_19714 * steps_18600 + gtid_19715] =
            res_19852;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19824
}
__kernel void mainzisegmap_19780(__global int *global_failure,
                                 int64_t paths_18599, __global
                                 unsigned char *mem_20154)
{
    #define segmap_group_sizze_19799 (mainzisegmap_group_sizze_19782)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20426;
    int32_t local_tid_20427;
    int64_t group_sizze_20430;
    int32_t wave_sizze_20429;
    int32_t group_tid_20428;
    
    global_tid_20426 = get_global_id(0);
    local_tid_20427 = get_local_id(0);
    group_sizze_20430 = get_local_size(0);
    wave_sizze_20429 = LOCKSTEP_WIDTH;
    group_tid_20428 = get_group_id(0);
    
    int32_t phys_tid_19780;
    
    phys_tid_19780 = global_tid_20426;
    
    int64_t gtid_19779;
    
    gtid_19779 = sext_i32_i64(group_tid_20428) * segmap_group_sizze_19799 +
        sext_i32_i64(local_tid_20427);
    if (slt64(gtid_19779, paths_18599)) {
        int32_t res_19803 = sext_i64_i32(gtid_19779);
        int32_t x_19804 = lshr32(res_19803, 16);
        int32_t x_19805 = res_19803 ^ x_19804;
        int32_t x_19806 = mul32(73244475, x_19805);
        int32_t x_19807 = lshr32(x_19806, 16);
        int32_t x_19808 = x_19806 ^ x_19807;
        int32_t x_19809 = mul32(73244475, x_19808);
        int32_t x_19810 = lshr32(x_19809, 16);
        int32_t x_19811 = x_19809 ^ x_19810;
        int32_t unsign_arg_19812 = 777822902 ^ x_19811;
        int32_t unsign_arg_19813 = mul32(48271, unsign_arg_19812);
        int32_t unsign_arg_19814 = umod32(unsign_arg_19813, 2147483647);
        
        ((__global int32_t *) mem_20154)[gtid_19779] = unsign_arg_19814;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19799
}
__kernel void mainzisegmap_20010(__global int *global_failure,
                                 int64_t res_18845, int64_t num_segments_19039,
                                 __global unsigned char *mem_20211, __global
                                 unsigned char *mem_20216, __global
                                 unsigned char *mem_20219, __global
                                 unsigned char *mem_20221)
{
    #define segmap_group_sizze_20013 (mainzisegmap_group_sizze_20012)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20951;
    int32_t local_tid_20952;
    int64_t group_sizze_20955;
    int32_t wave_sizze_20954;
    int32_t group_tid_20953;
    
    global_tid_20951 = get_global_id(0);
    local_tid_20952 = get_local_id(0);
    group_sizze_20955 = get_local_size(0);
    wave_sizze_20954 = LOCKSTEP_WIDTH;
    group_tid_20953 = get_group_id(0);
    
    int32_t phys_tid_20010;
    
    phys_tid_20010 = global_tid_20951;
    
    int64_t write_i_20009;
    
    write_i_20009 = sext_i32_i64(group_tid_20953) * segmap_group_sizze_20013 +
        sext_i32_i64(local_tid_20952);
    if (slt64(write_i_20009, res_18845)) {
        int64_t i_p_o_20052 = add64(1, write_i_20009);
        int64_t rot_i_20053 = smod64(i_p_o_20052, res_18845);
        bool x_19052 = ((__global bool *) mem_20211)[rot_i_20053];
        float write_value_19053 = ((__global float *) mem_20216)[write_i_20009];
        int64_t res_19054;
        
        if (x_19052) {
            int64_t x_19051 = ((__global int64_t *) mem_20219)[write_i_20009];
            int64_t res_19055 = sub64(x_19051, 1);
            
            res_19054 = res_19055;
        } else {
            res_19054 = -1;
        }
        if (sle64(0, res_19054) && slt64(res_19054, num_segments_19039)) {
            ((__global float *) mem_20221)[res_19054] = write_value_19053;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20013
}
__kernel void mainzisegred_large_20629(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_20667_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_20665_backing_aligned_1,
                                       int64_t res_18845,
                                       int64_t num_groups_19922, __global
                                       unsigned char *mem_20198,
                                       int32_t num_subhistos_20543, __global
                                       unsigned char *res_subhistos_mem_20544,
                                       int64_t groups_per_segment_20651,
                                       int64_t elements_per_thread_20652,
                                       int64_t virt_num_groups_20653,
                                       int64_t threads_per_segment_20655,
                                       __global
                                       unsigned char *group_res_arr_mem_20656,
                                       __global
                                       unsigned char *mainzicounter_mem_20658)
{
    #define seghist_group_sizze_19920 (mainziseghist_group_sizze_19919)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_20667_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_20667_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_20665_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_20665_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20660;
    int32_t local_tid_20661;
    int64_t group_sizze_20664;
    int32_t wave_sizze_20663;
    int32_t group_tid_20662;
    
    global_tid_20660 = get_global_id(0);
    local_tid_20661 = get_local_id(0);
    group_sizze_20664 = get_local_size(0);
    wave_sizze_20663 = LOCKSTEP_WIDTH;
    group_tid_20662 = get_group_id(0);
    
    int32_t flat_gtid_20629;
    
    flat_gtid_20629 = global_tid_20660;
    
    __local char *red_arr_mem_20665;
    
    red_arr_mem_20665 = (__local char *) red_arr_mem_20665_backing_0;
    
    __local char *sync_arr_mem_20667;
    
    sync_arr_mem_20667 = (__local char *) sync_arr_mem_20667_backing_1;
    
    int32_t phys_group_id_20669;
    
    phys_group_id_20669 = get_group_id(0);
    for (int32_t i_20670 = 0; i_20670 <
         sdiv_up32(sext_i64_i32(virt_num_groups_20653) - phys_group_id_20669,
                   sext_i64_i32(num_groups_19922)); i_20670++) {
        int32_t virt_group_id_20671 = phys_group_id_20669 + i_20670 *
                sext_i64_i32(num_groups_19922);
        int32_t flat_segment_id_20672 = squot32(virt_group_id_20671,
                                                sext_i64_i32(groups_per_segment_20651));
        int64_t global_tid_20673 = srem64(sext_i32_i64(virt_group_id_20671) *
                                          seghist_group_sizze_19920 +
                                          sext_i32_i64(local_tid_20661),
                                          seghist_group_sizze_19920 *
                                          groups_per_segment_20651);
        int64_t bucket_id_20627 = sext_i32_i64(flat_segment_id_20672);
        int64_t subhistogram_id_20628;
        int64_t x_acc_20674;
        int64_t chunk_sizze_20675;
        
        chunk_sizze_20675 = smin64(elements_per_thread_20652,
                                   sdiv_up64(num_subhistos_20543 -
                                             sext_i32_i64(sext_i64_i32(global_tid_20673)),
                                             threads_per_segment_20655));
        
        int64_t x_19926;
        int64_t x_19927;
        
        // neutral-initialise the accumulators
        {
            x_acc_20674 = 0;
        }
        for (int64_t i_20679 = 0; i_20679 < chunk_sizze_20675; i_20679++) {
            subhistogram_id_20628 =
                sext_i32_i64(sext_i64_i32(global_tid_20673)) +
                threads_per_segment_20655 * i_20679;
            // apply map function
            {
                // load accumulator
                {
                    x_19926 = x_acc_20674;
                }
                // load new values
                {
                    x_19927 = ((__global
                                int64_t *) res_subhistos_mem_20544)[subhistogram_id_20628 *
                                                                    res_18845 +
                                                                    bucket_id_20627];
                }
                // apply reduction operator
                {
                    int64_t res_19928 = smax64(x_19926, x_19927);
                    
                    // store in accumulator
                    {
                        x_acc_20674 = res_19928;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_19926 = x_acc_20674;
            ((__local
              int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661)] =
                x_19926;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_20680;
        int32_t skip_waves_20681;
        
        skip_waves_20681 = 1;
        
        int64_t x_20676;
        int64_t x_20677;
        
        offset_20680 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_20661,
                      sext_i64_i32(seghist_group_sizze_19920))) {
                x_20676 = ((__local
                            int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661 +
                                                          offset_20680)];
            }
        }
        offset_20680 = 1;
        while (slt32(offset_20680, wave_sizze_20663)) {
            if (slt32(local_tid_20661 + offset_20680,
                      sext_i64_i32(seghist_group_sizze_19920)) &&
                ((local_tid_20661 - squot32(local_tid_20661, wave_sizze_20663) *
                  wave_sizze_20663) & (2 * offset_20680 - 1)) == 0) {
                // read array element
                {
                    x_20677 = ((volatile __local
                                int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661 +
                                                              offset_20680)];
                }
                // apply reduction operation
                {
                    int64_t res_20678 = smax64(x_20676, x_20677);
                    
                    x_20676 = res_20678;
                }
                // write result of operation
                {
                    ((volatile __local
                      int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661)] =
                        x_20676;
                }
            }
            offset_20680 *= 2;
        }
        while (slt32(skip_waves_20681,
                     squot32(sext_i64_i32(seghist_group_sizze_19920) +
                             wave_sizze_20663 - 1, wave_sizze_20663))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_20680 = skip_waves_20681 * wave_sizze_20663;
            if (slt32(local_tid_20661 + offset_20680,
                      sext_i64_i32(seghist_group_sizze_19920)) &&
                ((local_tid_20661 - squot32(local_tid_20661, wave_sizze_20663) *
                  wave_sizze_20663) == 0 && (squot32(local_tid_20661,
                                                     wave_sizze_20663) & (2 *
                                                                          skip_waves_20681 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_20677 = ((__local
                                int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661 +
                                                              offset_20680)];
                }
                // apply reduction operation
                {
                    int64_t res_20678 = smax64(x_20676, x_20677);
                    
                    x_20676 = res_20678;
                }
                // write result of operation
                {
                    ((__local
                      int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661)] =
                        x_20676;
                }
            }
            skip_waves_20681 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (sext_i32_i64(local_tid_20661) == 0) {
                x_acc_20674 = x_20676;
            }
        }
        if (groups_per_segment_20651 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_20661 == 0) {
                    ((__global int64_t *) mem_20198)[bucket_id_20627] =
                        x_acc_20674;
                }
            }
        } else {
            int32_t old_counter_20682;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_20661 == 0) {
                    ((__global
                      int64_t *) group_res_arr_mem_20656)[sext_i32_i64(virt_group_id_20671) *
                                                          seghist_group_sizze_19920] =
                        x_acc_20674;
                    mem_fence_global();
                    old_counter_20682 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_20658)[sext_i32_i64(srem32(flat_segment_id_20672,
                                                                                                     10240))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_20667)[0] =
                        old_counter_20682 == groups_per_segment_20651 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_20683;
            
            is_last_group_20683 = ((__local bool *) sync_arr_mem_20667)[0];
            if (is_last_group_20683) {
                if (local_tid_20661 == 0) {
                    old_counter_20682 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_20658)[sext_i32_i64(srem32(flat_segment_id_20672,
                                                                                                     10240))],
                                              (int) (0 -
                                                     groups_per_segment_20651));
                }
                // read in the per-group-results
                {
                    int64_t read_per_thread_20684 =
                            sdiv_up64(groups_per_segment_20651,
                                      seghist_group_sizze_19920);
                    
                    x_19926 = 0;
                    for (int64_t i_20685 = 0; i_20685 < read_per_thread_20684;
                         i_20685++) {
                        int64_t group_res_id_20686 =
                                sext_i32_i64(local_tid_20661) *
                                read_per_thread_20684 + i_20685;
                        int64_t index_of_group_res_20687 =
                                sext_i32_i64(flat_segment_id_20672) *
                                groups_per_segment_20651 + group_res_id_20686;
                        
                        if (slt64(group_res_id_20686,
                                  groups_per_segment_20651)) {
                            x_19927 = ((__global
                                        int64_t *) group_res_arr_mem_20656)[index_of_group_res_20687 *
                                                                            seghist_group_sizze_19920];
                            
                            int64_t res_19928;
                            
                            res_19928 = smax64(x_19926, x_19927);
                            x_19926 = res_19928;
                        }
                    }
                }
                ((__local
                  int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661)] =
                    x_19926;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_20688;
                    int32_t skip_waves_20689;
                    
                    skip_waves_20689 = 1;
                    
                    int64_t x_20676;
                    int64_t x_20677;
                    
                    offset_20688 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_20661,
                                  sext_i64_i32(seghist_group_sizze_19920))) {
                            x_20676 = ((__local
                                        int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661 +
                                                                      offset_20688)];
                        }
                    }
                    offset_20688 = 1;
                    while (slt32(offset_20688, wave_sizze_20663)) {
                        if (slt32(local_tid_20661 + offset_20688,
                                  sext_i64_i32(seghist_group_sizze_19920)) &&
                            ((local_tid_20661 - squot32(local_tid_20661,
                                                        wave_sizze_20663) *
                              wave_sizze_20663) & (2 * offset_20688 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_20677 = ((volatile __local
                                            int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661 +
                                                                          offset_20688)];
                            }
                            // apply reduction operation
                            {
                                int64_t res_20678 = smax64(x_20676, x_20677);
                                
                                x_20676 = res_20678;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661)] =
                                    x_20676;
                            }
                        }
                        offset_20688 *= 2;
                    }
                    while (slt32(skip_waves_20689,
                                 squot32(sext_i64_i32(seghist_group_sizze_19920) +
                                         wave_sizze_20663 - 1,
                                         wave_sizze_20663))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_20688 = skip_waves_20689 * wave_sizze_20663;
                        if (slt32(local_tid_20661 + offset_20688,
                                  sext_i64_i32(seghist_group_sizze_19920)) &&
                            ((local_tid_20661 - squot32(local_tid_20661,
                                                        wave_sizze_20663) *
                              wave_sizze_20663) == 0 &&
                             (squot32(local_tid_20661, wave_sizze_20663) & (2 *
                                                                            skip_waves_20689 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_20677 = ((__local
                                            int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661 +
                                                                          offset_20688)];
                            }
                            // apply reduction operation
                            {
                                int64_t res_20678 = smax64(x_20676, x_20677);
                                
                                x_20676 = res_20678;
                            }
                            // write result of operation
                            {
                                ((__local
                                  int64_t *) red_arr_mem_20665)[sext_i32_i64(local_tid_20661)] =
                                    x_20676;
                            }
                        }
                        skip_waves_20689 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_20661 == 0) {
                            ((__global int64_t *) mem_20198)[bucket_id_20627] =
                                x_20676;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef seghist_group_sizze_19920
}
__kernel void mainzisegred_nonseg_19095(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_20333_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_20331_backing_aligned_1,
                                        int64_t n_18596,
                                        int64_t num_groups_19090, __global
                                        unsigned char *swap_term_mem_20096,
                                        __global
                                        unsigned char *payments_mem_20097,
                                        __global unsigned char *mem_20101,
                                        __global
                                        unsigned char *mainzicounter_mem_20321,
                                        __global
                                        unsigned char *group_res_arr_mem_20323,
                                        int64_t num_threads_20325)
{
    #define segred_group_sizze_19088 (mainzisegred_group_sizze_19087)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_20333_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_20333_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_20331_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_20331_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20326;
    int32_t local_tid_20327;
    int64_t group_sizze_20330;
    int32_t wave_sizze_20329;
    int32_t group_tid_20328;
    
    global_tid_20326 = get_global_id(0);
    local_tid_20327 = get_local_id(0);
    group_sizze_20330 = get_local_size(0);
    wave_sizze_20329 = LOCKSTEP_WIDTH;
    group_tid_20328 = get_group_id(0);
    
    int32_t phys_tid_19095;
    
    phys_tid_19095 = global_tid_20326;
    
    __local char *sync_arr_mem_20331;
    
    sync_arr_mem_20331 = (__local char *) sync_arr_mem_20331_backing_0;
    
    __local char *red_arr_mem_20333;
    
    red_arr_mem_20333 = (__local char *) red_arr_mem_20333_backing_1;
    
    int64_t dummy_19093;
    
    dummy_19093 = 0;
    
    int64_t gtid_19094;
    
    gtid_19094 = 0;
    
    float x_acc_20335;
    int64_t chunk_sizze_20336;
    
    chunk_sizze_20336 = smin64(sdiv_up64(n_18596,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_19088 *
                                         num_groups_19090))),
                               sdiv_up64(n_18596 - sext_i32_i64(phys_tid_19095),
                                         num_threads_20325));
    
    float x_18614;
    float x_18615;
    
    // neutral-initialise the accumulators
    {
        x_acc_20335 = -INFINITY;
    }
    for (int64_t i_20340 = 0; i_20340 < chunk_sizze_20336; i_20340++) {
        gtid_19094 = sext_i32_i64(phys_tid_19095) + num_threads_20325 * i_20340;
        // apply map function
        {
            float x_18617 = ((__global
                              float *) swap_term_mem_20096)[gtid_19094];
            int64_t x_18618 = ((__global
                                int64_t *) payments_mem_20097)[gtid_19094];
            float res_18619 = sitofp_i64_f32(x_18618);
            float res_18620 = x_18617 * res_18619;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_18614 = x_acc_20335;
            }
            // load new values
            {
                x_18615 = res_18620;
            }
            // apply reduction operator
            {
                float res_18616 = fmax32(x_18614, x_18615);
                
                // store in accumulator
                {
                    x_acc_20335 = res_18616;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_18614 = x_acc_20335;
        ((__local float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327)] =
            x_18614;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_20341;
    int32_t skip_waves_20342;
    
    skip_waves_20342 = 1;
    
    float x_20337;
    float x_20338;
    
    offset_20341 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_20327, sext_i64_i32(segred_group_sizze_19088))) {
            x_20337 = ((__local
                        float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327 +
                                                    offset_20341)];
        }
    }
    offset_20341 = 1;
    while (slt32(offset_20341, wave_sizze_20329)) {
        if (slt32(local_tid_20327 + offset_20341,
                  sext_i64_i32(segred_group_sizze_19088)) && ((local_tid_20327 -
                                                               squot32(local_tid_20327,
                                                                       wave_sizze_20329) *
                                                               wave_sizze_20329) &
                                                              (2 *
                                                               offset_20341 -
                                                               1)) == 0) {
            // read array element
            {
                x_20338 = ((volatile __local
                            float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327 +
                                                        offset_20341)];
            }
            // apply reduction operation
            {
                float res_20339 = fmax32(x_20337, x_20338);
                
                x_20337 = res_20339;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327)] =
                    x_20337;
            }
        }
        offset_20341 *= 2;
    }
    while (slt32(skip_waves_20342,
                 squot32(sext_i64_i32(segred_group_sizze_19088) +
                         wave_sizze_20329 - 1, wave_sizze_20329))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_20341 = skip_waves_20342 * wave_sizze_20329;
        if (slt32(local_tid_20327 + offset_20341,
                  sext_i64_i32(segred_group_sizze_19088)) && ((local_tid_20327 -
                                                               squot32(local_tid_20327,
                                                                       wave_sizze_20329) *
                                                               wave_sizze_20329) ==
                                                              0 &&
                                                              (squot32(local_tid_20327,
                                                                       wave_sizze_20329) &
                                                               (2 *
                                                                skip_waves_20342 -
                                                                1)) == 0)) {
            // read array element
            {
                x_20338 = ((__local
                            float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327 +
                                                        offset_20341)];
            }
            // apply reduction operation
            {
                float res_20339 = fmax32(x_20337, x_20338);
                
                x_20337 = res_20339;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327)] =
                    x_20337;
            }
        }
        skip_waves_20342 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_20327) == 0) {
            x_acc_20335 = x_20337;
        }
    }
    
    int32_t old_counter_20343;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_20327 == 0) {
            ((__global
              float *) group_res_arr_mem_20323)[sext_i32_i64(group_tid_20328) *
                                                segred_group_sizze_19088] =
                x_acc_20335;
            mem_fence_global();
            old_counter_20343 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20321)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_20331)[0] = old_counter_20343 ==
                num_groups_19090 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_20344;
    
    is_last_group_20344 = ((__local bool *) sync_arr_mem_20331)[0];
    if (is_last_group_20344) {
        if (local_tid_20327 == 0) {
            old_counter_20343 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20321)[0],
                                                      (int) (0 -
                                                             num_groups_19090));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_20345 = sdiv_up64(num_groups_19090,
                                                      segred_group_sizze_19088);
            
            x_18614 = -INFINITY;
            for (int64_t i_20346 = 0; i_20346 < read_per_thread_20345;
                 i_20346++) {
                int64_t group_res_id_20347 = sext_i32_i64(local_tid_20327) *
                        read_per_thread_20345 + i_20346;
                int64_t index_of_group_res_20348 = group_res_id_20347;
                
                if (slt64(group_res_id_20347, num_groups_19090)) {
                    x_18615 = ((__global
                                float *) group_res_arr_mem_20323)[index_of_group_res_20348 *
                                                                  segred_group_sizze_19088];
                    
                    float res_18616;
                    
                    res_18616 = fmax32(x_18614, x_18615);
                    x_18614 = res_18616;
                }
            }
        }
        ((__local float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327)] =
            x_18614;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_20349;
            int32_t skip_waves_20350;
            
            skip_waves_20350 = 1;
            
            float x_20337;
            float x_20338;
            
            offset_20349 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_20327,
                          sext_i64_i32(segred_group_sizze_19088))) {
                    x_20337 = ((__local
                                float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327 +
                                                            offset_20349)];
                }
            }
            offset_20349 = 1;
            while (slt32(offset_20349, wave_sizze_20329)) {
                if (slt32(local_tid_20327 + offset_20349,
                          sext_i64_i32(segred_group_sizze_19088)) &&
                    ((local_tid_20327 - squot32(local_tid_20327,
                                                wave_sizze_20329) *
                      wave_sizze_20329) & (2 * offset_20349 - 1)) == 0) {
                    // read array element
                    {
                        x_20338 = ((volatile __local
                                    float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327 +
                                                                offset_20349)];
                    }
                    // apply reduction operation
                    {
                        float res_20339 = fmax32(x_20337, x_20338);
                        
                        x_20337 = res_20339;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327)] =
                            x_20337;
                    }
                }
                offset_20349 *= 2;
            }
            while (slt32(skip_waves_20350,
                         squot32(sext_i64_i32(segred_group_sizze_19088) +
                                 wave_sizze_20329 - 1, wave_sizze_20329))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_20349 = skip_waves_20350 * wave_sizze_20329;
                if (slt32(local_tid_20327 + offset_20349,
                          sext_i64_i32(segred_group_sizze_19088)) &&
                    ((local_tid_20327 - squot32(local_tid_20327,
                                                wave_sizze_20329) *
                      wave_sizze_20329) == 0 && (squot32(local_tid_20327,
                                                         wave_sizze_20329) &
                                                 (2 * skip_waves_20350 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_20338 = ((__local
                                    float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327 +
                                                                offset_20349)];
                    }
                    // apply reduction operation
                    {
                        float res_20339 = fmax32(x_20337, x_20338);
                        
                        x_20337 = res_20339;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_20333)[sext_i32_i64(local_tid_20327)] =
                            x_20337;
                    }
                }
                skip_waves_20350 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_20327 == 0) {
                    ((__global float *) mem_20101)[0] = x_20337;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_19088
}
__kernel void mainzisegred_nonseg_19917(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_20516_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_20514_backing_aligned_1,
                                        int64_t paths_18599,
                                        int64_t num_groups_19912, __global
                                        unsigned char *mem_20193, __global
                                        unsigned char *mem_20196, __global
                                        unsigned char *mainzicounter_mem_20504,
                                        __global
                                        unsigned char *group_res_arr_mem_20506,
                                        int64_t num_threads_20508)
{
    #define segred_group_sizze_19910 (mainzisegred_group_sizze_19909)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_20516_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_20516_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_20514_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_20514_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20509;
    int32_t local_tid_20510;
    int64_t group_sizze_20513;
    int32_t wave_sizze_20512;
    int32_t group_tid_20511;
    
    global_tid_20509 = get_global_id(0);
    local_tid_20510 = get_local_id(0);
    group_sizze_20513 = get_local_size(0);
    wave_sizze_20512 = LOCKSTEP_WIDTH;
    group_tid_20511 = get_group_id(0);
    
    int32_t phys_tid_19917;
    
    phys_tid_19917 = global_tid_20509;
    
    __local char *sync_arr_mem_20514;
    
    sync_arr_mem_20514 = (__local char *) sync_arr_mem_20514_backing_0;
    
    __local char *red_arr_mem_20516;
    
    red_arr_mem_20516 = (__local char *) red_arr_mem_20516_backing_1;
    
    int64_t dummy_19915;
    
    dummy_19915 = 0;
    
    int64_t gtid_19916;
    
    gtid_19916 = 0;
    
    int64_t x_acc_20518;
    int64_t chunk_sizze_20519;
    
    chunk_sizze_20519 = smin64(sdiv_up64(paths_18599,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_19910 *
                                         num_groups_19912))),
                               sdiv_up64(paths_18599 -
                                         sext_i32_i64(phys_tid_19917),
                                         num_threads_20508));
    
    int64_t x_18846;
    int64_t x_18847;
    
    // neutral-initialise the accumulators
    {
        x_acc_20518 = 0;
    }
    for (int64_t i_20523 = 0; i_20523 < chunk_sizze_20519; i_20523++) {
        gtid_19916 = sext_i32_i64(phys_tid_19917) + num_threads_20508 * i_20523;
        // apply map function
        {
            int64_t x_18849 = ((__global int64_t *) mem_20193)[gtid_19916];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_18846 = x_acc_20518;
            }
            // load new values
            {
                x_18847 = x_18849;
            }
            // apply reduction operator
            {
                int64_t res_18848 = add64(x_18846, x_18847);
                
                // store in accumulator
                {
                    x_acc_20518 = res_18848;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_18846 = x_acc_20518;
        ((__local int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510)] =
            x_18846;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_20524;
    int32_t skip_waves_20525;
    
    skip_waves_20525 = 1;
    
    int64_t x_20520;
    int64_t x_20521;
    
    offset_20524 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_20510, sext_i64_i32(segred_group_sizze_19910))) {
            x_20520 = ((__local
                        int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510 +
                                                      offset_20524)];
        }
    }
    offset_20524 = 1;
    while (slt32(offset_20524, wave_sizze_20512)) {
        if (slt32(local_tid_20510 + offset_20524,
                  sext_i64_i32(segred_group_sizze_19910)) && ((local_tid_20510 -
                                                               squot32(local_tid_20510,
                                                                       wave_sizze_20512) *
                                                               wave_sizze_20512) &
                                                              (2 *
                                                               offset_20524 -
                                                               1)) == 0) {
            // read array element
            {
                x_20521 = ((volatile __local
                            int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510 +
                                                          offset_20524)];
            }
            // apply reduction operation
            {
                int64_t res_20522 = add64(x_20520, x_20521);
                
                x_20520 = res_20522;
            }
            // write result of operation
            {
                ((volatile __local
                  int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510)] =
                    x_20520;
            }
        }
        offset_20524 *= 2;
    }
    while (slt32(skip_waves_20525,
                 squot32(sext_i64_i32(segred_group_sizze_19910) +
                         wave_sizze_20512 - 1, wave_sizze_20512))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_20524 = skip_waves_20525 * wave_sizze_20512;
        if (slt32(local_tid_20510 + offset_20524,
                  sext_i64_i32(segred_group_sizze_19910)) && ((local_tid_20510 -
                                                               squot32(local_tid_20510,
                                                                       wave_sizze_20512) *
                                                               wave_sizze_20512) ==
                                                              0 &&
                                                              (squot32(local_tid_20510,
                                                                       wave_sizze_20512) &
                                                               (2 *
                                                                skip_waves_20525 -
                                                                1)) == 0)) {
            // read array element
            {
                x_20521 = ((__local
                            int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510 +
                                                          offset_20524)];
            }
            // apply reduction operation
            {
                int64_t res_20522 = add64(x_20520, x_20521);
                
                x_20520 = res_20522;
            }
            // write result of operation
            {
                ((__local
                  int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510)] =
                    x_20520;
            }
        }
        skip_waves_20525 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_20510) == 0) {
            x_acc_20518 = x_20520;
        }
    }
    
    int32_t old_counter_20526;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_20510 == 0) {
            ((__global
              int64_t *) group_res_arr_mem_20506)[sext_i32_i64(group_tid_20511) *
                                                  segred_group_sizze_19910] =
                x_acc_20518;
            mem_fence_global();
            old_counter_20526 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20504)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_20514)[0] = old_counter_20526 ==
                num_groups_19912 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_20527;
    
    is_last_group_20527 = ((__local bool *) sync_arr_mem_20514)[0];
    if (is_last_group_20527) {
        if (local_tid_20510 == 0) {
            old_counter_20526 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20504)[0],
                                                      (int) (0 -
                                                             num_groups_19912));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_20528 = sdiv_up64(num_groups_19912,
                                                      segred_group_sizze_19910);
            
            x_18846 = 0;
            for (int64_t i_20529 = 0; i_20529 < read_per_thread_20528;
                 i_20529++) {
                int64_t group_res_id_20530 = sext_i32_i64(local_tid_20510) *
                        read_per_thread_20528 + i_20529;
                int64_t index_of_group_res_20531 = group_res_id_20530;
                
                if (slt64(group_res_id_20530, num_groups_19912)) {
                    x_18847 = ((__global
                                int64_t *) group_res_arr_mem_20506)[index_of_group_res_20531 *
                                                                    segred_group_sizze_19910];
                    
                    int64_t res_18848;
                    
                    res_18848 = add64(x_18846, x_18847);
                    x_18846 = res_18848;
                }
            }
        }
        ((__local int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510)] =
            x_18846;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_20532;
            int32_t skip_waves_20533;
            
            skip_waves_20533 = 1;
            
            int64_t x_20520;
            int64_t x_20521;
            
            offset_20532 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_20510,
                          sext_i64_i32(segred_group_sizze_19910))) {
                    x_20520 = ((__local
                                int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510 +
                                                              offset_20532)];
                }
            }
            offset_20532 = 1;
            while (slt32(offset_20532, wave_sizze_20512)) {
                if (slt32(local_tid_20510 + offset_20532,
                          sext_i64_i32(segred_group_sizze_19910)) &&
                    ((local_tid_20510 - squot32(local_tid_20510,
                                                wave_sizze_20512) *
                      wave_sizze_20512) & (2 * offset_20532 - 1)) == 0) {
                    // read array element
                    {
                        x_20521 = ((volatile __local
                                    int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510 +
                                                                  offset_20532)];
                    }
                    // apply reduction operation
                    {
                        int64_t res_20522 = add64(x_20520, x_20521);
                        
                        x_20520 = res_20522;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510)] =
                            x_20520;
                    }
                }
                offset_20532 *= 2;
            }
            while (slt32(skip_waves_20533,
                         squot32(sext_i64_i32(segred_group_sizze_19910) +
                                 wave_sizze_20512 - 1, wave_sizze_20512))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_20532 = skip_waves_20533 * wave_sizze_20512;
                if (slt32(local_tid_20510 + offset_20532,
                          sext_i64_i32(segred_group_sizze_19910)) &&
                    ((local_tid_20510 - squot32(local_tid_20510,
                                                wave_sizze_20512) *
                      wave_sizze_20512) == 0 && (squot32(local_tid_20510,
                                                         wave_sizze_20512) &
                                                 (2 * skip_waves_20533 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_20521 = ((__local
                                    int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510 +
                                                                  offset_20532)];
                    }
                    // apply reduction operation
                    {
                        int64_t res_20522 = add64(x_20520, x_20521);
                        
                        x_20520 = res_20522;
                    }
                    // write result of operation
                    {
                        ((__local
                          int64_t *) red_arr_mem_20516)[sext_i32_i64(local_tid_20510)] =
                            x_20520;
                    }
                }
                skip_waves_20533 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_20510 == 0) {
                    ((__global int64_t *) mem_20196)[0] = x_20520;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_19910
}
__kernel void mainzisegred_nonseg_20024(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_20968_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_20966_backing_aligned_1,
                                        int64_t paths_18599,
                                        int64_t num_groups_20019, __global
                                        unsigned char *mem_20221, __global
                                        unsigned char *mem_20225, __global
                                        unsigned char *mainzicounter_mem_20956,
                                        __global
                                        unsigned char *group_res_arr_mem_20958,
                                        int64_t num_threads_20960)
{
    #define segred_group_sizze_20017 (mainzisegred_group_sizze_20016)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_20968_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_20968_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_20966_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_20966_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20961;
    int32_t local_tid_20962;
    int64_t group_sizze_20965;
    int32_t wave_sizze_20964;
    int32_t group_tid_20963;
    
    global_tid_20961 = get_global_id(0);
    local_tid_20962 = get_local_id(0);
    group_sizze_20965 = get_local_size(0);
    wave_sizze_20964 = LOCKSTEP_WIDTH;
    group_tid_20963 = get_group_id(0);
    
    int32_t phys_tid_20024;
    
    phys_tid_20024 = global_tid_20961;
    
    __local char *sync_arr_mem_20966;
    
    sync_arr_mem_20966 = (__local char *) sync_arr_mem_20966_backing_0;
    
    __local char *red_arr_mem_20968;
    
    red_arr_mem_20968 = (__local char *) red_arr_mem_20968_backing_1;
    
    int64_t dummy_20022;
    
    dummy_20022 = 0;
    
    int64_t gtid_20023;
    
    gtid_20023 = 0;
    
    float x_acc_20970;
    int64_t chunk_sizze_20971;
    
    chunk_sizze_20971 = smin64(sdiv_up64(paths_18599,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_20017 *
                                         num_groups_20019))),
                               sdiv_up64(paths_18599 -
                                         sext_i32_i64(phys_tid_20024),
                                         num_threads_20960));
    
    float x_19060;
    float x_19061;
    
    // neutral-initialise the accumulators
    {
        x_acc_20970 = 0.0F;
    }
    for (int64_t i_20975 = 0; i_20975 < chunk_sizze_20971; i_20975++) {
        gtid_20023 = sext_i32_i64(phys_tid_20024) + num_threads_20960 * i_20975;
        // apply map function
        {
            float x_19063 = ((__global float *) mem_20221)[gtid_20023];
            float res_19064 = fmax32(0.0F, x_19063);
            
            // save map-out results
            { }
            // load accumulator
            {
                x_19060 = x_acc_20970;
            }
            // load new values
            {
                x_19061 = res_19064;
            }
            // apply reduction operator
            {
                float res_19062 = x_19060 + x_19061;
                
                // store in accumulator
                {
                    x_acc_20970 = res_19062;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_19060 = x_acc_20970;
        ((__local float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962)] =
            x_19060;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_20976;
    int32_t skip_waves_20977;
    
    skip_waves_20977 = 1;
    
    float x_20972;
    float x_20973;
    
    offset_20976 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_20962, sext_i64_i32(segred_group_sizze_20017))) {
            x_20972 = ((__local
                        float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962 +
                                                    offset_20976)];
        }
    }
    offset_20976 = 1;
    while (slt32(offset_20976, wave_sizze_20964)) {
        if (slt32(local_tid_20962 + offset_20976,
                  sext_i64_i32(segred_group_sizze_20017)) && ((local_tid_20962 -
                                                               squot32(local_tid_20962,
                                                                       wave_sizze_20964) *
                                                               wave_sizze_20964) &
                                                              (2 *
                                                               offset_20976 -
                                                               1)) == 0) {
            // read array element
            {
                x_20973 = ((volatile __local
                            float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962 +
                                                        offset_20976)];
            }
            // apply reduction operation
            {
                float res_20974 = x_20972 + x_20973;
                
                x_20972 = res_20974;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962)] =
                    x_20972;
            }
        }
        offset_20976 *= 2;
    }
    while (slt32(skip_waves_20977,
                 squot32(sext_i64_i32(segred_group_sizze_20017) +
                         wave_sizze_20964 - 1, wave_sizze_20964))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_20976 = skip_waves_20977 * wave_sizze_20964;
        if (slt32(local_tid_20962 + offset_20976,
                  sext_i64_i32(segred_group_sizze_20017)) && ((local_tid_20962 -
                                                               squot32(local_tid_20962,
                                                                       wave_sizze_20964) *
                                                               wave_sizze_20964) ==
                                                              0 &&
                                                              (squot32(local_tid_20962,
                                                                       wave_sizze_20964) &
                                                               (2 *
                                                                skip_waves_20977 -
                                                                1)) == 0)) {
            // read array element
            {
                x_20973 = ((__local
                            float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962 +
                                                        offset_20976)];
            }
            // apply reduction operation
            {
                float res_20974 = x_20972 + x_20973;
                
                x_20972 = res_20974;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962)] =
                    x_20972;
            }
        }
        skip_waves_20977 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_20962) == 0) {
            x_acc_20970 = x_20972;
        }
    }
    
    int32_t old_counter_20978;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_20962 == 0) {
            ((__global
              float *) group_res_arr_mem_20958)[sext_i32_i64(group_tid_20963) *
                                                segred_group_sizze_20017] =
                x_acc_20970;
            mem_fence_global();
            old_counter_20978 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20956)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_20966)[0] = old_counter_20978 ==
                num_groups_20019 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_20979;
    
    is_last_group_20979 = ((__local bool *) sync_arr_mem_20966)[0];
    if (is_last_group_20979) {
        if (local_tid_20962 == 0) {
            old_counter_20978 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20956)[0],
                                                      (int) (0 -
                                                             num_groups_20019));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_20980 = sdiv_up64(num_groups_20019,
                                                      segred_group_sizze_20017);
            
            x_19060 = 0.0F;
            for (int64_t i_20981 = 0; i_20981 < read_per_thread_20980;
                 i_20981++) {
                int64_t group_res_id_20982 = sext_i32_i64(local_tid_20962) *
                        read_per_thread_20980 + i_20981;
                int64_t index_of_group_res_20983 = group_res_id_20982;
                
                if (slt64(group_res_id_20982, num_groups_20019)) {
                    x_19061 = ((__global
                                float *) group_res_arr_mem_20958)[index_of_group_res_20983 *
                                                                  segred_group_sizze_20017];
                    
                    float res_19062;
                    
                    res_19062 = x_19060 + x_19061;
                    x_19060 = res_19062;
                }
            }
        }
        ((__local float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962)] =
            x_19060;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_20984;
            int32_t skip_waves_20985;
            
            skip_waves_20985 = 1;
            
            float x_20972;
            float x_20973;
            
            offset_20984 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_20962,
                          sext_i64_i32(segred_group_sizze_20017))) {
                    x_20972 = ((__local
                                float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962 +
                                                            offset_20984)];
                }
            }
            offset_20984 = 1;
            while (slt32(offset_20984, wave_sizze_20964)) {
                if (slt32(local_tid_20962 + offset_20984,
                          sext_i64_i32(segred_group_sizze_20017)) &&
                    ((local_tid_20962 - squot32(local_tid_20962,
                                                wave_sizze_20964) *
                      wave_sizze_20964) & (2 * offset_20984 - 1)) == 0) {
                    // read array element
                    {
                        x_20973 = ((volatile __local
                                    float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962 +
                                                                offset_20984)];
                    }
                    // apply reduction operation
                    {
                        float res_20974 = x_20972 + x_20973;
                        
                        x_20972 = res_20974;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962)] =
                            x_20972;
                    }
                }
                offset_20984 *= 2;
            }
            while (slt32(skip_waves_20985,
                         squot32(sext_i64_i32(segred_group_sizze_20017) +
                                 wave_sizze_20964 - 1, wave_sizze_20964))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_20984 = skip_waves_20985 * wave_sizze_20964;
                if (slt32(local_tid_20962 + offset_20984,
                          sext_i64_i32(segred_group_sizze_20017)) &&
                    ((local_tid_20962 - squot32(local_tid_20962,
                                                wave_sizze_20964) *
                      wave_sizze_20964) == 0 && (squot32(local_tid_20962,
                                                         wave_sizze_20964) &
                                                 (2 * skip_waves_20985 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_20973 = ((__local
                                    float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962 +
                                                                offset_20984)];
                    }
                    // apply reduction operation
                    {
                        float res_20974 = x_20972 + x_20973;
                        
                        x_20972 = res_20974;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_20968)[sext_i32_i64(local_tid_20962)] =
                            x_20972;
                    }
                }
                skip_waves_20985 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_20962 == 0) {
                    ((__global float *) mem_20225)[0] = x_20972;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20017
}
__kernel void mainzisegred_nonseg_20256(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_20363_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_20361_backing_aligned_1,
                                        int64_t n_18596, __global
                                        unsigned char *payments_mem_20097,
                                        __global unsigned char *mem_20268,
                                        __global
                                        unsigned char *mainzicounter_mem_20351,
                                        __global
                                        unsigned char *group_res_arr_mem_20353,
                                        int64_t num_threads_20355)
{
    #define segred_num_groups_20250 (mainzisegred_num_groups_20249)
    #define segred_group_sizze_20252 (mainzisegred_group_sizze_20251)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_20363_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_20363_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_20361_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_20361_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20356;
    int32_t local_tid_20357;
    int64_t group_sizze_20360;
    int32_t wave_sizze_20359;
    int32_t group_tid_20358;
    
    global_tid_20356 = get_global_id(0);
    local_tid_20357 = get_local_id(0);
    group_sizze_20360 = get_local_size(0);
    wave_sizze_20359 = LOCKSTEP_WIDTH;
    group_tid_20358 = get_group_id(0);
    
    int32_t phys_tid_20256;
    
    phys_tid_20256 = global_tid_20356;
    
    __local char *sync_arr_mem_20361;
    
    sync_arr_mem_20361 = (__local char *) sync_arr_mem_20361_backing_0;
    
    __local char *red_arr_mem_20363;
    
    red_arr_mem_20363 = (__local char *) red_arr_mem_20363_backing_1;
    
    int64_t dummy_20254;
    
    dummy_20254 = 0;
    
    int64_t gtid_20255;
    
    gtid_20255 = 0;
    
    int64_t x_acc_20365;
    int64_t chunk_sizze_20366;
    
    chunk_sizze_20366 = smin64(sdiv_up64(n_18596,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_20252 *
                                         segred_num_groups_20250))),
                               sdiv_up64(n_18596 - sext_i32_i64(phys_tid_20256),
                                         num_threads_20355));
    
    int64_t x_20257;
    int64_t y_20258;
    
    // neutral-initialise the accumulators
    {
        x_acc_20365 = 0;
    }
    for (int64_t i_20370 = 0; i_20370 < chunk_sizze_20366; i_20370++) {
        gtid_20255 = sext_i32_i64(phys_tid_20256) + num_threads_20355 * i_20370;
        // apply map function
        {
            int64_t res_20261 = ((__global
                                  int64_t *) payments_mem_20097)[gtid_20255];
            int64_t bytes_20262 = 4 * res_20261;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_20257 = x_acc_20365;
            }
            // load new values
            {
                y_20258 = bytes_20262;
            }
            // apply reduction operator
            {
                int64_t zz_20259 = smax64(x_20257, y_20258);
                
                // store in accumulator
                {
                    x_acc_20365 = zz_20259;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_20257 = x_acc_20365;
        ((__local int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357)] =
            x_20257;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_20371;
    int32_t skip_waves_20372;
    
    skip_waves_20372 = 1;
    
    int64_t x_20367;
    int64_t y_20368;
    
    offset_20371 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_20357, sext_i64_i32(segred_group_sizze_20252))) {
            x_20367 = ((__local
                        int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357 +
                                                      offset_20371)];
        }
    }
    offset_20371 = 1;
    while (slt32(offset_20371, wave_sizze_20359)) {
        if (slt32(local_tid_20357 + offset_20371,
                  sext_i64_i32(segred_group_sizze_20252)) && ((local_tid_20357 -
                                                               squot32(local_tid_20357,
                                                                       wave_sizze_20359) *
                                                               wave_sizze_20359) &
                                                              (2 *
                                                               offset_20371 -
                                                               1)) == 0) {
            // read array element
            {
                y_20368 = ((volatile __local
                            int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357 +
                                                          offset_20371)];
            }
            // apply reduction operation
            {
                int64_t zz_20369 = smax64(x_20367, y_20368);
                
                x_20367 = zz_20369;
            }
            // write result of operation
            {
                ((volatile __local
                  int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357)] =
                    x_20367;
            }
        }
        offset_20371 *= 2;
    }
    while (slt32(skip_waves_20372,
                 squot32(sext_i64_i32(segred_group_sizze_20252) +
                         wave_sizze_20359 - 1, wave_sizze_20359))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_20371 = skip_waves_20372 * wave_sizze_20359;
        if (slt32(local_tid_20357 + offset_20371,
                  sext_i64_i32(segred_group_sizze_20252)) && ((local_tid_20357 -
                                                               squot32(local_tid_20357,
                                                                       wave_sizze_20359) *
                                                               wave_sizze_20359) ==
                                                              0 &&
                                                              (squot32(local_tid_20357,
                                                                       wave_sizze_20359) &
                                                               (2 *
                                                                skip_waves_20372 -
                                                                1)) == 0)) {
            // read array element
            {
                y_20368 = ((__local
                            int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357 +
                                                          offset_20371)];
            }
            // apply reduction operation
            {
                int64_t zz_20369 = smax64(x_20367, y_20368);
                
                x_20367 = zz_20369;
            }
            // write result of operation
            {
                ((__local
                  int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357)] =
                    x_20367;
            }
        }
        skip_waves_20372 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_20357) == 0) {
            x_acc_20365 = x_20367;
        }
    }
    
    int32_t old_counter_20373;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_20357 == 0) {
            ((__global
              int64_t *) group_res_arr_mem_20353)[sext_i32_i64(group_tid_20358) *
                                                  segred_group_sizze_20252] =
                x_acc_20365;
            mem_fence_global();
            old_counter_20373 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20351)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_20361)[0] = old_counter_20373 ==
                segred_num_groups_20250 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_20374;
    
    is_last_group_20374 = ((__local bool *) sync_arr_mem_20361)[0];
    if (is_last_group_20374) {
        if (local_tid_20357 == 0) {
            old_counter_20373 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20351)[0],
                                                      (int) (0 -
                                                             segred_num_groups_20250));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_20375 = sdiv_up64(segred_num_groups_20250,
                                                      segred_group_sizze_20252);
            
            x_20257 = 0;
            for (int64_t i_20376 = 0; i_20376 < read_per_thread_20375;
                 i_20376++) {
                int64_t group_res_id_20377 = sext_i32_i64(local_tid_20357) *
                        read_per_thread_20375 + i_20376;
                int64_t index_of_group_res_20378 = group_res_id_20377;
                
                if (slt64(group_res_id_20377, segred_num_groups_20250)) {
                    y_20258 = ((__global
                                int64_t *) group_res_arr_mem_20353)[index_of_group_res_20378 *
                                                                    segred_group_sizze_20252];
                    
                    int64_t zz_20259;
                    
                    zz_20259 = smax64(x_20257, y_20258);
                    x_20257 = zz_20259;
                }
            }
        }
        ((__local int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357)] =
            x_20257;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_20379;
            int32_t skip_waves_20380;
            
            skip_waves_20380 = 1;
            
            int64_t x_20367;
            int64_t y_20368;
            
            offset_20379 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_20357,
                          sext_i64_i32(segred_group_sizze_20252))) {
                    x_20367 = ((__local
                                int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357 +
                                                              offset_20379)];
                }
            }
            offset_20379 = 1;
            while (slt32(offset_20379, wave_sizze_20359)) {
                if (slt32(local_tid_20357 + offset_20379,
                          sext_i64_i32(segred_group_sizze_20252)) &&
                    ((local_tid_20357 - squot32(local_tid_20357,
                                                wave_sizze_20359) *
                      wave_sizze_20359) & (2 * offset_20379 - 1)) == 0) {
                    // read array element
                    {
                        y_20368 = ((volatile __local
                                    int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357 +
                                                                  offset_20379)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_20369 = smax64(x_20367, y_20368);
                        
                        x_20367 = zz_20369;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357)] =
                            x_20367;
                    }
                }
                offset_20379 *= 2;
            }
            while (slt32(skip_waves_20380,
                         squot32(sext_i64_i32(segred_group_sizze_20252) +
                                 wave_sizze_20359 - 1, wave_sizze_20359))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_20379 = skip_waves_20380 * wave_sizze_20359;
                if (slt32(local_tid_20357 + offset_20379,
                          sext_i64_i32(segred_group_sizze_20252)) &&
                    ((local_tid_20357 - squot32(local_tid_20357,
                                                wave_sizze_20359) *
                      wave_sizze_20359) == 0 && (squot32(local_tid_20357,
                                                         wave_sizze_20359) &
                                                 (2 * skip_waves_20380 - 1)) ==
                     0)) {
                    // read array element
                    {
                        y_20368 = ((__local
                                    int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357 +
                                                                  offset_20379)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_20369 = smax64(x_20367, y_20368);
                        
                        x_20367 = zz_20369;
                    }
                    // write result of operation
                    {
                        ((__local
                          int64_t *) red_arr_mem_20363)[sext_i32_i64(local_tid_20357)] =
                            x_20367;
                    }
                }
                skip_waves_20380 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_20357 == 0) {
                    ((__global int64_t *) mem_20268)[0] = x_20367;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_num_groups_20250
    #undef segred_group_sizze_20252
}
__kernel void mainzisegred_nonseg_20288(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_20400_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_20398_backing_aligned_1,
                                        int64_t n_18596, __global
                                        unsigned char *payments_mem_20097,
                                        __global unsigned char *mem_20300,
                                        __global
                                        unsigned char *mainzicounter_mem_20388,
                                        __global
                                        unsigned char *group_res_arr_mem_20390,
                                        int64_t num_threads_20392)
{
    #define segred_num_groups_20282 (mainzisegred_num_groups_20281)
    #define segred_group_sizze_20284 (mainzisegred_group_sizze_20283)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_20400_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_20400_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_20398_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_20398_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20393;
    int32_t local_tid_20394;
    int64_t group_sizze_20397;
    int32_t wave_sizze_20396;
    int32_t group_tid_20395;
    
    global_tid_20393 = get_global_id(0);
    local_tid_20394 = get_local_id(0);
    group_sizze_20397 = get_local_size(0);
    wave_sizze_20396 = LOCKSTEP_WIDTH;
    group_tid_20395 = get_group_id(0);
    
    int32_t phys_tid_20288;
    
    phys_tid_20288 = global_tid_20393;
    
    __local char *sync_arr_mem_20398;
    
    sync_arr_mem_20398 = (__local char *) sync_arr_mem_20398_backing_0;
    
    __local char *red_arr_mem_20400;
    
    red_arr_mem_20400 = (__local char *) red_arr_mem_20400_backing_1;
    
    int64_t dummy_20286;
    
    dummy_20286 = 0;
    
    int64_t gtid_20287;
    
    gtid_20287 = 0;
    
    int64_t x_acc_20402;
    int64_t chunk_sizze_20403;
    
    chunk_sizze_20403 = smin64(sdiv_up64(n_18596,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_20284 *
                                         segred_num_groups_20282))),
                               sdiv_up64(n_18596 - sext_i32_i64(phys_tid_20288),
                                         num_threads_20392));
    
    int64_t x_20289;
    int64_t y_20290;
    
    // neutral-initialise the accumulators
    {
        x_acc_20402 = 0;
    }
    for (int64_t i_20407 = 0; i_20407 < chunk_sizze_20403; i_20407++) {
        gtid_20287 = sext_i32_i64(phys_tid_20288) + num_threads_20392 * i_20407;
        // apply map function
        {
            int64_t res_20293 = ((__global
                                  int64_t *) payments_mem_20097)[gtid_20287];
            int64_t bytes_20294 = 4 * res_20293;
            
            // save map-out results
            { }
            // load accumulator
            {
                x_20289 = x_acc_20402;
            }
            // load new values
            {
                y_20290 = bytes_20294;
            }
            // apply reduction operator
            {
                int64_t zz_20291 = smax64(x_20289, y_20290);
                
                // store in accumulator
                {
                    x_acc_20402 = zz_20291;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_20289 = x_acc_20402;
        ((__local int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394)] =
            x_20289;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_20408;
    int32_t skip_waves_20409;
    
    skip_waves_20409 = 1;
    
    int64_t x_20404;
    int64_t y_20405;
    
    offset_20408 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_20394, sext_i64_i32(segred_group_sizze_20284))) {
            x_20404 = ((__local
                        int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394 +
                                                      offset_20408)];
        }
    }
    offset_20408 = 1;
    while (slt32(offset_20408, wave_sizze_20396)) {
        if (slt32(local_tid_20394 + offset_20408,
                  sext_i64_i32(segred_group_sizze_20284)) && ((local_tid_20394 -
                                                               squot32(local_tid_20394,
                                                                       wave_sizze_20396) *
                                                               wave_sizze_20396) &
                                                              (2 *
                                                               offset_20408 -
                                                               1)) == 0) {
            // read array element
            {
                y_20405 = ((volatile __local
                            int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394 +
                                                          offset_20408)];
            }
            // apply reduction operation
            {
                int64_t zz_20406 = smax64(x_20404, y_20405);
                
                x_20404 = zz_20406;
            }
            // write result of operation
            {
                ((volatile __local
                  int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394)] =
                    x_20404;
            }
        }
        offset_20408 *= 2;
    }
    while (slt32(skip_waves_20409,
                 squot32(sext_i64_i32(segred_group_sizze_20284) +
                         wave_sizze_20396 - 1, wave_sizze_20396))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_20408 = skip_waves_20409 * wave_sizze_20396;
        if (slt32(local_tid_20394 + offset_20408,
                  sext_i64_i32(segred_group_sizze_20284)) && ((local_tid_20394 -
                                                               squot32(local_tid_20394,
                                                                       wave_sizze_20396) *
                                                               wave_sizze_20396) ==
                                                              0 &&
                                                              (squot32(local_tid_20394,
                                                                       wave_sizze_20396) &
                                                               (2 *
                                                                skip_waves_20409 -
                                                                1)) == 0)) {
            // read array element
            {
                y_20405 = ((__local
                            int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394 +
                                                          offset_20408)];
            }
            // apply reduction operation
            {
                int64_t zz_20406 = smax64(x_20404, y_20405);
                
                x_20404 = zz_20406;
            }
            // write result of operation
            {
                ((__local
                  int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394)] =
                    x_20404;
            }
        }
        skip_waves_20409 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_20394) == 0) {
            x_acc_20402 = x_20404;
        }
    }
    
    int32_t old_counter_20410;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_20394 == 0) {
            ((__global
              int64_t *) group_res_arr_mem_20390)[sext_i32_i64(group_tid_20395) *
                                                  segred_group_sizze_20284] =
                x_acc_20402;
            mem_fence_global();
            old_counter_20410 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20388)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_20398)[0] = old_counter_20410 ==
                segred_num_groups_20282 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_20411;
    
    is_last_group_20411 = ((__local bool *) sync_arr_mem_20398)[0];
    if (is_last_group_20411) {
        if (local_tid_20394 == 0) {
            old_counter_20410 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_20388)[0],
                                                      (int) (0 -
                                                             segred_num_groups_20282));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_20412 = sdiv_up64(segred_num_groups_20282,
                                                      segred_group_sizze_20284);
            
            x_20289 = 0;
            for (int64_t i_20413 = 0; i_20413 < read_per_thread_20412;
                 i_20413++) {
                int64_t group_res_id_20414 = sext_i32_i64(local_tid_20394) *
                        read_per_thread_20412 + i_20413;
                int64_t index_of_group_res_20415 = group_res_id_20414;
                
                if (slt64(group_res_id_20414, segred_num_groups_20282)) {
                    y_20290 = ((__global
                                int64_t *) group_res_arr_mem_20390)[index_of_group_res_20415 *
                                                                    segred_group_sizze_20284];
                    
                    int64_t zz_20291;
                    
                    zz_20291 = smax64(x_20289, y_20290);
                    x_20289 = zz_20291;
                }
            }
        }
        ((__local int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394)] =
            x_20289;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_20416;
            int32_t skip_waves_20417;
            
            skip_waves_20417 = 1;
            
            int64_t x_20404;
            int64_t y_20405;
            
            offset_20416 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_20394,
                          sext_i64_i32(segred_group_sizze_20284))) {
                    x_20404 = ((__local
                                int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394 +
                                                              offset_20416)];
                }
            }
            offset_20416 = 1;
            while (slt32(offset_20416, wave_sizze_20396)) {
                if (slt32(local_tid_20394 + offset_20416,
                          sext_i64_i32(segred_group_sizze_20284)) &&
                    ((local_tid_20394 - squot32(local_tid_20394,
                                                wave_sizze_20396) *
                      wave_sizze_20396) & (2 * offset_20416 - 1)) == 0) {
                    // read array element
                    {
                        y_20405 = ((volatile __local
                                    int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394 +
                                                                  offset_20416)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_20406 = smax64(x_20404, y_20405);
                        
                        x_20404 = zz_20406;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394)] =
                            x_20404;
                    }
                }
                offset_20416 *= 2;
            }
            while (slt32(skip_waves_20417,
                         squot32(sext_i64_i32(segred_group_sizze_20284) +
                                 wave_sizze_20396 - 1, wave_sizze_20396))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_20416 = skip_waves_20417 * wave_sizze_20396;
                if (slt32(local_tid_20394 + offset_20416,
                          sext_i64_i32(segred_group_sizze_20284)) &&
                    ((local_tid_20394 - squot32(local_tid_20394,
                                                wave_sizze_20396) *
                      wave_sizze_20396) == 0 && (squot32(local_tid_20394,
                                                         wave_sizze_20396) &
                                                 (2 * skip_waves_20417 - 1)) ==
                     0)) {
                    // read array element
                    {
                        y_20405 = ((__local
                                    int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394 +
                                                                  offset_20416)];
                    }
                    // apply reduction operation
                    {
                        int64_t zz_20406 = smax64(x_20404, y_20405);
                        
                        x_20404 = zz_20406;
                    }
                    // write result of operation
                    {
                        ((__local
                          int64_t *) red_arr_mem_20400)[sext_i32_i64(local_tid_20394)] =
                            x_20404;
                    }
                }
                skip_waves_20417 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_20394 == 0) {
                    ((__global int64_t *) mem_20300)[0] = x_20404;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_num_groups_20282
    #undef segred_group_sizze_20284
}
__kernel void mainzisegred_small_20629(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_20637_backing_aligned_0,
                                       int64_t res_18845,
                                       int64_t num_groups_19922, __global
                                       unsigned char *mem_20198,
                                       int32_t num_subhistos_20543, __global
                                       unsigned char *res_subhistos_mem_20544,
                                       int64_t segment_sizze_nonzzero_20630)
{
    #define seghist_group_sizze_19920 (mainziseghist_group_sizze_19919)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_20637_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_20637_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_20632;
    int32_t local_tid_20633;
    int64_t group_sizze_20636;
    int32_t wave_sizze_20635;
    int32_t group_tid_20634;
    
    global_tid_20632 = get_global_id(0);
    local_tid_20633 = get_local_id(0);
    group_sizze_20636 = get_local_size(0);
    wave_sizze_20635 = LOCKSTEP_WIDTH;
    group_tid_20634 = get_group_id(0);
    
    int32_t flat_gtid_20629;
    
    flat_gtid_20629 = global_tid_20632;
    
    __local char *red_arr_mem_20637;
    
    red_arr_mem_20637 = (__local char *) red_arr_mem_20637_backing_0;
    
    int32_t phys_group_id_20639;
    
    phys_group_id_20639 = get_group_id(0);
    for (int32_t i_20640 = 0; i_20640 <
         sdiv_up32(sext_i64_i32(sdiv_up64(res_18845,
                                          squot64(seghist_group_sizze_19920,
                                                  segment_sizze_nonzzero_20630))) -
                   phys_group_id_20639, sext_i64_i32(num_groups_19922));
         i_20640++) {
        int32_t virt_group_id_20641 = phys_group_id_20639 + i_20640 *
                sext_i64_i32(num_groups_19922);
        int64_t bucket_id_20627 = squot64(sext_i32_i64(local_tid_20633),
                                          segment_sizze_nonzzero_20630) +
                sext_i32_i64(virt_group_id_20641) *
                squot64(seghist_group_sizze_19920,
                        segment_sizze_nonzzero_20630);
        int64_t subhistogram_id_20628 = srem64(sext_i32_i64(local_tid_20633),
                                               num_subhistos_20543);
        
        // apply map function if in bounds
        {
            if (slt64(0, num_subhistos_20543) && (slt64(bucket_id_20627,
                                                        res_18845) &&
                                                  slt64(sext_i32_i64(local_tid_20633),
                                                        num_subhistos_20543 *
                                                        squot64(seghist_group_sizze_19920,
                                                                segment_sizze_nonzzero_20630)))) {
                // save results to be reduced
                {
                    ((__local
                      int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)] =
                        ((__global
                          int64_t *) res_subhistos_mem_20544)[subhistogram_id_20628 *
                                                              res_18845 +
                                                              bucket_id_20627];
                }
            } else {
                ((__local
                  int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)] =
                    0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt64(0, num_subhistos_20543)) {
            // perform segmented scan to imitate reduction
            {
                int64_t x_19926;
                int64_t x_19927;
                int64_t x_20642;
                int64_t x_20643;
                bool ltid_in_bounds_20645;
                
                ltid_in_bounds_20645 = slt64(sext_i32_i64(local_tid_20633),
                                             num_subhistos_20543 *
                                             squot64(seghist_group_sizze_19920,
                                                     segment_sizze_nonzzero_20630));
                
                int32_t skip_threads_20646;
                
                // read input for in-block scan
                {
                    if (ltid_in_bounds_20645) {
                        x_19927 = ((volatile __local
                                    int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)];
                        if ((local_tid_20633 - squot32(local_tid_20633, 32) *
                             32) == 0) {
                            x_19926 = x_19927;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_20646 = 1;
                    while (slt32(skip_threads_20646, 32)) {
                        if (sle32(skip_threads_20646, local_tid_20633 -
                                  squot32(local_tid_20633, 32) * 32) &&
                            ltid_in_bounds_20645) {
                            // read operands
                            {
                                x_19926 = ((volatile __local
                                            int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633) -
                                                                          sext_i32_i64(skip_threads_20646)];
                            }
                            // perform operation
                            {
                                bool inactive_20647 =
                                     slt64(srem64(sext_i32_i64(local_tid_20633),
                                                  num_subhistos_20543),
                                           sext_i32_i64(local_tid_20633) -
                                           sext_i32_i64(local_tid_20633 -
                                           skip_threads_20646));
                                
                                if (inactive_20647) {
                                    x_19926 = x_19927;
                                }
                                if (!inactive_20647) {
                                    int64_t res_19928 = smax64(x_19926,
                                                               x_19927);
                                    
                                    x_19926 = res_19928;
                                }
                            }
                        }
                        if (sle32(wave_sizze_20635, skip_threads_20646)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_20646, local_tid_20633 -
                                  squot32(local_tid_20633, 32) * 32) &&
                            ltid_in_bounds_20645) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)] =
                                    x_19926;
                                x_19927 = x_19926;
                            }
                        }
                        if (sle32(wave_sizze_20635, skip_threads_20646)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_20646 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_20633 - squot32(local_tid_20633, 32) * 32) ==
                        31 && ltid_in_bounds_20645) {
                        ((volatile __local
                          int64_t *) red_arr_mem_20637)[sext_i32_i64(squot32(local_tid_20633,
                                                                             32))] =
                            x_19926;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_20648;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_20633, 32) == 0 &&
                            ltid_in_bounds_20645) {
                            x_20643 = ((volatile __local
                                        int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)];
                            if ((local_tid_20633 - squot32(local_tid_20633,
                                                           32) * 32) == 0) {
                                x_20642 = x_20643;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_20648 = 1;
                        while (slt32(skip_threads_20648, 32)) {
                            if (sle32(skip_threads_20648, local_tid_20633 -
                                      squot32(local_tid_20633, 32) * 32) &&
                                (squot32(local_tid_20633, 32) == 0 &&
                                 ltid_in_bounds_20645)) {
                                // read operands
                                {
                                    x_20642 = ((volatile __local
                                                int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633) -
                                                                              sext_i32_i64(skip_threads_20648)];
                                }
                                // perform operation
                                {
                                    bool inactive_20649 =
                                         slt64(srem64(sext_i32_i64(local_tid_20633 *
                                                      32 + 32 - 1),
                                                      num_subhistos_20543),
                                               sext_i32_i64(local_tid_20633 *
                                               32 + 32 - 1) -
                                               sext_i32_i64((local_tid_20633 -
                                                             skip_threads_20648) *
                                               32 + 32 - 1));
                                    
                                    if (inactive_20649) {
                                        x_20642 = x_20643;
                                    }
                                    if (!inactive_20649) {
                                        int64_t res_20644 = smax64(x_20642,
                                                                   x_20643);
                                        
                                        x_20642 = res_20644;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_20635, skip_threads_20648)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_20648, local_tid_20633 -
                                      squot32(local_tid_20633, 32) * 32) &&
                                (squot32(local_tid_20633, 32) == 0 &&
                                 ltid_in_bounds_20645)) {
                                // write result
                                {
                                    ((volatile __local
                                      int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)] =
                                        x_20642;
                                    x_20643 = x_20642;
                                }
                            }
                            if (sle32(wave_sizze_20635, skip_threads_20648)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_20648 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_20633, 32) == 0 ||
                          !ltid_in_bounds_20645)) {
                        // read operands
                        {
                            x_19927 = x_19926;
                            x_19926 = ((__local
                                        int64_t *) red_arr_mem_20637)[sext_i32_i64(squot32(local_tid_20633,
                                                                                           32)) -
                                                                      1];
                        }
                        // perform operation
                        {
                            bool inactive_20650 =
                                 slt64(srem64(sext_i32_i64(local_tid_20633),
                                              num_subhistos_20543),
                                       sext_i32_i64(local_tid_20633) -
                                       sext_i32_i64(squot32(local_tid_20633,
                                                            32) * 32 - 1));
                            
                            if (inactive_20650) {
                                x_19926 = x_19927;
                            }
                            if (!inactive_20650) {
                                int64_t res_19928 = smax64(x_19926, x_19927);
                                
                                x_19926 = res_19928;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)] =
                                x_19926;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_20633, 32) == 0) {
                        ((__local
                          int64_t *) red_arr_mem_20637)[sext_i32_i64(local_tid_20633)] =
                            x_19927;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt64(sext_i32_i64(virt_group_id_20641) *
                      squot64(seghist_group_sizze_19920,
                              segment_sizze_nonzzero_20630) +
                      sext_i32_i64(local_tid_20633), res_18845) &&
                slt64(sext_i32_i64(local_tid_20633),
                      squot64(seghist_group_sizze_19920,
                              segment_sizze_nonzzero_20630))) {
                ((__global
                  int64_t *) mem_20198)[sext_i32_i64(virt_group_id_20641) *
                                        squot64(seghist_group_sizze_19920,
                                                segment_sizze_nonzzero_20630) +
                                        sext_i32_i64(local_tid_20633)] =
                    ((__local
                      int64_t *) red_arr_mem_20637)[(sext_i32_i64(local_tid_20633) +
                                                     1) *
                                                    segment_sizze_nonzzero_20630 -
                                                    1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef seghist_group_sizze_19920
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
    self.failure_msgs=["Range {}..{}...{} is invalid.\n-> #0  cva.fut:54:29-52\n   #1  cva.fut:102:25-65\n   #2  cva.fut:116:16-62\n   #3  cva.fut:112:17-116:85\n   #4  cva.fut:107:1-177:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:103:47-70\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-177:18\n",
     "Index [{}:] out of bounds for array of shape [{}].\n-> #0  cva.fut:104:74-90\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-177:18\n",
     "Range {}..{}...{} is invalid.\n-> #0  cva.fut:54:29-52\n   #1  cva.fut:102:25-65\n   #2  cva.fut:116:16-62\n   #3  cva.fut:112:17-116:85\n   #4  cva.fut:107:1-177:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:103:47-70\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-177:18\n",
     "Index [{}:] out of bounds for array of shape [{}].\n-> #0  cva.fut:104:74-90\n   #1  cva.fut:116:16-62\n   #2  cva.fut:112:17-116:85\n   #3  cva.fut:107:1-177:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:71:97-104\n   #1  cva.fut:130:32-62\n   #2  cva.fut:130:22-69\n   #3  cva.fut:107:1-177:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:71:58-105\n   #1  cva.fut:130:32-62\n   #2  cva.fut:130:22-69\n   #3  cva.fut:107:1-177:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:142:37-88\n   #6  cva.fut:141:18-144:79\n   #7  cva.fut:107:1-177:18\n"]
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
                                       all_sizes={"builtin#replicate_f32.group_size_20949": {"class": "group_size",
                                                                                   "value": None},
                                        "builtin#replicate_i64.group_size_20541": {"class": "group_size",
                                                                                   "value": None},
                                        "main.L2_size_20606": {"class": "L2_for_histogram", "value": 4194304},
                                        "main.seghist_group_size_19919": {"class": "group_size", "value": None},
                                        "main.seghist_num_groups_19921": {"class": "num_groups", "value": None},
                                        "main.segmap_group_size_19111": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19287": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19620": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19719": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19782": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20012": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_19622": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_19087": {"class": "group_size", "value": None},
                                        "main.segred_group_size_19909": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20016": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20251": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20283": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_19089": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_19911": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20018": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20249": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20281": {"class": "num_groups", "value": None},
                                        "main.segscan_group_size_19901": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_19934": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_19942": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_19950": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_20002": {"class": "group_size", "value": None},
                                        "main.segscan_num_groups_19903": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_19936": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_19944": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_19952": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_20004": {"class": "num_groups", "value": None},
                                        "main.suff_outer_par_0": {"class": "threshold ()", "value": None}})
    self.builtinzhreplicate_f32zireplicate_20946_var = program.builtinzhreplicate_f32zireplicate_20946
    self.builtinzhreplicate_i64zireplicate_20538_var = program.builtinzhreplicate_i64zireplicate_20538
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainziscan_stage1_19907_var = program.mainziscan_stage1_19907
    self.mainziscan_stage1_19940_var = program.mainziscan_stage1_19940
    self.mainziscan_stage1_19948_var = program.mainziscan_stage1_19948
    self.mainziscan_stage1_19956_var = program.mainziscan_stage1_19956
    self.mainziscan_stage1_20008_var = program.mainziscan_stage1_20008
    self.mainziscan_stage2_19907_var = program.mainziscan_stage2_19907
    self.mainziscan_stage2_19940_var = program.mainziscan_stage2_19940
    self.mainziscan_stage2_19948_var = program.mainziscan_stage2_19948
    self.mainziscan_stage2_19956_var = program.mainziscan_stage2_19956
    self.mainziscan_stage2_20008_var = program.mainziscan_stage2_20008
    self.mainziscan_stage3_19907_var = program.mainziscan_stage3_19907
    self.mainziscan_stage3_19940_var = program.mainziscan_stage3_19940
    self.mainziscan_stage3_19948_var = program.mainziscan_stage3_19948
    self.mainziscan_stage3_19956_var = program.mainziscan_stage3_19956
    self.mainziscan_stage3_20008_var = program.mainziscan_stage3_20008
    self.mainziseghist_global_19925_var = program.mainziseghist_global_19925
    self.mainziseghist_local_19925_var = program.mainziseghist_local_19925
    self.mainzisegmap_19109_var = program.mainzisegmap_19109
    self.mainzisegmap_19285_var = program.mainzisegmap_19285
    self.mainzisegmap_19618_var = program.mainzisegmap_19618
    self.mainzisegmap_19716_var = program.mainzisegmap_19716
    self.mainzisegmap_19780_var = program.mainzisegmap_19780
    self.mainzisegmap_20010_var = program.mainzisegmap_20010
    self.mainzisegred_large_20629_var = program.mainzisegred_large_20629
    self.mainzisegred_nonseg_19095_var = program.mainzisegred_nonseg_19095
    self.mainzisegred_nonseg_19917_var = program.mainzisegred_nonseg_19917
    self.mainzisegred_nonseg_20024_var = program.mainzisegred_nonseg_20024
    self.mainzisegred_nonseg_20256_var = program.mainzisegred_nonseg_20256
    self.mainzisegred_nonseg_20288_var = program.mainzisegred_nonseg_20288
    self.mainzisegred_small_20629_var = program.mainzisegred_small_20629
    self.constants = {}
    mainzicounter_mem_20321 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_20986 = opencl_alloc(self, 40, "static_mem_20986")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_20986,
                      normaliseArray(mainzicounter_mem_20321),
                      is_blocking=synchronous)
    self.mainzicounter_mem_20321 = static_mem_20986
    mainzicounter_mem_20351 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_20988 = opencl_alloc(self, 40, "static_mem_20988")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_20988,
                      normaliseArray(mainzicounter_mem_20351),
                      is_blocking=synchronous)
    self.mainzicounter_mem_20351 = static_mem_20988
    mainzicounter_mem_20388 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_20990 = opencl_alloc(self, 40, "static_mem_20990")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_20990,
                      normaliseArray(mainzicounter_mem_20388),
                      is_blocking=synchronous)
    self.mainzicounter_mem_20388 = static_mem_20990
    mainzicounter_mem_20504 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_20996 = opencl_alloc(self, 40, "static_mem_20996")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_20996,
                      normaliseArray(mainzicounter_mem_20504),
                      is_blocking=synchronous)
    self.mainzicounter_mem_20504 = static_mem_20996
    mainzihist_locks_mem_20614 = np.zeros(100151, dtype=np.int32)
    static_mem_21000 = opencl_alloc(self, 400604, "static_mem_21000")
    if (400604 != 0):
      cl.enqueue_copy(self.queue, static_mem_21000,
                      normaliseArray(mainzihist_locks_mem_20614),
                      is_blocking=synchronous)
    self.mainzihist_locks_mem_20614 = static_mem_21000
    mainzicounter_mem_20658 = np.zeros(10240, dtype=np.int32)
    static_mem_21003 = opencl_alloc(self, 40960, "static_mem_21003")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_21003,
                      normaliseArray(mainzicounter_mem_20658),
                      is_blocking=synchronous)
    self.mainzicounter_mem_20658 = static_mem_21003
    mainzicounter_mem_20956 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_21005 = opencl_alloc(self, 40, "static_mem_21005")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_21005,
                      normaliseArray(mainzicounter_mem_20956),
                      is_blocking=synchronous)
    self.mainzicounter_mem_20956 = static_mem_21005
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
  def futhark_builtinzhreplicate_f32(self, mem_20942, num_elems_20943,
                                     val_20944):
    group_sizze_20949 = self.sizes["builtin#replicate_f32.group_size_20949"]
    num_groups_20950 = sdiv_up64(num_elems_20943, group_sizze_20949)
    if ((1 * (np.long(num_groups_20950) * np.long(group_sizze_20949))) != 0):
      self.builtinzhreplicate_f32zireplicate_20946_var.set_args(mem_20942,
                                                                np.int32(num_elems_20943),
                                                                np.float32(val_20944))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f32zireplicate_20946_var,
                                 ((np.long(num_groups_20950) * np.long(group_sizze_20949)),),
                                 (np.long(group_sizze_20949),))
      if synchronous:
        sync(self)
    return ()
  def futhark_builtinzhreplicate_i64(self, mem_20534, num_elems_20535,
                                     val_20536):
    group_sizze_20541 = self.sizes["builtin#replicate_i64.group_size_20541"]
    num_groups_20542 = sdiv_up64(num_elems_20535, group_sizze_20541)
    if ((1 * (np.long(num_groups_20542) * np.long(group_sizze_20541))) != 0):
      self.builtinzhreplicate_i64zireplicate_20538_var.set_args(mem_20534,
                                                                np.int32(num_elems_20535),
                                                                np.int64(val_20536))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_i64zireplicate_20538_var,
                                 ((np.long(num_groups_20542) * np.long(group_sizze_20541)),),
                                 (np.long(group_sizze_20541),))
      if synchronous:
        sync(self)
    return ()
  def futhark_main(self, swap_term_mem_20096, payments_mem_20097,
                   notional_mem_20098, n_18596, n_18597, n_18598, paths_18599,
                   steps_18600, a_18604, b_18605, sigma_18606, r0_18607):
    dim_match_18608 = (n_18596 == n_18597)
    empty_or_match_cert_18609 = True
    assert dim_match_18608, ("Error: %s\n\nBacktrace:\n-> #0  cva.fut:107:1-177:18\n" % ("function arguments of wrong shape",))
    segred_group_sizze_19088 = self.sizes["main.segred_group_size_19087"]
    max_num_groups_20320 = self.sizes["main.segred_num_groups_19089"]
    num_groups_19090 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(n_18596,
                                                            sext_i32_i64(segred_group_sizze_19088)),
                                                  sext_i32_i64(max_num_groups_20320))))
    mem_20101 = opencl_alloc(self, np.int64(4), "mem_20101")
    mainzicounter_mem_20321 = self.mainzicounter_mem_20321
    group_res_arr_mem_20323 = opencl_alloc(self,
                                           (np.int32(4) * (segred_group_sizze_19088 * num_groups_19090)),
                                           "group_res_arr_mem_20323")
    num_threads_20325 = (num_groups_19090 * segred_group_sizze_19088)
    if ((1 * (np.long(num_groups_19090) * np.long(segred_group_sizze_19088))) != 0):
      self.mainzisegred_nonseg_19095_var.set_args(self.global_failure,
                                                  cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_19088))),
                                                  cl.LocalMemory(np.long(np.int32(1))),
                                                  np.int64(n_18596),
                                                  np.int64(num_groups_19090),
                                                  swap_term_mem_20096,
                                                  payments_mem_20097, mem_20101,
                                                  mainzicounter_mem_20321,
                                                  group_res_arr_mem_20323,
                                                  np.int64(num_threads_20325))
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegred_nonseg_19095_var,
                                 ((np.long(num_groups_19090) * np.long(segred_group_sizze_19088)),),
                                 (np.long(segred_group_sizze_19088),))
      if synchronous:
        sync(self)
    read_res_20987 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_20987, mem_20101,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_18613 = read_res_20987[0]
    mem_20101 = None
    res_18621 = sitofp_i64_f32(steps_18600)
    dt_18622 = (res_18613 / res_18621)
    x_18624 = fpow32(a_18604, np.float32(2.0))
    x_18625 = (b_18605 * x_18624)
    x_18626 = fpow32(sigma_18606, np.float32(2.0))
    y_18627 = (x_18626 / np.float32(2.0))
    y_18628 = (x_18625 - y_18627)
    y_18629 = (np.float32(4.0) * a_18604)
    suff_outer_par_19097 = (self.sizes["main.suff_outer_par_0"] <= n_18596)
    segmap_group_sizze_19187 = self.sizes["main.segmap_group_size_19111"]
    segmap_group_sizze_19363 = self.sizes["main.segmap_group_size_19287"]
    bytes_20102 = (np.int64(4) * n_18596)
    bytes_20104 = (np.int64(8) * n_18596)
    segred_num_groups_20250 = self.sizes["main.segred_num_groups_20249"]
    segred_group_sizze_20252 = self.sizes["main.segred_group_size_20251"]
    segred_num_groups_20282 = self.sizes["main.segred_num_groups_20281"]
    segred_group_sizze_20284 = self.sizes["main.segred_group_size_20283"]
    local_memory_capacity_20425 = self.max_local_memory
    if ((sle64((np.int32(1) + (np.int32(8) * segred_group_sizze_20252)),
               sext_i32_i64(local_memory_capacity_20425)) and sle64(np.int64(0),
                                                                    sext_i32_i64(local_memory_capacity_20425))) and suff_outer_par_19097):
      segmap_usable_groups_19188 = sdiv_up64(n_18596, segmap_group_sizze_19187)
      mem_20103 = opencl_alloc(self, bytes_20102, "mem_20103")
      if ((n_18596 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_20103, swap_term_mem_20096,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_18596 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_20105 = opencl_alloc(self, bytes_20104, "mem_20105")
      if ((n_18596 * np.int32(8)) != 0):
        cl.enqueue_copy(self.queue, mem_20105, payments_mem_20097,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_18596 * np.int32(8))))
      if synchronous:
        sync(self)
      mem_20107 = opencl_alloc(self, bytes_20102, "mem_20107")
      if ((n_18596 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_20107, notional_mem_20098,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_18596 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_20124 = opencl_alloc(self, bytes_20102, "mem_20124")
      num_threads_20240 = (segmap_group_sizze_19187 * segmap_usable_groups_19188)
      mem_20268 = opencl_alloc(self, np.int64(8), "mem_20268")
      mainzicounter_mem_20351 = self.mainzicounter_mem_20351
      group_res_arr_mem_20353 = opencl_alloc(self,
                                             (np.int32(8) * (segred_group_sizze_20252 * segred_num_groups_20250)),
                                             "group_res_arr_mem_20353")
      num_threads_20355 = (segred_num_groups_20250 * segred_group_sizze_20252)
      if ((1 * (np.long(segred_num_groups_20250) * np.long(segred_group_sizze_20252))) != 0):
        self.mainzisegred_nonseg_20256_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(8) * segred_group_sizze_20252))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(n_18596),
                                                    payments_mem_20097,
                                                    mem_20268,
                                                    mainzicounter_mem_20351,
                                                    group_res_arr_mem_20353,
                                                    np.int64(num_threads_20355))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_20256_var,
                                   ((np.long(segred_num_groups_20250) * np.long(segred_group_sizze_20252)),),
                                   (np.long(segred_group_sizze_20252),))
        if synchronous:
          sync(self)
      read_res_20989 = np.empty(1, dtype=ct.c_int64)
      cl.enqueue_copy(self.queue, read_res_20989, mem_20268,
                      device_offset=(np.long(np.int64(0)) * 8),
                      is_blocking=synchronous)
      sync(self)
      max_per_thread_20246 = read_res_20989[0]
      mem_20268 = None
      sizze_sum_20263 = (num_threads_20240 * max_per_thread_20246)
      mem_20110 = opencl_alloc(self, sizze_sum_20263, "mem_20110")
      if ((1 * (np.long(segmap_usable_groups_19188) * np.long(segmap_group_sizze_19187))) != 0):
        self.mainzisegmap_19109_var.set_args(self.global_failure,
                                             self.failure_is_an_option,
                                             self.global_failure_args,
                                             np.int64(n_18596),
                                             np.float32(a_18604),
                                             np.float32(r0_18607),
                                             np.float32(x_18624),
                                             np.float32(x_18626),
                                             np.float32(y_18628),
                                             np.float32(y_18629),
                                             swap_term_mem_20096,
                                             payments_mem_20097, mem_20110,
                                             mem_20124,
                                             np.int64(num_threads_20240))
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19109_var,
                                   ((np.long(segmap_usable_groups_19188) * np.long(segmap_group_sizze_19187)),),
                                   (np.long(segmap_group_sizze_19187),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      mem_20110 = None
      res_mem_20148 = mem_20124
      res_mem_20149 = mem_20107
      res_mem_20150 = mem_20105
      res_mem_20151 = mem_20103
    else:
      segmap_usable_groups_19364 = sdiv_up64(n_18596, segmap_group_sizze_19363)
      mem_20126 = opencl_alloc(self, bytes_20102, "mem_20126")
      if ((n_18596 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_20126, swap_term_mem_20096,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_18596 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_20128 = opencl_alloc(self, bytes_20104, "mem_20128")
      if ((n_18596 * np.int32(8)) != 0):
        cl.enqueue_copy(self.queue, mem_20128, payments_mem_20097,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_18596 * np.int32(8))))
      if synchronous:
        sync(self)
      mem_20130 = opencl_alloc(self, bytes_20102, "mem_20130")
      if ((n_18596 * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_20130, notional_mem_20098,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((n_18596 * np.int32(4))))
      if synchronous:
        sync(self)
      mem_20147 = opencl_alloc(self, bytes_20102, "mem_20147")
      num_threads_20272 = (segmap_group_sizze_19363 * segmap_usable_groups_19364)
      mem_20300 = opencl_alloc(self, np.int64(8), "mem_20300")
      mainzicounter_mem_20388 = self.mainzicounter_mem_20388
      group_res_arr_mem_20390 = opencl_alloc(self,
                                             (np.int32(8) * (segred_group_sizze_20284 * segred_num_groups_20282)),
                                             "group_res_arr_mem_20390")
      num_threads_20392 = (segred_num_groups_20282 * segred_group_sizze_20284)
      if ((1 * (np.long(segred_num_groups_20282) * np.long(segred_group_sizze_20284))) != 0):
        self.mainzisegred_nonseg_20288_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(8) * segred_group_sizze_20284))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(n_18596),
                                                    payments_mem_20097,
                                                    mem_20300,
                                                    mainzicounter_mem_20388,
                                                    group_res_arr_mem_20390,
                                                    np.int64(num_threads_20392))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_20288_var,
                                   ((np.long(segred_num_groups_20282) * np.long(segred_group_sizze_20284)),),
                                   (np.long(segred_group_sizze_20284),))
        if synchronous:
          sync(self)
      read_res_20991 = np.empty(1, dtype=ct.c_int64)
      cl.enqueue_copy(self.queue, read_res_20991, mem_20300,
                      device_offset=(np.long(np.int64(0)) * 8),
                      is_blocking=synchronous)
      sync(self)
      max_per_thread_20278 = read_res_20991[0]
      mem_20300 = None
      sizze_sum_20295 = (num_threads_20272 * max_per_thread_20278)
      mem_20133 = opencl_alloc(self, sizze_sum_20295, "mem_20133")
      if ((1 * (np.long(segmap_usable_groups_19364) * np.long(segmap_group_sizze_19363))) != 0):
        self.mainzisegmap_19285_var.set_args(self.global_failure,
                                             self.failure_is_an_option,
                                             self.global_failure_args,
                                             np.int64(n_18596),
                                             np.float32(a_18604),
                                             np.float32(r0_18607),
                                             np.float32(x_18624),
                                             np.float32(x_18626),
                                             np.float32(y_18628),
                                             np.float32(y_18629),
                                             swap_term_mem_20096,
                                             payments_mem_20097, mem_20133,
                                             mem_20147,
                                             np.int64(num_threads_20272))
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19285_var,
                                   ((np.long(segmap_usable_groups_19364) * np.long(segmap_group_sizze_19363)),),
                                   (np.long(segmap_group_sizze_19363),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      mem_20133 = None
      res_mem_20148 = mem_20147
      res_mem_20149 = mem_20130
      res_mem_20150 = mem_20128
      res_mem_20151 = mem_20126
    sims_per_year_18706 = (res_18621 / res_18613)
    bounds_invalid_upwards_18707 = slt64(steps_18600, np.int64(1))
    valid_18708 = not(bounds_invalid_upwards_18707)
    range_valid_c_18709 = True
    assert valid_18708, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:60:56-67\n   #1  cva.fut:118:17-44\n   #2  cva.fut:107:1-177:18\n" % ("Range ",
                                                                                                                                                    np.int64(1),
                                                                                                                                                    "..",
                                                                                                                                                    np.int64(2),
                                                                                                                                                    "...",
                                                                                                                                                    steps_18600,
                                                                                                                                                    " is invalid."))
    bounds_invalid_upwards_18719 = slt64(paths_18599, np.int64(0))
    valid_18720 = not(bounds_invalid_upwards_18719)
    range_valid_c_18721 = True
    assert valid_18720, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:122:19-49\n   #4  cva.fut:107:1-177:18\n" % ("Range ",
                                                                                                                                                                                                                                                                np.int64(0),
                                                                                                                                                                                                                                                                "..",
                                                                                                                                                                                                                                                                np.int64(1),
                                                                                                                                                                                                                                                                "..<",
                                                                                                                                                                                                                                                                paths_18599,
                                                                                                                                                                                                                                                                " is invalid."))
    upper_bound_18724 = (steps_18600 - np.int64(1))
    res_18725 = futhark_sqrt32(dt_18622)
    segmap_group_sizze_19799 = self.sizes["main.segmap_group_size_19782"]
    segmap_usable_groups_19800 = sdiv_up64(paths_18599,
                                           segmap_group_sizze_19799)
    bytes_20153 = (np.int64(4) * paths_18599)
    mem_20154 = opencl_alloc(self, bytes_20153, "mem_20154")
    if ((1 * (np.long(segmap_usable_groups_19800) * np.long(segmap_group_sizze_19799))) != 0):
      self.mainzisegmap_19780_var.set_args(self.global_failure,
                                           np.int64(paths_18599), mem_20154)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19780_var,
                                 ((np.long(segmap_usable_groups_19800) * np.long(segmap_group_sizze_19799)),),
                                 (np.long(segmap_group_sizze_19799),))
      if synchronous:
        sync(self)
    nest_sizze_19823 = (paths_18599 * steps_18600)
    segmap_group_sizze_19824 = self.sizes["main.segmap_group_size_19719"]
    segmap_usable_groups_19825 = sdiv_up64(nest_sizze_19823,
                                           segmap_group_sizze_19824)
    bytes_20156 = (np.int64(4) * nest_sizze_19823)
    mem_20158 = opencl_alloc(self, bytes_20156, "mem_20158")
    if ((1 * (np.long(segmap_usable_groups_19825) * np.long(segmap_group_sizze_19824))) != 0):
      self.mainzisegmap_19716_var.set_args(self.global_failure,
                                           np.int64(paths_18599),
                                           np.int64(steps_18600), mem_20154,
                                           mem_20158)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19716_var,
                                 ((np.long(segmap_usable_groups_19825) * np.long(segmap_group_sizze_19824)),),
                                 (np.long(segmap_group_sizze_19824),))
      if synchronous:
        sync(self)
    mem_20154 = None
    segmap_group_sizze_19869 = self.sizes["main.segmap_group_size_19620"]
    max_num_groups_20436 = self.sizes["main.segmap_num_groups_19622"]
    num_groups_19870 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_18599,
                                                            sext_i32_i64(segmap_group_sizze_19869)),
                                                  sext_i32_i64(max_num_groups_20436))))
    mem_20161 = opencl_alloc(self, bytes_20156, "mem_20161")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_20161, np.int64(0),
                                                mem_20158, np.int64(0),
                                                np.int64(1), steps_18600,
                                                paths_18599)
    mem_20158 = None
    mem_20179 = opencl_alloc(self, bytes_20156, "mem_20179")
    bytes_20163 = (np.int64(4) * steps_18600)
    num_threads_20306 = (segmap_group_sizze_19869 * num_groups_19870)
    total_sizze_20307 = (bytes_20163 * num_threads_20306)
    mem_20164 = opencl_alloc(self, total_sizze_20307, "mem_20164")
    if ((1 * (np.long(num_groups_19870) * np.long(segmap_group_sizze_19869))) != 0):
      self.mainzisegmap_19618_var.set_args(self.global_failure,
                                           self.failure_is_an_option,
                                           self.global_failure_args,
                                           np.int64(paths_18599),
                                           np.int64(steps_18600),
                                           np.float32(a_18604),
                                           np.float32(b_18605),
                                           np.float32(sigma_18606),
                                           np.float32(r0_18607),
                                           np.float32(dt_18622),
                                           np.int64(upper_bound_18724),
                                           np.float32(res_18725),
                                           np.int64(num_groups_19870),
                                           mem_20161, mem_20164, mem_20179)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19618_var,
                                 ((np.long(num_groups_19870) * np.long(segmap_group_sizze_19869)),),
                                 (np.long(segmap_group_sizze_19869),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_20161 = None
    mem_20164 = None
    y_18787 = slt64(np.int64(0), n_18596)
    index_certs_18788 = True
    assert y_18787, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:136:53-60\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  cva.fut:135:30-137:39\n   #4  cva.fut:135:20-137:51\n   #5  cva.fut:107:1-177:18\n" % ("Index [",
                                                                                                                                                                                                                                                      np.int64(0),
                                                                                                                                                                                                                                                      "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                      n_18596,
                                                                                                                                                                                                                                                      "]."))
    read_res_20992 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_20992, res_mem_20148,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_18789 = read_res_20992[0]
    res_mem_20148 = None
    read_res_20993 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_20993, res_mem_20149,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_18790 = read_res_20993[0]
    res_mem_20149 = None
    read_res_20994 = np.empty(1, dtype=ct.c_int64)
    cl.enqueue_copy(self.queue, read_res_20994, res_mem_20150,
                    device_offset=(np.long(np.int64(0)) * 8),
                    is_blocking=synchronous)
    sync(self)
    res_18791 = read_res_20994[0]
    res_mem_20150 = None
    read_res_20995 = np.empty(1, dtype=ct.c_float)
    cl.enqueue_copy(self.queue, read_res_20995, res_mem_20151,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_18792 = read_res_20995[0]
    res_mem_20151 = None
    res_18800 = sitofp_i64_f32(paths_18599)
    mem_20181 = opencl_alloc(self, bytes_20163, "mem_20181")
    segscan_group_sizze_19902 = self.sizes["main.segscan_group_size_19901"]
    max_num_groups_20448 = self.sizes["main.segscan_num_groups_19903"]
    num_groups_19904 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_18599,
                                                            sext_i32_i64(segscan_group_sizze_19902)),
                                                  sext_i32_i64(max_num_groups_20448))))
    segred_group_sizze_19910 = self.sizes["main.segred_group_size_19909"]
    max_num_groups_20449 = self.sizes["main.segred_num_groups_19911"]
    num_groups_19912 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_18599,
                                                            sext_i32_i64(segred_group_sizze_19910)),
                                                  sext_i32_i64(max_num_groups_20449))))
    seghist_group_sizze_19920 = self.sizes["main.seghist_group_size_19919"]
    max_num_groups_20450 = self.sizes["main.seghist_num_groups_19921"]
    num_groups_19922 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_18599,
                                                            sext_i32_i64(seghist_group_sizze_19920)),
                                                  sext_i32_i64(max_num_groups_20450))))
    segscan_group_sizze_19935 = self.sizes["main.segscan_group_size_19934"]
    segscan_group_sizze_19943 = self.sizes["main.segscan_group_size_19942"]
    segscan_group_sizze_19951 = self.sizes["main.segscan_group_size_19950"]
    segscan_group_sizze_20003 = self.sizes["main.segscan_group_size_20002"]
    segred_group_sizze_20017 = self.sizes["main.segred_group_size_20016"]
    max_num_groups_20451 = self.sizes["main.segred_num_groups_20018"]
    num_groups_20019 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_18599,
                                                            sext_i32_i64(segred_group_sizze_20017)),
                                                  sext_i32_i64(max_num_groups_20451))))
    bytes_20190 = (np.int64(8) * paths_18599)
    mem_20191 = opencl_alloc(self, bytes_20190, "mem_20191")
    mem_20193 = opencl_alloc(self, bytes_20190, "mem_20193")
    mem_20196 = opencl_alloc(self, np.int64(8), "mem_20196")
    mem_20225 = opencl_alloc(self, np.int64(4), "mem_20225")
    redout_19896 = np.float32(0.0)
    i_19898 = np.int64(0)
    one_21008 = np.int64(1)
    for counter_21007 in range(steps_18600):
      index_primexp_20055 = (np.int64(1) + i_19898)
      res_18826 = sitofp_i64_f32(index_primexp_20055)
      res_18827 = (res_18826 / sims_per_year_18706)
      x_18836 = (res_18827 / res_18792)
      ceil_arg_18837 = (x_18836 - np.float32(1.0))
      res_18838 = futhark_ceil32(ceil_arg_18837)
      res_18839 = fptosi_f32_i64(res_18838)
      max_arg_18840 = (res_18791 - res_18839)
      res_18841 = smax64(np.int64(0), max_arg_18840)
      cond_18842 = (res_18841 == np.int64(0))
      if cond_18842:
        res_18843 = np.int64(1)
      else:
        res_18843 = res_18841
      if slt64(np.int64(0), paths_18599):
        stage1_max_num_groups_20454 = self.max_group_size
        stage1_num_groups_20455 = smin64(stage1_max_num_groups_20454,
                                         num_groups_19904)
        num_threads_20456 = sext_i64_i32((stage1_num_groups_20455 * segscan_group_sizze_19902))
        if ((1 * (np.long(stage1_num_groups_20455) * np.long(segscan_group_sizze_19902))) != 0):
          self.mainziscan_stage1_19907_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_19902)))),
                                                    np.int64(paths_18599),
                                                    np.int64(res_18843),
                                                    mem_20191, mem_20193,
                                                    np.int32(num_threads_20456))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_19907_var,
                                     ((np.long(stage1_num_groups_20455) * np.long(segscan_group_sizze_19902)),),
                                     (np.long(segscan_group_sizze_19902),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_20455))) != 0):
          self.mainziscan_stage2_19907_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_20455)))),
                                                    np.int64(paths_18599),
                                                    mem_20191,
                                                    np.int64(stage1_num_groups_20455),
                                                    np.int32(num_threads_20456))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_19907_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_20455)),),
                                     (np.long(stage1_num_groups_20455),))
          if synchronous:
            sync(self)
        required_groups_20492 = sext_i64_i32(sdiv_up64(paths_18599,
                                                       segscan_group_sizze_19902))
        if ((1 * (np.long(num_groups_19904) * np.long(segscan_group_sizze_19902))) != 0):
          self.mainziscan_stage3_19907_var.set_args(self.global_failure,
                                                    np.int64(paths_18599),
                                                    np.int64(num_groups_19904),
                                                    mem_20191,
                                                    np.int32(num_threads_20456),
                                                    np.int32(required_groups_20492))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_19907_var,
                                     ((np.long(num_groups_19904) * np.long(segscan_group_sizze_19902)),),
                                     (np.long(segscan_group_sizze_19902),))
          if synchronous:
            sync(self)
      mainzicounter_mem_20504 = self.mainzicounter_mem_20504
      group_res_arr_mem_20506 = opencl_alloc(self,
                                             (np.int32(8) * (segred_group_sizze_19910 * num_groups_19912)),
                                             "group_res_arr_mem_20506")
      num_threads_20508 = (num_groups_19912 * segred_group_sizze_19910)
      if ((1 * (np.long(num_groups_19912) * np.long(segred_group_sizze_19910))) != 0):
        self.mainzisegred_nonseg_19917_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(8) * segred_group_sizze_19910))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(paths_18599),
                                                    np.int64(num_groups_19912),
                                                    mem_20193, mem_20196,
                                                    mainzicounter_mem_20504,
                                                    group_res_arr_mem_20506,
                                                    np.int64(num_threads_20508))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_19917_var,
                                   ((np.long(num_groups_19912) * np.long(segred_group_sizze_19910)),),
                                   (np.long(segred_group_sizze_19910),))
        if synchronous:
          sync(self)
      read_res_20997 = np.empty(1, dtype=ct.c_int64)
      cl.enqueue_copy(self.queue, read_res_20997, mem_20196,
                      device_offset=(np.long(np.int64(0)) * 8),
                      is_blocking=synchronous)
      sync(self)
      res_18845 = read_res_20997[0]
      bounds_invalid_upwards_18850 = slt64(res_18845, np.int64(0))
      valid_18851 = not(bounds_invalid_upwards_18850)
      range_valid_c_18852 = True
      assert valid_18851, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:142:37-88\n   #6  cva.fut:141:18-144:79\n   #7  cva.fut:107:1-177:18\n" % ("Range ",
                                                                                                                                                                                                                                                                                                                                                                                                                                  np.int64(0),
                                                                                                                                                                                                                                                                                                                                                                                                                                  "..",
                                                                                                                                                                                                                                                                                                                                                                                                                                  np.int64(1),
                                                                                                                                                                                                                                                                                                                                                                                                                                  "..<",
                                                                                                                                                                                                                                                                                                                                                                                                                                  res_18845,
                                                                                                                                                                                                                                                                                                                                                                                                                                  " is invalid."))
      bytes_20197 = (np.int64(8) * res_18845)
      mem_20198 = opencl_alloc(self, bytes_20197, "mem_20198")
      self.futhark_builtinzhreplicate_i64(mem_20198, res_18845, np.int64(0))
      h_20546 = (np.int32(8) * res_18845)
      seg_h_20547 = (np.int32(8) * res_18845)
      if (seg_h_20547 == np.int64(0)):
        pass
      else:
        hist_H_20548 = res_18845
        hist_el_sizze_20549 = (sdiv_up64(h_20546, hist_H_20548) + np.int64(4))
        hist_N_20550 = paths_18599
        hist_RF_20551 = np.int64(1)
        hist_L_20552 = self.max_local_memory
        max_group_sizze_20553 = self.max_group_size
        num_groups_20554 = sdiv_up64(sext_i32_i64(sext_i64_i32((num_groups_19922 * seghist_group_sizze_19920))),
                                     max_group_sizze_20553)
        hist_m_prime_20555 = (sitofp_i64_f64(smin64(sext_i32_i64(squot32(hist_L_20552,
                                                                         hist_el_sizze_20549)),
                                                    sdiv_up64(hist_N_20550,
                                                              num_groups_20554))) / sitofp_i64_f64(hist_H_20548))
        hist_M0_20556 = smax64(np.int64(1),
                               smin64(fptosi_f64_i64(hist_m_prime_20555),
                                      max_group_sizze_20553))
        hist_Nout_20557 = np.int64(1)
        hist_Nin_20558 = paths_18599
        work_asymp_M_max_20559 = squot64((hist_Nout_20557 * hist_N_20550),
                                         ((np.int64(2) * num_groups_20554) * hist_H_20548))
        hist_M_20560 = sext_i64_i32(smin64(hist_M0_20556,
                                           work_asymp_M_max_20559))
        hist_C_20561 = sdiv_up64(max_group_sizze_20553,
                                 sext_i32_i64(smax32(np.int32(1),
                                                     hist_M_20560)))
        local_mem_needed_20562 = (hist_el_sizze_20549 * sext_i32_i64(hist_M_20560))
        hist_S_20563 = sext_i64_i32(sdiv_up64((hist_H_20548 * local_mem_needed_20562),
                                              hist_L_20552))
        if (sle64(hist_H_20548,
                  hist_Nin_20558) and (sle64(local_mem_needed_20562,
                                             hist_L_20552) and (sle32(hist_S_20563,
                                                                      np.int32(6)) and (sle64(hist_C_20561,
                                                                                              max_group_sizze_20553) and slt32(np.int32(0),
                                                                                                                               hist_M_20560))))):
          num_segments_20564 = np.int64(1)
          num_subhistos_20543 = (num_groups_20554 * num_segments_20564)
          if (num_subhistos_20543 == np.int64(1)):
            res_subhistos_mem_20544 = mem_20198
          else:
            res_subhistos_mem_20544 = opencl_alloc(self,
                                                   ((sext_i32_i64(num_subhistos_20543) * res_18845) * np.int32(8)),
                                                   "res_subhistos_mem_20544")
            self.futhark_builtinzhreplicate_i64(res_subhistos_mem_20544,
                                                (num_subhistos_20543 * res_18845),
                                                np.int64(0))
            if ((res_18845 * np.int32(8)) != 0):
              cl.enqueue_copy(self.queue, res_subhistos_mem_20544, mem_20198,
                              dest_offset=np.long(np.int64(0)),
                              src_offset=np.long(np.int64(0)),
                              byte_count=np.long((res_18845 * np.int32(8))))
            if synchronous:
              sync(self)
          chk_i_20565 = np.int32(0)
          one_20999 = np.int32(1)
          for counter_20998 in range(hist_S_20563):
            num_segments_20566 = np.int64(1)
            hist_H_chk_20567 = sdiv_up64(res_18845, sext_i32_i64(hist_S_20563))
            histo_sizze_20568 = hist_H_chk_20567
            init_per_thread_20569 = sext_i64_i32(sdiv_up64((sext_i32_i64(hist_M_20560) * histo_sizze_20568),
                                                           max_group_sizze_20553))
            if ((1 * (np.long(num_groups_20554) * np.long(max_group_sizze_20553))) != 0):
              self.mainziseghist_local_19925_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long((np.int32(4) * (hist_M_20560 * hist_H_chk_20567)))),
                                                          cl.LocalMemory(np.long((np.int32(8) * (hist_M_20560 * hist_H_chk_20567)))),
                                                          np.int64(paths_18599),
                                                          np.int64(res_18845),
                                                          mem_20191,
                                                          res_subhistos_mem_20544,
                                                          np.int32(max_group_sizze_20553),
                                                          np.int64(num_groups_20554),
                                                          np.int32(hist_M_20560),
                                                          np.int32(chk_i_20565),
                                                          np.int64(num_segments_20566),
                                                          np.int64(hist_H_chk_20567),
                                                          np.int64(histo_sizze_20568),
                                                          np.int32(init_per_thread_20569))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainziseghist_local_19925_var,
                                         ((np.long(num_groups_20554) * np.long(max_group_sizze_20553)),),
                                         (np.long(max_group_sizze_20553),))
              if synchronous:
                sync(self)
            chk_i_20565 += one_20999
        else:
          hist_H_20601 = res_18845
          hist_RF_20602 = ((np.float64(0.0) + sitofp_i32_f64(np.int64(1))) / np.float64(1.0))
          hist_el_sizze_20603 = squot32(sext_i64_i32((np.int32(4) + np.int32(8))),
                                        np.int32(2))
          hist_C_max_20604 = fmin64(sitofp_i32_f64(sext_i64_i32((num_groups_19922 * seghist_group_sizze_19920))),
                                    (sitofp_i32_f64(hist_H_20601) / np.float64(2.0)))
          hist_M_min_20605 = smax32(np.int32(1),
                                    sext_i64_i32(fptosi_f64_i64((sitofp_i32_f64(sext_i64_i32((num_groups_19922 * seghist_group_sizze_19920))) / hist_C_max_20604))))
          L2_sizze_20606 = self.sizes["main.L2_size_20606"]
          hist_RACE_exp_20607 = fmax64(np.float64(1.0),
                                       ((np.float64(0.75) * hist_RF_20602) / (np.float64(64.0) / sitofp_i32_f64(hist_el_sizze_20603))))
          if slt64(paths_18599, hist_H_20601):
            hist_S_20608 = np.int32(1)
          else:
            hist_S_20608 = sext_i64_i32(sdiv_up64(((sext_i32_i64(hist_M_min_20605) * hist_H_20601) * sext_i32_i64(hist_el_sizze_20603)),
                                                  fptosi_f64_i64(((np.float64(0.4) * sitofp_i32_f64(L2_sizze_20606)) * hist_RACE_exp_20607))))
          hist_H_chk_20609 = sdiv_up64(res_18845, sext_i32_i64(hist_S_20608))
          hist_k_max_20610 = (fmin64(((np.float64(0.4) * (sitofp_i32_f64(L2_sizze_20606) / sitofp_i32_f64(sext_i64_i32((np.int32(4) + np.int32(8)))))) * hist_RACE_exp_20607),
                                     sitofp_i32_f64(paths_18599)) / sitofp_i32_f64(sext_i64_i32((num_groups_19922 * seghist_group_sizze_19920))))
          hist_u_20611 = np.int64(1)
          hist_C_20612 = fmin64(sitofp_i32_f64(sext_i64_i32((num_groups_19922 * seghist_group_sizze_19920))),
                                (sitofp_i32_f64((hist_u_20611 * hist_H_chk_20609)) / hist_k_max_20610))
          hist_M_20613 = smax32(hist_M_min_20605,
                                sext_i64_i32(fptosi_f64_i64((sitofp_i32_f64(sext_i64_i32((num_groups_19922 * seghist_group_sizze_19920))) / hist_C_20612))))
          num_subhistos_20543 = sext_i32_i64(hist_M_20613)
          if (hist_M_20613 == np.int32(1)):
            res_subhistos_mem_20544 = mem_20198
          else:
            if (num_subhistos_20543 == np.int64(1)):
              res_subhistos_mem_20544 = mem_20198
            else:
              res_subhistos_mem_20544 = opencl_alloc(self,
                                                     ((sext_i32_i64(num_subhistos_20543) * res_18845) * np.int32(8)),
                                                     "res_subhistos_mem_20544")
              self.futhark_builtinzhreplicate_i64(res_subhistos_mem_20544,
                                                  (num_subhistos_20543 * res_18845),
                                                  np.int64(0))
              if ((res_18845 * np.int32(8)) != 0):
                cl.enqueue_copy(self.queue, res_subhistos_mem_20544, mem_20198,
                                dest_offset=np.long(np.int64(0)),
                                src_offset=np.long(np.int64(0)),
                                byte_count=np.long((res_18845 * np.int32(8))))
              if synchronous:
                sync(self)
          mainzihist_locks_mem_20614 = self.mainzihist_locks_mem_20614
          chk_i_20616 = np.int32(0)
          one_21002 = np.int32(1)
          for counter_21001 in range(hist_S_20608):
            hist_H_chk_20617 = sdiv_up64(res_18845, sext_i32_i64(hist_S_20608))
            if ((1 * (np.long(num_groups_19922) * np.long(seghist_group_sizze_19920))) != 0):
              self.mainziseghist_global_19925_var.set_args(self.global_failure,
                                                           np.int64(paths_18599),
                                                           np.int64(res_18845),
                                                           np.int64(num_groups_19922),
                                                           mem_20191,
                                                           np.int32(num_subhistos_20543),
                                                           res_subhistos_mem_20544,
                                                           mainzihist_locks_mem_20614,
                                                           np.int32(chk_i_20616),
                                                           np.int64(hist_H_chk_20617))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainziseghist_global_19925_var,
                                         ((np.long(num_groups_19922) * np.long(seghist_group_sizze_19920)),),
                                         (np.long(seghist_group_sizze_19920),))
              if synchronous:
                sync(self)
            chk_i_20616 += one_21002
        if (num_subhistos_20543 == np.int64(1)):
          mem_20198 = res_subhistos_mem_20544
        else:
          if slt64((num_subhistos_20543 * np.int64(2)),
                   seghist_group_sizze_19920):
            segment_sizze_nonzzero_20630 = smax64(np.int64(1),
                                                  num_subhistos_20543)
            num_threads_20631 = (num_groups_19922 * seghist_group_sizze_19920)
            if ((1 * (np.long(num_groups_19922) * np.long(seghist_group_sizze_19920))) != 0):
              self.mainzisegred_small_20629_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long((np.int32(8) * seghist_group_sizze_19920))),
                                                         np.int64(res_18845),
                                                         np.int64(num_groups_19922),
                                                         mem_20198,
                                                         np.int32(num_subhistos_20543),
                                                         res_subhistos_mem_20544,
                                                         np.int64(segment_sizze_nonzzero_20630))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_small_20629_var,
                                         ((np.long(num_groups_19922) * np.long(seghist_group_sizze_19920)),),
                                         (np.long(seghist_group_sizze_19920),))
              if synchronous:
                sync(self)
          else:
            groups_per_segment_20651 = sdiv_up64(num_groups_19922,
                                                 smax64(np.int64(1), res_18845))
            elements_per_thread_20652 = sdiv_up64(num_subhistos_20543,
                                                  (seghist_group_sizze_19920 * groups_per_segment_20651))
            virt_num_groups_20653 = (groups_per_segment_20651 * res_18845)
            num_threads_20654 = (num_groups_19922 * seghist_group_sizze_19920)
            threads_per_segment_20655 = (groups_per_segment_20651 * seghist_group_sizze_19920)
            group_res_arr_mem_20656 = opencl_alloc(self,
                                                   (np.int32(8) * (seghist_group_sizze_19920 * virt_num_groups_20653)),
                                                   "group_res_arr_mem_20656")
            mainzicounter_mem_20658 = self.mainzicounter_mem_20658
            if ((1 * (np.long(num_groups_19922) * np.long(seghist_group_sizze_19920))) != 0):
              self.mainzisegred_large_20629_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long(np.int32(1))),
                                                         cl.LocalMemory(np.long((np.int32(8) * seghist_group_sizze_19920))),
                                                         np.int64(res_18845),
                                                         np.int64(num_groups_19922),
                                                         mem_20198,
                                                         np.int32(num_subhistos_20543),
                                                         res_subhistos_mem_20544,
                                                         np.int64(groups_per_segment_20651),
                                                         np.int64(elements_per_thread_20652),
                                                         np.int64(virt_num_groups_20653),
                                                         np.int64(threads_per_segment_20655),
                                                         group_res_arr_mem_20656,
                                                         mainzicounter_mem_20658)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_large_20629_var,
                                         ((np.long(num_groups_19922) * np.long(seghist_group_sizze_19920)),),
                                         (np.long(seghist_group_sizze_19920),))
              if synchronous:
                sync(self)
      max_num_groups_20690 = self.sizes["main.segscan_num_groups_19936"]
      num_groups_19937 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_18845,
                                                              sext_i32_i64(segscan_group_sizze_19935)),
                                                    sext_i32_i64(max_num_groups_20690))))
      mem_20202 = opencl_alloc(self, res_18845, "mem_20202")
      mem_20204 = opencl_alloc(self, bytes_20197, "mem_20204")
      if slt64(np.int64(0), res_18845):
        stage1_max_num_groups_20691 = self.max_group_size
        stage1_num_groups_20692 = smin64(stage1_max_num_groups_20691,
                                         num_groups_19937)
        num_threads_20693 = sext_i64_i32((stage1_num_groups_20692 * segscan_group_sizze_19935))
        if ((1 * (np.long(stage1_num_groups_20692) * np.long(segscan_group_sizze_19935))) != 0):
          self.mainziscan_stage1_19940_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_19935)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_19935)))),
                                                    np.int64(res_18845),
                                                    mem_20198, mem_20202,
                                                    mem_20204,
                                                    np.int32(num_threads_20693))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_19940_var,
                                     ((np.long(stage1_num_groups_20692) * np.long(segscan_group_sizze_19935)),),
                                     (np.long(segscan_group_sizze_19935),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_20692))) != 0):
          self.mainziscan_stage2_19940_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_20692)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_20692)))),
                                                    np.int64(res_18845),
                                                    mem_20202, mem_20204,
                                                    np.int64(stage1_num_groups_20692),
                                                    np.int32(num_threads_20693))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_19940_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_20692)),),
                                     (np.long(stage1_num_groups_20692),))
          if synchronous:
            sync(self)
        required_groups_20745 = sext_i64_i32(sdiv_up64(res_18845,
                                                       segscan_group_sizze_19935))
        if ((1 * (np.long(num_groups_19937) * np.long(segscan_group_sizze_19935))) != 0):
          self.mainziscan_stage3_19940_var.set_args(self.global_failure,
                                                    np.int64(res_18845),
                                                    np.int64(num_groups_19937),
                                                    mem_20202, mem_20204,
                                                    np.int32(num_threads_20693),
                                                    np.int32(required_groups_20745))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_19940_var,
                                     ((np.long(num_groups_19937) * np.long(segscan_group_sizze_19935)),),
                                     (np.long(segscan_group_sizze_19935),))
          if synchronous:
            sync(self)
      mem_20198 = None
      mem_20202 = None
      max_num_groups_20757 = self.sizes["main.segscan_num_groups_19944"]
      num_groups_19945 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_18845,
                                                              sext_i32_i64(segscan_group_sizze_19943)),
                                                    sext_i32_i64(max_num_groups_20757))))
      mem_20207 = opencl_alloc(self, res_18845, "mem_20207")
      mem_20209 = opencl_alloc(self, bytes_20197, "mem_20209")
      mem_20211 = opencl_alloc(self, res_18845, "mem_20211")
      if slt64(np.int64(0), res_18845):
        stage1_max_num_groups_20758 = self.max_group_size
        stage1_num_groups_20759 = smin64(stage1_max_num_groups_20758,
                                         num_groups_19945)
        num_threads_20760 = sext_i64_i32((stage1_num_groups_20759 * segscan_group_sizze_19943))
        if ((1 * (np.long(stage1_num_groups_20759) * np.long(segscan_group_sizze_19943))) != 0):
          self.mainziscan_stage1_19948_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_19943)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_19943)))),
                                                    np.int64(res_18845),
                                                    mem_20204, mem_20207,
                                                    mem_20209, mem_20211,
                                                    np.int32(num_threads_20760))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_19948_var,
                                     ((np.long(stage1_num_groups_20759) * np.long(segscan_group_sizze_19943)),),
                                     (np.long(segscan_group_sizze_19943),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_20759))) != 0):
          self.mainziscan_stage2_19948_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_20759)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_20759)))),
                                                    np.int64(res_18845),
                                                    mem_20207, mem_20209,
                                                    np.int64(stage1_num_groups_20759),
                                                    np.int32(num_threads_20760))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_19948_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_20759)),),
                                     (np.long(stage1_num_groups_20759),))
          if synchronous:
            sync(self)
        required_groups_20812 = sext_i64_i32(sdiv_up64(res_18845,
                                                       segscan_group_sizze_19943))
        if ((1 * (np.long(num_groups_19945) * np.long(segscan_group_sizze_19943))) != 0):
          self.mainziscan_stage3_19948_var.set_args(self.global_failure,
                                                    np.int64(res_18845),
                                                    np.int64(num_groups_19945),
                                                    mem_20207, mem_20209,
                                                    np.int32(num_threads_20760),
                                                    np.int32(required_groups_20812))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_19948_var,
                                     ((np.long(num_groups_19945) * np.long(segscan_group_sizze_19943)),),
                                     (np.long(segscan_group_sizze_19943),))
          if synchronous:
            sync(self)
      mem_20207 = None
      max_num_groups_20824 = self.sizes["main.segscan_num_groups_19952"]
      num_groups_19953 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_18845,
                                                              sext_i32_i64(segscan_group_sizze_19951)),
                                                    sext_i32_i64(max_num_groups_20824))))
      mem_20214 = opencl_alloc(self, res_18845, "mem_20214")
      bytes_20215 = (np.int64(4) * res_18845)
      mem_20216 = opencl_alloc(self, bytes_20215, "mem_20216")
      if slt64(np.int64(0), res_18845):
        stage1_max_num_groups_20825 = self.max_group_size
        stage1_num_groups_20826 = smin64(stage1_max_num_groups_20825,
                                         num_groups_19953)
        num_threads_20827 = sext_i64_i32((stage1_num_groups_20826 * segscan_group_sizze_19951))
        if ((1 * (np.long(stage1_num_groups_20826) * np.long(segscan_group_sizze_19951))) != 0):
          self.mainziscan_stage1_19956_var.set_args(self.global_failure,
                                                    self.failure_is_an_option,
                                                    self.global_failure_args,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * segscan_group_sizze_19951)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_19951)))),
                                                    np.int64(paths_18599),
                                                    np.float32(a_18604),
                                                    np.float32(b_18605),
                                                    np.float32(sigma_18606),
                                                    np.float32(res_18789),
                                                    np.float32(res_18790),
                                                    np.int64(res_18791),
                                                    np.float32(res_18792),
                                                    np.float32(res_18827),
                                                    np.int64(res_18845),
                                                    np.int64(i_19898),
                                                    mem_20179, mem_20204,
                                                    mem_20209, mem_20211,
                                                    mem_20214, mem_20216,
                                                    np.int32(num_threads_20827))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_19956_var,
                                     ((np.long(stage1_num_groups_20826) * np.long(segscan_group_sizze_19951)),),
                                     (np.long(segscan_group_sizze_19951),))
          if synchronous:
            sync(self)
        self.failure_is_an_option = np.int32(1)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_20826))) != 0):
          self.mainziscan_stage2_19956_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * stage1_num_groups_20826)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_20826)))),
                                                    np.int64(res_18845),
                                                    mem_20214, mem_20216,
                                                    np.int64(stage1_num_groups_20826),
                                                    np.int32(num_threads_20827))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_19956_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_20826)),),
                                     (np.long(stage1_num_groups_20826),))
          if synchronous:
            sync(self)
        required_groups_20879 = sext_i64_i32(sdiv_up64(res_18845,
                                                       segscan_group_sizze_19951))
        if ((1 * (np.long(num_groups_19953) * np.long(segscan_group_sizze_19951))) != 0):
          self.mainziscan_stage3_19956_var.set_args(self.global_failure,
                                                    np.int64(res_18845),
                                                    np.int64(num_groups_19953),
                                                    mem_20214, mem_20216,
                                                    np.int32(num_threads_20827),
                                                    np.int32(required_groups_20879))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_19956_var,
                                     ((np.long(num_groups_19953) * np.long(segscan_group_sizze_19951)),),
                                     (np.long(segscan_group_sizze_19951),))
          if synchronous:
            sync(self)
      mem_20204 = None
      mem_20209 = None
      mem_20214 = None
      max_num_groups_20891 = self.sizes["main.segscan_num_groups_20004"]
      num_groups_20005 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_18845,
                                                              sext_i32_i64(segscan_group_sizze_20003)),
                                                    sext_i32_i64(max_num_groups_20891))))
      mem_20219 = opencl_alloc(self, bytes_20197, "mem_20219")
      if slt64(np.int64(0), res_18845):
        stage1_max_num_groups_20892 = self.max_group_size
        stage1_num_groups_20893 = smin64(stage1_max_num_groups_20892,
                                         num_groups_20005)
        num_threads_20894 = sext_i64_i32((stage1_num_groups_20893 * segscan_group_sizze_20003))
        if ((1 * (np.long(stage1_num_groups_20893) * np.long(segscan_group_sizze_20003))) != 0):
          self.mainziscan_stage1_20008_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_20003)))),
                                                    np.int64(res_18845),
                                                    mem_20211, mem_20219,
                                                    np.int32(num_threads_20894))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_20008_var,
                                     ((np.long(stage1_num_groups_20893) * np.long(segscan_group_sizze_20003)),),
                                     (np.long(segscan_group_sizze_20003),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_20893))) != 0):
          self.mainziscan_stage2_20008_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_20893)))),
                                                    np.int64(res_18845),
                                                    mem_20219,
                                                    np.int64(stage1_num_groups_20893),
                                                    np.int32(num_threads_20894))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_20008_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_20893)),),
                                     (np.long(stage1_num_groups_20893),))
          if synchronous:
            sync(self)
        required_groups_20930 = sext_i64_i32(sdiv_up64(res_18845,
                                                       segscan_group_sizze_20003))
        if ((1 * (np.long(num_groups_20005) * np.long(segscan_group_sizze_20003))) != 0):
          self.mainziscan_stage3_20008_var.set_args(self.global_failure,
                                                    np.int64(res_18845),
                                                    np.int64(num_groups_20005),
                                                    mem_20219,
                                                    np.int32(num_threads_20894),
                                                    np.int32(required_groups_20930))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_20008_var,
                                     ((np.long(num_groups_20005) * np.long(segscan_group_sizze_20003)),),
                                     (np.long(segscan_group_sizze_20003),))
          if synchronous:
            sync(self)
      cond_19038 = slt64(np.int64(0), res_18845)
      if cond_19038:
        i_19040 = (res_18845 - np.int64(1))
        x_19041 = sle64(np.int64(0), i_19040)
        y_19042 = slt64(i_19040, res_18845)
        bounds_check_19043 = (x_19041 and y_19042)
        index_certs_19044 = True
        assert bounds_check_19043, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:142:37-88\n   #5  cva.fut:141:18-144:79\n   #6  cva.fut:107:1-177:18\n" % ("Index [",
                                                                                                                                                                                                                                                                                                                                                                                                   i_19040,
                                                                                                                                                                                                                                                                                                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                                                                                                                                                   res_18845,
                                                                                                                                                                                                                                                                                                                                                                                                   "]."))
        read_res_21004 = np.empty(1, dtype=ct.c_int64)
        cl.enqueue_copy(self.queue, read_res_21004, mem_20219,
                        device_offset=(np.long(i_19040) * 8),
                        is_blocking=synchronous)
        sync(self)
        res_19045 = read_res_21004[0]
        num_segments_19039 = res_19045
      else:
        num_segments_19039 = np.int64(0)
      bounds_invalid_upwards_19046 = slt64(num_segments_19039, np.int64(0))
      valid_19047 = not(bounds_invalid_upwards_19046)
      range_valid_c_19048 = True
      assert valid_19047, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:142:37-88\n   #6  cva.fut:141:18-144:79\n   #7  cva.fut:107:1-177:18\n" % ("Range ",
                                                                                                                                                                                                                                                                                                                                                                                                                                 np.int64(0),
                                                                                                                                                                                                                                                                                                                                                                                                                                 "..",
                                                                                                                                                                                                                                                                                                                                                                                                                                 np.int64(1),
                                                                                                                                                                                                                                                                                                                                                                                                                                 "..<",
                                                                                                                                                                                                                                                                                                                                                                                                                                 num_segments_19039,
                                                                                                                                                                                                                                                                                                                                                                                                                                 " is invalid."))
      bytes_20220 = (np.int64(4) * num_segments_19039)
      mem_20221 = opencl_alloc(self, bytes_20220, "mem_20221")
      self.futhark_builtinzhreplicate_f32(mem_20221, num_segments_19039,
                                          np.float32(0.0))
      segmap_group_sizze_20013 = self.sizes["main.segmap_group_size_20012"]
      segmap_usable_groups_20014 = sdiv_up64(res_18845,
                                             segmap_group_sizze_20013)
      if ((1 * (np.long(segmap_usable_groups_20014) * np.long(segmap_group_sizze_20013))) != 0):
        self.mainzisegmap_20010_var.set_args(self.global_failure,
                                             np.int64(res_18845),
                                             np.int64(num_segments_19039),
                                             mem_20211, mem_20216, mem_20219,
                                             mem_20221)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20010_var,
                                   ((np.long(segmap_usable_groups_20014) * np.long(segmap_group_sizze_20013)),),
                                   (np.long(segmap_group_sizze_20013),))
        if synchronous:
          sync(self)
      mem_20211 = None
      mem_20216 = None
      mem_20219 = None
      dim_match_19056 = (paths_18599 == num_segments_19039)
      empty_or_match_cert_19057 = True
      assert dim_match_19056, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:142:37-88\n   #2  cva.fut:141:18-144:79\n   #3  cva.fut:107:1-177:18\n" % ("Value of (core language) shape (",
                                                                                                                                                                                                                           num_segments_19039,
                                                                                                                                                                                                                           ") cannot match shape of type `[",
                                                                                                                                                                                                                           paths_18599,
                                                                                                                                                                                                                           "]b`."))
      mainzicounter_mem_20956 = self.mainzicounter_mem_20956
      group_res_arr_mem_20958 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_20017 * num_groups_20019)),
                                             "group_res_arr_mem_20958")
      num_threads_20960 = (num_groups_20019 * segred_group_sizze_20017)
      if ((1 * (np.long(num_groups_20019) * np.long(segred_group_sizze_20017))) != 0):
        self.mainzisegred_nonseg_20024_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_20017))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(paths_18599),
                                                    np.int64(num_groups_20019),
                                                    mem_20221, mem_20225,
                                                    mainzicounter_mem_20956,
                                                    group_res_arr_mem_20958,
                                                    np.int64(num_threads_20960))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_20024_var,
                                   ((np.long(num_groups_20019) * np.long(segred_group_sizze_20017)),),
                                   (np.long(segred_group_sizze_20017),))
        if synchronous:
          sync(self)
      mem_20221 = None
      read_res_21006 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_21006, mem_20225,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_19059 = read_res_21006[0]
      res_19065 = (res_19059 / res_18800)
      negate_arg_19066 = (a_18604 * res_18827)
      exp_arg_19067 = (np.float32(0.0) - negate_arg_19066)
      res_19068 = fpow32(np.float32(2.7182817459106445), exp_arg_19067)
      x_19069 = (np.float32(1.0) - res_19068)
      B_19070 = (x_19069 / a_18604)
      x_19071 = (B_19070 - res_18827)
      x_19072 = (y_18628 * x_19071)
      A1_19073 = (x_19072 / x_18624)
      y_19074 = fpow32(B_19070, np.float32(2.0))
      x_19075 = (x_18626 * y_19074)
      A2_19076 = (x_19075 / y_18629)
      exp_arg_19077 = (A1_19073 - A2_19076)
      res_19078 = fpow32(np.float32(2.7182817459106445), exp_arg_19077)
      negate_arg_19079 = (np.float32(5.000000074505806e-2) * B_19070)
      exp_arg_19080 = (np.float32(0.0) - negate_arg_19079)
      res_19081 = fpow32(np.float32(2.7182817459106445), exp_arg_19080)
      res_19082 = (res_19078 * res_19081)
      res_19083 = (res_19065 * res_19082)
      res_18815 = (res_19083 + redout_19896)
      cl.enqueue_copy(self.queue, mem_20181, np.array(res_19083,
                                                      dtype=ct.c_float),
                      device_offset=(np.long(i_19898) * 4),
                      is_blocking=synchronous)
      redout_tmp_20452 = res_18815
      redout_19896 = redout_tmp_20452
      i_19898 += one_21008
    res_18811 = redout_19896
    mem_20179 = None
    mem_20191 = None
    mem_20193 = None
    mem_20196 = None
    mem_20225 = None
    CVA_19084 = (np.float32(6.000000052154064e-3) * res_18811)
    mem_20232 = opencl_alloc(self, bytes_20163, "mem_20232")
    if ((steps_18600 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_20232, mem_20181,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((steps_18600 * np.int32(4))))
    if synchronous:
      sync(self)
    mem_20181 = None
    out_arrsizze_20319 = steps_18600
    out_mem_20318 = mem_20232
    scalar_out_20317 = CVA_19084
    return (scalar_out_20317, out_mem_20318, out_arrsizze_20319)
  def main(self, paths_18599_ext, steps_18600_ext, swap_term_mem_20096_ext,
           payments_mem_20097_ext, notional_mem_20098_ext, a_18604_ext,
           b_18605_ext, sigma_18606_ext, r0_18607_ext):
    try:
      paths_18599 = np.int64(ct.c_int64(paths_18599_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(paths_18599_ext),
                                                                                                                            paths_18599_ext))
    try:
      steps_18600 = np.int64(ct.c_int64(steps_18600_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(steps_18600_ext),
                                                                                                                            steps_18600_ext))
    try:
      assert ((type(swap_term_mem_20096_ext) in [np.ndarray,
                                                 cl.array.Array]) and (swap_term_mem_20096_ext.dtype == np.float32)), "Parameter has unexpected type"
      n_18596 = np.int32(swap_term_mem_20096_ext.shape[0])
      if (type(swap_term_mem_20096_ext) == cl.array.Array):
        swap_term_mem_20096 = swap_term_mem_20096_ext.data
      else:
        swap_term_mem_20096 = opencl_alloc(self,
                                           np.int64(swap_term_mem_20096_ext.nbytes),
                                           "swap_term_mem_20096")
        if (np.int64(swap_term_mem_20096_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, swap_term_mem_20096,
                          normaliseArray(swap_term_mem_20096_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(swap_term_mem_20096_ext),
                                                                                                                            swap_term_mem_20096_ext))
    try:
      assert ((type(payments_mem_20097_ext) in [np.ndarray,
                                                cl.array.Array]) and (payments_mem_20097_ext.dtype == np.int64)), "Parameter has unexpected type"
      n_18597 = np.int32(payments_mem_20097_ext.shape[0])
      if (type(payments_mem_20097_ext) == cl.array.Array):
        payments_mem_20097 = payments_mem_20097_ext.data
      else:
        payments_mem_20097 = opencl_alloc(self,
                                          np.int64(payments_mem_20097_ext.nbytes),
                                          "payments_mem_20097")
        if (np.int64(payments_mem_20097_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, payments_mem_20097,
                          normaliseArray(payments_mem_20097_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i64",
                                                                                                                            type(payments_mem_20097_ext),
                                                                                                                            payments_mem_20097_ext))
    try:
      assert ((type(notional_mem_20098_ext) in [np.ndarray,
                                                cl.array.Array]) and (notional_mem_20098_ext.dtype == np.float32)), "Parameter has unexpected type"
      n_18598 = np.int32(notional_mem_20098_ext.shape[0])
      if (type(notional_mem_20098_ext) == cl.array.Array):
        notional_mem_20098 = notional_mem_20098_ext.data
      else:
        notional_mem_20098 = opencl_alloc(self,
                                          np.int64(notional_mem_20098_ext.nbytes),
                                          "notional_mem_20098")
        if (np.int64(notional_mem_20098_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, notional_mem_20098,
                          normaliseArray(notional_mem_20098_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(notional_mem_20098_ext),
                                                                                                                            notional_mem_20098_ext))
    try:
      a_18604 = np.float32(ct.c_float(a_18604_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_18604_ext),
                                                                                                                            a_18604_ext))
    try:
      b_18605 = np.float32(ct.c_float(b_18605_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_18605_ext),
                                                                                                                            b_18605_ext))
    try:
      sigma_18606 = np.float32(ct.c_float(sigma_18606_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_18606_ext),
                                                                                                                            sigma_18606_ext))
    try:
      r0_18607 = np.float32(ct.c_float(r0_18607_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_18607_ext),
                                                                                                                            r0_18607_ext))
    (scalar_out_20317, out_mem_20318,
     out_arrsizze_20319) = self.futhark_main(swap_term_mem_20096,
                                             payments_mem_20097,
                                             notional_mem_20098, n_18596,
                                             n_18597, n_18598, paths_18599,
                                             steps_18600, a_18604, b_18605,
                                             sigma_18606, r0_18607)
    sync(self)
    return (np.float32(scalar_out_20317), cl.array.Array(self.queue,
                                                         (out_arrsizze_20319,),
                                                         ct.c_float,
                                                         data=out_mem_20318))